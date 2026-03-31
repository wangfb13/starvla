#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore
from numpy.typing import NDArray  # type: ignore

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_wrist_fisheye_x5 import WirstFisheyeCameraX5Config

logger = logging.getLogger(__name__)


class WristFisheyeX5Camera(Camera):
    def __init__(self, config: WirstFisheyeCameraX5Config):
        super().__init__(config)

        self.config = config
        self.ip = config.x5_ip
        self.port = config.port
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.intrinsics = config.intrinsics
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width = self.width
        self.capture_height = self.height
        if self.width and self.height and self.rotation in [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]:
            self.capture_width, self.capture_height = self.height, self.width

        self.context: Any | None = None
        self.socket: Any | None = None
        self._connected = False
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.latest_frame_id = 0
        self.last_consumed_frame_id = 0
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.ip}:{self.port})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self.context is not None and self.socket is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("Wrist fisheye X5 cameras require manual IP/port configuration.")

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        try:
            import zmq
        except Exception as e:
            raise ImportError(f"Could not import pyzmq: {e}") from e

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{self.ip}:{self.port}")
        self._connected = True

        if self.width is None or self.height is None:
            temp_frame = self._read_from_hardware()
            raw_height, raw_width = temp_frame.shape[:2]
            self.capture_width = raw_width
            self.capture_height = raw_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width = raw_height
                self.height = raw_width
            else:
                self.width = raw_width
                self.height = raw_height

        self._start_read_thread()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < max(self.warmup_s, 1):
                self.async_read(timeout_ms=max(self.warmup_s, 1) * 1000)
                time.sleep(0.1)

        logger.info("%s connected.", self)

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        self.last_consumed_frame_id = 0
        return self.async_read(timeout_ms=1000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        deadline = time.perf_counter() + timeout_ms / 1000
        while True:
            with self.frame_lock:
                frame = self.latest_frame
                frame_id = self.latest_frame_id
                if frame is not None and frame_id != self.last_consumed_frame_id:
                    self.last_consumed_frame_id = frame_id
                    return frame.copy()

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"{self} timed out waiting for a new frame.")

            self.new_frame_event.wait(remaining)
            self.new_frame_event.clear()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        with self.frame_lock:
            if self.latest_frame is None or self.latest_timestamp is None:
                raise RuntimeError(f"{self} has not captured any frames yet.")
            age_ms = (time.perf_counter() - self.latest_timestamp) * 1000
            if age_ms > max_age_ms:
                raise TimeoutError(f"{self} latest frame is older than {max_age_ms}ms.")
            return self.latest_frame.copy()

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_read_thread()
        self._cleanup()
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.latest_frame_id = 0
            self.last_consumed_frame_id = 0
            self.new_frame_event.clear()

    def _cleanup(self) -> None:
        self._connected = False
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None

    def _read_from_hardware(self) -> NDArray[Any]:
        if not self.is_connected or self.socket is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            payload = self.socket.recv()
        except Exception as e:
            if type(e).__name__ == "Again":
                raise TimeoutError(f"{self} timed out waiting for ZMQ payload.") from e
            raise

        frame_bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"{self} failed to decode JPEG frame.")

        if self.capture_width and self.capture_height:
            if frame_bgr.shape[1] != self.capture_width or frame_bgr.shape[0] != self.capture_height:
                logger.debug(
                    "%s resolution mismatch: got %sx%s, expected %sx%s",
                    self,
                    frame_bgr.shape[1],
                    frame_bgr.shape[0],
                    self.capture_width,
                    self.capture_height,
                )

        if self.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame = frame_bgr

        if self.rotation is not None:
            frame = cv2.rotate(frame, self.rotation)

        return frame

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, daemon=True, name=f"{self}_read_loop")
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self} stop_event is not initialized.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
                    self.latest_frame_id += 1
                self.new_frame_event.set()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning("%s read error: %s", self, e)
                time.sleep(0.01)


WirstFisheyeCameraX5 = WristFisheyeX5Camera
