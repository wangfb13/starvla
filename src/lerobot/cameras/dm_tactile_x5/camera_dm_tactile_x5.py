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

import json
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np  # type: ignore
from numpy.typing import NDArray  # type: ignore

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_dm_tactile_x5 import DmTactileCameraX5Config

logger = logging.getLogger(__name__)


class DmTactileCameraX5(Camera):
    def __init__(self, config: DmTactileCameraX5Config):
        super().__init__(config)

        self.config = config
        self.pub_addr = config.pub_addr
        self.topic = config.topic
        self.left_serial_number = config.left_serial_number.strip() if config.left_serial_number else None
        self.right_serial_number = config.right_serial_number.strip() if config.right_serial_number else None
        self.warmup_s = config.warmup_s
        self.inference_mode = config.inference_mode

        self.context: Any | None = None
        self.socket: Any | None = None
        self._connected = False
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frames: dict[str, NDArray[Any] | None] = {"left": None, "right": None}
        self.latest_timestamps: dict[str, float | None] = {"left": None, "right": None}
        self.latest_frame_id = 0
        self.last_consumed_frame_id = 0
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.pub_addr})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self.context is not None and self.socket is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("DM tactile X5 cameras require manual network configuration.")

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        try:
            import zmq
        except Exception as e:
            raise ImportError(f"Could not import pyzmq: {e}") from e

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVTIMEO, 200)
        self.socket.setsockopt(zmq.LINGER, 0)
        topic = self.topic if isinstance(self.topic, bytes) else str(self.topic).encode("utf-8")
        self.socket.setsockopt(zmq.SUBSCRIBE, topic)
        self.socket.connect(self.pub_addr)
        self._connected = True

        self._start_read_thread()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < max(self.warmup_s, 1):
                self.async_read(timeout_ms=max(self.warmup_s, 1) * 1000)
                time.sleep(0.1)

        logger.info("%s connected.", self)

    @check_if_not_connected
    def read(self) -> dict[str, NDArray[Any] | None]:
        self.last_consumed_frame_id = 0
        return self.async_read(timeout_ms=1000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> dict[str, NDArray[Any] | None]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        deadline = time.perf_counter() + timeout_ms / 1000
        while True:
            with self.frame_lock:
                ready = self._required_sides_ready()
                if ready and self.latest_frame_id != self.last_consumed_frame_id:
                    self.last_consumed_frame_id = self.latest_frame_id
                    return self._copy_frames()

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"{self} timed out waiting for tactile frames.")

            self.new_frame_event.wait(remaining)
            self.new_frame_event.clear()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> dict[str, NDArray[Any] | None]:
        with self.frame_lock:
            if not self._required_sides_ready():
                raise RuntimeError(f"{self} has not captured all required tactile frames yet.")

            for side in self._required_sides():
                timestamp = self.latest_timestamps[side]
                if timestamp is None:
                    raise RuntimeError(f"{self} has no {side} tactile timestamp.")
                age_ms = (time.perf_counter() - timestamp) * 1000
                if age_ms > max_age_ms:
                    raise TimeoutError(f"{self} latest {side} tactile frame is older than {max_age_ms}ms.")

            return self._copy_frames()

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_read_thread()
        self._cleanup()
        with self.frame_lock:
            self.latest_frames = {"left": None, "right": None}
            self.latest_timestamps = {"left": None, "right": None}
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

    def _required_sides(self) -> list[str]:
        sides: list[str] = []
        if self.left_serial_number is not None:
            sides.append("left")
        if self.right_serial_number is not None:
            sides.append("right")
        return sides

    def _required_sides_ready(self) -> bool:
        return all(self.latest_frames[side] is not None for side in self._required_sides())

    def _resolve_side(self, serial: str) -> str | None:
        if self.left_serial_number and serial == self.left_serial_number:
            return "left"
        if self.right_serial_number and serial == self.right_serial_number:
            return "right"
        return None

    def _validate_frame_resolution(self, frame: NDArray[Any]) -> None:
        if self.width is not None and self.height is not None:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                raise RuntimeError(
                    f"{self} tactile frame resolution mismatch: got {frame.shape[1]}x{frame.shape[0]}, expected {self.width}x{self.height}."
                )

    def _resize_inference_frame(self, frame: NDArray[Any]) -> NDArray[Any]:
        if self.width is None or self.height is None:
            return frame
        if frame.shape[1] == self.width and frame.shape[0] == self.height:
            return frame

        try:
            import cv2
        except Exception as e:
            raise ImportError(
                f"{self} requires OpenCV to resize tactile inference frames from "
                f"{frame.shape[1]}x{frame.shape[0]} to {self.width}x{self.height}: {e}"
            ) from e

        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        return resized.astype(frame.dtype, copy=False)

    def _decode_raw_rgb_frame(self, raw_b: bytes, meta: dict[str, Any]) -> NDArray[Any]:
        raw_info = meta.get("raw", {})
        if not raw_info.get("present") or not raw_info.get("shape"):
            raise RuntimeError(f"{self} received tactile frame without raw shape metadata.")

        shape = tuple(raw_info["shape"])
        frame = np.frombuffer(raw_b, dtype=np.uint8).reshape(shape)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.ndim != 3:
            raise RuntimeError(f"{self} expected a 2D/3D tactile frame, got shape {frame.shape}.")

        self._validate_frame_resolution(frame)

        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        elif frame.shape[2] != 3:
            raise RuntimeError(f"{self} expected 1 or 3 tactile channels, got {frame.shape[2]}.")

        return frame

    def _decode_depth_deformation_frame(self, raw_b: bytes, meta: dict[str, Any]) -> NDArray[Any]:
        frame_meta = meta.get("frame", {})
        shape = tuple(frame_meta.get("shape", ()))
        if len(shape) != 3 or shape[2] != 3:
            raise RuntimeError(f"{self} expected uint16 tactile frame shape [H,W,3], got {shape}.")

        dtype = str(frame_meta.get("dtype", "uint16")).lower()
        if dtype != "uint16":
            raise RuntimeError(f"{self} expected uint16 tactile payload in inference mode, got {dtype}.")

        frame = np.frombuffer(raw_b, dtype=np.uint16).reshape(shape)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"{self} expected 3-channel tactile frame, got shape {frame.shape}.")

        return self._resize_inference_frame(frame)

    def _read_from_hardware(self) -> tuple[str, NDArray[Any]]:
        if not self.is_connected or self.socket is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            topic_b, meta_b, raw_b = self.socket.recv_multipart()
        except Exception as e:
            if type(e).__name__ == "Again":
                raise TimeoutError(f"{self} timed out waiting for tactile payload.") from e
            raise

        serial = topic_b.decode("utf-8", errors="ignore").strip()
        side = self._resolve_side(serial)
        if side is None:
            raise RuntimeError(f"{self} received tactile frame for unexpected serial `{serial}`.")

        meta = json.loads(meta_b.decode("utf-8"))
        if self.inference_mode:
            frame = self._decode_depth_deformation_frame(raw_b, meta)
        else:
            frame = self._decode_raw_rgb_frame(raw_b, meta)

        return side, frame

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
                side, frame = self._read_from_hardware()
                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_frames[side] = frame
                    self.latest_timestamps[side] = capture_time
                    self.latest_frame_id += 1
                self.new_frame_event.set()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning("%s read error: %s", self, e)
                time.sleep(0.001)

    def _copy_frames(self) -> dict[str, NDArray[Any] | None]:
        if self.inference_mode:
            return {
                "left_tactile_depth_deformation": None
                if self.latest_frames["left"] is None
                else self.latest_frames["left"].copy(),
                "right_tactile_depth_deformation": None
                if self.latest_frames["right"] is None
                else self.latest_frames["right"].copy(),
            }
        return {
            "left_tactile": None if self.latest_frames["left"] is None else self.latest_frames["left"].copy(),
            "right_tactile": None if self.latest_frames["right"] is None else self.latest_frames["right"].copy(),
        }
