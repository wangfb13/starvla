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


def _center_crop_bounds(width: int, height: int, crop_size: int) -> tuple[int, int]:
    if crop_size > width or crop_size > height:
        raise ValueError(f"crop_size={crop_size} exceeds frame size {width}x{height}")
    x0 = (width - crop_size) // 2
    y0 = (height - crop_size) // 2
    return x0, y0


def _center_crop(image: NDArray[Any], crop_size: int) -> NDArray[Any]:
    height, width = image.shape[:2]
    x0, y0 = _center_crop_bounds(width, height, crop_size)
    return image[y0 : y0 + crop_size, x0 : x0 + crop_size]


def _scale_camera_matrix(
    camera_matrix: NDArray[Any],
    source_wh: tuple[int, int],
    target_wh: tuple[int, int],
) -> NDArray[Any]:
    src_w, src_h = source_wh
    dst_w, dst_h = target_wh
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid calibration resolution {source_wh}")

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    scaled = camera_matrix.copy().astype(np.float64)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


class WristFisheyeX5Camera(Camera):
    def __init__(self, config: WirstFisheyeCameraX5Config):
        super().__init__(config)

        self.config = config
        self.ip = config.x5_ip
        self.port = config.port
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.intrinsics = config.intrinsics
        self.enable_undistort_crop = config.enable_undistort_crop
        self.undistort_crop_size = config.undistort_crop_size
        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self._undistort_cache: dict[tuple[int, int, int], tuple[NDArray[Any], NDArray[Any], int]] = {}

        self.capture_width = self.width
        self.capture_height = self.height
        if self.width and self.height and self.rotation in [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]:
            self.capture_width, self.capture_height = self.height, self.width

        if self.enable_undistort_crop and self.capture_width and self.capture_height:
            crop_size = self.undistort_crop_size or min(self.capture_width, self.capture_height)
            if crop_size > min(self.capture_width, self.capture_height):
                raise ValueError(
                    f"`undistort_crop_size={crop_size}` exceeds configured stream size "
                    f"{self.capture_width}x{self.capture_height}"
                )
            self.width = crop_size
            self.height = crop_size

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

    def _get_intrinsics_entry(
        self,
        frame_wh: tuple[int, int] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any], tuple[int, int]]:
        if not self.intrinsics:
            raise ValueError("Fisheye undistort is enabled but no intrinsics are available.")

        preferred_key = None if frame_wh is None else f"{frame_wh[0]}x{frame_wh[1]}"
        selected_key = preferred_key if preferred_key in self.intrinsics else next(iter(self.intrinsics))
        entry = self.intrinsics[selected_key]
        camera_matrix = np.array(entry["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(entry["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

        image_size = entry.get("image_size")
        if image_size is None:
            try:
                width_str, height_str = selected_key.split("x", maxsplit=1)
                image_size = [int(width_str), int(height_str)]
            except Exception as exc:
                raise ValueError(
                    "Unable to infer fisheye calibration image_size from intrinsics entry. "
                    f"Got key={selected_key!r}, entry keys={list(entry.keys())}"
                ) from exc

        calib_wh = (int(image_size[0]), int(image_size[1]))
        return camera_matrix, dist_coeffs, calib_wh

    def _get_undistort_maps(self, frame_shape: tuple[int, ...]) -> tuple[NDArray[Any], NDArray[Any], int]:
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        crop_size = self.undistort_crop_size or min(frame_w, frame_h)
        cache_key = (frame_w, frame_h, crop_size)
        cached = self._undistort_cache.get(cache_key)
        if cached is not None:
            return cached

        camera_matrix, dist_coeffs, calib_wh = self._get_intrinsics_entry((frame_w, frame_h))
        scaled_camera_matrix = _scale_camera_matrix(camera_matrix, calib_wh, (frame_w, frame_h))
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            scaled_camera_matrix,
            dist_coeffs,
            np.eye(3),
            scaled_camera_matrix.copy(),
            (frame_w, frame_h),
            cv2.CV_16SC2,
        )
        cached = (map1, map2, crop_size)
        self._undistort_cache[cache_key] = cached
        return cached

    def _maybe_undistort_crop(self, frame: NDArray[Any]) -> NDArray[Any]:
        if not self.enable_undistort_crop:
            return frame

        map1, map2, crop_size = self._get_undistort_maps(frame.shape)
        undistorted = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return _center_crop(undistorted, crop_size)

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

        frame_bgr = self._maybe_undistort_crop(frame_bgr)

        if self.capture_width and self.capture_height:
            expected_width = self.width if self.enable_undistort_crop else self.capture_width
            expected_height = self.height if self.enable_undistort_crop else self.capture_height
            if frame_bgr.shape[1] != expected_width or frame_bgr.shape[0] != expected_height:
                logger.debug(
                    "%s resolution mismatch: got %sx%s, expected %sx%s",
                    self,
                    frame_bgr.shape[1],
                    frame_bgr.shape[0],
                    expected_width,
                    expected_height,
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
