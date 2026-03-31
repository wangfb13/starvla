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
from collections import Counter
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
from .configuration_orbbec import OrbbecCameraConfig
from .point_cloud_utils import create_point_cloud_from_depth_oak_mm

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)

        self.config = config
        self.serial_number = config.serial_number
        self.name = config.name
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.use_pointcloud = config.use_pointcloud
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

        self.pipeline: Any | None = None
        self.pipeline_profile: Any | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_points: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.latest_frame_id = 0
        self.last_consumed_frame_id = 0
        self.new_frame_event: Event = Event()
        self.point_cloud_filter: Any | None = None
        self._sdk: Any | None = None

    def __str__(self) -> str:
        identifier = self.serial_number or self.name
        return f"{self.__class__.__name__}({identifier})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None and self.pipeline_profile is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        try:
            from pyorbbecsdk import Context  # type: ignore
        except Exception as e:
            raise ImportError(f"Could not import pyorbbecsdk: {e}") from e

        cameras: list[dict[str, Any]] = []
        context = Context()
        device_list = context.query_devices()
        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            cameras.append(
                {
                    "type": "Orbbec",
                    "id": str(info.get_serial_number()),
                    "name": str(info.get_name()),
                }
            )
        return cameras

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self._sdk = self._import_sdk()
        sdk = self._sdk

        serial_number = self.serial_number
        if serial_number is None and self.name is not None:
            serial_number = self._find_serial_number_from_name(self.name)
            self.serial_number = serial_number

        if serial_number is None:
            raise ValueError("Unable to resolve Orbbec camera serial number.")

        context = sdk.Context()
        device_list = context.query_devices()
        target_device = device_list.get_device_by_serial_number(serial_number)
        if target_device is None:
            raise ConnectionError(
                f"Failed to find {self}. Run `lerobot-find-cameras orbbec` to inspect available devices."
            )

        self.pipeline = sdk.Pipeline(target_device)
        config = sdk.Config()

        color_profile_list = self.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        if self.fps and self.capture_width and self.capture_height:
            color_profile = color_profile_list.get_video_stream_profile(
                self.capture_width,
                self.capture_height,
                sdk.OBFormat.RGB,
                self.fps,
            )
        else:
            color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        if self.use_depth:
            depth_profile_list = self.pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
            closest_width, closest_height = self._find_closest_resolution(
                (
                    self.capture_width or color_profile.get_width(),
                    self.capture_height or color_profile.get_height(),
                ),
                self._supported_depth_resolutions(depth_profile_list),
            )
            depth_profile = depth_profile_list.get_video_stream_profile(
                closest_width,
                closest_height,
                sdk.OBFormat.Y16,
                self.fps or color_profile.get_fps(),
            )
            config.enable_stream(depth_profile)
            config.set_align_mode(sdk.OBAlignMode.HW_MODE)
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            try:
                self.pipeline.enable_frame_sync()
            except Exception as e:  # nosec B110
                logger.debug("%s frame sync enable failed: %s", self, e)

            if self.use_pointcloud:
                self.point_cloud_filter = sdk.PointCloudFilter()
                self.point_cloud_filter.set_create_point_format(sdk.OBFormat.POINT)

        try:
            self.pipeline_profile = self.pipeline.start(config)
        except Exception as e:
            self.pipeline = None
            self.pipeline_profile = None
            raise ConnectionError(
                f"Failed to open {self}. Run `lerobot-find-cameras orbbec` to inspect available devices."
            ) from e

        self._configure_capture_settings(sdk)
        self._start_read_thread()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < max(self.warmup_s, 1):
                self.async_read(timeout_ms=max(self.warmup_s, 1) * 1000)
                time.sleep(0.1)

        logger.info("%s connected.", self)

    @check_if_not_connected
    def read(self) -> dict[str, NDArray[Any]] | NDArray[Any]:
        self.last_consumed_frame_id = 0
        return self.async_read(timeout_ms=1000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> dict[str, NDArray[Any]] | NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        deadline = time.perf_counter() + timeout_ms / 1000
        while True:
            with self.frame_lock:
                ready = self.latest_color_frame is not None and (
                    not self.use_depth or self.latest_depth_frame is not None
                )
                frame_id = self.latest_frame_id
                if ready and frame_id != self.last_consumed_frame_id:
                    self.last_consumed_frame_id = frame_id
                    return self._build_output(copy_arrays=True)

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"{self} timed out waiting for a new frame.")

            self.new_frame_event.wait(remaining)
            self.new_frame_event.clear()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> dict[str, NDArray[Any]] | NDArray[Any]:
        with self.frame_lock:
            if self.latest_timestamp is None:
                raise RuntimeError(f"{self} has not captured any frames yet.")
            age_ms = (time.perf_counter() - self.latest_timestamp) * 1000
            if age_ms > max_age_ms:
                raise TimeoutError(f"{self} latest frame is older than {max_age_ms}ms.")
            return self._build_output(copy_arrays=True)

    @check_if_not_connected
    def read_depth(self) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"{self} depth stream is not enabled.")

        with self.frame_lock:
            if self.latest_depth_frame is None:
                raise RuntimeError(f"{self} has not captured any depth frames yet.")
            return self.latest_depth_frame.copy()

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_read_thread()
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.pipeline_profile = None
        self.point_cloud_filter = None
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_points = None
            self.latest_timestamp = None
            self.latest_frame_id = 0
            self.last_consumed_frame_id = 0
            self.new_frame_event.clear()

    def _import_sdk(self) -> Any:
        try:
            import pyorbbecsdk as sdk  # type: ignore
        except Exception as e:
            raise ImportError(f"Could not import pyorbbecsdk: {e}") from e
        return sdk

    def _find_serial_number_from_name(self, name: str) -> str:
        camera_infos = self.find_cameras()
        camera_names = [cam["name"] for cam in camera_infos]
        if Counter(camera_names)[name] > 1:
            raise ValueError(
                f"Multiple Orbbec cameras named '{name}' were detected. Use `serial_number` instead."
            )

        for cam in camera_infos:
            if cam["name"] == name:
                return str(cam["id"])

        raise ValueError(f"No Orbbec camera found with name '{name}'.")

    def _supported_depth_resolutions(self, depth_profile_list: Any) -> list[tuple[int, int]]:
        supported: list[tuple[int, int]] = []
        for index in range(depth_profile_list.get_count()):
            profile = depth_profile_list.get_stream_profile_by_index(index)
            if profile.is_video_stream_profile():
                video_profile = profile.as_video_stream_profile()
                resolution = (video_profile.get_width(), video_profile.get_height())
                if resolution not in supported:
                    supported.append(resolution)
        return supported

    def _find_closest_resolution(
        self, target_resolution: tuple[int, int], supported_resolutions: list[tuple[int, int]]
    ) -> tuple[int, int]:
        if not supported_resolutions:
            raise RuntimeError(f"{self} does not report any supported depth resolutions.")

        def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        return min(supported_resolutions, key=lambda resolution: distance(resolution, target_resolution))

    def _configure_capture_settings(self, sdk: Any) -> None:
        if self.pipeline_profile is None:
            raise RuntimeError(f"{self} pipeline profile is not initialized.")

        stream = self.pipeline_profile.get_stream(sdk.OBSensorType.COLOR_SENSOR).as_video_stream_profile()
        if self.fps is None:
            self.fps = stream.get_fps()

        if self.capture_width is None or self.capture_height is None:
            self.capture_width = int(stream.get_width())
            self.capture_height = int(stream.get_height())

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width = self.capture_height
            self.height = self.capture_width
        else:
            self.width = self.capture_width
            self.height = self.capture_height

    def _read_from_hardware(self) -> None:
        if self.pipeline is None or self._sdk is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sdk = self._sdk
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            raise TimeoutError(f"{self} did not return frames in time.")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} returned an empty color frame.")

        color_image = self._extract_color_frame(color_frame, sdk)

        depth_image = None
        points = None
        if self.use_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                raise RuntimeError(f"{self} returned an empty depth frame.")
            depth_image = self._extract_depth_frame(depth_frame)

            if self.use_pointcloud and self.point_cloud_filter is not None:
                point_cloud_frame = self.point_cloud_filter.process(depth_frame)
                if point_cloud_frame is not None:
                    raw_points = np.asarray(self.point_cloud_filter.calculate(point_cloud_frame))
                    points = create_point_cloud_from_depth_oak_mm(raw_points, far=3000, near=20)

        capture_time = time.perf_counter()
        with self.frame_lock:
            self.latest_color_frame = color_image
            self.latest_depth_frame = depth_image
            self.latest_points = points
            self.latest_timestamp = capture_time
            self.latest_frame_id += 1
        self.new_frame_event.set()

    def _extract_color_frame(self, color_frame: Any, sdk: Any) -> NDArray[Any]:
        width = int(color_frame.get_width())
        height = int(color_frame.get_height())
        buffer = np.asarray(color_frame.get_data())
        frame_format = color_frame.get_format()

        if frame_format == sdk.OBFormat.RGB:
            color_image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
            if self.color_mode == ColorMode.BGR:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        else:
            decoded = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(f"{self} failed to decode color frame.")
            color_image = decoded
            if self.color_mode == ColorMode.RGB:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        return color_image

    def _extract_depth_frame(self, depth_frame: Any) -> NDArray[Any]:
        width = int(depth_frame.get_width())
        height = int(depth_frame.get_height())
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > 200) & (depth_data < 5000), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)

        if self.rotation is not None:
            depth_data = cv2.rotate(depth_data, self.rotation)

        return depth_data

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
                self._read_from_hardware()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning("%s read error: %s", self, e)
                time.sleep(0.01)

    def _build_output(self, *, copy_arrays: bool) -> dict[str, NDArray[Any]] | NDArray[Any]:
        color = self.latest_color_frame
        depth = self.latest_depth_frame
        points = self.latest_points

        if color is None:
            raise RuntimeError(f"{self} has no color frame available.")

        color_out = color.copy() if copy_arrays else color
        if not self.use_depth:
            return color_out

        output: dict[str, NDArray[Any]] = {"color": color_out}
        if depth is not None:
            output["depth"] = depth.copy() if copy_arrays else depth
        if points is not None:
            output["points"] = points.copy() if copy_arrays else points
        return output
