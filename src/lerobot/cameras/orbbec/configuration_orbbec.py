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
from dataclasses import dataclass
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["OrbbecCameraConfig", "ColorMode", "Cv2Rotation"]


def _resolve_intrinsics_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path.replace("${HF_LEROBOT_HOME}", str(HF_LEROBOT_HOME))


@CameraConfig.register_subclass("OrbbecCamera")
@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    name: str | None = None
    serial_number: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    channels: int | None = None
    use_depth: bool = False
    use_pointcloud: bool = False
    force_hardware_reset: bool = True
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    intrinsics_file: str | None = None
    intrinsics: dict | None = None
    warmup_s: int = 1
    mock: bool = False

    def __post_init__(self) -> None:
        if not bool(self.name) and not bool(self.serial_number):
            raise ValueError(
                f"One of `name` or `serial_number` must be set, got {self.name=} and {self.serial_number=}."
            )

        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.channels = 3

        if self.use_pointcloud and not self.use_depth:
            raise ValueError("Enabling point cloud output requires `use_depth=True`.")

        values = (self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError("`width` and `height` must both be set or both be None.")

        self.intrinsics_file = _resolve_intrinsics_path(self.intrinsics_file)
        if self.intrinsics_file and not self.intrinsics:
            self.intrinsics = self._load_intrinsics(Path(self.intrinsics_file))

    def _load_intrinsics(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"Please generate the Orbbec intrinsics file first: {path}"
            )

        with path.open() as f:
            intrinsics_data = json.load(f)

        target_resolutions = ["1280x720", "640x480", "640x360"]
        return {
            resolution: {
                "fx": calib["intrinsics"]["fx"],
                "fy": calib["intrinsics"]["fy"],
                "ppx": calib["intrinsics"]["cx"],
                "ppy": calib["intrinsics"]["cy"],
            }
            for resolution in target_resolutions
            for calib in intrinsics_data["color_calibrations"]
            if calib["resolution"] == resolution
        }
