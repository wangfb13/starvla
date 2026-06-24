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
from typing import Any

import yaml

from lerobot.utils.constants import HF_LEROBOT_HOME

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["WirstFisheyeCameraX5Config", "ColorMode", "Cv2Rotation"]


def _resolve_intrinsics_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path.replace("${HF_LEROBOT_HOME}", str(HF_LEROBOT_HOME))


@CameraConfig.register_subclass("wirstFisheyeX5")
@dataclass
class WirstFisheyeCameraX5Config(CameraConfig):
    x5_ip: str | None = None
    port: int = 5552
    frame_header_size: int = 4
    frame_header_format: str = ">I"
    color_mode: ColorMode = ColorMode.RGB
    channels: int | None = None
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    intrinsics_file: str | None = None
    intrinsics: dict | None = None
    enable_undistort_crop: bool = False
    undistort_crop_size: int | None = None
    warmup_s: int = 1
    mock: bool = False
    use_gpu_decode: bool = False

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.channels = 3

        if not self.x5_ip:
            raise ValueError("`x5_ip` must be provided for the wrist fisheye X5 camera.")

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )

        if self.undistort_crop_size is not None and self.undistort_crop_size <= 0:
            raise ValueError("`undistort_crop_size` must be a positive integer when provided.")

        self.intrinsics_file = _resolve_intrinsics_path(self.intrinsics_file)
        if self.enable_undistort_crop and not self.intrinsics_file and not self.intrinsics:
            raise ValueError(
                "`enable_undistort_crop=True` requires either `intrinsics_file` or preloaded `intrinsics`."
            )
        if self.intrinsics_file and not self.intrinsics:
            self.intrinsics = self._load_intrinsics(Path(self.intrinsics_file))

    def _load_intrinsics(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Please generate the wrist fisheye camera intrinsics file first: {path}"
            )

        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open() as f:
                intrinsics_data = json.load(f)

            dist_coeffs = intrinsics_data.get("dist_coeffs", intrinsics_data.get("distortion_coeffs"))
            image_size = intrinsics_data.get("image_size", intrinsics_data.get("resolution"))
            required_values = {
                "camera_matrix": intrinsics_data.get("camera_matrix"),
                "dist_coeffs/distortion_coeffs": dist_coeffs,
                "image_size/resolution": image_size,
            }
            missing_keys = [key for key, value in required_values.items() if value is None]
            if missing_keys:
                raise KeyError(
                    "Expected wrist fisheye JSON intrinsics to contain camera_matrix, "
                    "dist_coeffs/distortion_coeffs and image_size/resolution, "
                    f"missing {missing_keys}."
                )

            camera_matrix = intrinsics_data["camera_matrix"]
            if intrinsics_data.get("distortion_model") not in {None, "equidistant", "fisheye"}:
                raise ValueError(
                    f"Unsupported distortion_model={intrinsics_data.get('distortion_model')!r} in {path}."
                )
        elif suffix in {".yaml", ".yml", ".yam"}:
            with path.open() as f:
                intrinsics_data = yaml.safe_load(f)

            if not isinstance(intrinsics_data, dict) or not intrinsics_data:
                raise ValueError(f"Expected non-empty YAML camera intrinsics in {path}.")

            cam_key = "cam0" if "cam0" in intrinsics_data else next(iter(intrinsics_data))
            cam_data = intrinsics_data[cam_key]
            required_keys = ["intrinsics", "distortion_coeffs", "resolution"]
            missing_keys = [key for key in required_keys if key not in cam_data]
            if missing_keys:
                raise KeyError(
                    f"Expected keys {required_keys} in YAML intrinsics file under `{cam_key}`, missing {missing_keys}."
                )

            fx, fy, cx, cy = cam_data["intrinsics"]
            camera_matrix = [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
            dist_coeffs = cam_data["distortion_coeffs"]
            image_size = cam_data["resolution"]
        else:
            raise ValueError(
                f"Unsupported intrinsics file extension `{suffix}` for {path}. Expected .json or .yaml/.yml."
            )

        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise ValueError(
                f"Expected `image_size`/`resolution` to contain [width, height], got {image_size}."
            )
        image_width, image_height = int(image_size[0]), int(image_size[1])
        target_resolution = f"{image_width}x{image_height}"

        return {
            target_resolution: {
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "image_size": [image_width, image_height],
            }
        }
