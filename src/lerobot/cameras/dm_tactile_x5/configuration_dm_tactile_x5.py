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

from dataclasses import dataclass
from typing import Any

from ..configs import CameraConfig

__all__ = ["DmTactileCameraX5Config"]


@CameraConfig.register_subclass("dmTactileX5")
@dataclass
class DmTactileCameraX5Config(CameraConfig):
    x5_ip: str | None = None
    port: int = 39010
    topic: str | bytes = b""
    left_serial_number: str | None = None
    right_serial_number: str | None = None
    color_mode: str = "custom"
    channels: int | None = None
    rotation: int | None = None
    roi: list[list[int]] | None = None
    warmup_s: int = 1
    inference_mode: bool = False
    mock: bool = False

    def __post_init__(self) -> None:
        self._validate_identifiers()

        if self.color_mode != "custom":
            raise ValueError(f"`color_mode` must be 'custom', got {self.color_mode}.")

        if not self.x5_ip:
            raise ValueError("`x5_ip` must be provided for the DM tactile X5 camera.")

        if self.topic is None:
            raise ValueError("`topic` must be provided for the DM tactile X5 camera stream.")

        values = (self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError("`width` and `height` must both be set or both be None.")

        if self.rotation is not None:
            raise ValueError(f"`rotation` must be None for DM tactile X5, got {self.rotation}.")

        if self.roi is not None and len(self.roi) != 4:
            raise ValueError(f"`roi` must contain 4 coordinates, got {self.roi}.")

        self.channels = 3

    @property
    def pub_addr(self) -> str:
        return f"tcp://{self.x5_ip}:{self.port}"

    def _validate_identifiers(self) -> None:
        left_has_serial = bool(self.left_serial_number and self.left_serial_number.strip())
        right_has_serial = bool(self.right_serial_number and self.right_serial_number.strip())
        if not left_has_serial and not right_has_serial:
            raise ValueError(
                "At least one tactile side must be configured. Set `left_serial_number` or `right_serial_number`."
            )
