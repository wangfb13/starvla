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

import numpy as np
from numpy.typing import NDArray


def grid_sample_pcd(point_cloud: NDArray[np.floating], grid_size: float = 5.0) -> NDArray[np.floating]:
    coords = point_cloud[:, :3]
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    _, indices = np.unique(keys, return_index=True)
    return point_cloud[indices]


def create_point_cloud_from_depth_oak_mm(
    cloud: NDArray[np.floating],
    *,
    far: float = 3000,
    near: float = 100,
    num_points: int = 4096,
) -> NDArray[np.float32]:
    clip_low_x = -500
    clip_high_x = 500
    cloud = cloud[(cloud[:, 0] > clip_low_x) & (cloud[:, 0] < clip_high_x)]
    cloud = cloud[(cloud[:, 2] < far) & (cloud[:, 2] > near)]
    cloud = grid_sample_pcd(cloud, grid_size=5)

    if num_points > cloud.shape[0]:
        num_pad = num_points - cloud.shape[0]
        pad_points = np.zeros((num_pad, 3), dtype=cloud.dtype)
        cloud = np.concatenate([cloud, pad_points], axis=0)
    else:
        selected_idx = np.random.choice(cloud.shape[0], num_points, replace=True)
        cloud = cloud[selected_idx]

    return cloud.astype(np.float32, copy=False)

