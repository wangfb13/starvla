#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE

ARM_NAMES = ("left", "right")
ARM_CANONICAL_DIM = 10
CANONICAL_ACTION_DIM = 20
CANONICAL_STATE_DIM = 22


def canonical_action_feature_names() -> list[str]:
    names: list[str] = []
    for arm in ARM_NAMES:
        names.extend(
            [
                f"{arm}_x",
                f"{arm}_y",
                f"{arm}_z",
                f"{arm}_rot6d_0",
                f"{arm}_rot6d_1",
                f"{arm}_rot6d_2",
                f"{arm}_rot6d_3",
                f"{arm}_rot6d_4",
                f"{arm}_rot6d_5",
                f"gripper_{arm}",
            ]
        )
    return names


def canonical_state_feature_names() -> list[str]:
    return canonical_action_feature_names() + ["left_valid", "right_valid"]


def extend_relative_exclude_joints(exclude_joints: list[str]) -> list[str]:
    updated = list(exclude_joints)
    if "rot6d" not in updated:
        updated.append("rot6d")
    return updated


@dataclass(frozen=True)
class ArmSourceSpec:
    pos_indices: tuple[int, int, int] | None
    rot6d_indices: tuple[int, int, int, int, int, int] | None
    quat_indices: tuple[int, int, int, int] | None
    gripper_index: int | None


def _find_index(names: list[str] | None, candidates: list[str]) -> int | None:
    if names is None:
        return None
    lowered = {name.lower(): idx for idx, name in enumerate(names)}
    for candidate in candidates:
        idx = lowered.get(candidate.lower())
        if idx is not None:
            return idx
    return None


def _find_all_indices(names: list[str] | None, expected: list[str]) -> tuple[int, ...] | None:
    if names is None:
        return None
    indices = []
    for expected_name in expected:
        idx = _find_index(names, [expected_name])
        if idx is None:
            return None
        indices.append(idx)
    return tuple(indices)


def build_arm_source_spec(names: list[str] | None, arm: str) -> ArmSourceSpec:
    pos_indices = _find_all_indices(names, [f"{arm}_x", f"{arm}_y", f"{arm}_z"])
    rot6d_indices = _find_all_indices(names, [f"{arm}_rot6d_{i}" for i in range(6)])
    quat_indices = _find_all_indices(names, [f"{arm}_qx", f"{arm}_qy", f"{arm}_qz", f"{arm}_qw"])
    gripper_index = _find_index(names, [f"gripper_{arm}", f"{arm}_gripper", "gripper"])
    return ArmSourceSpec(
        pos_indices=pos_indices if pos_indices is not None and len(pos_indices) == 3 else None,
        rot6d_indices=rot6d_indices if rot6d_indices is not None and len(rot6d_indices) == 6 else None,
        quat_indices=quat_indices if quat_indices is not None and len(quat_indices) == 4 else None,
        gripper_index=gripper_index,
    )


def _take_last_dim(tensor: Tensor, indices: tuple[int, ...] | list[int]) -> Tensor:
    index = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, dim=-1, index=index)


def _valid_mask(values: Tensor, missing_value: float | None) -> Tensor:
    valid = torch.isfinite(values).all(dim=-1)
    if missing_value is not None:
        missing_tensor = torch.tensor(missing_value, dtype=values.dtype, device=values.device)
        valid = valid & ~torch.isclose(values, missing_tensor).any(dim=-1)
    return valid


def _identity_rot6d(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=dtype)


def quat_xyzw_to_rot6d(quat: Tensor) -> Tensor:
    quat = quat.to(torch.float32)
    quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    safe_quat = torch.where(quat_norm > 1e-8, quat / quat_norm, torch.tensor([0.0, 0.0, 0.0, 1.0], device=quat.device))
    qx, qy, qz, qw = safe_quat.unbind(dim=-1)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    col0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)], dim=-1)
    col1 = torch.stack([2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)], dim=-1)
    return torch.cat([col0, col1], dim=-1)


def _canonicalize_arm(tensor: Tensor, spec: ArmSourceSpec, missing_value: float | None) -> tuple[Tensor, Tensor, Tensor]:
    leading_shape = tensor.shape[:-1]
    zeros = torch.zeros((*leading_shape, 1), device=tensor.device, dtype=tensor.dtype)
    pos_default = torch.zeros((*leading_shape, 3), device=tensor.device, dtype=tensor.dtype)
    rot_default = _identity_rot6d(tensor.device, tensor.dtype).expand(*leading_shape, 6)

    pos = pos_default
    pos_mask = torch.zeros((*leading_shape, 3), device=tensor.device, dtype=tensor.dtype)
    if spec.pos_indices is not None:
        raw_pos = _take_last_dim(tensor, spec.pos_indices)
        valid = _valid_mask(raw_pos, missing_value)
        pos = torch.where(valid.unsqueeze(-1), raw_pos, pos_default)
        pos_mask = valid.unsqueeze(-1).expand_as(pos_mask).to(tensor.dtype)

    rot = rot_default
    rot_mask = torch.zeros((*leading_shape, 6), device=tensor.device, dtype=tensor.dtype)
    if spec.rot6d_indices is not None:
        raw_rot6d = _take_last_dim(tensor, spec.rot6d_indices)
        valid = _valid_mask(raw_rot6d, missing_value)
        rot = torch.where(valid.unsqueeze(-1), raw_rot6d, rot_default)
        rot_mask = valid.unsqueeze(-1).expand_as(rot_mask).to(tensor.dtype)
    elif spec.quat_indices is not None:
        raw_quat = _take_last_dim(tensor, spec.quat_indices)
        valid = _valid_mask(raw_quat, missing_value)
        converted_rot = quat_xyzw_to_rot6d(raw_quat).to(tensor.dtype)
        rot = torch.where(valid.unsqueeze(-1), converted_rot, rot_default)
        rot_mask = valid.unsqueeze(-1).expand_as(rot_mask).to(tensor.dtype)

    gripper = zeros
    gripper_mask = torch.zeros((*leading_shape, 1), device=tensor.device, dtype=tensor.dtype)
    if spec.gripper_index is not None:
        raw_gripper = _take_last_dim(tensor, [spec.gripper_index])
        valid = _valid_mask(raw_gripper, missing_value)
        gripper = torch.where(valid.unsqueeze(-1), raw_gripper, zeros)
        gripper_mask = valid.unsqueeze(-1).expand_as(gripper_mask).to(tensor.dtype)

    canonical = torch.cat([pos, rot, gripper], dim=-1)
    mask = torch.cat([pos_mask, rot_mask, gripper_mask], dim=-1)
    arm_valid = (mask.sum(dim=-1) > 0).to(tensor.dtype).unsqueeze(-1)
    return canonical, mask, arm_valid


def canonicalize_bimanual_state_action(
    *,
    state: Tensor,
    action: Tensor | None,
    source_state_names: list[str] | None,
    source_action_names: list[str] | None,
    missing_value: float | None,
) -> dict[str, Tensor]:
    state_outputs = []
    state_masks = []
    state_valid_bits = []
    action_outputs: list[Tensor] = []
    action_masks: list[Tensor] = []

    for arm in ARM_NAMES:
        state_spec = build_arm_source_spec(source_state_names, arm)
        action_spec = build_arm_source_spec(source_action_names, arm) if action is not None else None

        state_canonical, state_mask, state_valid = _canonicalize_arm(state, state_spec, missing_value)

        state_outputs.append(state_canonical)
        state_masks.append(state_mask)
        state_valid_bits.append(state_valid)
        if action is not None and action_spec is not None:
            action_canonical, action_mask, _ = _canonicalize_arm(action, action_spec, missing_value)
            action_outputs.append(action_canonical)
            action_masks.append(action_mask)

    canonical_state = torch.cat(state_outputs + state_valid_bits, dim=-1)
    canonical_state_mask = torch.cat(state_masks, dim=-1)

    output = {
        OBS_STATE: canonical_state,
        "state_mask": canonical_state_mask,
    }
    if action is not None:
        output[ACTION] = torch.cat(action_outputs, dim=-1)
        output["action_mask"] = torch.cat(action_masks, dim=-1)
    return output


def _default_dim_stats(name: str) -> dict[str, float]:
    if name.endswith("_valid"):
        return {"min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0, "q01": 0.0, "q10": 0.0, "q50": 0.0, "q90": 1.0, "q99": 1.0}
    if "gripper" in name:
        return {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0, "q01": -1.0, "q10": -1.0, "q50": 0.0, "q90": 1.0, "q99": 1.0}
    return {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0, "q01": -1.0, "q10": -0.5, "q50": 0.0, "q90": 0.5, "q99": 1.0}


def _compute_masked_stats(values: Tensor, mask: Tensor, names: list[str]) -> dict[str, np.ndarray]:
    values_np = values.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    stats = {key: [] for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")}

    for dim, name in enumerate(names):
        dim_values = values_np[..., dim]
        dim_mask = mask_np[..., dim]
        valid_values = dim_values[dim_mask]
        if valid_values.size == 0:
            defaults = _default_dim_stats(name)
            for key, value in defaults.items():
                stats[key].append(value)
            continue

        stats["min"].append(float(valid_values.min()))
        stats["max"].append(float(valid_values.max()))
        stats["mean"].append(float(valid_values.mean()))
        stats["std"].append(float(valid_values.std()))
        stats["q01"].append(float(np.quantile(valid_values, 0.01)))
        stats["q10"].append(float(np.quantile(valid_values, 0.10)))
        stats["q50"].append(float(np.quantile(valid_values, 0.50)))
        stats["q90"].append(float(np.quantile(valid_values, 0.90)))
        stats["q99"].append(float(np.quantile(valid_values, 0.99)))

    output = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}
    output["count"] = np.asarray([values.shape[0]], dtype=np.float32)
    return output


def build_canonical_bimanual_stats(
    dataset: Any,
    *,
    source_state_names: list[str] | None,
    source_action_names: list[str] | None,
    missing_value: float | None,
    base_stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if dataset is None:
        raise ValueError("dataset is required to build canonical rot6d statistics")

    hf_dataset = dataset.hf_dataset
    raw_states = np.asarray(hf_dataset[OBS_STATE], dtype=np.float32)
    raw_actions = np.asarray(hf_dataset[ACTION], dtype=np.float32)

    canonical = canonicalize_bimanual_state_action(
        state=torch.from_numpy(raw_states),
        action=torch.from_numpy(raw_actions),
        source_state_names=source_state_names,
        source_action_names=source_action_names,
        missing_value=missing_value,
    )

    stats = deepcopy(base_stats if base_stats is not None else dataset.meta.stats)
    state_mask = torch.cat(
        [
            canonical["state_mask"],
            torch.ones((canonical[OBS_STATE].shape[0], 2), dtype=canonical[OBS_STATE].dtype),
        ],
        dim=-1,
    )

    stats[OBS_STATE] = _compute_masked_stats(canonical[OBS_STATE], state_mask, canonical_state_feature_names())
    stats[ACTION] = _compute_masked_stats(canonical[ACTION], canonical["action_mask"], canonical_action_feature_names())
    return stats
