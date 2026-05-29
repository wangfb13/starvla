# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
    "to_keyframe_relative_actions",
    "to_keyframe_absolute_actions",
]


LOGGER = logging.getLogger(__name__)


def _rotation_6d_to_matrix(rot6d: Tensor) -> Tensor:
    if rot6d.ndim < 1 or rot6d.shape[-1] != 6:
        raise ValueError(f"Expected trailing rot6d dim 6, got shape {tuple(rot6d.shape)}")
    col1 = rot6d[..., 0:3]
    col2 = rot6d[..., 3:6]
    basis1 = F.normalize(col1, dim=-1)
    proj = (basis1 * col2).sum(dim=-1, keepdim=True) * basis1
    basis2 = F.normalize(col2 - proj, dim=-1)
    basis3 = torch.cross(basis1, basis2, dim=-1)
    return torch.stack((basis1, basis2, basis3), dim=-1)


def _matrix_to_rotation_6d(rot_mats: Tensor) -> Tensor:
    if rot_mats.ndim < 2 or rot_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected trailing shape (3, 3), got {tuple(rot_mats.shape)}")
    col1 = rot_mats[..., :, 0]
    col2 = rot_mats[..., :, 1]
    return torch.cat((col1, col2), dim=-1)


def _parse_keyframe_spec(spec: str) -> tuple[int, int]:
    parts = str(spec).split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid relative_action_keyframes entry '{spec}'. Expected 'pos_start:rot6d_start'.")
    return int(parts[0]), int(parts[1])


def _keyframe_specs(keyframes: Sequence[str] | None, exclude_keys: Sequence[str] | None) -> list[tuple[str, int, int]]:
    excluded = {str(item) for item in (exclude_keys or [])}
    specs: list[tuple[str, int, int]] = []
    for spec in keyframes or []:
        if str(spec) in excluded:
            continue
        pos_start, rot_start = _parse_keyframe_spec(str(spec))
        specs.append((str(spec), pos_start, rot_start))
    return specs


def _mask_keyframe_dims(mask: list[bool], specs: Sequence[tuple[str, int, int]]) -> list[bool]:
    masked = list(mask)
    for _, pos_start, rot_start in specs:
        for idx in range(pos_start, pos_start + 3):
            if 0 <= idx < len(masked):
                masked[idx] = False
        for idx in range(rot_start, rot_start + 6):
            if 0 <= idx < len(masked):
                masked[idx] = False
    return masked


def _compatible_keyframe_specs(
    specs: Sequence[tuple[str, int, int]],
    *,
    action_dim: int,
    state_dim: int,
) -> list[tuple[str, int, int]]:
    compatible: list[tuple[str, int, int]] = []
    for spec, pos_start, rot_start in specs:
        required_dim = max(pos_start + 3, rot_start + 6)
        if action_dim >= required_dim and state_dim >= required_dim:
            compatible.append((spec, pos_start, rot_start))
    return compatible


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


def to_keyframe_relative_actions(
    actions: Tensor,
    state: Tensor,
    keyframes: Sequence[str],
    exclude_keys: Sequence[str] | None = None,
) -> Tensor:
    """Convert keyframed xyz/rot6d actions to local SE(3) relative actions.

    Each keyframe spec is `pos_start:rot6d_start`. For that block:
    rel_pos = R_state^T @ (pos_action - pos_state)
    rel_rot = R_state^T @ R_action
    """
    specs = _keyframe_specs(keyframes, exclude_keys)
    if not specs:
        return actions

    action_out = actions.clone()
    if state.device != action_out.device or state.dtype != action_out.dtype:
        state = state.to(device=action_out.device, dtype=action_out.dtype)

    if state.ndim == 1:
        state_ref = state.unsqueeze(0)
    else:
        state_ref = state
    if action_out.ndim == 2:
        action_seq = action_out.unsqueeze(1)
        squeeze_seq = True
    else:
        action_seq = action_out
        squeeze_seq = False

    for spec, pos_start, rot_start in specs:
        pos_slice = slice(pos_start, pos_start + 3)
        rot_slice = slice(rot_start, rot_start + 6)
        required_dim = max(pos_start + 3, rot_start + 6)
        if state_ref.shape[-1] < required_dim or action_seq.shape[-1] < required_dim:
            LOGGER.warning(
                "Skipping relative-action keyframe %s because dims are incompatible: "
                "state_dim=%s action_dim=%s required_dim=%s",
                spec,
                state_ref.shape[-1],
                action_seq.shape[-1],
                required_dim,
            )
            continue

        current_pos = state_ref[:, pos_slice].unsqueeze(1)
        current_rot = _rotation_6d_to_matrix(state_ref[:, rot_slice])
        target_pos = action_seq[:, :, pos_slice]
        target_rot = _rotation_6d_to_matrix(action_seq[:, :, rot_slice])

        current_rot_t = current_rot.transpose(-1, -2).unsqueeze(1)
        rel_pos = torch.matmul(current_rot_t, (target_pos - current_pos).unsqueeze(-1)).squeeze(-1)
        rel_rot = torch.matmul(current_rot_t, target_rot)

        action_seq[:, :, pos_slice] = rel_pos
        action_seq[:, :, rot_slice] = _matrix_to_rotation_6d(rel_rot)

    if squeeze_seq:
        return action_seq.squeeze(1)
    return action_seq


def to_keyframe_absolute_actions(
    actions: Tensor,
    state: Tensor,
    keyframes: Sequence[str],
    exclude_keys: Sequence[str] | None = None,
) -> Tensor:
    """Convert local SE(3) relative keyframed actions back to absolute xyz/rot6d."""
    specs = _keyframe_specs(keyframes, exclude_keys)
    if not specs:
        return actions

    action_out = actions.clone()
    if state.device != action_out.device or state.dtype != action_out.dtype:
        state = state.to(device=action_out.device, dtype=action_out.dtype)

    if state.ndim == 1:
        state_ref = state.unsqueeze(0)
    else:
        state_ref = state
    if action_out.ndim == 2:
        action_seq = action_out.unsqueeze(1)
        squeeze_seq = True
    else:
        action_seq = action_out
        squeeze_seq = False

    for spec, pos_start, rot_start in specs:
        pos_slice = slice(pos_start, pos_start + 3)
        rot_slice = slice(rot_start, rot_start + 6)
        required_dim = max(pos_start + 3, rot_start + 6)
        if state_ref.shape[-1] < required_dim or action_seq.shape[-1] < required_dim:
            LOGGER.warning(
                "Skipping absolute-action keyframe %s because dims are incompatible: "
                "state_dim=%s action_dim=%s required_dim=%s",
                spec,
                state_ref.shape[-1],
                action_seq.shape[-1],
                required_dim,
            )
            continue

        current_pos = state_ref[:, pos_slice].unsqueeze(1)
        current_rot = _rotation_6d_to_matrix(state_ref[:, rot_slice])
        rel_pos = action_seq[:, :, pos_slice]
        rel_rot = _rotation_6d_to_matrix(action_seq[:, :, rot_slice])

        abs_pos = torch.matmul(current_rot.unsqueeze(1), rel_pos.unsqueeze(-1)).squeeze(-1) + current_pos
        abs_rot = torch.matmul(current_rot.unsqueeze(1), rel_rot)

        action_seq[:, :, pos_slice] = abs_pos
        action_seq[:, :, rot_slice] = _matrix_to_rotation_6d(abs_rot)

    if squeeze_seq:
        return action_seq.squeeze(1)
    return action_seq


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    relative_action_keyframes: list[str] = field(default_factory=list)
    relative_exclude_keys: list[str] = field(default_factory=list)
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # Always cache state for the paired AbsoluteActionsProcessorStep
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        specs = _keyframe_specs(self.relative_action_keyframes, self.relative_exclude_keys)
        if specs:
            compatible_specs = _compatible_keyframe_specs(
                specs,
                action_dim=action.shape[-1],
                state_dim=state.shape[-1],
            )
            action = to_keyframe_relative_actions(
                action,
                state,
                self.relative_action_keyframes,
                self.relative_exclude_keys,
            )
            mask = _mask_keyframe_dims(mask, compatible_specs)
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "relative_action_keyframes": self.relative_action_keyframes,
            "relative_exclude_keys": self.relative_exclude_keys,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        if self.relative_step._last_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        specs = _keyframe_specs(
            self.relative_step.relative_action_keyframes,
            self.relative_step.relative_exclude_keys,
        )
        if specs:
            compatible_specs = _compatible_keyframe_specs(
                specs,
                action_dim=action.shape[-1],
                state_dim=self.relative_step._last_state.shape[-1],
            )
            mask = _mask_keyframe_dims(mask, compatible_specs)
        action = to_absolute_actions(action, self.relative_step._last_state, mask)
        if specs:
            action = to_keyframe_absolute_actions(
                action,
                self.relative_step._last_state,
                self.relative_step.relative_action_keyframes,
                self.relative_step.relative_exclude_keys,
            )
        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
