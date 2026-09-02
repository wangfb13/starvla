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
    relative_state_history_to_current: bool = False
    relative_scalar_state_history_indices: list[int] = field(default_factory=list)
    # Inference-only adapter for checkpoints whose DiT state was trained in an
    # episode-start anchor frame.  The raw state remains cached in _last_state so
    # AbsoluteActionsProcessorStep still decodes local actions in the robot base frame.
    online_anchor_state_to_episode_start: bool = False
    # Inference-only physical gripper calibration. Each value is the UMI canonical
    # gripper value that corresponds to robot state/command 1.0 (fully open).
    # Robot->model multiplies by this value; model->robot applies its inverse.
    online_gripper_umi_open_values: dict[str, float] = field(default_factory=dict)
    # Inference-only open-gripper latch. While latched, the postprocessor emits the
    # calibrated fully-open endpoint. An arm releases permanently for the episode
    # after its predicted UMI gripper mean stays below the configured threshold.
    online_gripper_open_latch: bool = False
    online_gripper_close_thresholds: dict[str, float] = field(default_factory=dict)
    online_gripper_latch_prefix_steps: int = 4
    online_gripper_latch_consecutive_chunks: int = 2
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _episode_anchor_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _online_gripper_latch_released: dict[str, bool] = field(default_factory=dict, init=False, repr=False)
    _online_gripper_latch_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

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

    def _calibrate_online_grippers(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        """Map robot gripper values to UMI coordinates, or invert that mapping."""
        if not self.online_gripper_umi_open_values or self.action_names is None:
            return values

        calibrated = values.clone()
        for idx, name in enumerate(self.action_names[: values.shape[-1]]):
            if name not in self.online_gripper_umi_open_values:
                continue
            umi_open = float(self.online_gripper_umi_open_values[name])
            if not 0.0 < umi_open <= 1.0:
                raise ValueError(
                    "online_gripper_umi_open_values must be in (0, 1], "
                    f"got {name}={umi_open}."
                )
            if inverse:
                calibrated[..., idx] = (calibrated[..., idx] / umi_open).clamp(0.0, 1.0)
            else:
                calibrated[..., idx] = calibrated[..., idx].clamp(0.0, 1.0) * umi_open
        return calibrated

    def _apply_online_gripper_open_latch(self, actions: torch.Tensor) -> torch.Tensor:
        """Hold each gripper open until its UMI prediction shows a stable close intent."""
        if (
            not self.online_gripper_open_latch
            or self.action_names is None
            or not self.online_gripper_umi_open_values
        ):
            return actions
        if actions.ndim < 2:
            raise ValueError(
                "online_gripper_open_latch expects a predicted action chunk with at least "
                f"2 dimensions, got shape={tuple(actions.shape)}."
            )
        if self.online_gripper_latch_prefix_steps < 1:
            raise ValueError("online_gripper_latch_prefix_steps must be at least 1.")
        if self.online_gripper_latch_consecutive_chunks < 1:
            raise ValueError("online_gripper_latch_consecutive_chunks must be at least 1.")

        latched_actions = actions.clone()
        prefix_steps = min(self.online_gripper_latch_prefix_steps, actions.shape[-2])
        for idx, name in enumerate(self.action_names[: actions.shape[-1]]):
            if name not in self.online_gripper_close_thresholds:
                continue
            if name not in self.online_gripper_umi_open_values:
                raise ValueError(
                    f"Missing online_gripper_umi_open_values entry for latched gripper '{name}'."
                )
            if self._online_gripper_latch_released.get(name, False):
                continue

            threshold = float(self.online_gripper_close_thresholds[name])
            predicted_mean = float(actions[..., :prefix_steps, idx].detach().float().mean().cpu())
            count = self._online_gripper_latch_counts.get(name, 0)
            count = count + 1 if predicted_mean < threshold else 0
            self._online_gripper_latch_counts[name] = count

            if count >= self.online_gripper_latch_consecutive_chunks:
                self._online_gripper_latch_released[name] = True
                LOGGER.info(
                    "Released online open-gripper latch for %s: UMI prefix mean %.6f < %.6f "
                    "for %d consecutive chunks.",
                    name,
                    predicted_mean,
                    threshold,
                    count,
                )
                continue

            # This is still in UMI coordinates. The subsequent inverse calibration
            # maps umi_open exactly to robot command 1.0.
            latched_actions[..., idx] = float(self.online_gripper_umi_open_values[name])

        return latched_actions

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        action = transition.get(TransitionKey.ACTION)

        if self.relative_state_history_to_current and state is not None:
            if state.ndim != 3 or state.shape[-2] < 2:
                raise ValueError(
                    "relative_state_history_to_current expects state [B,H,D] with H>=2, "
                    f"got {tuple(state.shape)}"
                )
            current_state = state[:, -1, :]
        else:
            current_state = state

        # Always cache the current raw state for the paired AbsoluteActionsProcessorStep.
        # Do not replace this with the episode-relative DiT state: action decoding
        # must use the current base-frame TCP pose.
        if current_state is not None:
            self._last_state = current_state.detach().clone()

        new_transition = transition.copy()

        # The checkpoint was trained on UMI gripper coordinates, while the online
        # robot reports 1.0 at its own fully-open endpoint. Apply the calibration
        # only to observation-only inference transitions; training sidecars are
        # already in UMI coordinates.
        model_state = state
        if action is None and state is not None:
            model_state = self._calibrate_online_grippers(state, inverse=False)

        # Training sidecars already contain episode-anchor-relative state. Apply
        # this conversion only to observation-only online inference transitions.
        if self.relative_state_history_to_current and model_state is not None:
            model_state = to_keyframe_relative_actions(
                model_state,
                model_state[:, -1, :],
                self.relative_action_keyframes,
                self.relative_exclude_keys,
            )
            if self.relative_scalar_state_history_indices:
                indices = list(dict.fromkeys(self.relative_scalar_state_history_indices))
                if any(index < 0 or index >= model_state.shape[-1] for index in indices):
                    raise ValueError(
                        "relative_scalar_state_history_indices are outside the state dimension: "
                        f"indices={indices}, state_dim={model_state.shape[-1]}"
                    )
                model_state = model_state.clone()
                current_values = model_state[:, -1:, indices].clone()
                model_state[:, :-1, indices] -= current_values
                model_state[:, -1:, indices] = current_values
            new_observation = observation.copy()
            new_observation[OBS_STATE] = model_state
            new_transition[TransitionKey.OBSERVATION] = new_observation
        elif self.online_anchor_state_to_episode_start and action is None and model_state is not None:
            if self._episode_anchor_state is None:
                self._episode_anchor_state = model_state.detach().clone()
            elif self._episode_anchor_state.shape != model_state.shape:
                raise RuntimeError(
                    "Online episode-anchor state shape changed without a processor reset: "
                    f"anchor={tuple(self._episode_anchor_state.shape)} current={tuple(model_state.shape)}"
                )

            anchored_state = to_keyframe_relative_actions(
                model_state,
                self._episode_anchor_state,
                self.relative_action_keyframes,
                self.relative_exclude_keys,
            )
            new_observation = observation.copy()
            new_observation[OBS_STATE] = anchored_state
            new_transition[TransitionKey.OBSERVATION] = new_observation
        elif model_state is not None and model_state is not state:
            new_observation = observation.copy()
            new_observation[OBS_STATE] = model_state
            new_transition[TransitionKey.OBSERVATION] = new_observation

        if not self.enabled:
            return new_transition

        if action is None or current_state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        specs = _keyframe_specs(self.relative_action_keyframes, self.relative_exclude_keys)
        if specs:
            compatible_specs = _compatible_keyframe_specs(
                specs,
                action_dim=action.shape[-1],
                state_dim=current_state.shape[-1],
            )
            action = to_keyframe_relative_actions(
                action,
                current_state,
                self.relative_action_keyframes,
                self.relative_exclude_keys,
            )
            mask = _mask_keyframe_dims(mask, compatible_specs)
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, current_state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "relative_action_keyframes": self.relative_action_keyframes,
            "relative_exclude_keys": self.relative_exclude_keys,
            "relative_state_history_to_current": self.relative_state_history_to_current,
            "relative_scalar_state_history_indices": self.relative_scalar_state_history_indices,
            "online_anchor_state_to_episode_start": self.online_anchor_state_to_episode_start,
            "online_gripper_umi_open_values": self.online_gripper_umi_open_values,
            "online_gripper_open_latch": self.online_gripper_open_latch,
            "online_gripper_close_thresholds": self.online_gripper_close_thresholds,
            "online_gripper_latch_prefix_steps": self.online_gripper_latch_prefix_steps,
            "online_gripper_latch_consecutive_chunks": self.online_gripper_latch_consecutive_chunks,
        }

    def reset(self) -> None:
        self._last_state = None
        self._episode_anchor_state = None
        self._online_gripper_latch_released.clear()
        self._online_gripper_latch_counts.clear()

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
        action = self.relative_step._apply_online_gripper_open_latch(action)
        action = self.relative_step._calibrate_online_grippers(action, inverse=True)
        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
