#!/usr/bin/env python

"""Pre/post processors for the generic StarVLA wrapper policy."""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.processor_pi05 import Pi05ApplyMasksProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_starvla import StarVLAConfig


@ProcessorStepRegistry.register(name="starvla_prepare_tactile_history")
@dataclass
class StarVLAPrepareTactileHistoryProcessorStep(ProcessorStep):
    """Build raw physical normal/tangent histories from one or two hands."""

    enabled: bool = False
    history_len: int = 30
    force_indices: tuple[int, ...] = ()
    output_key: str = "observation.tactile_history"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}) or {})
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("Tactile history conditioning requires observation.state.")
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(f"Expected state [B,T,D] or [B,D], got {tuple(state.shape)}")
        if len(self.force_indices) not in {6, 12} or max(self.force_indices) >= state.shape[-1]:
            raise ValueError(
                f"Invalid tactile force indices {self.force_indices} for state shape {tuple(state.shape)}"
            )

        index = torch.as_tensor(self.force_indices, device=state.device, dtype=torch.long)
        sensor_count = len(self.force_indices) // 3
        # [B,T,S,3], using two fingertip sensors per selected hand.
        force = state.index_select(-1, index).reshape(*state.shape[:-1], sensor_count, 3)
        normal = force[..., 2].abs()
        tangent = torch.linalg.vector_norm(force[..., :2], dim=-1)
        if sensor_count == 2:
            tactile = torch.stack(
                (normal[..., 0] + normal[..., 1], tangent[..., 0] + tangent[..., 1]),
                dim=-1,
            )
        else:
            tactile = torch.stack(
                (
                    normal[..., 0] + normal[..., 1],
                    tangent[..., 0] + tangent[..., 1],
                    normal[..., 2] + normal[..., 3],
                    tangent[..., 2] + tangent[..., 3],
                ),
                dim=-1,
            )

        # Dataset history retrieval clamps pre-episode indices to frame 0;
        # replace those repeated samples with the required left zero padding.
        pad_mask = observation.get(f"{OBS_STATE}_is_pad")
        if pad_mask is not None:
            if pad_mask.ndim == 1:
                pad_mask = pad_mask.unsqueeze(0)
            tactile = tactile.masked_fill(pad_mask.to(device=tactile.device).unsqueeze(-1), 0.0)

        observation[self.output_key] = tactile
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def get_config(self) -> dict[str, Any]:
        # Persist the physical force history definition. ProcessorStep's default
        # get_config() is empty, which otherwise reloads this step disabled.
        return {
            "enabled": bool(self.enabled),
            "history_len": int(self.history_len),
            "force_indices": list(self.force_indices),
            "output_key": self.output_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.enabled:
            return features
        transformed = {ptype: feats.copy() for ptype, feats in features.items()}
        transformed.setdefault(PipelineFeatureType.OBSERVATION, {})
        transformed[PipelineFeatureType.OBSERVATION][self.output_key] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.history_len, 2 if len(self.force_indices) == 6 else 4),
        )
        return transformed


def make_starvla_pre_post_processors(
    config: StarVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    processor_features = {**config.normalizer_input_features(), **config.output_features}

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
        relative_action_keyframes=config.relative_action_keyframes,
        relative_exclude_keys=config.relative_exclude_keys,
        relative_state_history_to_current=config.relative_state_history_to_current,
        relative_scalar_state_history_indices=config.relative_scalar_state_history_indices,
        inter_arm_relative_state=config.inter_arm_relative_state,
        inter_arm_state_start=config.inter_arm_state_start,
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
    ]
    if config.use_tactile:
        input_steps.append(
            StarVLAPrepareTactileHistoryProcessorStep(
                enabled=True,
                history_len=config.tactile_history_len,
                force_indices=tuple(config.tactile_force_indices),
                output_key=config.tactile_history_key,
            )
        )
    input_steps.extend([
        relative_step,
        NormalizerProcessorStep(
            features=processor_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05ApplyMasksProcessorStep(
            action_mask_observation_key=config.action_mask_observation_key,
            state_mask_observation_key=config.state_mask_observation_key,
        ),
        DeviceProcessorStep(device=config.device),
    ])

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
