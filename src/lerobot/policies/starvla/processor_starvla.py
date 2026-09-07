#!/usr/bin/env python

"""Pre/post processors for the generic StarVLA wrapper policy."""

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
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.policies.pi05.processor_pi05 import Pi05ApplyMasksProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_starvla import StarVLAConfig


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
    ]

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
