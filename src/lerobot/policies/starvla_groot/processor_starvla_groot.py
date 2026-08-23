#!/usr/bin/env python

# Copyright 2025 starVLA community & The HuggingFace Inc. team. All rights reserved.
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
"""Pre/post-processor pipelines for the StarvlaGroot (QwenGR00T) policy.

Unlike PI05 there is no tokenizer step: the Qwen processor tokenizes images and
language inside the policy via ``apply_chat_template``. The raw ``task`` string
is carried through complementary data and read by the policy at forward time.
State and action are normalized here (dataset stats) and the action is
un-normalized in the post-processor.
"""

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
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.policies.pi05.processor_pi05 import (
    Pi05ApplyMasksProcessorStep,
    Pi05CanonicalizeBimanualRot6DProcessorStep,
)

from .configuration_starvla_groot import StarvlaGrootConfig


def make_starvla_groot_pre_post_processors(
    config: StarvlaGrootConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    processor_features = {**config.normalizer_input_features(), **config.output_features}
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
        relative_action_keyframes=getattr(config, "relative_action_keyframes", []),
        relative_exclude_keys=getattr(config, "relative_exclude_keys", []),
        online_anchor_state_to_episode_start=getattr(
            config, "online_anchor_state_to_episode_start", False
        ),
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
    ]
    if config.canonicalize_bimanual_rot6d:
        input_steps.append(
            Pi05CanonicalizeBimanualRot6DProcessorStep(
                source_action_feature_names=config.source_action_feature_names,
                source_state_feature_names=config.source_state_feature_names,
                missing_value=config.canonical_missing_value,
                action_mask_observation_key=config.action_mask_observation_key,
                state_mask_observation_key=config.state_mask_observation_key,
            )
        )
    input_steps.extend(
        [
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
    )

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
