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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05_full.configuration_pi05 import PI05FullConfig
from lerobot.policies.pi05_full.canonical_bimanual import (
    CANONICAL_ACTION_DIM,
    CANONICAL_STATE_DIM,
    canonicalize_bimanual_state_action,
)
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_ACTION_MASK,
    OBS_STATE,
    OBS_STATE_MASK,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

PI05_FULL_SUBTASK_IS_FALLBACK_KEY = "pi05_full.subtask_is_fallback"


def _broadcast_feature_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=target.dtype, device=target.device)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-2)
    return mask


def pi05_full_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Preserve pi05_full-specific complementary fields that the generic batch converter drops."""
    transition = batch_to_transition(batch)
    complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
    for key in (
        "user_prompt",
        "next_subtask",
        "next_subtask_text_sidecar",
        "subtask_text_sidecar",
        PI05_FULL_SUBTASK_IS_FALLBACK_KEY,
    ):
        if key in batch:
            complementary_data[key] = batch[key]
    transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
    return transition


@ProcessorStepRegistry.register(name="pi05_full_canonicalize_bimanual_rot6d")
@dataclass
class Pi05CanonicalizeBimanualRot6DProcessorStep(ProcessorStep):
    source_action_feature_names: list[str] | None = None
    source_state_feature_names: list[str] | None = None
    missing_value: float | None = 9930.0
    action_mask_observation_key: str = OBS_ACTION_MASK
    state_mask_observation_key: str = OBS_STATE_MASK

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}) or {})
        state = observation.get(OBS_STATE)
        if state is None:
            return transition

        action = transition.get(TransitionKey.ACTION)
        canonical = canonicalize_bimanual_state_action(
            state=state,
            action=action,
            source_state_names=self.source_state_feature_names,
            source_action_names=self.source_action_feature_names,
            missing_value=self.missing_value,
        )

        observation[OBS_STATE] = canonical[OBS_STATE]
        observation[self.state_mask_observation_key] = canonical["state_mask"]
        observation[self.action_mask_observation_key] = canonical.get("action_mask", canonical["state_mask"])
        transition[TransitionKey.OBSERVATION] = observation

        if action is not None and ACTION in canonical:
            transition[TransitionKey.ACTION] = canonical[ACTION]

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {ptype: feats.copy() for ptype, feats in features.items()}
        transformed.setdefault(PipelineFeatureType.OBSERVATION, {})
        transformed.setdefault(PipelineFeatureType.ACTION, {})

        transformed[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(CANONICAL_STATE_DIM,),
        )
        transformed[PipelineFeatureType.OBSERVATION][self.state_mask_observation_key] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(CANONICAL_ACTION_DIM,),
        )
        transformed[PipelineFeatureType.OBSERVATION][self.action_mask_observation_key] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(CANONICAL_ACTION_DIM,),
        )
        transformed[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(CANONICAL_ACTION_DIM,),
        )
        return transformed


@ProcessorStepRegistry.register(name="pi05_full_apply_masks_processor_step")
@dataclass
class Pi05ApplyMasksProcessorStep(ProcessorStep):
    action_mask_observation_key: str = OBS_ACTION_MASK
    state_mask_observation_key: str = OBS_STATE_MASK

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}) or {})

        state = observation.get(OBS_STATE)
        state_mask = observation.get(self.state_mask_observation_key)
        if state is not None and state_mask is not None:
            state = state.clone()
            state_mask = state_mask.to(dtype=state.dtype, device=state.device)
            masked_dims = min(state_mask.shape[-1], state.shape[-1])
            state[..., :masked_dims] = state[..., :masked_dims] * state_mask[..., :masked_dims]
            observation[OBS_STATE] = state
            transition[TransitionKey.OBSERVATION] = observation

        action = transition.get(TransitionKey.ACTION)
        action_mask = observation.get(self.action_mask_observation_key)
        if action is not None and action_mask is not None:
            action = action.clone()
            action_mask = _broadcast_feature_mask(action_mask, action)
            masked_dims = min(action_mask.shape[-1], action.shape[-1])
            action[..., :masked_dims] = action[..., :masked_dims] * action_mask[..., :masked_dims]
            transition[TransitionKey.ACTION] = action

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi05_full_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    use_state_input: bool = True
    user_prompt_key: str = "user_prompt"
    command_key: str = "next_subtask"
    fallback_command_to_task: bool = True
    command_fallback_keys: tuple[str, ...] = ("subtask", "next_subtask_text_sidecar", "subtask_text_sidecar")

    @staticmethod
    def _to_text_batch(value: Any, key: str) -> list[str]:
        if value is None:
            raise ValueError(f"No {key} found in complementary data")
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
        raise ValueError(f"Complementary data '{key}' must be a string or list[str], got {type(value)!r}")

    @staticmethod
    def _clean_text(value: str) -> str:
        return value.strip().replace("_", " ").replace("\n", " ")

    @classmethod
    def _optional_text_batch(cls, value: Any, key: str) -> list[str] | None:
        if value is None:
            return None
        texts = cls._to_text_batch(value, key)
        if not any(cls._clean_text(text) for text in texts):
            return None
        return texts

    @classmethod
    def _format_action_prompt(cls, value: str, state_str: str | None) -> str:
        cleaned_text = cls._clean_text(value)
        if state_str is None:
            return f"Task: {cleaned_text}\nAction: "
        return f"Task: {cleaned_text}, State: {state_str};\nAction: "

    @classmethod
    def _format_subtask_prompt(cls, value: str) -> str:
        return f"Subtask: {cls._clean_text(value)};\n"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        tasks = self._optional_text_batch(complementary_data.get(self.task_key), self.task_key)
        user_prompts = self._optional_text_batch(complementary_data.get(self.user_prompt_key), self.user_prompt_key)
        if tasks is None and user_prompts is None:
            raise ValueError("No task or user_prompt found in complementary data")

        task_texts = tasks if tasks is not None else user_prompts
        user_prompt_texts = user_prompts if user_prompts is not None else tasks
        assert task_texts is not None
        assert user_prompt_texts is not None

        discretized_states = None
        if self.use_state_input:
            if state is None:
                raise ValueError("State is required for PI05 when use_state_input=True")

            # TODO: check if this necessary
            state = deepcopy(state)

            # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
            # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
            state_np = state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        if len(task_texts) != len(user_prompt_texts):
            raise ValueError("task and user_prompt must have the same batch size after preprocessing")

        def make_prompt_batch(texts: list[str]) -> list[str]:
            full_prompts: list[str] = []
            for i, text in enumerate(texts):
                state_str = None
                if discretized_states is not None:
                    state_str = " ".join(map(str, discretized_states[i]))
                full_prompts.append(self._format_action_prompt(text, state_str))
            return full_prompts

        complementary_data[self.task_key] = make_prompt_batch(task_texts)
        complementary_data[self.user_prompt_key] = make_prompt_batch(user_prompt_texts)

        commands = self._optional_text_batch(complementary_data.get(self.command_key), self.command_key)
        resolved_command_key = self.command_key if commands is not None else None
        if commands is None:
            for fallback_key in self.command_fallback_keys:
                commands = self._optional_text_batch(complementary_data.get(fallback_key), fallback_key)
                if commands is not None:
                    resolved_command_key = fallback_key
                    break
        used_task_fallback = False
        if commands is None and self.fallback_command_to_task:
            commands = user_prompt_texts
            used_task_fallback = True
        if commands is not None:
            if len(commands) != len(user_prompt_texts):
                raise ValueError("subtask/user_prompt batch size mismatch")
            formatted_commands = [self._format_subtask_prompt(text) for text in commands]
            complementary_data[self.command_key] = formatted_commands
            complementary_data["subtask"] = formatted_commands
            marker_device = state.device if isinstance(state, torch.Tensor) else None
            complementary_data[PI05_FULL_SUBTASK_IS_FALLBACK_KEY] = torch.full(
                (len(user_prompt_texts),),
                used_task_fallback,
                dtype=torch.bool,
                device=marker_device,
            )

        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "max_state_dim": self.max_state_dim,
            "task_key": self.task_key,
            "use_state_input": self.use_state_input,
            "user_prompt_key": self.user_prompt_key,
            "command_key": self.command_key,
            "fallback_command_to_task": self.fallback_command_to_task,
            "command_fallback_keys": list(self.command_fallback_keys),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_full_pre_post_processors(
    config: PI05FullConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
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
            # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
            # because the tokenizer step expects normalized state in [-1, 1] range for discretization
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            Pi05ApplyMasksProcessorStep(
                action_mask_observation_key=config.action_mask_observation_key,
                state_mask_observation_key=config.state_mask_observation_key,
            ),
            Pi05PrepareStateTokenizerProcessorStep(
                max_state_dim=config.max_state_dim,
                use_state_input=config.use_state_input,
            ),
            TokenizerProcessorStep(
                tokenizer_name=config.tokenizer_name,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=config.device),
        ]
    )

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=pi05_full_batch_to_transition,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
