#!/usr/bin/env python

import copy
from unittest.mock import patch

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.pi05_full import PI05FullConfig, PI05FullPolicy, make_pi05_full_pre_post_processors
from lerobot.policies.pi05_full.modeling_pi05 import (
    PI05FullPolicy as PI05FullPolicyImpl,
    make_prefix_att_2d_masks,
)
from lerobot.policies.pi05_full.processor_pi05 import (
    PI05_FULL_SUBTASK_IS_FALLBACK_KEY,
    Pi05PrepareStateTokenizerProcessorStep,
    pi05_full_batch_to_transition,
)
from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK,
    OBS_LANGUAGE_USER_PROMPT_TOKENS,
)


class FakeTokenizer:
    def __init__(self, eos_token_id: int = 99):
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        text: str | list[str],
        max_length: int,
        truncation: bool,
        padding: str,
        padding_side: str,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del truncation, padding, padding_side, return_tensors
        if isinstance(text, str):
            text = [text]

        encoded = []
        masks = []
        for item in text:
            ids = [ord(ch) % 97 + 1 for ch in item][:max_length]
            mask = [1] * len(ids)
            if len(ids) < max_length:
                pad = max_length - len(ids)
                ids = ids + [0] * pad
                mask = mask + [0] * pad
            encoded.append(ids)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def _make_test_config(use_state_input: bool = False) -> PI05FullConfig:
    config = PI05FullConfig(max_action_dim=7, max_state_dim=14, dtype="float32", use_state_input=use_state_input)
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }
    if config.requires_state_feature:
        config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        )
    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }
    return config


def _make_dataset_stats(include_state: bool = False) -> dict[str, dict[str, torch.Tensor]]:
    stats = {
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }
    if include_state:
        stats["observation.state"] = {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        }
    return stats


def test_make_policy_config_pi05_full():
    config = make_policy_config("pi05_full")

    assert isinstance(config, PI05FullConfig)
    assert config.type == "pi05_full"


def test_get_policy_class_pi05_full():
    policy_cls = get_policy_class("pi05_full")

    assert policy_cls is PI05FullPolicy


def test_prepare_state_tokenizer_uses_user_prompt_and_subtask_fallback():
    step = Pi05PrepareStateTokenizerProcessorStep(use_state_input=False)
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: None,
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick up red block"],
            "user_prompt": ["place blue peg carefully"],
            "subtask_text_sidecar": ["align peg with hole"],
        },
    }

    processed = step(copy.deepcopy(transition))
    complementary_data = processed[TransitionKey.COMPLEMENTARY_DATA]

    assert complementary_data["task"] == ["Task: pick up red block\nAction: "]
    assert complementary_data["user_prompt"] == ["Task: place blue peg carefully\nAction: "]
    assert complementary_data["next_subtask"] == ["Subtask: align peg with hole;\n"]
    assert complementary_data["subtask"] == ["Subtask: align peg with hole;\n"]
    assert complementary_data[PI05_FULL_SUBTASK_IS_FALLBACK_KEY].tolist() == [False]


def test_prepare_state_tokenizer_falls_back_subtask_to_user_prompt():
    step = Pi05PrepareStateTokenizerProcessorStep(use_state_input=False)
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: None,
        TransitionKey.COMPLEMENTARY_DATA: {
            "user_prompt": ["insert peg into fixture"],
        },
    }

    processed = step(copy.deepcopy(transition))
    complementary_data = processed[TransitionKey.COMPLEMENTARY_DATA]

    assert complementary_data["user_prompt"] == ["Task: insert peg into fixture\nAction: "]
    assert complementary_data["task"] == ["Task: insert peg into fixture\nAction: "]
    assert complementary_data["subtask"] == ["Subtask: insert peg into fixture;\n"]
    assert complementary_data[PI05_FULL_SUBTASK_IS_FALLBACK_KEY].tolist() == [True]


def test_pi05_full_batch_to_transition_preserves_extra_language_fields():
    batch = {
        "observation.images.base_0_rgb": torch.rand(1, 3, 4, 4),
        "task": ["task"],
        "user_prompt": ["prompt"],
        "subtask": ["subtask"],
        "next_subtask": ["next"],
        "next_subtask_text_sidecar": ["next sidecar"],
        "subtask_text_sidecar": ["subtask sidecar"],
        PI05_FULL_SUBTASK_IS_FALLBACK_KEY: torch.tensor([True]),
    }

    transition = pi05_full_batch_to_transition(batch)
    complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]

    assert complementary_data["user_prompt"] == ["prompt"]
    assert complementary_data["next_subtask"] == ["next"]
    assert complementary_data["next_subtask_text_sidecar"] == ["next sidecar"]
    assert complementary_data["subtask_text_sidecar"] == ["subtask sidecar"]
    assert complementary_data[PI05_FULL_SUBTASK_IS_FALLBACK_KEY].tolist() == [True]


def test_tokenizer_processor_adds_user_prompt_and_subtask_tokens():
    step = TokenizerProcessorStep(tokenizer=FakeTokenizer(), max_length=12)
    step._current_transition = {
        TransitionKey.OBSERVATION: {"observation.images.base_0_rgb": torch.rand(1, 3, 4, 4)},
        TransitionKey.ACTION: None,
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["Task: task\nAction: "],
            "user_prompt": ["Task: prompt\nAction: "],
            "subtask": ["Subtask: done;\n"],
        },
    }

    observation = step.observation(step.transition[TransitionKey.OBSERVATION])

    assert OBS_LANGUAGE_TOKENS in observation
    assert OBS_LANGUAGE_ATTENTION_MASK in observation
    assert OBS_LANGUAGE_USER_PROMPT_TOKENS in observation
    assert OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK in observation
    assert OBS_LANGUAGE_SUBTASK_TOKENS in observation
    assert OBS_LANGUAGE_SUBTASK_ATTENTION_MASK in observation
    assert observation[OBS_LANGUAGE_SUBTASK_TOKENS][0, 11].item() == 99
    assert observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK].dtype == torch.bool


def test_tokenizer_processor_registers_new_features():
    step = TokenizerProcessorStep(tokenizer=FakeTokenizer(), max_length=16)
    features = {
        PipelineFeatureType.OBSERVATION: {},
        PipelineFeatureType.ACTION: {},
    }

    transformed = step.transform_features(copy.deepcopy(features))
    observation_features = transformed[PipelineFeatureType.OBSERVATION]

    assert OBS_LANGUAGE_TOKENS in observation_features
    assert OBS_LANGUAGE_ATTENTION_MASK in observation_features
    assert OBS_LANGUAGE_USER_PROMPT_TOKENS in observation_features
    assert OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK in observation_features
    assert OBS_LANGUAGE_SUBTASK_TOKENS in observation_features
    assert OBS_LANGUAGE_SUBTASK_ATTENTION_MASK in observation_features


def test_policy_uses_user_prompt_tokens_when_available():
    task_tokens = torch.tensor([[1, 2, 3]])
    task_masks = torch.tensor([[True, True, False]])
    user_prompt_tokens = torch.tensor([[9, 8, 7]])
    user_prompt_masks = torch.tensor([[True, True, True]])

    tokens, masks = PI05FullPolicyImpl.get_language_inputs(
        {
            OBS_LANGUAGE_TOKENS: task_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: task_masks,
            OBS_LANGUAGE_USER_PROMPT_TOKENS: user_prompt_tokens,
            OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK: user_prompt_masks,
        }
    )

    assert torch.equal(tokens, user_prompt_tokens)
    assert torch.equal(masks, user_prompt_masks)


def test_policy_falls_back_to_task_tokens_without_user_prompt():
    task_tokens = torch.tensor([[1, 2, 3]])
    task_masks = torch.tensor([[True, True, False]])

    tokens, masks = PI05FullPolicyImpl.get_language_inputs(
        {
            OBS_LANGUAGE_TOKENS: task_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: task_masks,
        }
    )

    assert torch.equal(tokens, task_tokens)
    assert torch.equal(masks, task_masks)


def test_make_prefix_att_2d_masks_makes_subtask_causal():
    image_mask = torch.tensor([[True, True]])
    language_mask = torch.tensor([[True, True]])
    subtask_mask = torch.tensor([[True, True, False]])

    att = make_prefix_att_2d_masks([image_mask], language_mask, subtask_mask)

    assert att.shape == (1, 7, 7)
    assert bool(att[0, 4, 0]) is True
    assert bool(att[0, 4, 4]) is True
    assert bool(att[0, 4, 5]) is False
    assert bool(att[0, 5, 4]) is True


def test_policy_forward_combines_subtask_loss():
    class DummyModel:
        def forward(self, *args, **kwargs):
            del args, kwargs
            return {
                "flow_losses": torch.ones(1, 2, 3),
                "subtask_ce_loss_per_sample": torch.tensor([2.0]),
            }

    policy = object.__new__(PI05FullPolicyImpl)
    policy.config = type(
        "Cfg",
        (),
        {
            "output_features": {"action": type("F", (), {"shape": (3,)})()},
            "loss_weight_subtask_ce": 0.5,
            "action_mask_observation_key": "observation.action_mask",
            "max_action_dim": 3,
        },
    )()
    policy.model = DummyModel()
    policy._preprocess_images = lambda batch: ([batch["observation.images.base_0_rgb"]], [torch.ones(1, dtype=torch.bool)])
    policy.prepare_action = lambda batch: batch["action"]
    policy.prepare_action_mask = lambda batch: None
    policy.get_language_inputs = PI05FullPolicyImpl.get_language_inputs
    policy.get_subtask_inputs = PI05FullPolicyImpl.get_subtask_inputs

    batch = {
        "observation.images.base_0_rgb": torch.rand(1, 3, 8, 8),
        "action": torch.zeros(1, 2, 3),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
        OBS_LANGUAGE_SUBTASK_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
    }

    loss, loss_dict = PI05FullPolicyImpl.forward(policy, batch)

    assert torch.isclose(loss, torch.tensor(2.0))
    assert loss_dict["flow_loss"] == 1.0
    assert loss_dict["subtask_ce_loss"] == 2.0
    assert loss_dict["loss"] == 2.0


def test_policy_forward_without_subtask_keeps_zero_subtask_loss():
    class DummyModel:
        def forward(self, *args, **kwargs):
            del args, kwargs
            return {
                "flow_losses": torch.full((1, 1, 2), 3.0),
                "subtask_ce_loss_per_sample": torch.tensor([0.0]),
            }

    policy = object.__new__(PI05FullPolicyImpl)
    policy.config = type(
        "Cfg",
        (),
        {
            "output_features": {"action": type("F", (), {"shape": (2,)})()},
            "loss_weight_subtask_ce": 1.0,
            "action_mask_observation_key": "observation.action_mask",
            "max_action_dim": 2,
        },
    )()
    policy.model = DummyModel()
    policy._preprocess_images = lambda batch: ([batch["observation.images.base_0_rgb"]], [torch.ones(1, dtype=torch.bool)])
    policy.prepare_action = lambda batch: batch["action"]
    policy.prepare_action_mask = lambda batch: None
    policy.get_language_inputs = PI05FullPolicyImpl.get_language_inputs
    policy.get_subtask_inputs = PI05FullPolicyImpl.get_subtask_inputs

    batch = {
        "observation.images.base_0_rgb": torch.rand(1, 3, 8, 8),
        "action": torch.zeros(1, 1, 2),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
    }

    loss, loss_dict = PI05FullPolicyImpl.forward(policy, batch)

    assert torch.isclose(loss, torch.tensor(3.0))
    assert loss_dict["flow_loss"] == 3.0
    assert loss_dict["subtask_ce_loss"] == 0.0


def test_make_pi05_full_pre_post_processors_without_state_input():
    config = _make_test_config(use_state_input=False)
    with patch("lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
        preprocessor, _ = make_pi05_full_pre_post_processors(config=config, dataset_stats=_make_dataset_stats())

        batch = {
            "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224),
            "task": ["pick old task"],
            "user_prompt": ["pick new prompt"],
            "subtask": ["align object"],
        }

        processed = preprocessor(copy.deepcopy(batch))

    assert OBS_LANGUAGE_TOKENS in processed
    assert OBS_LANGUAGE_USER_PROMPT_TOKENS in processed
    assert OBS_LANGUAGE_SUBTASK_TOKENS in processed
    assert PI05_FULL_SUBTASK_IS_FALLBACK_KEY in processed


def _make_stub_policy_for_inference(interval: float = 1.0):
    policy = object.__new__(PI05FullPolicyImpl)
    torch.nn.Module.__init__(policy)
    policy.config = type(
        "Cfg",
        (),
        {
            "n_action_steps": 2,
            "chunk_size": 2,
            "max_action_dim": 3,
            "max_decoding_steps": 6,
            "temperature": 0.0,
            "subtask_regeneration_interval": interval,
            "tokenizer_name": "fake-tokenizer",
            "output_features": {"action": type("F", (), {"shape": (2,)})()},
            "action_mask_observation_key": "observation.action_mask",
            "rtc_config": None,
        },
    )()
    policy.model = type("DummyModel", (), {})()
    policy.model._paligemma_tokenizer = object()
    policy._preprocess_images = lambda batch: (
        [batch["observation.images.base_0_rgb"]],
        [torch.ones(batch["observation.images.base_0_rgb"].shape[0], dtype=torch.bool)],
    )
    policy.prepare_action_mask = lambda batch: None
    policy.get_language_inputs = PI05FullPolicyImpl.get_language_inputs
    policy.get_subtask_inputs = PI05FullPolicyImpl.get_subtask_inputs
    policy._ensure_paligemma_tokenizer = lambda: policy.model._paligemma_tokenizer
    PI05FullPolicyImpl.reset(policy)
    return policy


def test_policy_reset_clears_subtask_cache():
    policy = _make_stub_policy_for_inference()
    policy._cached_subtask_tokens = torch.ones(1, 2, dtype=torch.long)
    policy._cached_subtask_masks = torch.ones(1, 2, dtype=torch.bool)
    policy._cached_subtask_prompt_tokens = torch.ones(1, 2, dtype=torch.long)
    policy._cached_subtask_prompt_masks = torch.ones(1, 2, dtype=torch.bool)
    policy._last_subtask_time = 123.0

    PI05FullPolicyImpl.reset(policy)

    assert policy._cached_subtask_tokens is None
    assert policy._cached_subtask_masks is None
    assert policy._cached_subtask_prompt_tokens is None
    assert policy._cached_subtask_prompt_masks is None
    assert policy._last_subtask_time is None


def test_predict_action_chunk_uses_explicit_subtask_without_generation():
    policy = _make_stub_policy_for_inference()
    sample_calls = {}

    def fail_generate(*args, **kwargs):
        raise AssertionError("generate_subtask_tokens should not be called when explicit subtask is provided")

    def sample_actions(images, img_masks, tokens, masks, subtask_tokens=None, subtask_masks=None, **kwargs):
        del images, img_masks, tokens, masks, kwargs
        sample_calls["subtask_tokens"] = subtask_tokens.clone()
        sample_calls["subtask_masks"] = subtask_masks.clone()
        return torch.zeros(1, 2, 3)

    policy.model.generate_subtask_tokens = fail_generate
    policy.model.sample_actions = sample_actions

    explicit_tokens = torch.tensor([[7, 8, 0]], dtype=torch.long)
    explicit_masks = torch.tensor([[True, True, False]])
    batch = {
        "observation.images.base_0_rgb": torch.rand(1, 3, 8, 8),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 3]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, True]]),
        OBS_LANGUAGE_SUBTASK_TOKENS: explicit_tokens,
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK: explicit_masks,
        PI05_FULL_SUBTASK_IS_FALLBACK_KEY: torch.tensor([False]),
    }

    actions = PI05FullPolicyImpl.predict_action_chunk(policy, batch)

    assert actions.shape == (1, 2, 2)
    assert torch.equal(sample_calls["subtask_tokens"], explicit_tokens)
    assert torch.equal(sample_calls["subtask_masks"], explicit_masks)


def test_predict_action_chunk_generates_subtask_for_fallback_samples():
    policy = _make_stub_policy_for_inference(interval=100.0)
    generate_calls = {"count": 0}
    sample_calls = {}

    generated_tokens = torch.tensor([[101, 0, 0], [201, 202, 0]], dtype=torch.long)
    generated_masks = torch.tensor([[True, False, False], [True, True, False]])

    def generate_subtask_tokens(images, img_masks, tokens, masks, **kwargs):
        del images, img_masks, tokens, masks, kwargs
        generate_calls["count"] += 1
        return generated_tokens.clone(), generated_masks.clone()

    def sample_actions(images, img_masks, tokens, masks, subtask_tokens=None, subtask_masks=None, **kwargs):
        del images, img_masks, tokens, masks, kwargs
        sample_calls["subtask_tokens"] = subtask_tokens.clone()
        sample_calls["subtask_masks"] = subtask_masks.clone()
        return torch.zeros(2, 2, 3)

    policy.model.generate_subtask_tokens = generate_subtask_tokens
    policy.model.sample_actions = sample_actions

    explicit_tokens = torch.tensor([[11, 12], [21, 0]], dtype=torch.long)
    explicit_masks = torch.tensor([[True, True], [True, False]])
    batch = {
        "observation.images.base_0_rgb": torch.rand(2, 3, 8, 8),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, True], [True, True, True]]),
        OBS_LANGUAGE_SUBTASK_TOKENS: explicit_tokens,
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK: explicit_masks,
        PI05_FULL_SUBTASK_IS_FALLBACK_KEY: torch.tensor([False, True]),
    }

    actions = PI05FullPolicyImpl.predict_action_chunk(policy, batch)

    assert actions.shape == (2, 2, 2)
    assert generate_calls["count"] == 1
    assert torch.equal(sample_calls["subtask_tokens"][0], torch.tensor([11, 12, 0]))
    assert torch.equal(sample_calls["subtask_masks"][0], torch.tensor([True, True, False]))
    assert torch.equal(sample_calls["subtask_tokens"][1], generated_tokens[1])
    assert torch.equal(sample_calls["subtask_masks"][1], generated_masks[1])


def test_predict_action_chunk_reuses_cached_generated_subtasks_within_interval():
    policy = _make_stub_policy_for_inference(interval=100.0)
    generate_calls = {"count": 0}

    def generate_subtask_tokens(images, img_masks, tokens, masks, **kwargs):
        del images, img_masks, tokens, masks, kwargs
        generate_calls["count"] += 1
        return torch.tensor([[31, 0, 0]], dtype=torch.long), torch.tensor([[True, False, False]])

    def sample_actions(images, img_masks, tokens, masks, subtask_tokens=None, subtask_masks=None, **kwargs):
        del images, img_masks, tokens, masks, subtask_tokens, subtask_masks, kwargs
        return torch.zeros(1, 2, 3)

    policy.model.generate_subtask_tokens = generate_subtask_tokens
    policy.model.sample_actions = sample_actions

    batch = {
        "observation.images.base_0_rgb": torch.rand(1, 3, 8, 8),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 3]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, True]]),
        PI05_FULL_SUBTASK_IS_FALLBACK_KEY: torch.tensor([True]),
    }

    PI05FullPolicyImpl.predict_action_chunk(policy, batch)
    PI05FullPolicyImpl.predict_action_chunk(policy, batch)

    assert generate_calls["count"] == 1
