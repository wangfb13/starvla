#!/usr/bin/env python

"""LeRobot config for the vendored official StarVLA framework backend."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("starvla")
@dataclass
class StarVLAConfig(PreTrainedConfig):
    """Generic StarVLA policy wrapper.

    This config selects one official StarVLA framework by name and passes a
    framework config tree through to the vendored StarVLA backend. Common
    LeRobot-facing fields live here so dataset features, processors, and action
    chunking keep working with the rest of the LeRobot stack.
    """

    framework_name: str = "QwenGR00T"
    framework_config: dict[str, Any] = field(default_factory=dict)

    base_vlm: str | None = None
    attn_implementation: str = "sdpa"

    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16
    action_execution_skip_start_steps: int = 0

    action_dim: int | None = None
    state_dim: int | None = None
    use_state_input: bool = True
    image_keys: list[str] | None = None
    task_key: str = "task"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        if ACTION not in self.output_features:
            if self.action_dim is None:
                raise ValueError("StarVLA requires an 'action' output feature or policy.action_dim.")
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(int(self.action_dim),),
            )

        if self.action_dim is None:
            self.action_dim = int(self.output_features[ACTION].shape[0])

        if self.use_state_input and self.state_dim is not None and OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(int(self.state_dim),),
            )
        if OBS_STATE in self.input_features and self.state_dim is None:
            self.state_dim = int(self.input_features[OBS_STATE].shape[0])

        if self.image_keys is not None:
            missing = [key for key in self.image_keys if key not in self.input_features]
            if missing:
                raise ValueError(f"StarVLA image_keys are not present in input_features: {missing}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def normalizer_input_features(self) -> dict[str, PolicyFeature]:
        return {
            key: feature
            for key, feature in (self.input_features or {}).items()
            if feature.type is not FeatureType.VISUAL
        }

    def resolved_image_keys(self) -> list[str]:
        if self.image_keys is not None:
            return list(self.image_keys)
        return [
            key
            for key, feature in (self.input_features or {}).items()
            if feature.type is FeatureType.VISUAL
        ]

    def to_official_config(self) -> dict[str, Any]:
        cfg = dict(self.framework_config)
        framework = dict(cfg.get("framework", {}))
        framework["name"] = self.framework_name

        if self.base_vlm is not None:
            qwenvl = dict(framework.get("qwenvl", {}))
            qwenvl.setdefault("base_vlm", self.base_vlm)
            qwenvl.setdefault("attn_implementation", self.attn_implementation)
            framework["qwenvl"] = qwenvl

        action_model = dict(framework.get("action_model", {}))
        action_model.setdefault("action_dim", int(self.action_dim or 0))
        action_model.setdefault("state_dim", int(self.state_dim or 0))
        action_model.setdefault("action_horizon", int(self.chunk_size))
        framework["action_model"] = action_model

        cfg["framework"] = framework
        return cfg
