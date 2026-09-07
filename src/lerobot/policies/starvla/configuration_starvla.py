#!/usr/bin/env python

"""LeRobot config for the vendored official StarVLA framework backend."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_ACTION_MASK, OBS_STATE, OBS_STATE_MASK


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
    freeze_vlm: bool = False

    n_obs_steps: int = 1
    state_history_horizon: int = 1
    relative_state_history_to_current: bool = False
    relative_scalar_state_history_indices: list[int] = field(default_factory=list)
    inter_arm_relative_state: bool = False
    inter_arm_state_start: int = 22
    state_full_dropout_prob: float = 0.0
    inter_arm_state_dropout_prob: float = 0.0
    state_valid_indices: list[int] = field(default_factory=lambda: [20, 21])
    state_available_index: int | None = None
    chunk_size: int = 16
    n_action_steps: int = 16
    drop_n_last_frames: int = 0
    action_gap_frames: int = 6
    action_gap_dataset_overrides: list[str] = field(default_factory=list)
    apply_action_gap_to_real_datasets: bool = False
    action_execution_skip_start_steps: int = 0

    action_dim: int | None = None
    state_dim: int | None = None
    use_state_input: bool = True
    use_relative_actions: bool = False
    relative_action_keyframes: list[str] = field(default_factory=list)
    relative_exclude_keys: list[str] = field(default_factory=list)
    relative_exclude_joints: list[str] = field(default_factory=list)
    action_feature_names: list[str] | None = None
    state_feature_names: list[str] | None = None
    action_mask_observation_key: str = OBS_ACTION_MASK
    state_mask_observation_key: str = OBS_STATE_MASK
    image_keys: list[str] | None = None
    task_key: str = "task"
    task_override: str | None = None
    gradient_checkpointing: bool = False

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 2.5e-5
    vlm_optimizer_lr: float | None = None
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    @property
    def observation_delta_indices(self) -> None:
        return None

    def get_observation_delta_indices(self, feature_key: str) -> list | None:
        if feature_key == OBS_STATE and self.state_history_horizon > 1:
            return list(range(1 - self.state_history_horizon, 1))
        return None

    @property
    def action_delta_indices(self) -> list:
        start = int(self.action_gap_frames)
        return list(range(start, start + self.chunk_size))

    @staticmethod
    def parse_action_gap_dataset_override(spec: str) -> tuple[int, int]:
        dataset_index, gap_frames = str(spec).split(":", maxsplit=1)
        return int(dataset_index), int(gap_frames)

    @property
    def action_gap_frames_by_dataset_index(self) -> dict[int, int]:
        return {
            index: gap
            for index, gap in map(
                self.parse_action_gap_dataset_override, self.action_gap_dataset_overrides
            )
        }

    def get_action_delta_indices(self, dataset_index: int | None = None) -> list:
        start = int(self.action_gap_frames)
        if dataset_index is not None:
            start = self.action_gap_frames_by_dataset_index.get(int(dataset_index), start)
        return list(range(start, start + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.action_gap_frames < 0:
            raise ValueError("action_gap_frames must be non-negative.")
        if self.chunk_size < 1 or not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("Require chunk_size >= n_action_steps >= 1.")
        if not 0 <= self.action_execution_skip_start_steps < self.n_action_steps:
            raise ValueError("action_execution_skip_start_steps must be in [0, n_action_steps).")
        if self.state_history_horizon < 1:
            raise ValueError("state_history_horizon must be at least 1.")
        if self.relative_state_history_to_current and self.state_history_horizon < 2:
            raise ValueError("Current-relative state history requires at least two state frames.")
        if self.state_full_dropout_prob < 0 or self.inter_arm_state_dropout_prob < 0:
            raise ValueError("State dropout probabilities must be non-negative.")
        if self.state_full_dropout_prob + self.inter_arm_state_dropout_prob > 1:
            raise ValueError("State dropout probabilities must sum to at most 1.")
        if self.use_state_input and (self.state_dim is None or self.state_dim < 1):
            raise ValueError("State-conditioned StarVLA requires policy.state_dim >= 1.")

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

    @property
    def requires_state_feature(self) -> bool:
        return bool(self.use_state_input or self.use_relative_actions)

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
        action_model.setdefault("state_dim", int(self.state_dim or 0) if self.use_state_input else 0)
        action_model.setdefault("action_horizon", int(self.chunk_size))
        framework["action_model"] = action_model

        cfg["framework"] = framework

        datasets = dict(cfg.get("datasets", {}))
        vla_data = dict(datasets.get("vla_data", {}))
        vla_data.setdefault("obs_image_size", [224, 224])
        datasets["vla_data"] = vla_data
        cfg["datasets"] = datasets

        trainer = dict(cfg.get("trainer", {}))
        trainer.setdefault("repeated_diffusion_steps", 2)
        cfg["trainer"] = trainer
        return cfg
