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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_ACTION_MASK, OBS_IMAGES, OBS_STATE, OBS_STATE_MASK

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05_full")
@dataclass
class PI05FullConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute
    action_gap_frames: int = 0  # Future action offset applied before chunk sampling
    action_gap_dataset_overrides: list[str] = field(default_factory=list)
    apply_action_gap_to_real_datasets: bool = False

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32
    # Whether to include observation.state in PI0.5 inputs.
    # When enabled, state is discretized and appended to the language prompt.
    # When disabled, PI0.5 uses a task-only prompt.
    use_state_input: bool = True

    # Optional tactile prefix conditioning.
    # `flat` keeps a lightweight pooled/flattened branch for arbitrary tactile tensors.
    # `parsed_vit` uses a small ViT-style encoder for parsed tactile maps.
    enable_tactile: bool = False
    tactile_feature_keys: list[str] = field(default_factory=list)
    tactile_encoder_type: str = "flat"
    tactile_tokens: int = 16
    max_tactile_dim: int = 128
    tactile_dropout_prob: float = 0.1
    tactile_fallback_to_state: bool = False
    tactile_mask_key: str = "observation.tactile_present"
    tactile_slot_present_key: str = "observation.tactile_slot_present"
    enable_tactile_role_embeddings: bool = True
    # Backward-compatible alias. If set true it forces tactile_encoder_type=parsed_vit.
    enable_parsed_tactile_vit: bool = False
    tactile_input_channels: int = 3
    tactile_target_size: tuple[int, int] = (96, 128)
    tactile_patch_size: int = 16
    tactile_patch_encoder_dim: int = 384
    tactile_patch_encoder_depth: int = 12
    tactile_resampler_heads: int = 6
    tactile_token_budget_per_sensor: int = 8

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Canonical xyz/rot6d blocks converted with SE(3)-aware relative actions.
    # Each entry is "pos_start:rot6d_start"; canonical bimanual uses left=0:3, right=10:13.
    relative_action_keyframes: list[str] = field(default_factory=list)
    relative_exclude_keys: list[str] = field(default_factory=list)
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None
    state_feature_names: list[str] | None = None
    source_action_feature_names: list[str] | None = None
    source_state_feature_names: list[str] | None = None

    # Canonical mixed single-arm / dual-arm representation used before normalization.
    canonicalize_bimanual_rot6d: bool = False
    canonical_missing_value: float | None = 9930.0
    canonical_action_dim: int = 20
    canonical_state_dim: int = 22
    action_mask_observation_key: str = OBS_ACTION_MASK
    state_mask_observation_key: str = OBS_STATE_MASK

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    max_decoding_steps: int = 200
    temperature: float = 0.0
    subtask_regeneration_interval: float = 1.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections
    reset_action_expert_after_pretrained_load: bool = False
    loss_weight_flow: float = 1.0
    loss_weight_subtask_ce: float = 1.0

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.action_gap_frames < 0:
            raise ValueError("action_gap_frames must be non-negative.")
        if self.action_gap_dataset_overrides:
            seen_dataset_indices: set[int] = set()
            for spec in self.action_gap_dataset_overrides:
                dataset_index, gap_frames = self.parse_action_gap_dataset_override(spec)
                if dataset_index in seen_dataset_indices:
                    raise ValueError(
                        "action_gap_dataset_overrides cannot contain duplicate dataset indices."
                    )
                seen_dataset_indices.add(dataset_index)
                if gap_frames < 0:
                    raise ValueError("action_gap_dataset_overrides must use non-negative gap values.")

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.loss_weight_flow < 0:
            raise ValueError("loss_weight_flow must be non-negative.")
        if self.loss_weight_subtask_ce < 0:
            raise ValueError("loss_weight_subtask_ce must be non-negative.")
        if self.max_decoding_steps <= 0:
            raise ValueError("max_decoding_steps must be positive.")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if self.subtask_regeneration_interval < 0:
            raise ValueError("subtask_regeneration_interval must be non-negative.")
        if self.enable_parsed_tactile_vit:
            self.tactile_encoder_type = "parsed_vit"
        if self.tactile_encoder_type not in {"flat", "parsed_vit"}:
            raise ValueError(
                f"Unsupported tactile_encoder_type={self.tactile_encoder_type!r}. "
                "Expected one of {'flat', 'parsed_vit'}."
            )
        if self.enable_tactile:
            if self.tactile_tokens <= 0:
                raise ValueError("tactile_tokens must be positive., got {self.tactile_tokens}")
            if self.max_tactile_dim <= 0:
                raise ValueError("max_tactile_dim must be positive., got {self.max_tactile_dim}")
            if not 0.0 <= self.tactile_dropout_prob <= 1.0:
                raise ValueError("tactile_dropout_prob must be in [0, 1].")
            if self.tactile_encoder_type == "parsed_vit":
                if self.tactile_input_channels <= 0:
                    raise ValueError("tactile_input_channels must be positive.")
                if len(self.tactile_target_size) != 2 or min(self.tactile_target_size) <= 0:
                    raise ValueError("tactile_target_size must be a positive (H, W) pair.")
                if self.tactile_patch_size <= 0:
                    raise ValueError("tactile_patch_size must be positive.")
                if self.tactile_patch_encoder_dim <= 0:
                    raise ValueError("tactile_patch_encoder_dim must be positive.")
                if self.tactile_patch_encoder_depth <= 0:
                    raise ValueError("tactile_patch_encoder_depth must be positive.")
                if self.tactile_resampler_heads <= 0:
                    raise ValueError("tactile_resampler_heads must be positive.")
                if self.tactile_token_budget_per_sensor <= 0:
                    raise ValueError("tactile_token_budget_per_sensor must be positive.")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if self.requires_state_feature and OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=((self.canonical_state_dim,) if self.canonicalize_bimanual_rot6d else (self.max_state_dim,)),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=((self.canonical_action_dim,) if self.canonicalize_bimanual_rot6d else (self.max_action_dim,)),
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def requires_state_feature(self) -> bool:
        return self.use_state_input or self.use_relative_actions or self.canonicalize_bimanual_rot6d

    @property
    def action_delta_indices(self) -> list:
        start = int(self.action_gap_frames)
        return list(range(start, start + self.chunk_size))

    @staticmethod
    def parse_action_gap_dataset_override(spec: str) -> tuple[int, int]:
        parts = str(spec).split(":", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid action_gap_dataset_overrides entry '{spec}'. Expected 'dataset_index:gap_frames'."
            )
        dataset_index = int(parts[0])
        gap_frames = int(parts[1])
        if dataset_index < 0:
            raise ValueError("action_gap_dataset_overrides must use non-negative dataset indices.")
        if gap_frames < 0:
            raise ValueError("action_gap_dataset_overrides must use non-negative gap values.")
        return dataset_index, gap_frames

    @property
    def action_gap_frames_by_dataset_index(self) -> dict[int, int]:
        overrides: dict[int, int] = {}
        for spec in self.action_gap_dataset_overrides:
            dataset_index, gap_frames = self.parse_action_gap_dataset_override(spec)
            overrides[dataset_index] = gap_frames
        return overrides

    def get_action_delta_indices(self, dataset_index: int | None = None) -> list:
        start = int(self.action_gap_frames)
        if dataset_index is not None and self.action_gap_dataset_overrides:
            start = int(self.action_gap_frames_by_dataset_index.get(int(dataset_index), start))
        return list(range(start, start + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
