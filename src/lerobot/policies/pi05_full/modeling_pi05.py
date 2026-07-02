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

import builtins
import copy
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    AutoTokenizer = None
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05_full.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05FullConfig
from lerobot.policies.pi05_full.processor_pi05 import PI05_FULL_SUBTASK_IS_FALLBACK_KEY
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_ACTION_MASK,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK,
    OBS_LANGUAGE_USER_PROMPT_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def make_prefix_att_2d_masks(
    image_pad_masks: list[torch.Tensor],
    language_pad_masks: torch.Tensor,
    subtask_pad_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build prefix attention where image/language are bidirectional and subtask is causal on itself."""
    pad_masks = []
    segment_types = []

    for img_mask in image_pad_masks:
        num_img_embs = img_mask.shape[1]
        pad_masks.append(img_mask)
        segment_types.append(("image", num_img_embs))

    pad_masks.append(language_pad_masks)
    segment_types.append(("language", language_pad_masks.shape[1]))

    if subtask_pad_masks is not None:
        pad_masks.append(subtask_pad_masks)
        segment_types.append(("subtask", subtask_pad_masks.shape[1]))

    full_pad_mask = torch.cat(pad_masks, dim=1)
    bsize, total_len = full_pad_mask.shape
    device = full_pad_mask.device
    att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)

    context_types = {"image", "language"}
    offsets = []
    cursor = 0
    for seg_type, seg_len in segment_types:
        offsets.append((seg_type, cursor, cursor + seg_len))
        cursor += seg_len

    for query_type, query_start, query_end in offsets:
        for key_type, key_start, key_end in offsets:
            if query_type in context_types and key_type in context_types:
                att_2d_masks[:, query_start:query_end, key_start:key_end] = True
            elif query_type == "subtask" and key_type in context_types:
                att_2d_masks[:, query_start:query_end, key_start:key_end] = True
            elif query_type == "subtask" and key_type == "subtask":
                subtask_len = query_end - query_start
                causal_mask = torch.tril(torch.ones(subtask_len, subtask_len, dtype=torch.bool, device=device))
                att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None, :, :]

    pad_2d_masks = full_pad_mask[:, None, :] * full_pad_mask[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


def resize_without_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize images to a target size without padding.

    Keeps the same input/output layout handling as ``resize_with_pad_torch``.
    """
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    resized_images = F.interpolate(
        images,
        size=(height, width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    if channels_last:
        resized_images = resized_images.permute(0, 2, 3, 1)

    return resized_images


TACTILE_ROLE_ALIAS_ORDER = (
    "observation.images.tactile_left_left",
    "observation.images.tactile_left_right",
    "observation.images.tactile_right_left",
    "observation.images.tactile_right_right",
    "observation.depth_deformation.tactile_left_left",
    "observation.depth_deformation.tactile_left_right",
    "observation.depth_deformation.tactile_right_left",
    "observation.depth_deformation.tactile_right_right",
    "observation.depth_shear.tactile_left_left",
    "observation.depth_shear.tactile_left_right",
    "observation.depth_shear.tactile_right_left",
    "observation.depth_shear.tactile_right_right",
)
TACTILE_SENSOR_SLOT_ID_BY_KEY = {key: idx % 4 for idx, key in enumerate(TACTILE_ROLE_ALIAS_ORDER)}
TACTILE_HAND_SIDE_ID_BY_KEY = {key: 0 if (idx % 4) < 2 else 1 for idx, key in enumerate(TACTILE_ROLE_ALIAS_ORDER)}
TACTILE_FINGER_SLOT_ID_BY_KEY = {key: (idx % 4) % 2 for idx, key in enumerate(TACTILE_ROLE_ALIAS_ORDER)}


class TactilePatchEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + x_attn
        x = x + self.mlp(self.norm2(x))
        return x


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(
        self,
        config: PI05FullConfig,
        rtc_processor: RTCProcessor | None = None,
        paligemma_tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self._paligemma_tokenizer = paligemma_tokenizer

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.enable_tactile = config.enable_tactile
        self.tactile_encoder_type = config.tactile_encoder_type
        if self.enable_tactile:
            tactile_in_dim = 1 if self.tactile_encoder_type == "contact_stats" else config.max_tactile_dim
            self.tactile_in_proj = nn.Linear(tactile_in_dim, paligemma_config.width)
            self.tactile_fusion_layernorm = nn.LayerNorm(paligemma_config.width)
            self.tactile_gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
            self.tactile_modality_embedding = nn.Embedding(1, paligemma_config.width)
            self.tactile_hand_side_embedding = nn.Embedding(2, paligemma_config.width)
            self.tactile_finger_slot_embedding = nn.Embedding(2, paligemma_config.width)
            self.tactile_sensor_slot_embedding = nn.Embedding(4, paligemma_config.width)
            nn.init.zeros_(self.tactile_modality_embedding.weight)
            nn.init.zeros_(self.tactile_hand_side_embedding.weight)
            nn.init.zeros_(self.tactile_finger_slot_embedding.weight)
            nn.init.zeros_(self.tactile_sensor_slot_embedding.weight)

            if self.tactile_encoder_type == "parsed_vit":
                tactile_h, tactile_w = config.tactile_target_size
                patch_size = config.tactile_patch_size
                if tactile_h % patch_size != 0 or tactile_w % patch_size != 0:
                    raise ValueError(
                        f"tactile_target_size {config.tactile_target_size} must be divisible by "
                        f"tactile_patch_size {patch_size}"
                    )
                tactile_patch_count = (tactile_h // patch_size) * (tactile_w // patch_size)
                self.tactile_patch_proj = nn.Conv2d(
                    config.tactile_input_channels,
                    config.tactile_patch_encoder_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                self.tactile_patch_pos_embed = nn.Embedding(tactile_patch_count, config.tactile_patch_encoder_dim)
                self.tactile_patch_blocks = nn.ModuleList(
                    [
                        TactilePatchEncoderBlock(
                            config.tactile_patch_encoder_dim,
                            config.tactile_resampler_heads,
                        )
                        for _ in range(config.tactile_patch_encoder_depth)
                    ]
                )
                self.tactile_patch_layernorm = nn.LayerNorm(config.tactile_patch_encoder_dim)
                self.tactile_query_tokens = nn.Parameter(
                    torch.randn(1, config.tactile_token_budget_per_sensor, config.tactile_patch_encoder_dim) * 0.02
                )
                self.tactile_resampler_attn = nn.MultiheadAttention(
                    config.tactile_patch_encoder_dim,
                    config.tactile_resampler_heads,
                    batch_first=True,
                )
                self.tactile_resampler_ff = nn.Sequential(
                    nn.Linear(config.tactile_patch_encoder_dim, config.tactile_patch_encoder_dim * 4),
                    nn.GELU(),
                    nn.Linear(config.tactile_patch_encoder_dim * 4, config.tactile_patch_encoder_dim),
                )
                self.tactile_token_layernorm = nn.LayerNorm(config.tactile_patch_encoder_dim)
                self.tactile_in_proj = nn.Linear(config.tactile_patch_encoder_dim, paligemma_config.width)
                self.tactile_sensor_scale = nn.Embedding(4, 1)
                nn.init.ones_(self.tactile_sensor_scale.weight)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    @staticmethod
    def _reset_module_parameters(module: nn.Module) -> None:
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    def reset_action_expert_parameters(self) -> None:
        """Reinitialize action-specific modules after loading a warm-start checkpoint."""
        self.paligemma_with_expert.gemma_expert.apply(self._reset_module_parameters)
        for module in (
            self.action_in_proj,
            self.action_out_proj,
            self.time_mlp_in,
            self.time_mlp_out,
        ):
            self._reset_module_parameters(module)
        logging.info(
            "Reinitialized PI05Full action expert modules: gemma_expert, action_in_proj, "
            "action_out_proj, time_mlp_in, time_mlp_out"
        )

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks, dtype: torch.dtype | None = None):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        if dtype is None:
            dtype = torch.float32
        zero = torch.tensor(0.0, dtype=dtype, device=att_2d_masks.device)
        neg = torch.tensor(OPENPI_ATTENTION_MASK_VALUE, dtype=dtype, device=att_2d_masks.device)
        return torch.where(att_2d_masks_4d, zero, neg)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        subtask_tokens: torch.Tensor | None = None,
        subtask_masks: torch.Tensor | None = None,
        tactile_tokens: Tensor | None = None,
        tactile_token_masks: Tensor | None = None,
        tactile_role_metadata: dict[str, Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images, language, and optional subtask tokens for prefix processing."""
        embs = []
        image_pad_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            image_pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

        if self.enable_tactile and tactile_tokens is not None:
            bsize = tactile_tokens.shape[0]
            if tactile_token_masks is None:
                tactile_token_masks = torch.ones(
                    bsize,
                    tactile_tokens.shape[1],
                    dtype=torch.bool,
                    device=tactile_tokens.device,
                )
            else:
                tactile_token_masks = tactile_token_masks.to(device=tactile_tokens.device, dtype=torch.bool)

            if self.training and self.config.tactile_dropout_prob > 0:
                keep = torch.rand(bsize, device=tactile_tokens.device) >= self.config.tactile_dropout_prob
                tactile_token_masks = tactile_token_masks & keep[:, None]

            def tactile_embed_func(x):
                out = self.tactile_in_proj(x)
                out = self.tactile_fusion_layernorm(out)
                if self.config.enable_tactile_role_embeddings and tactile_role_metadata is not None:
                    role_emb = self.tactile_modality_embedding(tactile_role_metadata["modality_ids"])
                    role_emb = role_emb + self.tactile_hand_side_embedding(tactile_role_metadata["hand_side_ids"])
                    role_emb = role_emb + self.tactile_finger_slot_embedding(
                        tactile_role_metadata["finger_slot_ids"]
                    )
                    role_emb = role_emb + self.tactile_sensor_slot_embedding(
                        tactile_role_metadata["sensor_slot_ids"]
                    )
                    role_mask = tactile_role_metadata["token_active_mask"].to(device=out.device, dtype=out.dtype)
                    out = (out + role_emb) * role_mask[:, :, None]
                return out

            tactile_emb = self._apply_checkpoint(tactile_embed_func, tactile_tokens)
            tactile_emb = tactile_emb * torch.sigmoid(self.tactile_gate)
            embs.append(tactile_emb)
            image_pad_masks.append(tactile_token_masks)

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)

        subtask_pad_mask = None
        if self.config.enable_subtask_branch and subtask_tokens is not None and subtask_masks is not None:
            def subtask_embed_func(subtask_tokens):
                subtask_emb = self.paligemma_with_expert.embed_language_tokens(subtask_tokens)
                subtask_emb_dim = subtask_emb.shape[-1]
                return subtask_emb * math.sqrt(subtask_emb_dim)

            subtask_emb = self._apply_checkpoint(subtask_embed_func, subtask_tokens)
            embs.append(subtask_emb)
            subtask_pad_mask = subtask_masks

        embs = torch.cat(embs, dim=1)
        att_2d_masks = make_prefix_att_2d_masks(image_pad_masks, masks, subtask_pad_mask)
        pad_masks = torch.cat(
            [*image_pad_masks, masks] + ([subtask_pad_mask] if subtask_pad_mask is not None else []),
            dim=1,
        )

        return embs, pad_masks, att_2d_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        action_mask=None,
        noise=None,
        time=None,
        subtask_tokens: torch.Tensor | None = None,
        subtask_masks: torch.Tensor | None = None,
        tactile_tokens: Tensor | None = None,
        tactile_token_masks: Tensor | None = None,
        tactile_role_metadata: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Do a full training forward pass and compute flow and optional subtask CE losses."""
        if action_mask is not None:
            action_mask = action_mask.to(device=actions.device, dtype=actions.dtype)
            if action_mask.ndim == 2:
                action_mask = action_mask[:, None, :]
            actions = actions * action_mask

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if action_mask is not None:
            noise = noise * action_mask

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
            tactile_tokens=tactile_tokens,
            tactile_token_masks=tactile_token_masks,
            tactile_role_metadata=tactile_role_metadata,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]
        bsize = prefix_pad_masks.shape[0]
        prefix_to_suffix_mask = prefix_pad_masks[:, :, None].expand(bsize, prefix_len, suffix_len)
        suffix_to_prefix_mask = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        upper = torch.cat([prefix_att_masks, prefix_to_suffix_mask], dim=2)
        lower = torch.cat([suffix_to_prefix_mask, suffix_att_2d_masks], dim=2)
        att_2d_masks = torch.cat([upper, lower], dim=1)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        flow_losses = F.mse_loss(u_t, v_t, reduction="none")
        if action_mask is not None:
            flow_losses = flow_losses * action_mask
        predicted_actions = x_t - time_expanded * v_t

        subtask_ce_loss = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
        use_subtask_branch = self.config.enable_subtask_branch and subtask_tokens is not None and subtask_masks is not None
        if use_subtask_branch:
            lm_head = self.paligemma_with_expert.paligemma.lm_head
            language_len = tokens.shape[1]
            subtask_len = subtask_tokens.shape[1]
            subtask_hidden = prefix_out[:, -subtask_len:, :] if language_len == 0 else prefix_out[:, -subtask_len:, :]
            subtask_logits = lm_head(subtask_hidden)
            shifted_logits = subtask_logits[:, :-1, :]
            shifted_targets = subtask_tokens[:, 1:]
            shifted_masks = subtask_masks[:, 1:].to(dtype=torch.float32)

            if shifted_logits.numel() > 0:
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                flat_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))
                flat_targets = shifted_targets.reshape(-1)
                subtask_loss_per_token = loss_fct(flat_logits, flat_targets).reshape(shifted_targets.shape)
                masked_subtask_loss = subtask_loss_per_token * shifted_masks
                subtask_ce_loss = masked_subtask_loss.sum(dim=1) / shifted_masks.sum(dim=1).clamp(min=1.0)

        return {
            "flow_losses": flow_losses,
            "predicted_actions": predicted_actions,
            "subtask_ce_loss_per_sample": subtask_ce_loss,
        }

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        subtask_tokens: torch.Tensor | None = None,
        subtask_masks: torch.Tensor | None = None,
        tactile_tokens: Tensor | None = None,
        tactile_token_masks: Tensor | None = None,
        tactile_role_metadata: dict[str, Tensor] | None = None,
        action_mask=None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        if action_mask is not None:
            action_mask = action_mask.to(device=device, dtype=noise.dtype)
            if action_mask.ndim == 2:
                action_mask = action_mask[:, None, :].expand(bsize, self.config.chunk_size, -1)
            noise = noise * action_mask

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
            tactile_tokens=tactile_tokens,
            tactile_token_masks=tactile_token_masks,
            tactile_role_metadata=tactile_role_metadata,
        )
        prefix_att_2d_masks = prefix_att_masks
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            if action_mask is not None:
                v_t = v_t * action_mask
            x_t = x_t + dt * v_t
            if action_mask is not None:
                x_t = x_t * action_mask

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    @torch.no_grad()
    def generate_subtask_tokens(
        self,
        images,
        img_masks,
        tokens,
        masks,
        tactile_tokens: Tensor | None = None,
        tactile_token_masks: Tensor | None = None,
        tactile_role_metadata: dict[str, Tensor] | None = None,
        max_decoding_steps: int | None = None,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._paligemma_tokenizer is None:
            if AutoTokenizer is None:
                raise ImportError("transformers AutoTokenizer is required for pi05_full subtask generation.")
            self._paligemma_tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_decoding_steps
        if temperature is None:
            temperature = self.config.temperature

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        bos_token_id = self._paligemma_tokenizer.bos_token_id
        eos_token_id = self._paligemma_tokenizer.eos_token_id
        if bos_token_id is None or eos_token_id is None:
            raise RuntimeError("Tokenizer must define both bos_token_id and eos_token_id for subtask generation.")

        bos_token = torch.full((bsize, 1), bos_token_id, dtype=torch.long, device=device)
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens_in,
            masks_in,
            subtask_tokens=None,
            subtask_masks=None,
            tactile_tokens=tactile_tokens,
            tactile_token_masks=tactile_token_masks,
            tactile_role_metadata=tactile_role_metadata,
        )

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        last_logits = lm_head(prefix_out[:, -1:, :])
        if temperature > 0:
            probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

        generated_subtask_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
        generated_subtask_masks = torch.zeros((bsize, max_decoding_steps), dtype=torch.bool, device=device)
        generated_subtask_tokens[:, 0] = next_token.squeeze(-1)
        generated_subtask_masks[:, 0] = True

        finished = next_token.squeeze(-1) == eos_token_id
        current_pad_mask = prefix_pad_masks

        for t in range(1, max_decoding_steps):
            next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            if prefix_embs.dtype == torch.bfloat16:
                next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

            new_column = torch.ones((bsize, 1), dtype=torch.bool, device=device)
            current_pad_mask = torch.cat([current_pad_mask, new_column], dim=1)
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = self._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1), dtype=next_token_emb.dtype
            )

            (step_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )

            last_logits = lm_head(step_out[:, -1:, :])
            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            next_token = torch.where(finished.unsqueeze(-1), torch.zeros_like(next_token), next_token)
            generated_subtask_tokens[:, t] = next_token.squeeze(-1)
            generated_subtask_masks[:, t] = ~finished
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            if finished.all():
                break

        return generated_subtask_tokens, generated_subtask_masks

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05FullPolicy(PreTrainedPolicy):
    """PI05 Full Policy for LeRobot."""

    config_class = PI05FullConfig
    name = "pi05_full"

    def __init__(
        self,
        config: PI05FullConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            model_state_dict = model.state_dict()
            compatible_state_dict = {}
            shape_mismatched_keys = []
            for key, value in remapped_state_dict.items():
                target_value = model_state_dict.get(key)
                if target_value is not None and target_value.shape != value.shape:
                    shape_mismatched_keys.append((key, tuple(value.shape), tuple(target_value.shape)))
                    continue
                compatible_state_dict[key] = value

            if shape_mismatched_keys:
                print(f"Skipping {len(shape_mismatched_keys)} state dict keys with incompatible shapes")
                for key, checkpoint_shape, model_shape in shape_mismatched_keys[:10]:
                    print(f"  - {key}: checkpoint {checkpoint_shape} -> model {model_shape}")
                if len(shape_mismatched_keys) > 10:
                    print(f"  ... and {len(shape_mismatched_keys) - 10} more")

            # Load the remapped state dict into the model
            # Tactile-enabled models commonly warm-start from RGB-only checkpoints, so
            # tactile branch parameters are expected to be missing on first load.
            load_strict = strict and not shape_mismatched_keys and not getattr(model.config, "enable_tactile", False)
            missing_keys, unexpected_keys = model.load_state_dict(compatible_state_dict, strict=load_strict)
            skipped_key_names = {key for key, _, _ in shape_mismatched_keys}

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")
            elif shape_mismatched_keys and not unexpected_keys and set(missing_keys).issubset(skipped_key_names):
                print("Compatible pretrained keys loaded successfully; shape-mismatched keys were reinitialized.")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        if getattr(model.config, "reset_action_expert_after_pretrained_load", False):
            model.model.reset_action_expert_parameters()

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if hasattr(self, "_lerobot_processed_action_queue"):
            self._lerobot_processed_action_queue = deque(maxlen=self.config.n_action_steps)
        self._cached_subtask_tokens: Tensor | None = None
        self._cached_subtask_masks: Tensor | None = None
        self._cached_subtask_prompt_tokens: Tensor | None = None
        self._cached_subtask_prompt_masks: Tensor | None = None
        self._last_subtask_time: float | None = None
        self._inter_arm_position_indices_cache: tuple[tuple[int, int, int], tuple[int, int, int]] | None | bool = None
        self._arm_pose_indices_cache: tuple[tuple[int, ...], tuple[int, ...]] | None | bool = None

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    @staticmethod
    def _find_feature_index(names: list[str] | None, candidates: list[str]) -> int | None:
        if names is None:
            return None
        lowered = {str(name).lower(): idx for idx, name in enumerate(names)}
        for candidate in candidates:
            idx = lowered.get(candidate.lower())
            if idx is not None:
                return idx
        return None

    def _resolve_inter_arm_position_indices(self, action_dim: int) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        cached = getattr(self, "_inter_arm_position_indices_cache", None)
        if cached is False:
            return None
        if isinstance(cached, tuple):
            return cached

        action_names = getattr(self.config, "action_feature_names", None)
        left_candidates = {
            "x": ["left_x", "main_left_x", "left_arm_pose.x", "left_pose.x"],
            "y": ["left_y", "main_left_y", "left_arm_pose.y", "left_pose.y"],
            "z": ["left_z", "main_left_z", "left_arm_pose.z", "left_pose.z"],
        }
        right_candidates = {
            "x": ["right_x", "main_right_x", "right_arm_pose.x", "right_pose.x"],
            "y": ["right_y", "main_right_y", "right_arm_pose.y", "right_pose.y"],
            "z": ["right_z", "main_right_z", "right_arm_pose.z", "right_pose.z"],
        }

        left_indices = tuple(
            self._find_feature_index(action_names, left_candidates[axis]) for axis in ("x", "y", "z")
        )
        right_indices = tuple(
            self._find_feature_index(action_names, right_candidates[axis]) for axis in ("x", "y", "z")
        )

        if any(idx is None for idx in left_indices + right_indices):
            if action_dim >= 13:
                resolved = ((0, 1, 2), (10, 11, 12))
                self._inter_arm_position_indices_cache = resolved
                return resolved
            self._inter_arm_position_indices_cache = False
            return None

        resolved = (
            tuple(int(idx) for idx in left_indices),
            tuple(int(idx) for idx in right_indices),
        )
        self._inter_arm_position_indices_cache = resolved
        return resolved

    @staticmethod
    def _is_named_arm_pose_feature(name: str, arm: str) -> bool:
        lowered = str(name).lower()
        if "gripper" in lowered:
            return False
        arm_aliases = {
            "left": ("left_", "main_left_", "left_arm_pose.", "left_pose."),
            "right": ("right_", "main_right_", "right_arm_pose.", "right_pose."),
        }
        return any(alias in lowered for alias in arm_aliases[arm])

    def _resolve_arm_pose_indices(self, action_dim: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        cached = getattr(self, "_arm_pose_indices_cache", None)
        if cached is False:
            return None
        if isinstance(cached, tuple):
            return cached

        action_names = getattr(self.config, "action_feature_names", None)
        if action_names is not None:
            left_indices = tuple(
                idx for idx, name in enumerate(action_names) if self._is_named_arm_pose_feature(name, "left")
            )
            right_indices = tuple(
                idx for idx, name in enumerate(action_names) if self._is_named_arm_pose_feature(name, "right")
            )
            if left_indices and right_indices:
                resolved = (left_indices, right_indices)
                self._arm_pose_indices_cache = resolved
                return resolved

        if action_dim >= 19:
            resolved = (tuple(range(0, 9)), tuple(range(10, 19)))
            self._arm_pose_indices_cache = resolved
            return resolved

        self._arm_pose_indices_cache = False
        return None

    def _compute_inter_arm_position_loss(
        self,
        predicted_actions: Tensor,
        target_actions: Tensor,
        action_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.config.loss_weight_inter_arm_position <= 0:
            return None, None

        indices = self._resolve_inter_arm_position_indices(predicted_actions.shape[-1])
        if indices is None:
            return None, None

        left_indices, right_indices = indices
        pred_left = predicted_actions[..., list(left_indices)]
        pred_right = predicted_actions[..., list(right_indices)]
        target_left = target_actions[..., list(left_indices)]
        target_right = target_actions[..., list(right_indices)]

        pred_span = pred_right - pred_left
        target_span = target_right - target_left
        losses = F.mse_loss(pred_span, target_span, reduction="none")

        if action_mask is not None:
            expanded_mask = action_mask
            if expanded_mask.ndim == 2:
                expanded_mask = expanded_mask[:, None, :]
            left_valid = expanded_mask[..., list(left_indices)] > 0
            right_valid = expanded_mask[..., list(right_indices)] > 0
            pair_mask = (left_valid & right_valid).to(dtype=losses.dtype)
            losses = losses * pair_mask
            per_sample_denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
            per_sample_loss = losses.sum(dim=(1, 2)) / per_sample_denom
            scalar_loss = losses.sum() / pair_mask.sum().clamp_min(1.0)
        else:
            per_sample_loss = losses.mean(dim=(1, 2))
            scalar_loss = losses.mean()

        return per_sample_loss, scalar_loss

    @staticmethod
    def _masked_mean_last_dim(values: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return values.mean(dim=-1)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return (values * mask).sum(dim=-1) / denom

    @staticmethod
    def _expand_action_mask_over_time(mask: Tensor | None, target_time_dim: int) -> Tensor | None:
        if mask is None:
            return None
        if mask.ndim == 2:
            mask = mask[:, None, :]
        if mask.shape[1] == 1 and target_time_dim != 1:
            mask = mask.expand(-1, target_time_dim, -1)
        elif mask.shape[1] != target_time_dim:
            raise ValueError(
                "Action mask time dimension does not match predicted action chunk length: "
                f"mask.shape={tuple(mask.shape)} target_time_dim={target_time_dim}"
            )
        return mask

    def _compute_idle_arm_losses(
        self,
        predicted_actions: Tensor,
        target_actions: Tensor,
        action_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if self.config.loss_weight_idle_arm <= 0 and self.config.loss_weight_idle_smoothness <= 0:
            return None, None, None, None

        indices = self._resolve_arm_pose_indices(predicted_actions.shape[-1])
        if indices is None:
            return None, None, None, None

        expanded_mask = None
        if action_mask is not None:
            expanded_mask = self._expand_action_mask_over_time(action_mask, predicted_actions.shape[1])
            expanded_mask = expanded_mask.to(device=predicted_actions.device, dtype=predicted_actions.dtype)

        idle_weighted_losses: list[Tensor] = []
        idle_sample_counts: list[Tensor] = []
        smooth_weighted_losses: list[Tensor] = []
        smooth_sample_counts: list[Tensor] = []

        for arm_indices in indices:
            pred_arm = predicted_actions[..., list(arm_indices)]
            target_arm = target_actions[..., list(arm_indices)]
            arm_mask = expanded_mask[..., list(arm_indices)] if expanded_mask is not None else None

            activity = self._masked_mean_last_dim(target_arm.abs(), arm_mask)
            idle_step_mask = (activity < self.config.idle_arm_threshold).to(dtype=predicted_actions.dtype)

            pose_errors = F.mse_loss(pred_arm, target_arm, reduction="none")
            pose_step_losses = self._masked_mean_last_dim(pose_errors, arm_mask)
            idle_weighted_losses.append((pose_step_losses * idle_step_mask).sum(dim=1))
            idle_sample_counts.append(idle_step_mask.sum(dim=1))

            if pred_arm.shape[1] > 1:
                pred_step_delta = pred_arm[:, 1:, :] - pred_arm[:, :-1, :]
                if arm_mask is not None:
                    delta_mask = arm_mask[:, 1:, :] * arm_mask[:, :-1, :]
                else:
                    delta_mask = None
                smooth_step_losses = self._masked_mean_last_dim(pred_step_delta.pow(2), delta_mask)
                idle_pair_mask = idle_step_mask[:, 1:] * idle_step_mask[:, :-1]
                smooth_weighted_losses.append((smooth_step_losses * idle_pair_mask).sum(dim=1))
                smooth_sample_counts.append(idle_pair_mask.sum(dim=1))

        idle_loss_per_sample = None
        idle_loss = None
        if idle_weighted_losses:
            idle_total = torch.stack(idle_weighted_losses, dim=0).sum(dim=0)
            idle_counts = torch.stack(idle_sample_counts, dim=0).sum(dim=0)
            idle_loss_per_sample = idle_total / idle_counts.clamp_min(1.0)
            valid_idle_samples = idle_counts > 0
            idle_loss = (
                idle_total[valid_idle_samples].sum() / idle_counts[valid_idle_samples].sum().clamp_min(1.0)
                if torch.any(valid_idle_samples)
                else torch.zeros((), device=predicted_actions.device, dtype=predicted_actions.dtype)
            )

        idle_smoothness_per_sample = None
        idle_smoothness_loss = None
        if smooth_weighted_losses:
            smooth_total = torch.stack(smooth_weighted_losses, dim=0).sum(dim=0)
            smooth_counts = torch.stack(smooth_sample_counts, dim=0).sum(dim=0)
            idle_smoothness_per_sample = smooth_total / smooth_counts.clamp_min(1.0)
            valid_smooth_samples = smooth_counts > 0
            idle_smoothness_loss = (
                smooth_total[valid_smooth_samples].sum() / smooth_counts[valid_smooth_samples].sum().clamp_min(1.0)
                if torch.any(valid_smooth_samples)
                else torch.zeros((), device=predicted_actions.device, dtype=predicted_actions.dtype)
            )

        return idle_loss_per_sample, idle_loss, idle_smoothness_per_sample, idle_smoothness_loss

    def _infer_batch_size(self, batch: dict[str, Tensor]) -> int:
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        raise ValueError("Could not infer batch size from the batch.")

    def _resolve_tactile_keys(self, batch: dict[str, Tensor]) -> list[str]:
        if not self.config.enable_tactile:
            return []
        if self.config.tactile_feature_keys:
            return [key for key in self.config.tactile_feature_keys if key in batch]
        return [key for key in self.config.image_features if "tactile" in key and key in batch]

    def _resolve_rgb_image_keys(self, batch: dict[str, Tensor]) -> list[str]:
        tactile_keys = set(self._resolve_tactile_keys(batch))
        return [key for key in self.config.image_features if key in batch and key not in tactile_keys]

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        tactile_keys = set(self._resolve_tactile_keys(batch))
        rgb_image_features = [key for key in self.config.image_features if key not in tactile_keys]
        present_img_keys = [key for key in rgb_image_features if key in batch]
        missing_img_keys = [key for key in rgb_image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (rgb_image_features: {rgb_image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                # img = resize_with_pad_torch(img, *self.config.image_resolution)
                img = resize_without_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            missing_mask_key = f"{key}_is_missing"
            if missing_mask_key in batch:
                mask = ~batch[missing_mask_key].to(device=device, dtype=torch.bool)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _build_tactile_role_metadata(
        self,
        *,
        batch: dict[str, Tensor],
        tactile_keys: list[str],
        tactile_segments: list[tuple[str, int, int]],
        present: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, Tensor] | None:
        if not self.config.enable_tactile_role_embeddings or not tactile_segments:
            return None
        if any(key not in TACTILE_SENSOR_SLOT_ID_BY_KEY for key in tactile_keys):
            return None

        slot_present = batch.get(self.config.tactile_slot_present_key)
        if torch.is_tensor(slot_present):
            slot_present = slot_present.to(device=device, dtype=torch.bool)
            if slot_present.ndim == 1:
                slot_present = slot_present[None, :]
            else:
                slot_present = slot_present.reshape(slot_present.shape[0], -1)
        else:
            slot_present = None

        num_tokens = max(end for _, _, end in tactile_segments)
        modality_ids = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=device)
        hand_side_ids = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=device)
        finger_slot_ids = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=device)
        sensor_slot_ids = torch.zeros((batch_size, num_tokens), dtype=torch.long, device=device)
        token_active_mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)

        for key, start, end in tactile_segments:
            sensor_slot_id = TACTILE_SENSOR_SLOT_ID_BY_KEY[key]
            hand_side_ids[:, start:end] = TACTILE_HAND_SIDE_ID_BY_KEY[key]
            finger_slot_ids[:, start:end] = TACTILE_FINGER_SLOT_ID_BY_KEY[key]
            sensor_slot_ids[:, start:end] = sensor_slot_id
            if slot_present is not None and sensor_slot_id < slot_present.shape[1]:
                token_active_mask[:, start:end] = slot_present[:, sensor_slot_id : sensor_slot_id + 1]
            else:
                token_active_mask[:, start:end] = present[:, None]

        return {
            "modality_ids": modality_ids,
            "hand_side_ids": hand_side_ids,
            "finger_slot_ids": finger_slot_ids,
            "sensor_slot_ids": sensor_slot_ids,
            "token_active_mask": token_active_mask,
        }

    def _encode_parsed_tactile_sensor(self, tensor: Tensor, sensor_slot_id: int | None = None) -> Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"Parsed tactile tensor must be rank-4 [B,C,H,W], got {tuple(tensor.shape)}")
        target_h, target_w = self.config.tactile_target_size
        if tuple(tensor.shape[-2:]) != (target_h, target_w):
            tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        patches = self.model.tactile_patch_proj(tensor)
        patches = patches.flatten(2).transpose(1, 2)
        pos_ids = torch.arange(patches.shape[1], device=patches.device)
        patches = patches + self.model.tactile_patch_pos_embed(pos_ids)[None, :, :]
        for block in self.model.tactile_patch_blocks:
            patches = block(patches)
        patches = self.model.tactile_patch_layernorm(patches)
        query = self.model.tactile_query_tokens.expand(patches.shape[0], -1, -1)
        resampled, _ = self.model.tactile_resampler_attn(query, patches, patches, need_weights=False)
        resampled = resampled + self.model.tactile_resampler_ff(self.model.tactile_token_layernorm(resampled))
        if sensor_slot_id is not None:
            scale = self.model.tactile_sensor_scale.weight[sensor_slot_id].view(1, 1, 1)
            resampled = resampled * scale
        return resampled

    def _encode_contact_stats_sensor(self, tensor: Tensor) -> Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"Contact tactile tensor must be rank-4 [B,C,H,W], got {tuple(tensor.shape)}")
        tensor = tensor.to(dtype=torch.float32)
        if tensor.shape[1] > 1:
            tensor = tensor.mean(dim=1, keepdim=True)
        flattened = tensor.flatten(start_dim=2)
        topk_ratio = self.config.tactile_contact_topk_ratio
        k = max(1, math.ceil(flattened.shape[-1] * topk_ratio))
        contact_score = torch.topk(flattened, k=k, dim=-1).values.mean(dim=-1, keepdim=True)
        return contact_score

    def _prepare_tactile(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor | None, Tensor | None, dict[str, Tensor] | None]:
        if not self.config.enable_tactile:
            return None, None, None

        tactile_keys = self._resolve_tactile_keys(batch)
        if not tactile_keys and not self.config.tactile_fallback_to_state:
            return None, None, None

        device = next(self.parameters()).device
        batch_size = self._infer_batch_size(batch)
        tactile_segments: list[tuple[str, int, int]] = []
        cursor = 0

        if self.config.tactile_encoder_type == "parsed_vit" and tactile_keys:
            parsed_tokens_per_sensor: list[Tensor] = []
            for key in tactile_keys:
                tensor = batch[key].to(device=device, dtype=torch.float32)
                if tensor.ndim == 5:
                    tensor = tensor[:, -1]
                if tensor.ndim == 4 and tensor.shape[-1] in {1, 3} and tensor.shape[1] not in {1, 3}:
                    tensor = tensor.permute(0, 3, 1, 2)
                if tensor.ndim != 4:
                    raise ValueError(f"Parsed tactile feature {key} must be rank-4, got {tuple(tensor.shape)}")
                if tensor.shape[1] != self.config.tactile_input_channels:
                    raise ValueError(
                        f"Parsed tactile feature {key} must have {self.config.tactile_input_channels} channels, "
                        f"got {tensor.shape[1]}"
                    )
                sensor_slot_id = TACTILE_SENSOR_SLOT_ID_BY_KEY.get(key)
                encoded = self._encode_parsed_tactile_sensor(tensor, sensor_slot_id=sensor_slot_id)
                parsed_tokens_per_sensor.append(encoded)
                tactile_segments.append((key, cursor, cursor + encoded.shape[1]))
                cursor += encoded.shape[1]

            tactile_tokens = torch.cat(parsed_tokens_per_sensor, dim=1)
        elif self.config.tactile_encoder_type == "contact_stats" and tactile_keys:
            contact_tokens_per_sensor: list[Tensor] = []
            for key in tactile_keys:
                tensor = batch[key].to(device=device, dtype=torch.float32)
                if tensor.ndim == 5:
                    tensor = tensor[:, -1]
                if tensor.ndim == 4 and tensor.shape[-1] in {1, 3} and tensor.shape[1] not in {1, 3}:
                    tensor = tensor.permute(0, 3, 1, 2)
                if tensor.ndim != 4:
                    raise ValueError(f"Contact tactile feature {key} must be rank-4, got {tuple(tensor.shape)}")
                encoded = self._encode_contact_stats_sensor(tensor)
                contact_tokens_per_sensor.append(encoded)
                tactile_segments.append((key, cursor, cursor + encoded.shape[1]))
                cursor += encoded.shape[1]

            tactile_tokens = torch.cat(contact_tokens_per_sensor, dim=1)
        else:
            tactile_vectors: list[Tensor] = []
            num_tactile_keys = max(1, len(tactile_keys))
            target_dim = self.config.tactile_tokens * self.config.max_tactile_dim
            per_key_budget = max(1, target_dim // num_tactile_keys)

            for key in tactile_keys:
                tensor = batch[key].to(device=device, dtype=torch.float32)
                if tensor.ndim == 5:
                    tensor = tensor[:, -1]
                if tensor.ndim == 4 and tensor.shape[-1] in {1, 3} and tensor.shape[1] not in {1, 3}:
                    tensor = tensor.permute(0, 3, 1, 2)
                if tensor.ndim == 4 and tensor.shape[1] in {1, 3}:
                    channels = int(tensor.shape[1])
                    max_side = int(max(1, math.floor(math.sqrt(per_key_budget / max(1, channels)))))
                    pool_side = max(1, min(16, max_side))
                    tensor = F.adaptive_avg_pool2d(tensor, output_size=(pool_side, pool_side))
                flattened = tensor.reshape(tensor.shape[0], -1)
                tactile_vectors.append(flattened)
                token_start = cursor // self.config.max_tactile_dim
                cursor += flattened.shape[1]
                token_end = min(self.config.tactile_tokens, math.ceil(cursor / self.config.max_tactile_dim))
                if token_end > token_start:
                    tactile_segments.append((key, token_start, token_end))

            if not tactile_vectors and self.config.tactile_fallback_to_state and OBS_STATE in batch:
                state = batch[OBS_STATE].to(device=device, dtype=torch.float32)
                tactile_vectors.append(state.reshape(state.shape[0], -1))

            if not tactile_vectors:
                return None, None, None

            tactile_vector = torch.cat(tactile_vectors, dim=1)
            tactile_vector = pad_vector(tactile_vector, self.config.tactile_tokens * self.config.max_tactile_dim)
            tactile_vector = tactile_vector[:, : self.config.tactile_tokens * self.config.max_tactile_dim]
            tactile_tokens = tactile_vector.view(
                tactile_vector.shape[0],
                self.config.tactile_tokens,
                self.config.max_tactile_dim,
            )

        present = batch.get(self.config.tactile_mask_key)
        if torch.is_tensor(present):
            present = present.to(device=device, dtype=torch.bool)
            if present.ndim > 1:
                present = present.reshape(present.shape[0], -1).any(dim=1)
        else:
            present = torch.ones(batch_size, dtype=torch.bool, device=device)

        tactile_role_metadata = self._build_tactile_role_metadata(
            batch=batch,
            tactile_keys=tactile_keys,
            tactile_segments=tactile_segments,
            present=present,
            batch_size=batch_size,
            device=device,
        )
        tactile_token_masks = torch.ones(
            tactile_tokens.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        if tactile_role_metadata is not None:
            tactile_token_masks = tactile_token_masks & tactile_role_metadata["token_active_mask"]
        else:
            tactile_token_masks = tactile_token_masks & present[:, None]
        return tactile_tokens, tactile_token_masks, tactile_role_metadata

    def _collect_tactile_metrics(
        self,
        batch: dict[str, Tensor],
        tactile_tokens: Tensor | None,
        tactile_token_masks: Tensor | None,
    ) -> dict[str, float]:
        if not self.config.enable_tactile:
            return {}

        metrics: dict[str, float] = {
            "tactile_num_sensors": float(len(self._resolve_tactile_keys(batch))),
        }
        if hasattr(self.model, "tactile_gate"):
            metrics["tactile_gate"] = float(torch.sigmoid(self.model.tactile_gate.detach()).item())

        if tactile_tokens is None:
            metrics["tactile_present_mean"] = 0.0
            metrics["tactile_token_mask_mean"] = 0.0
            return metrics

        if tactile_token_masks is None:
            tactile_token_masks = torch.ones(
                tactile_tokens.shape[:2],
                dtype=torch.bool,
                device=tactile_tokens.device,
            )
        else:
            tactile_token_masks = tactile_token_masks.to(device=tactile_tokens.device, dtype=torch.bool)

        metrics["tactile_token_mask_mean"] = float(tactile_token_masks.float().mean().item())
        metrics["tactile_present_mean"] = float(tactile_token_masks.any(dim=1).float().mean().item())

        valid_token_rows = tactile_tokens[tactile_token_masks]
        if valid_token_rows.numel() == 0:
            metrics["tactile_token_abs_mean"] = 0.0
            metrics["tactile_token_std"] = 0.0
            return metrics

        metrics["tactile_token_abs_mean"] = float(valid_token_rows.abs().mean().item())
        metrics["tactile_token_std"] = (
            float(valid_token_rows.float().std(unbiased=False).item()) if valid_token_rows.numel() > 1 else 0.0
        )

        if self.config.tactile_encoder_type == "contact_stats" and valid_token_rows.shape[-1] == 1:
            contact_values = valid_token_rows[:, 0]
            metrics["tactile_contact_mean"] = float(contact_values.mean().item())
            metrics["tactile_contact_max"] = float(contact_values.max().item())
            metrics["tactile_contact_std"] = (
                float(contact_values.float().std(unbiased=False).item()) if contact_values.numel() > 1 else 0.0
            )

            tactile_keys = self._resolve_tactile_keys(batch)
            for sensor_idx, key in enumerate(tactile_keys[: tactile_tokens.shape[1]]):
                sensor_mask = tactile_token_masks[:, sensor_idx]
                if not torch.any(sensor_mask):
                    continue
                sensor_values = tactile_tokens[:, sensor_idx, 0][sensor_mask]
                sensor_name = key.split(".")[-1]
                metrics[f"tactile_contact_{sensor_name}_mean"] = float(sensor_values.mean().item())

        return metrics

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def prepare_action_mask(self, batch):
        action_mask = batch.get(self.config.action_mask_observation_key, batch.get(OBS_ACTION_MASK))
        sidecar_action_mask = batch.get("action_valid_mask")
        if action_mask is None:
            action_mask = sidecar_action_mask
        elif sidecar_action_mask is not None and torch.is_tensor(action_mask) and torch.is_tensor(sidecar_action_mask):
            if action_mask.sum() <= 0 and sidecar_action_mask.sum() > 0:
                action_mask = sidecar_action_mask
        if action_mask is None:
            return None
        return pad_vector(action_mask, self.config.max_action_dim)

    @staticmethod
    def _debug_tensor_summary(batch: dict[str, Tensor], key: str) -> str:
        value = batch.get(key)
        if value is None:
            return f"{key}=None"
        if not torch.is_tensor(value):
            return f"{key}=type({type(value).__name__})"
        return (
            f"{key}:shape={tuple(value.shape)} "
            f"sum={float(value.float().sum().detach().cpu()):.3f} "
            f"mean={float(value.float().mean().detach().cpu()):.3f}"
        )

    @staticmethod
    def get_language_inputs(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        tokens = batch.get(OBS_LANGUAGE_USER_PROMPT_TOKENS, batch[OBS_LANGUAGE_TOKENS])
        masks = batch.get(OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK, batch[OBS_LANGUAGE_ATTENTION_MASK])
        return tokens, masks

    @staticmethod
    def get_subtask_inputs(batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        return batch.get(OBS_LANGUAGE_SUBTASK_TOKENS), batch.get(OBS_LANGUAGE_SUBTASK_ATTENTION_MASK)

    @staticmethod
    def _coerce_bool_batch(
        value: Tensor | bool | list[bool] | tuple[bool, ...] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=torch.bool)
        else:
            tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        tensor = tensor.reshape(-1)
        if tensor.numel() == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size)
        if tensor.numel() != batch_size:
            raise ValueError(
                f"Expected boolean batch of size {batch_size}, got {tensor.numel()} elements instead."
            )
        return tensor

    @staticmethod
    def _pad_sequence_dim(tensor: Tensor, target_len: int, pad_value: int | bool = 0) -> Tensor:
        current_len = tensor.shape[1]
        if current_len == target_len:
            return tensor
        if current_len > target_len:
            return tensor[:, :target_len]
        return F.pad(tensor, (0, target_len - current_len), value=pad_value)

    def _ensure_paligemma_tokenizer(self):
        if self.model._paligemma_tokenizer is not None:
            return self.model._paligemma_tokenizer
        if AutoTokenizer is None:
            raise ImportError("transformers AutoTokenizer is required for pi05_full subtask generation.")
        try:
            self.model._paligemma_tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer '{self.config.tokenizer_name}' for pi05_full subtask generation."
            ) from exc
        return self.model._paligemma_tokenizer

    def _invalidate_subtask_cache(self) -> None:
        self._cached_subtask_tokens = None
        self._cached_subtask_masks = None
        self._cached_subtask_prompt_tokens = None
        self._cached_subtask_prompt_masks = None
        self._last_subtask_time = None

    def _resolve_subtask_inputs(
        self,
        batch: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None, Tensor]:
        if not self.config.enable_subtask_branch:
            fallback_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return None, None, fallback_mask
        subtask_tokens, subtask_masks = self.get_subtask_inputs(batch)
        if (subtask_tokens is None) != (subtask_masks is None):
            raise ValueError("subtask tokens and attention masks must either both be present or both be absent.")

        if subtask_tokens is None or subtask_masks is None:
            fallback_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            return None, None, fallback_mask

        fallback_mask = self._coerce_bool_batch(
            batch.get(PI05_FULL_SUBTASK_IS_FALLBACK_KEY),
            batch_size=batch_size,
            device=device,
        )
        return subtask_tokens, subtask_masks, fallback_mask

    def _cached_prompt_matches(self, tokens: Tensor, masks: Tensor) -> bool:
        if self._cached_subtask_prompt_tokens is None or self._cached_subtask_prompt_masks is None:
            return False
        current_tokens = tokens.detach().cpu()
        current_masks = masks.detach().cpu()
        return torch.equal(self._cached_subtask_prompt_tokens, current_tokens) and torch.equal(
            self._cached_subtask_prompt_masks, current_masks
        )

    def _should_regenerate_subtasks(self, tokens: Tensor, masks: Tensor, current_time: float) -> bool:
        if self._cached_subtask_tokens is None or self._cached_subtask_masks is None:
            return True
        if self._last_subtask_time is None:
            return True
        if not self._cached_prompt_matches(tokens, masks):
            return True
        interval = self.config.subtask_regeneration_interval
        if interval <= 0:
            return True
        return (current_time - self._last_subtask_time) >= interval

    def _get_or_generate_subtasks(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        tactile_tokens: Tensor | None = None,
        tactile_token_masks: Tensor | None = None,
        tactile_role_metadata: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not self.config.enable_subtask_branch:
            raise RuntimeError("Subtask branch is disabled in pi05_full config; generation should not be called.")
        current_time = time.time()
        if self._should_regenerate_subtasks(tokens, masks, current_time):
            self._ensure_paligemma_tokenizer()
            subtask_tokens, subtask_masks = self.model.generate_subtask_tokens(
                images,
                img_masks,
                tokens,
                masks,
                tactile_tokens=tactile_tokens,
                tactile_token_masks=tactile_token_masks,
                tactile_role_metadata=tactile_role_metadata,
                max_decoding_steps=self.config.max_decoding_steps,
                temperature=self.config.temperature,
            )
            self._cached_subtask_tokens = subtask_tokens.detach().cpu()
            self._cached_subtask_masks = subtask_masks.detach().cpu()
            self._cached_subtask_prompt_tokens = tokens.detach().cpu()
            self._cached_subtask_prompt_masks = masks.detach().cpu()
            self._last_subtask_time = current_time

        assert self._cached_subtask_tokens is not None
        assert self._cached_subtask_masks is not None
        return (
            self._cached_subtask_tokens.to(device=tokens.device),
            self._cached_subtask_masks.to(device=tokens.device),
        )

    def _select_inference_subtasks(
        self,
        batch: dict[str, Tensor],
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        tactile_tokens: Tensor | None = None,
        tactile_token_masks: Tensor | None = None,
        tactile_role_metadata: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if not self.config.enable_subtask_branch:
            return None, None
        batch_size = tokens.shape[0]
        explicit_tokens, explicit_masks, fallback_mask = self._resolve_subtask_inputs(
            batch,
            batch_size=batch_size,
            device=tokens.device,
        )

        if explicit_tokens is not None and not torch.any(fallback_mask):
            self._invalidate_subtask_cache()
            return explicit_tokens, explicit_masks

        generated_tokens, generated_masks = self._get_or_generate_subtasks(
            images,
            img_masks,
            tokens,
            masks,
            tactile_tokens=tactile_tokens,
            tactile_token_masks=tactile_token_masks,
            tactile_role_metadata=tactile_role_metadata,
        )
        if explicit_tokens is None or explicit_masks is None:
            return generated_tokens, generated_masks

        target_len = max(explicit_tokens.shape[1], generated_tokens.shape[1])
        explicit_tokens = self._pad_sequence_dim(explicit_tokens, target_len, pad_value=0)
        explicit_masks = self._pad_sequence_dim(explicit_masks, target_len, pad_value=False)
        generated_tokens = self._pad_sequence_dim(generated_tokens, target_len, pad_value=0)
        generated_masks = self._pad_sequence_dim(generated_masks, target_len, pad_value=False)

        selector = fallback_mask[:, None]
        merged_tokens = torch.where(selector, generated_tokens, explicit_tokens)
        merged_masks = torch.where(selector, generated_masks, explicit_masks)
        return merged_tokens, merged_masks

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tactile_tokens, tactile_token_masks, tactile_role_metadata = self._prepare_tactile(batch)
        tokens, masks = self.get_language_inputs(batch)
        action_mask = self.prepare_action_mask(batch)
        subtask_tokens = None
        subtask_masks = None
        if self.config.enable_subtask_branch:
            subtask_tokens, subtask_masks = self._select_inference_subtasks(
                batch,
                images,
                img_masks,
                tokens,
                masks,
                tactile_tokens=tactile_tokens,
                tactile_token_masks=tactile_token_masks,
                tactile_role_metadata=tactile_role_metadata,
            )

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
            tactile_tokens=tactile_tokens,
            tactile_token_masks=tactile_token_masks,
            tactile_role_metadata=tactile_role_metadata,
            action_mask=action_mask,
            **kwargs,
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        if action_mask is not None:
            action_mask = action_mask[..., :original_action_dim]
            if action_mask.ndim == 2:
                action_mask = action_mask[:, None, :]
            actions = actions * action_mask

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tactile_tokens, tactile_token_masks, tactile_role_metadata = self._prepare_tactile(batch)
        tokens, masks = self.get_language_inputs(batch)
        subtask_tokens, subtask_masks = self.get_subtask_inputs(batch) if self.config.enable_subtask_branch else (None, None)

        actions = self.prepare_action(batch)
        action_mask = self.prepare_action_mask(batch)
        if self.training and action_mask is None:
            debug_keys = (
                self.config.action_mask_observation_key,
                OBS_ACTION_MASK,
                "action_valid_mask",
                ACTION,
            )
            debug_summary = "; ".join(self._debug_tensor_summary(batch, key) for key in debug_keys)
            raise ValueError(
                "PI05 full did not receive an action mask during training. "
                "Masked pi05_full training requires observation.action_mask/action_valid_mask. "
                f"Batch debug: {debug_summary}"
            )

        model_outputs = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            action_mask=action_mask,
            subtask_tokens=subtask_tokens if self.config.enable_subtask_branch else None,
            subtask_masks=subtask_masks if self.config.enable_subtask_branch else None,
            tactile_tokens=tactile_tokens,
            tactile_token_masks=tactile_token_masks,
            tactile_role_metadata=tactile_role_metadata,
        )
        losses = model_outputs["flow_losses"]
        predicted_actions = model_outputs["predicted_actions"]
        subtask_ce_loss_per_sample = model_outputs["subtask_ce_loss_per_sample"]

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        predicted_actions = predicted_actions[:, :, :original_action_dim]
        target_actions = actions[:, :, :original_action_dim]
        expanded_action_mask = None
        if action_mask is not None:
            expanded_action_mask = action_mask[..., :original_action_dim]
            if expanded_action_mask.ndim == 2:
                expanded_action_mask = expanded_action_mask[:, None, :]
            expanded_action_mask = expanded_action_mask.expand_as(losses)

        if expanded_action_mask is not None:
            per_dim_denom = expanded_action_mask.sum(dim=[0, 1]).clamp_min(1.0)
            loss_per_dim = losses.sum(dim=[0, 1]) / per_dim_denom
        else:
            loss_per_dim = losses.mean(dim=[0, 1])

        subtask_ce_loss = subtask_ce_loss_per_sample.mean()
        loss_dict = {
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
            "subtask_ce_loss": subtask_ce_loss.item(),
        }
        loss_dict.update(self._collect_tactile_metrics(batch, tactile_tokens, tactile_token_masks))
        inter_arm_pos_loss_per_sample, inter_arm_pos_loss = self._compute_inter_arm_position_loss(
            predicted_actions,
            target_actions,
            action_mask[..., :original_action_dim] if action_mask is not None else None,
        )
        idle_arm_loss_per_sample, idle_arm_loss, idle_smoothness_per_sample, idle_smoothness_loss = (
            self._compute_idle_arm_losses(
                predicted_actions,
                target_actions,
                action_mask[..., :original_action_dim] if action_mask is not None else None,
            )
        )
        if inter_arm_pos_loss is not None:
            loss_dict["inter_arm_pos_loss"] = inter_arm_pos_loss.item()
        if idle_arm_loss is not None:
            loss_dict["idle_arm_loss"] = idle_arm_loss.item()
        if idle_smoothness_loss is not None:
            loss_dict["idle_smoothness_loss"] = idle_smoothness_loss.item()
        if expanded_action_mask is not None:
            loss_dict["action_mask_mean"] = expanded_action_mask.float().mean().item()
            loss_dict["action_mask_sum"] = expanded_action_mask.float().sum().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            if expanded_action_mask is not None:
                per_sample_denom = expanded_action_mask.sum(dim=(1, 2)).clamp_min(1.0)
                flow_per_sample_loss = losses.sum(dim=(1, 2)) / per_sample_denom
            else:
                flow_per_sample_loss = losses.mean(dim=(1, 2))
            per_sample_loss = (
                self.config.loss_weight_flow * flow_per_sample_loss
                + self.config.loss_weight_subtask_ce * subtask_ce_loss_per_sample
            )
            if inter_arm_pos_loss_per_sample is not None:
                per_sample_loss = (
                    per_sample_loss
                    + self.config.loss_weight_inter_arm_position * inter_arm_pos_loss_per_sample
                )
            if idle_arm_loss_per_sample is not None:
                per_sample_loss = per_sample_loss + self.config.loss_weight_idle_arm * idle_arm_loss_per_sample
            if idle_smoothness_per_sample is not None:
                per_sample_loss = (
                    per_sample_loss
                    + self.config.loss_weight_idle_smoothness * idle_smoothness_per_sample
                )
            loss_dict["flow_loss"] = flow_per_sample_loss.mean().item()
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            if expanded_action_mask is not None:
                mask_sum = expanded_action_mask.sum()
                if self.training and mask_sum <= 0:
                    debug_keys = (
                        self.config.action_mask_observation_key,
                        OBS_ACTION_MASK,
                        "action_valid_mask",
                        ACTION,
                    )
                    debug_summary = "; ".join(self._debug_tensor_summary(batch, key) for key in debug_keys)
                    raise ValueError(
                        "PI05 full received an all-zero action mask during training; "
                        "flow loss would be fully masked out. Check observation.action_mask/action_valid_mask "
                        "in the dataset sidecar and preprocessing pipeline. "
                        f"Batch debug: {debug_summary}"
                    )
                flow_loss = losses.sum() / mask_sum.clamp_min(1.0)
            else:
                flow_loss = losses.mean()
            loss = self.config.loss_weight_flow * flow_loss + self.config.loss_weight_subtask_ce * subtask_ce_loss
            if inter_arm_pos_loss is not None:
                loss = loss + self.config.loss_weight_inter_arm_position * inter_arm_pos_loss
            if idle_arm_loss is not None:
                loss = loss + self.config.loss_weight_idle_arm * idle_arm_loss
            if idle_smoothness_loss is not None:
                loss = loss + self.config.loss_weight_idle_smoothness * idle_smoothness_loss
            loss_dict["flow_loss"] = flow_loss.item()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
