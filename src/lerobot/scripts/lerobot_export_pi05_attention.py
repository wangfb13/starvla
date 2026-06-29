#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Export PI05 Full training-sample attention overlays for a checkpoint.

This script runs a pi05_full checkpoint on training dataset samples and records
the full self-attention matrices produced during the training forward pass.
It then converts attention from action/suffix queries onto image patch tokens
into per-camera heatmaps and overlays them on the original training images.

Notes:
1. The exported maps come from the *training forward* of pi05_full, i.e.
   action-token queries attending to image tokens while denoising noised ground-
   truth actions. This matches how the model is optimized during training.
2. The script reuses `train_config.json` from the checkpoint so dataset roots,
   sidecars and rename maps stay aligned with the training run by default.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05_full import modeling_pi05


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@dataclass
class LayerAttentionRecord:
    layer_idx: int
    attention: torch.Tensor


class AttentionCollector:
    def __init__(self) -> None:
        self.records: list[LayerAttentionRecord] = []

    def clear(self) -> None:
        self.records.clear()

    def add(self, layer_idx: int, attention: torch.Tensor) -> None:
        self.records.append(LayerAttentionRecord(layer_idx=layer_idx, attention=attention.detach().cpu()))


def _gemma_eager_attention_with_weights(
    layer,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = modeling_pi05.modeling_gemma.repeat_kv(key_states, layer.num_key_value_groups)
    value_states = modeling_pi05.modeling_gemma.repeat_kv(value_states, layer.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _patched_compute_layer_complete_factory(collector: AttentionCollector):
    def _patched_compute_layer_complete(
        layer_idx,
        inputs_embeds,
        attention_mask,
        position_ids,
        adarms_cond,
        paligemma,
        gemma_expert,
    ):
        models = [paligemma.model.language_model, gemma_expert.model]
        query_states = []
        key_states = []
        value_states = []
        gates = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            hidden_states, gate = modeling_pi05.layernorm_forward(
                layer.input_layernorm, hidden_states, adarms_cond[i]
            )
            gates.append(gate)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

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
        query_states, key_states = modeling_pi05.modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
        att_output, attn_weights = _gemma_eager_attention_with_weights(
            paligemma.model.language_model.layers[layer_idx].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
        collector.add(layer_idx, attn_weights[0])

        batch_size = query_states.shape[0]
        head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
        att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

        outputs_embeds = []
        start_pos = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            end_pos = start_pos + hidden_states.shape[1]
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
            out_emb = modeling_pi05._gated_residual(hidden_states, out_emb, gates[i])
            after_first_residual = out_emb.clone()
            out_emb, gate = modeling_pi05.layernorm_forward(
                layer.post_attention_layernorm, out_emb, adarms_cond[i]
            )
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)
            out_emb = modeling_pi05._gated_residual(after_first_residual, out_emb, gate)
            outputs_embeds.append(out_emb)
            start_pos = end_pos
        return outputs_embeds

    return _patched_compute_layer_complete


@contextmanager
def _capture_pi05_attention(collector: AttentionCollector):
    original = modeling_pi05.compute_layer_complete
    modeling_pi05.compute_layer_complete = _patched_compute_layer_complete_factory(collector)
    try:
        yield
    finally:
        modeling_pi05.compute_layer_complete = original


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Checkpoint pretrained_model directory.")
    parser.add_argument(
        "--train-config-path",
        default=None,
        help="Optional train_config.json path. Defaults to <policy-path>/train_config.json.",
    )
    parser.add_argument("--dataset-root", default=None, help="Optional override for dataset root.")
    parser.add_argument("--dataset-roots-file", default=None, help="Optional override for dataset roots file.")
    parser.add_argument("--dataset-info-path", default=None, help="Optional override for dataset info.json template.")
    parser.add_argument(
        "--dataset-sidecar-root",
        default=None,
        help="Optional override for dataset sidecar root.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode indices to visualize. Example: 0,1,2",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional comma-separated dataset indices. If set, `--max-samples/--stride` are ignored.",
    )
    parser.add_argument("--max-samples", type=int, default=8, help="Number of dataset items to export.")
    parser.add_argument("--stride", type=int, default=500, help="Stride when scanning the dataset.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated attention layers to average. Defaults to all captured layers.",
    )
    parser.add_argument(
        "--query-reduction",
        default="mean",
        choices=["mean", "last"],
        help="How to reduce suffix/action query tokens before projecting to image patches.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video decoder backend. Use pyav if torchcodec/ffmpeg is unavailable.",
    )
    parser.add_argument("--device", default=None, help="cuda/cpu. Auto-detected if omitted.")
    parser.add_argument("--output-dir", default=None, help="Directory where overlays and json files are saved.")
    parser.add_argument(
        "--render-videos",
        action="store_true",
        help="Also convert exported overlay PNGs into per-camera MP4 files and a side-by-side montage.",
    )
    parser.add_argument(
        "--render-video-fps",
        type=float,
        default=None,
        help="Optional FPS override for rendered MP4 files. Defaults to the dataset FPS.",
    )
    return parser


def _to_python_scalar(value: Any) -> int | float | str:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return str(tuple(value.shape))
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _convert_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().to(torch.float32)
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    image = image.clamp(0.0, 1.0).numpy()
    return np.round(image * 255.0).astype(np.uint8)


def _infer_grid_shape(num_tokens: int) -> tuple[int, int]:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(
            f"Image token count {num_tokens} is not a perfect square. "
            "Please update the script if your vision backbone uses a non-square grid."
        )
    return side, side


def _heatmap_from_attention(
    attention_records: list[LayerAttentionRecord],
    *,
    layer_indices: list[int],
    query_slice: slice,
    key_slice: slice,
    query_reduction: str,
) -> np.ndarray:
    selected = [record.attention for record in attention_records if record.layer_idx in layer_indices]
    if not selected:
        raise ValueError(f"No captured attention layers matched requested indices: {layer_indices}")

    heatmaps = []
    for attention in selected:
        # attention: [heads, query_tokens, key_tokens]
        suffix_to_image = attention[:, query_slice, key_slice]
        if query_reduction == "last":
            reduced = suffix_to_image[:, -1:, :]
        else:
            reduced = suffix_to_image
        token_scores = reduced.to(dtype=torch.float32).mean(dim=(0, 1))
        heatmaps.append(token_scores.numpy())
    stacked = np.stack(heatmaps, axis=0)
    return stacked.mean(axis=0)


def _overlay_heatmap(image_uint8: np.ndarray, token_heatmap: np.ndarray, alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w = _infer_grid_shape(int(token_heatmap.shape[0]))
    heat = token_heatmap.reshape(grid_h, grid_w)
    heat = heat - heat.min()
    denom = float(heat.max())
    if denom > 0:
        heat = heat / denom
    heat_resized = cv2.resize(heat.astype(np.float32), (image_uint8.shape[1], image_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.round(heat_resized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.clip(image_uint8.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha, 0, 255)
    return overlay.astype(np.uint8), np.round(heat_resized * 255.0).astype(np.uint8)


def _build_sample_indices(dataset_len: int, explicit_indices: list[int] | None, max_samples: int, stride: int) -> list[int]:
    if explicit_indices is not None:
        return [idx for idx in explicit_indices if 0 <= idx < dataset_len]
    indices = list(range(0, dataset_len, max(1, stride)))
    return indices[: max(1, max_samples)]


def _get_sample_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ("episode_index", "frame_index", "timestamp", "task", "dataset_index"):
        if key in item:
            metadata[key] = _to_python_scalar(item[key])
    return metadata


def _resolve_original_image_key(processed_key: str, rename_map: dict[str, str]) -> str:
    reverse_map = {dst: src for src, dst in rename_map.items()}
    return reverse_map.get(processed_key, processed_key)


def _sanitize_video_stem(image_key: str) -> str:
    return image_key.replace("/", "__")


def _stack_frames_horizontally(frames_bgr: list[np.ndarray]) -> np.ndarray:
    if not frames_bgr:
        raise ValueError("Expected at least one frame to stack into a montage video.")
    if len(frames_bgr) == 1:
        return frames_bgr[0]
    return np.concatenate(frames_bgr, axis=1)


def _write_video_from_exported_frames(
    *,
    output_dir: Path,
    sample_indices: list[int],
    rgb_keys: list[str],
    fps: float,
) -> list[Path]:
    if not sample_indices or not rgb_keys:
        return []

    first_frame_path = output_dir / f"sample_{sample_indices[0]:06d}" / f"{_sanitize_video_stem(rgb_keys[0])}.overlay.png"
    first_frame = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise FileNotFoundError(f"Could not read first overlay frame at {first_frame_path}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    written_paths: list[Path] = []
    writers: dict[str, cv2.VideoWriter] = {}
    montage_writer: cv2.VideoWriter | None = None

    try:
        for image_key in rgb_keys:
            video_path = output_dir / f"{_sanitize_video_stem(image_key)}.overlay.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {video_path}")
            writers[image_key] = writer
            written_paths.append(video_path)

        montage_path = output_dir / "attention_overlay_montage.mp4"
        montage_writer = cv2.VideoWriter(str(montage_path), fourcc, fps, (width * len(rgb_keys), height))
        if not montage_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {montage_path}")
        written_paths.append(montage_path)

        for dataset_idx in sample_indices:
            frame_row: list[np.ndarray] = []
            for image_key in rgb_keys:
                overlay_path = output_dir / f"sample_{dataset_idx:06d}" / f"{_sanitize_video_stem(image_key)}.overlay.png"
                frame_bgr = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise FileNotFoundError(f"Could not read overlay frame at {overlay_path}")
                if frame_bgr.shape[:2] != (height, width):
                    frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
                writers[image_key].write(frame_bgr)
                frame_row.append(frame_bgr)

            montage_writer.write(_stack_frames_horizontally(frame_row))
    finally:
        for writer in writers.values():
            writer.release()
        if montage_writer is not None:
            montage_writer.release()

    return written_paths


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    policy_path = Path(args.policy_path).resolve()
    train_config_path = (
        Path(args.train_config_path).resolve()
        if args.train_config_path is not None
        else policy_path / "train_config.json"
    )
    explicit_indices = _parse_int_list(args.indices)
    requested_layers = _parse_int_list(args.layers)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = TrainPipelineConfig.from_pretrained(train_config_path)
    if args.dataset_root is not None:
        train_cfg.dataset.root = args.dataset_root
        train_cfg.dataset.roots_file = None
    if args.dataset_roots_file is not None:
        train_cfg.dataset.roots_file = args.dataset_roots_file
        train_cfg.dataset.root = None
    if args.dataset_info_path is not None:
        train_cfg.dataset.info_path = args.dataset_info_path
    if args.dataset_sidecar_root is not None:
        train_cfg.dataset.sidecar_root = args.dataset_sidecar_root
    train_cfg.dataset.video_backend = args.video_backend
    train_cfg.num_workers = 0
    train_cfg.policy.device = device
    train_cfg.policy.use_amp = False
    train_cfg.dataset.episodes = _parse_int_list(args.episodes)

    dataset = make_dataset(train_cfg)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_cfg.use_amp = False
    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=train_cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": train_cfg.rename_map},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
        },
    )
    policy.eval()

    output_dir = Path(args.output_dir) if args.output_dir else policy_path / "attention_exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = _build_sample_indices(len(dataset), explicit_indices, args.max_samples, args.stride)
    collector = AttentionCollector()
    exported_rgb_keys: list[str] | None = None

    with torch.inference_mode():
        for dataset_idx in sample_indices:
            item = dataset[dataset_idx]
            raw_batch = {
                key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                for key, value in item.items()
            }
            batch = preprocessor(raw_batch)

            rgb_keys = list(policy._resolve_rgb_image_keys(batch))
            images, img_masks = policy._preprocess_images(batch)
            tactile_tokens, tactile_token_masks, tactile_role_metadata = policy._prepare_tactile(batch)
            tokens, masks = policy.get_language_inputs(batch)
            subtask_tokens, subtask_masks = (
                policy.get_subtask_inputs(batch) if policy.config.enable_subtask_branch else (None, None)
            )
            actions = policy.prepare_action(batch)
            action_mask = policy.prepare_action_mask(batch)
            if action_mask is None:
                raise ValueError(
                    f"Dataset item {dataset_idx} does not contain an action mask required for pi05_full training forward."
                )

            collector.clear()
            with _capture_pi05_attention(collector):
                zero_noise = torch.zeros_like(actions)
                fixed_time = torch.full((actions.shape[0],), 0.5, dtype=torch.float32, device=actions.device)
                _ = policy.model.forward(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    actions,
                    action_mask=action_mask,
                    noise=zero_noise,
                    time=fixed_time,
                    subtask_tokens=subtask_tokens if policy.config.enable_subtask_branch else None,
                    subtask_masks=subtask_masks if policy.config.enable_subtask_branch else None,
                    tactile_tokens=tactile_tokens,
                    tactile_token_masks=tactile_token_masks,
                    tactile_role_metadata=tactile_role_metadata,
                )

            if not collector.records:
                raise RuntimeError(f"No attention records were captured for dataset index {dataset_idx}.")

            layer_indices = requested_layers or sorted({record.layer_idx for record in collector.records})
            if exported_rgb_keys is None:
                exported_rgb_keys = list(rgb_keys)
            elif exported_rgb_keys != list(rgb_keys):
                raise ValueError(
                    "Exported samples produced different RGB key sets, which would make video rendering ambiguous: "
                    f"{exported_rgb_keys} vs {list(rgb_keys)}"
                )

            image_token_counts: list[int] = []
            for image in images[: len(rgb_keys)]:
                img_emb = policy.model.paligemma_with_expert.embed_image(image)
                image_token_counts.append(int(img_emb.shape[1]))

            prefix_len = sum(image_token_counts) + int(masks.shape[1])
            if policy.config.enable_subtask_branch and subtask_tokens is not None and subtask_masks is not None:
                prefix_len += int(subtask_tokens.shape[1])
            suffix_len = int(policy.config.chunk_size)

            sample_dir = output_dir / f"sample_{dataset_idx:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            segment_summaries = []
            cursor = 0
            for image_key, image_token_count in zip(rgb_keys, image_token_counts, strict=True):
                key_slice = slice(cursor, cursor + image_token_count)
                cursor += image_token_count
                query_slice = slice(prefix_len, prefix_len + suffix_len)
                token_heatmap = _heatmap_from_attention(
                    collector.records,
                    layer_indices=layer_indices,
                    query_slice=query_slice,
                    key_slice=key_slice,
                    query_reduction=args.query_reduction,
                )

                original_image_key = _resolve_original_image_key(image_key, train_cfg.rename_map)
                if original_image_key not in raw_batch:
                    raise KeyError(
                        f"Could not map processed image key {image_key!r} back to original key. "
                        f"Tried {original_image_key!r}. Available raw keys: {sorted(raw_batch.keys())}"
                    )
                original_image = _convert_image_to_uint8(raw_batch[original_image_key])
                overlay_rgb, heat_gray = _overlay_heatmap(original_image, token_heatmap)

                base_name = image_key.replace("/", "__")
                original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                heat_color_bgr = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)

                cv2.imwrite(str(sample_dir / f"{base_name}.original.png"), original_bgr)
                cv2.imwrite(str(sample_dir / f"{base_name}.overlay.png"), overlay_bgr)
                cv2.imwrite(str(sample_dir / f"{base_name}.heatmap.png"), heat_color_bgr)

                segment_summaries.append(
                    {
                        "image_key": image_key,
                        "image_token_count": image_token_count,
                        "key_start": key_slice.start,
                        "key_end": key_slice.stop,
                        "heatmap_min": float(token_heatmap.min()),
                        "heatmap_max": float(token_heatmap.max()),
                    }
                )

            metadata = {
                "dataset_index": dataset_idx,
                "policy_path": str(policy_path),
                "train_config_path": str(train_config_path),
                "layers": layer_indices,
                "query_reduction": args.query_reduction,
                "sample_metadata": _get_sample_metadata(item),
                "prefix_length": prefix_len,
                "suffix_length": suffix_len,
                "rgb_image_keys": rgb_keys,
                "segments": segment_summaries,
            }
            with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"[exported] dataset_index={dataset_idx} -> {sample_dir}")

    if args.render_videos:
        if exported_rgb_keys is None:
            raise RuntimeError("No samples were exported, so no videos can be rendered.")
        fps = float(args.render_video_fps) if args.render_video_fps is not None else float(dataset.fps)
        video_paths = _write_video_from_exported_frames(
            output_dir=output_dir,
            sample_indices=sample_indices,
            rgb_keys=exported_rgb_keys,
            fps=fps,
        )
        for video_path in video_paths:
            print(f"[video] {video_path}")


if __name__ == "__main__":
    main()
