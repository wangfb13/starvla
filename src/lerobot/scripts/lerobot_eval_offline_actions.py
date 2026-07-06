#!/usr/bin/env python

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
"""Offline action-alignment evaluation for behavior cloning policies.

This script compares predicted actions against dataset actions using logged
observations only. For PI0.5 / PI05 it reports two complementary views:

1. Chunk alignment: compare `predict_action_chunk(observation_t)` to the
   dataset action chunk starting at frame `t`.
2. Rollout-like step alignment: feed observations sequentially and compare the
   action returned by `select_action()` (which uses the internal action queue)
   against the first action in the dataset chunk for each frame.

The metrics are reported in two spaces:

1. `model_space`: normalized / processor-space actions actually used by the
   policy.
2. `absolute_canonical_space`: postprocessed actions after unnormalization and
   relative-to-absolute conversion. For PI05 this is the 20D canonical action
   space, not the raw 111D dataset action schema.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ACTION_MASK


def _parse_episodes(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _normalize_mask(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(like, dtype=torch.float32)

    mask = mask.to(device=like.device, dtype=torch.float32)
    while mask.ndim > like.ndim:
        mask = mask[:, 0, ...]
    if mask.shape[-1] != like.shape[-1]:
        mask = mask[..., : like.shape[-1]]
    if mask.ndim == like.ndim - 1:
        mask = mask.unsqueeze(-2)
    return mask.expand_as(like)


def _get_action_mask(policy, batch: dict[str, Any], like: torch.Tensor) -> torch.Tensor | None:
    """Best-effort action-mask extraction across policy implementations."""
    if hasattr(policy, "prepare_action_mask"):
        return policy.prepare_action_mask(batch)

    if hasattr(policy, "_prepare_action_mask"):
        return policy._prepare_action_mask(batch, like.device, like.dtype)

    action_mask_key = getattr(getattr(policy, "config", None), "action_mask_observation_key", OBS_ACTION_MASK)
    mask = batch.get(action_mask_key, batch.get(OBS_ACTION_MASK))
    if mask is None:
        mask = batch.get("action_valid_mask")
    return mask


@dataclass
class ErrorStats:
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    total_count: float = 0.0
    cosine_sum: float = 0.0
    cosine_count: int = 0
    per_dim_abs_sum: torch.Tensor | None = None
    per_dim_count: torch.Tensor | None = None

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        pred = pred.detach().to(dtype=torch.float64, device="cpu")
        target = target.detach().to(dtype=torch.float64, device="cpu")
        mask = mask.detach().to(dtype=torch.float64, device="cpu")

        abs_err = (pred - target).abs() * mask
        sq_err = (pred - target).pow(2) * mask

        self.sum_abs += float(abs_err.sum().item())
        self.sum_sq += float(sq_err.sum().item())
        self.total_count += float(mask.sum().item())

        if self.per_dim_abs_sum is None:
            self.per_dim_abs_sum = abs_err.sum(dim=tuple(range(abs_err.ndim - 1)))
            self.per_dim_count = mask.sum(dim=tuple(range(mask.ndim - 1)))
        else:
            self.per_dim_abs_sum += abs_err.sum(dim=tuple(range(abs_err.ndim - 1)))
            self.per_dim_count += mask.sum(dim=tuple(range(mask.ndim - 1)))

        flat_pred = pred.reshape(-1)
        flat_target = target.reshape(-1)
        flat_mask = mask.reshape(-1) > 0
        if flat_mask.any():
            masked_pred = flat_pred[flat_mask]
            masked_target = flat_target[flat_mask]
            denom = masked_pred.norm().item() * masked_target.norm().item()
            if denom > 0:
                cos = F.cosine_similarity(
                    masked_pred.unsqueeze(0).to(dtype=torch.float32),
                    masked_target.unsqueeze(0).to(dtype=torch.float32),
                    dim=1,
                )
                self.cosine_sum += float(cos.item())
                self.cosine_count += 1

    def to_dict(self, action_names: list[str] | None = None) -> dict[str, Any]:
        mae = self.sum_abs / self.total_count if self.total_count > 0 else math.nan
        rmse = math.sqrt(self.sum_sq / self.total_count) if self.total_count > 0 else math.nan
        cosine = self.cosine_sum / self.cosine_count if self.cosine_count > 0 else math.nan

        per_dim_mae: list[float] = []
        if self.per_dim_abs_sum is not None and self.per_dim_count is not None:
            for abs_sum, count in zip(self.per_dim_abs_sum.tolist(), self.per_dim_count.tolist(), strict=True):
                per_dim_mae.append(abs_sum / count if count > 0 else math.nan)

        result = {
            "mae": mae,
            "rmse": rmse,
            "cosine": cosine,
            "valid_values": self.total_count,
            "per_dim_mae": per_dim_mae,
        }
        if action_names is not None:
            result["per_dim_mae_by_name"] = {
                name: per_dim_mae[idx] for idx, name in enumerate(action_names[: len(per_dim_mae)])
            }
        return result


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Checkpoint pretrained_model directory.")
    parser.add_argument(
        "--train-config-path",
        default=None,
        help="Optional train_config.json path. Defaults to <policy-path>/train_config.json.",
    )
    parser.add_argument("--dataset-root", default=None, help="Optional override for dataset root.")
    parser.add_argument(
        "--dataset-info-path",
        default=None,
        help="Optional override for dataset info template, e.g. meta/info_fisheye.json.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode indices to evaluate. Example: 0,1,2",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video decoder backend. Use pyav if torchcodec/ffmpeg is unavailable.",
    )
    parser.add_argument("--device", default=None, help="cuda/cpu. Auto-detected if omitted.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap.")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth frame.")
    parser.add_argument("--output-json", default=None, help="Where to save the JSON metrics.")
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    policy_path = Path(args.policy_path).resolve()
    train_config_path = (
        Path(args.train_config_path).resolve()
        if args.train_config_path is not None
        else policy_path / "train_config.json"
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    episodes = _parse_episodes(args.episodes)

    train_cfg = TrainPipelineConfig.from_pretrained(train_config_path)
    if args.dataset_root is not None:
        train_cfg.dataset.root = args.dataset_root
    if args.dataset_info_path is not None:
        train_cfg.dataset.info_path = args.dataset_info_path
    if episodes is not None:
        train_cfg.dataset.episodes = episodes
    train_cfg.dataset.video_backend = args.video_backend
    train_cfg.num_workers = 0
    train_cfg.policy.device = device
    train_cfg.policy.use_amp = False

    dataset = make_dataset(train_cfg)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_cfg.use_amp = False

    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=train_cfg.rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
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

    action_names = getattr(policy_cfg, "action_feature_names", None)
    model_chunk_stats = ErrorStats()
    absolute_chunk_stats = ErrorStats()
    model_step_stats = ErrorStats()
    absolute_step_stats = ErrorStats()

    policy.eval()
    policy.reset()

    current_episode: int | None = None
    processed_frames = 0
    visited_frames = 0
    start = time.perf_counter()

    with torch.inference_mode():
        for idx in range(len(dataset)):
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break
            if idx % args.stride != 0:
                continue

            raw_item = dataset[idx]
            visited_frames += 1

            episode_index = _to_int(raw_item["episode_index"])
            if current_episode != episode_index:
                policy.reset()
                current_episode = episode_index

            raw_batch = {
                key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                for key, value in raw_item.items()
            }

            processed_batch = preprocessor(raw_batch)
            gt_chunk_model = processed_batch[ACTION]
            pred_chunk_model = policy.predict_action_chunk(processed_batch)

            chunk_mask = _normalize_mask(_get_action_mask(policy, processed_batch, gt_chunk_model), gt_chunk_model)
            model_chunk_stats.update(pred_chunk_model, gt_chunk_model, chunk_mask)

            gt_chunk_absolute = postprocessor(gt_chunk_model.clone())
            pred_chunk_absolute = postprocessor(pred_chunk_model.clone())
            absolute_chunk_stats.update(
                pred_chunk_absolute,
                gt_chunk_absolute,
                chunk_mask.to(device="cpu"),
            )

            pred_step_model = policy.select_action(processed_batch)
            gt_step_model = gt_chunk_model[:, 0, :]
            step_mask = _normalize_mask(_get_action_mask(policy, processed_batch, gt_step_model), gt_step_model)
            model_step_stats.update(pred_step_model, gt_step_model, step_mask)

            gt_step_absolute = gt_chunk_absolute[:, 0, :]
            pred_step_absolute = postprocessor(pred_step_model.clone())
            absolute_step_stats.update(
                pred_step_absolute,
                gt_step_absolute,
                step_mask.to(device="cpu"),
            )

            processed_frames += 1
            if processed_frames % 10 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"processed_frames={processed_frames} visited_frames={visited_frames} "
                    f"chunk_mae={model_chunk_stats.sum_abs / max(model_chunk_stats.total_count, 1.0):.6f} "
                    f"step_mae={model_step_stats.sum_abs / max(model_step_stats.total_count, 1.0):.6f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    elapsed = time.perf_counter() - start
    result = {
        "policy_path": str(policy_path),
        "train_config_path": str(train_config_path),
        "dataset_root": str(train_cfg.dataset.root),
        "dataset_info_path": train_cfg.dataset.info_path,
        "video_backend": args.video_backend,
        "device": device,
        "episodes": episodes,
        "max_frames": args.max_frames,
        "stride": args.stride,
        "processed_frames": processed_frames,
        "visited_frames": visited_frames,
        "elapsed_s": elapsed,
        "notes": [
            "chunk metrics compare predict_action_chunk() against the dataset chunk starting at each frame",
            "rollout_like_step metrics compare select_action() against the first action in each dataset chunk",
            "absolute_canonical_space is postprocessed 20D canonical action space, not the raw dataset action schema",
        ],
        "metrics": {
            "chunk": {
                "model_space": model_chunk_stats.to_dict(action_names=action_names),
                "absolute_canonical_space": absolute_chunk_stats.to_dict(action_names=action_names),
            },
            "rollout_like_step": {
                "model_space": model_step_stats.to_dict(action_names=action_names),
                "absolute_canonical_space": absolute_step_stats.to_dict(action_names=action_names),
            },
        },
    }

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else policy_path / "offline_action_alignment.json"
    )
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"saved metrics to {output_json}", flush=True)


if __name__ == "__main__":
    main()
