#!/usr/bin/env python3

"""Offline chunk comparison for PI05-Full state-as-action checkpoints.

This script evaluates a single episode, selects a frame with the largest
future xyz span inside one predicted chunk, and saves:
- full 20D gt/pred chunk curves
- xyz-only curves
- a JSON summary with the selected frame and errors

Example usage:

python /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/offline_eval_pi05full_state_as_action_chunk.py \
  --checkpoint-dir /path/to/checkpoints/030000 \
  --dataset-root /path/to/dataset \
  --sidecar-path /path/to/framewise_guidance.parquet \
  --output-dir /path/to/save_dir

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05_full.modeling_pi05 import PI05FullPolicy


ACTION_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "gripper_left",
    "right_x",
    "right_y",
    "right_z",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "gripper_right",
]
LEFT_XYZ = (0, 1, 2)
RIGHT_XYZ = (10, 11, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=None, help="Manually pick a frame instead of auto-selecting.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-backend", type=str, default="pyav")
    return parser.parse_args()


def _to_scalar(value):
    if torch.is_tensor(value):
        return value.item()
    return value


def _load_policy_bundle(checkpoint_dir: Path, device: str):
    pretrained_dir = checkpoint_dir / "pretrained_model"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cfg = PreTrainedConfig.from_pretrained(pretrained_dir, local_files_only=True)
    cfg.device = device
    if cfg.type == "pi05":
        policy_cls = PI05Policy
    elif cfg.type == "pi05_full":
        policy_cls = PI05FullPolicy
    else:
        raise ValueError(f"Unsupported policy type: {cfg.type}")

    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=pretrained_dir,
        config=cfg,
        local_files_only=True,
        strict=True,
    )
    policy.to(torch.device(device))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return pretrained_dir, cfg, policy, preprocessor, postprocessor


def _get_xyz_dims(action_mask: np.ndarray) -> list[int]:
    dims: list[int] = []
    if action_mask[:, LEFT_XYZ].mean() > 0.5:
        dims.extend(LEFT_XYZ)
    if action_mask[:, RIGHT_XYZ].mean() > 0.5:
        dims.extend(RIGHT_XYZ)
    return dims


def _select_frame(actions: np.ndarray, xyz_dims: list[int], chunk_horizon: int, frame_index: int | None) -> tuple[int, float]:
    if frame_index is not None:
        return frame_index, float("nan")

    valid_start_max = actions.shape[0] - chunk_horizon
    if valid_start_max <= 0:
        raise RuntimeError("Episode is shorter than the model chunk horizon.")

    scores = np.empty(valid_start_max, dtype=np.float32)
    for start in range(valid_start_max):
        xyz_chunk = actions[start : start + chunk_horizon][:, xyz_dims]
        scores[start] = float(np.ptp(xyz_chunk, axis=0).sum())

    best_start = int(np.argmax(scores))
    return best_start, float(scores[best_start])


def _plot_full_chunk(output_path: Path, action_names: list[str], gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, title: str) -> None:
    horizon, action_dim = gt.shape
    fig, axes = plt.subplots(5, 4, figsize=(18, 16), sharex=True, dpi=180)
    axes = axes.flatten()
    xs = np.arange(horizon)

    for dim in range(action_dim):
        ax = axes[dim]
        if mask[:, dim].mean() <= 0.5:
            ax.text(0.5, 0.5, "invalid dim", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_title(action_names[dim])
            ax.grid(True, alpha=0.2)
            continue
        ax.plot(xs, gt[:, dim], label="gt", linewidth=2.0, color="C0")
        ax.plot(xs, pred[:, dim], label="pred", linewidth=2.0, linestyle="--", color="C1")
        ax.set_title(action_names[dim])
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend()

    for ax in axes[-4:]:
        ax.set_xlabel("chunk step")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_xyz_only(output_path: Path, xyz_names: list[str], xyz_dims: list[int], gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    horizon = gt.shape[0]
    cols = 2
    rows = int(math.ceil(len(xyz_dims) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), sharex=True, dpi=180)
    axes = np.array(axes).reshape(-1)
    xs = np.arange(horizon)

    for idx, (dim, name) in enumerate(zip(xyz_dims, xyz_names)):
        ax = axes[idx]
        ax.plot(xs, gt[:, dim], label="gt", linewidth=2.2, color="C0")
        ax.plot(xs, pred[:, dim], label="pred", linewidth=2.2, linestyle="--", color="C1")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(xyz_dims) :]:
        ax.axis("off")

    axes[0].legend()
    for ax in axes[max(0, len(axes) - cols) :]:
        if ax.has_data():
            ax.set_xlabel("chunk step")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_dir, cfg, policy, preprocessor, postprocessor = _load_policy_bundle(args.checkpoint_dir, args.device)
    dataset = LeRobotDataset(
        repo_id=f"offline_eval_{args.output_dir.name}",
        root=str(args.dataset_root),
        sidecar_path=str(args.sidecar_path),
        episodes=[args.episode_index],
        video_backend=args.video_backend,
        download_videos=False,
    )

    sample0 = dataset[0]
    processed0 = preprocessor(sample0)
    with torch.inference_mode():
        pred0 = policy.predict_action_chunk(processed0)
        pred0_abs = postprocessor(pred0)
    chunk_horizon = int(pred0_abs.shape[1])
    action_dim = int(pred0_abs.shape[2])

    length = len(dataset)
    actions = np.empty((length, action_dim), dtype=np.float32)
    masks = np.empty((length, action_dim), dtype=np.float32)
    frame_indices = np.empty(length, dtype=np.int64)
    timestamps = np.empty(length, dtype=np.float32)

    for i in range(length):
        item = dataset[i]
        actions[i] = item["action"].detach().cpu().numpy().astype(np.float32)[:action_dim]
        action_mask = item.get("action_valid_mask")
        if action_mask is None:
            masks[i] = 1.0
        else:
            masks[i] = torch.as_tensor(action_mask).detach().cpu().numpy().astype(np.float32)[:action_dim]
        frame_indices[i] = int(_to_scalar(item["frame_index"]))
        timestamps[i] = float(_to_scalar(item["timestamp"]))

    xyz_dims = _get_xyz_dims(masks)
    if not xyz_dims:
        raise RuntimeError("No valid xyz dimensions found in this episode.")
    xyz_names = [ACTION_NAMES[i] for i in xyz_dims]

    selected_start, selection_score = _select_frame(actions, xyz_dims, chunk_horizon, args.frame_index)
    gt_chunk = actions[selected_start : selected_start + chunk_horizon]
    mask_chunk = masks[selected_start : selected_start + chunk_horizon]

    raw_item = dataset[selected_start]
    processed = preprocessor(raw_item)
    with torch.inference_mode():
        pred_chunk = policy.predict_action_chunk(processed)
        pred_chunk_abs = postprocessor(pred_chunk)[0].detach().cpu().numpy().astype(np.float32)

    valid_mask_bool = mask_chunk > 0.5
    abs_diff = np.abs(pred_chunk_abs - gt_chunk)
    xyz_abs_diff = np.abs(pred_chunk_abs[:, xyz_dims] - gt_chunk[:, xyz_dims])

    masked_abs_diff = abs_diff[valid_mask_bool]
    masked_xyz_abs_diff = xyz_abs_diff[valid_mask_bool[:, xyz_dims]]
    mean_abs_error = float(masked_abs_diff.mean()) if masked_abs_diff.size else float("nan")
    mean_abs_error_xyz = float(masked_xyz_abs_diff.mean()) if masked_xyz_abs_diff.size else float("nan")

    title = (
        f"episode {args.episode_index} frame {frame_indices[selected_start]} (dataset idx {selected_start})\n"
        f"xyz-span selection score={selection_score:.4f}, horizon={chunk_horizon}"
    )

    full_plot_path = args.output_dir / "episode0_large_xyz_frame_gt_vs_pred_full_chunk.png"
    xyz_plot_path = args.output_dir / "episode0_large_xyz_frame_gt_vs_pred_xyz_only.png"
    summary_path = args.output_dir / "episode0_large_xyz_frame_summary.json"

    _plot_full_chunk(full_plot_path, ACTION_NAMES[:action_dim], gt_chunk, pred_chunk_abs, mask_chunk, title)
    _plot_xyz_only(xyz_plot_path, xyz_names, xyz_dims, gt_chunk, pred_chunk_abs, title)

    summary = {
        "policy_path": str(pretrained_dir),
        "dataset_root": str(args.dataset_root),
        "sidecar_path": str(args.sidecar_path),
        "episode_index": args.episode_index,
        "frame_index": int(frame_indices[selected_start]),
        "dataset_index": selected_start,
        "timestamp": float(timestamps[selected_start]),
        "chunk_horizon": chunk_horizon,
        "action_dim": action_dim,
        "xyz_dims": xyz_dims,
        "xyz_names": xyz_names,
        "selection_rule": "max future chunk xyz span sum over valid xyz dims",
        "selection_score_xyz_span_sum": float(selection_score),
        "valid_left_xyz": bool(masks[:, LEFT_XYZ].mean() > 0.5),
        "valid_right_xyz": bool(masks[:, RIGHT_XYZ].mean() > 0.5),
        "mean_abs_error_over_chunk_valid_dims": mean_abs_error,
        "mean_abs_error_xyz_only_valid_dims": mean_abs_error_xyz,
        "gt_xyz_start": gt_chunk[0, xyz_dims].tolist(),
        "gt_xyz_end": gt_chunk[-1, xyz_dims].tolist(),
        "pred_xyz_start": pred_chunk_abs[0, xyz_dims].tolist(),
        "pred_xyz_end": pred_chunk_abs[-1, xyz_dims].tolist(),
        "full_plot_path": str(full_plot_path),
        "xyz_plot_path": str(xyz_plot_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {full_plot_path}")
    print(f"saved: {xyz_plot_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
