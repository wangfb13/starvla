#!/usr/bin/env python3

"""Compare PI05/PI05-Full gripper predictions near a grasp-like interval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, action="append", required=True)
    parser.add_argument("--label", type=str, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument("--gripper-dim", type=int, default=9)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--context-radius", type=int, default=80)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="largest_increase",
        choices=("largest_increase", "largest_range"),
    )
    return parser.parse_args()


def load_policy_bundle(checkpoint_dir: Path, device: str):
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
    return policy, preprocessor, postprocessor


def load_episode_arrays(
    dataset_root: Path, sidecar_path: Path, episode_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_table = pa_dataset.dataset(str(dataset_root / "data"), format="parquet").to_table(
        columns=["frame_index", "timestamp"],
        filter=(pa_dataset.field("episode_index") == episode_index),
    )
    raw_rows = raw_table.to_pydict()
    frame_indices = np.asarray(raw_rows["frame_index"], dtype=np.int64)
    timestamps = np.asarray(raw_rows["timestamp"], dtype=np.float32)

    sidecar_table = pq.read_table(sidecar_path, columns=["episode_index", "frame_index", "action"])
    sidecar_rows = sidecar_table.to_pydict()
    sidecar_mask = np.asarray(sidecar_rows["episode_index"], dtype=np.int64) == episode_index
    sidecar_frame_indices = np.asarray(sidecar_rows["frame_index"], dtype=np.int64)[sidecar_mask]
    actions = np.asarray(sidecar_rows["action"], dtype=np.float32)[sidecar_mask]

    if len(frame_indices) != len(sidecar_frame_indices):
        raise RuntimeError(
            f"Raw dataset frame count ({len(frame_indices)}) does not match sidecar frame count ({len(sidecar_frame_indices)})."
        )
    if not np.array_equal(frame_indices, sidecar_frame_indices):
        raise RuntimeError("Raw dataset frame_index does not align with sidecar frame_index for the selected episode.")

    return frame_indices, timestamps, actions


def select_start(actions: np.ndarray, gripper_dim: int, chunk_size: int, mode: str) -> tuple[int, dict[str, float]]:
    valid_count = len(actions) - chunk_size + 1
    if valid_count <= 0:
        raise RuntimeError("Episode shorter than chunk size.")
    best_start = 0
    best_score = -float("inf")
    best_payload: dict[str, float] = {}
    values = actions[:, gripper_dim]
    for start in range(valid_count):
        chunk = values[start : start + chunk_size]
        increase = float(chunk[-1] - chunk[0])
        value_range = float(chunk.max() - chunk.min())
        score = increase if mode == "largest_increase" else value_range
        if score > best_score:
            best_score = score
            best_start = start
            best_payload = {
                "score": score,
                "increase": increase,
                "range": value_range,
                "start_value": float(chunk[0]),
                "end_value": float(chunk[-1]),
                "max_value": float(chunk.max()),
                "min_value": float(chunk.min()),
            }
    return best_start, best_payload


def predict_chunk(dataset: LeRobotDataset, index: int, policy, preprocessor, postprocessor) -> np.ndarray:
    item = dataset[index]
    processed = preprocessor(item)
    with torch.inference_mode():
        pred = policy.predict_action_chunk(processed)
        pred_abs = postprocessor(pred)[0].detach().cpu().numpy().astype(np.float32)
    return pred_abs


def plot_context(
    output_path: Path,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    values: np.ndarray,
    start: int,
    chunk_size: int,
    radius: int,
    gripper_name: str,
) -> None:
    lo = max(0, start - radius)
    hi = min(len(values), start + chunk_size + radius)
    xs = frame_indices[lo:hi]
    ys = values[lo:hi]

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=180)
    ax.plot(xs, ys, color="#1f77b4", linewidth=2.2)
    ax.axvspan(frame_indices[start], frame_indices[min(start + chunk_size - 1, len(frame_indices) - 1)], color="#ffdd99", alpha=0.35)
    ax.axvline(frame_indices[start], color="#d62728", linestyle="--", linewidth=1.8, label="selected start")
    ax.set_title(f"{gripper_name} context around selected grasp-like window")
    ax.set_xlabel("frame_index")
    ax.set_ylabel(gripper_name)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_chunk_compare(
    output_path: Path,
    gt_chunk: np.ndarray,
    preds: list[np.ndarray],
    labels: list[str],
    gripper_dim: int,
    gripper_name: str,
) -> None:
    horizon = gt_chunk.shape[0]
    xs = np.arange(horizon)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=180, sharex=True)

    axes[0].plot(xs, gt_chunk[:, gripper_dim], color="black", linewidth=2.4, label="gt")
    for pred, label in zip(preds, labels, strict=True):
        axes[0].plot(xs, pred[:horizon, gripper_dim], linewidth=2.0, linestyle="--", label=label)
    axes[0].set_title(f"{gripper_name} over selected 50-step chunk")
    axes[0].set_xlabel("chunk step")
    axes[0].set_ylabel(gripper_name)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    # Right subplot zooms into residuals to make small differences easier to see.
    for pred, label in zip(preds, labels, strict=True):
        residual = pred[:horizon, gripper_dim] - gt_chunk[:, gripper_dim]
        axes[1].plot(xs, residual, linewidth=2.0, label=label)
    axes[1].axhline(0.0, color="black", linewidth=1.2, alpha=0.4)
    axes[1].set_title(f"{gripper_name} prediction residual")
    axes[1].set_xlabel("chunk step")
    axes[1].set_ylabel("pred - gt")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.checkpoint_dir) != len(args.label):
        raise ValueError("--checkpoint-dir and --label must have the same count.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices, timestamps, actions = load_episode_arrays(args.dataset_root, args.sidecar_path, args.episode_index)
    start = args.frame_start
    selection_payload = None
    if start is None:
        start, selection_payload = select_start(actions, args.gripper_dim, args.chunk_size, args.selection_mode)

    dataset = LeRobotDataset(
        repo_id=f"gripper_compare_{args.output_dir.name}",
        root=str(args.dataset_root),
        sidecar_path=str(args.sidecar_path),
        episodes=[args.episode_index],
        video_backend=args.video_backend,
        download_videos=False,
    )

    preds = []
    for checkpoint_dir in args.checkpoint_dir:
        policy, preprocessor, postprocessor = load_policy_bundle(checkpoint_dir, args.device)
        preds.append(predict_chunk(dataset, start, policy, preprocessor, postprocessor))

    gt_chunk = actions[start : start + args.chunk_size]
    gripper_name = ACTION_NAMES[args.gripper_dim]

    context_path = args.output_dir / "gripper_context_window.png"
    compare_path = args.output_dir / "gripper_chunk_compare.png"
    plot_context(
        context_path,
        frame_indices,
        timestamps,
        actions[:, args.gripper_dim],
        start,
        args.chunk_size,
        args.context_radius,
        gripper_name,
    )
    plot_chunk_compare(compare_path, gt_chunk, preds, args.label, args.gripper_dim, gripper_name)

    summary = {
        "episode_index": args.episode_index,
        "selected_start_index": int(start),
        "selected_frame_index": int(frame_indices[start]),
        "selected_timestamp": float(timestamps[start]),
        "gripper_dim": int(args.gripper_dim),
        "gripper_name": gripper_name,
        "selection_mode": args.selection_mode,
        "selection_payload": selection_payload,
        "checkpoints": [str(p) for p in args.checkpoint_dir],
        "labels": args.label,
        "context_plot": str(context_path),
        "compare_plot": str(compare_path),
        "gt_chunk_gripper": gt_chunk[:, args.gripper_dim].tolist(),
        "pred_chunks_gripper": {
            label: pred[: args.chunk_size, args.gripper_dim].tolist() for label, pred in zip(args.label, preds, strict=True)
        },
    }
    summary_path = args.output_dir / "gripper_compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
