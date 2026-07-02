#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.video_utils import decode_video_frames


TACTILE_KEYS = [
    "observation.depth_deformation.tactile_left_left",
    "observation.depth_deformation.tactile_left_right",
    "observation.depth_deformation.tactile_right_left",
    "observation.depth_deformation.tactile_right_right",
]

RGB_KEYS = [
    "observation.images.cam_left",
    "observation.images.cam_right",
]

DEFAULT_TOLERANCE_S = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one episode with tactile contact_stats, state, and action trajectories."
    )
    parser.add_argument("dataset_root", type=Path, help="LeRobot dataset root.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save outputs.")
    parser.add_argument("--video-backend", type=str, default="pyav", help="Video backend for RGB decoding.")
    parser.add_argument("--tolerance-s", type=float, default=DEFAULT_TOLERANCE_S, help="Decode timestamp tolerance.")
    parser.add_argument(
        "--tactile-contact-topk-ratio",
        type=float,
        default=0.05,
        help="Top-k ratio used by contact_stats tactile preprocessing.",
    )
    parser.add_argument(
        "--strip-samples",
        type=int,
        default=8,
        help="How many evenly spaced timestamps to decode for the frame strip.",
    )
    return parser.parse_args()


def _load_info(dataset_root: Path) -> dict:
    return json.loads((dataset_root / "meta" / "info.json").read_text())


def _resolve_video_extension(feature_key: str) -> str:
    lowered = feature_key.lower()
    if any(token in lowered for token in ("deformation", "shear", "depth")):
        return ".mov"
    return ".mp4"


def _load_episode_row(dataset_root: Path, episode_index: int) -> dict[str, object]:
    table = pa_dataset.dataset(dataset_root / "meta" / "episodes").to_table(
        filter=(pa_dataset.field("episode_index") == episode_index)
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"Episode {episode_index} not found in {dataset_root}")
    return rows[0]


def _resolve_episode_video_path(dataset_root: Path, episode_row: dict[str, object], feature_key: str) -> Path:
    chunk_index = int(episode_row[f"videos/{feature_key}/chunk_index"])
    file_index = int(episode_row[f"videos/{feature_key}/file_index"])
    ext = _resolve_video_extension(feature_key)
    return dataset_root / "videos" / feature_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}{ext}"


def _load_episode_table(dataset_root: Path, episode_index: int):
    return pa_dataset.dataset(dataset_root / "data").to_table(
        columns=["frame_index", "timestamp", "action", "observation.state"],
        filter=(pa_dataset.field("episode_index") == episode_index),
    )


def _sort_episode_arrays(table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = table.to_pydict()
    frame_index = np.asarray(rows["frame_index"], dtype=np.int64)
    timestamps = np.asarray(rows["timestamp"], dtype=np.float64)
    action = np.asarray(rows["action"], dtype=np.float32)
    state = np.asarray(rows["observation.state"], dtype=np.float32)
    order = np.argsort(frame_index)
    return frame_index[order], timestamps[order], action[order], state[order]


def _compute_contact_stats(frames: torch.Tensor, topk_ratio: float) -> np.ndarray:
    tactile = torch.as_tensor(frames, dtype=torch.float32)
    if tactile.ndim != 4:
        raise ValueError(f"Expected tactile frames [T,C,H,W], got {tuple(tactile.shape)}")
    if tactile.shape[1] > 1:
        tactile = tactile.mean(dim=1, keepdim=True)
    flat = tactile.flatten(start_dim=2).squeeze(1)
    sorted_values = torch.sort(flat, dim=1, descending=True).values
    k = max(1, math.ceil(sorted_values.shape[1] * topk_ratio))
    scores = sorted_values[:, :k].mean(dim=1)
    return scores.detach().cpu().numpy().astype(np.float32)


def _normalize_series(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin + 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _summarize_series(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _write_framewise_csv(
    output_path: Path,
    frame_index: np.ndarray,
    timestamps: np.ndarray,
    action: np.ndarray,
    state: np.ndarray,
    action_names: list[str],
    state_names: list[str],
    tactile_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "frame_index": frame_index,
            "timestamp_s": timestamps,
        }
    )
    for sensor_key, values in tactile_scores.items():
        df[f"contact_{sensor_key.split('.')[-1]}"] = values

    contact_left = 0.5 * (
        tactile_scores["observation.depth_deformation.tactile_left_left"]
        + tactile_scores["observation.depth_deformation.tactile_left_right"]
    )
    contact_right = 0.5 * (
        tactile_scores["observation.depth_deformation.tactile_right_left"]
        + tactile_scores["observation.depth_deformation.tactile_right_right"]
    )
    df["contact_left_mean"] = contact_left
    df["contact_right_mean"] = contact_right
    df["contact_bimanual_mean"] = 0.5 * (contact_left + contact_right)

    selected_action_names = [
        "left_x",
        "left_y",
        "left_z",
        "right_x",
        "right_y",
        "right_z",
        "gripper_left",
        "gripper_right",
    ]
    selected_state_names = selected_action_names + [
        "left_Acc_X",
        "left_Acc_Y",
        "left_Acc_Z",
        "right_Acc_X",
        "right_Acc_Y",
        "right_Acc_Z",
    ]

    for name in selected_action_names:
        idx = action_names.index(name)
        df[f"action_{name}"] = action[:, idx]
    for name in selected_state_names:
        idx = state_names.index(name)
        df[f"state_{name}"] = state[:, idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _plot_contact_and_gripper(
    output_path: Path,
    timestamps: np.ndarray,
    tactile_scores: dict[str, np.ndarray],
    action: np.ndarray,
    state: np.ndarray,
    action_names: list[str],
    state_names: list[str],
    topk_ratio: float,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), dpi=180, sharex=True)

    sensor_colors = {
        "observation.depth_deformation.tactile_left_left": "#1f77b4",
        "observation.depth_deformation.tactile_left_right": "#17becf",
        "observation.depth_deformation.tactile_right_left": "#ff7f0e",
        "observation.depth_deformation.tactile_right_right": "#d62728",
    }

    for sensor_key, values in tactile_scores.items():
        axes[0].plot(
            timestamps,
            values,
            label=sensor_key.split(".")[-1],
            linewidth=1.2,
            color=sensor_colors[sensor_key],
        )
    axes[0].set_title(f"Tactile contact_stats trajectory (top-k ratio={topk_ratio:.3f})")
    axes[0].set_ylabel("contact score")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    state_gl = state[:, state_names.index("gripper_left")]
    state_gr = state[:, state_names.index("gripper_right")]
    action_gl = action[:, action_names.index("gripper_left")]
    action_gr = action[:, action_names.index("gripper_right")]

    axes[1].plot(timestamps, state_gl, label="state gripper_left", color="#2ca02c", linewidth=1.4)
    axes[1].plot(timestamps, action_gl, label="action gripper_left", color="#2ca02c", linewidth=1.0, alpha=0.7)
    axes[1].plot(timestamps, state_gr, label="state gripper_right", color="#9467bd", linewidth=1.4)
    axes[1].plot(timestamps, action_gr, label="action gripper_right", color="#9467bd", linewidth=1.0, alpha=0.7)
    axes[1].set_title("Gripper state/action")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    contact_left = 0.5 * (
        tactile_scores["observation.depth_deformation.tactile_left_left"]
        + tactile_scores["observation.depth_deformation.tactile_left_right"]
    )
    contact_right = 0.5 * (
        tactile_scores["observation.depth_deformation.tactile_right_left"]
        + tactile_scores["observation.depth_deformation.tactile_right_right"]
    )
    axes[2].plot(timestamps, _normalize_series(contact_left), label="contact left mean", color="#2ca02c", linewidth=1.5)
    axes[2].plot(
        timestamps, _normalize_series(state_gl), label="state gripper_left", color="#2ca02c", linestyle="--", linewidth=1.0
    )
    axes[2].plot(timestamps, _normalize_series(contact_right), label="contact right mean", color="#9467bd", linewidth=1.5)
    axes[2].plot(
        timestamps, _normalize_series(state_gr), label="state gripper_right", color="#9467bd", linestyle="--", linewidth=1.0
    )
    axes[2].set_title("Normalized contact vs gripper state")
    axes[2].set_ylabel("normalized")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)

    axes[3].plot(
        timestamps,
        np.gradient(contact_left, timestamps, edge_order=1),
        label="d(contact_left)/dt",
        color="#2ca02c",
        linewidth=1.1,
    )
    axes[3].plot(
        timestamps,
        np.gradient(contact_right, timestamps, edge_order=1),
        label="d(contact_right)/dt",
        color="#9467bd",
        linewidth=1.1,
    )
    axes[3].plot(
        timestamps,
        np.gradient(state_gl, timestamps, edge_order=1),
        label="d(gripper_left_state)/dt",
        color="#2ca02c",
        linestyle="--",
        linewidth=0.9,
    )
    axes[3].plot(
        timestamps,
        np.gradient(state_gr, timestamps, edge_order=1),
        label="d(gripper_right_state)/dt",
        color="#9467bd",
        linestyle="--",
        linewidth=0.9,
    )
    axes[3].set_title("Contact and gripper change rates")
    axes[3].set_ylabel("rate")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(True, alpha=0.2)
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_pose_action_overview(
    output_path: Path,
    timestamps: np.ndarray,
    action: np.ndarray,
    state: np.ndarray,
    action_names: list[str],
    state_names: list[str],
) -> None:
    pose_names = ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), dpi=180, sharex=True)
    for ax, name in zip(axes.flatten(), pose_names, strict=True):
        ax.plot(timestamps, state[:, state_names.index(name)], label=f"state {name}", linewidth=1.4)
        ax.plot(timestamps, action[:, action_names.index(name)], label=f"action {name}", linewidth=1.0, alpha=0.7)
        ax.set_title(name)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("TCP pose state/action trajectories", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    output_path: Path,
    frame_index: np.ndarray,
    action: np.ndarray,
    state: np.ndarray,
    action_names: list[str],
    state_names: list[str],
    tactile_scores: dict[str, np.ndarray],
) -> None:
    selected_action_names = [
        "left_x",
        "left_y",
        "left_z",
        "right_x",
        "right_y",
        "right_z",
        "arm_left_1",
        "arm_left_2",
        "arm_left_3",
        "arm_left_4",
        "arm_left_5",
        "arm_left_6",
        "arm_left_7",
        "arm_right_1",
        "arm_right_2",
        "arm_right_3",
        "arm_right_4",
        "arm_right_5",
        "arm_right_6",
        "arm_right_7",
        "gripper_left",
        "gripper_right",
    ]
    selected_state_names = selected_action_names + [
        "left_Acc_X",
        "left_Acc_Y",
        "left_Acc_Z",
        "right_Acc_X",
        "right_Acc_Y",
        "right_Acc_Z",
    ]

    action_matrix = np.stack([action[:, action_names.index(name)] for name in selected_action_names], axis=0)
    state_matrix = np.stack([state[:, state_names.index(name)] for name in selected_state_names], axis=0)
    tactile_matrix = np.stack([tactile_scores[key] for key in TACTILE_KEYS], axis=0)

    def zscore_rows(matrix: np.ndarray) -> np.ndarray:
        mean = matrix.mean(axis=1, keepdims=True)
        std = matrix.std(axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (matrix - mean) / std

    fig, axes = plt.subplots(3, 1, figsize=(18, 11), dpi=180, sharex=True)
    im0 = axes[0].imshow(zscore_rows(tactile_matrix), aspect="auto", cmap="magma", interpolation="nearest")
    axes[0].set_yticks(range(len(TACTILE_KEYS)))
    axes[0].set_yticklabels([key.split(".")[-1] for key in TACTILE_KEYS], fontsize=8)
    axes[0].set_title("Tactile contact_stats heatmap (row-wise z-score)")
    fig.colorbar(im0, ax=axes[0], fraction=0.012, pad=0.01)

    im1 = axes[1].imshow(zscore_rows(state_matrix), aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[1].set_yticks(range(len(selected_state_names)))
    axes[1].set_yticklabels(selected_state_names, fontsize=7)
    axes[1].set_title("Selected state dimensions heatmap (row-wise z-score)")
    fig.colorbar(im1, ax=axes[1], fraction=0.012, pad=0.01)

    im2 = axes[2].imshow(zscore_rows(action_matrix), aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[2].set_yticks(range(len(selected_action_names)))
    axes[2].set_yticklabels(selected_action_names, fontsize=7)
    axes[2].set_title("Selected action dimensions heatmap (row-wise z-score)")
    tick_count = min(12, len(frame_index))
    tick_positions = np.linspace(0, len(frame_index) - 1, tick_count, dtype=int)
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels([str(int(frame_index[idx])) for idx in tick_positions], rotation=0, fontsize=8)
    axes[2].set_xlabel("frame index")
    fig.colorbar(im2, ax=axes[2], fraction=0.012, pad=0.01)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _normalize_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.float32:
        frame = frame.astype(np.float32)
    if frame.max() > 1.0:
        frame = frame / 255.0
    return np.clip(frame, 0.0, 1.0)


def _normalize_tactile_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.shape[2] >= 3:
        frame = frame[..., :3]
    if frame.max() > frame.min():
        frame = (frame - frame.min()) / (frame.max() - frame.min())
    else:
        frame = np.zeros_like(frame)
    return np.clip(frame, 0.0, 1.0)


def _plot_visual_strip(
    output_path: Path,
    dataset_root: Path,
    episode_row: dict[str, object],
    timestamps: np.ndarray,
    strip_samples: int,
    tolerance_s: float,
    video_backend: str,
) -> None:
    sample_indices = np.linspace(0, len(timestamps) - 1, strip_samples, dtype=int)
    sample_timestamps = [float(timestamps[idx]) for idx in sample_indices]

    decoded_rgb = {}
    for key in RGB_KEYS:
        video_path = _resolve_episode_video_path(dataset_root, episode_row, key)
        frames = decode_video_frames(video_path, sample_timestamps, tolerance_s, backend=video_backend)
        decoded_rgb[key] = frames.permute(0, 2, 3, 1).cpu().numpy()

    decoded_tactile = {}
    for key in TACTILE_KEYS:
        video_path = _resolve_episode_video_path(dataset_root, episode_row, key)
        frames = decode_video_frames(video_path, sample_timestamps, tolerance_s, backend="pyav")
        decoded_tactile[key] = frames.permute(0, 2, 3, 1).cpu().numpy()

    nrows = len(RGB_KEYS) + len(TACTILE_KEYS)
    fig, axes = plt.subplots(nrows, strip_samples, figsize=(2.3 * strip_samples, 2.1 * nrows), dpi=180)

    for col, timestamp_s in enumerate(sample_timestamps):
        axes[0, col].set_title(f"t={timestamp_s:.1f}s", fontsize=9)

    for row, key in enumerate(RGB_KEYS):
        for col in range(strip_samples):
            axes[row, col].imshow(_normalize_rgb_frame(decoded_rgb[key][col]))
            axes[row, col].axis("off")
        axes[row, 0].text(
            -0.1,
            0.5,
            key.split(".")[-1],
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=9,
        )

    tactile_offset = len(RGB_KEYS)
    for row, key in enumerate(TACTILE_KEYS, start=tactile_offset):
        for col in range(strip_samples):
            axes[row, col].imshow(_normalize_tactile_frame(decoded_tactile[key][col]))
            axes[row, col].axis("off")
        axes[row, 0].text(
            -0.1,
            0.5,
            key.split(".")[-1],
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=9,
        )

    fig.suptitle("Episode visual strip: RGB cameras and tactile streams", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    dataset_root: Path,
    episode_index: int,
    frame_count: int,
    fps: float,
    topk_ratio: float,
    tactile_scores: dict[str, np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    state_names: list[str],
    action_names: list[str],
) -> None:
    duration_s = frame_count / fps
    gripper_state_left = state[:, state_names.index("gripper_left")]
    gripper_state_right = state[:, state_names.index("gripper_right")]
    gripper_action_left = action[:, action_names.index("gripper_left")]
    gripper_action_right = action[:, action_names.index("gripper_right")]

    lines = [
        "# Episode Visualization Report",
        "",
        f"- Dataset: `{dataset_root}`",
        f"- Episode index: `{episode_index}`",
        f"- Frames: `{frame_count}`",
        f"- Duration: `{duration_s:.2f}s` at `{fps:.1f} Hz`",
        f"- Tactile contact top-k ratio: `{topk_ratio:.3f}`",
        "",
        "## Tactile Summary",
        "",
    ]
    for key in TACTILE_KEYS:
        stats = _summarize_series(tactile_scores[key])
        lines.append(
            f"- `{key.split('.')[-1]}`: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"p05={stats['p05']:.4f}, p95={stats['p95']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Gripper Summary",
            "",
            f"- `state gripper_left`: mean={gripper_state_left.mean():.4f}, min={gripper_state_left.min():.4f}, max={gripper_state_left.max():.4f}",
            f"- `state gripper_right`: mean={gripper_state_right.mean():.4f}, min={gripper_state_right.min():.4f}, max={gripper_state_right.max():.4f}",
            f"- `action gripper_left`: mean={gripper_action_left.mean():.4f}, min={gripper_action_left.min():.4f}, max={gripper_action_left.max():.4f}",
            f"- `action gripper_right`: mean={gripper_action_right.mean():.4f}, min={gripper_action_right.min():.4f}, max={gripper_action_right.max():.4f}",
            "",
            "## Output Files",
            "",
            "- `episode0_framewise.csv`: framewise tactile/state/action values",
            "- `episode0_contact_and_gripper.png`: tactile scores with gripper curves",
            "- `episode0_pose_state_action.png`: left/right xyz state/action trajectories",
            "- `episode0_heatmaps.png`: tactile/state/action heatmaps",
            "- `episode0_visual_strip.png`: sampled RGB and tactile frames",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = _load_info(dataset_root)
    fps = float(info["fps"])
    state_names = list(info["features"]["observation.state"]["names"])
    action_names = list(info["features"]["action"]["names"])

    episode_row = _load_episode_row(dataset_root, args.episode_index)
    episode_table = _load_episode_table(dataset_root, args.episode_index)
    frame_index, timestamps, action, state = _sort_episode_arrays(episode_table)

    tactile_scores: dict[str, np.ndarray] = {}
    for key in TACTILE_KEYS:
        video_path = _resolve_episode_video_path(dataset_root, episode_row, key)
        tactile_frames = decode_video_frames(video_path, timestamps.tolist(), args.tolerance_s, backend="pyav")
        tactile_scores[key] = _compute_contact_stats(tactile_frames, args.tactile_contact_topk_ratio)

    framewise_csv_path = output_dir / "episode0_framewise.csv"
    _write_framewise_csv(
        framewise_csv_path,
        frame_index,
        timestamps,
        action,
        state,
        action_names,
        state_names,
        tactile_scores,
    )

    _plot_contact_and_gripper(
        output_dir / "episode0_contact_and_gripper.png",
        timestamps,
        tactile_scores,
        action,
        state,
        action_names,
        state_names,
        args.tactile_contact_topk_ratio,
    )
    _plot_pose_action_overview(
        output_dir / "episode0_pose_state_action.png",
        timestamps,
        action,
        state,
        action_names,
        state_names,
    )
    _plot_heatmaps(
        output_dir / "episode0_heatmaps.png",
        frame_index,
        action,
        state,
        action_names,
        state_names,
        tactile_scores,
    )
    _plot_visual_strip(
        output_dir / "episode0_visual_strip.png",
        dataset_root,
        episode_row,
        timestamps,
        args.strip_samples,
        args.tolerance_s,
        args.video_backend,
    )
    _write_report(
        output_dir / "report.md",
        dataset_root,
        args.episode_index,
        len(frame_index),
        fps,
        args.tactile_contact_topk_ratio,
        tactile_scores,
        state,
        action,
        state_names,
        action_names,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "episode_index": args.episode_index,
        "frame_count": int(len(frame_index)),
        "duration_s": float(len(frame_index) / fps),
        "fps": fps,
        "tactile_contact_topk_ratio": float(args.tactile_contact_topk_ratio),
        "outputs": [
            "episode0_framewise.csv",
            "episode0_contact_and_gripper.png",
            "episode0_pose_state_action.png",
            "episode0_heatmaps.png",
            "episode0_visual_strip.png",
            "report.md",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
