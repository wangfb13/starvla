#!/usr/bin/env python3

"""Analyze continuity, stalls, and jumps in raw Sparky teleoperation trajectories."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


LEFT_XYZ = slice(0, 3)
RIGHT_XYZ = slice(7, 10)
MISSING_VALUE = 9930.0


@dataclass
class EpisodeMetrics:
    episode_index: int
    num_frames: int
    duration_s: float
    action_path_left: float
    action_path_right: float
    state_path_left: float
    state_path_right: float
    action_stall_ratio_left: float
    action_stall_ratio_right: float
    state_stall_ratio_left: float
    state_stall_ratio_right: float
    max_action_jump_left: float
    max_action_jump_right: float
    max_state_jump_left: float
    max_state_jump_right: float
    mean_action_state_gap_left: float
    mean_action_state_gap_right: float
    max_action_state_gap_left: float
    max_action_state_gap_right: float
    longest_action_stall_left: int
    longest_action_stall_right: int
    longest_state_stall_left: int
    longest_state_stall_right: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="/mnt/tacumi_data/lerobot_datasetV30_train_ready/sparky_ugripper/sparky_ugripper_fruit_packing_v2gripper_R7_115_v30_tcp_train_ready",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset_root>/visualizations/trajectory_continuity_raw.",
    )
    parser.add_argument(
        "--stall-threshold-m",
        type=float,
        default=0.0008,
        help="Frame-to-frame xyz motion below this is counted as a stall step.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=3,
        help="How many worst episodes to visualize for each anomaly type.",
    )
    return parser.parse_args()


def _load_raw_tables(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    all_episode = []
    all_frame = []
    all_action = []
    all_state = []
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action", "observation.state"])
        all_episode.append(np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64))
        all_frame.append(np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64))
        all_action.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        all_state.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))

    episode_index = np.concatenate(all_episode, axis=0)
    frame_index = np.concatenate(all_frame, axis=0)
    action = np.concatenate(all_action, axis=0)
    state = np.concatenate(all_state, axis=0)
    return episode_index, frame_index, action, state


def _valid_xyz(xyz: np.ndarray) -> np.ndarray:
    finite = np.isfinite(xyz).all(axis=1)
    not_missing = np.abs(xyz - MISSING_VALUE).max(axis=1) < 1e-3
    return finite & (~not_missing)


def _step_norms(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = _valid_xyz(xyz)
    if len(xyz) < 2:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
    valid_pairs = valid[:-1] & valid[1:]
    deltas = xyz[1:] - xyz[:-1]
    norms = np.linalg.norm(deltas, axis=1).astype(np.float32)
    norms[~valid_pairs] = np.nan
    return norms, valid_pairs


def _gap_norms(action_xyz: np.ndarray, state_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = _valid_xyz(action_xyz) & _valid_xyz(state_xyz)
    if not valid.any():
        return np.zeros(0, dtype=np.float32), valid
    gaps = np.linalg.norm(action_xyz - state_xyz, axis=1).astype(np.float32)
    gaps[~valid] = np.nan
    return gaps, valid


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask.tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _safe_nanmean(values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _safe_nanmax(values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmax(values))


def _compute_episode_metrics(
    episode_index: int,
    frame_index: np.ndarray,
    action_xyz_left: np.ndarray,
    action_xyz_right: np.ndarray,
    state_xyz_left: np.ndarray,
    state_xyz_right: np.ndarray,
    stall_threshold_m: float,
    fps: float = 30.0,
) -> EpisodeMetrics:
    a_left_steps, a_left_valid = _step_norms(action_xyz_left)
    a_right_steps, a_right_valid = _step_norms(action_xyz_right)
    s_left_steps, s_left_valid = _step_norms(state_xyz_left)
    s_right_steps, s_right_valid = _step_norms(state_xyz_right)

    gap_left, gap_left_valid = _gap_norms(action_xyz_left, state_xyz_left)
    gap_right, gap_right_valid = _gap_norms(action_xyz_right, state_xyz_right)

    a_left_stall = np.isfinite(a_left_steps) & (a_left_steps < stall_threshold_m)
    a_right_stall = np.isfinite(a_right_steps) & (a_right_steps < stall_threshold_m)
    s_left_stall = np.isfinite(s_left_steps) & (s_left_steps < stall_threshold_m)
    s_right_stall = np.isfinite(s_right_steps) & (s_right_steps < stall_threshold_m)

    def stall_ratio(mask: np.ndarray, valid_pairs: np.ndarray) -> float:
        denom = max(int(valid_pairs.sum()), 1)
        return float(mask.sum() / denom)

    return EpisodeMetrics(
        episode_index=episode_index,
        num_frames=int(len(frame_index)),
        duration_s=float(max(len(frame_index) - 1, 0) / fps),
        action_path_left=float(np.nansum(a_left_steps)),
        action_path_right=float(np.nansum(a_right_steps)),
        state_path_left=float(np.nansum(s_left_steps)),
        state_path_right=float(np.nansum(s_right_steps)),
        action_stall_ratio_left=stall_ratio(a_left_stall, a_left_valid),
        action_stall_ratio_right=stall_ratio(a_right_stall, a_right_valid),
        state_stall_ratio_left=stall_ratio(s_left_stall, s_left_valid),
        state_stall_ratio_right=stall_ratio(s_right_stall, s_right_valid),
        max_action_jump_left=_safe_nanmax(a_left_steps),
        max_action_jump_right=_safe_nanmax(a_right_steps),
        max_state_jump_left=_safe_nanmax(s_left_steps),
        max_state_jump_right=_safe_nanmax(s_right_steps),
        mean_action_state_gap_left=_safe_nanmean(gap_left),
        mean_action_state_gap_right=_safe_nanmean(gap_right),
        max_action_state_gap_left=_safe_nanmax(gap_left),
        max_action_state_gap_right=_safe_nanmax(gap_right),
        longest_action_stall_left=_longest_true_run(a_left_stall),
        longest_action_stall_right=_longest_true_run(a_right_stall),
        longest_state_stall_left=_longest_true_run(s_left_stall),
        longest_state_stall_right=_longest_true_run(s_right_stall),
    )


def _plot_distribution(values: np.ndarray, title: str, xlabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    clean = values[np.isfinite(values)]
    ax.hist(clean, bins=60, color="C0", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode count")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_episode_example(
    episode_index: int,
    frame_index: np.ndarray,
    action_xyz: np.ndarray,
    state_xyz: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, dpi=180)
    labels = ["x", "y", "z"]
    for dim, ax in enumerate(axes):
        ax.plot(frame_index, action_xyz[:, dim], label="action", linewidth=1.6, color="C1")
        ax.plot(frame_index, state_xyz[:, dim], label="state", linewidth=1.6, color="C0", alpha=0.85)
        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend()
    axes[-1].set_xlabel("frame index")
    fig.suptitle(f"Episode {episode_index} | {title}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.dataset_root / "visualizations" / "trajectory_continuity_raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_ids, frame_index, action, state = _load_raw_tables(args.dataset_root)
    action_left = action[:, LEFT_XYZ]
    action_right = action[:, RIGHT_XYZ]
    state_left = state[:, LEFT_XYZ]
    state_right = state[:, RIGHT_XYZ]

    unique_episode_ids = np.unique(episode_ids)
    metrics: list[EpisodeMetrics] = []
    for ep in unique_episode_ids.tolist():
        mask = episode_ids == ep
        metrics.append(
            _compute_episode_metrics(
                episode_index=int(ep),
                frame_index=frame_index[mask],
                action_xyz_left=action_left[mask],
                action_xyz_right=action_right[mask],
                state_xyz_left=state_left[mask],
                state_xyz_right=state_right[mask],
                stall_threshold_m=args.stall_threshold_m,
            )
        )

    metrics_dicts = [m.__dict__ for m in metrics]

    action_stall_right = np.asarray([m.action_stall_ratio_right for m in metrics], dtype=np.float32)
    state_stall_right = np.asarray([m.state_stall_ratio_right for m in metrics], dtype=np.float32)
    jump_right = np.asarray([m.max_action_jump_right for m in metrics], dtype=np.float32)
    gap_right = np.asarray([m.mean_action_state_gap_right for m in metrics], dtype=np.float32)

    _plot_distribution(
        action_stall_right,
        title="Right-arm action stall ratio per episode",
        xlabel="stall ratio",
        output_path=output_dir / "right_action_stall_ratio_hist.png",
    )
    _plot_distribution(
        jump_right,
        title="Right-arm max action jump per episode",
        xlabel="max frame-to-frame xyz jump (m)",
        output_path=output_dir / "right_action_jump_hist.png",
    )
    _plot_distribution(
        gap_right,
        title="Right-arm mean |action_xyz - state_xyz| per episode",
        xlabel="mean gap (m)",
        output_path=output_dir / "right_action_state_gap_hist.png",
    )

    worst_stall_eps = sorted(metrics, key=lambda x: x.longest_action_stall_right, reverse=True)[: args.example_count]
    worst_jump_eps = sorted(metrics, key=lambda x: x.max_action_jump_right, reverse=True)[: args.example_count]
    worst_gap_eps = sorted(metrics, key=lambda x: x.mean_action_state_gap_right, reverse=True)[: args.example_count]

    for rank, episode_metrics in enumerate(worst_stall_eps, start=1):
        ep_mask = episode_ids == episode_metrics.episode_index
        _plot_episode_example(
            episode_index=episode_metrics.episode_index,
            frame_index=frame_index[ep_mask],
            action_xyz=action_right[ep_mask],
            state_xyz=state_right[ep_mask],
            title=f"worst action stall #{rank}",
            output_path=output_dir / f"examples/worst_stall_right_{rank:02d}_ep{episode_metrics.episode_index}.png",
        )

    for rank, episode_metrics in enumerate(worst_jump_eps, start=1):
        ep_mask = episode_ids == episode_metrics.episode_index
        _plot_episode_example(
            episode_index=episode_metrics.episode_index,
            frame_index=frame_index[ep_mask],
            action_xyz=action_right[ep_mask],
            state_xyz=state_right[ep_mask],
            title=f"worst action jump #{rank}",
            output_path=output_dir / f"examples/worst_jump_right_{rank:02d}_ep{episode_metrics.episode_index}.png",
        )

    for rank, episode_metrics in enumerate(worst_gap_eps, start=1):
        ep_mask = episode_ids == episode_metrics.episode_index
        _plot_episode_example(
            episode_index=episode_metrics.episode_index,
            frame_index=frame_index[ep_mask],
            action_xyz=action_right[ep_mask],
            state_xyz=state_right[ep_mask],
            title=f"worst action-state gap #{rank}",
            output_path=output_dir / f"examples/worst_gap_right_{rank:02d}_ep{episode_metrics.episode_index}.png",
        )

    summary = {
        "dataset_root": str(args.dataset_root),
        "stall_threshold_m": args.stall_threshold_m,
        "episode_count": len(metrics),
        "global": {
            "mean_action_stall_ratio_right": float(np.nanmean(action_stall_right)),
            "mean_state_stall_ratio_right": float(np.nanmean(state_stall_right)),
            "mean_action_state_gap_right": float(np.nanmean(gap_right)),
            "p95_action_state_gap_right": float(np.nanpercentile(gap_right, 95)),
            "p95_max_action_jump_right": float(np.nanpercentile(jump_right, 95)),
            "max_action_jump_right": float(np.nanmax(jump_right)),
            "max_longest_action_stall_right": int(max(m.longest_action_stall_right for m in metrics)),
            "max_longest_state_stall_right": int(max(m.longest_state_stall_right for m in metrics)),
        },
        "worst_examples": {
            "stall_right": [m.__dict__ for m in worst_stall_eps],
            "jump_right": [m.__dict__ for m in worst_jump_eps],
            "gap_right": [m.__dict__ for m in worst_gap_eps],
        },
        "episodes": metrics_dicts,
    }

    summary_path = output_dir / "trajectory_continuity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved: {summary_path}")
    print(f"saved: {output_dir / 'right_action_stall_ratio_hist.png'}")
    print(f"saved: {output_dir / 'right_action_jump_hist.png'}")
    print(f"saved: {output_dir / 'right_action_state_gap_hist.png'}")
    print("top stall episodes:", [m.episode_index for m in worst_stall_eps])
    print("top jump episodes:", [m.episode_index for m in worst_jump_eps])
    print("top gap episodes:", [m.episode_index for m in worst_gap_eps])


if __name__ == "__main__":
    main()
