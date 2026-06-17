#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


LEFT_IDXS = (0, 1, 2)
RIGHT_IDXS = (7, 8, 9)
BINS = 101
XS = np.linspace(0.0, 1.0, BINS)


def load_episode_points(parquet_path: Path, idxs: tuple[int, int, int]) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table.column(0).to_pylist(), dtype=np.float32)
    return actions[:, idxs]


def collect_trajectories(dataset_root: Path, idxs: tuple[int, int, int]) -> list[np.ndarray]:
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {dataset_root / 'data'}")

    trajectories = []
    for path in parquet_paths:
        pts = load_episode_points(path, idxs)
        trajectories.append(pts)
    return trajectories


def interpolate_trajectory(traj: np.ndarray, bins: int = BINS) -> np.ndarray:
    if len(traj) == 1:
        return np.repeat(traj, bins, axis=0)
    t = np.linspace(0.0, 1.0, len(traj))
    out = np.empty((bins, traj.shape[1]), dtype=np.float32)
    for d in range(traj.shape[1]):
        out[:, d] = np.interp(XS, t, traj[:, d])
    return out


def interpolate_trajectories(trajectories: list[np.ndarray]) -> np.ndarray:
    return np.stack([interpolate_trajectory(traj) for traj in trajectories], axis=0)


def set_3d_limits(ax, all_points: np.ndarray) -> None:
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = np.max(maxs - mins) / 2.0
    if half_range == 0:
        half_range = 1.0
    pad = half_range * 0.05
    half_range += pad
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def summarize_trajectories(trajectories: list[np.ndarray]) -> dict:
    all_points = np.concatenate(trajectories, axis=0)
    starts = np.stack([pts[0] for pts in trajectories])
    ends = np.stack([pts[-1] for pts in trajectories])
    disp = np.linalg.norm(ends - starts, axis=1)

    return {
        "episodes": len(trajectories),
        "total_frames": int(all_points.shape[0]),
        "mean_episode_length": float(np.mean([len(pts) for pts in trajectories])),
        "xyz_min": all_points.min(axis=0).tolist(),
        "xyz_max": all_points.max(axis=0).tolist(),
        "xyz_mean": all_points.mean(axis=0).tolist(),
        "xyz_std": all_points.std(axis=0).tolist(),
        "xyz_p05": np.percentile(all_points, 5, axis=0).tolist(),
        "xyz_p50": np.percentile(all_points, 50, axis=0).tolist(),
        "xyz_p95": np.percentile(all_points, 95, axis=0).tolist(),
        "start_mean": starts.mean(axis=0).tolist(),
        "start_std": starts.std(axis=0).tolist(),
        "end_mean": ends.mean(axis=0).tolist(),
        "end_std": ends.std(axis=0).tolist(),
        "start_end_distance_mean": float(disp.mean()),
        "start_end_distance_median": float(np.median(disp)),
        "start_end_distance_p90": float(np.percentile(disp, 90)),
    }


def analyze_relative_spread(trajectories: list[np.ndarray]) -> dict:
    arr = interpolate_trajectories(trajectories)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    spread = np.linalg.norm(std, axis=1)
    peak_i = int(spread.argmax())
    peak_pts = arr[:, peak_i, :]
    peak_center = peak_pts.mean(axis=0)
    peak_dist = np.linalg.norm(peak_pts - peak_center, axis=1)
    outlier_order = np.argsort(peak_dist)[::-1][:8]

    return {
        "bins": BINS,
        "progress": XS.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "spread": spread.tolist(),
        "peak_bin": peak_i,
        "peak_progress": float(XS[peak_i]),
        "peak_spread": float(spread[peak_i]),
        "peak_center": peak_center.tolist(),
        "peak_outliers": [
            {
                "episode_index": int(i),
                "distance": float(peak_dist[i]),
                "point": peak_pts[i].tolist(),
            }
            for i in outlier_order
        ],
    }


def plot_arm(trajectories: list[np.ndarray], title: str, output_path: Path) -> None:
    all_points = np.concatenate(trajectories, axis=0)
    colors = plt.cm.turbo(np.linspace(0, 1, len(trajectories)))
    starts = np.stack([pts[0] for pts in trajectories])
    ends = np.stack([pts[-1] for pts in trajectories])
    mean_start = starts.mean(axis=0)
    mean_end = ends.mean(axis=0)

    fig = plt.figure(figsize=(9, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    for i, pts in enumerate(trajectories):
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=colors[i],
            alpha=0.22,
            linewidth=0.8,
        )

    ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], color="limegreen", s=8, alpha=0.65, label="start")
    ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2], color="crimson", s=8, alpha=0.65, label="end")
    ax.scatter(
        [mean_start[0]],
        [mean_start[1]],
        [mean_start[2]],
        color="green",
        s=90,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        label="mean start",
    )
    ax.scatter(
        [mean_end[0]],
        [mean_end[1]],
        [mean_end[2]],
        color="red",
        s=90,
        marker="X",
        edgecolors="black",
        linewidths=0.5,
        label="mean end",
    )

    set_3d_limits(ax, all_points)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_spread_curves(left_analysis: dict, right_analysis: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.plot(left_analysis["progress"], left_analysis["spread"], label="left arm", linewidth=2)
    ax.plot(right_analysis["progress"], right_analysis["spread"], label="right arm", linewidth=2)
    ax.scatter(
        [left_analysis["peak_progress"]],
        [left_analysis["peak_spread"]],
        color="C0",
        s=55,
        zorder=3,
    )
    ax.scatter(
        [right_analysis["peak_progress"]],
        [right_analysis["peak_spread"]],
        color="C1",
        s=55,
        zorder=3,
    )
    ax.axvline(left_analysis["peak_progress"], color="C0", linestyle="--", alpha=0.25)
    ax.axvline(right_analysis["peak_progress"], color="C1", linestyle="--", alpha=0.25)
    ax.set_xlabel("Relative progress")
    ax.set_ylabel("Spread = ||std_xyz||")
    ax.set_title("Episode-to-episode trajectory spread over relative progress")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_peak_scatter(trajectories: list[np.ndarray], analysis: dict, title: str, output_path: Path) -> None:
    arr = interpolate_trajectories(trajectories)
    peak_i = analysis["peak_bin"]
    pts = arr[:, peak_i, :]
    center = np.asarray(analysis["peak_center"], dtype=np.float32)
    outliers = analysis["peak_outliers"]
    outlier_indices = {item["episode_index"] for item in outliers}

    fig = plt.figure(figsize=(9, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="0.75", s=18, alpha=0.6)
    ax.scatter(center[0], center[1], center[2], color="black", s=80, marker="x", label="center")

    for idx, p in enumerate(pts):
        if idx not in outlier_indices:
            continue
        ax.scatter(p[0], p[1], p[2], color="crimson", s=40, edgecolors="black", linewidths=0.4)
        ax.text(p[0], p[1], p[2], str(idx), fontsize=7)

    ax.set_title(f"{title} at progress {analysis['peak_progress']:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True, alpha=0.2)
    all_points = pts
    set_3d_limits(ax, all_points)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Sparky R8_100 left/right action xyz trajectories.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="/mnt/cpfs/datasets/sparky_ugripper_fruit_packing_v2gripper_R8_100",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Defaults to <dataset_root>/visualizations.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir or (dataset_root / "visualizations")

    left_trajs = collect_trajectories(dataset_root, LEFT_IDXS)
    right_trajs = collect_trajectories(dataset_root, RIGHT_IDXS)
    left_analysis = analyze_relative_spread(left_trajs)
    right_analysis = analyze_relative_spread(right_trajs)

    left_out = output_dir / "r8_100_left_arm_action_xyz_3d.png"
    right_out = output_dir / "r8_100_right_arm_action_xyz_3d.png"
    spread_out = output_dir / "r8_100_action_xyz_spread_by_progress.png"
    peak_scatter_out = output_dir / "r8_100_action_xyz_peak_scatter.png"
    stats_out = output_dir / "r8_100_action_xyz_distribution_summary.json"

    plot_arm(left_trajs, "R8_100 Left Arm Action XYZ Trajectories", left_out)
    plot_arm(right_trajs, "R8_100 Right Arm Action XYZ Trajectories", right_out)
    plot_spread_curves(left_analysis, right_analysis, spread_out)
    plot_peak_scatter(left_trajs, left_analysis, "R8_100 Left Arm Peak-Spread Scatter", peak_scatter_out)
    # Reuse the same peak-scatter file by appending the right arm figure next to it.
    # Keep the output compact and easy to inspect.
    
    right_peak_scatter_out = output_dir / "r8_100_action_xyz_peak_scatter_right.png"
    plot_peak_scatter(right_trajs, right_analysis, "R8_100 Right Arm Peak-Spread Scatter", right_peak_scatter_out)

    stats = {
        "left_arm": summarize_trajectories(left_trajs),
        "left_arm_relative_spread": left_analysis,
        "right_arm": summarize_trajectories(right_trajs),
        "right_arm_relative_spread": right_analysis,
    }
    stats_out.write_text(json.dumps(stats, indent=2))

    print(f"saved: {left_out}")
    print(f"saved: {right_out}")
    print(f"saved: {spread_out}")
    print(f"saved: {peak_scatter_out}")
    print(f"saved: {right_peak_scatter_out}")
    print(f"saved: {stats_out}")
    print(f"episodes: {len(left_trajs)}")


if __name__ == "__main__":
    main()
