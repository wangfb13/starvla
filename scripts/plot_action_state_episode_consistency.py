#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from episode_parquet_reader import EpisodePair, iter_episode_pairs


LEFT_ACTION = slice(0, 3)
RIGHT_ACTION = slice(7, 10)
LEFT_STATE = slice(0, 3)
RIGHT_STATE = slice(7, 10)


def set_3d_limits(ax, all_points: np.ndarray) -> None:
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = np.max(maxs - mins) / 2.0
    if half_range == 0:
        half_range = 1.0
    half_range *= 1.05
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def episode_score(item: EpisodePair) -> dict:
    action_left, state_left = item.action[:, LEFT_ACTION], item.state[:, LEFT_STATE]
    action_right, state_right = item.action[:, RIGHT_ACTION], item.state[:, RIGHT_STATE]
    left_err = np.linalg.norm(action_left - state_left, axis=1)
    right_err = np.linalg.norm(action_right - state_right, axis=1)
    return {
        "episode": item.episode_name,
        "score": float((left_err.mean() + right_err.mean()) / 2.0),
        "left_mean_error": float(left_err.mean()),
        "right_mean_error": float(right_err.mean()),
    }


def plot_episode(item: EpisodePair, output_path: Path, title_prefix: str) -> dict:
    action_left, state_left = item.action[:, LEFT_ACTION], item.state[:, LEFT_STATE]
    action_right, state_right = item.action[:, RIGHT_ACTION], item.state[:, RIGHT_STATE]

    left_err = np.linalg.norm(action_left - state_left, axis=1)
    right_err = np.linalg.norm(action_right - state_right, axis=1)
    score = float((left_err.mean() + right_err.mean()) / 2.0)

    fig = plt.figure(figsize=(14, 6), dpi=180)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    arm_data = [
        (axes[0], "Left arm", action_left, state_left, float(left_err.mean())),
        (axes[1], "Right arm", action_right, state_right, float(right_err.mean())),
    ]

    for ax, arm_name, action, state, arm_err in arm_data:
        ax.plot(action[:, 0], action[:, 1], action[:, 2], color="tab:blue", linewidth=1.6, label="action")
        ax.plot(state[:, 0], state[:, 1], state[:, 2], color="tab:orange", linewidth=1.2, alpha=0.9, label="state")
        ax.scatter(action[0, 0], action[0, 1], action[0, 2], color="navy", s=26, marker="o")
        ax.scatter(action[-1, 0], action[-1, 1], action[-1, 2], color="navy", s=32, marker="X")
        ax.scatter(state[0, 0], state[0, 1], state[0, 2], color="darkorange", s=26, marker="o")
        ax.scatter(state[-1, 0], state[-1, 1], state[-1, 2], color="darkorange", s=32, marker="X")
        set_3d_limits(ax, np.concatenate([action, state], axis=0))
        ax.set_title(f"{arm_name} | mean err = {arm_err:.4f} m")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{title_prefix}: {item.episode_name} | combined score = {score:.4f} m", y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "episode": item.episode_name,
        "score": score,
        "left_mean_error": float(left_err.mean()),
        "right_mean_error": float(right_err.mean()),
        "image": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one consistent and one inconsistent action/state trajectory figure.")
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
        help="Directory to save outputs. Defaults to <dataset_root>/visualizations/action_state_consistency.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.dataset_root / "visualizations" / "action_state_consistency")
    items = list(iter_episode_pairs(args.dataset_root, slice(None), slice(None)))
    scores = [episode_score(item) for item in items]
    scores_sorted = sorted(scores, key=lambda x: x["score"])
    best = next(item for item in items if item.episode_name == scores_sorted[0]["episode"])
    worst = next(item for item in items if item.episode_name == scores_sorted[-1]["episode"])

    consistent_info = plot_episode(best, output_dir / "action_state_consistent_episode.png", "Most consistent")
    inconsistent_info = plot_episode(worst, output_dir / "action_state_inconsistent_episode.png", "Most inconsistent")

    summary = {
        "most_consistent": consistent_info,
        "most_inconsistent": inconsistent_info,
        "top10_best": scores_sorted[:10],
        "top10_worst": scores_sorted[-10:],
    }
    summary_path = output_dir / "action_state_episode_consistency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved: {consistent_info['image']}")
    print(f"saved: {inconsistent_info['image']}")
    print(f"saved: {summary_path}")
    print("best:", consistent_info["episode"], consistent_info["score"])
    print("worst:", inconsistent_info["episode"], inconsistent_info["score"])


if __name__ == "__main__":
    main()
