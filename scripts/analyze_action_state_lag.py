#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from episode_parquet_reader import iter_episode_pairs


LEFT_ACTION = slice(0, 3)
RIGHT_ACTION = slice(7, 10)
LEFT_STATE = slice(0, 3)
RIGHT_STATE = slice(7, 10)


def lag_distance(action: np.ndarray, state: np.ndarray, lag: int) -> np.ndarray:
    if lag > 0:
        a = action[:-lag]
        s = state[lag:]
    elif lag < 0:
        a = action[-lag:]
        s = state[:lag]
    else:
        a = action
        s = state
    return np.linalg.norm(a - s, axis=1)


def analyze_arm(dataset_root: Path, action_slice: slice, state_slice: slice, lags: np.ndarray) -> dict:
    per_episode_mean = []
    per_episode_best_lag = []
    per_episode_zero = []
    per_episode_best = []
    episode_names = []

    for item in iter_episode_pairs(dataset_root, action_slice, state_slice):
        action, state = item.action, item.state
        lag_means = []
        for lag in lags:
            d = lag_distance(action, state, int(lag))
            lag_means.append(float(d.mean()))
        lag_means = np.asarray(lag_means, dtype=np.float32)
        best_idx = int(lag_means.argmin())
        per_episode_mean.append(lag_means)
        per_episode_best_lag.append(int(lags[best_idx]))
        per_episode_zero.append(float(lag_means[lags.tolist().index(0)]))
        per_episode_best.append(float(lag_means[best_idx]))
        episode_names.append(item.episode_name)

    per_episode_mean = np.stack(per_episode_mean, axis=0)
    mean_curve = per_episode_mean.mean(axis=0)
    std_curve = per_episode_mean.std(axis=0)
    zero_idx = lags.tolist().index(0)
    global_best_idx = int(mean_curve.argmin())

    return {
        "lags": lags.tolist(),
        "mean_curve": mean_curve.tolist(),
        "std_curve": std_curve.tolist(),
        "global_best_lag": int(lags[global_best_idx]),
        "global_best_mean_error": float(mean_curve[global_best_idx]),
        "zero_lag_mean_error": float(mean_curve[zero_idx]),
        "improvement_vs_zero": float(mean_curve[zero_idx] - mean_curve[global_best_idx]),
        "per_episode_best_lag": per_episode_best_lag,
        "per_episode_zero_lag_error": per_episode_zero,
        "per_episode_best_error": per_episode_best,
        "episodes": episode_names,
    }


def plot_lag_curves(left: dict, right: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    lags = np.asarray(left["lags"])
    for label, stats, color in [("left arm", left, "C0"), ("right arm", right, "C1")]:
        mean_curve = np.asarray(stats["mean_curve"])
        std_curve = np.asarray(stats["std_curve"])
        ax.plot(lags, mean_curve, label=label, linewidth=2, color=color)
        ax.fill_between(lags, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
        ax.scatter([stats["global_best_lag"]], [stats["global_best_mean_error"]], color=color, s=55, zorder=3)
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Lag (frames). Positive means state is shifted later.")
    ax.set_ylabel("Mean |action_xyz - state_xyz| (m)")
    ax.set_title("Action-state tracking error under temporal lag")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_best_lag_hist(left: dict, right: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=180, sharey=True)
    for ax, stats, title, color in [
        (axes[0], left, "Left arm best lag per episode", "C0"),
        (axes[1], right, "Right arm best lag per episode", "C1"),
    ]:
        best_lags = np.asarray(stats["per_episode_best_lag"])
        bins = np.arange(best_lags.min() - 0.5, best_lags.max() + 1.5, 1.0)
        ax.hist(best_lags, bins=bins, color=color, alpha=0.75)
        ax.axvline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Best lag (frames)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Episode count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether state lags behind action in Sparky dataset.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="/mnt/cpfs/datasets/sparky_ugripper_fruit_packing_v2gripper_R8_100",
        type=Path,
    )
    parser.add_argument("--max-lag", type=int, default=30, help="Maximum absolute lag in frames.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save outputs. Defaults to <dataset_root>/visualizations/action_state_lag.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.dataset_root / "visualizations" / "action_state_lag")
    lags = np.arange(-args.max_lag, args.max_lag + 1, dtype=int)

    left = analyze_arm(args.dataset_root, LEFT_ACTION, LEFT_STATE, lags)
    right = analyze_arm(args.dataset_root, RIGHT_ACTION, RIGHT_STATE, lags)

    plot_lag_curves(left, right, output_dir / "action_state_lag_error_curve.png")
    plot_best_lag_hist(left, right, output_dir / "action_state_best_lag_hist.png")

    summary = {"left_arm": left, "right_arm": right}
    summary_path = output_dir / "action_state_lag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved: {output_dir / 'action_state_lag_error_curve.png'}")
    print(f"saved: {output_dir / 'action_state_best_lag_hist.png'}")
    print(f"saved: {summary_path}")
    print(
        "left:",
        left["global_best_lag"],
        left["global_best_mean_error"],
        left["zero_lag_mean_error"],
        left["improvement_vs_zero"],
    )
    print(
        "right:",
        right["global_best_lag"],
        right["global_best_mean_error"],
        right["zero_lag_mean_error"],
        right["improvement_vs_zero"],
    )


if __name__ == "__main__":
    main()
