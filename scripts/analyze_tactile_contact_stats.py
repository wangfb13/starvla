#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pa_dataset
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.video_utils import decode_video_frames


DEFAULT_TACTILE_KEYS = [
    "observation.depth_deformation.tactile_left_left",
    "observation.depth_deformation.tactile_left_right",
    "observation.depth_deformation.tactile_right_left",
    "observation.depth_deformation.tactile_right_right",
]

DEFAULT_TOLERANCE_S = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tactile contact_stats curves across episodes and recommend a top-k ratio."
    )
    parser.add_argument("dataset_root", type=Path, help="LeRobot dataset root.")
    parser.add_argument(
        "--episode-indices",
        type=str,
        default="",
        help="Comma-separated episode indices. Empty means use the first max_episodes episodes.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=8,
        help="How many leading episodes to analyze when --episode-indices is not set. Use 0 for all episodes.",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.02,0.05,0.10",
        help="Comma-separated tactile_contact_topk_ratio candidates.",
    )
    parser.add_argument(
        "--tactile-keys",
        type=str,
        default=",".join(DEFAULT_TACTILE_KEYS),
        help="Comma-separated tactile feature keys.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="Video backend passed to LeRobotDataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save analysis outputs. Defaults to <dataset_root>/meta/tactile_contact_stats.",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=0,
        help="Optional cap for quick inspection. Use 0 for all frames.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N processed episodes. Use 0 to disable.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help="Timestamp tolerance used by video decoding.",
    )
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    for value in values:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"Ratio must be in (0, 1], got {value}")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one non-empty string value.")
    return values


def _parse_episode_indices(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _resolve_episode_indices(dataset_root: Path, requested: list[int], max_episodes: int) -> list[int]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    total_episodes = int(info["total_episodes"])
    if requested:
        for episode_index in requested:
            if episode_index < 0 or episode_index >= total_episodes:
                raise ValueError(f"Episode index out of range: {episode_index} not in [0, {total_episodes})")
        return requested
    if max_episodes == 0:
        return list(range(total_episodes))
    return list(range(min(total_episodes, max_episodes)))


def _resolve_video_extension(feature_key: str) -> str:
    lowered = feature_key.lower()
    if any(token in lowered for token in ("deformation", "shear", "depth")):
        return ".mov"
    return ".mp4"


def _load_episode_rows(dataset_root: Path) -> dict[int, dict[str, object]]:
    table = pa_dataset.dataset(dataset_root / "meta" / "episodes").to_table()
    rows = table.to_pylist()
    return {int(row["episode_index"]): row for row in rows}


def _load_episode_timestamps(
    dataset_root: Path,
    episode_index: int,
    max_frames_per_episode: int,
) -> tuple[np.ndarray, np.ndarray]:
    table = pa_dataset.dataset(dataset_root / "data").to_table(
        columns=["episode_index", "frame_index", "timestamp"],
        filter=(pa_dataset.field("episode_index") == episode_index),
    )
    row_dict = table.to_pydict()
    frame_indices = np.asarray(row_dict["frame_index"], dtype=np.int64)
    timestamps = np.asarray(row_dict["timestamp"], dtype=np.float64)
    order = np.argsort(frame_indices)
    frame_indices = frame_indices[order]
    timestamps = timestamps[order]
    if max_frames_per_episode > 0:
        frame_indices = frame_indices[:max_frames_per_episode]
        timestamps = timestamps[:max_frames_per_episode]
    return frame_indices, timestamps


def _resolve_episode_video_path(dataset_root: Path, episode_row: dict[str, object], feature_key: str) -> Path:
    chunk_index = int(episode_row[f"videos/{feature_key}/chunk_index"])
    file_index = int(episode_row[f"videos/{feature_key}/file_index"])
    ext = _resolve_video_extension(feature_key)
    return dataset_root / "videos" / feature_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}{ext}"


def _compute_contact_curves(frames: torch.Tensor, ratios: list[float]) -> dict[str, np.ndarray]:
    frames = torch.as_tensor(frames, dtype=torch.float32)
    if frames.ndim != 4:
        raise ValueError(f"Expected decoded tactile frames rank-4 [T,C,H,W], got {tuple(frames.shape)}")
    if frames.shape[1] > 1:
        frames = frames.mean(dim=1, keepdim=True)
    flat = frames.flatten(start_dim=2).squeeze(1)
    sorted_values = torch.sort(flat, dim=1, descending=True).values
    cumulative = torch.cumsum(sorted_values, dim=1)
    total = int(sorted_values.shape[1])
    curves: dict[str, np.ndarray] = {}
    for ratio in ratios:
        k = max(1, math.ceil(total * ratio))
        scores = cumulative[:, k - 1] / float(k)
        curves[f"{ratio:.5f}"] = scores.detach().cpu().numpy().astype(np.float32)
    return curves


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin + 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _plot_episode_curves(
    episode_index: int,
    frame_indices: np.ndarray,
    sensor_curves: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    ratios = list(next(iter(sensor_curves.values())).keys())
    fig, axes = plt.subplots(len(sensor_curves), 2, figsize=(15, 3.6 * len(sensor_curves)), dpi=180, sharex=True)
    if len(sensor_curves) == 1:
        axes = np.asarray([axes])
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(ratios)))

    for row, (sensor_key, ratio_curves) in enumerate(sensor_curves.items()):
        label = sensor_key.split(".")[-1]
        ax_raw = axes[row, 0]
        ax_norm = axes[row, 1]
        for color, ratio in zip(colors, ratios, strict=True):
            values = ratio_curves[ratio]
            ax_raw.plot(frame_indices, values, color=color, linewidth=1.6, label=f"ratio={float(ratio):.3f}")
            ax_norm.plot(frame_indices, _normalize_curve(values), color=color, linewidth=1.6, label=f"ratio={float(ratio):.3f}")
        ax_raw.set_title(f"{label} raw contact score")
        ax_norm.set_title(f"{label} normalized trend")
        ax_raw.grid(True, alpha=0.2)
        ax_norm.grid(True, alpha=0.2)
        ax_raw.set_ylabel("score")
        ax_norm.set_ylabel("norm score")

    axes[-1, 0].set_xlabel("frame index")
    axes[-1, 1].set_xlabel("frame index")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Tactile contact_stats curves for episode {episode_index}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _compute_ratio_metrics(
    curves_by_ratio: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for ratio, curves in curves_by_ratio.items():
        spans = []
        jitters = []
        peak_to_median = []
        for curve in curves:
            spans.append(float(np.percentile(curve, 95) - np.percentile(curve, 5)))
            if len(curve) > 1:
                jitters.append(float(np.median(np.abs(np.diff(curve)))))
            else:
                jitters.append(0.0)
            peak_to_median.append(float(np.percentile(curve, 95) / max(np.median(curve), 1e-6)))
        span = float(np.mean(spans))
        jitter = float(np.mean(jitters))
        peak_ratio = float(np.mean(peak_to_median))
        score = span / max(jitter, 1e-6)
        metrics[ratio] = {
            "mean_span_p95_p5": span,
            "mean_jitter_median_abs_diff": jitter,
            "mean_peak_to_median_ratio": peak_ratio,
            "stability_score": score,
        }
    return metrics


def _plot_ratio_summary(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    ratios = list(metrics.keys())
    x = np.arange(len(ratios))
    spans = np.asarray([metrics[ratio]["mean_span_p95_p5"] for ratio in ratios], dtype=np.float32)
    jitters = np.asarray([metrics[ratio]["mean_jitter_median_abs_diff"] for ratio in ratios], dtype=np.float32)
    scores = np.asarray([metrics[ratio]["stability_score"] for ratio in ratios], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=180)
    axes[0].bar(x, spans, color="C0", alpha=0.85)
    axes[0].set_title("Dynamic span")
    axes[0].set_ylabel("p95 - p5")
    axes[1].bar(x, jitters, color="C1", alpha=0.85)
    axes[1].set_title("Frame-to-frame jitter")
    axes[1].set_ylabel("median |diff|")
    axes[2].bar(x, scores, color="C2", alpha=0.85)
    axes[2].set_title("Span / jitter")
    axes[2].set_ylabel("stability score")
    for ax in axes:
        ax.set_xticks(x, [f"{float(ratio):.3f}" for ratio in ratios])
        ax.set_xlabel("top-k ratio")
        ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    dataset_root: Path,
    ratios: list[float],
    episode_indices: list[int],
    tactile_keys: list[str],
    metrics: dict[str, dict[str, float]],
    recommendation: str,
) -> None:
    lines = [
        "# Tactile Contact Stats Analysis",
        "",
        f"- dataset_root: `{dataset_root}`",
        f"- analyzed_episodes: `{episode_indices}`",
        f"- tactile_keys: `{tactile_keys}`",
        f"- ratios: `{[round(r, 5) for r in ratios]}`",
        f"- recommended_ratio: `{recommendation}`",
        "",
        "## Ratio Summary",
        "",
        "| ratio | mean_span_p95_p5 | mean_jitter_median_abs_diff | mean_peak_to_median_ratio | stability_score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ratio, ratio_metrics in metrics.items():
        lines.append(
            f"| {float(ratio):.3f} | "
            f"{ratio_metrics['mean_span_p95_p5']:.6f} | "
            f"{ratio_metrics['mean_jitter_median_abs_diff']:.6f} | "
            f"{ratio_metrics['mean_peak_to_median_ratio']:.6f} | "
            f"{ratio_metrics['stability_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `mean_span_p95_p5` 越大，说明触觉曲线更有动态范围，更容易区分抓到和没抓到。",
            "- `mean_jitter_median_abs_diff` 越小，说明帧间抖动更少。",
            "- `stability_score=span/jitter` 越大，通常越适合先拿来做轻量接触信号。",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    ratios = _parse_float_list(args.ratios)
    tactile_keys = _parse_str_list(args.tactile_keys)
    episode_indices = _resolve_episode_indices(
        args.dataset_root,
        _parse_episode_indices(args.episode_indices),
        args.max_episodes,
    )
    output_dir = args.output_dir or (args.dataset_root / "meta" / "tactile_contact_stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_episode_scores: dict[int, dict[str, dict[str, list[float]]]] = {}
    frame_indices_by_episode: dict[int, list[int]] = defaultdict(list)
    curves_by_ratio: dict[str, list[np.ndarray]] = defaultdict(list)
    present_keys: set[str] = set()
    episode_rows = _load_episode_rows(args.dataset_root)

    for episode_counter, episode_index in enumerate(episode_indices, start=1):
        if episode_index not in episode_rows:
            raise RuntimeError(f"Episode {episode_index} not found in meta/episodes.")
        frame_indices, timestamps = _load_episode_timestamps(
            args.dataset_root,
            episode_index,
            args.max_frames_per_episode,
        )
        frame_indices_by_episode[episode_index] = frame_indices.tolist()
        episode_store = per_episode_scores.setdefault(
            episode_index,
            {key: {f"{ratio:.5f}": [] for ratio in ratios} for key in tactile_keys},
        )
        episode_row = episode_rows[episode_index]
        for key in tactile_keys:
            row_key = f"videos/{key}/file_index"
            if row_key not in episode_row:
                continue
            present_keys.add(key)
            video_path = _resolve_episode_video_path(args.dataset_root, episode_row, key)
            frames = decode_video_frames(
                video_path,
                timestamps.tolist(),
                tolerance_s=args.tolerance_s,
                backend=args.video_backend,
            )
            ratio_curves = _compute_contact_curves(frames, ratios)
            for ratio_key, curve in ratio_curves.items():
                episode_store[key][ratio_key] = curve.tolist()
        if args.progress_every > 0 and episode_counter % args.progress_every == 0:
            print(
                f"processed_episodes={episode_counter}/{len(episode_indices)} "
                f"episode_index={episode_index} frames={len(frame_indices)}"
            )

    if not present_keys:
        raise RuntimeError(f"None of the requested tactile keys were found: {tactile_keys}")

    tactile_keys = [key for key in tactile_keys if key in present_keys]

    for episode_index, sensor_map in per_episode_scores.items():
        frame_indices = np.asarray(frame_indices_by_episode[episode_index], dtype=np.int64)
        plot_ready: dict[str, dict[str, np.ndarray]] = {}
        for sensor_key in tactile_keys:
            ratio_curves: dict[str, np.ndarray] = {}
            for ratio_key, values in sensor_map[sensor_key].items():
                curve = np.asarray(values, dtype=np.float32)
                ratio_curves[ratio_key] = curve
                curves_by_ratio[ratio_key].append(curve)
            plot_ready[sensor_key] = ratio_curves
        _plot_episode_curves(
            episode_index,
            frame_indices,
            plot_ready,
            output_dir / f"episode_{episode_index:06d}_contact_stats.png",
        )

    metrics = _compute_ratio_metrics(curves_by_ratio)
    recommendation = max(metrics.items(), key=lambda item: item[1]["stability_score"])[0]

    _plot_ratio_summary(metrics, output_dir / "ratio_summary.png")
    summary = {
        "dataset_root": str(args.dataset_root),
        "episode_indices": episode_indices,
        "tactile_keys": tactile_keys,
        "ratios": ratios,
        "metrics": metrics,
        "recommended_ratio": float(recommendation),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(
        output_dir / "report.md",
        args.dataset_root,
        ratios,
        episode_indices,
        tactile_keys,
        metrics,
        recommendation,
    )

    print(f"saved: {output_dir}")
    print(f"recommended_ratio={float(recommendation):.5f}")
    for ratio_key, ratio_metrics in metrics.items():
        print(
            f"ratio={float(ratio_key):.5f} "
            f"span={ratio_metrics['mean_span_p95_p5']:.6f} "
            f"jitter={ratio_metrics['mean_jitter_median_abs_diff']:.6f} "
            f"score={ratio_metrics['stability_score']:.6f}"
        )


if __name__ == "__main__":
    main()
