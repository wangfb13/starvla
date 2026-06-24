#!/usr/bin/env python3

"""Offline chunk comparison for PI05 / PI05-Full state-as-action checkpoints.

Default mode keeps the original behavior:
- evaluate one episode
- auto-select the frame with the largest future xyz span in one chunk
- save full-chunk plots and a JSON summary

Stats mode is enabled with ``--stats-num-samples`` and adds:
- multi-frame evaluation
- random / uniform / hardest-frame sampling
- per-horizon metrics
- separate xyz / rot / gripper / all-valid MAE
- a hold-start baseline for context
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

try:
    import pyarrow.dataset as pa_dataset
except ImportError:
    pa_dataset = None

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
LEFT_ROT = tuple(range(3, 9))
RIGHT_ROT = tuple(range(13, 19))
GRIPPER = (9, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=None, help="Manually pick a frame instead of auto-selecting.")
    parser.add_argument(
        "--fast-select-from-parquet",
        action="store_true",
        help="Use raw parquet timestamps plus sidecar GT actions to speed up frame selection without changing GT source.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-backend", type=str, default="pyav")

    parser.add_argument(
        "--stats-num-samples",
        type=int,
        default=0,
        help="Enable multi-frame stats mode when > 0.",
    )
    parser.add_argument(
        "--stats-sample-mode",
        type=str,
        default="random",
        choices=("random", "uniform", "hardest"),
        help="Frame sampling strategy in stats mode.",
    )
    parser.add_argument(
        "--stats-horizons",
        type=str,
        default="1,10,50",
        help="Comma-separated horizons to evaluate in stats mode. Values are clipped to chunk size.",
    )
    parser.add_argument("--stats-seed", type=int, default=0)
    parser.add_argument(
        "--stats-sample-list-path",
        type=Path,
        default=None,
        help="Optional JSON file that stores sampled dataset indices for exact cross-checkpoint reuse.",
    )
    parser.add_argument(
        "--stats-progress-every",
        type=int,
        default=10,
        help="Print progress every N samples in stats mode.",
    )
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


def _load_action_mask(sample: dict, action_dim: int) -> np.ndarray:
    action_mask = sample.get("action_valid_mask")
    if action_mask is None:
        return np.ones((action_dim,), dtype=np.float32)
    return torch.as_tensor(action_mask).detach().cpu().numpy().astype(np.float32)[:action_dim]


def _get_xyz_dims(action_mask: np.ndarray) -> list[int]:
    dims: list[int] = []
    if action_mask[:, LEFT_XYZ].mean() > 0.5:
        dims.extend(LEFT_XYZ)
    if action_mask[:, RIGHT_XYZ].mean() > 0.5:
        dims.extend(RIGHT_XYZ)
    return dims


def _valid_start_indices(num_frames: int, chunk_horizon: int) -> np.ndarray:
    valid_count = num_frames - chunk_horizon + 1
    if valid_count <= 0:
        raise RuntimeError("Episode is shorter than the model chunk horizon.")
    return np.arange(valid_count, dtype=np.int64)


def _compute_window_scores(actions: np.ndarray, xyz_dims: list[int], chunk_horizon: int) -> np.ndarray:
    start_indices = _valid_start_indices(actions.shape[0], chunk_horizon)
    scores = np.empty(len(start_indices), dtype=np.float32)
    for idx, start in enumerate(start_indices):
        xyz_chunk = actions[start : start + chunk_horizon][:, xyz_dims]
        scores[idx] = float(np.ptp(xyz_chunk, axis=0).sum())
    return scores


def _select_frame(actions: np.ndarray, xyz_dims: list[int], chunk_horizon: int, frame_index: int | None) -> tuple[int, float]:
    if frame_index is not None:
        return frame_index, float("nan")
    scores = _compute_window_scores(actions, xyz_dims, chunk_horizon)
    best_start = int(np.argmax(scores))
    return best_start, float(scores[best_start])


def _parse_horizons(raw: str, chunk_horizon: int) -> list[int]:
    horizons = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        horizons.append(int(part))
    if not horizons:
        raise ValueError("No horizons parsed from --stats-horizons.")
    clipped = sorted({max(1, min(chunk_horizon, h)) for h in horizons})
    return clipped


def _select_stats_starts(
    start_indices: np.ndarray,
    scores: np.ndarray,
    num_samples: int,
    sample_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = min(num_samples, len(start_indices))
    if sample_mode == "hardest":
        order = np.argsort(scores)[::-1][:num_samples]
    elif sample_mode == "uniform":
        order = np.linspace(0, len(start_indices) - 1, num_samples, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        order = rng.choice(len(start_indices), size=num_samples, replace=False)
        order.sort()
    return start_indices[order], scores[order]


def _load_or_create_stats_sample_list(
    sample_list_path: Path | None,
    start_indices: np.ndarray,
    scores: np.ndarray,
    num_samples: int,
    sample_mode: str,
    seed: int,
    dataset_root: Path,
    episode_index: int,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_list_path is None:
        return _select_stats_starts(start_indices, scores, num_samples, sample_mode, seed)

    sample_list_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_list_path.exists():
        payload = json.loads(sample_list_path.read_text(encoding="utf-8"))
        saved_dataset = payload.get("dataset_root")
        saved_episode = payload.get("episode_index")
        if saved_dataset != str(dataset_root) or int(saved_episode) != int(episode_index):
            raise ValueError(
                "stats sample list does not match current dataset/episode: "
                f"{sample_list_path}"
            )
        saved_starts = np.asarray(payload["dataset_indices"], dtype=np.int64)
        score_map = {int(s): float(v) for s, v in zip(start_indices.tolist(), scores.tolist())}
        if not set(saved_starts.tolist()).issubset(set(score_map.keys())):
            raise ValueError(
                "stats sample list contains indices invalid for the current episode/chunk setup: "
                f"{sample_list_path}"
            )
        loaded_scores = np.asarray([score_map[int(idx)] for idx in saved_starts.tolist()], dtype=np.float32)
        print(f"[stats] loaded shared sample list: {sample_list_path}")
        return saved_starts, loaded_scores

    sampled_starts, sampled_scores = _select_stats_starts(start_indices, scores, num_samples, sample_mode, seed)
    payload = {
        "dataset_root": str(dataset_root),
        "episode_index": int(episode_index),
        "stats_num_samples": int(len(sampled_starts)),
        "stats_sample_mode": sample_mode,
        "stats_seed": int(seed),
        "dataset_indices": sampled_starts.tolist(),
        "frame_indices": [int(frame_indices[idx]) for idx in sampled_starts.tolist()],
        "timestamps": [float(timestamps[idx]) for idx in sampled_starts.tolist()],
        "selection_scores_xyz_span_sum": sampled_scores.tolist(),
    }
    sample_list_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stats] saved shared sample list: {sample_list_path}")
    return sampled_starts, sampled_scores


def _load_episode_arrays_fast(
    dataset_root: Path,
    sidecar_path: Path,
    episode_index: int,
    action_dim: int,
    fallback_action_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pa_dataset is None:
        raise RuntimeError("pyarrow is required for --fast-select-from-parquet, but it is not installed.")

    raw_table = pa_dataset.dataset(str(dataset_root / "data"), format="parquet").to_table(
        columns=["index", "episode_index", "frame_index", "timestamp"],
        filter=(pa_dataset.field("episode_index") == episode_index),
    )
    sidecar_ds = pa_dataset.dataset(str(sidecar_path), format="parquet")
    sidecar_columns = ["index", "episode_index", "frame_index", "action"]
    if "action_valid_mask" in sidecar_ds.schema.names:
        sidecar_columns.append("action_valid_mask")
    sidecar_table = sidecar_ds.to_table(
        columns=sidecar_columns,
        filter=(pa_dataset.field("episode_index") == episode_index),
    )

    raw_rows = raw_table.to_pydict()
    sidecar_rows = sidecar_table.to_pydict()
    if not raw_rows["index"]:
        raise RuntimeError(f"No raw parquet rows found for episode {episode_index} in {dataset_root}.")
    if not sidecar_rows["index"]:
        raise RuntimeError(f"No sidecar rows found for episode {episode_index} in {sidecar_path}.")

    raw_indices = np.asarray(raw_rows["index"], dtype=np.int64)
    raw_episode_ids = np.asarray(raw_rows["episode_index"], dtype=np.int64)
    raw_frame_indices = np.asarray(raw_rows["frame_index"], dtype=np.int64)
    raw_timestamps = np.asarray(raw_rows["timestamp"], dtype=np.float32)
    raw_order = np.argsort(raw_indices, kind="stable")

    sidecar_indices = np.asarray(sidecar_rows["index"], dtype=np.int64)
    sidecar_episode_ids = np.asarray(sidecar_rows["episode_index"], dtype=np.int64)
    sidecar_frame_indices = np.asarray(sidecar_rows["frame_index"], dtype=np.int64)
    sidecar_actions = np.asarray([row[:action_dim] for row in sidecar_rows["action"]], dtype=np.float32)
    sidecar_order = np.argsort(sidecar_indices, kind="stable")

    raw_indices = raw_indices[raw_order]
    raw_episode_ids = raw_episode_ids[raw_order]
    raw_frame_indices = raw_frame_indices[raw_order]
    raw_timestamps = raw_timestamps[raw_order]

    sidecar_indices = sidecar_indices[sidecar_order]
    sidecar_episode_ids = sidecar_episode_ids[sidecar_order]
    sidecar_frame_indices = sidecar_frame_indices[sidecar_order]
    sidecar_actions = sidecar_actions[sidecar_order]

    if len(raw_indices) != len(sidecar_indices):
        raise RuntimeError(
            "Raw parquet and sidecar row counts do not match for "
            f"episode {episode_index}: raw={len(raw_indices)} sidecar={len(sidecar_indices)}"
        )
    if not np.array_equal(raw_indices, sidecar_indices):
        raise RuntimeError(
            "Raw parquet and sidecar absolute indices do not align for "
            f"episode {episode_index}. First mismatch raw={raw_indices[:5].tolist()} "
            f"sidecar={sidecar_indices[:5].tolist()}"
        )
    if not np.array_equal(raw_episode_ids, sidecar_episode_ids):
        raise RuntimeError(f"Episode ids do not align between raw parquet and sidecar for episode {episode_index}.")
    if not np.array_equal(raw_frame_indices, sidecar_frame_indices):
        raise RuntimeError(
            "Frame indices do not align between raw parquet and sidecar for "
            f"episode {episode_index}."
        )

    actions = sidecar_actions
    frame_indices = sidecar_frame_indices
    timestamps = raw_timestamps
    if "action_valid_mask" in sidecar_rows:
        sidecar_masks = np.asarray([row[:action_dim] for row in sidecar_rows["action_valid_mask"]], dtype=np.float32)
        masks = sidecar_masks[sidecar_order]
    else:
        masks = np.tile(fallback_action_mask[None, :], (actions.shape[0], 1))
    return actions, masks, frame_indices, timestamps


def _load_episode_arrays(
    dataset: LeRobotDataset,
    dataset_root: Path,
    sidecar_path: Path,
    episode_index: int,
    action_dim: int,
    action_mask0: np.ndarray,
    fast_select_from_parquet: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fast_select_from_parquet:
        return _load_episode_arrays_fast(dataset_root, sidecar_path, episode_index, action_dim, action_mask0)

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
    return actions, masks, frame_indices, timestamps


def _validate_loaded_episode_arrays(
    dataset: LeRobotDataset,
    actions: np.ndarray,
    masks: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    action_dim: int,
) -> None:
    if len(actions) != len(dataset):
        raise RuntimeError(
            f"Loaded episode arrays length mismatch: arrays={len(actions)} dataset={len(dataset)}"
        )

    probe_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for probe_idx in probe_indices:
        item = dataset[probe_idx]
        item_action = item["action"].detach().cpu().numpy().astype(np.float32)[:action_dim]
        item_mask = item.get("action_valid_mask")
        if item_mask is None:
            item_mask_np = np.ones((action_dim,), dtype=np.float32)
        else:
            item_mask_np = torch.as_tensor(item_mask).detach().cpu().numpy().astype(np.float32)[:action_dim]
        item_frame_index = int(_to_scalar(item["frame_index"]))
        item_timestamp = float(_to_scalar(item["timestamp"]))

        if item_frame_index != int(frame_indices[probe_idx]):
            raise RuntimeError(
                "Loaded frame_index does not match dataset item at relative index "
                f"{probe_idx}: loaded={int(frame_indices[probe_idx])} dataset={item_frame_index}"
            )
        if not np.isclose(item_timestamp, float(timestamps[probe_idx]), atol=1e-5):
            raise RuntimeError(
                "Loaded timestamp does not match dataset item at relative index "
                f"{probe_idx}: loaded={float(timestamps[probe_idx])} dataset={item_timestamp}"
            )
        if not np.allclose(item_action, actions[probe_idx], atol=1e-6):
            diff = np.abs(item_action - actions[probe_idx])
            raise RuntimeError(
                "Loaded GT action does not match dataset sidecar action at relative index "
                f"{probe_idx}: max_diff={float(diff.max())}"
            )
        if not np.allclose(item_mask_np, masks[probe_idx], atol=1e-6):
            diff = np.abs(item_mask_np - masks[probe_idx])
            raise RuntimeError(
                "Loaded action_valid_mask does not match dataset sidecar mask at relative index "
                f"{probe_idx}: max_diff={float(diff.max())}"
            )


def _build_metric_groups(mask_episode: np.ndarray, action_dim: int) -> dict[str, list[int]]:
    valid_dims = [i for i in range(action_dim) if float(mask_episode[:, i].mean()) > 0.5]
    xyz_dims = [i for i in (*LEFT_XYZ, *RIGHT_XYZ) if i in valid_dims]
    rot_dims = [i for i in (*LEFT_ROT, *RIGHT_ROT) if i in valid_dims]
    gripper_dims = [i for i in GRIPPER if i in valid_dims]
    groups = {
        "all_valid": valid_dims,
        "xyz": xyz_dims,
        "rot6d": rot_dims,
        "gripper": gripper_dims,
    }
    return {name: dims for name, dims in groups.items() if dims}


def _masked_mae(pred: np.ndarray, gt: np.ndarray, mask_bool: np.ndarray) -> float:
    values = np.abs(pred - gt)[mask_bool]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def _evaluate_chunk_metrics(
    pred_chunk_abs: np.ndarray,
    gt_chunk: np.ndarray,
    mask_chunk: np.ndarray,
    groups: dict[str, list[int]],
    horizons: list[int],
) -> dict[str, dict[str, dict[str, float]]]:
    baseline_chunk = np.repeat(gt_chunk[:1], gt_chunk.shape[0], axis=0)
    results: dict[str, dict[str, dict[str, float]]] = {}

    for horizon in horizons:
        gt_h = gt_chunk[:horizon]
        pred_h = pred_chunk_abs[:horizon]
        base_h = baseline_chunk[:horizon]
        mask_h = mask_chunk[:horizon] > 0.5
        horizon_metrics: dict[str, dict[str, float]] = {}
        for group_name, dims in groups.items():
            pred_group = pred_h[:, dims]
            gt_group = gt_h[:, dims]
            base_group = base_h[:, dims]
            group_mask = mask_h[:, dims]
            model_mae = _masked_mae(pred_group, gt_group, group_mask)
            baseline_mae = _masked_mae(base_group, gt_group, group_mask)
            if math.isfinite(model_mae) and math.isfinite(baseline_mae):
                better_than_baseline = float(model_mae < baseline_mae)
                rel_improvement = float((baseline_mae - model_mae) / max(baseline_mae, 1e-8))
            else:
                better_than_baseline = float("nan")
                rel_improvement = float("nan")
            horizon_metrics[group_name] = {
                "model_mae": model_mae,
                "baseline_hold_start_mae": baseline_mae,
                "relative_improvement_vs_baseline": rel_improvement,
                "model_better_than_baseline": better_than_baseline,
            }
        results[str(horizon)] = horizon_metrics
    return results


def _summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _aggregate_stats(
    sample_metrics: list[dict],
    horizons: list[int],
    groups: dict[str, list[int]],
) -> dict[str, dict[str, dict[str, float]]]:
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for horizon in horizons:
        h_key = str(horizon)
        horizon_summary: dict[str, dict[str, float]] = {}
        for group_name in groups:
            model_vals = [sample["metrics"][h_key][group_name]["model_mae"] for sample in sample_metrics]
            baseline_vals = [sample["metrics"][h_key][group_name]["baseline_hold_start_mae"] for sample in sample_metrics]
            improv_vals = [sample["metrics"][h_key][group_name]["relative_improvement_vs_baseline"] for sample in sample_metrics]
            better_vals = [sample["metrics"][h_key][group_name]["model_better_than_baseline"] for sample in sample_metrics]
            horizon_summary[group_name] = {
                "model_mae": _summarize_values(model_vals),
                "baseline_hold_start_mae": _summarize_values(baseline_vals),
                "relative_improvement_vs_baseline": _summarize_values(improv_vals),
                "model_better_than_baseline_rate": _summarize_values(better_vals)["mean"],
            }
        aggregates[h_key] = horizon_summary
    return aggregates


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


def _plot_stats_summary(
    output_path: Path,
    horizons: list[int],
    aggregates: dict[str, dict[str, dict[str, float]]],
    groups: dict[str, list[int]],
) -> None:
    group_names = list(groups.keys())
    pretty_names = {
        "all_valid": "all",
        "xyz": "xyz",
        "rot6d": "rot6d",
        "gripper": "gripper",
    }
    xs = np.arange(len(horizons), dtype=np.float32)
    width = 0.18 if len(group_names) >= 4 else 0.22

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=180)
    metric_specs = [
        ("model_mae", "Model MAE"),
        ("baseline_hold_start_mae", "Baseline MAE"),
        ("relative_improvement_vs_baseline", "Relative Improvement vs Baseline"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        for idx, group_name in enumerate(group_names):
            offsets = xs + (idx - (len(group_names) - 1) / 2.0) * width
            means = [
                aggregates[str(horizon)][group_name][metric_key]["mean"]
                for horizon in horizons
            ]
            ax.bar(offsets, means, width=width, label=pretty_names.get(group_name, group_name))
        ax.set_xticks(xs)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_title(title)
        ax.set_xlabel("horizon")
        ax.grid(True, axis="y", alpha=0.25)
        if metric_key == "relative_improvement_vs_baseline":
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)

    axes[0].set_ylabel("value")
    axes[0].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _predict_chunk(
    dataset: LeRobotDataset,
    index: int,
    preprocessor,
    postprocessor,
    policy,
) -> np.ndarray:
    raw_item = dataset[index]
    processed = preprocessor(raw_item)
    with torch.inference_mode():
        pred_chunk = policy.predict_action_chunk(processed)
        pred_chunk_abs = postprocessor(pred_chunk)[0].detach().cpu().numpy().astype(np.float32)
    return pred_chunk_abs


def _run_single_frame_eval(
    args: argparse.Namespace,
    pretrained_dir: Path,
    dataset: LeRobotDataset,
    preprocessor,
    postprocessor,
    policy,
    actions: np.ndarray,
    masks: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    action_dim: int,
    chunk_horizon: int,
) -> None:
    xyz_dims = _get_xyz_dims(masks)
    if not xyz_dims:
        raise RuntimeError("No valid xyz dimensions found in this episode.")
    xyz_names = [ACTION_NAMES[i] for i in xyz_dims]

    selected_start, selection_score = _select_frame(actions, xyz_dims, chunk_horizon, args.frame_index)
    gt_chunk = actions[selected_start : selected_start + chunk_horizon]
    mask_chunk = masks[selected_start : selected_start + chunk_horizon]
    pred_chunk_abs = _predict_chunk(dataset, selected_start, preprocessor, postprocessor, policy)

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
        "mode": "single_frame",
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


def _run_stats_eval(
    args: argparse.Namespace,
    pretrained_dir: Path,
    dataset: LeRobotDataset,
    preprocessor,
    postprocessor,
    policy,
    actions: np.ndarray,
    masks: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    action_dim: int,
    chunk_horizon: int,
) -> None:
    if args.frame_index is not None:
        raise ValueError("--frame-index is only supported in single-frame mode.")

    xyz_dims = _get_xyz_dims(masks)
    if not xyz_dims:
        raise RuntimeError("No valid xyz dimensions found in this episode.")

    groups = _build_metric_groups(masks, action_dim)
    horizons = _parse_horizons(args.stats_horizons, chunk_horizon)
    start_indices = _valid_start_indices(actions.shape[0], chunk_horizon)
    scores = _compute_window_scores(actions, xyz_dims, chunk_horizon)
    sampled_starts, sampled_scores = _load_or_create_stats_sample_list(
        args.stats_sample_list_path,
        start_indices,
        scores,
        args.stats_num_samples,
        args.stats_sample_mode,
        args.stats_seed,
        args.dataset_root,
        args.episode_index,
        frame_indices,
        timestamps,
    )

    sample_summaries: list[dict] = []
    for sample_idx, (start, score) in enumerate(zip(sampled_starts.tolist(), sampled_scores.tolist()), start=1):
        pred_chunk_abs = _predict_chunk(dataset, start, preprocessor, postprocessor, policy)
        gt_chunk = actions[start : start + chunk_horizon]
        mask_chunk = masks[start : start + chunk_horizon]
        metrics = _evaluate_chunk_metrics(pred_chunk_abs, gt_chunk, mask_chunk, groups, horizons)
        sample_summaries.append(
            {
                "dataset_index": int(start),
                "frame_index": int(frame_indices[start]),
                "timestamp": float(timestamps[start]),
                "selection_score_xyz_span_sum": float(score),
                "metrics": metrics,
            }
        )
        if args.stats_progress_every > 0 and (
            sample_idx == len(sampled_starts) or sample_idx % args.stats_progress_every == 0
        ):
            print(f"[stats] evaluated {sample_idx}/{len(sampled_starts)} samples")

    aggregates = _aggregate_stats(sample_summaries, horizons, groups)
    plot_path = args.output_dir / "stats_mae_by_horizon.png"
    summary_path = args.output_dir / "stats_summary.json"
    _plot_stats_summary(plot_path, horizons, aggregates, groups)

    summary = {
        "mode": "stats",
        "policy_path": str(pretrained_dir),
        "dataset_root": str(args.dataset_root),
        "sidecar_path": str(args.sidecar_path),
        "episode_index": args.episode_index,
        "chunk_horizon": chunk_horizon,
        "action_dim": action_dim,
        "sample_count": len(sample_summaries),
        "stats_sample_mode": args.stats_sample_mode,
        "stats_seed": args.stats_seed,
        "stats_sample_list_path": str(args.stats_sample_list_path) if args.stats_sample_list_path else None,
        "horizons": horizons,
        "groups": {name: {"dims": dims, "names": [ACTION_NAMES[i] for i in dims]} for name, dims in groups.items()},
        "plot_path": str(plot_path),
        "aggregates": aggregates,
        "samples": sample_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    preview = {
        "mode": "stats",
        "sample_count": len(sample_summaries),
        "horizons": horizons,
        "plot_path": str(plot_path),
        "summary_path": str(summary_path),
        "xyz_mae_means": {
            str(h): aggregates[str(h)]["xyz"]["model_mae"]["mean"]
            for h in horizons
            if "xyz" in aggregates[str(h)]
        },
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    print(f"saved: {plot_path}")
    print(f"saved: {summary_path}")


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
    action_mask0 = _load_action_mask(sample0, action_dim)

    actions, masks, frame_indices, timestamps = _load_episode_arrays(
        dataset,
        args.dataset_root,
        args.sidecar_path,
        args.episode_index,
        action_dim,
        action_mask0,
        args.fast_select_from_parquet,
    )
    _validate_loaded_episode_arrays(dataset, actions, masks, frame_indices, timestamps, action_dim)

    if args.stats_num_samples > 0:
        _run_stats_eval(
            args,
            pretrained_dir,
            dataset,
            preprocessor,
            postprocessor,
            policy,
            actions,
            masks,
            frame_indices,
            timestamps,
            action_dim,
            chunk_horizon,
        )
    else:
        _run_single_frame_eval(
            args,
            pretrained_dir,
            dataset,
            preprocessor,
            postprocessor,
            policy,
            actions,
            masks,
            frame_indices,
            timestamps,
            action_dim,
            chunk_horizon,
        )


if __name__ == "__main__":
    main()
