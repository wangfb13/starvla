#!/usr/bin/env python3
"""Offline StarVLA prediction-vs-clean-label evaluation at one shared event.

The script deliberately keeps the clean demonstration action before any
flow-matching noise is created.  Inference receives a copy of the processed
batch with the ACTION key removed.  Two (or more) current frames from one
episode are compared at the same original event frame, not at chunk endpoints.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(240, repeat=True)
print("[event-eval] importing scientific/model dependencies", flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.relative_action_processor import to_keyframe_relative_actions
from lerobot.utils.constants import ACTION, OBS_STATE


MODEL_CAMERA_KEYS = {
    "observation.images.cam_left_undist",
    "observation.images.cam_right_undist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint pretrained_model directory.")
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--relative-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--current-frames",
        default=None,
        help="Comma-separated original frame numbers. Default uses event-32,event-10.",
    )
    parser.add_argument("--event-frame", type=int, default=None, help="Known original event frame; otherwise detect from label.")
    parser.add_argument("--auto-anchor-offsets", default="50,32,10")
    parser.add_argument("--close-threshold", type=float, default=0.55)
    parser.add_argument("--stable-steps", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def to_int(value) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def stable_close_index(
    gripper: np.ndarray,
    threshold: float,
    stable_steps: int,
    *,
    require_open_before: bool,
) -> int | None:
    values = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if stable_steps < 1 or len(values) < stable_steps:
        return None
    closed = values <= threshold
    hits = np.convolve(closed.astype(np.int16), np.ones(stable_steps, dtype=np.int16), mode="valid")
    for index in np.flatnonzero(hits == stable_steps):
        if not require_open_before or np.any(values[:index] > threshold):
            return int(index)
    return None


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12, None)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / np.clip(np.linalg.norm(a2_orth, axis=-1, keepdims=True), 1e-12, None)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-1)


def rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_rot = rot6d_to_matrix(pred[..., 3:9])
    target_rot = rot6d_to_matrix(target[..., 3:9])
    relative = np.einsum("...ji,...jk->...ik", target_rot, pred_rot)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def filtered_info(dataset_root: Path, output_dir: Path) -> Path:
    raw_info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    scalar_keys = {
        "observation.state",
        "action",
        "score",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    keep = MODEL_CAMERA_KEYS | scalar_keys
    missing = MODEL_CAMERA_KEYS - set(raw_info["features"])
    if missing:
        raise RuntimeError(f"Missing required training cameras: {sorted(missing)}")
    raw_info["features"] = {key: value for key, value in raw_info["features"].items() if key in keep}
    path = output_dir / "info_wrist_undist_offline.json"
    path.write_text(json.dumps(raw_info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_dataset(args: argparse.Namespace):
    info_path = filtered_info(args.dataset_root, args.output_dir)
    train_cfg = TrainPipelineConfig.from_pretrained(args.checkpoint / "train_config.json")
    train_cfg.dataset.root = str(args.dataset_root)
    train_cfg.dataset.repo_id = args.dataset_root.name
    train_cfg.dataset.roots_file = None
    train_cfg.dataset.info_path = str(info_path)
    train_cfg.dataset.sidecar_root = None
    train_cfg.dataset.sidecar_path = str(args.sidecar)
    train_cfg.dataset.relative_stats_path = str(args.relative_stats)
    train_cfg.dataset.video_backend = "pyav"
    train_cfg.dataset.episodes = None
    train_cfg.num_workers = 0
    train_cfg.policy.device = args.device
    train_cfg.policy.base_vlm = str(args.base_vlm)
    train_cfg.policy.use_amp = False
    return make_dataset(train_cfg)


def load_episode_labels(sidecar: Path, episode_index: int) -> pd.DataFrame:
    frame = pd.read_parquet(sidecar, columns=["index", "episode_index", "frame_index", "action"])
    frame = frame[frame.episode_index == episode_index].sort_values("frame_index").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"Episode {episode_index} not found in {sidecar}")
    return frame


def dataset_index_map(dataset_root: Path, episode_index: int) -> dict[int, int]:
    frame = pd.read_parquet(
        dataset_root / "data",
        columns=["index", "episode_index", "frame_index"],
        filters=[("episode_index", "==", episode_index)],
    )
    frame = frame[frame.episode_index == episode_index]
    return {int(row.frame_index): int(row.index) for row in frame.itertuples(index=False)}


def current_state_from_history(state: torch.Tensor) -> torch.Tensor:
    if state.ndim == 3:
        return state[:, -1, :]
    if state.ndim == 2:
        return state
    raise ValueError(f"Unexpected raw state shape: {tuple(state.shape)}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_prediction(
    *,
    pred_absolute: np.ndarray,
    target_absolute: np.ndarray,
    current_frame: int,
    gap: int,
    label_event_frame: int,
    threshold: float,
    stable_steps: int,
) -> dict[str, object]:
    pred_close = stable_close_index(
        pred_absolute[:, 9], threshold, stable_steps, require_open_before=False
    )
    target_close = stable_close_index(
        target_absolute[:, 9], threshold, stable_steps, require_open_before=True
    )
    event_index = label_event_frame - current_frame - gap
    contains_event = 0 <= event_index < len(target_absolute)
    xyz_error = np.linalg.norm(pred_absolute[:, :3] - target_absolute[:, :3], axis=-1) * 1000.0
    rot_error = rotation_error_deg(pred_absolute, target_absolute)
    event_same_index_error = float(
        1000.0 * np.linalg.norm(pred_absolute[event_index, :3] - target_absolute[event_index, :3])
    ) if contains_event else None
    event_to_event_error = None
    if pred_close is not None and target_close is not None:
        event_to_event_error = float(
            1000.0 * np.linalg.norm(pred_absolute[pred_close, :3] - target_absolute[target_close, :3])
        )
    return {
        "current_frame": current_frame,
        "chunk_original_frames": [current_frame + gap, current_frame + gap + len(target_absolute) - 1],
        "label_event_frame": label_event_frame,
        "label_event_chunk_index": int(event_index) if contains_event else None,
        "label_contains_event": contains_event,
        "label_contains_stable_close": target_close is not None,
        "label_already_closed_at_window_start": bool(np.all(target_absolute[:stable_steps, 9] <= threshold)),
        "close_recall_evaluable": target_close is not None,
        "close_recall_status": ("predicted_close" if pred_close is not None else "missed_label_close") if target_close is not None else "not_applicable_label_has_no_close_transition",
        "label_stable_close_chunk_index": target_close,
        "label_stable_close_original_frame": None if target_close is None else current_frame + gap + target_close,
        "prediction_contains_stable_close": pred_close is not None,
        "prediction_stable_close_chunk_index": pred_close,
        "prediction_stable_close_original_frame": None if pred_close is None else current_frame + gap + pred_close,
        "prediction_close_frame_error": None if pred_close is None or target_close is None else int(pred_close - target_close),
        "xyz_error_mm_by_chunk_index": xyz_error.tolist(),
        "rotation_error_deg_by_chunk_index": rot_error.tolist(),
        "gripper_abs_error_by_chunk_index": np.abs(pred_absolute[:, 9] - target_absolute[:, 9]).tolist(),
        "xyz_mae_mm": float(np.mean(xyz_error)),
        "xyz_error_at_label_event_mm": event_same_index_error,
        "event_to_event_close_position_error_mm": event_to_event_error,
        "rotation_error_at_label_event_deg": float(rot_error[event_index]) if contains_event else None,
        "gripper_pred_at_label_event": float(pred_absolute[event_index, 9]) if contains_event else None,
        "gripper_label_at_event": float(target_absolute[event_index, 9]) if contains_event else None,
        "pred_absolute_xyz_at_label_event_m": pred_absolute[event_index, :3].tolist() if contains_event else None,
        "label_absolute_xyz_at_event_m": target_absolute[event_index, :3].tolist() if contains_event else None,
        "pred_absolute_xyz_at_pred_close_m": None if pred_close is None else pred_absolute[pred_close, :3].tolist(),
        "label_absolute_xyz_at_label_close_m": None if target_close is None else target_absolute[target_close, :3].tolist(),
    }


def plot_samples(sample_outputs: list[dict[str, object]], gap: int, output: Path) -> None:
    rows = len(sample_outputs)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4.2 * rows), squeeze=False, constrained_layout=True)
    for row, sample in enumerate(sample_outputs):
        current = int(sample["current_frame"])
        frames = current + gap + np.arange(sample["target_absolute"].shape[0])
        target = sample["target_absolute"]
        predictions = sample["pred_absolute_all"]
        target_local = sample["target_local"]
        pred_local = sample["pred_local_all"]

        ax = axes[row, 0]
        for seed_i, prediction in enumerate(predictions):
            error = 1000.0 * np.linalg.norm(prediction[:, :3] - target[:, :3], axis=-1)
            ax.plot(frames, error, alpha=0.8, label=f"seed {seed_i}")
        ax.axvline(sample["label_event_frame"], color="black", linestyle="--", label="shared label event")
        ax.set(title=f"t={current}: position error", xlabel="original frame", ylabel="error [mm]")
        ax.grid(alpha=0.2)
        ax.legend()

        ax = axes[row, 1]
        ax.plot(frames, target[:, 9], color="black", linewidth=3, label="clean label")
        for seed_i, prediction in enumerate(predictions):
            ax.plot(frames, prediction[:, 9], alpha=0.75, label=f"pred seed {seed_i}")
        ax.axhline(sample["close_threshold"], color="red", linestyle=":", label="close threshold")
        ax.axvline(sample["label_event_frame"], color="black", linestyle="--")
        ax.set(title=f"t={current}: gripper", xlabel="original frame", ylabel="absolute gripper")
        ax.grid(alpha=0.2)
        ax.legend()

        ax = axes[row, 2]
        colors = ["#4c78a8", "#f58518", "#54a24b"]
        for dim, name in enumerate("xyz"):
            ax.plot(frames, 1000.0 * target_local[:, dim], color=colors[dim], linewidth=3, label=f"label {name}")
            for seed_i in range(len(pred_local)):
                ax.plot(
                    frames,
                    1000.0 * pred_local[seed_i, :, dim],
                    color=colors[dim],
                    alpha=0.28,
                    linestyle="--",
                )
        ax.axvline(sample["label_event_frame"], color="black", linestyle="--")
        ax.set(title=f"t={current}: local clean label vs prediction", xlabel="original frame", ylabel="local TCP [mm]")
        ax.grid(alpha=0.2)
        ax.legend(ncol=2)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.stable_steps < 1 or args.num_seeds < 1:
        raise ValueError("stable-steps and num-seeds must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[event-eval] loading full-episode clean labels", flush=True)
    episode_labels = load_episode_labels(args.sidecar, args.episode_index)
    episode_actions = np.stack(episode_labels.action.to_numpy()).astype(np.float32)
    full_close_local = stable_close_index(
        episode_actions[:, 9],
        args.close_threshold,
        args.stable_steps,
        require_open_before=True,
    )
    if args.event_frame is None:
        if full_close_local is None:
            raise RuntimeError("No stable close was found in the full episode clean label.")
        event_frame = int(episode_labels.iloc[full_close_local].frame_index)
    else:
        event_frame = int(args.event_frame)

    if args.current_frames:
        current_frames = parse_ints(args.current_frames)
    else:
        current_frames = sorted({max(0, event_frame - offset) for offset in parse_ints(args.auto_anchor_offsets)})
    if len(current_frames) < 1:
        raise ValueError("No current frames were selected.")

    print(f"[event-eval] event={event_frame} anchors={current_frames}; building training dataset", flush=True)
    dataset = build_dataset(args)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    cfg.pretrained_path = str(args.checkpoint)
    cfg.base_vlm = str(args.base_vlm)
    cfg.device = args.device
    cfg.use_amp = False
    cfg.online_anchor_state_to_episode_start = False
    gap = int(cfg.action_gap_frames)
    chunk_size = int(cfg.chunk_size)
    if chunk_size != 32 or gap != 6:
        raise RuntimeError(f"This experiment expects the checked training setup chunk=32 gap=6, got {chunk_size=} {gap=}")

    print("[event-eval] loading checkpoint policy", flush=True)
    policy = make_policy(cfg, ds_meta=dataset.meta, rename_map={})
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": {}},
            "starvla_prepare_tactile_history": {
                "enabled": True,
                "history_len": int(cfg.tactile_history_len),
                "force_indices": list(cfg.tactile_force_indices),
                "output_key": cfg.tactile_history_key,
            },
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.eval()
    print("[event-eval] policy and processors loaded", flush=True)
    index_map = dataset_index_map(args.dataset_root, args.episode_index)

    sample_outputs: list[dict[str, object]] = []
    summary_samples: list[dict[str, object]] = []
    saved_arrays: dict[str, np.ndarray] = {}
    for current_frame in current_frames:
        if current_frame not in index_map:
            raise ValueError(f"Frame {current_frame} is not present in episode {args.episode_index}")
        event_index = event_frame - current_frame - gap
        contains_event = 0 <= event_index < chunk_size
        dataset_index = index_map[current_frame]
        item = dataset[dataset_index]
        if to_int(item["episode_index"]) != args.episode_index or to_int(item["frame_index"]) != current_frame:
            raise RuntimeError("Dataset index mapping did not reproduce the requested episode/frame.")
        raw_batch = {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
            for key, value in item.items()
        }
        raw_target_absolute = raw_batch[ACTION].clone()
        expected_frames = current_frame + gap + np.arange(chunk_size)
        episode_by_frame = episode_labels.set_index("frame_index")
        if not set(expected_frames).issubset(set(episode_by_frame.index)):
            raise RuntimeError("Selected chunk includes unavailable/padded episode frames; choose an earlier anchor.")
        expected_target = np.stack(episode_by_frame.loc[expected_frames, "action"].to_numpy()).astype(np.float32)
        label_frame_mapping_max_abs = float(np.max(np.abs(to_numpy(raw_target_absolute[0]) - expected_target)))
        if label_frame_mapping_max_abs > 2e-6:
            raise RuntimeError(f"Dataset future label does not match original frames {expected_frames.tolist()}: max_abs={label_frame_mapping_max_abs}")
        raw_state_history = raw_batch[OBS_STATE].clone()
        current_state = current_state_from_history(raw_state_history)
        policy.reset()
        preprocessor.reset()
        batch = preprocessor(raw_batch)
        clean_target_model = batch[ACTION].detach().clone()
        inference_batch = {key: value for key, value in batch.items() if key != ACTION}
        if ACTION in inference_batch:
            raise AssertionError("Clean target leaked into inference batch.")

        target_absolute_t = postprocessor(clean_target_model.clone())
        roundtrip_max_abs = float(
            torch.max(torch.abs(target_absolute_t.cpu() - raw_target_absolute.cpu())).item()
        )
        if roundtrip_max_abs > 2e-5:
            raise RuntimeError(
                "Clean target preprocess/postprocess round trip does not match dataset label: "
                f"max_abs={roundtrip_max_abs}"
            )
        target_absolute = to_numpy(target_absolute_t[0])
        target_local = to_numpy(
            to_keyframe_relative_actions(
                target_absolute_t,
                current_state,
                cfg.relative_action_keyframes,
                cfg.relative_exclude_keys,
            )[0]
        )

        pred_model_all: list[np.ndarray] = []
        pred_absolute_all: list[np.ndarray] = []
        pred_local_all: list[np.ndarray] = []
        seed_summaries: list[dict[str, object]] = []
        for seed_index in range(args.num_seeds):
            actual_seed = args.seed + seed_index
            set_seed(actual_seed)
            print(f"[event-eval] inference t={current_frame} seed={actual_seed}", flush=True)
            with torch.inference_mode():
                prediction_model_t = policy.predict_action_chunk(inference_batch)
            prediction_absolute_t = postprocessor(prediction_model_t.clone())
            prediction_local_t = to_keyframe_relative_actions(
                prediction_absolute_t,
                current_state,
                cfg.relative_action_keyframes,
                cfg.relative_exclude_keys,
            )
            prediction_model = to_numpy(prediction_model_t[0])
            prediction_absolute = to_numpy(prediction_absolute_t[0])
            prediction_local = to_numpy(prediction_local_t[0])
            pred_model_all.append(prediction_model)
            pred_absolute_all.append(prediction_absolute)
            pred_local_all.append(prediction_local)
            one_summary = summarize_prediction(
                pred_absolute=prediction_absolute,
                target_absolute=target_absolute,
                current_frame=current_frame,
                gap=gap,
                label_event_frame=event_frame,
                threshold=args.close_threshold,
                stable_steps=args.stable_steps,
            )
            one_summary["seed"] = actual_seed
            seed_summaries.append(one_summary)

        pred_model_array = np.stack(pred_model_all)
        pred_absolute_array = np.stack(pred_absolute_all)
        pred_local_array = np.stack(pred_local_all)
        key = f"frame_{current_frame:04d}"
        saved_arrays[f"{key}_clean_target_model"] = to_numpy(clean_target_model[0])
        saved_arrays[f"{key}_target_absolute"] = target_absolute
        saved_arrays[f"{key}_target_local"] = target_local
        saved_arrays[f"{key}_pred_model"] = pred_model_array
        saved_arrays[f"{key}_pred_absolute"] = pred_absolute_array
        saved_arrays[f"{key}_pred_local"] = pred_local_array
        sample_outputs.append(
            {
                "current_frame": current_frame,
                "dataset_index": dataset_index,
                "label_event_frame": event_frame,
                "label_event_chunk_index": event_index if contains_event else None,
                "close_threshold": args.close_threshold,
                "target_absolute": target_absolute,
                "target_local": target_local,
                "pred_absolute_all": pred_absolute_array,
                "pred_local_all": pred_local_array,
            }
        )
        summary_samples.append(
            {
                "current_frame": current_frame,
                "dataset_index": dataset_index,
                "chunk_original_frames": [current_frame + gap, current_frame + gap + chunk_size - 1],
                "label_event_chunk_index": event_index if contains_event else None,
                "clean_target_roundtrip_max_abs": roundtrip_max_abs,
                "dataset_label_original_frame_mapping_max_abs": label_frame_mapping_max_abs,
                "inference_batch_contains_action": ACTION in inference_batch,
                "label_contains_full_stable_close_at_shared_event": bool(
                    contains_event and event_index + args.stable_steps <= chunk_size
                    and np.all(
                        target_absolute[event_index : event_index + args.stable_steps, 9]
                        <= args.close_threshold
                    )
                ),
                "seeds": seed_summaries,
            }
        )
        print(
            f"[event-eval] episode={args.episode_index} t={current_frame} "
            f"event={event_frame} chunk_index={event_index} roundtrip={roundtrip_max_abs:.3g}",
            flush=True,
        )

    event_samples = [sample for sample in sample_outputs if sample["label_event_chunk_index"] is not None]
    target_event_xyz = np.stack(
        [sample["target_absolute"][sample["label_event_chunk_index"], :3] for sample in event_samples]
    ) if event_samples else np.empty((0, 3), dtype=np.float32)
    target_event_pairwise = np.linalg.norm(
        target_event_xyz[:, None, :] - target_event_xyz[None, :, :], axis=-1
    )
    prediction_event_cross_anchor_mm: list[float] = []
    if len(event_samples) == 2:
        for seed_index in range(args.num_seeds):
            a, b = event_samples
            a_xyz = a["pred_absolute_all"][seed_index, a["label_event_chunk_index"], :3]
            b_xyz = b["pred_absolute_all"][seed_index, b["label_event_chunk_index"], :3]
            prediction_event_cross_anchor_mm.append(float(1000.0 * np.linalg.norm(a_xyz - b_xyz)))

    endpoint_frames = [int(sample["current_frame"] + gap + chunk_size - 1) for sample in sample_outputs]
    endpoint_target_xyz = np.stack([sample["target_absolute"][-1, :3] for sample in sample_outputs])
    endpoint_pairwise = np.linalg.norm(
        endpoint_target_xyz[:, None, :] - endpoint_target_xyz[None, :, :], axis=-1
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset_root),
        "sidecar": str(args.sidecar),
        "episode_index": args.episode_index,
        "shared_label_close_event_frame": event_frame,
        "close_detection": {
            "threshold": args.close_threshold,
            "stable_steps": args.stable_steps,
            "source": "full episode clean sidecar action, before normalization/noise",
        },
        "chunk": {"size": chunk_size, "gap": gap, "future_offsets": [gap, gap + chunk_size - 1]},
        "clean_label_contract": {
            "source": "dataset item ACTION cloned before inference",
            "model_space_label": "preprocessor output cloned before any model forward/noise",
            "inference_action_key_removed": True,
            "physical_comparison": "saved postprocessor unnormalization + relative-to-absolute decode",
        },
        "seeds": [args.seed + index for index in range(args.num_seeds)],
        "samples": summary_samples,
        "shared_event_comparison": {
            "current_frames": [sample["current_frame"] for sample in event_samples],
            "target_absolute_xyz_m_by_anchor": target_event_xyz.tolist(),
            "target_cross_anchor_max_distance_mm": float(1000.0 * target_event_pairwise.max()) if event_samples else None,
            "prediction_cross_anchor_distance_mm_by_seed": prediction_event_cross_anchor_mm,
        },
        "why_not_chunk_endpoints": {
            "endpoint_original_frames": endpoint_frames,
            "target_endpoint_cross_anchor_max_distance_mm": float(1000.0 * endpoint_pairwise.max()),
            "note": "Endpoints are different original frames and are not used for the shared-event verdict.",
        },
        "elapsed_s": time.perf_counter() - started,
    }
    np.savez_compressed(args.output_dir / "event_label_predictions.npz", **saved_arrays)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_samples(sample_outputs, gap, args.output_dir / "prediction_vs_clean_label.png")
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
