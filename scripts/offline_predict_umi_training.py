#!/usr/bin/env python3
"""Offline StarVLA prediction on the canonical UMI training sidecar."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.relative_action_processor import to_keyframe_relative_actions
from lerobot.utils.constants import ACTION, OBS_STATE


CAMERA_RENAME = {}
MODEL_CAMERA_KEYS = {
    "observation.images.cam_left_undist",
    "observation.images.cam_right_undist",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--relative-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def to_int(value) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def phase_summary(array: np.ndarray) -> dict:
    result = {}
    for phase in (0, 7, 15):
        result[str(phase)] = {
            "left_mean_mm": (array[:, phase, 0:3].mean(0) * 1000).tolist(),
            "right_mean_mm": (array[:, phase, 10:13].mean(0) * 1000).tolist(),
            "left_std_mm": (array[:, phase, 0:3].std(0) * 1000).tolist(),
            "right_std_mm": (array[:, phase, 10:13].std(0) * 1000).tolist(),
            "left_positive_fraction": (array[:, phase, 0:3] > 0).mean(0).tolist(),
            "right_positive_fraction": (array[:, phase, 10:13] > 0).mean(0).tolist(),
        }
    return result


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Recreate the filtered metadata used for training.  The raw UMI info also
    # advertises ego, wrist, raw and tactile streams that this checkpoint never
    # consumes; letting the dataset reader validate those unrelated streams can
    # fail before a sample is loaded.
    raw_info = json.loads((args.dataset_root / "meta/info.json").read_text())
    scalar_keys = {
        "observation.state", "action", "score", "timestamp", "frame_index",
        "episode_index", "index", "task_index",
    }
    keep_keys = MODEL_CAMERA_KEYS | scalar_keys
    missing_cameras = MODEL_CAMERA_KEYS - set(raw_info["features"])
    if missing_cameras:
        raise RuntimeError(f"Missing required training cameras: {sorted(missing_cameras)}")
    raw_info["features"] = {
        key: value for key, value in raw_info["features"].items() if key in keep_keys
    }
    filtered_info = args.output_dir / "info_wrist_undist_offline.json"
    filtered_info.write_text(json.dumps(raw_info, indent=2, ensure_ascii=False), encoding="utf-8")

    train_cfg = TrainPipelineConfig.from_pretrained(args.checkpoint / "train_config.json")
    train_cfg.dataset.root = str(args.dataset_root)
    train_cfg.dataset.info_path = str(filtered_info)
    train_cfg.dataset.sidecar_path = str(args.sidecar)
    train_cfg.dataset.relative_stats_path = str(args.relative_stats)
    train_cfg.dataset.video_backend = "pyav"
    train_cfg.dataset.episodes = None
    train_cfg.rename_map = CAMERA_RENAME
    train_cfg.num_workers = 0
    train_cfg.policy.device = "cuda"
    train_cfg.policy.base_vlm = str(args.base_vlm)
    train_cfg.policy.use_amp = False
    dataset = make_dataset(train_cfg)

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    cfg.pretrained_path = str(args.checkpoint)
    cfg.base_vlm = str(args.base_vlm)
    cfg.device = "cuda"
    cfg.use_amp = False
    cfg.online_anchor_state_to_episode_start = False
    policy = make_policy(cfg, ds_meta=dataset.meta, rename_map=CAMERA_RENAME)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": CAMERA_RENAME},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.eval()
    policy.reset()
    preprocessor.reset()

    collected = {
        "dataset_index": [], "episode_index": [], "frame_index": [],
        "state_raw": [], "state_normalized": [],
        "pred_relative": [], "target_relative": [],
        "pred_absolute": [], "target_absolute": [],
    }
    current_episode = None
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(0, len(dataset), args.stride):
            if len(collected["dataset_index"]) >= args.max_samples:
                break
            item = dataset[index]
            episode = to_int(item["episode_index"])
            if episode != current_episode:
                policy.reset()
                preprocessor.reset()
                current_episode = episode

            raw_batch = {
                key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                for key, value in item.items()
            }
            state_raw = raw_batch[OBS_STATE].clone()
            batch = preprocessor(raw_batch)
            seed = args.seed + index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            pred_model = policy.predict_action_chunk(batch)
            target_model = batch[ACTION]
            pred_absolute = postprocessor(pred_model.clone())
            target_absolute = postprocessor(target_model.clone())
            pred_relative = to_keyframe_relative_actions(
                pred_absolute,
                state_raw,
                cfg.relative_action_keyframes,
                cfg.relative_exclude_keys,
            )
            target_relative = to_keyframe_relative_actions(
                target_absolute,
                state_raw,
                cfg.relative_action_keyframes,
                cfg.relative_exclude_keys,
            )

            collected["dataset_index"].append(index)
            collected["episode_index"].append(episode)
            collected["frame_index"].append(to_int(item["frame_index"]))
            collected["state_raw"].append(to_numpy(state_raw[0]))
            collected["state_normalized"].append(to_numpy(batch[OBS_STATE][0]))
            collected["pred_relative"].append(to_numpy(pred_relative[0]))
            collected["target_relative"].append(to_numpy(target_relative[0]))
            collected["pred_absolute"].append(to_numpy(pred_absolute[0]))
            collected["target_absolute"].append(to_numpy(target_absolute[0]))
            if len(collected["dataset_index"]) % 20 == 0:
                print(
                    f"[umi-offline] samples={len(collected['dataset_index'])} "
                    f"episode={episode} frame={to_int(item['frame_index'])}",
                    flush=True,
                )

    arrays = {key: np.asarray(value) for key, value in collected.items()}
    np.savez_compressed(args.output_dir / "umi_offline_predictions.npz", **arrays)
    pred = arrays["pred_relative"][:, :16]
    target = arrays["target_relative"][:, :16]
    xyz = [0, 1, 2, 10, 11, 12]
    pred_end = pred[:, 15]
    target_end = target[:, 15]
    summary = {
        "dataset": str(args.dataset_root),
        "sidecar": str(args.sidecar),
        "checkpoint": str(args.checkpoint),
        "num_samples": int(len(pred)),
        "num_episodes_sampled": int(len(np.unique(arrays["episode_index"]))),
        "stride": args.stride,
        "elapsed_s": time.perf_counter() - started,
        "state_normalized_abs_max": float(np.abs(arrays["state_normalized"]).max()),
        "predicted_local_tcp_delta_phase_mm": phase_summary(pred),
        "target_local_tcp_delta_phase_mm": phase_summary(target),
        "prediction_vs_target": {
            "xyz_mae_m_first16": float(np.abs(pred[..., xyz] - target[..., xyz]).mean()),
            "right_endpoint_z_sign_agreement": float(
                np.mean(np.sign(pred_end[:, 12]) == np.sign(target_end[:, 12]))
            ),
            "right_endpoint_pred_z_mean_mm": float(pred_end[:, 12].mean() * 1000),
            "right_endpoint_target_z_mean_mm": float(target_end[:, 12].mean() * 1000),
            "right_endpoint_pred_z_positive_fraction": float(np.mean(pred_end[:, 12] > 0)),
            "right_endpoint_target_z_positive_fraction": float(np.mean(target_end[:, 12] > 0)),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
