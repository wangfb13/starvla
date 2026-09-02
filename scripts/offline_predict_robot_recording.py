#!/usr/bin/env python3
"""Replay a recorded Spark3 episode through StarVLA-GR00T without robot control.

The script intentionally imports no robot driver and publishes no command.  It
compares three DiT-state inputs while keeping images, prompt, model weights and
diffusion noise identical:

  A: deployed episode-start-relative state adapter
  B: identity TCP poses (grippers and validity retained)
  C: raw robot-base absolute canonical TCP state
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.async_inference.helpers import (
    augment_observation_for_online_pi05_full_sidecar,
    prepare_image,
    resize_robot_observation_image,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy
from lerobot.processor.relative_action_processor import to_keyframe_relative_actions
from lerobot.utils.constants import OBS_STATE


TASK = "Stack the red cup on top of another cup."
LEFT_XYZ = slice(0, 3)
RIGHT_XYZ = slice(10, 13)
IDENTITY_ROT6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
UMI_MEDIAN_GRIPPER = (0.76776001, 0.72916865)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def load_cfg(checkpoint: Path, base_vlm: Path, *, online_anchor: bool):
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.pretrained_path = str(checkpoint)
    cfg.base_vlm = str(base_vlm)
    cfg.device = "cuda"
    cfg.use_amp = False
    cfg.online_anchor_state_to_episode_start = online_anchor
    return cfg


def load_policy(checkpoint: Path, cfg) -> StarvlaGrootPolicy:
    policy = StarvlaGrootPolicy.from_pretrained(
        pretrained_name_or_path=checkpoint,
        config=cfg,
        local_files_only=True,
        strict=True,
    )
    return policy.to(torch.device("cuda")).eval()


def decode_frames(path: Path):
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            yield index, frame.to_ndarray(format="rgb24")


def raw_named_state(state: np.ndarray, names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, state, strict=True)}


def canonical_state(raw: dict[str, float], cfg) -> torch.Tensor:
    holder: dict = {}
    augment_observation_for_online_pi05_full_sidecar(
        observation=holder,
        raw_observation=raw,
        policy_config=cfg,
    )
    return holder[OBS_STATE].detach().clone()


def identity_pose_state(state: torch.Tensor) -> torch.Tensor:
    result = state.clone()
    result[0:3] = 0.0
    result[3:9] = IDENTITY_ROT6D
    result[10:13] = 0.0
    result[13:19] = IDENTITY_ROT6D
    return result


def make_observation(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    state: torch.Tensor,
    cfg,
) -> dict:
    height, width = tuple(int(x) for x in cfg.image_resolution)
    shape = (3, height, width)
    left = prepare_image(resize_robot_observation_image(torch.from_numpy(left_rgb), shape))
    right = prepare_image(resize_robot_observation_image(torch.from_numpy(right_rgb), shape))
    return {
        "observation.images.cam_left_undist": left,
        "observation.images.cam_right_undist": right,
        OBS_STATE: state.clone(),
        getattr(cfg, "state_mask_observation_key", "observation.state_mask"): torch.ones(1, 20),
        getattr(cfg, "action_mask_observation_key", "observation.action_mask"): torch.ones(1, 20),
        "observation.state_mask": torch.ones(1, 20),
        "observation.action_mask": torch.ones(1, 20),
        "task": TASK,
        "robot_type": "spark3_pose_gripper",
    }


def predict_from_prefix(policy, prefix, attention_mask, batch, seed: int) -> torch.Tensor:
    """Equivalent to policy.predict_action_chunk after a shared VLM prefix."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reps = max(1, int(policy.config.repeated_diffusion_steps))
    with torch.autocast("cuda", dtype=torch.float32):
        state = policy._get_state(batch, prefix.device, prefix.dtype)
        action_mask = policy._prepare_action_mask(batch, prefix.device, prefix.dtype)
        if reps > 1:
            batch_size = prefix.shape[0]
            prefix_rep = prefix.repeat(reps, 1, 1)
            attention_rep = attention_mask.repeat(reps, 1) if attention_mask is not None else None
            state_rep = state.repeat(reps, 1, 1) if state is not None else None
            mask_rep = action_mask.repeat(reps, 1, 1) if action_mask is not None else None
            pred_rep = policy.action_head.predict_action(
                prefix_rep,
                state_rep,
                encoder_attention_mask=attention_rep,
                action_mask=mask_rep,
            )
            prediction = pred_rep.view(reps, batch_size, *pred_rep.shape[1:]).mean(dim=0)
        else:
            prediction = policy.action_head.predict_action(
                prefix,
                state,
                encoder_attention_mask=attention_mask,
                action_mask=action_mask,
            )
    return prediction.float()


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat.astype(np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"Invalid quaternion: {quat.tolist()}")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    col0 = np.array([1 - 2 * (y*y + z*z), 2 * (x*y + z*w), 2 * (x*z - y*w)])
    col1 = np.array([2 * (x*y - z*w), 1 - 2 * (x*x + z*z), 2 * (y*z + x*w)])
    return np.concatenate((col0, col1)).astype(np.float32)


def raw_action_to_canonical(action: np.ndarray) -> np.ndarray:
    result = np.empty(20, dtype=np.float32)
    result[0:3] = action[0:3]
    result[3:9] = quat_xyzw_to_rot6d(action[3:7])
    result[9] = action[30]
    result[10:13] = action[7:10]
    result[13:19] = quat_xyzw_to_rot6d(action[10:14])
    result[19] = action[31]
    return result


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def xyz_phase_summary(array: np.ndarray) -> dict:
    # array: [chunks, horizon, 20], xyz values in meters
    result = {}
    for phase in (0, 7, 15):
        if phase >= array.shape[1]:
            continue
        result[str(phase)] = {
            "left_mean_mm": (array[:, phase, LEFT_XYZ].mean(axis=0) * 1000).tolist(),
            "right_mean_mm": (array[:, phase, RIGHT_XYZ].mean(axis=0) * 1000).tolist(),
            "left_std_mm": (array[:, phase, LEFT_XYZ].std(axis=0) * 1000).tolist(),
            "right_std_mm": (array[:, phase, RIGHT_XYZ].std(axis=0) * 1000).tolist(),
        }
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg_a = load_cfg(args.checkpoint, args.base_vlm, online_anchor=True)
    cfg_d = load_cfg(args.checkpoint, args.base_vlm, online_anchor=True)
    cfg_plain = load_cfg(args.checkpoint, args.base_vlm, online_anchor=False)
    policy = load_policy(args.checkpoint, cfg_a)
    pre_a, post_a = make_pre_post_processors(
        cfg_a,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    pre_plain, _ = make_pre_post_processors(
        cfg_plain,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    pre_d, _ = make_pre_post_processors(
        cfg_d,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    pre_a.reset()
    pre_plain.reset()
    pre_d.reset()
    policy.reset()

    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    state_names = info["features"]["observation.state"]["names"]
    table = pq.read_table(args.dataset_root / "data/chunk-000/file-000.parquet")
    rows = table.to_pydict()
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    raw_actions = np.asarray(rows["action"], dtype=np.float32)
    frame_indices = np.asarray(rows["frame_index"], dtype=np.int64)
    timestamps = np.asarray(rows["timestamp"], dtype=np.float32)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise RuntimeError("This diagnostic expects a single episode with contiguous frame_index.")

    starts = np.arange(args.start_frame, len(states), args.stride, dtype=np.int64)
    if args.max_chunks > 0:
        starts = starts[: args.max_chunks]
    selected = set(int(x) for x in starts)
    left_path = args.dataset_root / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4"
    right_path = args.dataset_root / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4"

    records: dict[str, list[np.ndarray]] = {
        "state_raw": [], "state_adapter": [],
        "state_normalized_a": [], "state_normalized_b": [], "state_normalized_c": [], "state_normalized_d": [],
        "pred_normalized_a": [], "pred_normalized_b": [], "pred_normalized_c": [], "pred_normalized_d": [],
        "pred_relative_a": [], "pred_relative_b": [], "pred_relative_c": [], "pred_relative_d": [],
        "pred_absolute_a": [], "pred_absolute_b": [], "pred_absolute_c": [], "pred_absolute_d": [],
        "recorded_relative": [], "recorded_relative_valid": [],
    }
    used_frames: list[int] = []
    used_timestamps: list[float] = []
    episode_anchor: torch.Tensor | None = None

    left_iter = decode_frames(left_path)
    right_iter = decode_frames(right_path)
    for (li, left_rgb), (ri, right_rgb) in zip(left_iter, right_iter, strict=True):
        if li != ri:
            raise RuntimeError(f"Video frame mismatch: left={li}, right={ri}")
        if li not in selected:
            continue
        raw = raw_named_state(states[li], state_names)
        state_raw = canonical_state(raw, cfg_a)
        if episode_anchor is None:
            episode_anchor = state_raw.clone()
        state_adapter = to_keyframe_relative_actions(
            state_raw.unsqueeze(0),
            episode_anchor.unsqueeze(0),
            cfg_a.relative_action_keyframes,
            cfg_a.relative_exclude_keys,
        )[0]
        state_b = identity_pose_state(state_raw)
        state_d = state_raw.clone()
        state_d[9] = UMI_MEDIAN_GRIPPER[0]
        state_d[19] = UMI_MEDIAN_GRIPPER[1]

        obs_a = make_observation(left_rgb, right_rgb, state_raw, cfg_a)
        obs_b = make_observation(left_rgb, right_rgb, state_b, cfg_plain)
        obs_c = make_observation(left_rgb, right_rgb, state_raw, cfg_plain)
        obs_d = make_observation(left_rgb, right_rgb, state_d, cfg_d)
        batch_a = pre_a(obs_a)
        batch_b = pre_plain(obs_b)
        batch_c = pre_plain(obs_c)
        batch_d = pre_d(obs_d)

        with torch.inference_mode():
            prefix, attention = policy._encode_prefix(batch_a)
            seed = args.seed + int(li)
            pred_a = predict_from_prefix(policy, prefix, attention, batch_a, seed)
            pred_b = predict_from_prefix(policy, prefix, attention, batch_b, seed)
            pred_c = predict_from_prefix(policy, prefix, attention, batch_c, seed)
            pred_d = predict_from_prefix(policy, prefix, attention, batch_d, seed)

            absolute_predictions = []
            relative_predictions = []
            for prediction in (pred_a, pred_b, pred_c, pred_d):
                absolute = post_a(prediction.clone())
                relative = to_keyframe_relative_actions(
                    absolute,
                    state_raw.unsqueeze(0),
                    cfg_a.relative_action_keyframes,
                    cfg_a.relative_exclude_keys,
                )
                absolute_predictions.append(absolute)
                relative_predictions.append(relative)

        horizon = min(16, len(raw_actions) - li)
        actual_abs = np.stack([raw_action_to_canonical(x) for x in raw_actions[li : li + horizon]])
        actual_abs_t = torch.from_numpy(actual_abs).unsqueeze(0)
        actual_rel = to_keyframe_relative_actions(
            actual_abs_t,
            state_raw.unsqueeze(0),
            cfg_a.relative_action_keyframes,
            cfg_a.relative_exclude_keys,
        )[0]
        padded_actual = np.full((16, 20), np.nan, dtype=np.float32)
        padded_actual[:horizon] = as_numpy(actual_rel)

        records["state_raw"].append(as_numpy(state_raw))
        records["state_adapter"].append(as_numpy(state_adapter))
        records["state_normalized_a"].append(as_numpy(batch_a[OBS_STATE][0]))
        records["state_normalized_b"].append(as_numpy(batch_b[OBS_STATE][0]))
        records["state_normalized_c"].append(as_numpy(batch_c[OBS_STATE][0]))
        records["state_normalized_d"].append(as_numpy(batch_d[OBS_STATE][0]))
        for label, prediction, relative, absolute in zip(
            ("a", "b", "c", "d"),
            (pred_a, pred_b, pred_c, pred_d),
            relative_predictions,
            absolute_predictions,
            strict=True,
        ):
            records[f"pred_normalized_{label}"].append(as_numpy(prediction[0]))
            records[f"pred_relative_{label}"].append(as_numpy(relative[0]))
            records[f"pred_absolute_{label}"].append(as_numpy(absolute[0]))
        records["recorded_relative"].append(padded_actual)
        records["recorded_relative_valid"].append(np.arange(16) < horizon)
        used_frames.append(li)
        used_timestamps.append(float(timestamps[li]))
        print(f"[offline] chunk={len(used_frames):03d}/{len(starts):03d} frame={li:04d}", flush=True)

    if len(used_frames) != len(starts):
        raise RuntimeError(f"Decoded {len(used_frames)} selected frames, expected {len(starts)}")

    arrays = {key: np.stack(value) for key, value in records.items()}
    arrays["frame_index"] = np.asarray(used_frames, dtype=np.int64)
    arrays["timestamp"] = np.asarray(used_timestamps, dtype=np.float32)
    np.savez_compressed(args.output_dir / "offline_predictions.npz", **arrays)

    rel_a = arrays["pred_relative_a"][:, :16]
    rel_b = arrays["pred_relative_b"][:, :16]
    rel_c = arrays["pred_relative_c"][:, :16]
    rel_d = arrays["pred_relative_d"][:, :16]
    actual = arrays["recorded_relative"]
    xyz_dims = [0, 1, 2, 10, 11, 12]
    valid_actual = np.isfinite(actual[..., xyz_dims])
    actual_mae = float(np.abs(rel_a[..., xyz_dims] - actual[..., xyz_dims])[valid_actual].mean())
    summary = {
        "dataset": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "model_sha256_expected": "040af7882edae8e4a331c9f497084749063c7ad837febef7e8ce98a2d9ff72dd",
        "num_chunks": len(used_frames),
        "frames": used_frames,
        "state_normalized_abs_max": {
            "A_episode_anchor": float(np.abs(arrays["state_normalized_a"]).max()),
            "B_identity_pose": float(np.abs(arrays["state_normalized_b"]).max()),
            "C_base_absolute": float(np.abs(arrays["state_normalized_c"]).max()),
            "D_umi_median_gripper": float(np.abs(arrays["state_normalized_d"]).max()),
        },
        "predicted_local_tcp_delta_phase_mm": {
            "A_episode_anchor": xyz_phase_summary(rel_a),
            "B_identity_pose": xyz_phase_summary(rel_b),
            "C_base_absolute": xyz_phase_summary(rel_c),
            "D_umi_median_gripper": xyz_phase_summary(rel_d),
        },
        "state_ablation_prediction_mae": {
            "A_vs_B_all20": float(np.mean(np.abs(rel_a - rel_b))),
            "A_vs_C_all20": float(np.mean(np.abs(rel_a - rel_c))),
            "A_vs_B_xyz_m": float(np.mean(np.abs(rel_a[..., xyz_dims] - rel_b[..., xyz_dims]))),
            "A_vs_C_xyz_m": float(np.mean(np.abs(rel_a[..., xyz_dims] - rel_c[..., xyz_dims]))),
            "A_vs_D_all20": float(np.mean(np.abs(rel_a - rel_d))),
            "A_vs_D_xyz_m": float(np.mean(np.abs(rel_a[..., xyz_dims] - rel_d[..., xyz_dims]))),
        },
        "recorded_action_comparison": {
            "A_vs_recorded_xyz_mae_m": actual_mae,
            "warning": "Recorded actions are asynchronous executed commands, not supervised UMI targets.",
        },
        "interpretation": {
            "offline_A_positive_bias": "Bias exists before robot control; checkpoint/visual/state conditioning is implicated.",
            "A_close_to_B_and_C": "DiT is weakly sensitive to state; state adapter is unlikely to be the main cause.",
            "A_far_from_B_or_C": "DiT is state-sensitive; inspect state convention and normalization.",
            "camera_extrinsics": "This recording contains pixels but no camera-to-TCP transform; extrinsics require a separate geometric check.",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
