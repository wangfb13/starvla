#!/usr/bin/env python3

"""Build a real-robot PI05 sidecar that reuses raw state as both canonical state and action, including gripper."""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from build_pi05_canonical_sidecar import (
    ACTION,
    ARM_NAMES,
    CANONICAL_ACTION_DIM,
    CANONICAL_STATE_DIM,
    GRIPPER_NAME_CANDIDATES,
    OBS_STATE,
    POSE_NAME_CANDIDATES,
    REAL_ROBOT_TYPES,
    build_arm_absolute_targets,
    build_pose_absolute_rot6d,
    build_previous_action,
    build_subtask_columns,
    collect_dataset_roots,
    compute_framewise_stats,
    compute_relative_action_chunk_stats,
    load_base_sidecar_map,
    load_frame_table,
    merge_base_sidecar,
    parse_csv_tokens,
    parse_relative_action_keyframes,
    read_json,
    resolve_base_sidecar,
    resolve_task_text,
    serialize_dict,
    take_columns_with_aliases,
    to_list_column,
    write_json,
)

LOGGER = logging.getLogger("build_pi05_real_state_as_action_state_gripper_sidecar")
DEFAULT_SCHEMA_NAME = "pi05_real_state_as_action_state_gripper_bimanual_rot6d_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=[],
        help="LeRobot dataset root. Repeat this flag to process multiple roots.",
    )
    parser.add_argument(
        "--dataset-roots-file",
        type=Path,
        action="append",
        default=[],
        help="Text file containing one LeRobot dataset root per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root. Each dataset gets a subdirectory named after its root.",
    )
    parser.add_argument(
        "--schema-name",
        default=DEFAULT_SCHEMA_NAME,
        help="Schema label written to framewise rows and summary.json.",
    )
    parser.add_argument(
        "--missing-value",
        type=float,
        default=9930.0,
        help="Sentinel value treated as missing in raw state vectors.",
    )
    parser.add_argument(
        "--real-mask-arm",
        choices=["none", "left", "right"],
        default="none",
        help="Masks the selected arm out of canonical state/action supervision.",
    )
    parser.add_argument(
        "--base-sidecar",
        type=Path,
        default=None,
        help="Optional single base sidecar to merge for single-dataset runs.",
    )
    parser.add_argument(
        "--base-sidecar-map",
        type=Path,
        default=None,
        help="Optional JSON mapping dataset root/name to a base framewise sidecar path.",
    )
    parser.add_argument(
        "--relative-stats-chunk-size",
        type=int,
        default=0,
        help="If >0, also write stats for relative action chunks with this chunk size.",
    )
    parser.add_argument(
        "--relative-stats-action-gap-frames",
        type=int,
        default=0,
        help="Future-frame gap used when computing optional relative action chunk stats.",
    )
    parser.add_argument(
        "--relative-action-keyframes",
        default="0:3,10:13",
        help="Comma-separated action slices converted by relative stats, e.g. `0:3,10:13`.",
    )
    parser.add_argument(
        "--relative-exclude-joints",
        default="",
        help="Comma-separated action-name tokens to keep absolute in optional relative stats.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecar/stat/summary files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    if not args.dataset_root and not args.dataset_roots_file:
        parser.error("At least one --dataset-root or --dataset-roots-file is required.")
    return args


def build_state_action_from_state_gripper(
    *,
    frame_df: pd.DataFrame,
    info: dict[str, Any],
    missing_value: float,
    real_mask_arm: str,
) -> dict[str, np.ndarray]:
    source_state_names = list(info["features"][OBS_STATE]["names"])
    state_index = {name: idx for idx, name in enumerate(source_state_names)}
    state_matrix = np.stack(frame_df[OBS_STATE].to_numpy()).astype(np.float32)
    episode_ids = frame_df["episode_index"].to_numpy(dtype=np.int64)

    state_arms: list[np.ndarray] = []
    state_masks: list[np.ndarray] = []
    state_present_bits: list[np.ndarray] = []
    action_arms: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    arm_anchor_counts = {arm: 0 for arm in ARM_NAMES}

    for arm in ARM_NAMES:
        pose_state, pose_state_valid = take_columns_with_aliases(
            state_matrix,
            state_index,
            POSE_NAME_CANDIDATES[arm],
            missing_value=missing_value,
        )
        grip_state, grip_state_valid = take_columns_with_aliases(
            state_matrix,
            state_index,
            GRIPPER_NAME_CANDIDATES[arm],
            missing_value=missing_value,
        )

        compact_state_pose, compact_state_pose_valid = build_pose_absolute_rot6d(
            pose_state,
            pose_state_valid,
        )
        state_arm = np.concatenate([compact_state_pose, grip_state], axis=1).astype(np.float32)
        state_arm_mask = np.concatenate([compact_state_pose_valid, grip_state_valid], axis=1).astype(np.float32)
        state_present = (compact_state_pose_valid.all(axis=1) | grip_state_valid.all(axis=1)).astype(np.float32)

        action_arm, action_mask = build_arm_absolute_targets(
            compact_pose=compact_state_pose,
            compact_pose_valid=compact_state_pose_valid,
            gripper=grip_state,
            gripper_valid=grip_state_valid,
        )

        if real_mask_arm == arm:
            state_arm_mask.fill(0.0)
            state_present.fill(0.0)
            action_mask.fill(0.0)

        state_arms.append(state_arm)
        state_masks.append(state_arm_mask)
        state_present_bits.append(state_present[:, None])
        action_arms.append(action_arm)
        action_masks.append(action_mask)

    canonical_state = np.concatenate(state_arms + state_present_bits, axis=1).astype(np.float32)
    state_mask = np.concatenate(state_masks, axis=1).astype(np.float32)
    canonical_action = np.concatenate(action_arms, axis=1).astype(np.float32)
    action_valid_mask = np.concatenate(action_masks, axis=1).astype(np.float32)
    prev_action, prev_action_valid_mask = build_previous_action(
        episode_ids=episode_ids,
        action=canonical_action,
        action_valid_mask=action_valid_mask,
    )
    arm_present_mask = np.concatenate(state_present_bits, axis=1).astype(np.float32)
    state_present = (arm_present_mask.max(axis=1) > 0).astype(np.float32)

    return {
        OBS_STATE: canonical_state,
        "state_mask": state_mask,
        ACTION: canonical_action,
        "action_valid_mask": action_valid_mask,
        "prev_action": prev_action,
        "prev_action_valid_mask": prev_action_valid_mask,
        "arm_present_mask": arm_present_mask,
        "state_present": state_present,
        "left_present": arm_present_mask[:, 0],
        "right_present": arm_present_mask[:, 1],
        "arm_anchor_counts": arm_anchor_counts,
    }


def build_sidecar_for_dataset(
    *,
    dataset_root: Path,
    output_root: Path,
    schema_name: str,
    missing_value: float,
    real_mask_arm: str,
    base_sidecar: Path | None,
    relative_stats_chunk_size: int,
    relative_stats_action_gap_frames: int,
    relative_action_keyframes: str,
    relative_exclude_joints: str,
    overwrite: bool,
) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")

    info = read_json(info_path)
    robot_type = str(info.get("robot_type", "")).strip()
    if robot_type not in REAL_ROBOT_TYPES:
        raise ValueError(
            f"{dataset_root} is not a supported real-robot dataset. "
            f"Expected robot_type in {sorted(REAL_ROBOT_TYPES)}, got {robot_type!r}."
        )

    output_dir = output_root / dataset_root.name
    sidecar_path = output_dir / "framewise_guidance.parquet"
    stats_output_path = output_dir / "stats.json"
    summary_path = output_dir / "summary.json"
    if not overwrite:
        for path in (sidecar_path, stats_output_path, summary_path):
            if path.exists():
                raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")

    LOGGER.info("processing real dataset=%s robot_type=%s", dataset_root, robot_type)
    if real_mask_arm != "none":
        LOGGER.info("masking arm: %s", real_mask_arm)

    base_stats = read_json(stats_path) if stats_path.exists() else {}
    frame_df = load_frame_table(dataset_root)
    canonical = build_state_action_from_state_gripper(
        frame_df=frame_df,
        info=info,
        missing_value=missing_value,
        real_mask_arm=real_mask_arm,
    )

    task_text = resolve_task_text(dataset_root)
    subtask_text, next_subtask_text = build_subtask_columns(frame_df, dataset_root)
    framewise_df = pd.DataFrame(
        {
            "index": frame_df["index"].astype(int),
            "episode_index": frame_df["episode_index"].astype(int),
            "frame_index": frame_df["frame_index"].astype(int),
            OBS_STATE: to_list_column(canonical[OBS_STATE]),
            ACTION: to_list_column(canonical[ACTION]),
            "action_valid_mask": to_list_column(canonical["action_valid_mask"]),
            "state_mask": to_list_column(canonical["state_mask"]),
            "arm_present_mask": to_list_column(canonical["arm_present_mask"]),
            "state_present": canonical["state_present"].astype(np.float32),
            "prev_action": to_list_column(canonical["prev_action"]),
            "prev_action_valid_mask": to_list_column(canonical["prev_action_valid_mask"]),
            "task_vlm_caption": task_text,
            "user_prompt_sidecar": "",
            "subtask_text_sidecar": subtask_text,
            "next_subtask_text_sidecar": next_subtask_text,
            "canonical_state_action_schema": schema_name,
            "canonical_dataset_type": "real",
            "left_arm_present": canonical["left_present"].astype(np.float32),
            "right_arm_present": canonical["right_present"].astype(np.float32),
        }
    )
    framewise_df = merge_base_sidecar(base_sidecar, framewise_df)

    framewise_stats = compute_framewise_stats(canonical, base_stats)
    stats_to_write = deepcopy(framewise_stats)
    relative_stats_path = None
    if relative_stats_chunk_size > 0:
        relative_keyframes = parse_relative_action_keyframes(relative_action_keyframes)
        relative_exclude_joint_tokens = parse_csv_tokens(relative_exclude_joints)
        relative_action_stats = compute_relative_action_chunk_stats(
            episode_ids=frame_df["episode_index"].to_numpy(dtype=np.int64),
            states=canonical[OBS_STATE],
            actions=canonical[ACTION],
            action_valid_mask=canonical["action_valid_mask"],
            chunk_size=relative_stats_chunk_size,
            action_gap_frames=relative_stats_action_gap_frames,
            relative_keyframes=relative_keyframes,
            relative_exclude_joints=relative_exclude_joint_tokens,
        )
        relative_stats = deepcopy(framewise_stats)
        relative_stats[ACTION] = relative_action_stats
        relative_stats_path = output_dir / (
            f"stats_relative_action_chunk{relative_stats_chunk_size}_gap{relative_stats_action_gap_frames}.json"
        )
        write_json(relative_stats_path, serialize_dict(relative_stats))

    output_dir.mkdir(parents=True, exist_ok=True)
    framewise_df.to_parquet(sidecar_path, index=False)
    write_json(stats_output_path, serialize_dict(stats_to_write))

    action_valid_ratio = canonical["action_valid_mask"].mean(axis=0)
    state_valid_ratio = np.concatenate(
        [
            canonical["state_mask"],
            np.ones((canonical[OBS_STATE].shape[0], 2), dtype=np.float32),
        ],
        axis=1,
    ).mean(axis=0)
    summary = {
        "dataset_root": str(dataset_root),
        "dataset_type": "real",
        "robot_type": robot_type,
        "schema_name": schema_name,
        "real_mask_arm": real_mask_arm,
        "state_source": "raw_observation_state",
        "action_source": "raw_observation_state",
        "sidecar_path": str(sidecar_path),
        "stats_path": str(stats_output_path),
        "relative_stats_path": str(relative_stats_path) if relative_stats_path is not None else None,
        "base_sidecar": str(base_sidecar) if base_sidecar is not None else None,
        "row_count": int(len(framewise_df)),
        "canonical_state_dim": CANONICAL_STATE_DIM,
        "canonical_action_dim": CANONICAL_ACTION_DIM,
        "state_pose_mode": "absolute_rot6d",
        "action_target_mode": "state_pose_and_state_gripper_reused_as_action_absolute_rot6d_framewise",
        "missing_value": missing_value,
        "left_arm_present_ratio": float(canonical["left_present"].mean()),
        "right_arm_present_ratio": float(canonical["right_present"].mean()),
        "mean_action_valid_ratio": float(canonical["action_valid_mask"].mean()),
        "action_valid_ratio_by_dim": action_valid_ratio.astype(float).tolist(),
        "state_valid_ratio_by_dim": state_valid_ratio.astype(float).tolist(),
        "arm_anchor_counts": canonical["arm_anchor_counts"],
        "task_vlm_caption_nonempty": bool(task_text.strip()),
        "subtask_frame_interval_mode": "inclusive_closed_interval_[start,end]",
        "subtask_text_nonempty_ratio": float((pd.Series(subtask_text).astype(str).str.strip() != "").mean()),
        "next_subtask_text_nonempty_ratio": float(
            (pd.Series(next_subtask_text).astype(str).str.strip() != "").mean()
        ),
        "recommended_policy": {
            "canonicalize_bimanual_rot6d": False,
            "use_relative_actions": True,
            "relative_action_keyframes": [
                item.strip() for item in relative_action_keyframes.split(",") if item.strip()
            ],
            "relative_exclude_joints": parse_csv_tokens(relative_exclude_joints),
            "action_gap_frames": relative_stats_action_gap_frames,
            "use_state_input": False,
            "action_valid_mask_key": "action_valid_mask",
            "state_present_key": "state_present",
            "max_state_dim": CANONICAL_STATE_DIM,
            "max_action_dim": CANONICAL_ACTION_DIM,
        },
    }
    write_json(summary_path, summary)
    LOGGER.info("wrote %s", sidecar_path)
    LOGGER.info("wrote %s", stats_output_path)
    if relative_stats_path is not None:
        LOGGER.info("wrote %s", relative_stats_path)
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
    )
    dataset_roots = collect_dataset_roots(args.dataset_root, args.dataset_roots_file)
    if not dataset_roots:
        raise ValueError("No dataset roots were provided.")
    if args.base_sidecar is not None and len(dataset_roots) != 1:
        raise ValueError("--base-sidecar can only be used with one --dataset-root. Use --base-sidecar-map instead.")

    base_sidecar_map = load_base_sidecar_map(args.base_sidecar_map)
    summaries = []
    for dataset_root in tqdm(dataset_roots, desc="Building real state-as-action sidecars (state gripper)"):
        base_sidecar = resolve_base_sidecar(
            dataset_root=dataset_root,
            single_base_sidecar=args.base_sidecar,
            base_sidecar_map=base_sidecar_map,
        )
        summaries.append(
            build_sidecar_for_dataset(
                dataset_root=dataset_root,
                output_root=args.output_root,
                schema_name=args.schema_name,
                missing_value=args.missing_value,
                real_mask_arm=args.real_mask_arm,
                base_sidecar=base_sidecar,
                relative_stats_chunk_size=args.relative_stats_chunk_size,
                relative_stats_action_gap_frames=args.relative_stats_action_gap_frames,
                relative_action_keyframes=args.relative_action_keyframes,
                relative_exclude_joints=args.relative_exclude_joints,
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "schema_name": args.schema_name,
        "dataset_count": len(summaries),
        "dataset_roots_file": [str(path) for path in args.dataset_roots_file],
        "sidecar_path_map": {
            Path(summary["dataset_root"]).name: summary["sidecar_path"] for summary in summaries
        },
        "stats_path_map": {
            Path(summary["dataset_root"]).name: summary["stats_path"] for summary in summaries
        },
        "relative_stats_path_map": {
            Path(summary["dataset_root"]).name: summary["relative_stats_path"]
            for summary in summaries
            if summary["relative_stats_path"] is not None
        },
        "summaries": summaries,
    }
    write_json(args.output_root / "build_manifest.json", manifest)
    LOGGER.info("wrote %s", args.output_root / "build_manifest.json")


if __name__ == "__main__":
    main()
