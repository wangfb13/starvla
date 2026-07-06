#!/usr/bin/env python3

"""Build a real-robot PI05 sidecar that uses observation.state joint angles as both state and action."""

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
    JOIN_KEYS,
    OBS_STATE,
    REAL_ROBOT_TYPES,
    build_previous_action,
    build_subtask_columns,
    collect_dataset_roots,
    compute_masked_stats,
    get_feature_stats,
    load_base_sidecar_map,
    load_frame_table,
    merge_base_sidecar,
    read_json,
    resolve_base_sidecar,
    resolve_task_text,
    serialize_dict,
    take_columns_with_aliases,
    to_list_column,
    write_json,
)

LOGGER = logging.getLogger("build_pi05_joint_state_as_action_sidecar")
DEFAULT_SCHEMA_NAME = "joint_state_as_action_bimanual_v1"

LEFT_JOINT_STATE_NAMES = [f"arm_left_{idx}" for idx in range(1, 8)]
RIGHT_JOINT_STATE_NAMES = [f"arm_right_{idx}" for idx in range(1, 8)]
LEFT_JOINT_FEATURE_NAMES = [f"main_arm_left_{idx}" for idx in range(1, 8)]
RIGHT_JOINT_FEATURE_NAMES = [f"main_arm_right_{idx}" for idx in range(1, 8)]
GRIPPER_FEATURE_NAMES = ["main_gripper_left", "main_gripper_right"]
JOINT_STATE_FEATURE_NAMES = (
    LEFT_JOINT_FEATURE_NAMES
    + ["main_gripper_left"]
    + RIGHT_JOINT_FEATURE_NAMES
    + ["main_gripper_right"]
)
JOINT_ACTION_FEATURE_NAMES = list(JOINT_STATE_FEATURE_NAMES)
STATE_PRESENCE_FEATURE_NAMES = ["left_arm_present", "right_arm_present"]
LEFT_ARM_SLICE = slice(0, 8)
RIGHT_ARM_SLICE = slice(8, 16)
JOINT_STATE_DIM = len(JOINT_STATE_FEATURE_NAMES) + len(STATE_PRESENCE_FEATURE_NAMES)
JOINT_ACTION_DIM = len(JOINT_ACTION_FEATURE_NAMES)
LEFT_ARM_PRESENT_NAME = "left_arm_present"
RIGHT_ARM_PRESENT_NAME = "right_arm_present"
LEFT_JOINT_CANDIDATES = [[name, f"main_{name}"] for name in LEFT_JOINT_STATE_NAMES]
RIGHT_JOINT_CANDIDATES = [[name, f"main_{name}"] for name in RIGHT_JOINT_STATE_NAMES]
LEFT_GRIPPER_CANDIDATES = [["gripper_left", "left_gripper", "main_gripper_left", "left_Position_Rad"]]
RIGHT_GRIPPER_CANDIDATES = [["gripper_right", "right_gripper", "main_gripper_right", "right_Position_Rad"]]


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
        help="Masks the selected arm out of joint state/action supervision.",
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


def build_joint_state_action_from_state(
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

    left_joints, left_joints_valid = take_columns_with_aliases(
        state_matrix,
        state_index,
        LEFT_JOINT_CANDIDATES,
        missing_value=missing_value,
    )
    right_joints, right_joints_valid = take_columns_with_aliases(
        state_matrix,
        state_index,
        RIGHT_JOINT_CANDIDATES,
        missing_value=missing_value,
    )
    left_gripper, left_gripper_valid = take_columns_with_aliases(
        state_matrix,
        state_index,
        LEFT_GRIPPER_CANDIDATES,
        missing_value=missing_value,
    )
    right_gripper, right_gripper_valid = take_columns_with_aliases(
        state_matrix,
        state_index,
        RIGHT_GRIPPER_CANDIDATES,
        missing_value=missing_value,
    )

    left_state = np.concatenate([left_joints, left_gripper], axis=1).astype(np.float32)
    right_state = np.concatenate([right_joints, right_gripper], axis=1).astype(np.float32)
    left_state_mask = np.concatenate([left_joints_valid, left_gripper_valid], axis=1).astype(np.float32)
    right_state_mask = np.concatenate([right_joints_valid, right_gripper_valid], axis=1).astype(np.float32)

    left_present = left_state_mask.all(axis=1).astype(np.float32)
    right_present = right_state_mask.all(axis=1).astype(np.float32)

    if real_mask_arm == "left":
        left_state_mask.fill(0.0)
        left_present.fill(0.0)
    elif real_mask_arm == "right":
        right_state_mask.fill(0.0)
        right_present.fill(0.0)

    arm_present_mask = np.stack([left_present, right_present], axis=1).astype(np.float32)
    state = np.concatenate([left_state, right_state, arm_present_mask], axis=1).astype(np.float32)
    state_mask = np.concatenate(
        [
            left_state_mask,
            right_state_mask,
            np.ones((state_matrix.shape[0], len(STATE_PRESENCE_FEATURE_NAMES)), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    action = np.concatenate([left_state, right_state], axis=1).astype(np.float32)
    action_valid_mask = state_mask.copy()
    action_valid_mask = action_valid_mask[:, :JOINT_ACTION_DIM]
    prev_action, prev_action_valid_mask = build_previous_action(
        episode_ids=episode_ids,
        action=action,
        action_valid_mask=action_valid_mask,
    )
    state_present = (arm_present_mask.max(axis=1) > 0).astype(np.float32)

    return {
        OBS_STATE: state,
        ACTION: action,
        "state_mask": state_mask,
        "action_valid_mask": action_valid_mask,
        "prev_action": prev_action,
        "prev_action_valid_mask": prev_action_valid_mask,
        "arm_present_mask": arm_present_mask,
        "state_present": state_present,
        "left_present": left_present,
        "right_present": right_present,
        "source_state_names": source_state_names,
    }


def compute_joint_framewise_stats(
    canonical: dict[str, np.ndarray],
    base_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    stats = deepcopy(base_stats) if isinstance(base_stats, dict) else {}
    stats[OBS_STATE] = compute_masked_stats(
        canonical[OBS_STATE],
        canonical["state_mask"],
        JOINT_STATE_FEATURE_NAMES + STATE_PRESENCE_FEATURE_NAMES,
    )
    stats[ACTION] = compute_masked_stats(
        canonical[ACTION],
        canonical["action_valid_mask"],
        JOINT_ACTION_FEATURE_NAMES,
    )
    stats["state_mask"] = get_feature_stats(
        canonical["state_mask"].astype(np.float32),
        axis=0,
        keepdims=False,
    )
    stats["action_valid_mask"] = get_feature_stats(
        canonical["action_valid_mask"].astype(np.float32),
        axis=0,
        keepdims=False,
    )
    stats["state_present"] = get_feature_stats(
        canonical["state_present"].astype(np.float32)[:, None],
        axis=0,
        keepdims=False,
    )
    stats["arm_present_mask"] = get_feature_stats(
        canonical["arm_present_mask"].astype(np.float32),
        axis=0,
        keepdims=False,
    )
    return stats


def build_sidecar_for_dataset(
    *,
    dataset_root: Path,
    output_root: Path,
    schema_name: str,
    missing_value: float,
    real_mask_arm: str,
    base_sidecar: Path | None,
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

    LOGGER.info("processing joint state-as-action dataset=%s robot_type=%s", dataset_root, robot_type)
    if real_mask_arm != "none":
        LOGGER.info("masking arm: %s", real_mask_arm)

    base_stats = read_json(stats_path) if stats_path.exists() else {}
    frame_df = load_frame_table(dataset_root)
    canonical = build_joint_state_action_from_state(
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
            "canonical_dataset_type": "real_joint_state",
            LEFT_ARM_PRESENT_NAME: canonical["left_present"].astype(np.float32),
            RIGHT_ARM_PRESENT_NAME: canonical["right_present"].astype(np.float32),
        }
    )
    framewise_df = merge_base_sidecar(base_sidecar, framewise_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    framewise_df.to_parquet(sidecar_path, index=False)
    stats_to_write = compute_joint_framewise_stats(canonical, base_stats)
    write_json(stats_output_path, serialize_dict(stats_to_write))

    summary = {
        "dataset_root": str(dataset_root),
        "dataset_type": "real",
        "robot_type": robot_type,
        "schema_name": schema_name,
        "real_mask_arm": real_mask_arm,
        "state_source": "raw_observation_state_joint_plus_gripper",
        "action_source": "raw_observation_state_joint_plus_gripper",
        "sidecar_path": str(sidecar_path),
        "stats_path": str(stats_output_path),
        "base_sidecar": str(base_sidecar) if base_sidecar is not None else None,
        "row_count": int(len(framewise_df)),
        "state_dim": JOINT_STATE_DIM,
        "action_dim": JOINT_ACTION_DIM,
        "state_feature_names": JOINT_STATE_FEATURE_NAMES + STATE_PRESENCE_FEATURE_NAMES,
        "action_feature_names": JOINT_ACTION_FEATURE_NAMES,
        "source_state_joint_names": LEFT_JOINT_STATE_NAMES + RIGHT_JOINT_STATE_NAMES + ["gripper_left", "gripper_right"],
        "arm_present_names": [LEFT_ARM_PRESENT_NAME, RIGHT_ARM_PRESENT_NAME],
        "missing_value": missing_value,
        "left_arm_present_ratio": float(canonical["left_present"].mean()),
        "right_arm_present_ratio": float(canonical["right_present"].mean()),
        "mean_action_valid_ratio": float(canonical["action_valid_mask"].mean()),
        "action_valid_ratio_by_dim": canonical["action_valid_mask"].mean(axis=0).astype(float).tolist(),
        "state_valid_ratio_by_dim": canonical["state_mask"].mean(axis=0).astype(float).tolist(),
        "task_vlm_caption_nonempty": bool(task_text.strip()),
        "subtask_frame_interval_mode": "inclusive_closed_interval_[start,end]",
        "subtask_text_nonempty_ratio": float((pd.Series(subtask_text).astype(str).str.strip() != "").mean()),
        "next_subtask_text_nonempty_ratio": float(
            (pd.Series(next_subtask_text).astype(str).str.strip() != "").mean()
        ),
        "recommended_robot_config": {
            "command_mode": "joint",
            "action_joint_names": JOINT_ACTION_FEATURE_NAMES,
        },
        "recommended_policy": {
            "canonicalize_bimanual_rot6d": False,
            "use_state_input": True,
            "state_mask_key": "state_mask",
            "action_valid_mask_key": "action_valid_mask",
            "state_present_key": "state_present",
            "max_state_dim": JOINT_STATE_DIM,
            "max_action_dim": JOINT_ACTION_DIM,
        },
    }
    write_json(summary_path, summary)
    LOGGER.info("wrote %s", sidecar_path)
    LOGGER.info("wrote %s", stats_output_path)
    LOGGER.info("wrote %s", summary_path)
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
    for dataset_root in tqdm(dataset_roots, desc="Building joint state-as-action sidecars"):
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
        "summaries": summaries,
    }
    write_json(args.output_root / "build_manifest.json", manifest)
    LOGGER.info("wrote %s", args.output_root / "build_manifest.json")


if __name__ == "__main__":
    main()
