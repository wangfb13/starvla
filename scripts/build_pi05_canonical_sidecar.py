#!/usr/bin/env python3

"""Build PI0.5/PI05 canonical framewise sidecars for UMI and real-robot datasets."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.utils import serialize_dict
from lerobot.policies.pi05_full.canonical_bimanual import (
    ARM_CANONICAL_DIM,
    ARM_NAMES,
    CANONICAL_ACTION_DIM,
    CANONICAL_STATE_DIM,
    canonical_action_feature_names,
    canonical_state_feature_names,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation import Rotation

LOGGER = logging.getLogger("build_pi05_canonical_sidecar")

SENTINEL = 9930.0
JOIN_KEYS = ["index", "episode_index", "frame_index"]
POSE_NAME_CANDIDATES = {
    "left": [
        ["left_x"],
        ["left_y"],
        ["left_z"],
        ["left_qx", "left_Quat_X"],
        ["left_qy", "left_Quat_Y"],
        ["left_qz", "left_Quat_Z"],
        ["left_qw", "left_Quat_W"],
    ],
    "right": [
        ["right_x"],
        ["right_y"],
        ["right_z"],
        ["right_qx", "right_Quat_X"],
        ["right_qy", "right_Quat_Y"],
        ["right_qz", "right_Quat_Z"],
        ["right_qw", "right_Quat_W"],
    ],
}
GRIPPER_NAME_CANDIDATES = {
    "left": [["gripper_left", "left_Position_Rad", "left_gripper", "main_gripper_left"]],
    "right": [["gripper_right", "right_Position_Rad", "right_gripper", "main_gripper_right"]],
}
SUBTASK_EPISODE_PATTERN = re.compile(r"(?:^|_)ep(\d+)(?:_|$)")
DEFAULT_SCHEMA_NAME = "pi05_canonical_bimanual_rot6d_v1"
REAL_ROBOT_TYPES = {"spark3_pose_gripper"}
UMI_ROBOT_TYPES = {"ugripper_left", "ugripper_right", "dual_ugripper", "dual_ugripper_v2_5"}
UMI_GRIPPER_RAW_MAX = 0.0
DEFAULT_UMI_GRIPPER_OPEN_ABS = 0.436
INTER_ARM_RELATIVE_STATE_NAMES = [
    "left_wrt_right_x",
    "left_wrt_right_y",
    "left_wrt_right_z",
    *[f"left_wrt_right_rot6d_{idx}" for idx in range(6)],
]
STATE_AVAILABLE_NAME = "state_available"


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
        help="Text file containing one LeRobot dataset root per line. Repeatable; blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "umi", "real"],
        default="auto",
        help=(
            "`auto` infers from meta/info.json robot_type. `umi` converts pose state/action to each episode "
            "anchor frame; `real` keeps pose in dataset frame."
        ),
    )
    parser.add_argument(
        "--umi-pose-frame",
        choices=["episode_anchor", "absolute"],
        default="episode_anchor",
        help=(
            "Pose frame used for UMI datasets. `episode_anchor` preserves the legacy behavior; "
            "`absolute` keeps xyz/orientation in the dataset (QR-world) frame while still applying "
            "UMI gripper normalization."
        ),
    )
    parser.add_argument(
        "--umi-gripper-normalization",
        choices=["fixed", "data_min"],
        default="fixed",
        help=(
            "`fixed` uses --umi-gripper-open-abs for both arms. `data_min` computes a separate "
            "fully-open magnitude for each arm from all valid state/action gripper values."
        ),
    )
    parser.add_argument(
        "--umi-gripper-open-abs",
        type=float,
        default=DEFAULT_UMI_GRIPPER_OPEN_ABS,
        help=(
            "Positive magnitude of the raw UMI gripper value treated as fully open. "
            "Values are clipped to [-value, 0] and divided by -value."
        ),
    )
    parser.add_argument(
        "--real-mask-arm",
        choices=["none", "left", "right"],
        default="none",
        help=(
            "Only used when resolved dataset_type is `real`. "
            "Masks the selected arm out of canonical state/action supervision."
        ),
    )
    parser.add_argument(
        "--canonical-arm",
        choices=["both", "left", "right"],
        default="both",
        help=(
            "Select the arm(s) emitted by the canonical sidecar. A single-arm sidecar places that "
            "arm's 10D TCP/gripper state first, emits a matching 10D action, and retains only that "
            "arm's appended tactile features."
        ),
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
        default=SENTINEL,
        help="Sentinel value treated as missing in raw state/action vectors.",
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
        help=(
            "Comma-separated action-name tokens to keep absolute when writing optional relative stats. "
            "Default is empty, so non-keyframe dims such as gripper are also chunk-relative."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecar/stat/summary files.",
    )
    parser.add_argument(
        "--add-inter-arm-relative-state",
        action="store_true",
        help=(
            "Append the 9D left-TCP pose in the right-TCP frame plus a state_available bit. "
            "The resulting state is 32D; the online/history processor may recompute the 9D block "
            "against the current right TCP for exact UMI history semantics."
        ),
    )
    parser.add_argument(
        "--append-state-feature-prefixes",
        default="",
        help=(
            "Comma-separated raw observation.state name prefixes to append after the canonical "
            "state (for example 'tactile_' to retain six-axis tactile force/torque channels)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    if not args.dataset_root and not args.dataset_roots_file:
        parser.error("At least one --dataset-root or --dataset-roots-file is required.")
    if args.umi_gripper_open_abs <= 0:
        parser.error("--umi-gripper-open-abs must be positive.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_dataset_roots_from_file(path: Path) -> list[Path]:
    roots: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.partition("#")[0].strip()
            if not line or line.startswith("#"):
                continue
            root = Path(line).expanduser()
            if not root.is_absolute():
                root = (path.parent / root).resolve()
            roots.append(root)
            LOGGER.debug("loaded dataset root from %s:%d -> %s", path, line_number, root)
    return roots


def collect_dataset_roots(root_args: list[Path], roots_files: list[Path]) -> list[Path]:
    roots: list[Path] = [root.expanduser() for root in root_args]
    for roots_file in roots_files:
        roots.extend(load_dataset_roots_from_file(roots_file.expanduser()))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            LOGGER.warning("skip duplicated dataset root: %s", root)
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def infer_dataset_type(info: dict[str, Any], requested_dataset_type: str, dataset_root: Path) -> str:
    if requested_dataset_type != "auto":
        return requested_dataset_type

    robot_type = str(info.get("robot_type", "")).strip()
    if robot_type in REAL_ROBOT_TYPES:
        return "real"
    if robot_type in UMI_ROBOT_TYPES:
        return "umi"
    raise ValueError(
        f"Cannot infer dataset type for {dataset_root}: unsupported robot_type={robot_type!r}. "
        "Pass --dataset-type umi or --dataset-type real to override, or add the robot_type mapping in this script."
    )


def name_to_index(names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def load_frame_table(dataset_root: Path) -> pd.DataFrame:
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    columns = JOIN_KEYS + [OBS_STATE, ACTION]
    frames = []
    for path in tqdm(parquet_paths, desc=f"Loading parquet {dataset_root.name}"):
        LOGGER.info("loading %s", path)
        frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True).sort_values("index").reset_index(drop=True)


def take_columns(
    matrix: np.ndarray,
    mapping: dict[str, int],
    names: list[str],
    *,
    missing_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((matrix.shape[0], len(names)), dtype=np.float32)
    valid = np.zeros((matrix.shape[0], len(names)), dtype=bool)
    for out_idx, name in enumerate(names):
        source_idx = mapping.get(name)
        if source_idx is None:
            continue
        column = matrix[:, source_idx].astype(np.float32)
        use = np.isfinite(column) & ~np.isclose(column, missing_value)
        if np.any(use):
            values[use, out_idx] = column[use]
        valid[:, out_idx] = use
    return values, valid


def take_columns_with_aliases(
    matrix: np.ndarray,
    mapping: dict[str, int],
    candidate_groups: list[list[str]],
    *,
    missing_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((matrix.shape[0], len(candidate_groups)), dtype=np.float32)
    valid = np.zeros((matrix.shape[0], len(candidate_groups)), dtype=bool)
    lowered_mapping = {name.lower(): idx for name, idx in mapping.items()}
    for out_idx, candidates in enumerate(candidate_groups):
        source_idx = None
        for candidate in candidates:
            source_idx = lowered_mapping.get(candidate.lower())
            if source_idx is not None:
                break
        if source_idx is None:
            continue
        column = matrix[:, source_idx].astype(np.float32)
        use = np.isfinite(column) & ~np.isclose(column, missing_value)
        if np.any(use):
            values[use, out_idx] = column[use]
        valid[:, out_idx] = use
    return values, valid


def normalize_umi_gripper(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    gripper_open_abs: float,
) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float32).copy()
    use = np.asarray(valid, dtype=bool)
    raw_min = -float(gripper_open_abs)
    normalized[use] = (
        np.clip(normalized[use], raw_min, UMI_GRIPPER_RAW_MAX)
        / raw_min
    )
    normalized[use] = np.clip(normalized[use], 0.0, 1.0)
    normalized[use] = normalized[use] + 0.0
    normalized[~use] = 0.0
    return normalized.astype(np.float32)


def resolve_umi_gripper_open_abs(
    *,
    state_values: np.ndarray,
    state_valid: np.ndarray,
    action_values: np.ndarray,
    action_valid: np.ndarray,
    normalization: str,
    fixed_open_abs: float,
) -> float:
    if normalization == "fixed":
        return float(fixed_open_abs)

    valid_values = np.concatenate(
        [state_values[np.asarray(state_valid, dtype=bool)], action_values[np.asarray(action_valid, dtype=bool)]]
    ).astype(np.float64)
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size == 0:
        raise ValueError("Cannot compute data-min UMI gripper normalization: no valid values were found.")
    raw_min = float(valid_values.min())
    if raw_min >= 0.0:
        raise ValueError(
            "Cannot compute data-min UMI gripper normalization: expected a negative raw minimum, "
            f"got {raw_min}."
        )
    return -raw_min


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float32)).as_matrix().astype(np.float32)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate([matrix[:3, 0], matrix[:3, 1]], axis=0).astype(np.float32)


def quat_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(quat_to_matrix(quat_xyzw))


def build_pose_absolute_rot6d(
    pose_values: np.ndarray,
    pose_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = pose_values.shape[0]
    output = np.zeros((row_count, 9), dtype=np.float32)
    valid = np.zeros((row_count, 9), dtype=bool)

    pose_present = pose_valid.all(axis=1)
    for row_idx in np.flatnonzero(pose_present):
        output[row_idx, :3] = pose_values[row_idx, :3]
        output[row_idx, 3:9] = quat_to_rot6d(pose_values[row_idx, 3:7])
        valid[row_idx, :] = True
    return output, valid


def build_left_pose_in_right_frame(
    left_pose_values: np.ndarray,
    left_pose_valid: np.ndarray,
    right_pose_values: np.ndarray,
    right_pose_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the left TCP pose in the simultaneous right TCP frame."""
    row_count = left_pose_values.shape[0]
    output = np.zeros((row_count, 9), dtype=np.float32)
    valid = np.zeros((row_count, 9), dtype=bool)
    pose_present = left_pose_valid.all(axis=1) & right_pose_valid.all(axis=1)
    for row_idx in np.flatnonzero(pose_present):
        left_rot = quat_to_matrix(left_pose_values[row_idx, 3:7])
        right_rot = quat_to_matrix(right_pose_values[row_idx, 3:7])
        output[row_idx, :3] = right_rot.T @ (
            left_pose_values[row_idx, :3] - right_pose_values[row_idx, :3]
        )
        output[row_idx, 3:9] = matrix_to_rot6d(right_rot.T @ left_rot)
        valid[row_idx, :] = True
    return output, valid


def collect_episode_anchor_poses(
    episode_ids: np.ndarray,
    pose_values: np.ndarray,
    pose_valid: np.ndarray,
) -> dict[int, np.ndarray]:
    anchor_by_episode: dict[int, np.ndarray] = {}
    for row_idx in range(pose_values.shape[0]):
        if not pose_valid[row_idx].all():
            continue
        episode_id = int(episode_ids[row_idx])
        if episode_id not in anchor_by_episode:
            anchor_by_episode[episode_id] = pose_values[row_idx].copy()
    return anchor_by_episode


def build_pose_in_episode_anchor_frame(
    episode_ids: np.ndarray,
    pose_values: np.ndarray,
    pose_valid: np.ndarray,
    anchor_by_episode: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    row_count = pose_values.shape[0]
    output = np.zeros((row_count, 9), dtype=np.float32)
    valid = np.zeros((row_count, 9), dtype=bool)

    for row_idx in range(row_count):
        if not pose_valid[row_idx].all():
            continue
        anchor_pose = anchor_by_episode.get(int(episode_ids[row_idx]))
        if anchor_pose is None:
            continue

        anchor_rot = Rotation.from_quat(anchor_pose[3:7])
        rel_pos = pose_values[row_idx, :3] - anchor_pose[:3]
        output[row_idx, :3] = anchor_rot.apply(rel_pos, inverse=True).astype(np.float32)

        anchor_matrix = quat_to_matrix(anchor_pose[3:7])
        target_matrix = quat_to_matrix(pose_values[row_idx, 3:7])
        output[row_idx, 3:9] = matrix_to_rot6d(anchor_matrix.T @ target_matrix)
        valid[row_idx, :] = True
    return output, valid


def build_arm_absolute_targets(
    compact_pose: np.ndarray,
    compact_pose_valid: np.ndarray,
    gripper: np.ndarray,
    gripper_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action = np.zeros((compact_pose.shape[0], ARM_CANONICAL_DIM), dtype=np.float32)
    mask = np.zeros((compact_pose.shape[0], ARM_CANONICAL_DIM), dtype=np.float32)

    pose_present = compact_pose_valid.all(axis=1)
    action[pose_present, :9] = compact_pose[pose_present, :9]
    mask[pose_present, :9] = 1.0

    grip_present = gripper_valid.all(axis=1)
    action[grip_present, 9] = gripper[grip_present, 0]
    mask[grip_present, 9] = 1.0
    return action, mask


def build_previous_action(
    episode_ids: np.ndarray,
    action: np.ndarray,
    action_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prev_action = np.zeros_like(action, dtype=np.float32)
    prev_mask = np.zeros_like(action_valid_mask, dtype=np.float32)
    same_episode = episode_ids[1:] == episode_ids[:-1]
    prev_action[1:][same_episode] = action[:-1][same_episode]
    prev_mask[1:][same_episode] = action_valid_mask[:-1][same_episode]
    return prev_action, prev_mask


def select_canonical_arm(canonical: dict[str, Any], arm: str) -> dict[str, Any]:
    """Project a canonical bimanual sidecar to one arm after the raw conversion."""
    all_action_names = canonical_action_feature_names()
    canonical["action_feature_names"] = all_action_names
    canonical["canonical_arm"] = arm
    if arm == "both":
        return canonical

    arm_index = ARM_NAMES.index(arm)
    arm_start = arm_index * ARM_CANONICAL_DIM
    arm_slice = slice(arm_start, arm_start + ARM_CANONICAL_DIM)
    state_names = list(canonical["state_feature_names"])
    tactile_prefix = f"tactile_{arm}_"
    tactile_indices = [index for index, name in enumerate(state_names) if name.startswith(tactile_prefix)]
    state_indices = list(range(arm_start, arm_start + ARM_CANONICAL_DIM)) + tactile_indices

    projected = dict(canonical)
    projected[OBS_STATE] = canonical[OBS_STATE][:, state_indices].astype(np.float32)
    projected["state_mask"] = canonical["state_mask"][:, state_indices].astype(np.float32)
    projected["state_feature_names"] = [state_names[index] for index in state_indices]
    projected[ACTION] = canonical[ACTION][:, arm_slice].astype(np.float32)
    projected["action_valid_mask"] = canonical["action_valid_mask"][:, arm_slice].astype(np.float32)
    projected["action_feature_names"] = all_action_names[arm_slice]
    projected["prev_action"] = canonical["prev_action"][:, arm_slice].astype(np.float32)
    projected["prev_action_valid_mask"] = canonical["prev_action_valid_mask"][:, arm_slice].astype(np.float32)
    projected["arm_present_mask"] = canonical["arm_present_mask"][:, arm_index : arm_index + 1]
    projected["state_present"] = projected["arm_present_mask"][:, 0].astype(np.float32)
    return projected


def build_state_action_from_raw(
    *,
    frame_df: pd.DataFrame,
    info: dict[str, Any],
    dataset_type: str,
    umi_pose_frame: str,
    umi_gripper_normalization: str,
    umi_gripper_open_abs: float,
    missing_value: float,
    real_mask_arm: str = "none",
    add_inter_arm_relative_state: bool = False,
    append_state_feature_prefixes: list[str] | None = None,
) -> dict[str, np.ndarray]:
    source_state_names = list(info["features"][OBS_STATE]["names"])
    source_action_names = list(info["features"][ACTION]["names"])
    state_index = name_to_index(source_state_names)
    action_index = name_to_index(source_action_names)

    state_matrix = np.stack(frame_df[OBS_STATE].to_numpy()).astype(np.float32)
    action_matrix = np.stack(frame_df[ACTION].to_numpy()).astype(np.float32)
    episode_ids = frame_df["episode_index"].to_numpy(dtype=np.int64)

    state_arms: list[np.ndarray] = []
    state_masks: list[np.ndarray] = []
    state_present_bits: list[np.ndarray] = []
    action_arms: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    arm_anchor_counts: dict[str, int] = {}
    gripper_open_abs_by_arm: dict[str, float] = {}
    raw_state_pose_by_arm: dict[str, np.ndarray] = {}
    raw_state_pose_valid_by_arm: dict[str, np.ndarray] = {}

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
        raw_state_pose_by_arm[arm] = pose_state
        raw_state_pose_valid_by_arm[arm] = pose_state_valid
        pose_action, pose_action_valid = take_columns_with_aliases(
            action_matrix,
            action_index,
            POSE_NAME_CANDIDATES[arm],
            missing_value=missing_value,
        )
        grip_action, grip_action_valid = take_columns_with_aliases(
            action_matrix,
            action_index,
            GRIPPER_NAME_CANDIDATES[arm],
            missing_value=missing_value,
        )

        arm_gripper_open_abs = float(umi_gripper_open_abs)
        if dataset_type == "umi":
            arm_gripper_open_abs = resolve_umi_gripper_open_abs(
                state_values=grip_state,
                state_valid=grip_state_valid,
                action_values=grip_action,
                action_valid=grip_action_valid,
                normalization=umi_gripper_normalization,
                fixed_open_abs=umi_gripper_open_abs,
            )
            gripper_open_abs_by_arm[arm] = arm_gripper_open_abs
            LOGGER.info(
                "UMI gripper normalization arm=%s mode=%s raw_min=%s raw_zero=0.0",
                arm,
                umi_gripper_normalization,
                -arm_gripper_open_abs,
            )

        if dataset_type == "umi" and umi_pose_frame == "episode_anchor":
            anchor_by_episode = collect_episode_anchor_poses(
                episode_ids=episode_ids,
                pose_values=pose_state,
                pose_valid=pose_state_valid,
            )
            compact_state_pose, compact_state_pose_valid = build_pose_in_episode_anchor_frame(
                episode_ids=episode_ids,
                pose_values=pose_state,
                pose_valid=pose_state_valid,
                anchor_by_episode=anchor_by_episode,
            )
            compact_action_pose, compact_action_pose_valid = build_pose_in_episode_anchor_frame(
                episode_ids=episode_ids,
                pose_values=pose_action,
                pose_valid=pose_action_valid,
                anchor_by_episode=anchor_by_episode,
            )
            grip_state = normalize_umi_gripper(
                grip_state, grip_state_valid, gripper_open_abs=arm_gripper_open_abs
            )
            grip_action = normalize_umi_gripper(
                grip_action, grip_action_valid, gripper_open_abs=arm_gripper_open_abs
            )
            arm_anchor_counts[arm] = len(anchor_by_episode)
        else:
            compact_state_pose, compact_state_pose_valid = build_pose_absolute_rot6d(
                pose_state,
                pose_state_valid,
            )
            compact_action_pose, compact_action_pose_valid = build_pose_absolute_rot6d(
                pose_action,
                pose_action_valid,
            )
            if dataset_type == "umi":
                grip_state = normalize_umi_gripper(
                    grip_state, grip_state_valid, gripper_open_abs=arm_gripper_open_abs
                )
                grip_action = normalize_umi_gripper(
                    grip_action, grip_action_valid, gripper_open_abs=arm_gripper_open_abs
                )
            arm_anchor_counts[arm] = 0

        state_arm = np.concatenate([compact_state_pose, grip_state], axis=1).astype(np.float32)
        state_arm_mask = np.concatenate([compact_state_pose_valid, grip_state_valid], axis=1).astype(np.float32)
        state_present = (compact_state_pose_valid.all(axis=1) | grip_state_valid.all(axis=1)).astype(np.float32)
        action_arm, action_mask = build_arm_absolute_targets(
            compact_pose=compact_action_pose,
            compact_pose_valid=compact_action_pose_valid,
            gripper=grip_action,
            gripper_valid=grip_action_valid,
        )

        if dataset_type == "real" and real_mask_arm == arm:
            state_arm_mask.fill(0.0)
            state_present.fill(0.0)
            action_mask.fill(0.0)

        state_arms.append(state_arm)
        state_masks.append(state_arm_mask)
        state_present_bits.append(state_present[:, None])
        action_arms.append(action_arm)
        action_masks.append(action_mask)

    canonical_state = np.concatenate(state_arms + state_present_bits, axis=1).astype(np.float32)
    # The two arm-valid bits are metadata and are always observable.
    state_mask = np.concatenate(
        state_masks + [np.ones((canonical_state.shape[0], 2), dtype=np.float32)], axis=1
    ).astype(np.float32)
    state_feature_names = canonical_state_feature_names()
    if add_inter_arm_relative_state:
        inter_arm_pose, inter_arm_valid = build_left_pose_in_right_frame(
            raw_state_pose_by_arm["left"],
            raw_state_pose_valid_by_arm["left"],
            raw_state_pose_by_arm["right"],
            raw_state_pose_valid_by_arm["right"],
        )
        canonical_state = np.concatenate(
            [canonical_state, inter_arm_pose, np.ones((canonical_state.shape[0], 1), dtype=np.float32)],
            axis=1,
        ).astype(np.float32)
        state_mask = np.concatenate(
            [state_mask, inter_arm_valid.astype(np.float32), np.ones((state_mask.shape[0], 1), dtype=np.float32)],
            axis=1,
        ).astype(np.float32)
        state_feature_names = state_feature_names + INTER_ARM_RELATIVE_STATE_NAMES + [STATE_AVAILABLE_NAME]
    append_state_feature_prefixes = append_state_feature_prefixes or []
    if append_state_feature_prefixes:
        extra_indices = [
            index
            for index, name in enumerate(source_state_names)
            if any(name.startswith(prefix) for prefix in append_state_feature_prefixes)
        ]
        if not extra_indices:
            raise ValueError(
                "No raw observation.state features matched --append-state-feature-prefixes="
                f"{append_state_feature_prefixes!r}. Available names: {source_state_names}"
            )
        extra_state = state_matrix[:, extra_indices].astype(np.float32)
        extra_valid = np.isfinite(extra_state) & (extra_state != missing_value)
        extra_state = np.where(extra_valid, extra_state, 0.0).astype(np.float32)
        canonical_state = np.concatenate([canonical_state, extra_state], axis=1).astype(np.float32)
        state_mask = np.concatenate([state_mask, extra_valid.astype(np.float32)], axis=1).astype(np.float32)
        state_feature_names = state_feature_names + [source_state_names[index] for index in extra_indices]
    canonical_action = np.concatenate(action_arms, axis=1).astype(np.float32)
    action_valid_mask = np.concatenate(action_masks, axis=1).astype(np.float32)
    prev_action, prev_action_valid_mask = build_previous_action(
        episode_ids=episode_ids,
        action=canonical_action,
        action_valid_mask=action_valid_mask,
    )
    arm_present_mask = np.concatenate(state_present_bits, axis=1).astype(np.float32)
    state_present = (arm_present_mask.max(axis=1) > 0).astype(np.float32)

    if arm_present_mask.sum() == 0 and action_valid_mask.sum() == 0:
        raise ValueError(
            "Canonical sidecar conversion found no valid arm state/action dimensions. "
            f"robot_type={info.get('robot_type')!r}, "
            f"state_names={source_state_names}, "
            f"action_names={source_action_names}. "
            "Check raw feature name aliases in build_pi05_canonical_sidecar.py."
        )

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
        "gripper_open_abs_by_arm": gripper_open_abs_by_arm,
        "state_feature_names": state_feature_names,
        "action_feature_names": canonical_action_feature_names(),
        "canonical_arm": "both",
    }


def resolve_task_text(dataset_root: Path) -> str:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return ""
    tasks = pd.read_parquet(tasks_path)
    for column in ("vlm_hybrid_task", "task", "__index_level_0__"):
        if column in tasks.columns:
            values = tasks[column].dropna().astype(str).str.strip()
            values = values[values != ""]
            if len(values) > 0:
                return str(values.iloc[0])
    if len(tasks.index) > 0 and isinstance(tasks.index[0], str):
        return str(tasks.index[0])
    return ""


def resolve_subtask_text(row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in row.index:
            value = row[column]
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(row.name, str) and row.name.strip():
        return row.name.strip()
    return ""


def resolve_subtask_episode_index(row: pd.Series) -> int | None:
    subtask_id = row.get("id")
    if isinstance(subtask_id, str):
        match = SUBTASK_EPISODE_PATTERN.search(subtask_id)
        if match is not None:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                pass

    meta = row.get("meta")
    if isinstance(meta, dict):
        episode_index = meta.get("episode_index")
        if episode_index is not None:
            try:
                return int(episode_index)
            except (TypeError, ValueError):
                pass

    episode_index = row.get("episode_index")
    if episode_index is not None:
        try:
            return int(episode_index)
        except (TypeError, ValueError):
            pass

    return None


def build_subtask_columns(frame_df: pd.DataFrame, dataset_root: Path) -> tuple[list[str], list[str]]:
    current = np.full(len(frame_df), "", dtype=object)
    next_text = np.full(len(frame_df), "", dtype=object)
    subtasks_path = dataset_root / "meta" / "subtasks.parquet"
    if not subtasks_path.exists():
        return current.tolist(), next_text.tolist()

    subtasks = pd.read_parquet(subtasks_path)
    grouped_rows: dict[int, list[tuple[int, int, str, str]]] = {}
    for _, row in tqdm(
        subtasks.iterrows(),
        total=len(subtasks),
        desc=f"Reading subtasks {dataset_root.name}",
    ):
        frames = row.get("frames")
        if not isinstance(frames, (list, tuple, np.ndarray)) or len(frames) != 2:
            continue
        episode_index = resolve_subtask_episode_index(row)
        if episode_index is None:
            continue
        try:
            start_frame = int(frames[0])
            end_frame = int(frames[1])
        except (TypeError, ValueError):
            continue
        # The source metadata stores frame ranges as inclusive [start, end].
        # Keep that contract here so subtask labels do not drop the right boundary.
        if end_frame < start_frame:
            continue

        subtask_text = resolve_subtask_text(row, ("subtask", "instruction", "description", "skill"))
        future_text = resolve_subtask_text(row, ("next_subtask", "next_instruction", "next_description"))
        grouped_rows.setdefault(episode_index, []).append((start_frame, end_frame, subtask_text, future_text))

    episode_values = frame_df["episode_index"].to_numpy(dtype=np.int64)
    frame_values = frame_df["frame_index"].to_numpy(dtype=np.int64)
    for episode_index, entries in tqdm(
        grouped_rows.items(),
        total=len(grouped_rows),
        desc=f"Applying subtasks {dataset_root.name}",
    ):
        entries.sort(key=lambda entry: entry[0])
        base_start = entries[0][0]
        for start_frame, end_frame, subtask_text, future_text in entries:
            local_start = start_frame - base_start
            local_end = end_frame - base_start
            if local_end < local_start:
                continue
            mask = (
                (episode_values == episode_index)
                & (frame_values >= local_start)
                & (frame_values <= local_end)
            )
            current[mask] = subtask_text
            next_text[mask] = future_text
    return current.tolist(), next_text.tolist()


def merge_base_sidecar(base_sidecar: Path | None, framewise_df: pd.DataFrame) -> pd.DataFrame:
    if base_sidecar is None:
        return framewise_df
    base_df = pd.read_parquet(base_sidecar)
    if "index" not in base_df.columns:
        raise ValueError(f"Base sidecar must contain an `index` column: {base_sidecar}")

    merged = base_df.merge(framewise_df, on=JOIN_KEYS, how="outer", suffixes=("", "__canonical"))
    for column in framewise_df.columns:
        if column in JOIN_KEYS:
            continue
        override_column = f"{column}__canonical"
        if override_column in merged.columns:
            merged[column] = merged[override_column]
            merged = merged.drop(columns=[override_column])
    return merged.sort_values("index").reset_index(drop=True)


def to_list_column(array: np.ndarray) -> list[list[float]]:
    return np.asarray(array, dtype=np.float32).tolist()


def compute_masked_stats(values: np.ndarray, mask: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    stats = {key: [] for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")}
    for dim, name in enumerate(names):
        valid_values = values[:, dim][mask[:, dim]]
        if valid_values.size == 0:
            defaults = default_dim_stats(name)
            for key, value in defaults.items():
                stats[key].append(value)
            continue
        stats["min"].append(float(valid_values.min()))
        stats["max"].append(float(valid_values.max()))
        stats["mean"].append(float(valid_values.mean()))
        stats["std"].append(float(valid_values.std()))
        stats["q01"].append(float(np.quantile(valid_values, 0.01)))
        stats["q10"].append(float(np.quantile(valid_values, 0.10)))
        stats["q50"].append(float(np.quantile(valid_values, 0.50)))
        stats["q90"].append(float(np.quantile(valid_values, 0.90)))
        stats["q99"].append(float(np.quantile(valid_values, 0.99)))
    output = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}
    output["count"] = np.asarray([values.shape[0]], dtype=np.float32)
    return output


def default_dim_stats(name: str) -> dict[str, float]:
    if name.endswith("_valid") or name.endswith("_present"):
        return {"min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0, "q01": 0.0, "q10": 0.0, "q50": 0.0, "q90": 1.0, "q99": 1.0}
    if "gripper" in name:
        return {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0, "q01": -1.0, "q10": -1.0, "q50": 0.0, "q90": 1.0, "q99": 1.0}
    return {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0, "q01": -1.0, "q10": -0.5, "q50": 0.0, "q90": 0.5, "q99": 1.0}


def compute_framewise_stats(canonical: dict[str, np.ndarray], base_stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = deepcopy(base_stats) if isinstance(base_stats, dict) else {}
    state_valid_mask = canonical["state_mask"]
    stats[OBS_STATE] = compute_masked_stats(
        canonical[OBS_STATE],
        state_valid_mask,
        canonical["state_feature_names"],
    )
    stats[ACTION] = compute_masked_stats(
        canonical[ACTION],
        canonical["action_valid_mask"],
        canonical["action_feature_names"],
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


def parse_relative_action_keyframes(spec: str) -> list[tuple[int, int]]:
    keyframes: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pos_start_str, rot_start_str = part.split(":", 1)
            keyframes.append((int(pos_start_str), int(rot_start_str)))
        else:
            raise ValueError(
                f"Invalid --relative-action-keyframes entry '{part}'. Expected `pos_start:rot6d_start`."
            )
    return keyframes


def parse_csv_tokens(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def relative_mask_from_action_names(action_names: list[str], exclude_joints: list[str]) -> np.ndarray:
    exclude_tokens = [token.lower() for token in exclude_joints if token]
    if not exclude_tokens:
        return np.ones(len(action_names), dtype=np.float32)
    mask = np.ones(len(action_names), dtype=np.float32)
    for idx, name in enumerate(action_names):
        action_name = str(name).lower()
        if any(token == action_name or token in action_name for token in exclude_tokens):
            mask[idx] = 0.0
    return mask


def rot6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    col1 = rot6d[..., 0:3]
    col2 = rot6d[..., 3:6]
    basis1 = col1 / np.clip(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8, None)
    proj = np.sum(basis1 * col2, axis=-1, keepdims=True) * basis1
    basis2_raw = col2 - proj
    basis2 = basis2_raw / np.clip(np.linalg.norm(basis2_raw, axis=-1, keepdims=True), 1e-8, None)
    basis3 = np.cross(basis1, basis2, axis=-1)
    return np.stack((basis1, basis2, basis3), axis=-1).astype(np.float32)


def apply_keyframe_relative_actions(
    chunk_actions: np.ndarray,
    state: np.ndarray,
    keyframes: list[tuple[int, int]],
) -> np.ndarray:
    output = chunk_actions.copy()
    if state.ndim == 1:
        state_ref = state[None, :]
    else:
        state_ref = state
    if output.ndim == 2:
        output_seq = output[None, :, :]
        squeeze_batch = True
    else:
        output_seq = output
        squeeze_batch = False
    for pos_start, rot_start in keyframes:
        pos_slice = slice(pos_start, pos_start + 3)
        rot_slice = slice(rot_start, rot_start + 6)
        required_dim = max(pos_start + 3, rot_start + 6)
        if output_seq.shape[-1] < required_dim or state_ref.shape[-1] < required_dim:
            continue
        current_pos = state_ref[:, pos_slice]
        current_rot = rot6d_to_matrix_np(state_ref[:, rot_slice])
        target_pos = output_seq[:, :, pos_slice]
        target_rot = rot6d_to_matrix_np(output_seq[:, :, rot_slice])
        current_rot_t = np.swapaxes(current_rot, -1, -2)
        rel_pos = np.matmul(current_rot_t[:, None, ...], (target_pos - current_pos[:, None, :])[..., None])
        rel_rot = np.matmul(current_rot_t[:, None, ...], target_rot)
        output_seq[:, :, pos_slice] = rel_pos.squeeze(-1)
        output_seq[:, :, rot_slice] = np.concatenate(
            [rel_rot[..., :, 0], rel_rot[..., :, 1]],
            axis=-1,
        ).astype(np.float32)
    if squeeze_batch:
        return output_seq[0]
    return output_seq


def compute_relative_action_chunk_stats(
    *,
    episode_ids: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    action_valid_mask: np.ndarray,
    chunk_size: int,
    action_gap_frames: int,
    relative_keyframes: list[tuple[int, int]],
    relative_exclude_joints: list[str],
    action_names: list[str],
) -> dict[str, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if action_gap_frames < 0:
        raise ValueError("action_gap_frames must be non-negative")

    episode_last_index: dict[int, int] = {}
    for idx, episode_id in enumerate(episode_ids):
        episode_last_index[int(episode_id)] = idx

    episode_last_indices = np.asarray(
        [episode_last_index[int(episode_id)] for episode_id in episode_ids],
        dtype=np.int64,
    )
    relative_values: list[np.ndarray] = []
    relative_valid: list[np.ndarray] = []
    total = len(actions)
    block_size = 4096
    chunk_offsets = np.arange(chunk_size, dtype=np.int64)
    action_dim = actions.shape[1]
    relative_mask = relative_mask_from_action_names(action_names, relative_exclude_joints)
    for pos_start, rot_start in relative_keyframes:
        relative_mask[pos_start : min(pos_start + 3, action_dim)] = 0.0
        relative_mask[rot_start : min(rot_start + 6, action_dim)] = 0.0
    for block_start in range(0, total, block_size):
        block_end = min(block_start + block_size, total)
        start_indices = np.arange(block_start, block_end, dtype=np.int64)
        query_indices = start_indices[:, None] + action_gap_frames + chunk_offsets[None, :]
        query_indices = np.minimum(query_indices, episode_last_indices[start_indices, None])

        chunk_actions = actions[query_indices].copy()
        chunk_masks = action_valid_mask[query_indices].copy()
        chunk_actions = apply_keyframe_relative_actions(
            chunk_actions,
            states[start_indices, :action_dim],
            relative_keyframes,
        )
        mask_dim = min(action_dim, states.shape[1])
        chunk_actions[:, :, :mask_dim] -= states[start_indices, None, :mask_dim] * relative_mask[:mask_dim]
        relative_values.append(chunk_actions.reshape(-1, action_dim))
        relative_valid.append(chunk_masks.reshape(-1, action_dim))

    if not relative_values:
        raise RuntimeError("No relative action chunks were available for stats computation.")

    values = np.concatenate(relative_values, axis=0)
    valid = np.concatenate(relative_valid, axis=0)
    return compute_masked_stats(values, valid, action_names)


def load_base_sidecar_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(key): Path(value) for key, value in payload.items()}


def resolve_base_sidecar(
    *,
    dataset_root: Path,
    single_base_sidecar: Path | None,
    base_sidecar_map: dict[str, Path],
) -> Path | None:
    if single_base_sidecar is not None:
        return single_base_sidecar
    candidates = [
        str(dataset_root),
        dataset_root.name,
        str(dataset_root.resolve()),
    ]
    for candidate in candidates:
        if candidate in base_sidecar_map:
            return base_sidecar_map[candidate]
    return None


def build_sidecar_for_dataset(
    *,
    dataset_root: Path,
    requested_dataset_type: str,
    umi_pose_frame: str,
    umi_gripper_normalization: str,
    umi_gripper_open_abs: float,
    real_mask_arm: str,
    canonical_arm: str,
    output_root: Path,
    schema_name: str,
    missing_value: float,
    base_sidecar: Path | None,
    relative_stats_chunk_size: int,
    relative_stats_action_gap_frames: int,
    relative_action_keyframes: str,
    relative_exclude_joints: str,
    overwrite: bool,
    add_inter_arm_relative_state: bool,
    append_state_feature_prefixes: list[str],
) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")

    info = read_json(info_path)
    dataset_type = infer_dataset_type(info, requested_dataset_type, dataset_root)
    output_dir = output_root / dataset_root.name
    sidecar_path = output_dir / "framewise_guidance.parquet"
    stats_output_path = output_dir / "stats.json"
    summary_path = output_dir / "summary.json"
    if not overwrite:
        for path in (sidecar_path, stats_output_path, summary_path):
            if path.exists():
                raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")

    LOGGER.info(
        "processing dataset=%s requested_type=%s resolved_type=%s robot_type=%s",
        dataset_root,
        requested_dataset_type,
        dataset_type,
        info.get("robot_type"),
    )
    if dataset_type == "real" and real_mask_arm != "none":
        LOGGER.info("masking arm for real dataset: %s", real_mask_arm)
    base_stats = read_json(stats_path) if stats_path.exists() else {}
    frame_df = load_frame_table(dataset_root)
    canonical = build_state_action_from_raw(
        frame_df=frame_df,
        info=info,
        dataset_type=dataset_type,
        umi_pose_frame=umi_pose_frame,
        umi_gripper_normalization=umi_gripper_normalization,
        umi_gripper_open_abs=umi_gripper_open_abs,
        missing_value=missing_value,
        real_mask_arm=real_mask_arm,
        add_inter_arm_relative_state=add_inter_arm_relative_state,
        append_state_feature_prefixes=append_state_feature_prefixes,
    )
    canonical = select_canonical_arm(canonical, canonical_arm)

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
            "canonical_dataset_type": dataset_type,
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
            action_names=canonical["action_feature_names"],
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
    state_valid_ratio = canonical["state_mask"].mean(axis=0)
    summary = {
        "dataset_root": str(dataset_root),
        "dataset_type": dataset_type,
        "umi_pose_frame": umi_pose_frame if dataset_type == "umi" else None,
        "real_mask_arm": real_mask_arm if dataset_type == "real" else "none",
        "robot_type": info.get("robot_type"),
        "schema_name": schema_name,
        "sidecar_path": str(sidecar_path),
        "stats_path": str(stats_output_path),
        "relative_stats_path": str(relative_stats_path) if relative_stats_path is not None else None,
        "base_sidecar": str(base_sidecar) if base_sidecar is not None else None,
        "row_count": int(len(framewise_df)),
        "canonical_state_dim": int(canonical[OBS_STATE].shape[1]),
        "canonical_action_dim": int(canonical[ACTION].shape[1]),
        "canonical_state_names": canonical["state_feature_names"],
        "canonical_action_names": canonical["action_feature_names"],
        "canonical_arm": canonical["canonical_arm"],
        "state_pose_mode": (
            "episode_anchor_relative_rot6d"
            if dataset_type == "umi" and umi_pose_frame == "episode_anchor"
            else "absolute_rot6d"
        ),
        "gripper_mode": (
            "per_arm_data_min_to_1_zero_to_0"
            if dataset_type == "umi" and umi_gripper_normalization == "data_min"
            else f"clip_raw_to_[{-umi_gripper_open_abs},0.0]_divide_by_{-umi_gripper_open_abs}"
            if dataset_type == "umi"
            else "framewise_raw"
        ),
        "umi_gripper_normalization": umi_gripper_normalization if dataset_type == "umi" else None,
        "umi_gripper_open_abs": (
            canonical["gripper_open_abs_by_arm"] if dataset_type == "umi" else None
        ),
        "action_target_mode": (
            "episode_anchor_relative_rot6d_gripper_framewise"
            if dataset_type == "umi" and umi_pose_frame == "episode_anchor"
            else "absolute_rot6d_gripper_framewise"
        ),
        "missing_value": missing_value,
        "left_arm_present_ratio": float(canonical["left_present"].mean()),
        "right_arm_present_ratio": float(canonical["right_present"].mean()),
        "mean_action_valid_ratio": float(canonical["action_valid_mask"].mean()),
        "action_valid_ratio_by_dim": action_valid_ratio.astype(float).tolist(),
        "state_valid_ratio_by_dim": state_valid_ratio.astype(float).tolist(),
        "arm_anchor_counts": canonical["arm_anchor_counts"],
        "task_vlm_caption_nonempty": bool(task_text.strip()),
        "subtask_text_nonempty_ratio": float((pd.Series(subtask_text).astype(str).str.strip() != "").mean()),
        "next_subtask_text_nonempty_ratio": float((pd.Series(next_subtask_text).astype(str).str.strip() != "").mean()),
        "recommended_policy": {
            "canonicalize_bimanual_rot6d": False,
            "use_relative_actions": umi_pose_frame != "absolute",
            "relative_action_keyframes": (
                []
                if umi_pose_frame == "absolute"
                else [item.strip() for item in relative_action_keyframes.split(",") if item.strip()]
            ),
            "relative_exclude_joints": (
                [] if umi_pose_frame == "absolute" else parse_csv_tokens(relative_exclude_joints)
            ),
            "action_gap_frames": relative_stats_action_gap_frames,
            "use_state_input": True,
            "action_valid_mask_key": "action_valid_mask",
            "state_present_key": "state_present",
            "max_state_dim": int(canonical[OBS_STATE].shape[1]),
            "max_action_dim": int(canonical[ACTION].shape[1]),
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
    for dataset_root in tqdm(dataset_roots, desc="Building sidecars"):
        base_sidecar = resolve_base_sidecar(
            dataset_root=dataset_root,
            single_base_sidecar=args.base_sidecar,
            base_sidecar_map=base_sidecar_map,
        )
        summaries.append(
            build_sidecar_for_dataset(
                dataset_root=dataset_root,
                requested_dataset_type=args.dataset_type,
                umi_pose_frame=args.umi_pose_frame,
                umi_gripper_normalization=args.umi_gripper_normalization,
                umi_gripper_open_abs=args.umi_gripper_open_abs,
                real_mask_arm=args.real_mask_arm,
                canonical_arm=args.canonical_arm,
                output_root=args.output_root,
                schema_name=args.schema_name,
                missing_value=args.missing_value,
                base_sidecar=base_sidecar,
                relative_stats_chunk_size=args.relative_stats_chunk_size,
                relative_stats_action_gap_frames=args.relative_stats_action_gap_frames,
                relative_action_keyframes=args.relative_action_keyframes,
                relative_exclude_joints=args.relative_exclude_joints,
                overwrite=args.overwrite,
                add_inter_arm_relative_state=args.add_inter_arm_relative_state,
                append_state_feature_prefixes=parse_csv_tokens(args.append_state_feature_prefixes),
            )
        )

    manifest = {
        "schema_name": args.schema_name,
        "requested_dataset_type": args.dataset_type,
        "umi_pose_frame": args.umi_pose_frame,
        "umi_gripper_normalization": args.umi_gripper_normalization,
        "umi_gripper_open_abs": args.umi_gripper_open_abs,
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
