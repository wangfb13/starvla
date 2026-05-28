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
POSE_NAMES = {
    "left": ["left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw"],
    "right": ["right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw"],
}
GRIPPER_NAMES = {"left": "gripper_left", "right": "gripper_right"}
SUBTASK_EPISODE_PATTERN = re.compile(r"_ep(\d+)_")
DEFAULT_SCHEMA_NAME = "pi05_canonical_bimanual_rot6d_v1"
REAL_ROBOT_TYPES = {"spark3_pose_gripper"}
UMI_ROBOT_TYPES = {"ugripper_left", "ugripper_right", "dual_ugripper"}


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
    for path in parquet_paths:
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


def build_state_action_from_raw(
    *,
    frame_df: pd.DataFrame,
    info: dict[str, Any],
    dataset_type: str,
    missing_value: float,
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

    for arm in ARM_NAMES:
        pose_state, pose_state_valid = take_columns(
            state_matrix,
            state_index,
            POSE_NAMES[arm],
            missing_value=missing_value,
        )
        grip_state, grip_state_valid = take_columns(
            state_matrix,
            state_index,
            [GRIPPER_NAMES[arm]],
            missing_value=missing_value,
        )
        pose_action, pose_action_valid = take_columns(
            action_matrix,
            action_index,
            POSE_NAMES[arm],
            missing_value=missing_value,
        )
        grip_action, grip_action_valid = take_columns(
            action_matrix,
            action_index,
            [GRIPPER_NAMES[arm]],
            missing_value=missing_value,
        )

        if dataset_type == "umi":
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


def build_subtask_columns(frame_df: pd.DataFrame, dataset_root: Path) -> tuple[list[str], list[str]]:
    current = np.full(len(frame_df), "", dtype=object)
    next_text = np.full(len(frame_df), "", dtype=object)
    subtasks_path = dataset_root / "meta" / "subtasks.parquet"
    if not subtasks_path.exists():
        return current.tolist(), next_text.tolist()

    subtasks = pd.read_parquet(subtasks_path)
    grouped_rows: dict[int, list[tuple[int, int, str, str]]] = {}
    for _, row in subtasks.iterrows():
        subtask_id = row.get("id")
        if not isinstance(subtask_id, str):
            continue
        match = SUBTASK_EPISODE_PATTERN.search(subtask_id)
        if match is None:
            continue
        frames = row.get("frames")
        if not isinstance(frames, (list, tuple, np.ndarray)) or len(frames) != 2:
            continue
        try:
            episode_index = int(match.group(1))
            start_frame = int(frames[0])
            end_frame = int(frames[1])
        except (TypeError, ValueError):
            continue
        if end_frame <= start_frame:
            continue

        subtask_text = resolve_subtask_text(row, ("subtask", "instruction", "description", "skill"))
        future_text = resolve_subtask_text(row, ("next_subtask", "next_instruction", "next_description"))
        grouped_rows.setdefault(episode_index, []).append((start_frame, end_frame, subtask_text, future_text))

    episode_values = frame_df["episode_index"].to_numpy(dtype=np.int64)
    frame_values = frame_df["frame_index"].to_numpy(dtype=np.int64)
    for episode_index, entries in grouped_rows.items():
        entries.sort(key=lambda entry: entry[0])
        base_start = entries[0][0]
        for start_frame, end_frame, subtask_text, future_text in entries:
            local_start = start_frame - base_start
            local_end = end_frame - base_start
            if local_end <= local_start:
                continue
            mask = (
                (episode_values == episode_index)
                & (frame_values >= local_start)
                & (frame_values < local_end)
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
    state_valid_mask = np.concatenate(
        [
            canonical["state_mask"],
            np.ones((canonical[OBS_STATE].shape[0], 2), dtype=np.float32),
        ],
        axis=1,
    )
    stats[OBS_STATE] = compute_masked_stats(
        canonical[OBS_STATE],
        state_valid_mask,
        canonical_state_feature_names(),
    )
    stats[ACTION] = compute_masked_stats(
        canonical[ACTION],
        canonical["action_valid_mask"],
        canonical_action_feature_names(),
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


def parse_slices(spec: str, dim: int) -> np.ndarray:
    mask = np.zeros((dim,), dtype=np.float32)
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_str, end_str = part.split(":", 1)
            start = int(start_str)
            end = int(end_str)
            mask[start:end] = 1.0
        else:
            mask[int(part)] = 1.0
    return mask


def compute_relative_action_chunk_stats(
    *,
    episode_ids: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    action_valid_mask: np.ndarray,
    chunk_size: int,
    action_gap_frames: int,
    relative_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if action_gap_frames < 0:
        raise ValueError("action_gap_frames must be non-negative")

    episode_last_index: dict[int, int] = {}
    for idx, episode_id in enumerate(episode_ids):
        episode_last_index[int(episode_id)] = idx

    relative_values: list[np.ndarray] = []
    relative_valid: list[np.ndarray] = []
    total = len(actions)
    for start_idx in range(total):
        ep = int(episode_ids[start_idx])
        ep_last_idx = episode_last_index[ep]
        query_indices = start_idx + action_gap_frames + np.arange(chunk_size)
        query_indices = np.minimum(query_indices, ep_last_idx)

        chunk_actions = actions[query_indices].copy()
        chunk_masks = action_valid_mask[query_indices].copy()
        chunk_actions -= states[start_idx, :CANONICAL_ACTION_DIM] * relative_mask
        relative_values.append(chunk_actions)
        relative_valid.append(chunk_masks)

    if not relative_values:
        raise RuntimeError("No relative action chunks were available for stats computation.")

    values = np.concatenate(relative_values, axis=0)
    valid = np.concatenate(relative_valid, axis=0)
    return compute_masked_stats(values, valid, canonical_action_feature_names())


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
    output_root: Path,
    schema_name: str,
    missing_value: float,
    base_sidecar: Path | None,
    relative_stats_chunk_size: int,
    relative_stats_action_gap_frames: int,
    relative_action_keyframes: str,
    overwrite: bool,
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
    base_stats = read_json(stats_path) if stats_path.exists() else {}
    frame_df = load_frame_table(dataset_root)
    canonical = build_state_action_from_raw(
        frame_df=frame_df,
        info=info,
        dataset_type=dataset_type,
        missing_value=missing_value,
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
        relative_mask = parse_slices(relative_action_keyframes, CANONICAL_ACTION_DIM)
        relative_action_stats = compute_relative_action_chunk_stats(
            episode_ids=frame_df["episode_index"].to_numpy(dtype=np.int64),
            states=canonical[OBS_STATE],
            actions=canonical[ACTION],
            action_valid_mask=canonical["action_valid_mask"],
            chunk_size=relative_stats_chunk_size,
            action_gap_frames=relative_stats_action_gap_frames,
            relative_mask=relative_mask,
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
        "dataset_type": dataset_type,
        "robot_type": info.get("robot_type"),
        "schema_name": schema_name,
        "sidecar_path": str(sidecar_path),
        "stats_path": str(stats_output_path),
        "relative_stats_path": str(relative_stats_path) if relative_stats_path is not None else None,
        "base_sidecar": str(base_sidecar) if base_sidecar is not None else None,
        "row_count": int(len(framewise_df)),
        "canonical_state_dim": CANONICAL_STATE_DIM,
        "canonical_action_dim": CANONICAL_ACTION_DIM,
        "canonical_state_names": canonical_state_feature_names(),
        "canonical_action_names": canonical_action_feature_names(),
        "state_pose_mode": "episode_anchor_relative_rot6d" if dataset_type == "umi" else "absolute_rot6d",
        "action_target_mode": "episode_anchor_relative_rot6d_gripper_absolute"
        if dataset_type == "umi"
        else "absolute_rot6d_gripper_absolute",
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
            "use_relative_actions": True,
            "relative_action_keyframes": [
                item.strip() for item in relative_action_keyframes.split(",") if item.strip()
            ],
            "action_gap_frames": relative_stats_action_gap_frames,
            "use_state_input": True,
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
    for dataset_root in dataset_roots:
        base_sidecar = resolve_base_sidecar(
            dataset_root=dataset_root,
            single_base_sidecar=args.base_sidecar,
            base_sidecar_map=base_sidecar_map,
        )
        summaries.append(
            build_sidecar_for_dataset(
                dataset_root=dataset_root,
                requested_dataset_type=args.dataset_type,
                output_root=args.output_root,
                schema_name=args.schema_name,
                missing_value=args.missing_value,
                base_sidecar=base_sidecar,
                relative_stats_chunk_size=args.relative_stats_chunk_size,
                relative_stats_action_gap_frames=args.relative_stats_action_gap_frames,
                relative_action_keyframes=args.relative_action_keyframes,
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "schema_name": args.schema_name,
        "requested_dataset_type": args.dataset_type,
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
