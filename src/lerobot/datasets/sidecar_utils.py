#!/usr/bin/env python

"""Utilities for loading framewise sidecars and sidecar stats."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.io_utils import cast_stats_to_numpy
from lerobot.utils.constants import ACTION, OBS_ACTION_MASK, OBS_STATE, OBS_STATE_MASK

LOGGER = logging.getLogger(__name__)

REAL_ROBOT_TYPES = {"spark3_pose_gripper"}
UMI_ROBOT_TYPES = {"ugripper_left", "ugripper_right", "dual_ugripper", "dual_ugripper_v2_5"}

_VECTOR_ALIAS_KEYS = {
    OBS_STATE_MASK: "state_mask",
    OBS_ACTION_MASK: "action_valid_mask",
}


def read_dataset_roots_file(path: str | Path) -> list[Path]:
    roots: list[Path] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            root = Path(line).expanduser()
            if not root.is_absolute():
                root = (path.parent / root).resolve()
            roots.append(root)
            LOGGER.debug("loaded dataset root from %s:%d -> %s", path, line_number, root)
    return roots


def dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            LOGGER.warning("skip duplicated dataset root: %s", path)
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def resolve_sidecar_path(
    *,
    dataset_root: str | Path,
    sidecar_root: str | Path | None = None,
    sidecar_path: str | Path | dict[str, str | Path] | None = None,
) -> Path | None:
    dataset_root = Path(dataset_root)
    if isinstance(sidecar_path, dict):
        for key in (str(dataset_root), dataset_root.name, str(dataset_root.resolve())):
            value = sidecar_path.get(key)
            if value is not None:
                return Path(value)
    elif sidecar_path is not None:
        return Path(sidecar_path)

    if sidecar_root is None:
        return None

    return Path(sidecar_root) / dataset_root.name / "framewise_guidance.parquet"


def resolve_sidecar_stats_path(
    *,
    dataset_root: str | Path,
    sidecar_root: str | Path | None = None,
    sidecar_path: str | Path | dict[str, str | Path] | None = None,
    stats_path: str | Path | dict[str, str | Path] | None = None,
    relative_stats_path: str | Path | dict[str, str | Path] | None = None,
    relative_chunk_size: int | None = None,
    relative_action_gap_frames: int | None = None,
    use_relative_actions: bool = False,
) -> Path | None:
    dataset_root = Path(dataset_root)
    explicit_path = relative_stats_path if use_relative_actions else stats_path
    if isinstance(explicit_path, dict):
        for key in (str(dataset_root), dataset_root.name, str(dataset_root.resolve())):
            value = explicit_path.get(key)
            if value is not None:
                return Path(value)
    elif explicit_path is not None:
        return Path(explicit_path)

    if sidecar_root is None:
        resolved_sidecar_path = resolve_sidecar_path(
            dataset_root=dataset_root,
            sidecar_path=sidecar_path,
        )
        if resolved_sidecar_path is None:
            return None
        sidecar_dir = resolved_sidecar_path.parent if resolved_sidecar_path.suffix else resolved_sidecar_path
    else:
        sidecar_dir = Path(sidecar_root) / dataset_root.name

    if use_relative_actions and relative_chunk_size and relative_action_gap_frames is not None:
        return sidecar_dir / f"stats_relative_action_chunk{relative_chunk_size}_gap{relative_action_gap_frames}.json"
    return sidecar_dir / "stats.json"


def load_stats_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        LOGGER.warning("Sidecar stats path does not exist: %s", path)
        return None
    with path.open("r", encoding="utf-8") as f:
        return cast_stats_to_numpy(json.load(f))


def filter_sidecar_normalization_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only stats used by PI0.5 canonical sidecar normalization."""
    if stats is None:
        return None
    filtered = {key: stats[key] for key in (OBS_STATE, ACTION) if key in stats}
    return filtered or None


def infer_gap_override_from_robot_type(robot_type: str | None) -> int | None:
    """Return the default action gap override implied by dataset robot type.

    UMI datasets keep the policy-level gap because their state/action delay is
    handled by future action sampling. Real robot datasets already encode that
    delay in the recorded state/action pair, so they should use gap 0.
    """
    if robot_type in REAL_ROBOT_TYPES:
        return 0
    if robot_type in UMI_ROBOT_TYPES:
        return None
    return None


def default_dim_stats(name: str) -> dict[str, float]:
    if name.endswith("_valid") or name.endswith("_present"):
        return {
            "min": 0.0,
            "max": 1.0,
            "mean": 0.0,
            "std": 1.0,
            "q01": 0.0,
            "q10": 0.0,
            "q50": 0.0,
            "q90": 1.0,
            "q99": 1.0,
        }
    if "gripper" in name:
        return {
            "min": -1.0,
            "max": 1.0,
            "mean": 0.0,
            "std": 1.0,
            "q01": -1.0,
            "q10": -1.0,
            "q50": 0.0,
            "q90": 1.0,
            "q99": 1.0,
        }
    return {
        "min": -1.0,
        "max": 1.0,
        "mean": 0.0,
        "std": 1.0,
        "q01": -1.0,
        "q10": -0.5,
        "q50": 0.0,
        "q90": 0.5,
        "q99": 1.0,
    }


def compute_masked_vector_stats(
    values: np.ndarray,
    mask: np.ndarray,
    names: list[str],
) -> dict[str, np.ndarray]:
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


def _relative_mask_from_action_names(action_dim: int, action_names: list[str], exclude_joints: list[str]) -> np.ndarray:
    exclude_tokens = [str(name).lower() for name in exclude_joints if name]
    if not exclude_tokens:
        return np.ones((action_dim,), dtype=np.float32)

    mask: list[float] = []
    for name in action_names[:action_dim]:
        action_name = str(name).lower()
        is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
        mask.append(0.0 if is_excluded else 1.0)

    if len(mask) < action_dim:
        mask.extend([1.0] * (action_dim - len(mask)))
    return np.asarray(mask, dtype=np.float32)


def _parse_relative_keyframe_spec(spec: str) -> tuple[int, int]:
    parts = str(spec).split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid relative_action_keyframes entry '{spec}'. Expected 'pos_start:rot6d_start'.")
    return int(parts[0]), int(parts[1])


def _relative_keyframe_specs(
    keyframes: list[str] | None,
    exclude_keys: list[str] | None = None,
) -> list[tuple[int, int]]:
    excluded = {str(item) for item in (exclude_keys or [])}
    specs: list[tuple[int, int]] = []
    for spec in keyframes or []:
        if str(spec) in excluded:
            continue
        specs.append(_parse_relative_keyframe_spec(str(spec)))
    return specs


def _rot6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    col1 = rot6d[..., 0:3]
    col2 = rot6d[..., 3:6]
    basis1 = col1 / np.clip(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8, None)
    proj = np.sum(basis1 * col2, axis=-1, keepdims=True) * basis1
    basis2_raw = col2 - proj
    basis2 = basis2_raw / np.clip(np.linalg.norm(basis2_raw, axis=-1, keepdims=True), 1e-8, None)
    basis3 = np.cross(basis1, basis2, axis=-1)
    return np.stack((basis1, basis2, basis3), axis=-1).astype(np.float32)


def _matrix_to_rot6d_np(rot_mats: np.ndarray) -> np.ndarray:
    rot_mats = np.asarray(rot_mats, dtype=np.float32)
    return np.concatenate((rot_mats[..., :, 0], rot_mats[..., :, 1]), axis=-1).astype(np.float32)


def _apply_keyframe_relative_np(
    chunk_actions: np.ndarray,
    state: np.ndarray,
    keyframe_specs: list[tuple[int, int]],
) -> np.ndarray:
    if not keyframe_specs:
        return chunk_actions
    output = chunk_actions.copy()
    state = np.asarray(state, dtype=np.float32)
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
    for pos_start, rot_start in keyframe_specs:
        pos_slice = slice(pos_start, pos_start + 3)
        rot_slice = slice(rot_start, rot_start + 6)
        required_dim = max(pos_start + 3, rot_start + 6)
        if output_seq.shape[-1] < required_dim or state_ref.shape[-1] < required_dim:
            continue
        current_pos = state_ref[:, pos_slice]
        current_rot = _rot6d_to_matrix_np(state_ref[:, rot_slice])
        target_pos = output_seq[:, :, pos_slice]
        target_rot = _rot6d_to_matrix_np(output_seq[:, :, rot_slice])
        current_rot_t = np.swapaxes(current_rot, -1, -2)
        rel_pos = np.matmul(current_rot_t[:, None, ...], (target_pos - current_pos[:, None, :])[..., None])
        rel_rot = np.matmul(current_rot_t[:, None, ...], target_rot)
        output_seq[:, :, pos_slice] = rel_pos.squeeze(-1)
        output_seq[:, :, rot_slice] = _matrix_to_rot6d_np(rel_rot)
    if squeeze_batch:
        return output_seq[0]
    return output_seq


def _default_value_for_series(series: pd.Series) -> object:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return False
    if pd.api.types.is_integer_dtype(dtype):
        return 0
    if pd.api.types.is_float_dtype(dtype):
        return 0.0
    return ""


def sidecar_value_is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, (np.ndarray, pd.Series, list, tuple)):
        return bool(np.all(missing))
    return bool(missing)


def materialize_sidecar_value(value: object, *, vector_dim: int | None = None) -> object:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return torch.as_tensor(value.copy())
    if isinstance(value, (list, tuple)):
        if vector_dim is not None:
            return torch.as_tensor(value, dtype=torch.float32)
        return value
    return value


class FramewiseSidecar:
    def __init__(self, path: str | Path | None, repo_id: str | None = None):
        self.path = Path(path) if path is not None else None
        self.repo_id = repo_id
        self._loaded = False
        self._frame_df: pd.DataFrame | None = None
        self._default_values: dict[str, object] = {}
        self._vector_dims: dict[str, int] = {}
        self._vector_stats: dict[str, dict[str, np.ndarray]] = {}

    @property
    def default_values(self) -> dict[str, object]:
        self.ensure_loaded()
        return self._default_values

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if self.path is None:
            return
        if not self.path.exists():
            LOGGER.warning("Sidecar path for %s does not exist: %s", self.repo_id, self.path)
            return

        sidecar_df = pd.read_parquet(self.path)
        if "index" not in sidecar_df.columns:
            raise ValueError(f"Sidecar parquet must contain an 'index' column: {self.path}")

        sidecar_df = sidecar_df.sort_values("index").reset_index(drop=True)
        if sidecar_df["index"].duplicated().any():
            raise ValueError(f"Sidecar parquet contains duplicated 'index' values: {self.path}")
        sidecar_df = sidecar_df.set_index("index", drop=False)

        for column in sidecar_df.columns:
            if column in {"index", "episode_index", "frame_index"}:
                continue
            self._default_values[column] = _default_value_for_series(sidecar_df[column])

        alias_defaults = {
            "subtask_text_sidecar": "subtask",
            "next_subtask_text_sidecar": "next_subtask",
            "user_prompt_sidecar": "user_prompt",
            "state_mask": OBS_STATE_MASK,
        }
        for source_key, alias_key in alias_defaults.items():
            if source_key in sidecar_df.columns:
                self._default_values.setdefault(alias_key, "")

        self._frame_df = sidecar_df

    def vector_dim(self, key: str) -> int | None:
        self.ensure_loaded()
        if key in _VECTOR_ALIAS_KEYS and (self._frame_df is None or key not in self._frame_df.columns):
            return self.vector_dim(_VECTOR_ALIAS_KEYS[key])
        if key in self._vector_dims:
            return self._vector_dims[key]
        if self._frame_df is None or key not in self._frame_df.columns:
            return None
        for value in self._frame_df[key]:
            if sidecar_value_is_missing(value):
                continue
            if torch.is_tensor(value):
                dim = int(value.numel())
            elif isinstance(value, np.ndarray):
                dim = int(value.size)
            elif isinstance(value, (list, tuple)):
                dim = int(np.asarray(value).size)
            else:
                return None
            self._vector_dims[key] = dim
            return dim
        return None

    def vector_stats(self, key: str) -> dict[str, np.ndarray] | None:
        self.ensure_loaded()
        if key in self._vector_stats:
            return self._vector_stats[key]
        if self._frame_df is None or key not in self._frame_df.columns:
            return None

        rows: list[np.ndarray] = []
        for value in self._frame_df[key]:
            if sidecar_value_is_missing(value):
                continue
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            rows.append(arr.astype(np.float32).reshape(-1))

        if not rows:
            return None

        matrix = np.stack(rows, axis=0)
        stats = get_feature_stats(matrix, axis=0, keepdims=False)
        self._vector_stats[key] = stats
        self._vector_dims.setdefault(key, int(matrix.shape[1]))
        return stats

    def masked_vector_stats(
        self,
        key: str,
        mask_key: str,
        names: list[str],
    ) -> dict[str, np.ndarray] | None:
        values = self._values_matrix(key)
        if values is None:
            return None

        mask = self._values_matrix(mask_key)
        if mask is None:
            return self.vector_stats(key)

        if mask.shape[1] < values.shape[1]:
            pad = np.ones((mask.shape[0], values.shape[1] - mask.shape[1]), dtype=np.float32)
            mask = np.concatenate([mask, pad], axis=1)
        elif mask.shape[1] > values.shape[1]:
            mask = mask[:, : values.shape[1]]

        stat_names = list(names[: values.shape[1]])
        if len(stat_names) < values.shape[1]:
            stat_names.extend([f"{key}_{idx}" for idx in range(len(stat_names), values.shape[1])])
        return compute_masked_vector_stats(values, mask, stat_names)

    def _values_matrix(self, key: str) -> np.ndarray | None:
        self.ensure_loaded()
        if self._frame_df is None or key not in self._frame_df.columns:
            return None

        rows: list[np.ndarray] = []
        for value in self._frame_df[key]:
            if sidecar_value_is_missing(value):
                return None
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            rows.append(arr.astype(np.float32).reshape(-1))
        if not rows:
            return None
        return np.stack(rows, axis=0)

    def relative_action_chunk_stats(
        self,
        *,
        chunk_size: int,
        action_gap_frames: int,
        action_names: list[str],
        relative_exclude_joints: list[str] | None = None,
        relative_action_keyframes: list[str] | None = None,
        relative_exclude_keys: list[str] | None = None,
    ) -> dict[str, np.ndarray] | None:
        self.ensure_loaded()
        if self._frame_df is None:
            return None
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if action_gap_frames < 0:
            raise ValueError("action_gap_frames must be non-negative")
        if "episode_index" not in self._frame_df.columns:
            return None

        states = self._values_matrix(OBS_STATE)
        actions = self._values_matrix(ACTION)
        action_valid_mask = self._values_matrix("action_valid_mask")
        if action_valid_mask is None:
            action_valid_mask = self._values_matrix(OBS_ACTION_MASK)
        if states is None or actions is None:
            return None
        if action_valid_mask is None:
            action_valid_mask = np.ones_like(actions, dtype=np.float32)

        action_dim = actions.shape[1]
        relative_mask = _relative_mask_from_action_names(
            action_dim,
            action_names=action_names,
            exclude_joints=relative_exclude_joints or [],
        )
        keyframe_specs = _relative_keyframe_specs(relative_action_keyframes, relative_exclude_keys)
        for pos_start, rot_start in keyframe_specs:
            relative_mask[pos_start : min(pos_start + 3, action_dim)] = 0.0
            relative_mask[rot_start : min(rot_start + 6, action_dim)] = 0.0
        episode_ids = self._frame_df["episode_index"].to_numpy(dtype=np.int64)
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
        chunk_offsets = np.arange(int(chunk_size), dtype=np.int64)
        for block_start in range(0, total, block_size):
            block_end = min(block_start + block_size, total)
            start_indices = np.arange(block_start, block_end, dtype=np.int64)
            query_indices = start_indices[:, None] + int(action_gap_frames) + chunk_offsets[None, :]
            query_indices = np.minimum(query_indices, episode_last_indices[start_indices, None])

            chunk_actions = actions[query_indices].copy()
            chunk_masks = action_valid_mask[query_indices].copy()
            chunk_actions = _apply_keyframe_relative_np(chunk_actions, states[start_indices], keyframe_specs)
            mask_dim = min(len(relative_mask), states.shape[1], action_dim)
            chunk_actions[:, :, :mask_dim] -= states[start_indices, None, :mask_dim] * relative_mask[:mask_dim]
            relative_values.append(chunk_actions.reshape(-1, action_dim))
            relative_valid.append(chunk_masks.reshape(-1, action_dim))

        if not relative_values:
            return None

        values = np.concatenate(relative_values, axis=0)
        valid = np.concatenate(relative_valid, axis=0)
        names = list(action_names[:action_dim])
        if len(names) < action_dim:
            names.extend([f"action_{idx}" for idx in range(len(names), action_dim)])
        return compute_masked_vector_stats(values, valid, names)

    def overlay(self, item: dict, abs_idx: int, *, skip_keys: set[str] | None = None) -> dict:
        self.ensure_loaded()
        if self._frame_df is None or abs_idx not in self._frame_df.index:
            return item
        skip_keys = skip_keys or set()

        sidecar_row = self._frame_df.loc[abs_idx]
        if isinstance(sidecar_row, pd.DataFrame):
            sidecar_row = sidecar_row.iloc[0]

        item = dict(item)
        for key, value in sidecar_row.items():
            if key in {"index", "episode_index", "frame_index"}:
                continue
            if key in skip_keys:
                continue
            if sidecar_value_is_missing(value):
                continue
            item[key] = materialize_sidecar_value(value, vector_dim=self.vector_dim(key))

        task_caption = item.get("task_vlm_caption", "")
        if isinstance(task_caption, str) and task_caption.strip():
            item["task"] = task_caption.strip()

        subtask_text = item.get("subtask_text_sidecar", "")
        if isinstance(subtask_text, str) and subtask_text.strip():
            item["subtask"] = subtask_text.strip()
        elif "subtask_text_sidecar" in self._default_values:
            item.setdefault("subtask", "")

        next_subtask_text = item.get("next_subtask_text_sidecar", "")
        if isinstance(next_subtask_text, str) and next_subtask_text.strip():
            item["next_subtask"] = next_subtask_text.strip()
        elif "next_subtask_text_sidecar" in self._default_values:
            item.setdefault("next_subtask", "")

        user_prompt = item.get("user_prompt_sidecar", "")
        if isinstance(user_prompt, str) and user_prompt.strip():
            item["user_prompt"] = user_prompt.strip()
        elif "user_prompt_sidecar" in self._default_values:
            item.setdefault("user_prompt", "")

        if "state_mask" in item:
            item[OBS_STATE_MASK] = item["state_mask"]
        if "action_valid_mask" in item:
            item[OBS_ACTION_MASK] = item["action_valid_mask"]

        return item

    def values_for_indices(self, key: str, indices: list[int]) -> torch.Tensor | None:
        self.ensure_loaded()
        if self._frame_df is None or key not in self._frame_df.columns:
            return None

        rows: list[torch.Tensor] = []
        vector_dim = self.vector_dim(key)
        for index in indices:
            if index not in self._frame_df.index:
                return None
            value = self._frame_df.at[index, key]
            if sidecar_value_is_missing(value):
                return None
            materialized = materialize_sidecar_value(value, vector_dim=vector_dim)
            if not torch.is_tensor(materialized):
                return None
            rows.append(materialized.to(dtype=torch.float32).reshape(-1))
        return torch.stack(rows, dim=0)


def apply_canonical_sidecar_feature_overrides(features: dict[str, dict], sidecar: FramewiseSidecar) -> dict[str, dict]:
    updated = dict(features)
    state_dim = sidecar.vector_dim("observation.state")
    action_dim = sidecar.vector_dim(ACTION)
    if state_dim is not None and "observation.state" in updated:
        updated["observation.state"] = dict(updated["observation.state"])
        updated["observation.state"]["shape"] = (state_dim,)
    if action_dim is not None and ACTION in updated:
        updated[ACTION] = dict(updated[ACTION])
        updated[ACTION]["shape"] = (action_dim,)
    return updated
