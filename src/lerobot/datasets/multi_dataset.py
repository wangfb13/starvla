#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from bisect import bisect_right
from collections.abc import Callable
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
import torch
import torch.utils

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import (
    _get_visual_tensor_shape,
    _is_visual_feature,
    _make_empty_visual_tensor,
)
from lerobot.datasets.feature_utils import get_delta_indices, get_hf_features_from_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoFrame
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ACTION_MASK, OBS_STATE_MASK

SPARKY_SINGLE_LEFT_IMAGE = "observation.images.cam_left"
SPARKY_SINGLE_RIGHT_IMAGE = "observation.images.cam_right"
SPARKY_DUAL_LEFT_IMAGE = "observation.images.left_cam_left"
SPARKY_DUAL_RIGHT_IMAGE = "observation.images.right_cam_right"
SPARKY_LEFT_HAND_LEFT_TACTILE = "observation.depth_deformation.tactile_left_left"
SPARKY_LEFT_HAND_RIGHT_TACTILE = "observation.depth_deformation.tactile_right_left"
SPARKY_RIGHT_HAND_LEFT_TACTILE = "observation.depth_deformation.tactile_left_right"
SPARKY_RIGHT_HAND_RIGHT_TACTILE = "observation.depth_deformation.tactile_right_right"


def _infer_sparky_visual_aliases(features: dict[str, dict]) -> dict[str, str]:
    """Map known Sparky single/dual-arm visual keys into a shared canonical schema."""
    aliases: dict[str, str] = {}

    if SPARKY_DUAL_LEFT_IMAGE in features and SPARKY_SINGLE_LEFT_IMAGE not in features:
        aliases[SPARKY_DUAL_LEFT_IMAGE] = SPARKY_SINGLE_LEFT_IMAGE
    if SPARKY_DUAL_RIGHT_IMAGE in features and SPARKY_SINGLE_RIGHT_IMAGE not in features:
        aliases[SPARKY_DUAL_RIGHT_IMAGE] = SPARKY_SINGLE_RIGHT_IMAGE

    is_right_only = (
        SPARKY_SINGLE_RIGHT_IMAGE in features
        and SPARKY_SINGLE_LEFT_IMAGE not in features
        and SPARKY_LEFT_HAND_LEFT_TACTILE in features
        and SPARKY_LEFT_HAND_RIGHT_TACTILE in features
        and SPARKY_RIGHT_HAND_LEFT_TACTILE not in features
        and SPARKY_RIGHT_HAND_RIGHT_TACTILE not in features
    )
    if is_right_only:
        aliases[SPARKY_LEFT_HAND_LEFT_TACTILE] = SPARKY_RIGHT_HAND_LEFT_TACTILE
        aliases[SPARKY_LEFT_HAND_RIGHT_TACTILE] = SPARKY_RIGHT_HAND_RIGHT_TACTILE

    return aliases


def _canonical_feature_key(key: str, aliases: dict[str, str]) -> str:
    return aliases.get(key, key)


def _canonicalize_stats_keys(stats: dict[str, dict], aliases: dict[str, str]) -> dict[str, dict]:
    canonical_stats: dict[str, dict] = {}
    for key, value in stats.items():
        canonical_key = _canonical_feature_key(key, aliases)
        if canonical_key not in canonical_stats:
            canonical_stats[canonical_key] = value
    return canonical_stats


def _apply_item_feature_aliases(item: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    if not aliases:
        return item

    item = dict(item)
    for source_key, target_key in aliases.items():
        if source_key in item and target_key not in item:
            item[target_key] = item[source_key]
        if source_key != target_key:
            # Treat aliases as renames so right-hand single-arm tactile data does not
            # also look like valid left-hand tactile data after canonicalization.
            item.pop(source_key, None)

        source_pad_key = f"{source_key}_is_pad"
        target_pad_key = f"{target_key}_is_pad"
        if source_pad_key in item and target_pad_key not in item:
            item[target_pad_key] = item[source_pad_key]
        if source_pad_key != target_pad_key:
            item.pop(source_pad_key, None)

        source_missing_key = f"{source_key}_is_missing"
        target_missing_key = f"{target_key}_is_missing"
        if source_missing_key in item and target_missing_key not in item:
            item[target_missing_key] = item[source_missing_key]
        if source_missing_key != target_missing_key:
            item.pop(source_missing_key, None)

    return item


def _delta_timestamps_for_reader(
    delta_timestamps: dict[str, list[float]] | None,
    aliases: dict[str, str],
    supported_keys: set[str],
) -> dict[str, list[float]] | None:
    if delta_timestamps is None:
        return None

    reader_delta_timestamps: dict[str, list[float]] = {}
    reverse_aliases = {target_key: source_key for source_key, target_key in aliases.items()}
    for key, value in delta_timestamps.items():
        reader_key = key if key in supported_keys else reverse_aliases.get(key)
        if reader_key in supported_keys:
            reader_delta_timestamps[reader_key] = value

    return reader_delta_timestamps or None


def _align_visual_tensor_to_feature_shape(
    value: Any,
    feature: dict[str, Any],
    num_frames: int | None = None,
) -> Any:
    if not isinstance(value, torch.Tensor):
        return value

    target_shape = _get_visual_tensor_shape(feature, num_frames=num_frames)
    if tuple(value.shape) == target_shape:
        return value

    if len(target_shape) == 3 and value.ndim == 3:
        channels, height, width = target_shape
        if tuple(value.shape) == (height, width, channels):
            return value.permute(2, 0, 1).contiguous()

    if len(target_shape) == 4 and value.ndim == 4:
        frames, channels, height, width = target_shape
        if tuple(value.shape) == (frames, height, width, channels):
            return value.permute(0, 3, 1, 2).contiguous()

    return value


def _filter_stats(
    stats: dict[str, dict],
    enabled_feature_keys: list[str],
) -> dict[str, dict]:
    return {key: value for key, value in stats.items() if key in enabled_feature_keys}


def _aggregate_tasks(datasets_list: list[LeRobotDataset]) -> pd.DataFrame | None:
    tasks: list[str] = []
    seen: set[str] = set()

    def add_task(task: object) -> None:
        if not isinstance(task, str):
            return
        task = task.strip()
        if not task or task in seen:
            return
        seen.add(task)
        tasks.append(task)

    for dataset in datasets_list:
        if dataset.meta.tasks is not None:
            for task in dataset.meta.tasks.index:
                add_task(task)

        for task in _load_sidecar_task_captions(dataset.sidecar_path):
            add_task(task)

    if not tasks:
        return None

    return pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task"))


def _load_sidecar_task_captions(sidecar_path: Path | None) -> list[str]:
    if sidecar_path is None or not sidecar_path.exists():
        return []

    try:
        task_df = pd.read_parquet(sidecar_path, columns=["task_vlm_caption"])
    except Exception as exc:
        if "task_vlm_caption" in str(exc):
            return []
        raise

    captions: list[str] = []
    for value in task_df["task_vlm_caption"].dropna().unique().tolist():
        if isinstance(value, str) and value.strip():
            captions.append(value.strip())
    return captions


def _aggregate_episodes(datasets_list: list[LeRobotDataset]) -> datasets.Dataset | None:
    aggregated: list[datasets.Dataset] = []
    frame_offset = 0
    episode_offset = 0

    for dataset in datasets_list:
        episode_ds = dataset.meta.episodes.with_format(None)
        episode_dict = {column: list(episode_ds[column]) for column in episode_ds.column_names}

        if "episode_index" in episode_dict:
            episode_dict["episode_index"] = [value + episode_offset for value in episode_dict["episode_index"]]
        if "dataset_from_index" in episode_dict:
            episode_dict["dataset_from_index"] = [
                value + frame_offset for value in episode_dict["dataset_from_index"]
            ]
        if "dataset_to_index" in episode_dict:
            episode_dict["dataset_to_index"] = [
                value + frame_offset for value in episode_dict["dataset_to_index"]
            ]

        aggregated.append(datasets.Dataset.from_dict(episode_dict, features=episode_ds.features))
        frame_offset += dataset.num_frames
        episode_offset += dataset.num_episodes

    if not aggregated:
        return None
    if len(aggregated) == 1:
        return aggregated[0]
    return datasets.concatenate_datasets(aggregated)


def _collect_multi_features(
    datasets_list: list[LeRobotDataset],
    feature_aliases_by_repo: dict[str, dict[str, str]],
    *,
    allow_missing_non_visual_features: bool = False,
) -> dict[str, dict]:
    def _feature_signature(feature: dict[str, Any]) -> tuple[Any, tuple[Any, ...], Any]:
        shape = feature.get("shape")
        shape_tuple = tuple(shape) if shape is not None else ()
        return feature.get("dtype"), shape_tuple, feature.get("names")

    features: dict[str, dict] = {}
    non_visual_keys: set[str] = set()

    for dataset in datasets_list:
        aliases = feature_aliases_by_repo.get(dataset.repo_id, {})
        for key, feature in dataset.features.items():
            canonical_key = _canonical_feature_key(key, aliases)
            if canonical_key in features:
                if _feature_signature(features[canonical_key]) != _feature_signature(feature):
                    raise ValueError(
                        f"Feature '{canonical_key}' has incompatible specs across datasets. "
                        f"Expected {features[canonical_key]}, got {feature} in {dataset.repo_id}."
                    )
            else:
                features[canonical_key] = deepcopy(feature)

            if not _is_visual_feature(feature):
                non_visual_keys.add(canonical_key)

    missing_non_visual: dict[str, list[str]] = {}
    for dataset in datasets_list:
        aliases = feature_aliases_by_repo.get(dataset.repo_id, {})
        canonical_keys = {_canonical_feature_key(key, aliases) for key in dataset.features}
        missing = [key for key in non_visual_keys if key not in canonical_keys]
        if missing:
            missing_non_visual[dataset.repo_id] = missing

    if missing_non_visual:
        if allow_missing_non_visual_features:
            dropped_keys = sorted({key for missing in missing_non_visual.values() for key in missing})
            for key in dropped_keys:
                features.pop(key, None)
            logging.warning(
                "Dropping non-visual features that are not present in every sidecar-enabled dataset: %s",
                dropped_keys,
            )
            return features
        raise ValueError(
            "All datasets must expose the same non-visual features for mixed training. "
            f"Missing keys by dataset: {missing_non_visual}"
        )

    return features


def _stats_array_dim(stats_dict: dict[str, Any]) -> int | None:
    for key in ("mean", "min", "max", "std", "q01", "q99"):
        value = stats_dict.get(key)
        if value is None:
            continue
        try:
            return int(len(value))
        except TypeError:
            continue
    return None


class MultiLeRobotDatasetMetadata:
    """Training-oriented metadata adapter for concatenated local datasets."""

    def __init__(
        self,
        datasets_list: list[LeRobotDataset],
        repo_ids: list[str],
        features: dict[str, dict],
        stats: dict[str, dict],
        extra_complementary_defaults: dict[str, object] | None = None,
    ):
        if len(datasets_list) == 0:
            raise ValueError("At least one dataset is required to build multi-dataset metadata.")

        first_meta = datasets_list[0].meta
        self.repo_id = "+".join(repo_ids)
        self.revision = datasets_list[0].revision
        self.root = datasets_list[0].root.parent if len(datasets_list) > 1 else datasets_list[0].root
        self.info = deepcopy(first_meta.info)
        self.info["features"] = deepcopy(features)
        self.info["total_frames"] = sum(dataset.num_frames for dataset in datasets_list)
        self.info["total_episodes"] = sum(dataset.num_episodes for dataset in datasets_list)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["camera_keys"] = [
            key for key, feature in self.info["features"].items() if _is_visual_feature(feature)
        ]

        self.tasks = _aggregate_tasks(datasets_list)
        self.subtasks = None
        self.episodes = _aggregate_episodes(datasets_list)
        self.stats = stats
        self.extra_complementary_defaults = extra_complementary_defaults or {}

        for key in ("observation.state", "action"):
            if key in self.info["features"] and key in self.stats:
                dim = _stats_array_dim(self.stats[key])
                if dim is not None:
                    self.info["features"][key] = deepcopy(self.info["features"][key])
                    self.info["features"][key]["shape"] = (dim,)

    @property
    def robot_type(self) -> str | None:
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "tactile", "depth"]]

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, ft in self.features.items()
            if ft["dtype"] in ["video", "image", "tactile", "depth"]
        ]

    @property
    def names(self) -> dict[str, list | dict]:
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict[str, tuple]:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    def canonicalize_item(
        self,
        item: dict[str, Any],
        delta_timestamps: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        canonical: dict[str, Any] = {}

        for key, feature in self.features.items():
            if _is_visual_feature(feature):
                missing_mask_key = f"{key}_is_missing"
                existing_missing = item.get(missing_mask_key)
                if existing_missing is not None:
                    if isinstance(existing_missing, torch.Tensor):
                        missing = bool(existing_missing.item())
                    else:
                        missing = bool(existing_missing)
                else:
                    missing = key not in item
                num_frames = None
                if delta_timestamps is not None and key in delta_timestamps:
                    num_frames = len(delta_timestamps[key])
                canonical[key] = (
                    _align_visual_tensor_to_feature_shape(item[key], feature, num_frames)
                    if key in item
                    else _make_empty_visual_tensor(feature, num_frames)
                )
                canonical[missing_mask_key] = (
                    existing_missing
                    if existing_missing is not None
                    else torch.tensor(missing, dtype=torch.bool)
                )
                continue

            if key not in item:
                raise KeyError(f"Required non-visual feature '{key}' is missing from multi-dataset item.")
            canonical[key] = item[key]

        for key, value in item.items():
            if key in canonical:
                continue
            if key in {
                "episode_index",
                "frame_index",
                "index",
                "timestamp",
                "task",
                "subtask",
                "next_subtask",
                "user_prompt",
                "subtask_text_sidecar",
                "next_subtask_text_sidecar",
                "dataset_index",
                OBS_ACTION_MASK,
                OBS_STATE_MASK,
            }:
                canonical[key] = value
            elif key.endswith("_is_pad") or key.endswith("_is_missing"):
                canonical[key] = value

        for key, default_value in self.extra_complementary_defaults.items():
            if key not in canonical:
                canonical[key] = default_value.clone() if torch.is_tensor(default_value) else default_value

        # Keep language-side complementary keys structurally stable so the default
        # DataLoader collate_fn does not fail when some frames lack subtask annotations.
        for key in (
            "subtask",
            "next_subtask",
            "user_prompt",
            "subtask_text_sidecar",
            "next_subtask_text_sidecar",
        ):
            canonical.setdefault(key, "")

        return canonical


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        root_paths_by_repo: dict[str, str | Path] | None = None,
        info_path: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        sidecar_paths: dict[str, str | Path] | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.repo_id = "+".join(repo_ids)
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.root_paths_by_repo = (
            {repo_id: Path(path) for repo_id, path in root_paths_by_repo.items()}
            if root_paths_by_repo is not None
            else None
        )
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root_paths_by_repo.get(repo_id, self.root / repo_id)
                if self.root_paths_by_repo is not None
                else self.root / repo_id,
                info_path=info_path,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=None,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
                sidecar_path=sidecar_paths.get(repo_id) if sidecar_paths is not None else None,
            )
            for repo_id in repo_ids
        ]

        if len(self._datasets) == 0:
            raise ValueError("At least one dataset is required.")

        first_fps = self._datasets[0].meta.fps
        fps_mismatch = {
            dataset.repo_id: dataset.meta.fps for dataset in self._datasets if dataset.meta.fps != first_fps
        }
        if fps_mismatch:
            raise ValueError(
                f"All datasets must have the same fps to be concatenated. Expected {first_fps}, got {fps_mismatch}."
            )

        self._feature_aliases_by_repo = {
            dataset.repo_id: _infer_sparky_visual_aliases(dataset.features) for dataset in self._datasets
        }
        self._extra_complementary_defaults = self._resolve_extra_complementary_defaults()
        multi_features = _collect_multi_features(
            self._datasets,
            self._feature_aliases_by_repo,
            allow_missing_non_visual_features=sidecar_paths is not None,
        )
        self.enabled_feature_keys = list(multi_features)
        self.episodes = None
        if sidecar_paths is not None:
            # Canonical sidecar training loads action/state stats from the sidecar files in the factory.
            # Avoid aggregating raw mixed-dataset stats here: source datasets can have heterogeneous robot
            # vectors and visual stat shapes, and those raw stats are not the normalization target anyway.
            self.stats = {}
        else:
            # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
            # with multiple robots of different ranges. Instead we should have one normalization
            # per robot.
            canonical_stats = [
                _canonicalize_stats_keys(dataset.meta.stats, self._feature_aliases_by_repo.get(dataset.repo_id, {}))
                for dataset in self._datasets
            ]
            self.stats = _filter_stats(
                aggregate_stats(canonical_stats),
                self.enabled_feature_keys,
            )
        self.meta = MultiLeRobotDatasetMetadata(
            self._datasets,
            repo_ids=self.repo_ids,
            features=multi_features,
            stats=self.stats,
            extra_complementary_defaults=self._extra_complementary_defaults,
        )
        self._cumulative_sizes: list[int] = []
        running_size = 0
        for dataset in self._datasets:
            running_size += dataset.num_frames
            self._cumulative_sizes.append(running_size)
        self.set_delta_timestamps(delta_timestamps)
        self.set_image_transforms(image_transforms)

    def set_delta_timestamps(self, delta_timestamps: dict[str, Any] | None) -> None:
        per_repo_delta_timestamps = (
            delta_timestamps is not None
            and set(delta_timestamps).issubset(self.repo_ids)
            and all(value is None or isinstance(value, dict) for value in delta_timestamps.values())
        )

        if per_repo_delta_timestamps:
            self.delta_timestamps = delta_timestamps
            self._delta_timestamps_by_repo = {
                repo_id: delta_timestamps.get(repo_id) for repo_id in self.repo_ids
            }
        else:
            if delta_timestamps is not None:
                unsupported_keys = set(delta_timestamps).difference(self.meta.features)
                if unsupported_keys:
                    raise ValueError(
                        "delta_timestamps contains keys that are not exposed by the multi-dataset schema: "
                        f"{sorted(unsupported_keys)}"
                    )
            self.delta_timestamps = delta_timestamps
            self._delta_timestamps_by_repo = dict.fromkeys(self.repo_ids, delta_timestamps)

        for dataset in self._datasets:
            supported_keys = set(getattr(dataset.meta, "reader_features", dataset.meta.features))
            child_delta_timestamps = _delta_timestamps_for_reader(
                self._delta_timestamps_by_repo.get(dataset.repo_id),
                self._feature_aliases_by_repo.get(dataset.repo_id, {}),
                supported_keys,
            )

            dataset.delta_timestamps = child_delta_timestamps
            if dataset.reader is not None:
                dataset.reader.delta_indices = None
                if child_delta_timestamps is not None:
                    dataset.reader.delta_indices = get_delta_indices(child_delta_timestamps, dataset.meta.fps)

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform for this dataset and its children."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in getattr(self, "_datasets", []):
            dataset.set_image_transforms(self.image_transforms)

    def clear_image_transforms(self) -> None:
        """Remove the transform from this dataset and its children."""
        self.set_image_transforms(None)

    def _resolve_extra_complementary_defaults(self) -> dict[str, object]:
        defaults: dict[str, object] = {}
        for dataset in self._datasets:
            for key, value in dataset.sidecar_default_values.items():
                defaults.setdefault(key, value)
            for key in dataset.sidecar_default_values:
                vector_dim = dataset.sidecar_vector_dim(key)
                if vector_dim is None:
                    continue
                defaults[key] = torch.zeros((vector_dim,), dtype=torch.float32)
        return defaults

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self.meta.fps

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        return get_hf_features_from_features(self.meta.features)

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image | VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        dataset_idx = bisect_right(self._cumulative_sizes, idx)
        start_idx = 0 if dataset_idx == 0 else self._cumulative_sizes[dataset_idx - 1]
        item = self._datasets[dataset_idx][idx - start_idx]
        repo_id = self.repo_ids[dataset_idx]
        item = _apply_item_feature_aliases(item, self._feature_aliases_by_repo.get(repo_id, {}))
        task = item.get("task")
        if isinstance(task, str) and self.meta.tasks is not None and "task_index" in item:
            if task not in self.meta.tasks.index:
                raise KeyError(
                    f"Task {task!r} from dataset {repo_id!r} is missing from aggregated multi-dataset tasks. "
                    "If this task comes from a sidecar override, make sure the sidecar contains "
                    "task_vlm_caption before MultiLeRobotDataset is constructed."
                )
            global_task_index = self.meta.tasks.loc[task, "task_index"]
            if isinstance(global_task_index, pd.Series):
                global_task_index = global_task_index.iloc[0]
            item["task_index"] = torch.tensor(int(global_task_index), dtype=torch.int64)
        item["dataset_index"] = torch.tensor(dataset_idx)
        delta_timestamps = self._delta_timestamps_by_repo.get(repo_id)
        return self.meta.canonicalize_item(item, delta_timestamps=delta_timestamps)

    def iter_hf_datasets(self):
        for dataset in self._datasets:
            yield dataset.hf_dataset

    def iter_datasets(self):
        yield from self._datasets

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
