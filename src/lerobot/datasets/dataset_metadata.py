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
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import packaging.version
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.feature_utils import _validate_feature_names, create_empty_dataset_info
from lerobot.datasets.io_utils import (
    get_file_size_in_mb,
    load_episodes,
    load_info,
    load_stats,
    load_subtasks,
    load_tasks,
    write_info,
    write_json,
    write_stats,
    write_tasks,
)
from lerobot.datasets.utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_FEATURES,
    INFO_PATH,
    check_version_compatibility,
    flatten_dict,
    get_safe_version,
    has_legacy_hub_download_metadata,
    is_valid_version,
    update_chunk_file_indices,
)
from lerobot.datasets.video_utils import get_video_info
from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE, OBS_ACTION_MASK, OBS_STATE_MASK

CODEBASE_VERSION = "v3.0"
VISUAL_FEATURE_DTYPES = {"image", "video", "tactile", "depth"}
CHANNEL_NAME_ALIASES = {"channel", "channels"}


def _uses_mov_container(video_key: str) -> bool:
    """Return whether a video key should be stored in a `.mov` container."""
    lowered = video_key.lower()
    return any(token in lowered for token in ("deformation", "shear", "depth"))


def _is_visual_feature(feature: dict[str, Any]) -> bool:
    return feature.get("dtype") in VISUAL_FEATURE_DTYPES


def _get_visual_tensor_shape(feature: dict[str, Any], num_frames: int | None = None) -> tuple[int, ...]:
    shape = tuple(feature["shape"])
    names = feature.get("names")
    if feature.get("dtype") in {"image", "video"} and len(shape) == 3 and names and len(names) == 3:
        if names[0] not in CHANNEL_NAME_ALIASES and names[-1] in CHANNEL_NAME_ALIASES:
            shape = (shape[2], shape[0], shape[1])
    if num_frames is not None and num_frames > 1:
        shape = (num_frames, *shape)
    return shape


def _make_empty_visual_tensor(feature: dict[str, Any], num_frames: int | None = None) -> torch.Tensor:
    return torch.zeros(_get_visual_tensor_shape(feature, num_frames=num_frames), dtype=torch.float32)


def _merge_info_with_template(base_info: dict, template_info: dict | None) -> dict:
    if template_info is None:
        return deepcopy(base_info)

    if "fps" in template_info and template_info["fps"] != base_info["fps"]:
        raise ValueError(
            f"Template info fps ({template_info['fps']}) must match dataset fps ({base_info['fps']})."
        )

    merged = deepcopy(base_info)
    merged_features: dict[str, dict[str, Any]] = {}

    base_features = base_info.get("features", {})
    template_features = template_info.get("features", {})

    for key, feature in base_features.items():
        if not _is_visual_feature(feature) or key in template_features:
            merged_features[key] = deepcopy(feature)

    for key, feature in template_features.items():
        if not _is_visual_feature(feature) and key not in base_features:
            raise ValueError(
                f"Template info feature '{key}' is not visual and does not exist in the source dataset."
            )
        merged_features[key] = deepcopy(feature)

    merged["features"] = merged_features
    if template_info.get("robot_type") is not None:
        merged["robot_type"] = template_info["robot_type"]

    return merged


class LeRobotDatasetMetadata:
    """Metadata container for a LeRobot dataset.

    Manages the ``info.json``, ``stats.json``, ``tasks.parquet``, and
    ``episodes/`` parquet files that describe a dataset's structure, content,
    and statistics.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        metadata_buffer_size: int = 10,
        info_path: str | Path | None = None,
        sidecar_path: str | Path | None = None,
    ):
        """Load or download metadata for an existing LeRobot dataset.

        Attempts to load metadata from local disk. If files are missing or
        ``force_cache_sync`` is ``True``, downloads the ``meta/`` directory from
        the Hub.

        Args:
            repo_id: Repository identifier (e.g. ``'lerobot/aloha_sim'``).
            root: Local directory for the dataset. When provided, Hub downloads
                are materialized directly into this directory. When omitted,
                existing local datasets are still looked up under
                ``$HF_LEROBOT_HOME/{repo_id}``, but Hub downloads use a
                revision-safe snapshot cache under
                ``$HF_LEROBOT_HOME/hub``.
            revision: Git revision (branch, tag, or commit hash). Defaults to
                the current codebase version.
            force_cache_sync: If ``True``, re-download metadata from the Hub
                even when local files exist.
            metadata_buffer_size: Number of episode metadata records to buffer
                in memory before flushing to parquet.
        """
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self._requested_root = Path(root) if root is not None else None
        self.root = self._requested_root if self._requested_root is not None else HF_LEROBOT_HOME / repo_id
        self._info_path = Path(info_path) if info_path is not None else None
        self.sidecar_path = Path(sidecar_path) if sidecar_path is not None else None
        self.storage_info: dict[str, Any] | None = None
        self._info_template: dict[str, Any] | None = None
        self._pq_writer = None
        self.latest_episode = None
        self._metadata_buffer: list[dict] = []
        self._metadata_buffer_size = metadata_buffer_size
        self._finalized = False

        try:
            if force_cache_sync or (
                self._requested_root is None and has_legacy_hub_download_metadata(self.root)
            ):
                raise FileNotFoundError
            self._load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            self._pull_from_repo(allow_patterns="meta/")
            self._load_metadata()

        if self.sidecar_path is not None:
            from lerobot.datasets.sidecar_utils import FramewiseSidecar, apply_canonical_sidecar_feature_overrides

            self._sidecar_for_metadata = FramewiseSidecar(self.sidecar_path, repo_id=self.repo_id)
            self.info["features"] = apply_canonical_sidecar_feature_overrides(
                self.info["features"],
                self._sidecar_for_metadata,
            )
        else:
            self._sidecar_for_metadata = None

    def _flush_metadata_buffer(self) -> None:
        """Write all buffered episode metadata to parquet file."""
        if not hasattr(self, "_metadata_buffer") or len(self._metadata_buffer) == 0:
            return

        combined_dict = {}
        for episode_dict in self._metadata_buffer:
            for key, value in episode_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                # Extract value and serialize numpy arrays
                # because PyArrow's from_pydict function doesn't support numpy arrays
                val = value[0] if isinstance(value, list) else value
                combined_dict[key].append(val.tolist() if isinstance(val, np.ndarray) else val)

        first_ep = self._metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        table = pa.Table.from_pydict(combined_dict)

        if not self._pq_writer:
            path = Path(self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pq_writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )

        self._pq_writer.write_table(table)

        self.latest_episode = self._metadata_buffer[-1]
        self._metadata_buffer.clear()

    def _close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        self._flush_metadata_buffer()

        writer = getattr(self, "_pq_writer", None)
        if writer is not None:
            writer.close()
            self._pq_writer = None

    def finalize(self) -> None:
        """Flush metadata buffer and close the parquet writer.

        Idempotent — safe to call multiple times.
        """
        if getattr(self, "_finalized", False):
            return
        self._close_writer()
        self._finalized = True

    def __del__(self):
        """Safety net: flush and close parquet writer on garbage collection."""
        # During interpreter shutdown, referenced objects may already be collected.
        with contextlib.suppress(Exception):
            self.finalize()

    def _load_metadata(self):
        self.storage_info = load_info(self.root)
        self._info_template = load_info(self.root, info_path=self._info_path) if self._info_path is not None else None
        self.info = _merge_info_with_template(self.storage_info, self._info_template)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks = load_tasks(self.root)
        self.subtasks = load_subtasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)

    def _pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        if self._requested_root is None:
            self.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                )
            )
            return

        self._requested_root.mkdir(exist_ok=True, parents=True)
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self._requested_root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        self.root = self._requested_root

    @property
    def url_root(self) -> str:
        """Hugging Face Hub URL root for this dataset."""
        return f"hf://datasets/{self.repo_id}"

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return packaging.version.parse(self.storage_info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        """Return the relative parquet file path for the given episode index.

        Args:
            ep_index: Zero-based episode index.

        Returns:
            Path to the parquet file containing this episode's data.

        Raises:
            IndexError: If ``ep_index`` is out of range.
        """
        if self.episodes is None:
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """Return the relative video file path for the given episode and video key.

        Args:
            ep_index: Zero-based episode index.
            vid_key: Feature key identifying the video stream
                (e.g. ``'observation.images.laptop'``).

        Returns:
            Path to the video file containing this episode's frames.

        Raises:
            IndexError: If ``ep_index`` is out of range.
        """
        if self.episodes is None:
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        return self.get_video_file_path_from_indices(vid_key, chunk_idx, file_idx)

    def get_video_file_path_from_indices(self, video_key: str, chunk_index: int, file_index: int) -> Path:
        """Return the relative video path for a known `(chunk_index, file_index)` pair."""
        if _uses_mov_container(video_key):
            if self.mov_path is None:
                raise ValueError(
                    f"Video key '{video_key}' requires mov_path metadata, but info.json does not define mov_path."
                )
            fpath = self.mov_path.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        else:
            fpath = self.video_path.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        return Path(fpath)

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["video_path"]

    @property
    def mov_path(self) -> str | None:
        """Formattable string for mov-backed video files."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info.get("mov_path")

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def storage_features(self) -> dict[str, dict]:
        """Features physically stored in parquet/video files."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["features"]

    @property
    def reader_features(self) -> dict[str, dict]:
        """Features that should be materialized from disk for the current training schema."""
        features: dict[str, dict] = {}
        effective_features = self.features
        for key, feature in self.storage_features.items():
            if _is_visual_feature(feature):
                if key in effective_features:
                    features[key] = feature
            else:
                features[key] = feature
        return features

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.reader_features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.reader_features.items() if ft["dtype"] in ["video", "tactile", "depth"]]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [
            key
            for key, ft in self.reader_features.items()
            if ft["dtype"] in ["video", "image", "tactile", "depth"]
        ]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["total_tasks"]

    @property
    def chunks_size(self) -> int:
        """Max number of files per chunk."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["chunks_size"]

    @property
    def data_files_size_in_mb(self) -> int:
        """Max size of data file in mega bytes."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["data_files_size_in_mb"]

    @property
    def video_files_size_in_mb(self) -> int:
        """Max size of video file in mega bytes."""
        if self.storage_info is None:
            raise ValueError("Storage metadata has not been loaded yet.")
        return self.storage_info["video_files_size_in_mb"]

    @property
    def uses_info_template(self) -> bool:
        return self._info_template is not None

    def canonicalize_item(
        self,
        item: dict[str, Any],
        delta_timestamps: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        if not self.uses_info_template:
            return item

        target_features = self.features
        canonical: dict[str, Any] = {}

        for key, feature in target_features.items():
            if _is_visual_feature(feature):
                missing = key not in item
                num_frames = None
                if delta_timestamps is not None and key in delta_timestamps:
                    num_frames = len(delta_timestamps[key])
                canonical[key] = item[key] if not missing else _make_empty_visual_tensor(feature, num_frames)
                canonical[f"{key}_is_missing"] = torch.tensor(missing, dtype=torch.bool)
                continue

            if key not in item:
                raise KeyError(f"Required non-visual feature '{key}' is missing from dataset item.")
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
                "action_valid_mask",
                "state_mask",
                OBS_ACTION_MASK,
                OBS_STATE_MASK,
            }:
                canonical[key] = value
            elif key.endswith("_is_pad") or key.endswith("_is_missing"):
                canonical[key] = value

        if "action_valid_mask" in canonical:
            canonical[OBS_ACTION_MASK] = canonical["action_valid_mask"]
        elif OBS_ACTION_MASK in canonical:
            canonical["action_valid_mask"] = canonical[OBS_ACTION_MASK]

        if "state_mask" in canonical:
            canonical[OBS_STATE_MASK] = canonical["state_mask"]
        elif OBS_STATE_MASK in canonical:
            canonical["state_mask"] = canonical[OBS_STATE_MASK]

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

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        if task in self.tasks.index:
            return int(self.tasks.loc[task].task_index)
        else:
            return None

    def save_episode_tasks(self, tasks: list[str]):
        """Register tasks for the current episode and persist to disk.

        New tasks that do not already exist in the dataset are assigned
        sequential task indices and appended to the tasks parquet file.

        Args:
            tasks: List of unique task descriptions in natural language.

        Raises:
            ValueError: If ``tasks`` contains duplicates.
        """
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Tasks are not unique: {tasks}")

        if self.tasks is None:
            new_tasks = tasks
            task_indices = range(len(tasks))
            self.tasks = pd.DataFrame({"task_index": task_indices}, index=pd.Index(tasks, name="task"))
        else:
            new_tasks = [task for task in tasks if task not in self.tasks.index]
            new_task_indices = range(len(self.tasks), len(self.tasks) + len(new_tasks))
            for task_idx, task in zip(new_task_indices, new_tasks, strict=False):
                self.tasks.loc[task] = task_idx

        if len(new_tasks) > 0:
            # Update on disk
            write_tasks(self.tasks, self.root)

    def _save_episode_metadata(self, episode_dict: dict) -> None:
        """Buffer episode metadata and write to parquet in batches for efficiency.

        This function accumulates episode metadata in a buffer and flushes it when the buffer
        reaches the configured size. This reduces I/O overhead by writing multiple episodes
        at once instead of one row at a time.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.
        """
        # Convert to list format for each value
        episode_dict = {key: [value] for key, value in episode_dict.items()}
        num_frames = episode_dict["length"][0]

        if self.latest_episode is None:
            # Initialize indices and frame count for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            if self.episodes is not None and len(self.episodes) > 0:
                # It means we are resuming recording, so we need to load the latest episode
                # Update the indices to avoid overwriting the latest episode
                chunk_idx = self.episodes[-1]["meta/episodes/chunk_index"]
                file_idx = self.episodes[-1]["meta/episodes/file_index"]
                latest_num_frames = self.episodes[-1]["dataset_to_index"]
                episode_dict["dataset_from_index"] = [latest_num_frames]
                episode_dict["dataset_to_index"] = [latest_num_frames + num_frames]

                # When resuming, move to the next file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
            else:
                episode_dict["dataset_from_index"] = [0]
                episode_dict["dataset_to_index"] = [num_frames]

            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
        else:
            chunk_idx = self.latest_episode["meta/episodes/chunk_index"][0]
            file_idx = self.latest_episode["meta/episodes/file_index"][0]

            latest_path = (
                self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
                if self._pq_writer is None
                else self._pq_writer.where
            )

            if Path(latest_path).exists():
                latest_size_in_mb = get_file_size_in_mb(Path(latest_path))
                latest_num_frames = self.latest_episode["episode_index"][0]

                av_size_per_frame = latest_size_in_mb / latest_num_frames if latest_num_frames > 0 else 0.0

                if latest_size_in_mb + av_size_per_frame * num_frames >= self.data_files_size_in_mb:
                    # Size limit is reached, flush buffer and prepare new parquet file
                    self._flush_metadata_buffer()
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
                    self._close_writer()

            # Update the existing pandas dataframe with new row
            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
            episode_dict["dataset_from_index"] = [self.latest_episode["dataset_to_index"][0]]
            episode_dict["dataset_to_index"] = [self.latest_episode["dataset_to_index"][0] + num_frames]

        # Add to buffer
        self._metadata_buffer.append(episode_dict)
        self.latest_episode = episode_dict

        if len(self._metadata_buffer) >= self._metadata_buffer_size:
            self._flush_metadata_buffer()

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:
        """Persist episode metadata, update dataset info, and aggregate stats.

        Writes the episode's metadata to the buffered parquet writer, increments
        the total episode/frame counters in ``info.json``, and merges the
        episode's statistics into the running dataset statistics.

        Args:
            episode_index: Zero-based index of the episode being saved.
            episode_length: Number of frames in this episode.
            episode_tasks: List of task descriptions for this episode.
            episode_stats: Per-feature statistics for this episode.
            episode_metadata: Additional metadata (chunk/file indices, frame
                ranges, video timestamps, etc.).
        """
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        episode_dict.update(episode_metadata)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)

        # Update info
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}

        write_info(self.info, self.root)

        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats is not None else episode_stats
        write_stats(self.stats, self.root)

    def update_video_info(self, video_key: str | None = None) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        for key in video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(0, key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def update_chunk_settings(
        self,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> None:
        """Update chunk and file size settings after dataset creation.

        This allows users to customize storage organization without modifying the constructor.
        These settings control how episodes are chunked and how large files can grow before
        creating new ones.

        Args:
            chunks_size: Maximum number of files per chunk directory. If None, keeps current value.
            data_files_size_in_mb: Maximum size for data parquet files in MB. If None, keeps current value.
            video_files_size_in_mb: Maximum size for video files in MB. If None, keeps current value.
        """
        if chunks_size is not None:
            if chunks_size <= 0:
                raise ValueError(f"chunks_size must be positive, got {chunks_size}")
            self.info["chunks_size"] = chunks_size

        if data_files_size_in_mb is not None:
            if data_files_size_in_mb <= 0:
                raise ValueError(f"data_files_size_in_mb must be positive, got {data_files_size_in_mb}")
            self.info["data_files_size_in_mb"] = data_files_size_in_mb

        if video_files_size_in_mb is not None:
            if video_files_size_in_mb <= 0:
                raise ValueError(f"video_files_size_in_mb must be positive, got {video_files_size_in_mb}")
            self.info["video_files_size_in_mb"] = video_files_size_in_mb

        # Update the info file on disk
        write_info(self.info, self.root)

    def get_chunk_settings(self) -> dict[str, int]:
        """Get current chunk and file size settings.

        Returns:
            Dict containing chunks_size, data_files_size_in_mb, and video_files_size_in_mb.
        """
        return {
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
        }

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
        metadata_buffer_size: int = 10,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> "LeRobotDatasetMetadata":
        """Create metadata for a new LeRobot dataset from scratch.

        Initializes the ``info.json`` file on disk with the provided feature
        schema and dataset settings. No episode data is written yet.

        Args:
            repo_id: Repository identifier (e.g. ``'user/my_dataset'``).
            fps: Frames per second used during data collection.
            features: Feature specification dict mapping feature names to their
                type/shape metadata.
            robot_type: Optional robot type string stored in metadata.
            root: Local directory for the dataset. Defaults to
                ``$HF_LEROBOT_HOME/{repo_id}``. Must not already exist.
            use_videos: If ``True``, visual modalities are encoded as MP4 videos.
            metadata_buffer_size: Number of episode metadata records to buffer
                before flushing to parquet.
            chunks_size: Max number of files per chunk directory. ``None`` uses
                the default.
            data_files_size_in_mb: Max parquet file size in MB. ``None`` uses the
                default.
            video_files_size_in_mb: Max video file size in MB. ``None`` uses the
                default.

        Returns:
            A new :class:`LeRobotDatasetMetadata` instance.
        """
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root) if root is not None else None
        obj.root = obj._requested_root if obj._requested_root is not None else HF_LEROBOT_HOME / repo_id
        obj._info_path = None
        obj._info_template = None

        obj.root.mkdir(parents=True, exist_ok=False)

        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks = None
        obj.subtasks = None
        obj.episodes = None
        obj.stats = None
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION,
            fps,
            features,
            use_videos,
            robot_type,
            chunks_size,
            data_files_size_in_mb,
            video_files_size_in_mb,
        )
        obj.storage_info = deepcopy(obj.info)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError(
                f"Features contain video keys {obj.video_keys}, but 'use_videos' is set to False. "
                "Either remove video features from the features dict, or set 'use_videos=True'."
            )
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        obj._pq_writer = None
        obj.latest_episode = None
        obj._metadata_buffer = []
        obj._metadata_buffer_size = metadata_buffer_size
        obj._finalized = False
        return obj
