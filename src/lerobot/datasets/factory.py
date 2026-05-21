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
import logging
from pprint import pformat
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def _is_local_lerobot_dataset(root: Path) -> bool:
    return (
        root.is_dir()
        and (root / "meta" / "info.json").is_file()
        and (root / "meta" / "stats.json").is_file()
        and (root / "data").is_dir()
    )


def _discover_child_datasets(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if _is_local_lerobot_dataset(child))


def _build_single_dataset(
    *,
    repo_id: str,
    root: str | Path | None,
    info_path: str | Path | None,
    cfg: TrainPipelineConfig,
    image_transforms: ImageTransforms | None,
):
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root, revision=cfg.dataset.revision, info_path=info_path)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    if not cfg.dataset.streaming:
        return LeRobotDataset(
            repo_id,
            root=root,
            info_path=info_path,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            tolerance_s=cfg.tolerance_s,
        )

    return StreamingLeRobotDataset(
        repo_id,
        root=root,
        info_path=info_path,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        max_num_shards=cfg.num_workers,
        tolerance_s=cfg.tolerance_s,
    )


def _build_multi_dataset(
    *,
    repo_ids: list[str],
    root: str | Path | None,
    info_path: str | Path | None,
    cfg: TrainPipelineConfig,
    image_transforms: ImageTransforms | None,
) -> MultiLeRobotDataset:
    if cfg.dataset.streaming:
        raise NotImplementedError("Streaming is not supported when loading multiple datasets at once.")
    if cfg.dataset.episodes is not None:
        raise NotImplementedError(
            "`dataset.episodes` is not supported when `dataset.root` resolves to multiple datasets."
        )

    dataset = MultiLeRobotDataset(
        repo_ids,
        root=root,
        info_path=info_path,
        delta_timestamps=None,
        image_transforms=image_transforms,
        video_backend=cfg.dataset.video_backend,
    )
    dataset.set_delta_timestamps(resolve_delta_timestamps(cfg.policy, dataset.meta))
    logging.info(
        "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
        f"{pformat(dataset.repo_id_to_index, indent=2)}"
    )
    return dataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    feature_source = getattr(ds_meta, "reader_features", ds_meta.features)
    for key in feature_source:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    dataset_root = Path(cfg.dataset.root) if cfg.dataset.root is not None else None
    discovered_repo_ids = _discover_child_datasets(dataset_root) if dataset_root is not None else []
    info_path = cfg.dataset.info_path

    if dataset_root is not None and _is_local_lerobot_dataset(dataset_root):
        if isinstance(cfg.dataset.repo_id, list):
            raise ValueError(
                "`dataset.root` points to a single dataset tree, so `dataset.repo_id` cannot be a list."
            )
        effective_repo_id = cfg.dataset.repo_id if isinstance(cfg.dataset.repo_id, str) else dataset_root.name
        dataset = _build_single_dataset(
            repo_id=effective_repo_id,
            root=dataset_root,
            info_path=info_path,
            cfg=cfg,
            image_transforms=image_transforms,
        )
    elif discovered_repo_ids:
        if cfg.dataset.repo_id is None:
            dataset = _build_multi_dataset(
                repo_ids=discovered_repo_ids,
                root=dataset_root,
                info_path=info_path,
                cfg=cfg,
                image_transforms=image_transforms,
            )
        elif isinstance(cfg.dataset.repo_id, str):
            if cfg.dataset.repo_id not in discovered_repo_ids:
                raise ValueError(
                    f"`dataset.repo_id={cfg.dataset.repo_id}` was not found under multi-dataset root {dataset_root}."
                )
            dataset = _build_single_dataset(
                repo_id=cfg.dataset.repo_id,
                root=dataset_root / cfg.dataset.repo_id,
                info_path=info_path,
                cfg=cfg,
                image_transforms=image_transforms,
            )
        else:
            missing_repo_ids = [repo_id for repo_id in cfg.dataset.repo_id if repo_id not in discovered_repo_ids]
            if missing_repo_ids:
                raise ValueError(
                    f"Some `dataset.repo_id` entries were not found under {dataset_root}: {missing_repo_ids}"
                )
            dataset = _build_multi_dataset(
                repo_ids=cfg.dataset.repo_id,
                root=dataset_root,
                info_path=info_path,
                cfg=cfg,
                image_transforms=image_transforms,
            )
    elif isinstance(cfg.dataset.repo_id, str):
        dataset = _build_single_dataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            info_path=info_path,
            cfg=cfg,
            image_transforms=image_transforms,
        )
    elif isinstance(cfg.dataset.repo_id, list):
        dataset = _build_multi_dataset(
            repo_ids=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            info_path=info_path,
            cfg=cfg,
            image_transforms=image_transforms,
        )
    else:
        raise ValueError(
            f"`dataset.root={cfg.dataset.root}` is neither a LeRobot dataset nor a multi-dataset root, "
            "and `dataset.repo_id` was not provided."
        )

    if cfg.dataset.use_imagenet_stats:
        visual_feature_keys = [
            key
            for key, feature in dataset.meta.features.items()
            if feature["dtype"] in {"image", "video", "tactile", "depth"}
        ]
        for key in visual_feature_keys:
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
