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
from lerobot.datasets.sidecar_utils import (
    dedupe_paths,
    filter_sidecar_normalization_stats,
    infer_gap_override_from_robot_type,
    load_stats_json,
    read_dataset_roots_file,
    resolve_sidecar_path,
    resolve_sidecar_stats_path,
)
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
    sidecar_path: str | Path | None = None,
):
    ds_meta = LeRobotDatasetMetadata(
        repo_id,
        root=root,
        revision=cfg.dataset.revision,
        info_path=info_path,
        sidecar_path=sidecar_path,
    )
    _apply_automatic_action_gap_overrides(cfg.policy, [(0, ds_meta)])
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta, dataset_index=0)
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
            sidecar_path=sidecar_path,
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
    root_paths_by_repo: dict[str, str | Path] | None = None,
    sidecar_paths: dict[str, str | Path] | None = None,
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
        root_paths_by_repo=root_paths_by_repo,
        info_path=info_path,
        delta_timestamps=None,
        image_transforms=image_transforms,
        video_backend=cfg.dataset.video_backend,
        sidecar_paths=sidecar_paths,
    )
    child_datasets = list(dataset.iter_datasets())
    _apply_automatic_action_gap_overrides(
        cfg.policy,
        [(dataset_index, child_dataset.meta) for dataset_index, child_dataset in enumerate(child_datasets)],
    )
    if _needs_per_dataset_action_delta_timestamps(cfg.policy, child_datasets):
        delta_timestamps_by_repo = {
            repo_id: resolve_delta_timestamps(cfg.policy, child_dataset.meta, dataset_index=dataset_index)
            for dataset_index, (repo_id, child_dataset) in enumerate(zip(repo_ids, child_datasets))
        }
        dataset.set_delta_timestamps(delta_timestamps_by_repo)
    else:
        dataset.set_delta_timestamps(resolve_delta_timestamps(cfg.policy, dataset.meta))
    logging.info(
        "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
        f"{pformat(dataset.repo_id_to_index, indent=2)}"
    )
    return dataset


def _resolve_roots_file_dataset_args(
    roots_file: str | Path | None,
) -> tuple[list[str], Path | None, dict[str, Path] | None]:
    if roots_file is None:
        return [], None, None
    roots = dedupe_paths(read_dataset_roots_file(roots_file))
    if not roots:
        raise ValueError(f"No dataset roots were found in dataset.roots_file={roots_file}")

    repo_ids = [root.name for root in roots]
    duplicate_repo_ids = sorted({repo_id for repo_id in repo_ids if repo_ids.count(repo_id) > 1})
    if duplicate_repo_ids:
        raise ValueError(
            "dataset.roots_file contains duplicate dataset directory names, which would collide as repo_id "
            f"and sidecar directory keys: {duplicate_repo_ids}"
        )

    parent = roots[0].parent
    if all(root.parent == parent for root in roots):
        return repo_ids, parent, None
    return repo_ids, None, dict(zip(repo_ids, roots, strict=True))


def _resolve_sidecar_paths_for_repo_ids(
    *,
    repo_ids: list[str],
    root: str | Path | None,
    sidecar_root: str | Path | None,
    sidecar_path: str | Path | dict[str, str | Path] | None,
    root_paths_by_repo: dict[str, str | Path] | None = None,
) -> dict[str, Path] | None:
    if sidecar_root is None and sidecar_path is None:
        return None
    if root is None and root_paths_by_repo is None:
        raise ValueError("dataset.sidecar_root/sidecar_path requires local dataset.root or dataset.roots_file.")

    root = Path(root) if root is not None else None
    sidecar_paths: dict[str, Path] = {}
    for repo_id in repo_ids:
        if root_paths_by_repo is not None and repo_id in root_paths_by_repo:
            dataset_root = Path(root_paths_by_repo[repo_id])
        elif root is not None:
            dataset_root = root / repo_id
        else:
            continue
        resolved = resolve_sidecar_path(
            dataset_root=dataset_root,
            sidecar_root=sidecar_root,
            sidecar_path=sidecar_path,
        )
        if resolved is not None:
            sidecar_paths[repo_id] = resolved
    return sidecar_paths or None


def _explicit_action_gap_override_indices(policy_cfg: PreTrainedConfig) -> set[int]:
    parser = getattr(policy_cfg, "parse_action_gap_dataset_override", None)
    explicit_indices: set[int] = set()
    for spec in getattr(policy_cfg, "action_gap_dataset_overrides", []) or []:
        if callable(parser):
            dataset_index, _ = parser(spec)
        else:
            dataset_index = int(str(spec).split(":", maxsplit=1)[0])
        explicit_indices.add(int(dataset_index))
    return explicit_indices


def _robot_type_from_meta(ds_meta: LeRobotDatasetMetadata) -> str | None:
    try:
        return getattr(ds_meta, "robot_type", None)
    except (KeyError, AttributeError):
        return None


def _apply_automatic_action_gap_overrides(
    policy_cfg: PreTrainedConfig,
    indexed_metas: list[tuple[int, LeRobotDatasetMetadata]],
) -> None:
    """Infer per-dataset action gaps from robot_type without overriding user choices.

    UMI datasets keep the policy-level gap. Real-robot datasets currently record
    state/action with the control delay already baked in, so they default to gap 0.
    """
    if not hasattr(policy_cfg, "action_gap_dataset_overrides"):
        return

    if bool(getattr(policy_cfg, "apply_action_gap_to_real_datasets", False)):
        return

    explicit_indices = _explicit_action_gap_override_indices(policy_cfg)
    auto_specs: list[str] = []
    for dataset_index, ds_meta in indexed_metas:
        if int(dataset_index) in explicit_indices:
            continue
        gap_override = infer_gap_override_from_robot_type(_robot_type_from_meta(ds_meta))
        if gap_override is None:
            continue
        auto_specs.append(f"{int(dataset_index)}:{int(gap_override)}")

    if not auto_specs:
        return

    existing_specs = list(getattr(policy_cfg, "action_gap_dataset_overrides", []) or [])
    policy_cfg.action_gap_dataset_overrides = existing_specs + auto_specs
    logging.info(
        "Applied automatic action gap overrides from dataset robot_type: %s",
        auto_specs,
    )


def _needs_per_dataset_action_delta_timestamps(
    policy_cfg: PreTrainedConfig,
    child_datasets: list[LeRobotDataset],
) -> bool:
    if getattr(policy_cfg, "action_gap_dataset_overrides", None):
        return True

    global_gap = getattr(policy_cfg, "action_gap_frames", None)
    for child_dataset in child_datasets:
        gap_override = infer_gap_override_from_robot_type(_robot_type_from_meta(child_dataset.meta))
        if gap_override is not None and gap_override != global_gap:
            return True
    return False


def _load_sidecar_stats_for_dataset(cfg: TrainPipelineConfig, dataset) -> dict | None:
    sidecar_stats_configured = any(
        value is not None
        for value in (
            cfg.dataset.sidecar_root,
            cfg.dataset.sidecar_path,
            cfg.dataset.stats_path,
            cfg.dataset.relative_stats_path,
        )
    )
    if not sidecar_stats_configured:
        return None

    use_relative_actions = bool(getattr(cfg.policy, "use_relative_actions", False))
    relative_chunk_size = getattr(cfg.policy, "chunk_size", None)
    default_relative_action_gap_frames = getattr(cfg.policy, "action_gap_frames", None)
    relative_exclude_joints = getattr(cfg.policy, "relative_exclude_joints", None) or []
    relative_action_keyframes = getattr(cfg.policy, "relative_action_keyframes", None) or []
    relative_exclude_keys = getattr(cfg.policy, "relative_exclude_keys", None) or []
    force_dynamic_relative_stats = use_relative_actions and bool(relative_action_keyframes)

    def _action_names_for_relative_stats(child_dataset) -> list[str] | None:
        action_names = getattr(cfg.policy, "action_feature_names", None)
        if action_names:
            return list(action_names)

        action_shape = tuple(child_dataset.meta.features.get(ACTION, {}).get("shape", ()))
        canonical_action_dim = getattr(cfg.policy, "canonical_action_dim", None)
        if cfg.policy.type in {"pi05", "pi05_full"} and action_shape == (canonical_action_dim,):
            if cfg.policy.type == "pi05_full":
                from lerobot.policies.pi05_full.canonical_bimanual import canonical_action_feature_names
            else:
                from lerobot.policies.pi05.canonical_bimanual import canonical_action_feature_names

            return canonical_action_feature_names()

        feature_names = child_dataset.meta.features.get(ACTION, {}).get("names")
        return list(feature_names) if feature_names is not None else None

    def _state_names_for_sidecar_stats(child_dataset) -> list[str] | None:
        state_names = getattr(cfg.policy, "state_feature_names", None)
        if state_names:
            return list(state_names)

        state_shape = tuple(child_dataset.meta.features.get("observation.state", {}).get("shape", ()))
        canonical_state_dim = getattr(cfg.policy, "canonical_state_dim", None)
        if cfg.policy.type in {"pi05", "pi05_full"} and state_shape == (canonical_state_dim,):
            if cfg.policy.type == "pi05_full":
                from lerobot.policies.pi05_full.canonical_bimanual import canonical_state_feature_names
            else:
                from lerobot.policies.pi05.canonical_bimanual import canonical_state_feature_names

            return canonical_state_feature_names()

        feature_names = child_dataset.meta.features.get("observation.state", {}).get("names")
        return list(feature_names) if feature_names is not None else None

    def _effective_gap_for_child(child_dataset, dataset_index: int | None = None) -> int | None:
        if dataset_index is not None:
            gap_overrides = getattr(cfg.policy, "action_gap_frames_by_dataset_index", {})
            if dataset_index in gap_overrides:
                return int(gap_overrides[dataset_index])
        robot_type_gap = infer_gap_override_from_robot_type(_robot_type_from_meta(child_dataset.meta))
        if robot_type_gap is not None:
            return robot_type_gap
        return default_relative_action_gap_frames

    def _dynamic_relative_stats(child_dataset, gap_frames: int | None) -> dict | None:
        if not use_relative_actions:
            return None
        sidecar = getattr(getattr(child_dataset, "reader", None), "sidecar", None)
        if sidecar is None:
            return None
        action_names = _action_names_for_relative_stats(child_dataset)
        if action_names is None:
            action_names = [f"action_{idx}" for idx in range(child_dataset.meta.features[ACTION]["shape"][0])]
        action_stats = sidecar.relative_action_chunk_stats(
            chunk_size=int(relative_chunk_size),
            action_gap_frames=int(gap_frames or 0),
            action_names=list(action_names),
            relative_exclude_joints=relative_exclude_joints,
            relative_action_keyframes=list(relative_action_keyframes),
            relative_exclude_keys=list(relative_exclude_keys),
        )
        state_names = _state_names_for_sidecar_stats(child_dataset)
        if state_names is None:
            state_names = [
                f"state_{idx}"
                for idx in range(child_dataset.meta.features["observation.state"]["shape"][0])
            ]
        state_stats = sidecar.masked_vector_stats("observation.state", "state_mask", state_names)
        if action_stats is None or state_stats is None:
            return None
        logging.info(
            "Computed dynamic sidecar relative stats for %s (chunk_size=%s, gap=%s).",
            child_dataset.root,
            relative_chunk_size,
            gap_frames,
        )
        return {"observation.state": state_stats, ACTION: action_stats}

    if hasattr(dataset, "iter_datasets"):
        source_stats = []
        child_datasets = list(dataset.iter_datasets())
        for dataset_index, child_dataset in enumerate(child_datasets):
            relative_action_gap_frames = _effective_gap_for_child(child_dataset, dataset_index)
            stats_path = resolve_sidecar_stats_path(
                dataset_root=child_dataset.root,
                sidecar_root=cfg.dataset.sidecar_root,
                sidecar_path=cfg.dataset.sidecar_path,
                stats_path=cfg.dataset.stats_path,
                relative_stats_path=cfg.dataset.relative_stats_path,
                relative_chunk_size=relative_chunk_size,
                relative_action_gap_frames=relative_action_gap_frames,
                use_relative_actions=use_relative_actions,
            )
            stats = None if force_dynamic_relative_stats else load_stats_json(stats_path)
            stats = filter_sidecar_normalization_stats(stats)
            if stats is None and use_relative_actions:
                stats = _dynamic_relative_stats(child_dataset, relative_action_gap_frames)
            if stats is not None:
                source_stats.append(stats)
            elif use_relative_actions:
                raise FileNotFoundError(
                    "Missing usable sidecar relative stats and dynamic computation failed for "
                    f"{child_dataset.root}. Expected chunk_size={relative_chunk_size} "
                    f"gap={relative_action_gap_frames}."
                )
            else:
                logging.warning(
                    "Missing usable sidecar stats for %s under sidecar stats path.",
                    child_dataset.root,
                )
        if len(source_stats) == len(child_datasets):
            from lerobot.datasets.compute_stats import aggregate_stats

            return aggregate_stats(source_stats)
        return None

    relative_action_gap_frames = _effective_gap_for_child(dataset, None)
    stats_path = resolve_sidecar_stats_path(
        dataset_root=dataset.root,
        sidecar_root=cfg.dataset.sidecar_root,
        sidecar_path=cfg.dataset.sidecar_path,
        stats_path=cfg.dataset.stats_path,
        relative_stats_path=cfg.dataset.relative_stats_path,
        relative_chunk_size=relative_chunk_size,
        relative_action_gap_frames=relative_action_gap_frames,
        use_relative_actions=use_relative_actions,
    )
    stats = None if force_dynamic_relative_stats else filter_sidecar_normalization_stats(load_stats_json(stats_path))
    if stats is None and use_relative_actions:
        stats = _dynamic_relative_stats(dataset, relative_action_gap_frames)
    if stats is None and use_relative_actions:
        raise FileNotFoundError(
            "Missing usable sidecar relative stats and dynamic computation failed for "
            f"{dataset.root}. Expected chunk_size={relative_chunk_size} gap={relative_action_gap_frames} "
        )
    return stats


def _resolve_action_delta_indices(
    cfg: PreTrainedConfig,
    dataset_index: int | None = None,
) -> list | None:
    if dataset_index is not None:
        getter = getattr(cfg, "get_action_delta_indices", None)
        if callable(getter):
            return getter(dataset_index)
    return cfg.action_delta_indices


def resolve_delta_timestamps(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata,
    *,
    dataset_index: int | None = None,
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
    action_delta_indices = _resolve_action_delta_indices(cfg, dataset_index)
    feature_source = getattr(ds_meta, "reader_features", ds_meta.features)
    for key in feature_source:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in action_delta_indices]
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

    roots_file_repo_ids, roots_file_parent, roots_file_paths_by_repo = _resolve_roots_file_dataset_args(
        cfg.dataset.roots_file
    )
    dataset_root = roots_file_parent if roots_file_parent is not None else (
        Path(cfg.dataset.root) if cfg.dataset.root is not None else None
    )
    discovered_repo_ids = _discover_child_datasets(dataset_root) if dataset_root is not None else []
    info_path = cfg.dataset.info_path

    if roots_file_repo_ids:
        if cfg.dataset.repo_id is not None:
            logging.warning("dataset.roots_file is set; ignoring dataset.repo_id=%s", cfg.dataset.repo_id)
        sidecar_paths = _resolve_sidecar_paths_for_repo_ids(
            repo_ids=roots_file_repo_ids,
            root=dataset_root,
            root_paths_by_repo=roots_file_paths_by_repo,
            sidecar_root=cfg.dataset.sidecar_root,
            sidecar_path=cfg.dataset.sidecar_path,
        )
        dataset = _build_multi_dataset(
            repo_ids=roots_file_repo_ids,
            root=dataset_root,
            root_paths_by_repo=roots_file_paths_by_repo,
            info_path=info_path,
            cfg=cfg,
            image_transforms=image_transforms,
            sidecar_paths=sidecar_paths,
        )
    elif dataset_root is not None and _is_local_lerobot_dataset(dataset_root):
        if isinstance(cfg.dataset.repo_id, list):
            raise ValueError(
                "`dataset.root` points to a single dataset tree, so `dataset.repo_id` cannot be a list."
            )
        effective_repo_id = cfg.dataset.repo_id if isinstance(cfg.dataset.repo_id, str) else dataset_root.name
        sidecar_path = resolve_sidecar_path(
            dataset_root=dataset_root,
            sidecar_root=cfg.dataset.sidecar_root,
            sidecar_path=cfg.dataset.sidecar_path,
        )
        dataset = _build_single_dataset(
            repo_id=effective_repo_id,
            root=dataset_root,
            info_path=info_path,
            cfg=cfg,
            image_transforms=image_transforms,
            sidecar_path=sidecar_path,
        )
    elif discovered_repo_ids:
        if cfg.dataset.repo_id is None:
            sidecar_paths = _resolve_sidecar_paths_for_repo_ids(
                repo_ids=discovered_repo_ids,
                root=dataset_root,
                root_paths_by_repo=None,
                sidecar_root=cfg.dataset.sidecar_root,
                sidecar_path=cfg.dataset.sidecar_path,
            )
            dataset = _build_multi_dataset(
                repo_ids=discovered_repo_ids,
                root=dataset_root,
                root_paths_by_repo=None,
                info_path=info_path,
                cfg=cfg,
                image_transforms=image_transforms,
                sidecar_paths=sidecar_paths,
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
                sidecar_path=resolve_sidecar_path(
                    dataset_root=dataset_root / cfg.dataset.repo_id,
                    sidecar_root=cfg.dataset.sidecar_root,
                    sidecar_path=cfg.dataset.sidecar_path,
                ),
            )
        else:
            missing_repo_ids = [repo_id for repo_id in cfg.dataset.repo_id if repo_id not in discovered_repo_ids]
            if missing_repo_ids:
                raise ValueError(
                    f"Some `dataset.repo_id` entries were not found under {dataset_root}: {missing_repo_ids}"
                )
            sidecar_paths = _resolve_sidecar_paths_for_repo_ids(
                repo_ids=cfg.dataset.repo_id,
                root=dataset_root,
                root_paths_by_repo=None,
                sidecar_root=cfg.dataset.sidecar_root,
                sidecar_path=cfg.dataset.sidecar_path,
            )
            dataset = _build_multi_dataset(
                repo_ids=cfg.dataset.repo_id,
                root=dataset_root,
                root_paths_by_repo=None,
                info_path=info_path,
                cfg=cfg,
                image_transforms=image_transforms,
                sidecar_paths=sidecar_paths,
            )
    elif isinstance(cfg.dataset.repo_id, str):
        sidecar_path = (
            resolve_sidecar_path(
                dataset_root=cfg.dataset.root,
                sidecar_root=cfg.dataset.sidecar_root,
                sidecar_path=cfg.dataset.sidecar_path,
            )
            if cfg.dataset.root is not None
            else cfg.dataset.sidecar_path
        )
        dataset = _build_single_dataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            info_path=info_path,
            cfg=cfg,
            image_transforms=image_transforms,
            sidecar_path=sidecar_path,
        )
    elif isinstance(cfg.dataset.repo_id, list):
        sidecar_paths = _resolve_sidecar_paths_for_repo_ids(
            repo_ids=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            root_paths_by_repo=None,
            sidecar_root=cfg.dataset.sidecar_root,
            sidecar_path=cfg.dataset.sidecar_path,
        )
        dataset = _build_multi_dataset(
            repo_ids=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            root_paths_by_repo=None,
            info_path=info_path,
            cfg=cfg,
            image_transforms=image_transforms,
            sidecar_paths=sidecar_paths,
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

    sidecar_stats = _load_sidecar_stats_for_dataset(cfg, dataset)
    dataset.meta.sidecar_stats_loaded = sidecar_stats is not None
    if sidecar_stats is not None:
        logging.info("Using precomputed sidecar stats for dataset normalization.")
        dataset.meta.stats.update(sidecar_stats)

    return dataset
