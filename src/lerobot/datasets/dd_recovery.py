"""Read-only, boundary-safe DD recovery views over existing LeRobot datasets.

The original action/state sidecar and checkpoint processors remain authoritative.
This module adds current-frame physical D/Dx/Dy; it
never estimates normalization statistics or changes source data/gel references.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

DD_KEY = "observation.tactile_dd"
DD_VIDEO_KEYS = (
    "observation.depth_deformation.tactile_right_left",
    "observation.depth_deformation.tactile_right_right",
)


def decode_dd_rgb48(rgb: np.ndarray) -> np.ndarray:
    """Decode uint16 RGB packing to float32 [3,H,W] in original SDK units."""
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected uint16 [H,W,3], got {rgb.dtype} {rgb.shape}")
    dd = rgb.astype(np.float32)
    dd[..., 1:] -= 30000.0  # Float subtraction is essential: uint16 would wrap.
    dd /= 1000.0
    return np.ascontiguousarray(dd.transpose(2, 0, 1))


def safe_anchor_frames(start: int, end: int, action_offsets: list[int]) -> np.ndarray:
    """Half-open segment; every action target must stay within this segment."""
    if not action_offsets or min(action_offsets) < 0:
        raise ValueError("Recovery action offsets must be nonempty and nonnegative")
    if start < 0 or end < start:
        raise ValueError("Invalid recovery segment")
    return np.arange(start, max(start, end - max(action_offsets)), dtype=np.int64)


def history_cut_mask(frame: int, start: int, offsets: list[int]) -> np.ndarray:
    return np.asarray(offsets, dtype=np.int64) + int(frame) < int(start)


def grouped_episode_split(episodes: list[int], seed: int) -> dict[int, str]:
    """Deterministic group split: 34 recovery episodes -> 24/5/5."""
    ids = np.asarray(sorted(episodes), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    holdout = max(1, int(round(len(ids) * 0.15))) if len(ids) >= 3 else 0
    n_train = len(ids) - 2 * holdout
    return {int(ep): ("train" if i < n_train else "val" if i < n_train + holdout else "test")
            for i, ep in enumerate(ids)}


class DDRecoveryDataset:
    """Map-style Dataset with source-safe anchors and lazily memory-mapped DD.

    base_datasets are FULL (no episode filtering) LeRobotDataset instances keyed
    by source alias. Source global row indices are validated during preparation.
    The class intentionally does not inherit torch Dataset, so preparation/tests
    can import decode and boundary helpers without importing torch.
    """

    def __init__(self, base_datasets: dict[str, Any], manifest_path: str | Path, split: str = "train",
                 max_open_caches: int = 8):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported DD recovery manifest schema")
        if not self.manifest.get("cache_ready"):
            raise ValueError("DD caches are not complete; run prepare_dd_recovery.py without --metadata-only")
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.base_datasets = base_datasets
        self.split = split
        self.max_open_caches = max_open_caches
        self._caches: OrderedDict[str, np.ndarray] = OrderedDict()
        self.anchors: list[dict[str, Any]] = []
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == split]
        for episode in self.episodes:
            for frame in safe_anchor_frames(episode["start"], episode["end"], self.manifest["action_offsets"]):
                self.anchors.append({
                    "source": episode["source"], "episode_index": episode["episode_index"],
                    "frame_index": int(frame), "source_index": episode["dataset_from_index"] + int(frame),
                    "segment_start": episode["start"], "segment_end": episode["end"],
                    "cache": episode["cache"],
                })
        self.num_frames = len(self.anchors)
        self.num_episodes = len(self.episodes)
        self.meta = next(iter(base_datasets.values())).meta if base_datasets else None
        self.repo_id = f"dd_recovery_{split}"

    def __len__(self):
        return len(self.anchors)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_caches"] = OrderedDict()  # Reopen memmaps inside DataLoader workers.
        return state

    def _dd(self, anchor: dict[str, Any]) -> np.ndarray:
        key = anchor["cache"]
        if key not in self._caches:
            p = Path(key)
            if not p.is_absolute():
                p = self.manifest_path.parent / p
            value = np.load(p, mmap_mode="r", allow_pickle=False)
            if value.ndim != 5 or tuple(value.shape[1:]) != (2, 3, 96, 128):
                raise ValueError(f"Invalid DD cache shape: {p}: {value.shape}")
            self._caches[key] = value
            while len(self._caches) > self.max_open_caches:
                self._caches.popitem(last=False)
        self._caches.move_to_end(key)
        # Copy and convert: memmap remains read-only; torch receives owned float32.
        return np.array(self._caches[key][anchor["frame_index"]], dtype=np.float32, copy=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        anchor = self.anchors[index]
        base = self.base_datasets[anchor["source"]]
        item = dict(base[anchor["source_index"]])
        for key, expected in (("episode_index", anchor["episode_index"]), ("frame_index", anchor["frame_index"])):
            if int(item[key]) != expected:
                raise RuntimeError(f"Base dataset row mismatch for {anchor}: {key}={item[key]}")
        if "action_is_pad" in item and bool(item["action_is_pad"].any()):
            raise RuntimeError(f"Unsafe padded action target for {anchor}")
        # A filtered anchor list alone does NOT isolate history. Replace all
        # pre-segment observation histories before checkpoint preprocessing.
        delta_timestamps = base.delta_timestamps or {}
        fps = float(self.manifest["fps"])
        for key, times in delta_timestamps.items():
            if not key.startswith("observation.") or key not in item:
                continue
            offsets = [int(round(t * fps)) for t in times]
            cut = history_cut_mask(anchor["frame_index"], anchor["segment_start"], offsets)
            value = item[key]
            if len(offsets) != value.shape[0]:
                raise ValueError(f"Unexpected history shape {key}: {value.shape}, offsets={offsets}")
            mask = torch.as_tensor(cut, dtype=torch.bool, device=value.device)
            pad_key = key + "_is_pad"
            original_mask = item.get(pad_key, torch.zeros_like(mask)).to(dtype=torch.bool)
            item[pad_key] = original_mask | mask
            if bool(mask.any()):
                retained = (~mask).nonzero(as_tuple=False).flatten()
                if not len(retained):
                    raise ValueError("History contains no retained observation")
                value = value.clone()
                # Keep rotations numerically valid; force processor zeros every
                # masked history row. No value from before the cut is returned.
                value[mask] = value[int(retained[0])].clone()
                item[key] = value
        item[DD_KEY] = torch.from_numpy(self._dd(anchor))
        return item

    def sampling_weights(self, normal_fraction: float = 0.7) -> np.ndarray:
        """70/30 source-group replay by default, uniform anchors within groups."""
        if not 0.0 < normal_fraction < 1.0:
            raise ValueError("normal_fraction must lie strictly between 0 and 1")
        recovery = np.asarray([a["source"] == "recovery" for a in self.anchors])
        if not recovery.any() or recovery.all():
            raise ValueError("Both recovery and normal anchors are required for replay")
        return np.where(recovery, (1 - normal_fraction) / recovery.sum(), normal_fraction / (~recovery).sum())


def make_dd_recovery_datasets(policy_config: Any, manifest_path: str | Path, *, info_path: str | Path,
                              video_backend: str = "pyav"):
    """Return train/val/test views; DO NOT compute or load dataset action stats."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads(Path(manifest_path).read_text())
    requested = list(policy_config.action_delta_indices)
    if requested != manifest["action_offsets"]:
        raise ValueError(f"Action offsets differ from prepared boundaries: {requested}")
    overrides = getattr(policy_config, "action_gap_dataset_overrides", None)
    if overrides:
        raise ValueError("DD recovery uses explicit common action offsets; dataset gap overrides are unsupported")
    bases = {}
    for source in manifest["sources"]:
        name = source["name"]
        root = Path(source["root"])
        sidecar = source.get("sidecar")
        if not sidecar or not Path(sidecar).is_file():
            raise FileNotFoundError(f"Missing canonical sidecar for {name}: {sidecar}")
        meta = LeRobotDatasetMetadata(root.name, root=root, info_path=info_path, sidecar_path=sidecar)
        unexpected = set(meta.video_keys) - {"observation.images.cam_left_undist", "observation.images.cam_right_undist"}
        if unexpected:
            raise ValueError(f"Use original filtered wrist-only info template; extra videos: {sorted(unexpected)}")
        delta = resolve_delta_timestamps(policy_config, meta)
        bases[name] = LeRobotDataset(root.name, root=root, info_path=info_path, sidecar_path=sidecar,
                                    delta_timestamps=delta, video_backend=video_backend, image_transforms=None)
        if len(bases[name]) != source["total_frames"]:
            raise ValueError(f"Source changed after preparation: {name} frame count")
    return tuple(DDRecoveryDataset(bases, manifest_path, split) for split in ("train", "val", "test"))
