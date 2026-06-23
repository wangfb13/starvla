#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class EpisodePair:
    episode_name: str
    action: np.ndarray
    state: np.ndarray


def _discover_data_files(dataset_root: Path) -> tuple[str, list[Path]]:
    episode_files = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if episode_files:
        return "per_episode", episode_files

    chunk_files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if chunk_files:
        return "chunked", chunk_files

    raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")


def iter_episode_pairs(dataset_root: Path, action_slice: slice, state_slice: slice):
    layout, data_files = _discover_data_files(dataset_root)

    if layout == "per_episode":
        for path in data_files:
            table = pq.read_table(path, columns=["action", "observation.state"])
            action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)[:, action_slice]
            state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)[:, state_slice]
            yield EpisodePair(path.stem, action, state)
        return

    for path in data_files:
        table = pq.read_table(path, columns=["action", "observation.state", "episode_index"])
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)[:, action_slice]
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)[:, state_slice]
        episode_index = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)

        # v3.0 files are typically sorted by episode_index and contiguous per episode.
        split_points = np.where(np.diff(episode_index) != 0)[0] + 1
        starts = np.concatenate(([0], split_points))
        ends = np.concatenate((split_points, [len(episode_index)]))

        for start, end in zip(starts, ends, strict=True):
            episode_id = int(episode_index[start])
            yield EpisodePair(
                episode_name=f"episode_{episode_id:06d}",
                action=action[start:end],
                state=state[start:end],
            )
