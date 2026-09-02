#!/usr/bin/env python

import json

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.sidecar_utils import (
    FramewiseSidecar,
    filter_sidecar_normalization_stats,
    infer_gap_override_from_robot_type,
    read_dataset_roots_file,
    resolve_sidecar_path,
    resolve_sidecar_stats_path,
)
from lerobot.utils.constants import ACTION, OBS_ACTION_MASK, OBS_STATE, OBS_STATE_MASK


def test_roots_file_and_sidecar_path_resolution(tmp_path):
    roots_file = tmp_path / "roots.txt"
    roots_file.write_text(
        """
        # comments are ignored
        datasets/source_a
        /absent/source_b  # inline comments are ignored
        """,
        encoding="utf-8",
    )

    roots = read_dataset_roots_file(roots_file)

    assert roots[0] == (tmp_path / "datasets/source_a").resolve()
    assert roots[1].as_posix() == "/absent/source_b"
    assert resolve_sidecar_path(
        dataset_root=roots[0],
        sidecar_root=tmp_path / "sidecars",
    ) == tmp_path / "sidecars/source_a/framewise_guidance.parquet"
    assert resolve_sidecar_path(
        dataset_root=roots[0],
        sidecar_path={"source_a": tmp_path / "explicit.parquet"},
    ) == tmp_path / "explicit.parquet"


def test_relative_stats_path_resolution_and_filtering(tmp_path):
    dataset_root = tmp_path / "dataset_a"
    stats_path = resolve_sidecar_stats_path(
        dataset_root=dataset_root,
        sidecar_root=tmp_path / "sidecars",
        relative_chunk_size=50,
        relative_action_gap_frames=6,
        use_relative_actions=True,
    )

    assert stats_path == tmp_path / "sidecars/dataset_a/stats_relative_action_chunk50_gap6.json"
    assert resolve_sidecar_stats_path(
        dataset_root=dataset_root,
        sidecar_path=tmp_path / "explicit/framewise_guidance.parquet",
        relative_chunk_size=50,
        relative_action_gap_frames=6,
        use_relative_actions=True,
    ) == tmp_path / "explicit/stats_relative_action_chunk50_gap6.json"

    raw_stats = {
        OBS_STATE: {"mean": [0.0, 1.0], "std": [1.0, 1.0]},
        ACTION: {"mean": [2.0, 3.0], "std": [1.0, 1.0]},
        "observation.images.cam_left": {"mean": [[[0.0]]], "std": [[[1.0]]]},
    }
    filtered = filter_sidecar_normalization_stats(raw_stats)

    assert set(filtered) == {OBS_STATE, ACTION}


def test_framewise_sidecar_overlays_vectors_text_and_masks(tmp_path):
    sidecar_path = tmp_path / "framewise_guidance.parquet"
    pd.DataFrame(
        [
            {
                "index": 0,
                OBS_STATE: np.array([1.0, 2.0], dtype=np.float32),
                ACTION: np.array([10.0, 20.0], dtype=np.float32),
                "state_mask": np.array([1.0, 0.0], dtype=np.float32),
                "action_valid_mask": np.array([1.0, 1.0], dtype=np.float32),
                "subtask_text_sidecar": "align peg",
                "next_subtask_text_sidecar": "insert peg",
                "user_prompt_sidecar": "finish the assembly",
            },
            {
                "index": 1,
                OBS_STATE: np.array([3.0, 4.0], dtype=np.float32),
                ACTION: np.array([30.0, 40.0], dtype=np.float32),
                "state_mask": np.array([1.0, 1.0], dtype=np.float32),
                "action_valid_mask": np.array([0.0, 1.0], dtype=np.float32),
                "subtask_text_sidecar": "",
                "next_subtask_text_sidecar": "",
                "user_prompt_sidecar": "",
            },
        ]
    ).to_parquet(sidecar_path, index=False)

    sidecar = FramewiseSidecar(sidecar_path, repo_id="test/repo")
    item = sidecar.overlay(
        {
            "index": torch.tensor(0),
            OBS_STATE: torch.zeros(2),
            ACTION: torch.zeros(2),
        },
        abs_idx=0,
    )

    assert torch.allclose(item[OBS_STATE], torch.tensor([1.0, 2.0]))
    assert torch.allclose(item[ACTION], torch.tensor([10.0, 20.0]))
    assert torch.allclose(item[OBS_STATE_MASK], torch.tensor([1.0, 0.0]))
    assert torch.allclose(item[OBS_ACTION_MASK], torch.tensor([1.0, 1.0]))
    assert item["subtask"] == "align peg"
    assert item["next_subtask"] == "insert peg"
    assert item["user_prompt"] == "finish the assembly"
    assert sidecar.vector_dim(OBS_STATE_MASK) == 2
    assert sidecar.vector_dim(OBS_ACTION_MASK) == 2

    chunk = sidecar.values_for_indices(ACTION, [0, 1])
    assert torch.allclose(chunk, torch.tensor([[10.0, 20.0], [30.0, 40.0]]))


def test_framewise_sidecar_dynamic_relative_action_chunk_stats(tmp_path):
    sidecar_path = tmp_path / "framewise_guidance.parquet"
    pd.DataFrame(
        [
            {
                "index": 0,
                "episode_index": 0,
                OBS_STATE: np.array([1.0, 10.0], dtype=np.float32),
                ACTION: np.array([2.0, 100.0], dtype=np.float32),
                "action_valid_mask": np.array([1.0, 1.0], dtype=np.float32),
            },
            {
                "index": 1,
                "episode_index": 0,
                OBS_STATE: np.array([2.0, 20.0], dtype=np.float32),
                ACTION: np.array([4.0, 200.0], dtype=np.float32),
                "action_valid_mask": np.array([1.0, 1.0], dtype=np.float32),
            },
            {
                "index": 2,
                "episode_index": 0,
                OBS_STATE: np.array([3.0, 30.0], dtype=np.float32),
                ACTION: np.array([8.0, 300.0], dtype=np.float32),
                "action_valid_mask": np.array([1.0, 1.0], dtype=np.float32),
            },
        ]
    ).to_parquet(sidecar_path, index=False)

    stats = FramewiseSidecar(sidecar_path).relative_action_chunk_stats(
        chunk_size=2,
        action_gap_frames=1,
        action_names=["x", "gripper"],
        relative_exclude_joints=["gripper"],
    )

    assert stats is not None
    # Start frames produce clamped chunks: [4-1, 8-1], [8-2, 8-2], [8-3, 8-3].
    assert stats["min"].tolist() == [3.0, 200.0]
    assert stats["max"].tolist() == [7.0, 300.0]
    assert stats["count"].tolist() == [6.0]


def test_framewise_sidecar_relative_state_history_stats_uses_current_anchor(tmp_path):
    sidecar_path = tmp_path / "framewise_guidance.parquet"
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rows = [
        {
            "index": index,
            "episode_index": 0,
            OBS_STATE: np.array([x, 0.0, 0.0, *identity_rot6d, 0.5], dtype=np.float32),
            "state_mask": np.ones(10, dtype=np.float32),
        }
        for index, x in enumerate([1.0, 2.0])
    ]
    pd.DataFrame(rows).to_parquet(sidecar_path, index=False)
    stats = FramewiseSidecar(sidecar_path).relative_state_history_stats(
        history_horizon=2,
        state_names=[f"state_{idx}" for idx in range(10)],
        relative_action_keyframes=["0:3"],
    )
    assert stats is not None
    assert stats["min"][0] == -1.0
    assert stats["max"][0] == 0.0
    assert stats["mean"][0] == -0.25


def test_framewise_sidecar_dynamic_keyframe_rot6d_relative_stats(tmp_path):
    sidecar_path = tmp_path / "framewise_guidance.parquet"
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    pd.DataFrame(
        [
            {
                "index": 0,
                "episode_index": 0,
                OBS_STATE: np.array([1.0, 2.0, 3.0, *identity_rot6d, 0.4], dtype=np.float32),
                ACTION: np.array([2.0, 4.0, 6.0, *identity_rot6d, 0.7], dtype=np.float32),
                "action_valid_mask": np.ones(10, dtype=np.float32),
            },
            {
                "index": 1,
                "episode_index": 0,
                OBS_STATE: np.array([2.0, 3.0, 4.0, *identity_rot6d, 0.5], dtype=np.float32),
                ACTION: np.array([4.0, 7.0, 10.0, *identity_rot6d, 0.8], dtype=np.float32),
                "action_valid_mask": np.ones(10, dtype=np.float32),
            },
        ]
    ).to_parquet(sidecar_path, index=False)

    stats = FramewiseSidecar(sidecar_path).relative_action_chunk_stats(
        chunk_size=1,
        action_gap_frames=0,
        action_names=[
            "left_x",
            "left_y",
            "left_z",
            "left_rot6d_0",
            "left_rot6d_1",
            "left_rot6d_2",
            "left_rot6d_3",
            "left_rot6d_4",
            "left_rot6d_5",
            "gripper_left",
        ],
        relative_exclude_joints=["gripper"],
        relative_action_keyframes=["0:3"],
    )

    assert stats is not None
    assert stats["min"][:3].tolist() == [1.0, 2.0, 3.0]
    assert stats["max"][:3].tolist() == [2.0, 4.0, 6.0]
    assert stats["min"][3:9].tolist() == identity_rot6d.tolist()
    assert stats["max"][3:9].tolist() == identity_rot6d.tolist()
    assert stats["min"][9] == 0.7
    assert stats["max"][9] == 0.8


def test_framewise_sidecar_masked_vector_stats_uses_mask(tmp_path):
    sidecar_path = tmp_path / "framewise_guidance.parquet"
    pd.DataFrame(
        [
            {
                "index": 0,
                OBS_STATE: np.array([1.0, 1000.0], dtype=np.float32),
                "state_mask": np.array([1.0, 0.0], dtype=np.float32),
            },
            {
                "index": 1,
                OBS_STATE: np.array([3.0, 5.0], dtype=np.float32),
                "state_mask": np.array([1.0, 1.0], dtype=np.float32),
            },
        ]
    ).to_parquet(sidecar_path, index=False)

    stats = FramewiseSidecar(sidecar_path).masked_vector_stats(
        OBS_STATE,
        "state_mask",
        ["x", "y"],
    )

    assert stats is not None
    assert stats["mean"].tolist() == [2.0, 5.0]
    assert stats["min"].tolist() == [1.0, 5.0]
    assert stats["max"].tolist() == [3.0, 5.0]


def test_robot_type_gap_inference():
    assert infer_gap_override_from_robot_type("spark3_pose_gripper") == 0
    assert infer_gap_override_from_robot_type("ugripper_left") is None
    assert infer_gap_override_from_robot_type("ugripper_right") is None
    assert infer_gap_override_from_robot_type("dual_ugripper") is None
    assert infer_gap_override_from_robot_type("unknown") is None


def test_sidecar_stats_json_casts_arrays(tmp_path):
    from lerobot.datasets.sidecar_utils import load_stats_json

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps({ACTION: {"mean": [1.0, 2.0], "std": [3.0, 4.0], "count": [2]}}),
        encoding="utf-8",
    )

    stats = load_stats_json(stats_path)

    assert isinstance(stats[ACTION]["mean"], np.ndarray)
    assert stats[ACTION]["mean"].tolist() == [1.0, 2.0]
