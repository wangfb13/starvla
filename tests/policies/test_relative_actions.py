"""Tests for relative action transforms — full pipeline validation.

Tests the complete flow matching OpenPI:
  raw actions → RelativeActions → Normalize(relative_stats) → model → Unnormalize → AbsoluteActions

Uses real dataset: lerobot-data-collection/dagger_final_1_21
"""

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import TransitionKey, batch_to_transition
from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_keyframe_absolute_actions,
    to_keyframe_relative_actions,
    to_absolute_actions,
    to_relative_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE

CHUNK_SIZE = 10
REPO_ID = "lerobot-data-collection/dagger_final_1_21"


@pytest.fixture(scope="module")
def dataset():
    return LeRobotDataset(REPO_ID, episodes=[0])


@pytest.fixture(scope="module")
def action_dim(dataset):
    return dataset.meta.features["action"]["shape"][0]


def _build_action_chunks(dataset, chunk_size, max_chunks=50):
    """Build action chunks from hf_dataset, like the training script does."""
    hf = dataset.hf_dataset
    total = len(hf)
    all_ep = torch.tensor([int(hf[i]["episode_index"]) for i in range(total)])
    chunks, states = [], []
    for i in range(total - chunk_size + 1):
        if all_ep[i] != all_ep[i + chunk_size - 1]:
            continue
        chunk_actions = torch.stack([hf[i + k]["action"] for k in range(chunk_size)]).float()
        state = hf[i]["observation.state"].float()
        chunks.append(chunk_actions)
        states.append(state)
        if len(chunks) >= max_chunks:
            break
    assert len(chunks) > 0, f"No valid chunks found. total={total}, ep_indices={all_ep.tolist()}"
    return torch.stack(chunks), torch.stack(states)


def _compute_relative_chunk_stats(action_chunks, states, mask):
    all_chunks = []
    for actions, state in zip(action_chunks, states, strict=True):
        relative = to_relative_actions(actions.unsqueeze(0), state.unsqueeze(0), mask).squeeze(0)
        all_chunks.append(relative.numpy())
    all_relative = np.concatenate(all_chunks, axis=0)
    return get_feature_stats(all_relative, axis=0, keepdims=all_relative.ndim == 1)


# Basic roundtrip tests


def test_roundtrip_3d(action_dim):
    actions = torch.randn(4, CHUNK_SIZE, action_dim)
    state = torch.randn(4, action_dim)
    mask = [True] * action_dim
    recovered = to_absolute_actions(to_relative_actions(actions, state, mask), state, mask)
    torch.testing.assert_close(recovered, actions)


def test_roundtrip_2d(action_dim):
    actions = torch.randn(4, action_dim)
    state = torch.randn(4, action_dim)
    mask = [True] * action_dim
    recovered = to_absolute_actions(to_relative_actions(actions, state, mask), state, mask)
    torch.testing.assert_close(recovered, actions)


def test_no_mutation(action_dim):
    actions = torch.randn(2, CHUNK_SIZE, action_dim)
    original = actions.clone()
    state = torch.randn(2, action_dim)
    to_relative_actions(actions, state, [True] * action_dim)
    torch.testing.assert_close(actions, original)


def test_exclude_joints_supports_partial_name_matching():
    names = [
        "right_joint_1.pos",
        "right_gripper.pos",
        "left_joint_1.pos",
        "left_gripper.pos",
    ]
    step = RelativeActionsProcessorStep(enabled=True, exclude_joints=["gripper"], action_names=names)
    assert step._build_mask(len(names)) == [True, False, True, False]


def test_keyframe_rot6d_relative_roundtrip():
    state = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.25,
                -1.0,
                -2.0,
                -3.0,
                0.0,
                1.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.75,
            ]
        ],
        dtype=torch.float32,
    )
    actions = state[:, None, :].repeat(1, 2, 1)
    actions[:, 0, 0:3] += torch.tensor([0.2, -0.1, 0.3])
    actions[:, 1, 10:13] += torch.tensor([-0.4, 0.1, 0.2])

    relative = to_keyframe_relative_actions(actions, state, ["0:3", "10:13"])
    recovered = to_keyframe_absolute_actions(relative, state, ["0:3", "10:13"])

    torch.testing.assert_close(recovered, actions, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(relative[..., 9], actions[..., 9])
    torch.testing.assert_close(relative[..., 19], actions[..., 19])


def test_processor_keyframes_do_not_apply_scalar_relative_to_rot6d_twice():
    names = [
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
    ]
    state = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4]])
    action = state[:, None, :].clone()
    action[..., 0] += 0.5
    transition = batch_to_transition({ACTION: action, OBS_STATE: state})

    relative_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=["gripper"],
        action_names=names,
        relative_action_keyframes=["0:3"],
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    relative_transition = relative_step(transition)
    relative_action = relative_transition[TransitionKey.ACTION]
    torch.testing.assert_close(relative_action[..., 0], torch.tensor([[0.5]]))
    torch.testing.assert_close(relative_action[..., 3:9], state[:, None, 3:9])
    torch.testing.assert_close(relative_action[..., 9], action[..., 9])

    recovered = absolute_step(relative_transition)[TransitionKey.ACTION]
    torch.testing.assert_close(recovered, action, atol=1e-5, rtol=1e-5)


def test_online_episode_anchor_state_keeps_raw_state_for_action_decode():
    identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z90_rot6d = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    anchor = torch.zeros(1, 22)
    anchor[:, 0:3] = torch.tensor([1.0, 2.0, 3.0])
    anchor[:, 3:9] = z90_rot6d
    anchor[:, 9] = 0.4
    anchor[:, 10:13] = torch.tensor([-1.0, -2.0, 0.5])
    anchor[:, 13:19] = identity_rot6d
    anchor[:, 19] = 0.6
    anchor[:, 20:22] = 1.0

    step = RelativeActionsProcessorStep(
        enabled=True,
        relative_action_keyframes=["0:3", "10:13"],
        online_anchor_state_to_episode_start=True,
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)

    first = step(batch_to_transition({OBS_STATE: anchor}))
    first_model_state = first[TransitionKey.OBSERVATION][OBS_STATE]
    torch.testing.assert_close(first_model_state[:, 0:3], torch.zeros(1, 3))
    torch.testing.assert_close(first_model_state[:, 3:9], identity_rot6d.unsqueeze(0))
    torch.testing.assert_close(first_model_state[:, 10:13], torch.zeros(1, 3))
    torch.testing.assert_close(first_model_state[:, 13:19], identity_rot6d.unsqueeze(0))
    torch.testing.assert_close(first_model_state[:, [9, 19, 20, 21]], anchor[:, [9, 19, 20, 21]])
    torch.testing.assert_close(step._last_state, anchor)

    current = anchor.clone()
    current[:, 0:3] += torch.tensor([0.0, 1.0, 0.0])
    current[:, 10:13] += torch.tensor([0.0, 0.2, 0.0])
    second = step(batch_to_transition({OBS_STATE: current}))
    second_model_state = second[TransitionKey.OBSERVATION][OBS_STATE]
    # The left TCP anchor is rotated +90 degrees around z, so +world-y is +local-x.
    torch.testing.assert_close(second_model_state[:, 0:3], torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(second_model_state[:, 10:13], torch.tensor([[0.0, 0.2, 0.0]]))
    torch.testing.assert_close(step._last_state, current)

    predicted_relative = torch.zeros(1, 1, 20)
    predicted_relative[..., 0] = 0.1
    predicted_relative[..., 3:9] = identity_rot6d
    predicted_relative[..., 13:19] = identity_rot6d
    decoded = absolute_step(batch_to_transition({ACTION: predicted_relative}))[TransitionKey.ACTION]
    # Decode from the current raw pose: left local +x becomes +world-y under z90.
    expected_left_position = current[:, 0:3] + torch.tensor([[0.0, 0.1, 0.0]])
    torch.testing.assert_close(decoded[:, 0, 0:3], expected_left_position, atol=1e-6, rtol=1e-6)

    step.reset()
    reanchored = step(batch_to_transition({OBS_STATE: current}))[TransitionKey.OBSERVATION][OBS_STATE]
    torch.testing.assert_close(reanchored[:, 0:3], torch.zeros(1, 3))
    torch.testing.assert_close(reanchored[:, 3:9], identity_rot6d.unsqueeze(0))


def test_online_episode_anchor_state_is_not_applied_to_training_transition():
    state = torch.zeros(1, 20)
    state[:, 0:3] = torch.tensor([1.0, 2.0, 3.0])
    state[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    state[:, 10:13] = torch.tensor([-1.0, -2.0, -3.0])
    state[:, 13:19] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action = state[:, None, :].clone()
    transition = batch_to_transition({ACTION: action, OBS_STATE: state})
    step = RelativeActionsProcessorStep(
        enabled=True,
        relative_action_keyframes=["0:3", "10:13"],
        online_anchor_state_to_episode_start=True,
    )

    result = step(transition)

    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], state)
    assert step._episode_anchor_state is None


def test_online_gripper_calibration_roundtrip_and_training_bypass():
    names = [
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
        "right_x",
        "right_y",
        "right_z",
        "right_rot6d_0",
        "right_rot6d_1",
        "right_rot6d_2",
        "right_rot6d_3",
        "right_rot6d_4",
        "right_rot6d_5",
        "gripper_right",
    ]
    scales = {"gripper_left": 0.803053081035614, "gripper_right": 0.812948226928711}
    robot_state = torch.zeros(1, 20)
    robot_state[:, [9, 19]] = 1.0
    step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=["gripper"],
        action_names=names,
        online_gripper_umi_open_values=scales,
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)

    online = step(batch_to_transition({OBS_STATE: robot_state}))
    model_state = online[TransitionKey.OBSERVATION][OBS_STATE]
    torch.testing.assert_close(model_state[:, 9], torch.tensor([scales["gripper_left"]]))
    torch.testing.assert_close(model_state[:, 19], torch.tensor([scales["gripper_right"]]))
    # TCP decoding must continue to use the uncalibrated raw robot state.
    torch.testing.assert_close(step._last_state, robot_state)

    umi_action = torch.zeros(1, 1, 20)
    umi_action[..., 9] = 0.765071
    umi_action[..., 19] = 0.750790
    robot_action = absolute_step(batch_to_transition({ACTION: umi_action}))[TransitionKey.ACTION]
    torch.testing.assert_close(robot_action[..., 9], torch.tensor([[0.765071 / scales["gripper_left"]]]))
    torch.testing.assert_close(robot_action[..., 19], torch.tensor([[0.750790 / scales["gripper_right"]]]))

    # Training transitions already carry UMI values and must not be calibrated again.
    training_action = torch.zeros(1, 1, 20)
    training = step(batch_to_transition({OBS_STATE: model_state, ACTION: training_action}))
    torch.testing.assert_close(training[TransitionKey.OBSERVATION][OBS_STATE], model_state)


def test_online_gripper_open_latch_releases_per_arm_and_resets():
    names = [f"feature_{idx}" for idx in range(20)]
    names[9] = "gripper_left"
    names[19] = "gripper_right"
    scales = {"gripper_left": 0.803053081035614, "gripper_right": 0.812948226928711}
    robot_state = torch.zeros(1, 20)
    robot_state[:, [9, 19]] = 1.0
    step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=["gripper"],
        action_names=names,
        online_gripper_umi_open_values=scales,
        online_gripper_open_latch=True,
        online_gripper_close_thresholds={"gripper_left": 0.76, "gripper_right": 0.77},
        online_gripper_latch_prefix_steps=4,
        online_gripper_latch_consecutive_chunks=2,
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)

    def decode(left: float, right: float) -> torch.Tensor:
        step(batch_to_transition({OBS_STATE: robot_state}))
        predicted = torch.zeros(1, 8, 20)
        predicted[..., 9] = left
        predicted[..., 19] = right
        return absolute_step(batch_to_transition({ACTION: predicted}))[TransitionKey.ACTION]

    first = decode(0.79, 0.78)
    torch.testing.assert_close(first[..., 9], torch.ones(1, 8))
    torch.testing.assert_close(first[..., 19], torch.ones(1, 8))

    first_close_signal = decode(0.78, 0.75)
    torch.testing.assert_close(first_close_signal[..., 9], torch.ones(1, 8))
    torch.testing.assert_close(first_close_signal[..., 19], torch.ones(1, 8))

    released_right = decode(0.78, 0.75)
    torch.testing.assert_close(released_right[..., 9], torch.ones(1, 8))
    torch.testing.assert_close(
        released_right[..., 19], torch.full((1, 8), 0.75 / scales["gripper_right"])
    )

    # Release is permanent within the episode; a later open-looking prediction
    # must not re-latch the right gripper.
    remains_released = decode(0.78, 0.80)
    torch.testing.assert_close(
        remains_released[..., 19], torch.full((1, 8), 0.80 / scales["gripper_right"])
    )

    step.reset()
    relatched = decode(0.78, 0.75)
    torch.testing.assert_close(relatched[..., 9], torch.ones(1, 8))
    torch.testing.assert_close(relatched[..., 19], torch.ones(1, 8))


# Chunk-level relative stats test


def test_chunk_stats_have_larger_std_than_frame_stats(dataset, action_dim):
    """Chunk-level relative stats should have larger std than per-frame relative stats."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    chunk_stats = _compute_relative_chunk_stats(action_chunks, states, mask)

    # Per-frame stats
    hf = dataset.hf_dataset
    n = min(500, len(hf))
    frame_actions = torch.stack([hf[i]["action"] for i in range(n)]).float()
    frame_states = torch.stack([hf[i]["observation.state"] for i in range(n)]).float()
    frame_relatives = to_relative_actions(frame_actions, frame_states, mask).numpy()
    frame_stats = get_feature_stats(frame_relatives, axis=0, keepdims=frame_relatives.ndim == 1)

    assert chunk_stats["std"].mean() >= frame_stats["std"].mean(), (
        f"Chunk std ({chunk_stats['std'].mean():.4f}) should be >= "
        f"frame std ({frame_stats['std'].mean():.4f})"
    )


# Full pipeline roundtrip: relative → normalize → unnormalize → absolute


def test_full_pipeline_roundtrip(dataset, action_dim):
    """Test the complete OpenPI pipeline: relative → normalize → unnormalize → absolute."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)
    stats = {ACTION: dict(relative_stats.items())}

    features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}

    relative_step = RelativeActionsProcessorStep(enabled=True)
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    original_actions = action_chunks[0].unsqueeze(0)
    state = states[0].unsqueeze(0)

    batch = {ACTION: original_actions, OBS_STATE: state}
    transition = batch_to_transition(batch)

    # Forward: relative → normalize
    t1 = relative_step(transition)
    t2 = normalizer(t1)

    normalized_action = t2[TransitionKey.ACTION]
    assert normalized_action.abs().mean() < 10, (
        f"Normalized actions should be in reasonable range, got mean abs {normalized_action.abs().mean():.2f}"
    )

    # Reverse: unnormalize → absolute
    t3 = unnormalizer(t2)
    t4 = absolute_step(t3)

    recovered_actions = t4[TransitionKey.ACTION]
    torch.testing.assert_close(recovered_actions, original_actions, atol=1e-4, rtol=1e-4)


def test_normalized_relative_values_are_reasonable(dataset, action_dim):
    """With correct chunk stats, normalized relative actions should be in a reasonable range."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)
    mean = torch.tensor(relative_stats["mean"]).float()
    std = torch.tensor(relative_stats["std"]).float()

    all_normalized = []
    for actions, state in zip(action_chunks, states, strict=True):
        relative = to_relative_actions(actions.unsqueeze(0), state.unsqueeze(0), mask).squeeze(0)
        normalized = (relative - mean) / (std + 1e-6)
        all_normalized.append(normalized)

    all_normalized = torch.cat(all_normalized, dim=0)

    pct_in_range = (all_normalized.abs() < 5).float().mean()
    assert pct_in_range > 0.9, (
        f"Only {pct_in_range * 100:.1f}% of normalized values in [-5, 5], expected >90%"
    )

    assert all_normalized.mean().abs() < 1.0, (
        f"Mean of normalized relative actions is {all_normalized.mean():.2f}, expected near 0"
    )


def test_processor_step_roundtrip(dataset, action_dim):
    """RelativeActionsProcessorStep applies relative offsets; to_absolute_actions recovers original."""
    hf = dataset.hf_dataset
    batch = {
        ACTION: torch.stack([hf[i]["action"] for i in range(4)]),
        OBS_STATE: torch.stack([hf[i]["observation.state"] for i in range(4)]),
    }
    original_actions = batch[ACTION].clone()
    transition = batch_to_transition(batch)

    step = RelativeActionsProcessorStep(enabled=True)
    relative_transition = step(transition)
    assert not torch.allclose(relative_transition[TransitionKey.ACTION], original_actions)

    state = transition[TransitionKey.OBSERVATION][OBS_STATE]
    mask = [True] * action_dim
    recovered = to_absolute_actions(relative_transition[TransitionKey.ACTION], state, mask)
    torch.testing.assert_close(recovered, original_actions)


def test_processor_step_disabled_is_noop(dataset, action_dim):
    """enabled=False should be a no-op."""
    hf = dataset.hf_dataset
    batch = {
        ACTION: torch.stack([hf[i]["action"] for i in range(2)]),
        OBS_STATE: torch.stack([hf[i]["observation.state"] for i in range(2)]),
    }
    original = batch[ACTION].clone()
    transition = batch_to_transition(batch)
    result = RelativeActionsProcessorStep(enabled=False)(transition)
    torch.testing.assert_close(result[TransitionKey.ACTION], original)


# Training batch shape validation


def test_relative_with_action_chunks(dataset, action_dim):
    """Verify relative actions work correctly with (B, chunk_size, action_dim) shaped actions."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)

    # Simulate a training batch: actions=(B, chunk_size, action_dim), state=(B, state_dim)
    batch_actions = action_chunks[:4]  # (4, chunk_size, action_dim)
    batch_states = states[:4]  # (4, state_dim)

    mask = [True] * action_dim
    relative = to_relative_actions(batch_actions, batch_states, mask)

    # First action in each chunk should be close to zero (action[t] - state[t] ≈ small)
    first_relatives = relative[:, 0, :]  # (B, action_dim)
    assert first_relatives.abs().mean() < relative.abs().mean(), (
        f"First action in chunk should have smaller relative offset than average. "
        f"First: {first_relatives.abs().mean():.4f}, Average: {relative.abs().mean():.4f}"
    )

    # Later actions should have larger relative offsets
    last_relatives = relative[:, -1, :]  # (B, action_dim)
    assert last_relatives.abs().mean() >= first_relatives.abs().mean(), (
        f"Last action in chunk should have >= relative offset than first. "
        f"Last: {last_relatives.abs().mean():.4f}, First: {first_relatives.abs().mean():.4f}"
    )

    # Roundtrip
    recovered = to_absolute_actions(relative, batch_states, mask)
    torch.testing.assert_close(recovered, batch_actions)


def test_relative_stats_match_actual_data_distribution(dataset, action_dim):
    """Verify computed stats match the actual relative-action distribution."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    # Compute stats like the training script does
    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)

    # Also compute directly
    all_relatives = []
    for actions, state in zip(action_chunks, states, strict=True):
        rel = to_relative_actions(actions.unsqueeze(0), state.unsqueeze(0), mask).squeeze(0)
        all_relatives.append(rel)
    all_relatives_tensor = torch.cat(all_relatives, dim=0)

    # Compare mean
    actual_mean = all_relatives_tensor.mean(dim=0).numpy()
    np.testing.assert_allclose(relative_stats["mean"], actual_mean, atol=0.01)

    # Compare std
    actual_std = all_relatives_tensor.std(dim=0).numpy()
    np.testing.assert_allclose(relative_stats["std"], actual_std, atol=0.1)

    # Verify q01 < mean < q99
    assert (relative_stats["q01"] < relative_stats["mean"]).all(), "q01 should be < mean"
    assert (relative_stats["mean"] < relative_stats["q99"]).all(), "mean should be < q99"


def test_quantile_normalization_roundtrip(dataset, action_dim):
    """Full roundtrip with QUANTILES normalization (what OpenPI uses for pi05)."""
    action_chunks, states = _build_action_chunks(dataset, CHUNK_SIZE)
    mask = [True] * action_dim

    relative_stats = _compute_relative_chunk_stats(action_chunks, states, mask)
    stats = {ACTION: dict(relative_stats.items())}

    features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.QUANTILES}

    relative_step = RelativeActionsProcessorStep(enabled=True)
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    original_actions = action_chunks[0].unsqueeze(0)
    state = states[0].unsqueeze(0)

    batch = {ACTION: original_actions, OBS_STATE: state}
    transition = batch_to_transition(batch)

    # Forward: relative → quantile normalize
    t1 = relative_step(transition)
    t2 = normalizer(t1)

    normalized = t2[TransitionKey.ACTION]
    # Most values should be in [-1, 1] with quantile normalization
    pct_in_range = (normalized.abs() < 2).float().mean()
    assert pct_in_range > 0.5, f"Only {pct_in_range * 100:.1f}% in [-2, 2] after quantile norm, expected >50%"

    # Reverse: unnormalize → absolute
    t3 = unnormalizer(t2)
    t4 = absolute_step(t3)

    recovered = t4[TransitionKey.ACTION]
    torch.testing.assert_close(recovered, original_actions, atol=1e-3, rtol=1e-3)


def test_state_not_modified_by_relative_processor(dataset, action_dim):
    """State should never be modified by the relative-action processor."""
    hf = dataset.hf_dataset
    batch = {
        ACTION: torch.stack([hf[i]["action"] for i in range(4)]),
        OBS_STATE: torch.stack([hf[i]["observation.state"] for i in range(4)]),
    }
    original_state = batch[OBS_STATE].clone()
    transition = batch_to_transition(batch)

    step = RelativeActionsProcessorStep(enabled=True)
    result = step(transition)

    result_state = result[TransitionKey.OBSERVATION][OBS_STATE]
    torch.testing.assert_close(result_state, original_state)
