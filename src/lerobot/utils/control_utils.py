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

########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.async_inference.helpers import (
    augment_observation_for_online_pi05_full_sidecar,
    should_use_online_pi05_full_sidecar_adapter,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pi05.canonical_bimanual import (
    CANONICAL_ACTION_DIM,
    canonical_bimanual_action_to_pose_gripper_dict,
    extract_arm_validity_from_action_mask,
    extract_arm_validity_from_canonical_state,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.types import PolicyAction
from lerobot.utils.constants import OBS_ACTION_MASK, OBS_STATE, OBS_STATE_MASK


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def _get_cached_preprocessor_state(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> torch.Tensor | None:
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    for step in preprocessor.steps:
        if isinstance(step, RelativeActionsProcessorStep):
            return step._last_state
    return None


def _build_spark3_default_action_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    from lerobot.robots.spark3_pose_gripper.schema import action_dict_from_observation

    return action_dict_from_observation(observation)


def _format_arm_pose(action: dict[str, float], arm: str) -> str:
    return "(%.4f, %.4f, %.4f)" % (
        action.get(f"main_{arm}_x", float("nan")),
        action.get(f"main_{arm}_y", float("nan")),
        action.get(f"main_{arm}_z", float("nan")),
    )


def _ensure_online_sidecar_batch_dims(
    observation: dict[str, Any],
    policy: PreTrainedPolicy,
) -> None:
    tensor_keys = {
        OBS_STATE,
        OBS_ACTION_MASK,
        getattr(policy.config, "action_mask_observation_key", OBS_ACTION_MASK),
        getattr(policy.config, "state_mask_observation_key", OBS_STATE_MASK),
        getattr(policy.config, "tactile_mask_key", "observation.tactile_present"),
        getattr(policy.config, "tactile_slot_present_key", "observation.tactile_slot_present"),
    }
    for key in tensor_keys:
        value = observation.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim <= 1:
            observation[key] = value.unsqueeze(0)


def _maybe_convert_pi05_canonical_action_to_robot_action(
    *,
    action: PolicyAction,
    raw_observation: dict[str, Any],
    prepared_observation: dict[str, Any] | None,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    robot_type: str | None,
):
    if robot_type != "spark3_pose_gripper":
        return action

    use_builtin_canonical = bool(getattr(policy.config, "canonicalize_bimanual_rot6d", False))
    use_online_sidecar = should_use_online_pi05_full_sidecar_adapter(
        policy_type=policy.name,
        policy_config=policy.config,
        robot_type=robot_type,
    )
    if not use_builtin_canonical and not use_online_sidecar:
        return action

    if not isinstance(action, torch.Tensor):
        return action
    if action.shape[-1] != CANONICAL_ACTION_DIM:
        return action

    action_tensor = action.squeeze(0) if action.ndim == 2 and action.shape[0] == 1 else action
    state = _get_cached_preprocessor_state(preprocessor)
    arm_valid = None
    if state is not None:
        try:
            arm_valid = extract_arm_validity_from_canonical_state(state)
        except ValueError:
            arm_valid = None
    if arm_valid is None and prepared_observation is not None:
        prepared_state = prepared_observation.get(OBS_STATE)
        if isinstance(prepared_state, torch.Tensor):
            prepared_state = prepared_state.squeeze(0) if prepared_state.ndim == 2 else prepared_state
            try:
                arm_valid = extract_arm_validity_from_canonical_state(prepared_state)
            except ValueError:
                arm_valid = None
    if arm_valid is None and prepared_observation is not None:
        action_mask_key = getattr(policy.config, "action_mask_observation_key", OBS_ACTION_MASK)
        prepared_action_mask = prepared_observation.get(action_mask_key, prepared_observation.get(OBS_ACTION_MASK))
        if isinstance(prepared_action_mask, torch.Tensor):
            prepared_action_mask = (
                prepared_action_mask.squeeze(0) if prepared_action_mask.ndim == 2 else prepared_action_mask
            )
            try:
                arm_valid = extract_arm_validity_from_action_mask(prepared_action_mask)
            except ValueError:
                arm_valid = None

    converted_action = canonical_bimanual_action_to_pose_gripper_dict(
        action_tensor,
        arm_valid=arm_valid,
    )
    robot_action = _build_spark3_default_action_from_observation(raw_observation)
    robot_action.update(converted_action)
    print(
        "[pi05-action-debug] arm_valid=%s updated_keys=%s left=%s right=%s gripper=(%.4f, %.4f)"
        % (
            arm_valid,
            sorted(converted_action.keys()),
            _format_arm_pose(converted_action, "left"),
            _format_arm_pose(converted_action, "right"),
            robot_action.get("main_gripper_left", float("nan")),
            robot_action.get("main_gripper_right", float("nan")),
        ),
        flush=True,
    )
    return robot_action


def _should_decode_action_chunks_with_shared_reference(
    policy: PreTrainedPolicy,
    action_override: Callable[[dict[str, Any]], PolicyAction] | None,
) -> bool:
    return (
        action_override is None
        and bool(getattr(policy.config, "use_relative_actions", False))
        and int(getattr(policy.config, "n_action_steps", 1)) > 1
    )


def _is_chunk_action_tensor(action: PolicyAction) -> bool:
    return isinstance(action, torch.Tensor) and action.ndim == 3


def _get_processed_action_queue(policy: PreTrainedPolicy) -> deque[PolicyAction]:
    queue = getattr(policy, "_lerobot_processed_action_queue", None)
    if queue is None:
        queue = deque(maxlen=max(1, int(getattr(policy.config, "n_action_steps", 1))))
        setattr(policy, "_lerobot_processed_action_queue", queue)
    return queue


def _get_trace_raw_action_queue(policy: PreTrainedPolicy) -> deque[torch.Tensor]:
    queue = getattr(policy, "_lerobot_trace_raw_action_queue", None)
    if queue is None:
        queue = deque(maxlen=max(1, int(getattr(policy.config, "n_action_steps", 1))))
        setattr(policy, "_lerobot_trace_raw_action_queue", queue)
    return queue


def _safe_queue_size(queue: Any) -> int | None:
    if queue is None:
        return None
    try:
        return len(queue)
    except TypeError:
        return None


def _get_policy_queue_size(policy: PreTrainedPolicy) -> int | None:
    processed_size = _safe_queue_size(getattr(policy, "_lerobot_processed_action_queue", None))
    action_size = _safe_queue_size(getattr(policy, "_action_queue", None))
    sizes = [size for size in (processed_size, action_size) if size is not None]
    if not sizes:
        return None
    return max(sizes)


def _try_dequeue_cached_policy_action(
    policy: PreTrainedPolicy,
    *,
    return_trace: dict[str, Any] | None = None,
) -> tuple[PolicyAction | None, bool]:
    processed_queue = getattr(policy, "_lerobot_processed_action_queue", None)
    processed_size = _safe_queue_size(processed_queue)
    if processed_size and processed_size > 0:
        raw_queue = _get_trace_raw_action_queue(policy)
        action = processed_queue.popleft()
        raw_action = raw_queue.popleft() if raw_queue else None
        if return_trace is not None:
            return_trace["fast_path_cache_hit"] = True
            return_trace["cache_queue_type"] = "processed_action_queue"
            if raw_action is not None:
                return_trace["raw_policy_action_tensor"] = raw_action
        return action, True

    action_queue = getattr(policy, "_action_queue", None)
    action_size = _safe_queue_size(action_queue)
    if action_size and action_size > 0:
        action = action_queue.popleft()
        if return_trace is not None:
            return_trace["fast_path_cache_hit"] = True
            return_trace["cache_queue_type"] = "raw_action_queue"
        return action, False

    if return_trace is not None:
        return_trace["fast_path_cache_hit"] = False
    return None, False


def _split_postprocessed_chunk(processed_chunk: PolicyAction) -> list[PolicyAction]:
    if not isinstance(processed_chunk, torch.Tensor):
        raise TypeError(
            "Chunked relative-action decoding currently expects the postprocessor to return a tensor, "
            f"got {type(processed_chunk)!r}."
        )

    if processed_chunk.ndim == 2:
        return [processed_chunk]
    if processed_chunk.ndim != 3:
        raise ValueError(
            "Chunked relative-action decoding expects a tensor with shape (B, T, D) or (B, D), "
            f"got {tuple(processed_chunk.shape)}."
        )
    return [processed_chunk[:, step_idx, :] for step_idx in range(processed_chunk.shape[1])]


def _predict_action_with_shared_reference_chunk(
    *,
    observation: dict[str, Any],
    raw_observation: dict[str, Any],
    prepared_observation: dict[str, Any] | None,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    robot_type: str | None,
    return_trace: dict[str, Any] | None = None,
) -> PolicyAction:
    queue = _get_processed_action_queue(policy)
    raw_queue = _get_trace_raw_action_queue(policy)
    if queue:
        action = queue.popleft()
        raw_action = raw_queue.popleft() if raw_queue else None
        if return_trace is not None and raw_action is not None:
            return_trace["raw_policy_action_tensor"] = raw_action
        return action

    raw_chunk = policy.predict_action_chunk(observation)[:, : policy.config.n_action_steps]
    processed_chunk = postprocessor(raw_chunk)

    for step_action in _split_postprocessed_chunk(processed_chunk):
        raw_queue.append(step_action.detach().cpu().clone())
        queue.append(
            _maybe_convert_pi05_canonical_action_to_robot_action(
                action=step_action,
                raw_observation=raw_observation,
                prepared_observation=prepared_observation,
                policy=policy,
                preprocessor=preprocessor,
                robot_type=robot_type,
            )
        )

    if not queue:
        raise RuntimeError("Policy returned an empty action chunk after shared-reference decoding.")

    action = queue.popleft()
    raw_action = raw_queue.popleft() if raw_queue else None
    if return_trace is not None and raw_action is not None:
        return_trace["raw_policy_action_tensor"] = raw_action
    return action


def _decode_override_chunk_with_shared_reference(
    *,
    raw_chunk: PolicyAction,
    raw_observation: dict[str, Any],
    prepared_observation: dict[str, Any] | None,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    robot_type: str | None,
    return_trace: dict[str, Any] | None = None,
) -> PolicyAction:
    queue = _get_processed_action_queue(policy)
    raw_queue = _get_trace_raw_action_queue(policy)
    if queue:
        action = queue.popleft()
        raw_action = raw_queue.popleft() if raw_queue else None
        if return_trace is not None and raw_action is not None:
            return_trace["raw_policy_action_tensor"] = raw_action
        return action

    processed_chunk = postprocessor(raw_chunk)
    for step_action in _split_postprocessed_chunk(processed_chunk):
        raw_queue.append(step_action.detach().cpu().clone())
        queue.append(
            _maybe_convert_pi05_canonical_action_to_robot_action(
                action=step_action,
                raw_observation=raw_observation,
                prepared_observation=prepared_observation,
                policy=policy,
                preprocessor=preprocessor,
                robot_type=robot_type,
            )
        )

    if not queue:
        raise RuntimeError("Manual action override returned an empty action chunk after shared-reference decoding.")

    action = queue.popleft()
    raw_action = raw_queue.popleft() if raw_queue else None
    if return_trace is not None and raw_action is not None:
        return_trace["raw_policy_action_tensor"] = raw_action
    return action


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
    raw_observation: dict[str, Any] | None = None,
    action_override: Callable[[dict[str, Any]], PolicyAction] | None = None,
    return_trace: dict[str, Any] | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action, unless
       `action_override` is provided, in which case the override is used instead.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A postprocessed policy action. For canonical PI0.5 on `spark3_pose_gripper`,
        this is converted into a robot action dictionary directly.
    """
    observation = copy(observation)
    raw_observation = copy(raw_observation) if raw_observation is not None else copy(observation)
    prepared_observation = None
    action_is_final = False
    trace = return_trace if return_trace is not None else None
    if trace is not None:
        trace.clear()
        trace["policy_queue_size_before"] = _get_policy_queue_size(policy)
        trace["n_action_steps"] = int(getattr(policy.config, "n_action_steps", 1))

    queue_size_before_select = _get_policy_queue_size(policy)
    if trace is not None:
        trace["policy_queue_size_before_select"] = queue_size_before_select

    cached_action = None
    cache_dequeue_ms = 0.0
    if action_override is None:
        start_t = time.perf_counter()
        cached_action, action_is_final = _try_dequeue_cached_policy_action(policy, return_trace=trace)
        cache_dequeue_ms = (time.perf_counter() - start_t) * 1000.0
    elif trace is not None:
        trace["fast_path_cache_hit"] = False

    if cached_action is not None:
        if trace is not None:
            trace["cache_dequeue_ms"] = cache_dequeue_ms
            trace["prepare_observation_ms"] = 0.0
            trace["preprocessor_ms"] = 0.0
            trace["policy_select_ms"] = 0.0
            trace["chunk_queue_refresh"] = False
            trace["chunk_decode_path"] = trace.get("cache_queue_type")

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
        ):
            action = cached_action
            if not action_is_final:
                start_t = time.perf_counter()
                action = postprocessor(action)
                if trace is not None:
                    trace["postprocessor_ms"] = (time.perf_counter() - start_t) * 1000.0
                    if isinstance(action, torch.Tensor):
                        trace["raw_policy_action_tensor"] = action.detach().cpu().clone()

            if not action_is_final:
                start_t = time.perf_counter()
                action = _maybe_convert_pi05_canonical_action_to_robot_action(
                    action=action,
                    raw_observation=raw_observation,
                    prepared_observation=None,
                    policy=policy,
                    preprocessor=preprocessor,
                    robot_type=robot_type,
                )
                if trace is not None:
                    trace["action_convert_ms"] = (time.perf_counter() - start_t) * 1000.0

        if trace is not None:
            trace["policy_queue_size_after"] = _get_policy_queue_size(policy)
            trace["predict_action_total_ms"] = (
                trace.get("cache_dequeue_ms", 0.0)
                + trace.get("postprocessor_ms", 0.0)
                + trace.get("action_convert_ms", 0.0)
            )
        return action

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        start_t = time.perf_counter()
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        if should_use_online_pi05_full_sidecar_adapter(
            policy_type=policy.name,
            policy_config=policy.config,
            robot_type=robot_type,
        ):
            augment_observation_for_online_pi05_full_sidecar(
                observation=observation,
                raw_observation=raw_observation,
                policy_config=policy.config,
            )
            _ensure_online_sidecar_batch_dims(observation, policy)
        prepared_observation = observation
        if trace is not None:
            trace["prepare_observation_ms"] = (time.perf_counter() - start_t) * 1000.0
            if OBS_STATE in prepared_observation:
                trace["prepared_model_input_state"] = prepared_observation[OBS_STATE].detach().cpu()
            action_mask_key = getattr(policy.config, "action_mask_observation_key", OBS_ACTION_MASK)
            state_mask_key = getattr(policy.config, "state_mask_observation_key", OBS_STATE_MASK)
            if state_mask_key in prepared_observation:
                trace["prepared_model_input_state_mask"] = prepared_observation[state_mask_key].detach().cpu()
            if action_mask_key in prepared_observation:
                trace["prepared_model_input_action_mask"] = prepared_observation[action_mask_key].detach().cpu()

        start_t = time.perf_counter()
        observation = preprocessor(observation)
        if trace is not None:
            trace["preprocessor_ms"] = (time.perf_counter() - start_t) * 1000.0
            if OBS_STATE in observation:
                trace["preprocessed_model_input_state"] = observation[OBS_STATE].detach().cpu()

        # Compute the next action either from the policy or from a manual override
        # that mimics the raw model output format before post-processing.
        start_t = time.perf_counter()
        if action_override is not None:
            action = action_override(prepared_observation)
            if _is_chunk_action_tensor(action):
                action = _decode_override_chunk_with_shared_reference(
                    raw_chunk=action,
                    raw_observation=raw_observation,
                    prepared_observation=prepared_observation,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    robot_type=robot_type,
                    return_trace=trace,
                )
                action_is_final = True
                if trace is not None:
                    trace["chunk_queue_refresh"] = True
                    trace["chunk_decode_path"] = "manual_override_shared_reference"
        elif _should_decode_action_chunks_with_shared_reference(policy, action_override):
            if trace is not None:
                trace["chunk_queue_refresh"] = queue_size_before_select == 0
                trace["chunk_decode_path"] = "shared_reference"
            action = _predict_action_with_shared_reference_chunk(
                observation=observation,
                raw_observation=raw_observation,
                prepared_observation=prepared_observation,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot_type=robot_type,
                return_trace=trace,
            )
            action_is_final = True
        else:
            if trace is not None:
                trace["chunk_queue_refresh"] = queue_size_before_select == 0 if queue_size_before_select is not None else None
                trace["chunk_decode_path"] = "policy_select_action"
            action = policy.select_action(observation)
        if trace is not None:
            trace["policy_select_ms"] = (time.perf_counter() - start_t) * 1000.0

        if not action_is_final:
            start_t = time.perf_counter()
            action = postprocessor(action)
            if trace is not None:
                trace["postprocessor_ms"] = (time.perf_counter() - start_t) * 1000.0
                if isinstance(action, torch.Tensor):
                    trace["raw_policy_action_tensor"] = action.detach().cpu().clone()

    if not action_is_final:
        start_t = time.perf_counter()
        action = _maybe_convert_pi05_canonical_action_to_robot_action(
            action=action,
            raw_observation=raw_observation,
            prepared_observation=prepared_observation,
            policy=policy,
            preprocessor=preprocessor,
            robot_type=robot_type,
        )
        if trace is not None:
            trace["action_convert_ms"] = (time.perf_counter() - start_t) * 1000.0

    if trace is not None:
        trace["policy_queue_size_after"] = _get_policy_queue_size(policy)
        trace["predict_action_total_ms"] = (
            trace.get("prepare_observation_ms", 0.0)
            + trace.get("preprocessor_ms", 0.0)
            + trace.get("policy_select_ms", 0.0)
            + trace.get("postprocessor_ms", 0.0)
            + trace.get("action_convert_ms", 0.0)
        )

    return action


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
