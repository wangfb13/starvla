# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import logging.handlers
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import PolicyFeature
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features

# NOTE: Configs need to be loaded for the client to be able to instantiate the policy config
from lerobot.policies import (  # noqa: F401
    ACTConfig,
    DiffusionConfig,
    PI0Config,
    PI05Config,
    PI05FullConfig,
    SmolVLAConfig,
    VQBeTConfig,
)
from lerobot.policies.pi05_full.canonical_bimanual import (
    CANONICAL_ACTION_DIM,
    CANONICAL_STATE_DIM,
    canonical_bimanual_action_chunk_to_pose_gripper_dicts,
    extract_arm_validity_from_action_mask,
    extract_arm_validity_from_canonical_state,
)
from lerobot.robots.spark3_pose_gripper.schema import action_dict_from_observation
from lerobot.robots.robot import Robot
from lerobot.utils.constants import OBS_ACTION_MASK, OBS_IMAGES, OBS_STATE, OBS_STATE_MASK, OBS_STR
from lerobot.utils.utils import init_logging

Action = torch.Tensor | dict[str, float]

# observation as received from the robot (can be numpy arrays, floats, etc.)
RawObservation = dict[str, Any]

# observation as those recorded in LeRobot dataset (keys are different)
LeRobotObservation = dict[str, torch.Tensor]

# observation, ready for policy inference (image keys resized)
Observation = dict[str, torch.Tensor]

SPARK3_ARM_KEYS = {
    "left": (
        "main_left_x",
        "main_left_y",
        "main_left_z",
        "main_left_qx",
        "main_left_qy",
        "main_left_qz",
        "main_left_w",
    ),
    "right": (
        "main_right_x",
        "main_right_y",
        "main_right_z",
        "main_right_qx",
        "main_right_qy",
        "main_right_qz",
        "main_right_w",
    ),
}
SPARK3_GRIPPER_KEYS = {"left": "main_gripper_left", "right": "main_gripper_right"}
TACTILE_SLOT_SUFFIX_TO_INDEX = {
    "tactile_left_left": 0,
    "tactile_left_right": 1,
    "tactile_right_left": 2,
    "tactile_right_right": 3,
}
SPARKY_TACTILE_RAW_KEY_TO_CANONICAL_KEY = {
    "cam_left_gripper_left_tactile_depth_deformation": "observation.depth_deformation.tactile_left_left",
    "cam_left_gripper_right_tactile_depth_deformation": "observation.depth_deformation.tactile_left_right",
    "cam_right_gripper_left_tactile_depth_deformation": "observation.depth_deformation.tactile_right_left",
    "cam_right_gripper_right_tactile_depth_deformation": "observation.depth_deformation.tactile_right_right",
}


def visualize_action_queue_size(action_queue_size: list[int]) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    ax.set_ylim(0, max(action_queue_size) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.plot(range(len(action_queue_size)), action_queue_size)
    plt.show()


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def resize_robot_observation_image(image: torch.tensor, resize_dims: tuple[int, int, int]) -> torch.tensor:
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W) for resizing from robot obsevation resolution to policy image resolution
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


def _is_finite_scalar(value: Any) -> bool:
    try:
        return torch.isfinite(torch.tensor(float(value), dtype=torch.float32)).item()
    except (TypeError, ValueError):
        return False


def _quat_xyzw_to_rot6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    quat_xyzw = quat_xyzw.to(dtype=torch.float32)
    norm = torch.linalg.norm(quat_xyzw)
    if not torch.isfinite(norm) or float(norm) <= 1e-8:
        raise ValueError("Quaternion norm is too small to convert to rot6d.")
    x, y, z, w = quat_xyzw / norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    col0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)))
    col1 = torch.stack((2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)))
    return torch.cat((col0, col1), dim=0).to(dtype=torch.float32)


def _extract_optional_mask(raw_observation: RawObservation, keys: tuple[str, ...], dim: int) -> torch.Tensor | None:
    for key in keys:
        if key not in raw_observation:
            continue
        value = raw_observation[key]
        try:
            mask = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if mask.numel() >= dim:
            return mask[:dim].clone()
    return None


def _extract_optional_arm_present(raw_observation: RawObservation) -> dict[str, bool] | None:
    if "arm_present_mask" in raw_observation:
        try:
            mask = torch.as_tensor(raw_observation["arm_present_mask"], dtype=torch.float32).reshape(-1)
        except (TypeError, ValueError):
            mask = None
        if mask is not None and mask.numel() >= 2:
            return {"left": bool(float(mask[0]) > 0.5), "right": bool(float(mask[1]) > 0.5)}

    left = raw_observation.get("left_arm_present", raw_observation.get("left_present"))
    right = raw_observation.get("right_arm_present", raw_observation.get("right_present"))
    if left is None and right is None:
        return None
    return {
        "left": True if left is None else bool(float(left) > 0.5),
        "right": True if right is None else bool(float(right) > 0.5),
    }


def _resolve_tactile_runtime_keys(observation: Observation, policy_config: Any) -> list[str]:
    configured = list(getattr(policy_config, "tactile_feature_keys", []) or [])
    if configured:
        return [key for key in configured if key in observation]
    tactile_prefixes = (
        f"{OBS_IMAGES}.",
        "observation.depth_deformation.",
        "observation.depth_shear.",
    )
    return [key for key in observation if key.startswith(tactile_prefixes) and "tactile" in key]


def _normalize_runtime_depth_deformation_tactile(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected tactile depth/deformation tensor in HWC shape [H,W,3], got {tuple(tensor.shape)}")

    tensor = tensor.to(dtype=torch.float32)
    depth = torch.clamp(tensor[..., :1] / 3000.0, 0.0, 1.0)
    motion = torch.clamp((tensor[..., 1:3] - 25000.0) / 10000.0, 0.0, 1.0)
    return torch.cat([depth, motion], dim=-1).permute(2, 0, 1).contiguous()


def _extract_runtime_tactile_observation(
    raw_observation: RawObservation,
    policy_config: Any,
) -> dict[str, torch.Tensor]:
    tactile_observation: dict[str, torch.Tensor] = {}
    configured = list(getattr(policy_config, "tactile_feature_keys", []) or [])

    for raw_key, canonical_key in SPARKY_TACTILE_RAW_KEY_TO_CANONICAL_KEY.items():
        if raw_key not in raw_observation:
            continue
        if configured and canonical_key not in configured:
            continue
        tactile_observation[canonical_key] = _normalize_runtime_depth_deformation_tactile(raw_observation[raw_key])

    return tactile_observation


def _build_spark3_online_pi05_state_and_masks(
    raw_observation: RawObservation,
    *,
    action_mask_key: str,
    state_mask_key: str,
) -> dict[str, torch.Tensor]:
    state_segments: list[torch.Tensor] = []
    state_masks: list[torch.Tensor] = []
    action_masks: list[torch.Tensor] = []
    arm_present: dict[str, bool] = {}

    explicit_arm_present = _extract_optional_arm_present(raw_observation)
    explicit_state_mask = _extract_optional_mask(
        raw_observation,
        (
            state_mask_key,
            "state_mask",
            OBS_STATE_MASK,
        ),
        CANONICAL_ACTION_DIM,
    )
    explicit_action_mask = _extract_optional_mask(
        raw_observation,
        (
            action_mask_key,
            "action_valid_mask",
            OBS_ACTION_MASK,
        ),
        CANONICAL_ACTION_DIM,
    )

    for arm in ("left", "right"):
        pose_values = []
        pose_valid = True
        for key in SPARK3_ARM_KEYS[arm]:
            if key not in raw_observation or not _is_finite_scalar(raw_observation[key]):
                pose_valid = False
                pose_values.append(0.0)
            else:
                pose_values.append(float(raw_observation[key]))

        grip_key = SPARK3_GRIPPER_KEYS[arm]
        grip_valid = grip_key in raw_observation and _is_finite_scalar(raw_observation[grip_key])
        grip_value = float(raw_observation[grip_key]) if grip_valid else 0.0

        pose_tensor = torch.zeros(9, dtype=torch.float32)
        if pose_valid:
            pose_tensor[:3] = torch.tensor(pose_values[:3], dtype=torch.float32)
            pose_tensor[3:9] = _quat_xyzw_to_rot6d(torch.tensor(pose_values[3:7], dtype=torch.float32))
        state_arm = torch.cat((pose_tensor, torch.tensor([grip_value], dtype=torch.float32)), dim=0)
        state_mask_arm = torch.cat(
            (
                torch.ones(9, dtype=torch.float32) if pose_valid else torch.zeros(9, dtype=torch.float32),
                torch.tensor([1.0 if grip_valid else 0.0], dtype=torch.float32),
            ),
            dim=0,
        )
        action_mask_arm = state_mask_arm.clone()

        arm_is_present = pose_valid or grip_valid
        if explicit_arm_present is not None:
            arm_is_present = bool(explicit_arm_present[arm])
            if not arm_is_present:
                state_mask_arm.zero_()
                action_mask_arm.zero_()

        state_segments.append(state_arm)
        state_masks.append(state_mask_arm)
        action_masks.append(action_mask_arm)
        arm_present[arm] = arm_is_present

    canonical_state = torch.cat(
        (
            torch.cat(state_segments, dim=0),
            torch.tensor(
                [1.0 if arm_present["left"] else 0.0, 1.0 if arm_present["right"] else 0.0],
                dtype=torch.float32,
            ),
        ),
        dim=0,
    )
    state_mask = torch.cat(state_masks, dim=0)
    action_mask = torch.cat(action_masks, dim=0)

    if explicit_state_mask is not None:
        state_mask = explicit_state_mask.to(dtype=torch.float32)
        if explicit_arm_present is None:
            arm_present = {
                "left": bool(float(state_mask[:10].max()) > 0.5),
                "right": bool(float(state_mask[10:20].max()) > 0.5),
            }
            canonical_state[-2:] = torch.tensor(
                [1.0 if arm_present["left"] else 0.0, 1.0 if arm_present["right"] else 0.0],
                dtype=torch.float32,
            )

    if explicit_action_mask is not None:
        action_mask = explicit_action_mask.to(dtype=torch.float32)

    if explicit_arm_present is not None:
        for arm_idx, arm in enumerate(("left", "right")):
            if arm_present[arm]:
                continue
            start = arm_idx * 10
            state_mask[start : start + 10] = 0.0
            action_mask[start : start + 10] = 0.0

    return {
        OBS_STATE: canonical_state,
        state_mask_key: state_mask,
        action_mask_key: action_mask,
        OBS_STATE_MASK: state_mask,
        OBS_ACTION_MASK: action_mask,
    }


def _should_use_online_pi05_full_sidecar_adapter(
    *,
    policy_type: str | None,
    policy_config: Any | None,
    robot_type: str | None,
) -> bool:
    if policy_type != "pi05_full":
        return False
    if robot_type != "spark3_pose_gripper":
        return False
    if policy_config is None:
        return False
    if getattr(policy_config, "canonicalize_bimanual_rot6d", False):
        return False
    return (
        getattr(policy_config, "canonical_action_dim", CANONICAL_ACTION_DIM) == CANONICAL_ACTION_DIM
        and getattr(policy_config, "canonical_state_dim", CANONICAL_STATE_DIM) == CANONICAL_STATE_DIM
    )


def should_use_online_pi05_full_sidecar_adapter(
    *,
    policy_type: str | None,
    policy_config: Any | None,
    robot_type: str | None,
) -> bool:
    return _should_use_online_pi05_full_sidecar_adapter(
        policy_type=policy_type,
        policy_config=policy_config,
        robot_type=robot_type,
    )


def _augment_observation_for_online_pi05_full_sidecar(
    *,
    observation: Observation,
    raw_observation: RawObservation,
    policy_config: Any,
) -> None:
    action_mask_key = getattr(policy_config, "action_mask_observation_key", OBS_ACTION_MASK)
    state_mask_key = getattr(policy_config, "state_mask_observation_key", OBS_STATE_MASK)

    online_state = _build_spark3_online_pi05_state_and_masks(
        raw_observation,
        action_mask_key=action_mask_key,
        state_mask_key=state_mask_key,
    )
    observation.update(online_state)

    if not getattr(policy_config, "enable_tactile", False):
        return

    observation.update(_extract_runtime_tactile_observation(raw_observation, policy_config))

    tactile_mask_key = getattr(policy_config, "tactile_mask_key", "observation.tactile_present")
    if tactile_mask_key in raw_observation:
        observation[tactile_mask_key] = torch.as_tensor(raw_observation[tactile_mask_key], dtype=torch.float32)
    else:
        tactile_keys = _resolve_tactile_runtime_keys(observation, policy_config)
        observation[tactile_mask_key] = torch.tensor([1.0 if tactile_keys else 0.0], dtype=torch.float32)

    tactile_slot_present_key = getattr(policy_config, "tactile_slot_present_key", "observation.tactile_slot_present")
    if tactile_slot_present_key in raw_observation:
        observation[tactile_slot_present_key] = torch.as_tensor(
            raw_observation[tactile_slot_present_key], dtype=torch.float32
        )
        return

    tactile_keys = _resolve_tactile_runtime_keys(observation, policy_config)
    if not tactile_keys:
        return

    slot_present = torch.zeros(4, dtype=torch.float32)
    for key in tactile_keys:
        for suffix, slot_idx in TACTILE_SLOT_SUFFIX_TO_INDEX.items():
            if key.endswith(suffix):
                slot_present[slot_idx] = 1.0
                break
    if float(slot_present.sum()) > 0.0:
        observation[tactile_slot_present_key] = slot_present


def augment_observation_for_online_pi05_full_sidecar(
    *,
    observation: Observation,
    raw_observation: RawObservation,
    policy_config: Any,
) -> None:
    _augment_observation_for_online_pi05_full_sidecar(
        observation=observation,
        raw_observation=raw_observation,
        policy_config=policy_config,
    )


# TODO(Steven): Consider implementing a pipeline step for this
def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
    *,
    policy_type: str | None = None,
    policy_config: Any | None = None,
    robot_type: str | None = None,
) -> Observation:
    observation = {}

    observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    if _should_use_online_pi05_full_sidecar_adapter(
        policy_type=policy_type,
        policy_config=policy_config,
        robot_type=robot_type,
    ):
        _augment_observation_for_online_pi05_full_sidecar(
            observation=observation,
            raw_observation=raw_observation,
            policy_config=policy_config,
        )

    for k, v in observation.items():
        if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions in observations
            if "image" in k:
                # Policy expects images in shape (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0)
        else:
            observation[k] = v

    return observation


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def extract_state_from_raw_observation(
    lerobot_obs: RawObservation,
) -> torch.Tensor:
    """Extract the state from a raw observation."""
    state = torch.tensor(lerobot_obs[OBS_STATE])

    if state.ndim == 1:
        state = state.unsqueeze(0)

    return state


def extract_images_from_raw_observation(
    lerobot_obs: RawObservation,
    camera_key: str,
) -> dict[str, torch.Tensor]:
    """Extract the images from a raw observation."""
    return torch.tensor(lerobot_obs[camera_key])


def make_lerobot_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
) -> LeRobotObservation:
    """Make a lerobot observation from a raw observation."""
    return build_dataset_frame(lerobot_features, robot_obs, prefix=OBS_STR)


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    """Matches keys from the raw robot_obs dict to the keys expected by a given policy (passed as
    policy_image_features)."""
    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)

    # 2. Greps all observation.images.<> keys
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        image_k: extract_images_from_raw_observation(lerobot_obs, image_k) for image_k in image_keys
    }

    # Turns the image features to (C, H, W) with H, W matching the policy image features.
    # This reduces the resolution of the images
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
        if key in policy_image_features
    }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

    return {**state_dict, **image_dict}


def build_spark3_default_action_from_raw_observation(raw_observation: RawObservation) -> dict[str, float]:
    return action_dict_from_observation(raw_observation)


def maybe_convert_pi05_canonical_action_chunk_to_robot_actions(
    *,
    action_chunk: torch.Tensor,
    raw_observation: RawObservation,
    policy_type: str,
    policy_config: Any,
    robot_type: str | None,
    canonical_state: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
) -> list[dict[str, float]] | None:
    if policy_type not in {"pi05", "pi05_full"}:
        return None
    if robot_type != "spark3_pose_gripper":
        return None
    use_builtin_canonical = bool(getattr(policy_config, "canonicalize_bimanual_rot6d", False))
    use_online_sidecar = _should_use_online_pi05_full_sidecar_adapter(
        policy_type=policy_type,
        policy_config=policy_config,
        robot_type=robot_type,
    )
    if not use_builtin_canonical and not use_online_sidecar:
        return None
    if action_chunk.shape[-1] != CANONICAL_ACTION_DIM:
        return None

    arm_valid = None
    if canonical_state is not None:
        try:
            arm_valid = extract_arm_validity_from_canonical_state(canonical_state)
        except ValueError:
            arm_valid = None
    if arm_valid is None and action_mask is not None:
        try:
            arm_valid = extract_arm_validity_from_action_mask(action_mask)
        except ValueError:
            arm_valid = None

    default_action = build_spark3_default_action_from_raw_observation(raw_observation)
    converted_chunk = canonical_bimanual_action_chunk_to_pose_gripper_dicts(action_chunk, arm_valid=arm_valid)
    return [{**default_action, **action_dict} for action_dict in converted_chunk]


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Get a logger using the standardized logging setup from utils.py.

    Args:
        name: Logger name (e.g., 'policy_server', 'robot_client')
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logs directory if logging to file
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None

    # Initialize the standardized logging
    init_logging(log_file=log_file, display_pid=False)

    # Return a named logger
    return logging.getLogger(name)


@dataclass
class TimedData:
    """A data object with timestamp and timestep information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        data: The actual data to wrap a timestamp around.
        timestep: The timestep of the data.
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation

    def get_canonical_state(self) -> torch.Tensor | None:
        canonical_state = self.observation.get(OBS_STATE)
        if not isinstance(canonical_state, torch.Tensor):
            return None
        return canonical_state

    def get_action_mask(self) -> torch.Tensor | None:
        action_mask = self.observation.get(OBS_ACTION_MASK)
        if not isinstance(action_mask, torch.Tensor):
            return None
        return action_mask


@dataclass
class FPSTracker:
    """Utility class to track FPS metrics over time."""

    target_fps: float
    first_timestamp: float = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """Calculate average FPS vs target"""
        self.total_obs_count += 1

        # Initialize first observation time
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        # Calculate overall average FPS (since start)
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0

        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        """Reset the FPS tracker state"""
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"
    robot_type: str | None = None
    rename_map: dict[str, str] = field(default_factory=dict)


def _compare_observation_states(obs1_state: torch.Tensor, obs2_state: torch.Tensor, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    return bool(torch.linalg.norm(obs1_state - obs2_state) < atol)


def observations_similar(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict], atol: float = 1
) -> bool:
    """Check if two observations are similar, under a tolerance threshold. Measures distance between
    observations as the difference in joint-space between the two observations.

    NOTE(fracapuano): This is a very simple check, and it is enough for the current use case.
    An immediate next step is to use (fast) perceptual difference metrics comparing some camera views,
    to surpass this joint-space similarity check.
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)
