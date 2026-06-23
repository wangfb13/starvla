#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.utils.rotation import Rotation

POSE_COMPONENTS = ("x", "y", "z", "qx", "qy", "qz", "qw")
BASE_TWIST_COMPONENTS = ("linear", "angular")

SCALAR_SENSOR_MSG_TYPES = {"float64", "uint16"}
ARRAY_SENSOR_MSG_TYPES = {"float64_multi_array", "uint16_multi_array"}
SUPPORTED_SENSOR_MSG_TYPES = SCALAR_SENSOR_MSG_TYPES | ARRAY_SENSOR_MSG_TYPES

DEFAULT_LEFT_ARM_POSE = [0.19165, 0.222878, 1.130936, -0.011698, -0.701645, 0.054166, 0.710369]
DEFAULT_RIGHT_ARM_POSE = [0.19165, -0.221378, 1.130936, 0.011698, -0.701645, -0.054166, 0.710369]
DEFAULT_HEAD_POSE = [0.0, 0.0, 1.7, 0.0, 0.0, 0.0, 1.0]
DEFAULT_HIP_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
DEFAULT_BASE_TWIST = [0.0, 0.0]
DEFAULT_LEFT_FLANGE_TO_TCP_TRANSFORM = [
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.004],
    [0.0, 0.0, -1.0, -0.12],
    [0.0, 0.0, 0.0, 1.0],
]
DEFAULT_RIGHT_FLANGE_TO_TCP_TRANSFORM = [
    [0.0, -1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, -0.004],
    [0.0, 0.0, -1.0, -0.12],
    [0.0, 0.0, 0.0, 1.0],
]
# Alternative TCP preset kept for quick switching if the real robot is using v2.
DEFAULT_LEFT_FLANGE_TO_TCP_TRANSFORM_V2 = [
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.01035],
    [0.0, 0.0, -1.0, -0.12971],
    [0.0, 0.0, 0.0, 1.0],
]
DEFAULT_RIGHT_FLANGE_TO_TCP_TRANSFORM_V2 = [
    [0.0, -1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, -0.01035],
    [0.0, 0.0, -1.0, -0.12971],
    [0.0, 0.0, 0.0, 1.0],
]

PI0_OBSERVATION_STATE_NAMES = (
    "main_left_x",
    "main_left_y",
    "main_left_z",
    "main_left_qx",
    "main_left_qy",
    "main_left_qz",
    "main_left_w",
    "main_right_x",
    "main_right_y",
    "main_right_z",
    "main_right_qx",
    "main_right_qy",
    "main_right_qz",
    "main_right_w",
    "main_head_x",
    "main_head_y",
    "main_head_z",
    "main_head_qx",
    "main_head_qy",
    "main_head_qz",
    "main_head_w",
    "main_arm_left_1",
    "main_arm_left_2",
    "main_arm_left_3",
    "main_arm_left_4",
    "main_arm_left_5",
    "main_arm_left_6",
    "main_arm_left_7",
    "main_arm_right_1",
    "main_arm_right_2",
    "main_arm_right_3",
    "main_arm_right_4",
    "main_arm_right_5",
    "main_arm_right_6",
    "main_arm_right_7",
    "main_head_pitch",
    "main_head_yaw",
    "main_hip_pitch",
    "main_hip_yaw",
    "main_knee",
    "main_left_wheel",
    "main_right_wheel",
    "main_gripper_left",
    "main_gripper_right",
)

DEFAULT_JOINT_NAME_MAPPING = {
    "arm_left_1_joint": "main_arm_left_1",
    "arm_left_2_joint": "main_arm_left_2",
    "arm_left_3_joint": "main_arm_left_3",
    "arm_left_4_joint": "main_arm_left_4",
    "arm_left_5_joint": "main_arm_left_5",
    "arm_left_6_joint": "main_arm_left_6",
    "arm_left_7_joint": "main_arm_left_7",
    "arm_right_1_joint": "main_arm_right_1",
    "arm_right_2_joint": "main_arm_right_2",
    "arm_right_3_joint": "main_arm_right_3",
    "arm_right_4_joint": "main_arm_right_4",
    "arm_right_5_joint": "main_arm_right_5",
    "arm_right_6_joint": "main_arm_right_6",
    "arm_right_7_joint": "main_arm_right_7",
    "head_pitch_joint": "main_head_pitch",
    "head_yaw_joint": "main_head_yaw",
    "hip_pitch_joint": "main_hip_pitch",
    "hip_yaw_joint": "main_hip_yaw",
    "knee_joint": "main_knee",
    "left_wheel_joint": "main_left_wheel",
    "right_wheel_joint": "main_right_wheel",
}


@dataclass
class Spark3EndEffectorConfig:
    command_topic: str
    state_topic: str | None = None
    dofs: int = 1
    source: str | None = None
    component_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dofs < 1:
            raise ValueError(f"`dofs` must be positive, got {self.dofs}.")
        if self.source not in {None, "left", "right"}:
            raise ValueError(f"`source` must be one of None/'left'/'right', got {self.source}.")
        if self.component_names and len(self.component_names) != self.dofs:
            raise ValueError(
                f"`component_names` length ({len(self.component_names)}) must match `dofs` ({self.dofs})."
            )


@dataclass
class Spark3ObservationTopicConfig:
    topic: str
    msg_type: str = "float64"
    keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.msg_type not in SUPPORTED_SENSOR_MSG_TYPES:
            raise ValueError(
                f"`msg_type` must be one of {sorted(SUPPORTED_SENSOR_MSG_TYPES)}, got {self.msg_type}."
            )
        if self.msg_type in ARRAY_SENSOR_MSG_TYPES and not self.keys:
            raise ValueError("Array sensor topics require explicit `keys` so dataset features are stable.")
        if self.msg_type in SCALAR_SENSOR_MSG_TYPES and len(self.keys) > 1:
            raise ValueError("Scalar sensor topics can define at most one output key.")


@dataclass
class Spark3PoseTransformConfig:
    transform_matrix: list[list[float]]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.transform_matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"`transform_matrix` must be 4x4, got shape {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("`transform_matrix` must contain only finite values.")
        self.transform_matrix = matrix.tolist()

    def as_numpy(self) -> np.ndarray:
        return np.asarray(self.transform_matrix, dtype=float)


def default_joint_name_mapping() -> dict[str, str]:
    return dict(DEFAULT_JOINT_NAME_MAPPING)


def default_end_effectors() -> dict[str, Spark3EndEffectorConfig]:
    return {
        "main_gripper_left": Spark3EndEffectorConfig(
            command_topic="/left_gripper_command",
            state_topic="/left_gripper_state",
            dofs=1,
            source="left",
        ),
        "main_gripper_right": Spark3EndEffectorConfig(
            command_topic="/right_gripper_command",
            state_topic="/right_gripper_state",
            dofs=1,
            source="right",
        ),
    }


def default_left_flange_to_tcp_transform() -> Spark3PoseTransformConfig:
    return Spark3PoseTransformConfig(transform_matrix=[row.copy() for row in DEFAULT_LEFT_FLANGE_TO_TCP_TRANSFORM])


def default_right_flange_to_tcp_transform() -> Spark3PoseTransformConfig:
    return Spark3PoseTransformConfig(transform_matrix=[row.copy() for row in DEFAULT_RIGHT_FLANGE_TO_TCP_TRANSFORM])


def default_left_flange_to_tcp_transform_v2() -> Spark3PoseTransformConfig:
    return Spark3PoseTransformConfig(transform_matrix=[row.copy() for row in DEFAULT_LEFT_FLANGE_TO_TCP_TRANSFORM_V2])


def default_right_flange_to_tcp_transform_v2() -> Spark3PoseTransformConfig:
    return Spark3PoseTransformConfig(
        transform_matrix=[row.copy() for row in DEFAULT_RIGHT_FLANGE_TO_TCP_TRANSFORM_V2]
    )


def ordered_joint_names(joint_name_mapping: dict[str, str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in joint_name_mapping.values():
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _legacy_pose_component(component: str) -> str:
    return "w" if component == "qw" else component


def _pose_feature_name(prefix: str, component: str) -> str:
    if prefix.startswith("main_"):
        return f"{prefix}_{_legacy_pose_component(component)}"
    return f"{prefix}.{component}"


def pose_feature_names(prefix: str) -> list[str]:
    return [_pose_feature_name(prefix, component) for component in POSE_COMPONENTS]


def pose_feature_dict(prefix: str) -> dict[str, type]:
    return {name: float for name in pose_feature_names(prefix)}


def pose_dict(prefix: str, values: list[float] | tuple[float, ...]) -> dict[str, float]:
    if len(values) != len(POSE_COMPONENTS):
        raise ValueError(
            f"{prefix} expects {len(POSE_COMPONENTS)} values, got {len(values)}."
        )
    return {
        _pose_feature_name(prefix, component): float(value)
        for component, value in zip(POSE_COMPONENTS, values, strict=True)
    }


def pose_aliases(prefix: str, component: str) -> list[str]:
    legacy_component = _legacy_pose_component(component)
    aliases = [_pose_feature_name(prefix, component)]

    if prefix == "left_arm_pose":
        aliases.extend([f"main_left_{legacy_component}", f"left_{component}"])
    elif prefix == "right_arm_pose":
        aliases.extend([f"main_right_{legacy_component}", f"right_{component}"])
    elif prefix == "head_pose":
        aliases.extend([f"main_head_{legacy_component}", f"head_{component}"])
    elif prefix == "hip_pose":
        aliases.extend([f"main_hip_{legacy_component}", f"hip_{component}"])

    return aliases


def pose_values_from_action(
    action: dict[str, Any], prefix: str, default: list[float] | tuple[float, ...]
) -> list[float]:
    values = []
    for idx, component in enumerate(POSE_COMPONENTS):
        value = default[idx]
        for key in pose_aliases(prefix, component):
            if key in action:
                value = action[key]
                break
        values.append(float(value))
    return values


def extract_pose_values(pose: Any) -> list[float]:
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def _pose_values_to_transform_matrix(values: list[float] | tuple[float, ...]) -> np.ndarray:
    if len(values) != len(POSE_COMPONENTS):
        raise ValueError(f"Expected {len(POSE_COMPONENTS)} pose values, got {len(values)}.")

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_quat(np.asarray(values[3:], dtype=float)).as_matrix()
    transform[:3, 3] = np.asarray(values[:3], dtype=float)
    return transform


def _transform_matrix_to_pose_values(transform: np.ndarray) -> list[float]:
    rotation = Rotation.from_matrix(transform[:3, :3]).as_quat().tolist()
    translation = transform[:3, 3].tolist()
    return [*translation, *rotation]


def pose_values_flange_to_tcp(
    values: list[float] | tuple[float, ...],
    transform_config: Spark3PoseTransformConfig | None,
) -> list[float]:
    if transform_config is None:
        return [float(value) for value in values]

    flange_pose = _pose_values_to_transform_matrix(values)
    tcp_pose = flange_pose @ transform_config.as_numpy()
    return _transform_matrix_to_pose_values(tcp_pose)


def pose_values_tcp_to_flange(
    values: list[float] | tuple[float, ...],
    transform_config: Spark3PoseTransformConfig | None,
) -> list[float]:
    if transform_config is None:
        return [float(value) for value in values]

    tcp_pose = _pose_values_to_transform_matrix(values)
    flange_pose = tcp_pose @ np.linalg.inv(transform_config.as_numpy())
    return _transform_matrix_to_pose_values(flange_pose)


def base_twist_feature_dict() -> dict[str, type]:
    return {"main_v": float, "main_w": float}


def base_twist_dict(values: list[float] | tuple[float, ...]) -> dict[str, float]:
    if len(values) != len(BASE_TWIST_COMPONENTS):
        raise ValueError(
            f"base twist expects {len(BASE_TWIST_COMPONENTS)} values, got {len(values)}."
        )
    return {"main_v": float(values[0]), "main_w": float(values[1])}


def base_twist_values_from_action(
    action: dict[str, Any], default: list[float] | tuple[float, ...]
) -> list[float]:
    return [
        float(action.get("main_v", action.get("base_twist.linear", default[0]))),
        float(action.get("main_w", action.get("base_twist.angular", default[1]))),
    ]


def end_effector_components(config: Spark3EndEffectorConfig) -> list[str]:
    if config.component_names:
        return list(config.component_names)
    if config.dofs == 1:
        return [""]
    return [f"joint_{idx}" for idx in range(config.dofs)]


def end_effector_feature_names(name: str, config: Spark3EndEffectorConfig) -> list[str]:
    return [
        name if component == "" else f"{name}.{component}"
        for component in end_effector_components(config)
    ]


def end_effector_feature_dict(
    end_effectors: dict[str, Spark3EndEffectorConfig],
) -> dict[str, type]:
    features: dict[str, type] = {}
    for name, config in end_effectors.items():
        features.update({feature_name: float for feature_name in end_effector_feature_names(name, config)})
    return features


def end_effector_dict(
    name: str,
    config: Spark3EndEffectorConfig,
    values: list[float] | tuple[float, ...],
) -> dict[str, float]:
    if len(values) != config.dofs:
        raise ValueError(f"{name} expects {config.dofs} values, got {len(values)}.")
    return {
        feature_name: float(value)
        for feature_name, value in zip(end_effector_feature_names(name, config), values, strict=True)
    }


def end_effector_values_from_action(
    action: dict[str, Any],
    name: str,
    config: Spark3EndEffectorConfig,
    default: list[float] | tuple[float, ...] | None = None,
) -> list[float]:
    fallback = list(default) if default is not None else [0.0] * config.dofs
    values = []
    for idx, feature_name in enumerate(end_effector_feature_names(name, config)):
        aliases = [feature_name]
        if name == "main_gripper_left":
            aliases.extend(["left_gripper.pos", "gripper_left", "left_gripper"])
        elif name == "main_gripper_right":
            aliases.extend(["right_gripper.pos", "gripper_right", "right_gripper"])
        value = fallback[idx]
        for key in aliases:
            if key in action:
                value = action[key]
                break
        values.append(float(value))
    return values


def sensor_feature_names(name: str, config: Spark3ObservationTopicConfig) -> list[str]:
    if config.keys:
        return list(config.keys)
    return [name]


def sensor_feature_dict(
    sensors: dict[str, Spark3ObservationTopicConfig],
) -> dict[str, type]:
    features: dict[str, type] = {}
    for name, config in sensors.items():
        features.update({feature_name: float for feature_name in sensor_feature_names(name, config)})
    return features


def sensor_dict(
    name: str,
    config: Spark3ObservationTopicConfig,
    values: list[float] | tuple[float, ...],
) -> dict[str, float]:
    feature_names = sensor_feature_names(name, config)
    if len(values) != len(feature_names):
        raise ValueError(f"{name} expects {len(feature_names)} values, got {len(values)}.")
    return {
        feature_name: float(value)
        for feature_name, value in zip(feature_names, values, strict=True)
    }


def parse_sensor_message_values(config: Spark3ObservationTopicConfig, msg: Any) -> list[float]:
    if config.msg_type == "float64":
        return [float(msg.data)]
    if config.msg_type == "uint16":
        return [float(msg.data)]
    if config.msg_type == "float64_multi_array":
        return [float(value) for value in msg.data]
    if config.msg_type == "uint16_multi_array":
        return [float(value) for value in msg.data]
    raise ValueError(f"Unsupported sensor message type {config.msg_type}.")


def default_action_dict(
    end_effectors: dict[str, Spark3EndEffectorConfig],
    *,
    left_arm_pose: list[float] | tuple[float, ...] = DEFAULT_LEFT_ARM_POSE,
    right_arm_pose: list[float] | tuple[float, ...] = DEFAULT_RIGHT_ARM_POSE,
    head_pose: list[float] | tuple[float, ...] = DEFAULT_HEAD_POSE,
    hip_pose: list[float] | tuple[float, ...] = DEFAULT_HIP_POSE,
    base_twist: list[float] | tuple[float, ...] = DEFAULT_BASE_TWIST,
) -> dict[str, float]:
    action: dict[str, float] = {}
    action.update(pose_dict("main_left", left_arm_pose))
    action.update(pose_dict("main_right", right_arm_pose))
    action.update(pose_dict("main_head", head_pose))
    action.update(pose_dict("main_hip", hip_pose))
    action.update(base_twist_dict(base_twist))
    for name, config in end_effectors.items():
        action.update(end_effector_dict(name, config, [0.0] * config.dofs))
    return action


def action_dict_from_observation(
    observation: dict[str, Any],
    end_effectors: dict[str, Spark3EndEffectorConfig] | None = None,
) -> dict[str, float]:
    configured_end_effectors = end_effectors if end_effectors is not None else default_end_effectors()
    action = default_action_dict(configured_end_effectors)
    for key in list(action):
        if key in observation:
            action[key] = float(observation[key])
    return action


def action_feature_dict(end_effectors: dict[str, Spark3EndEffectorConfig]) -> dict[str, type]:
    features: dict[str, type] = {}
    features.update(pose_feature_dict("main_left"))
    features.update(pose_feature_dict("main_right"))
    features.update(pose_feature_dict("main_head"))
    features.update(pose_feature_dict("main_hip"))
    features.update(base_twist_feature_dict())
    features.update(end_effector_feature_dict(end_effectors))
    return features


def observation_feature_dict(
    *,
    joint_names: tuple[str, ...],
    end_effectors: dict[str, Spark3EndEffectorConfig],
    sensors: dict[str, Spark3ObservationTopicConfig],
    camera_features: dict[str, tuple[int, int, int]],
    state_names: tuple[str, ...] | None = None,
) -> dict[str, type | tuple[int, int, int]]:
    features: dict[str, type | tuple[int, int, int]] = {}
    features.update(pose_feature_dict("main_left"))
    features.update(pose_feature_dict("main_right"))
    features.update(pose_feature_dict("main_head"))
    features.update({joint_name: float for joint_name in joint_names})
    features.update(end_effector_feature_dict(end_effectors))
    features.update(sensor_feature_dict(sensors))
    features.update(camera_features)
    if state_names is not None:
        features = filter_observation_state(features, state_names, camera_features.keys())
    return features


def filter_observation_state(
    observation: dict[str, Any],
    state_names: tuple[str, ...],
    camera_names: Any,
) -> dict[str, Any]:
    camera_name_set = set(camera_names)
    missing = [name for name in state_names if name not in observation]
    if missing:
        raise ValueError(f"Observation is missing configured state keys: {missing}")
    filtered = {key: observation[key] for key in state_names}
    filtered.update({key: value for key, value in observation.items() if key in camera_name_set})
    return filtered


def camera_feature_dict(camera_configs: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    features: dict[str, tuple[int, int, int]] = {}
    for camera_name, config in camera_configs.items():
        if "tactile" in camera_name:
            tactile_base_name = camera_name[:-8] if camera_name.endswith("_tactile") else camera_name
            if getattr(config, "inference_mode", False):
                features[f"{tactile_base_name}_left_tactile_depth_deformation"] = (config.height, config.width, 3)
                features[f"{tactile_base_name}_right_tactile_depth_deformation"] = (config.height, config.width, 3)
            else:
                features[f"{tactile_base_name}_left_tactile"] = (config.height, config.width, 3)
                features[f"{tactile_base_name}_right_tactile"] = (config.height, config.width, 3)
            continue

        features[camera_name] = (config.height, config.width, 3)
        if getattr(config, "use_depth", False):
            features[f"{camera_name}_depth"] = (config.height, config.width, 1)
    return features


def split_camera_observation(camera_name: str, frame: Any) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    if isinstance(frame, dict):
        if "color" in frame:
            obs[camera_name] = frame["color"]
        if "depth" in frame:
            obs[f"{camera_name}_depth"] = expand_depth_frame(frame["depth"])

        tactile_base_name = camera_name[:-8] if camera_name.endswith("_tactile") else camera_name
        if "left_tactile_depth_deformation" in frame:
            obs[f"{tactile_base_name}_left_tactile_depth_deformation"] = frame["left_tactile_depth_deformation"]
        if "right_tactile_depth_deformation" in frame:
            obs[f"{tactile_base_name}_right_tactile_depth_deformation"] = frame["right_tactile_depth_deformation"]
        if "left_tactile" in frame:
            obs[f"{tactile_base_name}_left_tactile"] = frame["left_tactile"]
        if "right_tactile" in frame:
            obs[f"{tactile_base_name}_right_tactile"] = frame["right_tactile"]
        return obs

    obs[camera_name] = frame
    return obs


def expand_depth_frame(depth_frame: Any) -> Any:
    if depth_frame is None:
        return None
    if hasattr(depth_frame, "ndim") and depth_frame.ndim == 2:
        return depth_frame[..., None]
    return depth_frame


def parse_goal_pose_message(
    msg: Any,
    end_effectors: dict[str, Spark3EndEffectorConfig],
    *,
    default_head_pose: list[float] | tuple[float, ...] = DEFAULT_HEAD_POSE,
    default_hip_pose: list[float] | tuple[float, ...] = DEFAULT_HIP_POSE,
    default_base_twist: list[float] | tuple[float, ...] = DEFAULT_BASE_TWIST,
    left_flange_to_tcp_transform: Spark3PoseTransformConfig | None = None,
    right_flange_to_tcp_transform: Spark3PoseTransformConfig | None = None,
) -> dict[str, float]:
    whole_pose = getattr(msg, "wbc_whole_pose", msg)
    action: dict[str, float] = {}
    left_pose = pose_values_flange_to_tcp(
        extract_pose_values(whole_pose.left_arm_goal_pose.pose),
        left_flange_to_tcp_transform,
    )
    right_pose = pose_values_flange_to_tcp(
        extract_pose_values(whole_pose.right_arm_goal_pose.pose),
        right_flange_to_tcp_transform,
    )
    action.update(pose_dict("main_left", left_pose))
    action.update(pose_dict("main_right", right_pose))

    if hasattr(whole_pose, "head_goal_pose"):
        action.update(pose_dict("main_head", extract_pose_values(whole_pose.head_goal_pose.pose)))
    else:
        action.update(pose_dict("main_head", default_head_pose))

    if hasattr(whole_pose, "hip_goal_pose") and hasattr(whole_pose.hip_goal_pose, "pose"):
        action.update(pose_dict("main_hip", extract_pose_values(whole_pose.hip_goal_pose.pose)))
    else:
        action.update(pose_dict("main_hip", default_hip_pose))

    linear_velocity = float(getattr(whole_pose, "linear_velocity", default_base_twist[0]))
    angular_velocity = float(getattr(whole_pose, "angular_velocity", default_base_twist[1]))
    action.update(base_twist_dict([linear_velocity, angular_velocity]))

    for name, config in end_effectors.items():
        if config.source == "left" and hasattr(msg, "left_ee_pos"):
            values = list(msg.left_ee_pos.data)
        elif config.source == "right" and hasattr(msg, "right_ee_pos"):
            values = list(msg.right_ee_pos.data)
        else:
            values = [0.0] * config.dofs
        action.update(end_effector_dict(name, config, values))

    return action


def parse_state_pose_message(
    msg: Any,
    *,
    left_flange_to_tcp_transform: Spark3PoseTransformConfig | None = None,
    right_flange_to_tcp_transform: Spark3PoseTransformConfig | None = None,
) -> dict[str, float]:
    obs: dict[str, float] = {}
    left_pose = pose_values_flange_to_tcp(
        extract_pose_values(msg.left_arm_state_pose.pose),
        left_flange_to_tcp_transform,
    )
    right_pose = pose_values_flange_to_tcp(
        extract_pose_values(msg.right_arm_state_pose.pose),
        right_flange_to_tcp_transform,
    )
    obs.update(pose_dict("main_left", left_pose))
    obs.update(pose_dict("main_right", right_pose))
    obs.update(pose_dict("main_head", extract_pose_values(msg.head_state_pose.pose)))
    return obs


def parse_joint_state_message(
    msg: Any,
    joint_name_mapping: dict[str, str],
) -> dict[str, float]:
    joint_positions: dict[str, float] = {}
    for idx, joint_name in enumerate(msg.name):
        mapped_name = joint_name_mapping.get(joint_name)
        if mapped_name is None and joint_name.endswith("_joint"):
            mapped_name = f"main_{joint_name.removesuffix('_joint')}"
        if mapped_name is None:
            continue
        if idx >= len(msg.position):
            continue
        joint_positions[mapped_name] = float(msg.position[idx])
    return joint_positions
