#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .schema import (
    DEFAULT_BASE_TWIST,
    DEFAULT_HEAD_POSE,
    DEFAULT_HIP_POSE,
    DEFAULT_LEFT_ARM_POSE,
    DEFAULT_RIGHT_ARM_POSE,
    Spark3EndEffectorConfig,
    Spark3ObservationTopicConfig,
    Spark3PoseTransformConfig,
    default_end_effectors,
    default_joint_name_mapping,
    default_left_flange_to_tcp_transform_v2,
    default_right_flange_to_tcp_transform_v2,
)


@RobotConfig.register_subclass("spark3_pose_gripper")
@dataclass
class Spark3PoseGripperConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    state_topic: str = "/wbc_state_pose"
    command_topic: str = "/wbc_goal_pose"
    joint_state_topic: str = "/joint_states"
    pose_selection: int = 3
    connect_timeout_s: float = 10.0
    ros_node_name: str = "lerobot_spark3_pose_gripper"
    trigger_initial_pose_service: str | None = None
    use_pi0_observation_state: bool = False
    joint_name_mapping: dict[str, str] = field(default_factory=default_joint_name_mapping)
    end_effectors: dict[str, Spark3EndEffectorConfig] = field(default_factory=default_end_effectors)
    sensors: dict[str, Spark3ObservationTopicConfig] = field(default_factory=dict)
    # Default to the measured TCP v2 preset for new Spark pose/gripper configs.
    left_flange_to_tcp_transform: Spark3PoseTransformConfig | None = field(
        default_factory=default_left_flange_to_tcp_transform_v2
    )
    # Default to the measured TCP v2 preset for new Spark pose/gripper configs.
    right_flange_to_tcp_transform: Spark3PoseTransformConfig | None = field(
        default_factory=default_right_flange_to_tcp_transform_v2
    )
    default_left_arm_pose: list[float] = field(default_factory=lambda: DEFAULT_LEFT_ARM_POSE.copy())
    default_right_arm_pose: list[float] = field(default_factory=lambda: DEFAULT_RIGHT_ARM_POSE.copy())
    default_head_pose: list[float] = field(default_factory=lambda: DEFAULT_HEAD_POSE.copy())
    default_hip_pose: list[float] = field(default_factory=lambda: DEFAULT_HIP_POSE.copy())
    default_base_twist: list[float] = field(default_factory=lambda: DEFAULT_BASE_TWIST.copy())

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, pose in {
            "default_left_arm_pose": self.default_left_arm_pose,
            "default_right_arm_pose": self.default_right_arm_pose,
            "default_head_pose": self.default_head_pose,
            "default_hip_pose": self.default_hip_pose,
        }.items():
            if len(pose) != 7:
                raise ValueError(f"`{name}` must contain 7 values, got {len(pose)}.")
        if len(self.default_base_twist) != 2:
            raise ValueError(
                f"`default_base_twist` must contain 2 values, got {len(self.default_base_twist)}."
            )
