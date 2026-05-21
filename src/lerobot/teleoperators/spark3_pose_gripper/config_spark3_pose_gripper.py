#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig
from lerobot.robots.spark3_pose_gripper.schema import (
    DEFAULT_BASE_TWIST,
    DEFAULT_HEAD_POSE,
    DEFAULT_HIP_POSE,
    DEFAULT_LEFT_ARM_POSE,
    DEFAULT_RIGHT_ARM_POSE,
    Spark3EndEffectorConfig,
    Spark3PoseTransformConfig,
    default_end_effectors,
    default_left_flange_to_tcp_transform,
    default_right_flange_to_tcp_transform,
)


@TeleoperatorConfig.register_subclass("spark3_pose_gripper")
@dataclass
class Spark3PoseGripperTeleoperatorConfig(TeleoperatorConfig):
    teleop_topic: str = "/wbc_goal_pose_ee"
    connect_timeout_s: float = 10.0
    ros_node_name: str = "lerobot_spark3_pose_gripper_teleoperator"
    end_effectors: dict[str, Spark3EndEffectorConfig] = field(default_factory=default_end_effectors)
    # Keep teleop action parsing on the same TCP preset as the robot config.
    # To switch to v2, replace this factory with the `_v2` helper in `schema.py`.
    left_flange_to_tcp_transform: Spark3PoseTransformConfig | None = field(
        default_factory=default_left_flange_to_tcp_transform
    )
    # Keep teleop action parsing on the same TCP preset as the robot config.
    # To switch to v2, replace this factory with the `_v2` helper in `schema.py`.
    right_flange_to_tcp_transform: Spark3PoseTransformConfig | None = field(
        default_factory=default_right_flange_to_tcp_transform
    )
    default_left_arm_pose: list[float] = field(default_factory=lambda: DEFAULT_LEFT_ARM_POSE.copy())
    default_right_arm_pose: list[float] = field(default_factory=lambda: DEFAULT_RIGHT_ARM_POSE.copy())
    default_head_pose: list[float] = field(default_factory=lambda: DEFAULT_HEAD_POSE.copy())
    default_hip_pose: list[float] = field(default_factory=lambda: DEFAULT_HIP_POSE.copy())
    default_base_twist: list[float] = field(default_factory=lambda: DEFAULT_BASE_TWIST.copy())

    def __post_init__(self) -> None:
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
