#!/usr/bin/env python

from types import SimpleNamespace

import pytest

from lerobot.robots.spark3_pose_gripper import Spark3PoseGripper, Spark3PoseGripperConfig
from lerobot.robots.spark3_pose_gripper.schema import (
    Spark3EndEffectorConfig,
    Spark3ObservationTopicConfig,
    action_feature_dict,
    default_end_effectors,
    default_left_flange_to_tcp_transform,
    default_right_flange_to_tcp_transform,
    parse_goal_pose_message,
    parse_joint_state_message,
    parse_state_pose_message,
    pose_values_flange_to_tcp,
    pose_values_tcp_to_flange,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.spark3_pose_gripper import (
    Spark3PoseGripperTeleoperator,
    Spark3PoseGripperTeleoperatorConfig,
)
from lerobot.teleoperators.utils import make_teleoperator_from_config


def _pose(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> SimpleNamespace:
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
    )


def test_make_robot_and_teleoperator_from_config():
    robot = make_robot_from_config(Spark3PoseGripperConfig())
    teleop = make_teleoperator_from_config(Spark3PoseGripperTeleoperatorConfig())

    assert isinstance(robot, Spark3PoseGripper)
    assert isinstance(teleop, Spark3PoseGripperTeleoperator)


def test_action_feature_schema_contains_pose_twist_and_grippers():
    features = action_feature_dict(default_end_effectors())

    assert "main_left_x" in features
    assert "main_right_w" in features
    assert "main_head_z" in features
    assert "main_hip_qz" in features
    assert "main_v" in features
    assert "main_w" in features
    assert "main_gripper_left" in features
    assert "main_gripper_right" in features


def test_parse_goal_pose_message_keeps_pose_and_ee_values():
    msg = SimpleNamespace(
        wbc_whole_pose=SimpleNamespace(
            left_arm_goal_pose=SimpleNamespace(pose=_pose(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)),
            right_arm_goal_pose=SimpleNamespace(pose=_pose(-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7)),
            head_goal_pose=SimpleNamespace(pose=_pose(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)),
            linear_velocity=0.8,
            angular_velocity=-0.9,
        ),
        left_ee_pos=SimpleNamespace(data=[0.25]),
        right_ee_pos=SimpleNamespace(data=[0.75]),
    )

    action = parse_goal_pose_message(msg, default_end_effectors())

    assert action["main_left_x"] == 0.1
    assert action["main_right_qz"] == -0.6
    assert action["main_head_w"] == 1.6
    assert action["main_v"] == 0.8
    assert action["main_w"] == -0.9
    assert action["main_gripper_left"] == 0.25
    assert action["main_gripper_right"] == 0.75


def test_parse_goal_pose_message_converts_flange_pose_to_tcp():
    msg = SimpleNamespace(
        wbc_whole_pose=SimpleNamespace(
            left_arm_goal_pose=SimpleNamespace(pose=_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
            right_arm_goal_pose=SimpleNamespace(pose=_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
            head_goal_pose=SimpleNamespace(pose=_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
            linear_velocity=0.0,
            angular_velocity=0.0,
        ),
        left_ee_pos=SimpleNamespace(data=[0.0]),
        right_ee_pos=SimpleNamespace(data=[0.0]),
    )

    action = parse_goal_pose_message(
        msg,
        default_end_effectors(),
        left_flange_to_tcp_transform=default_left_flange_to_tcp_transform(),
        right_flange_to_tcp_transform=default_right_flange_to_tcp_transform(),
    )

    assert action["main_left_x"] == pytest.approx(0.0, abs=1e-6)
    assert action["main_left_y"] == pytest.approx(0.004, abs=1e-6)
    assert action["main_left_z"] == pytest.approx(-0.12, abs=1e-6)
    assert action["main_right_x"] == pytest.approx(0.0, abs=1e-6)
    assert action["main_right_y"] == pytest.approx(-0.004, abs=1e-6)
    assert action["main_right_z"] == pytest.approx(-0.12, abs=1e-6)


def test_parse_state_pose_and_joint_state_messages():
    state_msg = SimpleNamespace(
        left_arm_state_pose=SimpleNamespace(pose=_pose(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)),
        right_arm_state_pose=SimpleNamespace(pose=_pose(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)),
        head_state_pose=SimpleNamespace(pose=_pose(1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1)),
    )
    joint_state_msg = SimpleNamespace(
        name=["arm_left_1_joint", "arm_right_7_joint", "head_pitch_joint"],
        position=[0.11, 0.77, -0.33],
    )

    pose_obs = parse_state_pose_message(state_msg)
    joint_obs = parse_joint_state_message(joint_state_msg, Spark3PoseGripperConfig().joint_name_mapping)

    assert pose_obs["main_left_y"] == 0.2
    assert pose_obs["main_right_z"] == 1.0
    assert pose_obs["main_head_w"] == 2.1
    assert joint_obs["main_arm_left_1"] == 0.11
    assert joint_obs["main_arm_right_7"] == 0.77
    assert joint_obs["main_head_pitch"] == -0.33


def test_pose_transform_round_trip_keeps_original_pose():
    raw_left_pose = [0.25, 0.1, 1.05, 0.0, 0.0, 0.0, 1.0]
    raw_right_pose = [0.25, -0.1, 1.05, 0.0, 0.0, 0.0, 1.0]

    left_tcp_pose = pose_values_flange_to_tcp(raw_left_pose, default_left_flange_to_tcp_transform())
    right_tcp_pose = pose_values_flange_to_tcp(raw_right_pose, default_right_flange_to_tcp_transform())

    recovered_left_pose = pose_values_tcp_to_flange(left_tcp_pose, default_left_flange_to_tcp_transform())
    recovered_right_pose = pose_values_tcp_to_flange(right_tcp_pose, default_right_flange_to_tcp_transform())

    assert recovered_left_pose == pytest.approx(raw_left_pose, abs=1e-6)
    assert recovered_right_pose == pytest.approx(raw_right_pose, abs=1e-6)


def test_parse_state_pose_message_converts_flange_pose_to_tcp():
    state_msg = SimpleNamespace(
        left_arm_state_pose=SimpleNamespace(pose=_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        right_arm_state_pose=SimpleNamespace(pose=_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        head_state_pose=SimpleNamespace(pose=_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    )

    pose_obs = parse_state_pose_message(
        state_msg,
        left_flange_to_tcp_transform=default_left_flange_to_tcp_transform(),
        right_flange_to_tcp_transform=default_right_flange_to_tcp_transform(),
    )

    assert pose_obs["main_left_x"] == pytest.approx(0.0, abs=1e-6)
    assert pose_obs["main_left_y"] == pytest.approx(0.004, abs=1e-6)
    assert pose_obs["main_left_z"] == pytest.approx(-0.12, abs=1e-6)
    assert pose_obs["main_right_x"] == pytest.approx(0.0, abs=1e-6)
    assert pose_obs["main_right_y"] == pytest.approx(-0.004, abs=1e-6)
    assert pose_obs["main_right_z"] == pytest.approx(-0.12, abs=1e-6)


def test_nested_sensor_and_end_effector_configs_validate():
    sensor_cfg = Spark3ObservationTopicConfig(
        topic="/sensor_topic",
        msg_type="float64_multi_array",
        keys=["sensor.fx", "sensor.fy"],
    )
    ee_cfg = Spark3EndEffectorConfig(
        command_topic="/hand_command",
        state_topic="/hand_state",
        dofs=2,
        source="left",
        component_names=["finger_0", "finger_1"],
    )

    cfg = Spark3PoseGripperConfig(
        sensors={"force": sensor_cfg},
        end_effectors={"left_hand": ee_cfg},
    )

    assert cfg.sensors["force"].keys == ["sensor.fx", "sensor.fy"]
    assert cfg.end_effectors["left_hand"].component_names == ["finger_0", "finger_1"]
