#!/usr/bin/env python

from types import SimpleNamespace

from lerobot.robots.spark3_pose_gripper import Spark3PoseGripper, Spark3PoseGripperConfig
from lerobot.robots.spark3_pose_gripper.schema import (
    Spark3EndEffectorConfig,
    Spark3ObservationTopicConfig,
    action_feature_dict,
    default_end_effectors,
    parse_goal_pose_message,
    parse_joint_state_message,
    parse_state_pose_message,
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

    assert "left_arm_pose.x" in features
    assert "right_arm_pose.qw" in features
    assert "head_pose.z" in features
    assert "hip_pose.qz" in features
    assert "base_twist.linear" in features
    assert "base_twist.angular" in features
    assert "left_gripper.pos" in features
    assert "right_gripper.pos" in features


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

    assert action["left_arm_pose.x"] == 0.1
    assert action["right_arm_pose.qz"] == -0.6
    assert action["head_pose.qw"] == 1.6
    assert action["base_twist.linear"] == 0.8
    assert action["base_twist.angular"] == -0.9
    assert action["left_gripper.pos"] == 0.25
    assert action["right_gripper.pos"] == 0.75


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

    assert pose_obs["left_arm_pose.y"] == 0.2
    assert pose_obs["right_arm_pose.z"] == 1.0
    assert pose_obs["head_pose.qw"] == 2.1
    assert joint_obs["arm_left_1"] == 0.11
    assert joint_obs["arm_right_7"] == 0.77
    assert joint_obs["head_pitch"] == -0.33


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
