#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from functools import cached_property
from threading import Event, Lock
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_spark3_pose_gripper import Spark3PoseGripperConfig
from .ros_utils import Spark3RosRuntime, ensure_ros_node, load_ros_runtime
from .schema import (
    action_feature_dict,
    base_twist_values_from_action,
    camera_feature_dict,
    clamp_end_effector_values,
    default_action_dict,
    end_effector_dict,
    end_effector_feature_names,
    end_effector_values_from_action,
    expand_depth_frame,
    filter_observation_state,
    joint_action_feature_dict,
    observation_feature_dict,
    ordered_joint_names,
    parse_joint_state_message,
    parse_sensor_message_values,
    parse_state_pose_message,
    PI0_OBSERVATION_STATE_NAMES,
    pose_dict,
    pose_values_flange_to_tcp,
    pose_values_from_action,
    pose_values_tcp_to_flange,
    sensor_dict,
    split_camera_observation,
)

logger = logging.getLogger(__name__)


class _EndEffectorHandle:
    def __init__(self, runtime: Spark3RosRuntime, name: str, config: Any):
        self.runtime = runtime
        self.name = name
        self.config = config
        self.values = [0.0] * config.dofs
        self.last_command_values = [0.0] * config.dofs
        self.ready = Event()
        self.publisher = None
        self.subscriber = None

    def connect(self) -> None:
        if self.config.dofs == 1:
            self.publisher = self.runtime.rospy.Publisher(
                self.config.command_topic,
                self.runtime.Float64,
                queue_size=5,
            )
        else:
            self.publisher = self.runtime.rospy.Publisher(
                self.config.command_topic,
                self.runtime.Float64MultiArray,
                queue_size=5,
            )
        logger.info(
            "[spark3-gripper-connect] name=%s command_topic=%s state_topic=%s dofs=%s",
            self.name,
            self.config.command_topic,
            self.config.state_topic,
            self.config.dofs,
        )

        if self.config.state_topic is None:
            self.ready.set()
            return

        msg_type = self.runtime.Float64 if self.config.dofs == 1 else self.runtime.Float64MultiArray
        self.subscriber = self.runtime.rospy.Subscriber(
            self.config.state_topic,
            msg_type,
            self._handle_state,
        )

    def _handle_state(self, msg: Any) -> None:
        if self.config.dofs == 1:
            self.values = [float(msg.data)]
        else:
            self.values = [float(value) for value in msg.data]
        self.ready.set()

    def publish(self, values: list[float]) -> None:
        if self.publisher is None:
            raise RuntimeError(f"{self.name} publisher is not initialized.")
        values = clamp_end_effector_values(values, self.config)
        self.last_command_values = list(values)
        if self.config.dofs == 1:
            msg = self.runtime.Float64()
            msg.data = float(values[0])
        else:
            msg = self.runtime.Float64MultiArray()
            msg.data = list(values)
        self.publisher.publish(msg)

    def observation(self) -> dict[str, float]:
        return end_effector_dict(self.name, self.config, self.values)

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command_topic": self.config.command_topic,
            "state_topic": self.config.state_topic,
            "command_values": list(self.last_command_values),
            "state_values": list(self.values),
        }

    def disconnect(self) -> None:
        if self.subscriber is not None and hasattr(self.subscriber, "unregister"):
            self.subscriber.unregister()
        if self.publisher is not None and hasattr(self.publisher, "unregister"):
            self.publisher.unregister()
        self.ready.clear()


class _SensorHandle:
    def __init__(self, runtime: Spark3RosRuntime, name: str, config: Any):
        self.runtime = runtime
        self.name = name
        self.config = config
        self.values = [0.0] * max(1, len(config.keys))
        self.ready = Event()
        self.subscriber = None

    def connect(self) -> None:
        msg_type = {
            "float64": self.runtime.Float64,
            "uint16": self.runtime.UInt16,
            "float64_multi_array": self.runtime.Float64MultiArray,
            "uint16_multi_array": self.runtime.UInt16MultiArray,
        }[self.config.msg_type]
        self.subscriber = self.runtime.rospy.Subscriber(
            self.config.topic,
            msg_type,
            self._handle_state,
        )

    def _handle_state(self, msg: Any) -> None:
        self.values = parse_sensor_message_values(self.config, msg)
        self.ready.set()

    def observation(self) -> dict[str, float]:
        return sensor_dict(self.name, self.config, self.values)

    def disconnect(self) -> None:
        if self.subscriber is not None and hasattr(self.subscriber, "unregister"):
            self.subscriber.unregister()
        self.ready.clear()


class Spark3PoseGripper(Robot):
    config_class = Spark3PoseGripperConfig
    name = "spark3_pose_gripper"

    def __init__(self, config: Spark3PoseGripperConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._joint_names = ordered_joint_names(config.joint_name_mapping)
        self._action_joint_names = tuple(config.action_joint_names) if config.action_joint_names else self._joint_names
        self._end_effector_action_names = tuple(self.config.end_effectors.keys())
        if config.joint_command_names:
            self._joint_command_names = tuple(config.joint_command_names)
        else:
            self._joint_command_names = tuple(
                name for name in self._action_joint_names if name not in self._end_effector_action_names
            )
        self._internal_to_ros_joint_names = self._build_internal_to_ros_joint_names()
        self._joint_positions = {joint_name: 0.0 for joint_name in self._joint_names}
        initial_pose_observation = default_action_dict(
            {},
            left_arm_pose=config.default_left_arm_pose,
            right_arm_pose=config.default_right_arm_pose,
            head_pose=config.default_head_pose,
            hip_pose=config.default_hip_pose,
            base_twist=config.default_base_twist,
        )
        initial_pose_observation.update(
            pose_dict(
                "main_left",
                pose_values_flange_to_tcp(
                    config.default_left_arm_pose,
                    config.left_flange_to_tcp_transform,
                ),
            )
        )
        initial_pose_observation.update(
            pose_dict(
                "main_right",
                pose_values_flange_to_tcp(
                    config.default_right_arm_pose,
                    config.right_flange_to_tcp_transform,
                ),
            )
        )
        self._pose_observation = {
            **initial_pose_observation
        }
        self._last_action = default_action_dict(
            config.end_effectors,
            left_arm_pose=config.default_left_arm_pose,
            right_arm_pose=config.default_right_arm_pose,
            head_pose=config.default_head_pose,
            hip_pose=config.default_hip_pose,
            base_twist=config.default_base_twist,
        )
        self._last_action.update(
            pose_dict(
                "main_left",
                pose_values_flange_to_tcp(
                    config.default_left_arm_pose,
                    config.left_flange_to_tcp_transform,
                ),
            )
        )
        self._last_action.update(
            pose_dict(
                "main_right",
                pose_values_flange_to_tcp(
                    config.default_right_arm_pose,
                    config.right_flange_to_tcp_transform,
                ),
            )
        )
        self._last_action.update({joint_name: 0.0 for joint_name in self._action_joint_names})
        self._is_connected = False
        self._state_ready = Event()
        self._joint_ready = Event()
        self._state_lock = Lock()
        self._ros: Spark3RosRuntime | None = None
        self._wbc_goal_publisher = None
        self._joint_command_publisher = None
        self._pose_subscriber = None
        self._joint_subscriber = None
        self._end_effector_handles: dict[str, _EndEffectorHandle] = {}
        self._sensor_handles: dict[str, _SensorHandle] = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return observation_feature_dict(
            joint_names=self._joint_names,
            end_effectors=self.config.end_effectors,
            sensors=self.config.sensors,
            camera_features=camera_feature_dict(self.config.cameras),
            state_names=PI0_OBSERVATION_STATE_NAMES if self.config.use_pi0_observation_state else None,
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.command_mode == "joint":
            return joint_action_feature_dict(self._action_joint_names, self.config.end_effectors)
        return action_feature_dict(self.config.end_effectors)

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        try:
            self._ros = load_ros_runtime()
            ensure_ros_node(self._ros, self.config.ros_node_name)

            if self.config.command_mode == "pose":
                self._wbc_goal_publisher = self._ros.rospy.Publisher(
                    self.config.command_topic,
                    self._ros.WbcPoseExpand,
                    queue_size=5,
                )
            else:
                self._joint_command_publisher = self._ros.rospy.Publisher(
                    self.config.joint_command_topic,
                    self._ros.JointState,
                    queue_size=5,
                )
            self._pose_subscriber = self._ros.rospy.Subscriber(
                self.config.state_topic,
                self._ros.WbcState,
                self._handle_pose_state,
            )

            self._joint_subscriber = self._ros.rospy.Subscriber(
                self.config.joint_state_topic,
                self._ros.JointState,
                self._handle_joint_state,
            )

            self._end_effector_handles = {
                name: _EndEffectorHandle(self._ros, name, ee_config)
                for name, ee_config in self.config.end_effectors.items()
            }
            for handle in self._end_effector_handles.values():
                handle.connect()

            self._sensor_handles = {
                name: _SensorHandle(self._ros, name, sensor_config)
                for name, sensor_config in self.config.sensors.items()
            }
            for handle in self._sensor_handles.values():
                handle.connect()

            for camera in self.cameras.values():
                camera.connect()

            self._wait_until_ready()

            if self.config.trigger_initial_pose_service:
                self._trigger_initial_pose_service()

            self._is_connected = True
            logger.info("%s connected.", self)
        except Exception:
            self._cleanup_resources()
            raise

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.config.connect_timeout_s
        while time.time() < deadline:
            end_effectors_ready = all(handle.ready.is_set() for handle in self._end_effector_handles.values())
            sensors_ready = all(handle.ready.is_set() for handle in self._sensor_handles.values())
            if self._state_ready.is_set() and self._joint_ready.is_set() and end_effectors_ready and sensors_ready:
                return
            time.sleep(0.05)

        missing = []
        if not self._state_ready.is_set():
            missing.append(self.config.state_topic)
        if not self._joint_ready.is_set():
            missing.append(self.config.joint_state_topic)
        missing.extend(
            handle.config.state_topic
            for handle in self._end_effector_handles.values()
            if handle.config.state_topic and not handle.ready.is_set()
        )
        missing.extend(
            handle.config.topic for handle in self._sensor_handles.values() if not handle.ready.is_set()
        )
        raise TimeoutError(f"Timed out waiting for Spark3 topics: {missing}")

    def _trigger_initial_pose_service(self) -> None:
        if self._ros is None or self.config.trigger_initial_pose_service is None:
            return

        from std_srvs.srv import Trigger

        self._ros.rospy.wait_for_service(self.config.trigger_initial_pose_service, timeout=self.config.connect_timeout_s)
        trigger_service = self._ros.rospy.ServiceProxy(self.config.trigger_initial_pose_service, Trigger)
        response = trigger_service()
        if not response.success:
            raise RuntimeError(
                f"Initial pose service {self.config.trigger_initial_pose_service} failed: {response.message}"
            )

    def _handle_pose_state(self, msg: Any) -> None:
        with self._state_lock:
            self._pose_observation = parse_state_pose_message(
                msg,
                left_flange_to_tcp_transform=self.config.left_flange_to_tcp_transform,
                right_flange_to_tcp_transform=self.config.right_flange_to_tcp_transform,
            )
        self._state_ready.set()

    def _handle_joint_state(self, msg: Any) -> None:
        joint_positions = parse_joint_state_message(msg, self.config.joint_name_mapping)
        with self._state_lock:
            self._joint_positions.update(joint_positions)
        self._joint_ready.set()

    def _make_pose_stamped(self, pose_values: list[float]) -> Any:
        if self._ros is None:
            raise RuntimeError("ROS runtime is not initialized.")
        pose_stamped = self._ros.PoseStamped()
        pose_stamped.header = self._ros.Header(
            stamp=self._ros.rospy.Time.now(),
            frame_id="base_link",
        )
        pose_stamped.pose.position = self._ros.Point(*pose_values[:3])
        pose_stamped.pose.orientation = self._ros.Quaternion(*pose_values[3:])
        return pose_stamped

    def _build_internal_to_ros_joint_names(self) -> dict[str, str]:
        ros_to_internal = self.config.joint_name_mapping
        internal_to_ros: dict[str, str] = {}
        for ros_name, internal_name in ros_to_internal.items():
            internal_to_ros.setdefault(internal_name, ros_name)
        return internal_to_ros

    def _infer_ros_joint_name(self, internal_joint_name: str) -> str:
        if internal_joint_name in self._internal_to_ros_joint_names:
            return self._internal_to_ros_joint_names[internal_joint_name]
        if internal_joint_name.startswith("main_"):
            return f"{internal_joint_name.removeprefix('main_')}_joint"
        return internal_joint_name

    def _build_command_message(self, action: RobotAction) -> Any:
        if self._ros is None:
            raise RuntimeError("ROS runtime is not initialized.")
        msg = self._ros.WbcPoseExpand()
        left_pose = pose_values_tcp_to_flange(
            pose_values_from_action(action, "left_arm_pose", self.config.default_left_arm_pose),
            self.config.left_flange_to_tcp_transform,
        )
        msg.left_arm_goal_pose = self._make_pose_stamped(
            left_pose
        )
        right_pose = pose_values_tcp_to_flange(
            pose_values_from_action(action, "right_arm_pose", self.config.default_right_arm_pose),
            self.config.right_flange_to_tcp_transform,
        )
        msg.right_arm_goal_pose = self._make_pose_stamped(
            right_pose
        )
        msg.head_goal_pose = self._make_pose_stamped(
            pose_values_from_action(action, "head_pose", self.config.default_head_pose)
        )
        msg.hip_goal_pose = self._make_pose_stamped(
            pose_values_from_action(action, "hip_pose", self.config.default_hip_pose)
        )
        base_twist = base_twist_values_from_action(action, self.config.default_base_twist)
        msg.linear_velocity = float(base_twist[0])
        msg.angular_velocity = float(base_twist[1])
        msg.pose_selection = int(self.config.pose_selection)
        return msg

    def _build_joint_command_message(self, action: RobotAction) -> Any:
        if self._ros is None:
            raise RuntimeError("ROS runtime is not initialized.")
        msg = self._ros.JointState()
        msg.header = self._ros.Header(stamp=self._ros.rospy.Time.now(), frame_id="")
        msg.name = [self._infer_ros_joint_name(joint_name) for joint_name in self._joint_command_names]
        msg.position = [
            float(action.get(joint_name, self._joint_positions.get(joint_name, 0.0)))
            for joint_name in self._joint_command_names
        ]
        return msg

    @staticmethod
    def _format_named_values(names: tuple[str, ...] | list[str], values: dict[str, float]) -> str:
        return ", ".join(f"{name}={float(values[name]):.4f}" for name in names if name in values)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        with self._state_lock:
            obs: dict[str, Any] = dict(self._pose_observation)
            joint_positions = dict(self._joint_positions)

        for joint_name in self._joint_names:
            obs[joint_name] = float(joint_positions.get(joint_name, 0.0))

        obs["left_arm_present"] = 1.0
        obs["right_arm_present"] = 1.0

        for handle in self._end_effector_handles.values():
            obs.update(handle.observation())

        for handle in self._sensor_handles.values():
            obs.update(handle.observation())

        for camera_name, camera in self.cameras.items():
            frame = camera.read_latest()
            obs.update(split_camera_observation(camera_name, frame))
            depth_key = f"{camera_name}_depth"
            if depth_key not in obs and getattr(self.config.cameras[camera_name], "use_depth", False):
                depth_frame = self._read_depth_frame(camera)
                if depth_frame is not None:
                    obs[depth_key] = depth_frame

        if self.config.use_pi0_observation_state:
            obs = filter_observation_state(
                obs,
                PI0_OBSERVATION_STATE_NAMES,
                camera_feature_dict(self.config.cameras).keys(),
            )

        return obs

    def _read_depth_frame(self, camera: Any) -> Any:
        if hasattr(camera, "frame_lock") and hasattr(camera, "latest_depth_frame"):
            with camera.frame_lock:
                depth_frame = None if camera.latest_depth_frame is None else camera.latest_depth_frame.copy()
            return expand_depth_frame(depth_frame)

        if hasattr(camera, "read_depth"):
            return expand_depth_frame(camera.read_depth())

        return None

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        command = dict(self._last_action)
        command.update(action)
        clamped_end_effector_values: dict[str, list[float]] = {}
        for name, handle in self._end_effector_handles.items():
            raw_values = end_effector_values_from_action(command, name, handle.config)
            clamped_values = clamp_end_effector_values(raw_values, handle.config)
            clamped_end_effector_values[name] = clamped_values
            for feature_name, value in zip(end_effector_feature_names(name, handle.config), clamped_values, strict=True):
                command[feature_name] = float(value)

        if self.config.command_mode == "pose":
            if self._wbc_goal_publisher is None:
                raise RuntimeError("WBC goal publisher is not initialized.")
            print(
                "[spark3-send-action:pose] left=(%.4f, %.4f, %.4f) right=(%.4f, %.4f, %.4f) gripper=(%.4f, %.4f)"
                % (
                    command.get("main_left_x", float("nan")),
                    command.get("main_left_y", float("nan")),
                    command.get("main_left_z", float("nan")),
                    command.get("main_right_x", float("nan")),
                    command.get("main_right_y", float("nan")),
                    command.get("main_right_z", float("nan")),
                    command.get("main_gripper_left", float("nan")),
                    command.get("main_gripper_right", float("nan")),
                ),
                flush=True,
            )
            self._wbc_goal_publisher.publish(self._build_command_message(command))
        else:
            if self._joint_command_publisher is None:
                raise RuntimeError("Joint command publisher is not initialized.")
            action_debug_values = {
                joint_name: float(command.get(joint_name, self._joint_positions.get(joint_name, 0.0)))
                for joint_name in self._action_joint_names
            }
            joint_command_msg = self._build_joint_command_message(command)
            joint_command_debug_values = {
                ros_joint_name: float(position)
                for ros_joint_name, position in zip(joint_command_msg.name, joint_command_msg.position, strict=True)
            }
            print(
                "[spark3-send-action:joint:action] %s"
                % self._format_named_values(self._action_joint_names, action_debug_values),
                flush=True,
            )
            print(
                "[spark3-send-action:joint:command] %s"
                % self._format_named_values(joint_command_msg.name, joint_command_debug_values),
                flush=True,
            )
            self._joint_command_publisher.publish(joint_command_msg)

        for name, handle in self._end_effector_handles.items():
            handle.publish(clamped_end_effector_values[name])

        if self._end_effector_handles:
            gripper_debug = []
            for handle in self._end_effector_handles.values():
                snapshot = handle.debug_snapshot()
                if len(snapshot["command_values"]) == 1 and len(snapshot["state_values"]) == 1:
                    gripper_debug.append(
                        "%s topic=%s cmd=%.4f state=%.4f"
                        % (
                            snapshot["name"],
                            snapshot["command_topic"],
                            float(snapshot["command_values"][0]),
                            float(snapshot["state_values"][0]),
                        )
                    )
                else:
                    gripper_debug.append(
                        "%s topic=%s cmd=%s state=%s"
                        % (
                            snapshot["name"],
                            snapshot["command_topic"],
                            snapshot["command_values"],
                            snapshot["state_values"],
                        )
                    )
            print("[spark3-send-action:gripper] %s" % " | ".join(gripper_debug), flush=True)

        self._last_action = command
        return command

    @check_if_not_connected
    def disconnect(self) -> None:
        self._cleanup_resources()
        logger.info("%s disconnected.", self)

    def _cleanup_resources(self) -> None:
        for handle in self._end_effector_handles.values():
            handle.disconnect()
        for handle in self._sensor_handles.values():
            handle.disconnect()
        self._end_effector_handles = {}
        self._sensor_handles = {}

        if self._pose_subscriber is not None and hasattr(self._pose_subscriber, "unregister"):
            self._pose_subscriber.unregister()
        if self._joint_subscriber is not None and hasattr(self._joint_subscriber, "unregister"):
            self._joint_subscriber.unregister()
        if self._wbc_goal_publisher is not None and hasattr(self._wbc_goal_publisher, "unregister"):
            self._wbc_goal_publisher.unregister()
        if self._joint_command_publisher is not None and hasattr(self._joint_command_publisher, "unregister"):
            self._joint_command_publisher.unregister()

        for camera in self.cameras.values():
            camera.disconnect()

        self._pose_subscriber = None
        self._joint_subscriber = None
        self._wbc_goal_publisher = None
        self._joint_command_publisher = None
        self._state_ready.clear()
        self._joint_ready.clear()
        self._is_connected = False
        self._ros = None
