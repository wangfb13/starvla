#!/usr/bin/env python

from __future__ import annotations

import logging
import time
from functools import cached_property
from threading import Event, Lock
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.spark3_pose_gripper.ros_utils import ensure_ros_node, load_ros_runtime
from lerobot.robots.spark3_pose_gripper.schema import (
    action_feature_dict,
    default_action_dict,
    parse_goal_pose_message,
    pose_dict,
    pose_values_flange_to_tcp,
)

from .config_spark3_pose_gripper import Spark3PoseGripperTeleoperatorConfig

logger = logging.getLogger(__name__)


class Spark3PoseGripperTeleoperator(Teleoperator):
    config_class = Spark3PoseGripperTeleoperatorConfig
    name = "spark3_pose_gripper"

    def __init__(self, config: Spark3PoseGripperTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._latest_action = default_action_dict(
            config.end_effectors,
            left_arm_pose=config.default_left_arm_pose,
            right_arm_pose=config.default_right_arm_pose,
            head_pose=config.default_head_pose,
            hip_pose=config.default_hip_pose,
            base_twist=config.default_base_twist,
        )
        self._latest_action.update(
            pose_dict(
                "main_left",
                pose_values_flange_to_tcp(
                    config.default_left_arm_pose,
                    config.left_flange_to_tcp_transform,
                ),
            )
        )
        self._latest_action.update(
            pose_dict(
                "main_right",
                pose_values_flange_to_tcp(
                    config.default_right_arm_pose,
                    config.right_flange_to_tcp_transform,
                ),
            )
        )
        self._action_lock = Lock()
        self._action_ready = Event()
        self._is_connected = False
        self._ros = None
        self._subscriber = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return action_feature_dict(self.config.end_effectors)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

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
            self._subscriber = self._ros.rospy.Subscriber(
                self.config.teleop_topic,
                self._ros.WbcPoseEe,
                self._handle_action,
            )
            self._wait_until_ready()
            self._is_connected = True
            logger.info("%s connected.", self)
        except Exception:
            self._cleanup_resources()
            raise

    def _wait_until_ready(self) -> None:
        if self._action_ready.wait(timeout=self.config.connect_timeout_s):
            return
        raise TimeoutError(f"Timed out waiting for teleop topic {self.config.teleop_topic}.")

    def _handle_action(self, msg: Any) -> None:
        action = parse_goal_pose_message(
            msg,
            self.config.end_effectors,
            default_head_pose=self.config.default_head_pose,
            default_hip_pose=self.config.default_hip_pose,
            default_base_twist=self.config.default_base_twist,
            left_flange_to_tcp_transform=self.config.left_flange_to_tcp_transform,
            right_flange_to_tcp_transform=self.config.right_flange_to_tcp_transform,
        )
        with self._action_lock:
            self._latest_action = action
        self._action_ready.set()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        with self._action_lock:
            return dict(self._latest_action)

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    @check_if_not_connected
    def disconnect(self) -> None:
        self._cleanup_resources()
        logger.info("%s disconnected.", self)

    def _cleanup_resources(self) -> None:
        if self._subscriber is not None and hasattr(self._subscriber, "unregister"):
            self._subscriber.unregister()
        self._subscriber = None
        self._action_ready.clear()
        self._is_connected = False
        self._ros = None
