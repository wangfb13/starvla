#!/usr/bin/env python

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

_ROS_NODE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Spark3RosRuntime:
    rospy: Any
    Float64: type
    Float64MultiArray: type
    UInt16: type
    UInt16MultiArray: type
    Header: type
    Point: type
    Quaternion: type
    PoseStamped: type
    WbcPoseEe: type
    WbcPoseExpand: type
    WbcState: type


def load_ros_runtime() -> Spark3RosRuntime:
    try:
        import rospy
        from geometry_msgs.msg import Point, PoseStamped, Quaternion
        from sensor_msgs.msg import JointState  # noqa: F401
        from std_msgs.msg import Float64, Float64MultiArray, Header, UInt16, UInt16MultiArray
        from wbc_msgs.msg import wbc_pose_ee, wbc_pose_expand, wbc_state
    except ImportError as e:
        raise ImportError(
            "spark3_pose_gripper requires a ROS1 Python environment with "
            "`rospy`, `geometry_msgs`, `sensor_msgs`, `std_msgs`, and `wbc_msgs`."
        ) from e

    return Spark3RosRuntime(
        rospy=rospy,
        Float64=Float64,
        Float64MultiArray=Float64MultiArray,
        UInt16=UInt16,
        UInt16MultiArray=UInt16MultiArray,
        Header=Header,
        Point=Point,
        Quaternion=Quaternion,
        PoseStamped=PoseStamped,
        WbcPoseEe=wbc_pose_ee,
        WbcPoseExpand=wbc_pose_expand,
        WbcState=wbc_state,
    )


def ensure_ros_node(runtime: Spark3RosRuntime, node_name: str) -> None:
    with _ROS_NODE_LOCK:
        is_initialized = getattr(getattr(runtime.rospy, "core", None), "is_initialized", lambda: False)
        if is_initialized():
            return
        runtime.rospy.init_node(node_name, anonymous=True, disable_signals=True)
