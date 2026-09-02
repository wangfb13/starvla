#!/usr/bin/env python3
"""Run lerobot_record with QR-world <-> robot-base TCP pose conversion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

POSE_SUFFIXES = ("x", "y", "z", "qx", "qy", "qz", "w")


def load_transform(path: str) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    matrix = payload.get("T_base_from_qr") if isinstance(payload, dict) else payload
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Calibration must contain a finite 4x4 T_base_from_qr matrix.")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("T_base_from_qr last row must be [0, 0, 0, 1].")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("T_base_from_qr rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("T_base_from_qr rotation determinant must be +1.")
    return transform


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Invalid zero/non-finite TCP quaternion.")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                      (m[0, 2] - m[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                      (m[1, 0] - m[0, 1]) / s])
    return q / np.linalg.norm(q)


def read_pose(values: dict[str, Any], arm: str) -> np.ndarray | None:
    keys = [f"main_{arm}_{suffix}" for suffix in POSE_SUFFIXES]
    if not all(key in values for key in keys):
        return None
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [float(values[key]) for key in keys[:3]]
    pose[:3, :3] = quaternion_xyzw_to_matrix(np.array([float(values[key]) for key in keys[3:]]))
    return pose


def write_pose(values: dict[str, Any], arm: str, pose: np.ndarray) -> None:
    xyz = pose[:3, 3]
    quat = matrix_to_quaternion_xyzw(pose[:3, :3])
    for suffix, value in zip(POSE_SUFFIXES, np.concatenate((xyz, quat)), strict=True):
        values[f"main_{arm}_{suffix}"] = float(value)


def transform_arms(values: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    result = dict(values)
    for arm in ("left", "right"):
        pose = read_pose(result, arm)
        if pose is not None:
            write_pose(result, arm, transform @ pose)
    return result


def install_frame_adapter(t_base_from_qr: np.ndarray) -> None:
    from lerobot.async_inference import helpers
    from lerobot.utils import control_utils

    t_qr_from_base = np.linalg.inv(t_base_from_qr)
    original_build = helpers._build_spark3_online_pi05_state_and_masks
    original_convert = control_utils._maybe_convert_pi05_canonical_action_to_robot_action

    def build_qr_state(raw_observation, *args, **kwargs):
        return original_build(transform_arms(raw_observation, t_qr_from_base), *args, **kwargs)

    def convert_qr_action(*args, **kwargs):
        action = original_convert(*args, **kwargs)
        if not isinstance(action, dict):
            raise RuntimeError("Expected canonical StarVLA action to become a robot action dict.")
        return transform_arms(action, t_base_from_qr)

    helpers._build_spark3_online_pi05_state_and_masks = build_qr_state
    control_utils._maybe_convert_pi05_canonical_action_to_robot_action = convert_qr_action


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qr-calibration", required=True)
    args, remaining = parser.parse_known_args()
    transform = load_transform(args.qr_calibration)
    print(f"[absolute-qr] loaded T_base_from_qr from {args.qr_calibration}", flush=True)
    print(transform, flush=True)
    install_frame_adapter(transform)
    sys.argv = [sys.argv[0], *remaining]
    from lerobot.scripts.lerobot_record import main as record_main
    record_main()


if __name__ == "__main__":
    main()
