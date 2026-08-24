#!/usr/bin/env python3
"""Convert a LeRobot UMI dataset into an action-only Spark3 replay dataset.

The input action is expected to contain, for both arms, a gripper angle followed
by a QR-frame TCP pose represented as xyz + XYZW quaternion. The output action
contains base_link-frame absolute TCP poses and [0, 1] Spark3 gripper commands.

The calibration matrix is interpreted as ``base_link_from_qr`` by default:

    T_base_tcp = T_base_qr @ T_qr_tcp

This tool never connects to a robot and never sends commands.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTION = "action"
OUTPUT_ACTION_NAMES = [
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
    "main_gripper_left",
    "main_gripper_right",
]

INPUT_NAME_CANDIDATES = {
    "left_gripper": ("left_position_rad", "gripper_left", "left_gripper", "main_gripper_left"),
    "left_x": ("left_x", "main_left_x"),
    "left_y": ("left_y", "main_left_y"),
    "left_z": ("left_z", "main_left_z"),
    "left_qx": ("left_quat_x", "left_qx", "main_left_qx"),
    "left_qy": ("left_quat_y", "left_qy", "main_left_qy"),
    "left_qz": ("left_quat_z", "left_qz", "main_left_qz"),
    "left_qw": ("left_quat_w", "left_qw", "main_left_w"),
    "right_gripper": (
        "right_position_rad",
        "gripper_right",
        "right_gripper",
        "main_gripper_right",
    ),
    "right_x": ("right_x", "main_right_x"),
    "right_y": ("right_y", "main_right_y"),
    "right_z": ("right_z", "main_right_z"),
    "right_qx": ("right_quat_x", "right_qx", "main_right_qx"),
    "right_qy": ("right_quat_y", "right_qy", "main_right_qy"),
    "right_qz": ("right_quat_z", "right_qz", "main_right_qz"),
    "right_qw": ("right_quat_w", "right_qw", "main_right_w"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert QR-frame UMI TCP actions to a Spark3 base_link absolute replay dataset."
    )
    parser.add_argument("dataset_root", type=Path, help="Input LeRobot UMI dataset root.")
    parser.add_argument(
        "matrix_path",
        type=Path,
        help="4x4 base_link_from_qr calibration matrix (.json/.yaml/.npy/.txt/.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root. Default: <dataset_root>_spark3_replay.",
    )
    parser.add_argument(
        "--invert-matrix",
        action="store_true",
        help="Invert the supplied matrix (use when the file stores qr_from_base_link).",
    )
    parser.add_argument("--left-umi-open", type=float, default=-0.436)
    parser.add_argument("--left-umi-closed", type=float, default=0.0)
    parser.add_argument("--right-umi-open", type=float, default=-0.436)
    parser.add_argument("--right-umi-closed", type=float, default=0.0)
    parser.add_argument("--robot-open", type=float, default=1.0)
    parser.add_argument("--robot-closed", type=float, default=0.0)
    parser.add_argument(
        "--max-translation-step-m",
        type=float,
        default=0.05,
        help="Fail if consecutive TCP positions jump farther than this within an episode; <=0 disables.",
    )
    parser.add_argument(
        "--max-rotation-step-deg",
        type=float,
        default=30.0,
        help="Fail if consecutive TCP rotations jump farther than this within an episode; <=0 disables.",
    )
    return parser.parse_args()


def _extract_matrix_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in (
            "base_from_qr",
            "base_link_from_qr",
            "T_base_qr",
            "transform_matrix",
            "matrix",
        ):
            if key in payload:
                return payload[key]
        raise ValueError(
            "Matrix mapping must contain one of: base_from_qr, base_link_from_qr, "
            "T_base_qr, transform_matrix, matrix."
        )
    return payload


def load_transform(path: Path, *, invert: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration matrix file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif suffix == ".json":
        matrix = np.asarray(_extract_matrix_payload(json.loads(path.read_text())), dtype=np.float64)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML calibration files.") from exc
        matrix = np.asarray(_extract_matrix_payload(yaml.safe_load(path.read_text())), dtype=np.float64)
    else:
        delimiter = "," if suffix == ".csv" else None
        matrix = np.loadtxt(path, dtype=np.float64, delimiter=delimiter)

    if matrix.shape != (4, 4):
        raise ValueError(f"Calibration matrix must have shape (4, 4), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Calibration matrix contains NaN or infinity.")
    if invert:
        matrix = np.linalg.inv(matrix)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"Invalid homogeneous-transform last row: {matrix[3].tolist()}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("Calibration rotation is not orthonormal within tolerance 2e-3.")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=2e-3):
        raise ValueError(f"Calibration rotation determinant must be +1, got {determinant:.8f}.")
    return matrix


def resolve_input_indices(action_names: list[str]) -> dict[str, int]:
    normalized = {str(name).strip().lower(): idx for idx, name in enumerate(action_names)}
    result: dict[str, int] = {}
    for target, candidates in INPUT_NAME_CANDIDATES.items():
        for candidate in candidates:
            if candidate in normalized:
                result[target] = normalized[candidate]
                break
        else:
            raise ValueError(
                f"Input action is missing '{target}'. Accepted names={list(candidates)}; "
                f"available names={action_names}."
            )
    return result


def map_gripper(
    raw: np.ndarray,
    *,
    umi_open: float,
    umi_closed: float,
    robot_open: float,
    robot_closed: float,
) -> np.ndarray:
    denominator = umi_open - umi_closed
    if abs(denominator) < 1e-12:
        raise ValueError("UMI gripper open and closed endpoints must differ.")
    fraction = (raw - umi_closed) / denominator
    mapped = robot_closed + np.clip(fraction, 0.0, 1.0) * (robot_open - robot_closed)
    return np.clip(mapped, min(robot_closed, robot_open), max(robot_closed, robot_open))


def _continuous_quaternions(quaternions: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    output = quaternions.copy()
    previous_by_episode: dict[int, np.ndarray] = {}
    for idx, episode in enumerate(episodes.tolist()):
        previous = previous_by_episode.get(int(episode))
        if previous is not None and float(np.dot(previous, output[idx])) < 0.0:
            output[idx] *= -1.0
        previous_by_episode[int(episode)] = output[idx].copy()
    return output


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized XYZW quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def multiply_quaternions_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for one left XYZW quaternion and an (N, 4) right array."""
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64).T
    output = np.column_stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )
    return output / np.linalg.norm(output, axis=1, keepdims=True)


def transform_actions(
    actions: np.ndarray,
    episodes: np.ndarray,
    indices: dict[str, int],
    base_from_qr: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action array, got shape={actions.shape}.")

    base_quaternion = matrix_to_quaternion_xyzw(base_from_qr[:3, :3])
    output = np.empty((actions.shape[0], len(OUTPUT_ACTION_NAMES)), dtype=np.float32)

    for arm, output_start in (("left", 0), ("right", 7)):
        positions_qr = actions[:, [indices[f"{arm}_x"], indices[f"{arm}_y"], indices[f"{arm}_z"]]]
        quaternions_qr = actions[
            :,
            [indices[f"{arm}_qx"], indices[f"{arm}_qy"], indices[f"{arm}_qz"], indices[f"{arm}_qw"]],
        ]
        norms = np.linalg.norm(quaternions_qr, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
            bad = int(np.flatnonzero((~np.isfinite(norms)) | (norms < 1e-8))[0])
            raise ValueError(f"Invalid {arm} quaternion at local row {bad}: {quaternions_qr[bad].tolist()}")
        quaternions_qr = quaternions_qr / norms[:, None]

        positions_base = positions_qr @ base_from_qr[:3, :3].T + base_from_qr[:3, 3]
        quaternions_base = multiply_quaternions_xyzw(base_quaternion, quaternions_qr)
        quaternions_base = _continuous_quaternions(quaternions_base, episodes)
        output[:, output_start : output_start + 3] = positions_base
        output[:, output_start + 3 : output_start + 7] = quaternions_base

    output[:, 14] = map_gripper(
        actions[:, indices["left_gripper"]],
        umi_open=args.left_umi_open,
        umi_closed=args.left_umi_closed,
        robot_open=args.robot_open,
        robot_closed=args.robot_closed,
    )
    output[:, 15] = map_gripper(
        actions[:, indices["right_gripper"]],
        umi_open=args.right_umi_open,
        umi_closed=args.right_umi_closed,
        robot_open=args.robot_open,
        robot_closed=args.robot_closed,
    )
    return output


def validate_steps(
    actions: np.ndarray,
    episodes: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_deg: float,
) -> None:
    if len(actions) < 2:
        return
    same_episode = episodes[1:] == episodes[:-1]
    for arm, start in (("left", 0), ("right", 7)):
        translation_steps = np.linalg.norm(np.diff(actions[:, start : start + 3], axis=0), axis=1)
        if max_translation_m > 0 and np.any(same_episode & (translation_steps > max_translation_m)):
            row = int(np.flatnonzero(same_episode & (translation_steps > max_translation_m))[0] + 1)
            raise ValueError(
                f"{arm} TCP translation jump at output row {row}: "
                f"{translation_steps[row - 1]:.6f} m > {max_translation_m:.6f} m."
            )
        if max_rotation_deg > 0:
            quaternions = actions[:, start + 3 : start + 7].astype(np.float64)
            dots = np.sum(quaternions[:-1] * quaternions[1:], axis=1)
            rotation_steps = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)) * (180.0 / np.pi)
            if np.any(same_episode & (rotation_steps > max_rotation_deg)):
                row = int(np.flatnonzero(same_episode & (rotation_steps > max_rotation_deg))[0] + 1)
                raise ValueError(
                    f"{arm} TCP rotation jump at output row {row}: "
                    f"{rotation_steps[row - 1]:.6f} deg > {max_rotation_deg:.6f} deg."
                )


def action_stats(actions: np.ndarray) -> dict[str, list[float] | list[int]]:
    quantiles = np.quantile(actions.astype(np.float64), [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": actions.min(axis=0).astype(float).tolist(),
        "max": actions.max(axis=0).astype(float).tolist(),
        "mean": actions.astype(np.float64).mean(axis=0).tolist(),
        "std": actions.astype(np.float64).std(axis=0).tolist(),
        "count": [int(actions.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def replace_action_column(table: pa.Table, actions: np.ndarray) -> pa.Table:
    flat = pa.array(actions.reshape(-1), type=pa.float32())
    action_array = pa.FixedSizeListArray.from_arrays(flat, len(OUTPUT_ACTION_NAMES))
    action_index = table.schema.get_field_index(ACTION)
    if action_index < 0:
        raise ValueError("Parquet file is missing its action column.")
    field = pa.field(ACTION, pa.list_(pa.float32(), len(OUTPUT_ACTION_NAMES)))
    return table.set_column(action_index, field, action_array).replace_schema_metadata(None)


def convert_dataset(args: argparse.Namespace) -> Path:
    source = args.dataset_root.resolve()
    info_path = source / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Input is not a LeRobot dataset; missing {info_path}")
    output = (
        args.output_root.resolve()
        if args.output_root is not None
        else source.with_name(f"{source.name}_spark3_replay")
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex[:8]}")

    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Only LeRobot v3.0 is supported, got {info.get('codebase_version')!r}.")
    input_action_feature = info.get("features", {}).get(ACTION)
    if not input_action_feature or not input_action_feature.get("names"):
        raise ValueError("Input meta/info.json must define action feature names.")
    input_indices = resolve_input_indices(list(input_action_feature["names"]))
    base_from_qr = load_transform(args.matrix_path.resolve(), invert=args.invert_matrix)

    all_actions: list[np.ndarray] = []
    parquet_feature_names: set[str] = set()
    try:
        (temporary / "data").mkdir(parents=True)
        (temporary / "meta").mkdir(parents=True)
        parquet_paths = sorted((source / "data").glob("**/*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet data files found under {source / 'data'}")

        for parquet_path in parquet_paths:
            table = pq.read_table(parquet_path)
            parquet_feature_names.update(table.column_names)
            raw_actions = np.asarray(table[ACTION].to_pylist(), dtype=np.float64)
            episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
            converted = transform_actions(raw_actions, episodes, input_indices, base_from_qr, args)
            validate_steps(
                converted,
                episodes,
                max_translation_m=args.max_translation_step_m,
                max_rotation_deg=args.max_rotation_step_deg,
            )
            destination = temporary / parquet_path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(replace_action_column(table, converted), destination, compression="zstd")
            all_actions.append(converted)
            print(f"[converted] {parquet_path.relative_to(source)} rows={len(converted)}")

        converted_actions = np.concatenate(all_actions, axis=0)
        for metadata_name in ("tasks.parquet",):
            metadata_source = source / "meta" / metadata_name
            if metadata_source.exists():
                shutil.copy2(metadata_source, temporary / "meta" / metadata_name)
        episodes_source = source / "meta/episodes"
        if episodes_source.exists():
            shutil.copytree(episodes_source, temporary / "meta/episodes")

        output_info = dict(info)
        output_info["robot_type"] = "spark3_pose_gripper_replay"
        output_info["features"] = {
            key: value
            for key, value in info["features"].items()
            if key in parquet_feature_names
        }
        output_info["features"][ACTION] = {
            "dtype": "float32",
            "shape": [len(OUTPUT_ACTION_NAMES)],
            "names": OUTPUT_ACTION_NAMES,
        }
        (temporary / "meta/info.json").write_text(json.dumps(output_info, indent=2) + "\n")

        source_stats_path = source / "meta/stats.json"
        stats = json.loads(source_stats_path.read_text()) if source_stats_path.exists() else {}
        stats[ACTION] = action_stats(converted_actions)
        for feature_name in list(stats):
            if feature_name not in output_info["features"]:
                stats.pop(feature_name, None)
        (temporary / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")

        report = {
            "source_dataset": str(source),
            "base_from_qr": base_from_qr.tolist(),
            "matrix_was_inverted": bool(args.invert_matrix),
            "input_action_names": input_action_feature["names"],
            "output_action_names": OUTPUT_ACTION_NAMES,
            "gripper_mapping": {
                "left": {
                    "umi_open": args.left_umi_open,
                    "umi_closed": args.left_umi_closed,
                },
                "right": {
                    "umi_open": args.right_umi_open,
                    "umi_closed": args.right_umi_closed,
                },
                "robot_open": args.robot_open,
                "robot_closed": args.robot_closed,
                "clipped": True,
            },
            "frames": int(converted_actions.shape[0]),
            "episodes": int(info["total_episodes"]),
            "fps": int(info["fps"]),
            "videos_copied": False,
        }
        (temporary / "meta/replay_conversion.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"[PASS] replay dataset written to {output}")
    print(f"[info] episodes are preserved; select one with --dataset.episode=N (0-{info['total_episodes'] - 1})")
    print("[safety] This conversion does not move the robot. Validate the first pose and pre-position safely.")
    return output


def main() -> None:
    args = parse_args()
    try:
        convert_dataset(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
