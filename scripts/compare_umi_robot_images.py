#!/usr/bin/env python3
"""Extract matched UMI/robot wrist-camera grids and basic domain statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig  # noqa: F401


CAMERAS = (
    "observation.images.cam_left_undist",
    "observation.images.cam_right_undist",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--umi-root", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--relative-stats", type=Path, required=True)
    p.add_argument("--robot-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=20)
    return p.parse_args()


def image_uint8(value: torch.Tensor) -> np.ndarray:
    x = value.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)
    array = x.numpy()
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)[..., :3]


def resize_rgb(image: np.ndarray, size: int = 224) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def stats(images: list[np.ndarray]) -> dict:
    means, stds, brightness, saturation, sharpness = [], [], [], [], []
    for rgb in images:
        rgb = resize_rgb(rgb)
        f = rgb.astype(np.float32) / 255.0
        means.append(f.mean((0, 1)))
        stds.append(f.std((0, 1)))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        brightness.append(float(hsv[..., 2].mean() / 255.0))
        saturation.append(float(hsv[..., 1].mean() / 255.0))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return {
        "rgb_mean": np.mean(means, axis=0).tolist(),
        "rgb_std_within_image": np.mean(stds, axis=0).tolist(),
        "brightness_mean": float(np.mean(brightness)),
        "brightness_std_across_images": float(np.std(brightness)),
        "saturation_mean": float(np.mean(saturation)),
        "sharpness_laplacian_mean": float(np.mean(sharpness)),
        "sharpness_laplacian_median": float(np.median(sharpness)),
    }


def annotate(rgb: np.ndarray, label: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(bgr, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def montage(images: list[np.ndarray], labels: list[str], columns: int = 5) -> np.ndarray:
    tiles = [annotate(resize_rgb(image), label) for image, label in zip(images, labels, strict=True)]
    rows = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * columns - len(tiles)))
    return np.concatenate(
        [np.concatenate(tiles[row * columns:(row + 1) * columns], axis=1) for row in range(rows)],
        axis=0,
    )


def decode_selected(path: Path, selected: set[int]) -> dict[int, np.ndarray]:
    result = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected:
                result[index] = frame.to_ndarray(format="rgb24")
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_info = args.output_dir / "info_wrist_undist_offline.json"
    info = json.loads((args.umi_root / "meta/info.json").read_text())
    scalar = {"observation.state", "action", "score", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    info["features"] = {k: v for k, v in info["features"].items() if k in set(CAMERAS) | scalar}
    filtered_info.write_text(json.dumps(info, indent=2), encoding="utf-8")

    cfg = TrainPipelineConfig.from_pretrained(args.checkpoint / "train_config.json")
    cfg.dataset.root = str(args.umi_root)
    cfg.dataset.info_path = str(filtered_info)
    cfg.dataset.sidecar_path = str(args.sidecar)
    cfg.dataset.relative_stats_path = str(args.relative_stats)
    cfg.dataset.video_backend = "pyav"
    cfg.dataset.episodes = None
    cfg.rename_map = {}
    cfg.num_workers = 0
    dataset = make_dataset(cfg)

    umi_indices = np.linspace(0, len(dataset) - 1, args.num_samples, dtype=np.int64)
    umi = {"left": [], "right": []}
    umi_labels = []
    for index in umi_indices:
        item = dataset[int(index)]
        umi["left"].append(image_uint8(item[CAMERAS[0]]))
        umi["right"].append(image_uint8(item[CAMERAS[1]]))
        umi_labels.append(f"ep{int(item['episode_index'])}/f{int(item['frame_index'])}")

    total_robot = int(json.loads((args.robot_root / "meta/info.json").read_text())["total_frames"])
    robot_indices = np.linspace(0, total_robot - 1, args.num_samples, dtype=np.int64)
    selected = set(int(x) for x in robot_indices)
    robot = {
        "left": decode_selected(args.robot_root / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4", selected),
        "right": decode_selected(args.robot_root / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4", selected),
    }
    robot_left = [robot["left"][int(i)] for i in robot_indices]
    robot_right = [robot["right"][int(i)] for i in robot_indices]
    robot_labels = [f"f{int(i)}" for i in robot_indices]

    groups = {
        "umi_left": umi["left"], "umi_right": umi["right"],
        "robot_left": robot_left, "robot_right": robot_right,
    }
    for name, images in groups.items():
        labels = umi_labels if name.startswith("umi") else robot_labels
        rgb_grid = montage(images, labels)
        cv2.imwrite(str(args.output_dir / f"{name}_grid.jpg"), cv2.cvtColor(rgb_grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])

    comparison_rows = []
    for idx in range(args.num_samples):
        row = np.concatenate([
            annotate(resize_rgb(umi["left"][idx]), f"UMI L {umi_labels[idx]}"),
            annotate(resize_rgb(umi["right"][idx]), f"UMI R {umi_labels[idx]}"),
            annotate(resize_rgb(robot_left[idx]), f"Robot L {robot_labels[idx]}"),
            annotate(resize_rgb(robot_right[idx]), f"Robot R {robot_labels[idx]}"),
        ], axis=1)
        comparison_rows.append(row)
    comparison = np.concatenate(comparison_rows, axis=0)
    cv2.imwrite(str(args.output_dir / "paired_comparison.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])

    result = {
        "umi_indices": umi_indices.tolist(),
        "robot_indices": robot_indices.tolist(),
        "image_statistics": {name: stats(images) for name, images in groups.items()},
    }
    (args.output_dir / "image_statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
