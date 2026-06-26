#!/usr/bin/env python3

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path
from typing import Any

import cv2  # type: ignore

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render policy inference trace markers onto recorded episode videos."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Dataset repo id. Defaults to the dataset root directory name.",
    )
    parser.add_argument("--episode-index", type=int, required=True, help="Zero-based episode index.")
    parser.add_argument(
        "--video-key",
        default=None,
        help="Optional single video key to render. If omitted, render all dataset video keys.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Optional explicit trace json path. Defaults to meta/inference_traces/episode-xxxxxx.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to meta/inference_traces/rendered.",
    )
    return parser.parse_args()


def _default_trace_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "meta" / "inference_traces" / f"episode-{episode_index:06d}.json"


def _normalize_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_overlay_lines(event: dict[str, Any]) -> list[str]:
    lines = [
        f"INF frame={event.get('recorded_frame_index', '?')}",
        f"latency={event.get('inference_latency_ms', 0.0):.1f} ms  t={event.get('inference_start_s', 0.0):.3f}s",
    ]

    raw_observation = event.get("raw_observation", {})
    if isinstance(raw_observation, dict):
        scalar_items: list[str] = []
        for key, value in raw_observation.items():
            if isinstance(value, (int, float, bool, str)):
                scalar_items.append(f"{key}={_normalize_scalar(value)}")
            if len(scalar_items) >= 6:
                break
        if scalar_items:
            lines.append("obs: " + " | ".join(scalar_items[:3]))
            if len(scalar_items) > 3:
                lines.append("obs+: " + " | ".join(scalar_items[3:6]))

    policy_output = event.get("policy_output")
    if isinstance(policy_output, dict):
        scalar_items = []
        for key, value in policy_output.items():
            if isinstance(value, (int, float, bool, str)):
                scalar_items.append(f"{key}={_normalize_scalar(value)}")
            if len(scalar_items) >= 4:
                break
        if scalar_items:
            lines.append("act: " + " | ".join(scalar_items))

    image_keys = event.get("image_keys")
    if isinstance(image_keys, list) and image_keys:
        lines.append("img: " + ", ".join(str(key) for key in image_keys[:2]))

    return lines


def _draw_overlay(frame, event: dict[str, Any]) -> Any:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (0, 255, 0), 4)

    lines = _build_overlay_lines(event)
    overlay = frame.copy()
    panel_height = 34 + 28 * len(lines)
    cv2.rectangle(overlay, (12, 12), (min(width - 12, 1100), min(height - 12, panel_height)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    y = 40
    for line in lines:
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28
    return frame


def _render_video(
    video_path: Path,
    output_path: Path,
    events_by_frame: dict[int, dict[str, Any]],
    fps_fallback: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video: {output_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            event = events_by_frame.get(frame_idx)
            if event is not None:
                frame = _draw_overlay(frame, event)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    dataset_repo_id = args.dataset_repo_id or dataset_root.name
    trace_path = args.trace_path or _default_trace_path(dataset_root, args.episode_index)

    if not trace_path.exists():
        raise FileNotFoundError(f"Inference trace not found: {trace_path}")

    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
        download_videos=False,
    )

    with trace_path.open("r", encoding="utf-8") as f:
        trace_payload = json.load(f)

    trace_events = trace_payload.get("events", [])
    events_by_frame = {
        int(event["recorded_frame_index"]): event
        for event in trace_events
        if "recorded_frame_index" in event
    }
    if not events_by_frame:
        raise ValueError(f"No inference events found in trace: {trace_path}")

    output_dir = args.output_dir or dataset_root / "meta" / "inference_traces" / "rendered"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_keys = [args.video_key] if args.video_key else list(dataset.meta.video_keys)
    for video_key in video_keys:
        video_path = dataset.root / dataset.meta.get_video_file_path(args.episode_index, video_key)
        safe_video_key = video_key.replace("/", "__")
        output_path = output_dir / f"episode-{args.episode_index:06d}.{safe_video_key}.mp4"
        _render_video(video_path, output_path, events_by_frame, float(trace_payload.get("fps", dataset.fps)))
        print(f"[rendered] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
