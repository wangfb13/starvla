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

import av  # type: ignore
import cv2  # type: ignore
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a rich inference-trace video with camera inputs and full state/action overlays."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Dataset repo id. Defaults to the dataset root directory name.",
    )
    parser.add_argument("--episode-index", type=int, required=True, help="Zero-based episode index.")
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Optional explicit trace json path. If omitted, auto-detects the episode trace.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output mp4 path. Defaults to meta/inference_traces/rich_render/episode-xxxxxx.rich.mp4.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on rendered frames for testing.",
    )
    parser.add_argument("--output-width", type=int, default=1920, help="Output video width.")
    parser.add_argument("--output-height", type=int, default=1080, help="Output video height.")
    return parser.parse_args()


def _resolve_trace_path(dataset_root: Path, episode_index: int, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    canonical = dataset_root / "meta" / "inference_traces" / f"episode-{episode_index:06d}.json"
    if canonical.exists():
        return canonical

    trace_dir = dataset_root / "meta" / "inference_traces"
    candidates = sorted(trace_dir.glob("episode*.json"))
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not determine trace path for episode {episode_index}. "
        f"Expected {canonical} or exactly one json in {trace_dir}."
    )


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _scalar_items(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, dict):
        return []

    items: list[tuple[str, str]] = []
    for key, item in value.items():
        if isinstance(item, (int, float, bool, str)):
            items.append((str(key), _format_scalar(item)))
        elif isinstance(item, list) and len(item) <= 6 and all(isinstance(v, (int, float, bool, str)) for v in item):
            items.append((str(key), "[" + ", ".join(_format_scalar(v) for v in item) + "]"))
        else:
            try:
                length = len(item)  # type: ignore[arg-type]
                items.append((str(key), f"<{type(item).__name__}:{length}>"))
            except TypeError:
                items.append((str(key), f"<{type(item).__name__}>"))
    return items


def _split_evenly(items: list[tuple[str, str]], num_groups: int) -> list[list[tuple[str, str]]]:
    groups = [[] for _ in range(num_groups)]
    for idx, item in enumerate(items):
        groups[idx % num_groups].append(item)
    return groups


def _draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (230, 235, 240),
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    src_height, src_width = frame.shape[:2]
    scale = min(target_width / src_width, target_height / src_height)
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


class SequentialVideoReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.container = av.open(str(video_path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self._decoder = self.container.decode(self.stream)
        self._next_index = 0

    def get_frame(self, target_index: int) -> np.ndarray:
        if target_index < self._next_index:
            raise ValueError(f"Cannot seek backwards in sequential reader: {target_index} < {self._next_index}")

        for frame in self._decoder:
            array = frame.to_ndarray(format="rgb24")
            current_index = self._next_index
            self._next_index += 1
            if current_index == target_index:
                return array

        raise IndexError(f"Frame {target_index} out of range for {self.video_path}")

    def close(self) -> None:
        self.container.close()


def _compose_canvas(
    *,
    event: dict[str, Any],
    frames_by_key: dict[str, np.ndarray],
    output_width: int,
    output_height: int,
) -> np.ndarray:
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    top_height = output_height // 2
    bottom_y = top_height
    bottom_height = output_height - top_height

    video_keys = list(frames_by_key.keys())
    cell_width = output_width // max(1, len(video_keys))

    for idx, video_key in enumerate(video_keys):
        cell_x = idx * cell_width
        resized = _fit_frame(frames_by_key[video_key], cell_width, top_height)
        y0 = (top_height - resized.shape[0]) // 2
        x0 = cell_x + (cell_width - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized[:, :, ::-1]
        cv2.rectangle(canvas, (cell_x + 8, 8), (cell_x + cell_width - 8, top_height - 8), (72, 196, 136), 3)
        label = video_key.replace("observation.images.", "")
        _draw_text(canvas, label, (cell_x + 22, 36), scale=0.68, color=(72, 196, 136), thickness=2)

    cv2.rectangle(canvas, (0, bottom_y), (output_width, output_height), (18, 24, 30), -1)
    cv2.line(canvas, (0, bottom_y), (output_width, bottom_y), (60, 70, 82), 2)

    frame_idx = event.get("recorded_frame_index", "?")
    latency_ms = float(event.get("inference_latency_ms", 0.0))
    loop_ts = float(event.get("loop_timestamp_s", 0.0))
    inf_start = float(event.get("inference_start_s", 0.0))
    inf_end = float(event.get("inference_end_s", 0.0))
    image_keys = event.get("image_keys", [])
    model_inputs = event.get("model_inputs", {})
    obs_state = model_inputs.get("observation.state") if isinstance(model_inputs, dict) else None
    obs_state_dim = len(obs_state) if isinstance(obs_state, list) else None

    _draw_text(
        canvas,
        f"frame={frame_idx}  latency={latency_ms:.2f}ms  loop_t={loop_ts:.3f}s  inf=[{inf_start:.3f}, {inf_end:.3f}]s",
        (24, bottom_y + 34),
        scale=0.64,
        color=(240, 245, 250),
        thickness=2,
    )
    _draw_text(
        canvas,
        "image_keys=" + ", ".join(str(key) for key in image_keys),
        (24, bottom_y + 64),
        scale=0.48,
        color=(180, 190, 200),
    )
    if obs_state_dim is not None:
        _draw_text(
            canvas,
            f"model_inputs.observation.state dim={obs_state_dim}",
            (24, bottom_y + 90),
            scale=0.48,
            color=(180, 190, 200),
        )

    raw_items = _scalar_items(event.get("raw_observation", {}))
    action_items = _scalar_items(event.get("policy_output", {}))
    raw_groups = _split_evenly(raw_items, 2)
    action_groups = _split_evenly(action_items, 2)

    section_top = bottom_y + 126
    section_bottom = output_height - 20
    line_height = 18
    max_rows = max(1, (section_bottom - section_top - 28) // line_height)
    col_width = output_width // 4

    section_specs = [
        ("Raw Obs A", raw_groups[0], 0, (110, 196, 255)),
        ("Raw Obs B", raw_groups[1], 1, (110, 196, 255)),
        ("Policy Out A", action_groups[0], 2, (255, 193, 94)),
        ("Policy Out B", action_groups[1], 3, (255, 193, 94)),
    ]

    for title, items, col_idx, color in section_specs:
        x0 = col_idx * col_width + 18
        x1 = (col_idx + 1) * col_width - 14
        _draw_text(canvas, title, (x0, section_top), scale=0.56, color=color, thickness=2)
        cv2.line(canvas, (x0, section_top + 8), (x1, section_top + 8), color, 1)
        y = section_top + 28
        for name, value in items[:max_rows]:
            line = f"{name}={value}"
            if len(line) > 38:
                line = line[:35] + "..."
            _draw_text(canvas, line, (x0, y), scale=0.42, color=(235, 238, 242))
            y += line_height

        hidden = max(0, len(items) - max_rows)
        if hidden > 0:
            _draw_text(canvas, f"... +{hidden} more", (x0, section_bottom), scale=0.42, color=(140, 150, 160))

    return canvas


def _render_trace_video(
    *,
    dataset: LeRobotDataset,
    trace_payload: dict[str, Any],
    output_path: Path,
    episode_index: int,
    output_width: int,
    output_height: int,
    max_frames: int | None,
) -> None:
    events = trace_payload.get("events", [])
    if not isinstance(events, list) or not events:
        raise ValueError("Trace payload does not contain any events.")

    if max_frames is not None:
        events = events[:max_frames]

    video_keys = list(dataset.meta.video_keys)
    video_paths = {
        key: dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        for key in video_keys
    }
    readers = {key: SequentialVideoReader(path) for key, path in video_paths.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(trace_payload.get("fps", dataset.fps))
    fps_rate = max(1, int(round(fps)))
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=fps_rate)
    stream.width = output_width
    stream.height = output_height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    try:
        for idx, event in enumerate(events, start=1):
            frame_index = int(event["recorded_frame_index"])
            frames_by_key = {key: readers[key].get_frame(frame_index) for key in video_keys}
            canvas = _compose_canvas(
                event=event,
                frames_by_key=frames_by_key,
                output_width=output_width,
                output_height=output_height,
            )
            video_frame = av.VideoFrame.from_ndarray(canvas[:, :, ::-1], format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

            if idx % 250 == 0:
                print(f"[render] {output_path.name}: {idx}/{len(events)} frames")

        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
        for reader in readers.values():
            reader.close()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    dataset_repo_id = args.dataset_repo_id or dataset_root.name
    trace_path = _resolve_trace_path(dataset_root, args.episode_index, args.trace_path)

    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=dataset_root,
        download_videos=False,
    )

    with trace_path.open("r", encoding="utf-8") as f:
        trace_payload = json.load(f)

    output_path = args.output_path
    if output_path is None:
        output_path = (
            dataset_root
            / "meta"
            / "inference_traces"
            / "rich_render"
            / f"episode-{args.episode_index:06d}.rich.mp4"
        )

    _render_trace_video(
        dataset=dataset,
        trace_payload=trace_payload,
        output_path=output_path,
        episode_index=args.episode_index,
        output_width=args.output_width,
        output_height=args.output_height,
        max_frames=args.max_frames,
    )
    print(f"[rendered] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
