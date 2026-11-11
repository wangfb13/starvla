#!/usr/bin/env python3
"""Prepare read-only DD caches and audited recovery action-training boundaries.

Example:
  python scripts/prepare_dd_recovery.py \
    --source recovery=/path/to/recovery --source normal1=/path/to/normal1 \
    --source normal2=/path/to/normal2 --sidecar recovery=/path/to/framewise.parquet \
    --sidecar normal1=/path/to/framewise.parquet --sidecar normal2=/path/to/framewise.parquet \
    --annotations scripts/dd_recovery_annotations/recovery_phase_annotations.csv \
    --output-root artifacts/dd_recovery_261021 --workers 4

No source videos, metadata, weights, or normalization statistics are modified.
No contact classification labels are used. The reviewed CSV defines ONLY action
supervision start/end boundaries. Caches preserve the original gel reference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_module_path = Path(__file__).resolve().parents[1] / "src/lerobot/datasets/dd_recovery.py"
_spec = importlib.util.spec_from_file_location("dd_recovery_preparation_helpers", _module_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
DD_VIDEO_KEYS = _helpers.DD_VIDEO_KEYS
decode_dd_rgb48 = _helpers.decode_dd_rgb48
grouped_episode_split = _helpers.grouped_episode_split
safe_anchor_frames = _helpers.safe_anchor_frames


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def assignments(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or name in result or not name:
            raise ValueError(f"Expected distinct name=path assignments, got {value!r}")
        result[name] = str(Path(path).resolve())
    return result


def source_video_path(root: Path, info: dict, episode: dict, key: str) -> Path:
    prefix = f"videos/{key}"
    path = root / info["video_path"].format(
        video_key=key, chunk_index=int(episode[f"{prefix}/chunk_index"]),
        file_index=int(episode[f"{prefix}/file_index"]), episode_index=int(episode["episode_index"]))
    if path.exists():
        return path
    # Current trainready metadata uses a common .mp4 template while the tactile
    # FFV1 payload uses .mkv. Resolve explicitly rather than episode=file guessing.
    tactile = path.with_suffix(".mkv")
    if tactile.exists():
        return tactile
    raise FileNotFoundError(f"No tactile video for {key}: {path} or {tactile}")


def decode_episode_cache(job: dict) -> dict:
    import av
    output = Path(job["output"])
    stamp = output.with_suffix(".json")
    signature = job["signature"]
    if output.exists() and stamp.exists():
        previous = json.loads(stamp.read_text())
        if previous.get("signature") == signature:
            arr = np.load(output, mmap_mode="r", allow_pickle=False)
            if (tuple(arr.shape) == (job["length"], 2, 3, 96, 128) and arr.dtype == np.float16
                    and previous.get("cache_sha256") == sha256(output)):
                return previous | {"reused": True}
        raise RuntimeError(f"Existing cache has a different source/signature: {output}; use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.npy")
    cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                    shape=(job["length"], 2, 3, 96, 128))
    max_error = 0.0
    max_quantization_error = 0.0
    ranges = []
    for finger, video in enumerate(job["videos"]):
        targets = np.asarray(job["timestamps"], dtype=np.float64) + video["from_timestamp"]
        tolerance = 0.51 / job["fps"]
        cursor = 0
        minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
        with av.open(video["path"]) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 1
            previous_frame = None
            previous_time = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError(f"Video frame lacks timestamp: {video['path']}")
                timestamp = float(frame.pts * stream.time_base)
                # Process a target once its nearest bracket is available; this
                # works with shared files and non-zero episode start timestamps.
                while cursor < len(targets) and targets[cursor] <= timestamp:
                    use_previous = previous_time is not None and abs(previous_time - targets[cursor]) < abs(timestamp - targets[cursor])
                    selected = previous_frame if use_previous else frame
                    selected_time = previous_time if use_previous else timestamp
                    error = abs(selected_time - targets[cursor])
                    if error > tolerance:
                        raise ValueError(f"DD timestamp mismatch {video['path']}, target={targets[cursor]}, got={selected_time}")
                    raw = selected.to_ndarray(format="rgb48le")
                    if raw.shape != (96, 128, 3):
                        raise ValueError(f"Unexpected tactile shape {raw.shape} in {video['path']}")
                    dd = decode_dd_rgb48(raw)
                    stored = dd.astype(np.float16)
                    if not np.isfinite(stored).all():
                        raise ValueError(f"Non-finite DD after decoding: {video['path']}")
                    cache[cursor, finger] = stored
                    minimum = np.minimum(minimum, dd.min(axis=(1, 2)))
                    maximum = np.maximum(maximum, dd.max(axis=(1, 2)))
                    max_error = max(max_error, error)
                    max_quantization_error = max(max_quantization_error, float(np.max(np.abs(stored.astype(np.float32) - dd))))
                    cursor += 1
                previous_frame, previous_time = frame, timestamp
                if cursor == len(targets):
                    break
            # Floating-point timestamps can put the final target infinitesimally
            # after the final video PTS. Accept only within the same tolerance.
            while cursor < len(targets) and previous_frame is not None:
                error = abs(previous_time - targets[cursor])
                if error > tolerance:
                    break
                dd = decode_dd_rgb48(previous_frame.to_ndarray(format="rgb48le"))
                stored = dd.astype(np.float16)
                if not np.isfinite(stored).all():
                    raise ValueError(f"Non-finite final DD frame: {video['path']}")
                cache[cursor, finger] = stored
                max_quantization_error = max(max_quantization_error, float(np.max(np.abs(stored.astype(np.float32) - dd))))
                minimum = np.minimum(minimum, dd.min(axis=(1, 2)))
                maximum = np.maximum(maximum, dd.max(axis=(1, 2)))
                max_error = max(max_error, error)
                cursor += 1
            if cursor != len(targets):
                raise ValueError(f"Video too short: {video['path']} decoded {cursor}/{len(targets)} targets")
        ranges.append({"min": minimum.tolist(), "max": maximum.tolist()})
    cache.flush()
    del cache
    os.replace(temporary, output)
    result = {"signature": signature, "frames": job["length"], "shape": [job["length"], 2, 3, 96, 128],
              "dtype": "float16", "max_timestamp_error_s": max_error,
              "max_float16_error_sdk_units": max_quantization_error, "channel_ranges": ranges,
              "source_videos": job["videos"], "cache_sha256": sha256(output)}
    atomic_json(stamp, result)
    return result


def prepare(args):
    sources = assignments(args.source)
    sidecars = assignments(args.sidecar)
    if set(sources) != {"recovery", "normal1", "normal2"}:
        raise ValueError("Provide exactly recovery, normal1 and normal2 sources")
    annotations = pd.read_csv(args.annotations)
    annotations = annotations.set_index("episode_index", verify_integrity=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    offsets = list(range(args.action_gap, args.action_gap + args.chunk_size))
    manifest = {"schema_version": 1, "seed": args.seed, "fps": 30, "action_offsets": offsets,
                "history_policy": "pre-segment entries replaced with retained values and marked padded; existing processor writes zero force history",
                "classification": "none: DD conditions action prediction only", "sources": [], "episodes": [],
                "annotations_sha256": sha256(args.annotations), "cache_ready": False,
                "split_caveat": "Normal validation/test are retention checks: the base 060000 checkpoint already saw these normal episodes.",
                "cache_decoder": "PyAV uint16 rgb48le; (R,G-30000,B-30000)/1000; float16 storage, float32 input; no new reference"}
    jobs = []
    total_prefix = total_recovery = total_anchors = 0
    for source_number, name in enumerate(("recovery", "normal1", "normal2")):
        root_string = sources[name]
        root = Path(root_string)
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        if int(info["fps"]) != 30:
            raise ValueError(f"Unexpected FPS: {root}")
        episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
        data_files = sorted((root / "data").rglob("*.parquet"))
        if not episode_files or not data_files:
            raise FileNotFoundError(f"Missing episode metadata/data in {root}")
        episodes = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
        cols = ["index", "episode_index", "frame_index", "timestamp", "observation.state"]
        rows = pd.concat([pd.read_parquet(p, columns=cols) for p in data_files], ignore_index=True).sort_values("index")
        if len(rows) != int(info["total_frames"]) or not np.array_equal(rows["index"], np.arange(len(rows))):
            raise ValueError(f"Source row indices not contiguous: {root}")
        raw_state = np.stack(rows["observation.state"].to_numpy())
        forces = raw_state[:, 132:144]
        if forces.shape[1] != 12 or not np.isfinite(forces).all() or np.any(forces == 9930):
            raise ValueError(f"Invalid right-finger wrench source: {root}")
        if name == "recovery" and set(episodes["episode_index"]) != set(annotations.index):
            raise ValueError("Recovery annotation episodes differ from source episodes")
        split = grouped_episode_split(episodes["episode_index"].tolist(), args.seed + source_number)
        data_hashes = {str(p.relative_to(root)): sha256(p) for p in data_files + episode_files}
        manifest["sources"].append({"name": name, "root": str(root), "sidecar": sidecars.get(name),
                                    "total_frames": len(rows), "total_episodes": len(episodes),
                                    "info_sha256": sha256(info_path), "parquet_sha256": data_hashes})
        for ep in episodes.sort_values("episode_index").to_dict("records"):
            eid = int(ep["episode_index"])
            length = int(ep["length"])
            e_rows = rows[rows["episode_index"] == eid].sort_values("frame_index")
            if len(e_rows) != length or not np.array_equal(e_rows["frame_index"], np.arange(length)):
                raise ValueError(f"Noncontiguous frames {name}:{eid}")
            source_start = int(ep["dataset_from_index"])
            if not np.array_equal(e_rows["index"], np.arange(source_start, source_start + length)):
                raise ValueError(f"Episode row mapping mismatch {name}:{eid}")
            start, end = 0, length
            if name == "recovery":
                annotation = annotations.loc[eid]
                if int(annotation["episode_length"]) != length:
                    raise ValueError(f"Reviewed annotation length mismatch episode {eid}")
                start, end = int(annotation["recovery_bc_start"]), int(annotation["recovery_bc_end_exclusive"])
                if start < int(annotation["already_bad_contact_start"]) or end > length:
                    raise ValueError(f"Unsafe reviewed recovery boundary episode {eid}")
                total_prefix += start
                total_recovery += end - start
            anchors = safe_anchor_frames(start, end, offsets)
            if not len(anchors) or np.any(anchors[:, None] + np.asarray(offsets) >= end):
                raise ValueError(f"Invalid action targets {name}:{eid}")
            total_anchors += len(anchors)
            cache_rel = f"dd_cache/{name}/episode_{eid:05d}.npy"
            videos = []
            for key in DD_VIDEO_KEYS:
                path = source_video_path(root, info, ep, key)
                videos.append({"path": str(path), "sha256": sha256(path),
                               "from_timestamp": float(ep[f"videos/{key}/from_timestamp"]), "key": key})
            timestamps = e_rows["timestamp"].astype(float).tolist()
            signature_payload = {"videos": videos, "timestamps": timestamps, "decoder_version": 1}
            signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
            jobs.append({"output": str(output_root / cache_rel), "signature": signature, "videos": videos,
                         "timestamps": timestamps, "length": length, "fps": 30})
            manifest["episodes"].append({"source": name, "episode_index": eid, "length": length,
                "dataset_from_index": source_start, "start": start, "end": end, "split": split[eid],
                "anchor_count": len(anchors), "cache": cache_rel, "cache_signature": signature})
        print(f"Validated source {name}: {len(episodes)} episodes, {len(rows)} frames", flush=True)
    manifest["summary"] = {"excluded_recovery_prefix_frames": total_prefix,
        "retained_recovery_frames": total_recovery, "all_source_safe_anchors": total_anchors,
        "recovery_safe_anchors": sum(e["anchor_count"] for e in manifest["episodes"] if e["source"] == "recovery"),
        "splits": {split: {"episodes": sum(e["split"] == split for e in manifest["episodes"]),
                            "anchors": sum(e["anchor_count"] for e in manifest["episodes"] if e["split"] == split)}
                   for split in ("train", "val", "test")},
        "raw_cache_bytes": sum(j["length"] for j in jobs) * 2 * 3 * 96 * 128 * 2}
    atomic_json(output_root / "manifest.pending.json", manifest)
    if args.metadata_only:
        print(json.dumps(manifest["summary"], indent=2), flush=True)
        return
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(decode_episode_cache, job): job for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"DD cache {n}/{len(jobs)} ready: {futures[future]['output']}", flush=True)
    manifest["cache_ready"] = True
    manifest["summary"]["max_timestamp_error_s"] = max(r["max_timestamp_error_s"] for r in results)
    manifest["summary"]["max_float16_error_sdk_units"] = max(r["max_float16_error_sdk_units"] for r in results)
    atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    print(f"READY {output_root / 'manifest.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", required=True, help="recovery|normal1|normal2=/source/root")
    p.add_argument("--sidecar", action="append", default=[], help="source=/canonical/framewise_guidance.parquet")
    p.add_argument("--annotations", type=Path, default=Path(__file__).parent / "dd_recovery_annotations/recovery_phase_annotations.csv")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--action-gap", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--metadata-only", action="store_true")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
