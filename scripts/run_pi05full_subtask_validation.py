#!/usr/bin/env python3

"""Run qualitative next-subtask validation on a PI05-Full checkpoint."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05_full.modeling_pi05 import PI05FullPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)


RGB_KEY_PRIORITY = [
    "observation.images.left_cam_left",
    "observation.images.right_cam_right",
    "observation.images.cam_left",
    "observation.images.cam_right",
    "observation.images.image",
    "observation.images.image2",
]


@dataclass
class PanelEntry:
    rel_key: str
    dataset_root: str
    sidecar_path: str
    robot_type: str
    abs_index: int
    episode_index: int
    frame_index: int
    subtask_text: str
    next_subtask_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _clean_text(value: str) -> str:
    cleaned = str(value).replace("\n", " ").strip()
    cleaned = cleaned.replace("Subtask:", "").replace("Task:", "").strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    return " ".join(cleaned.split())


def _norm_text(value: str) -> str:
    return _clean_text(value).lower()


def _similarity(prediction: str, target: str) -> float:
    return float(SequenceMatcher(a=_norm_text(prediction), b=_norm_text(target)).ratio())


def load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def _read_candidates(record: dict[str, Any]) -> pd.DataFrame:
    sidecar = pd.read_parquet(
        record["twindelta_sidecar"],
        columns=[
            "index",
            "episode_index",
            "frame_index",
            "subtask_text_sidecar",
            "next_subtask_text_sidecar",
        ],
    )
    sidecar["subtask_text_sidecar"] = sidecar["subtask_text_sidecar"].fillna("").astype(str)
    sidecar["next_subtask_text_sidecar"] = sidecar["next_subtask_text_sidecar"].fillna("").astype(str)
    sidecar = sidecar[
        sidecar["next_subtask_text_sidecar"].str.strip().astype(bool)
        & sidecar["subtask_text_sidecar"].str.strip().astype(bool)
    ].copy()
    if len(sidecar) == 0:
        return sidecar
    group_cols = ["episode_index", "subtask_text_sidecar", "next_subtask_text_sidecar"]
    representative_rows = []
    for _, group in sidecar.groupby(group_cols, sort=False):
        group = group.sort_values("frame_index")
        representative_rows.append(group.iloc[len(group) // 2])
    rep_df = pd.DataFrame(representative_rows).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return rep_df


def build_panel(records: list[dict[str, Any]], panel_json: Path, num_samples: int) -> list[PanelEntry]:
    if panel_json.exists():
        payload = json.loads(panel_json.read_text(encoding="utf-8"))
        return [PanelEntry(**entry) for entry in payload["entries"]]

    per_type: dict[str, list[PanelEntry]] = {"left": [], "right": [], "dual": [], "unknown": []}
    for record in records:
        candidates = _read_candidates(record)
        if len(candidates) == 0:
            continue
        row = candidates.iloc[len(candidates) // 2]
        robot_type = str(record.get("robot_type", "unknown"))
        per_type.setdefault(robot_type, [])
        per_type[robot_type].append(
            PanelEntry(
                rel_key=str(record["rel_key"]),
                dataset_root=str(record["dataset_root"]),
                sidecar_path=str(record["twindelta_sidecar"]),
                robot_type=robot_type,
                abs_index=int(row["index"]),
                episode_index=int(row["episode_index"]),
                frame_index=int(row["frame_index"]),
                subtask_text=_clean_text(row["subtask_text_sidecar"]),
                next_subtask_text=_clean_text(row["next_subtask_text_sidecar"]),
            )
        )

    total_available = sum(len(items) for items in per_type.values())
    target = min(num_samples, total_available)
    if target <= 0:
        raise SystemExit("No valid next-subtask sidecar entries found for validation.")

    weighted_types = [key for key in ("left", "right", "dual") if key in per_type and per_type[key]]
    selected: list[PanelEntry] = []
    if weighted_types:
        total_pool = sum(len(per_type[key]) for key in weighted_types)
        quotas = {key: max(1, round(target * len(per_type[key]) / total_pool)) for key in weighted_types}
        while sum(quotas.values()) > target:
            biggest = max(quotas, key=quotas.get)
            if quotas[biggest] > 1:
                quotas[biggest] -= 1
            else:
                break
        while sum(quotas.values()) < target:
            smallest = min(weighted_types, key=lambda key: quotas[key] / max(len(per_type[key]), 1))
            quotas[smallest] += 1
        for key in weighted_types:
            selected.extend(per_type[key][: min(quotas[key], len(per_type[key]))])

    if len(selected) < target:
        leftovers = []
        for key, entries in per_type.items():
            leftovers.extend(entries)
        seen = {(entry.rel_key, entry.abs_index) for entry in selected}
        for entry in leftovers:
            marker = (entry.rel_key, entry.abs_index)
            if marker in seen:
                continue
            selected.append(entry)
            seen.add(marker)
            if len(selected) >= target:
                break

    payload = {"num_samples": len(selected), "entries": [asdict(entry) for entry in selected]}
    panel_json.parent.mkdir(parents=True, exist_ok=True)
    panel_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected rank-3 image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] in {1, 3}:
        image = image.permute(1, 2, 0)
    image = image.numpy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return Image.fromarray(image)


def choose_rgb_images(item: dict[str, Any]) -> list[tuple[str, torch.Tensor]]:
    chosen = []
    for key in RGB_KEY_PRIORITY:
        value = item.get(key)
        if torch.is_tensor(value) and value.ndim == 3:
            chosen.append((key, value))
    if chosen:
        return chosen[:2]
    fallback = []
    for key, value in item.items():
        if not key.startswith("observation.images.") or "tactile" in key:
            continue
        if torch.is_tensor(value) and value.ndim == 3:
            fallback.append((key, value))
    return fallback[:2]


def coerce_sample_tensors(item: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            converted[key] = value
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S", "O"}:
                converted[key] = value
            else:
                converted[key] = torch.from_numpy(value)
        elif isinstance(value, np.generic):
            converted[key] = torch.as_tensor(value.item())
        else:
            converted[key] = value
    return converted


def normalize_single_sample_batch(batch: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(batch)
    value = normalized.get("observation.tactile_present")
    if torch.is_tensor(value) and value.ndim == 0:
        normalized["observation.tactile_present"] = value.unsqueeze(0)
    return normalized


def save_image_strip(images: list[tuple[str, torch.Tensor]], output_path: Path) -> None:
    if not images:
        canvas = Image.new("RGB", (480, 160), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 70), "No RGB frame available", fill=(30, 30, 30))
        canvas.save(output_path)
        return

    pil_images = []
    labels = []
    for key, tensor in images:
        image = tensor_to_pil_image(tensor).convert("RGB")
        image.thumbnail((320, 240))
        pil_images.append(image)
        labels.append(key.replace("observation.images.", ""))

    width = sum(img.width for img in pil_images) + 12 * (len(pil_images) - 1)
    height = max(img.height for img in pil_images) + 28
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    cursor_x = 0
    for image, label in zip(pil_images, labels, strict=True):
        canvas.paste(image, (cursor_x, 0))
        draw.text((cursor_x + 4, image.height + 4), label, fill=(20, 20, 20))
        cursor_x += image.width + 12
    canvas.save(output_path)


def decode_generated(tokenizer: Any, tokens: torch.Tensor, masks: torch.Tensor) -> str:
    active = masks.detach().cpu().bool().reshape(-1)
    seq = tokens.detach().cpu().reshape(-1)
    if active.numel() != seq.numel():
        raise ValueError("Token/mask shape mismatch while decoding generated text.")
    seq = seq[active] if active.any() else seq[:0]
    text = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
    return _clean_text(text)


def load_policy_and_preprocessor(checkpoint_dir: Path, device: str) -> tuple[PI05FullPolicy, Any]:
    pretrained_dir = checkpoint_dir / "pretrained_model"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cfg = PreTrainedConfig.from_pretrained(pretrained_dir, local_files_only=True)
    cfg.device = device
    policy = PI05FullPolicy.from_pretrained(
        pretrained_name_or_path=pretrained_dir,
        config=cfg,
        local_files_only=True,
        strict=True,
    )
    policy.to(torch.device(device))
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={
            "tokenizer_processor": {
                "tokenizer_name": cfg.tokenizer_name,
            },
            "device_processor": {
                "device": device,
            },
        },
    )
    return policy, preprocessor


def run_validation(
    job_name: str,
    panel: list[PanelEntry],
    checkpoint_dir: Path,
    output_dir: Path,
    video_backend: str,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    policy, preprocessor = load_policy_and_preprocessor(checkpoint_dir, device=device)
    dataset_cache: dict[tuple[str, str], LeRobotDataset] = {}
    rows = []

    with torch.inference_mode():
        for sample_idx, entry in enumerate(panel):
            cache_key = (entry.dataset_root, entry.sidecar_path)
            if cache_key not in dataset_cache:
                dataset_cache[cache_key] = LeRobotDataset(
                    repo_id=f"subtask_eval_{len(dataset_cache)}",
                    root=entry.dataset_root,
                    sidecar_path=entry.sidecar_path,
                    video_backend=video_backend,
                    download_videos=False,
                )
            dataset = dataset_cache[cache_key]
            item = dataset[entry.abs_index]
            item = coerce_sample_tensors(item)
            processed = preprocessor(item)
            processed = normalize_single_sample_batch(processed)

            images, img_masks = policy._preprocess_images(processed)
            task_tokens, task_masks = policy.get_language_inputs(processed)
            pred_tokens, pred_masks = policy.model.generate_subtask_tokens(
                images,
                img_masks,
                task_tokens,
                task_masks,
                max_decoding_steps=policy.config.max_decoding_steps,
                temperature=policy.config.temperature,
            )
            tokenizer = policy.model._paligemma_tokenizer
            if tokenizer is None:
                raise RuntimeError("pi05_full subtask tokenizer was not initialized after generation.")

            predicted = decode_generated(tokenizer, pred_tokens[0], pred_masks[0])
            gt_from_item = _clean_text(item.get("next_subtask", entry.next_subtask_text))
            current_subtask = _clean_text(item.get("subtask", entry.subtask_text))
            task_text = _clean_text(item.get("task", ""))
            token_gt = decode_generated(
                tokenizer,
                processed[OBS_LANGUAGE_SUBTASK_TOKENS][0],
                processed[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK][0],
            )
            image_path = image_dir / f"{sample_idx:03d}.png"
            save_image_strip(choose_rgb_images(item), image_path)

            exact = _norm_text(predicted) == _norm_text(gt_from_item)
            similarity = _similarity(predicted, gt_from_item)
            rows.append(
                {
                    "sample_id": sample_idx,
                    "rel_key": entry.rel_key,
                    "robot_type": entry.robot_type,
                    "episode_index": entry.episode_index,
                    "frame_index": entry.frame_index,
                    "task": task_text,
                    "current_subtask": current_subtask,
                    "ground_truth_next_subtask": gt_from_item,
                    "ground_truth_tokenized": token_gt,
                    "predicted_next_subtask": predicted,
                    "exact_match": bool(exact),
                    "similarity": float(similarity),
                    "image_path": str(image_path.relative_to(output_dir)),
                }
            )

    exact_rate = float(np.mean([row["exact_match"] for row in rows])) if rows else 0.0
    mean_similarity = float(np.mean([row["similarity"] for row in rows])) if rows else 0.0
    return {
        "job_name": job_name,
        "checkpoint_dir": str(checkpoint_dir),
        "sample_count": len(rows),
        "exact_match_rate": exact_rate,
        "mean_similarity": mean_similarity,
        "rows": rows,
    }


def write_reports(result: dict[str, Any], output_dir: Path) -> None:
    summary_json = output_dir / "summary.json"
    rows_json = output_dir / "samples.json"
    rows_csv = output_dir / "samples.csv"
    report_md = output_dir / "report.md"
    report_html = output_dir / "report.html"

    summary_json.write_text(
        json.dumps(
            {
                "job_name": result["job_name"],
                "checkpoint_dir": result["checkpoint_dir"],
                "sample_count": result["sample_count"],
                "exact_match_rate": result["exact_match_rate"],
                "mean_similarity": result["mean_similarity"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows_json.write_text(json.dumps(result["rows"], indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(result["rows"]).to_csv(rows_csv, index=False)

    md_lines = [
        "# PI05-Full next-subtask qualitative validation",
        "",
        f"- job: `{result['job_name']}`",
        f"- checkpoint: `{result['checkpoint_dir']}`",
        f"- sample_count: `{result['sample_count']}`",
        f"- exact_match_rate: `{result['exact_match_rate']:.3f}`",
        f"- mean_similarity: `{result['mean_similarity']:.3f}`",
        "",
    ]
    for row in result["rows"]:
        md_lines.extend(
            [
                f"## Sample {row['sample_id']:03d} — `{row['rel_key']}`",
                "",
                f"![sample]({row['image_path']})",
                "",
                f"- robot_type: `{row['robot_type']}`",
                f"- task: {row['task']}",
                f"- current subtask: {row['current_subtask']}",
                f"- ground truth next subtask: **{row['ground_truth_next_subtask']}**",
                f"- predicted next subtask: **{row['predicted_next_subtask']}**",
                f"- exact match: `{row['exact_match']}`",
                f"- similarity: `{row['similarity']:.3f}`",
                "",
            ]
        )
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    html_rows = []
    for row in result["rows"]:
        exact_class = "ok" if row["exact_match"] else "bad"
        html_rows.append(
            f"""
            <tr>
              <td>{row['sample_id']:03d}</td>
              <td><img src="{html.escape(row['image_path'])}" style="max-width:420px;border:1px solid #ccc;"></td>
              <td>{html.escape(row['robot_type'])}<br><small>{html.escape(row['rel_key'])}</small></td>
              <td><b>Task</b><br>{html.escape(row['task'])}<br><br><b>Current</b><br>{html.escape(row['current_subtask'])}</td>
              <td>{html.escape(row['ground_truth_next_subtask'])}</td>
              <td>{html.escape(row['predicted_next_subtask'])}</td>
              <td class="{exact_class}">{row['exact_match']}</td>
              <td>{row['similarity']:.3f}</td>
            </tr>
            """
        )
    report_html.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PI05-Full next-subtask validation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f3f3; }}
    .ok {{ background: #e8f6e8; }}
    .bad {{ background: #fdeaea; }}
    .summary {{ margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h1>PI05-Full next-subtask qualitative validation</h1>
  <div class="summary">
    <p><b>job</b>: {html.escape(result['job_name'])}</p>
    <p><b>checkpoint</b>: {html.escape(result['checkpoint_dir'])}</p>
    <p><b>sample_count</b>: {result['sample_count']}</p>
    <p><b>exact_match_rate</b>: {result['exact_match_rate']:.3f}</p>
    <p><b>mean_similarity</b>: {result['mean_similarity']:.3f}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Frame</th>
        <th>Source</th>
        <th>Context</th>
        <th>Ground Truth</th>
        <th>Prediction</th>
        <th>Exact</th>
        <th>Sim</th>
      </tr>
    </thead>
    <tbody>
      {''.join(html_rows)}
    </tbody>
  </table>
</body>
</html>""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    records = load_manifest_records(args.manifest)
    panel = build_panel(records, args.panel_json, args.num_samples)
    result = run_validation(
        job_name=args.job_name,
        panel=panel,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        video_backend=args.video_backend,
        device=args.device,
    )
    write_reports(result, args.output_dir)
    print(
        json.dumps(
            {
                "checkpoint_dir": str(args.checkpoint_dir),
                "panel_json": str(args.panel_json),
                "output_dir": str(args.output_dir),
                "sample_count": result["sample_count"],
                "exact_match_rate": result["exact_match_rate"],
                "mean_similarity": result["mean_similarity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
