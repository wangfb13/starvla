#!/usr/bin/env python3

"""Offline action prediction report for PI05 / PI05-Full checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05_full.modeling_pi05 import PI05FullPolicy


RGB_KEY_PRIORITY = [
    "observation.images.left_cam_left",
    "observation.images.right_cam_right",
    "observation.images.cam_left",
    "observation.images.cam_right",
    "observation.images.image",
    "observation.images.image2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sidecar-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=0)
    parser.add_argument("--job-name", type=str, default="")
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


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


def load_policy_and_processors(
    checkpoint_dir: Path,
    device: str,
) -> tuple[PreTrainedConfig, PI05Policy | PI05FullPolicy, Any, Any]:
    pretrained_dir = checkpoint_dir / "pretrained_model"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cfg = PreTrainedConfig.from_pretrained(pretrained_dir, local_files_only=True)
    cfg.device = device
    if cfg.type == "pi05":
        policy_cls = PI05Policy
    elif cfg.type == "pi05_full":
        policy_cls = PI05FullPolicy
    else:
        raise ValueError(f"Unsupported policy type for this report: {cfg.type}")

    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=pretrained_dir,
        config=cfg,
        local_files_only=True,
        strict=True,
    )
    policy.to(torch.device(device))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={
            "device_processor": {
                "device": device,
            },
        },
    )
    return cfg, policy, preprocessor, postprocessor


def build_sample_indices(
    dataset_len: int,
    num_samples: int,
    start_index: int,
    sample_stride: int,
    seed: int,
) -> list[int]:
    if dataset_len <= 0:
        return []
    if sample_stride > 0:
        indices = list(range(start_index, dataset_len, sample_stride))
        return indices[:num_samples]
    rng = np.random.default_rng(seed)
    population = np.arange(start_index, dataset_len, dtype=np.int64)
    if len(population) == 0:
        return []
    if len(population) <= num_samples:
        return population.tolist()
    chosen = np.sort(rng.choice(population, size=num_samples, replace=False))
    return chosen.tolist()


def predict_action_chunk_absolute(
    policy: PI05Policy | PI05FullPolicy,
    postprocessor: Any,
    processed: dict[str, Any],
) -> torch.Tensor:
    predicted = policy.predict_action_chunk(processed)
    predicted_abs = postprocessor(predicted)
    if not torch.is_tensor(predicted_abs):
        raise TypeError(f"Expected tensor output from postprocessor, got {type(predicted_abs)}")
    return predicted_abs.detach().cpu()


def compute_action_metrics(
    pred_chunk: torch.Tensor,
    gt_chunk: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> dict[str, Any]:
    pred = pred_chunk.to(dtype=torch.float32)
    gt = gt_chunk.to(dtype=torch.float32)
    horizon = min(pred.shape[0], gt.shape[0])
    action_dim = min(pred.shape[1], gt.shape[1])
    pred = pred[:horizon, :action_dim]
    gt = gt[:horizon, :action_dim]

    if action_mask is None:
        mask = torch.ones_like(gt, dtype=torch.bool)
    else:
        mask = action_mask[:horizon, :action_dim].to(dtype=torch.bool)
    mask_f = mask.to(dtype=torch.float32)

    diff = pred - gt
    abs_diff = diff.abs()
    sq_diff = diff.square()

    denom = mask_f.sum().item()
    if denom <= 0:
        mae = float("nan")
        rmse = float("nan")
    else:
        mae = float((abs_diff * mask_f).sum().item() / denom)
        rmse = float(math.sqrt((sq_diff * mask_f).sum().item() / denom))

    per_step_denom = mask_f.sum(dim=1).clamp_min(1.0)
    per_dim_denom = mask_f.sum(dim=0).clamp_min(1.0)
    per_step_mae = ((abs_diff * mask_f).sum(dim=1) / per_step_denom).cpu().numpy()
    per_dim_mae = ((abs_diff * mask_f).sum(dim=0) / per_dim_denom).cpu().numpy()

    return {
        "pred": pred,
        "gt": gt,
        "mask": mask,
        "mae": mae,
        "rmse": rmse,
        "per_step_mae": per_step_mae,
        "per_dim_mae": per_dim_mae,
    }


def plot_summary(per_dim_mae: np.ndarray, per_step_mae: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].bar(np.arange(len(per_dim_mae)), per_dim_mae)
    axes[0].set_title("Mean Absolute Error by Action Dimension")
    axes[0].set_xlabel("Action Dimension")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(np.arange(len(per_step_mae)), per_step_mae, marker="o", linewidth=2)
    axes[1].set_title("Mean Absolute Error by Chunk Step")
    axes[1].set_xlabel("Chunk Step")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_plots.png", dpi=160)
    plt.close(fig)


def plot_sample_action_curves(
    sample_id: int,
    pred: torch.Tensor,
    gt: torch.Tensor,
    output_path: Path,
    max_dims: int = 6,
) -> None:
    dims = min(max_dims, pred.shape[1], gt.shape[1])
    fig, axes = plt.subplots(dims, 1, figsize=(10, 2.4 * dims), sharex=True)
    if dims == 1:
        axes = [axes]

    xs = np.arange(pred.shape[0])
    for dim in range(dims):
        ax = axes[dim]
        ax.plot(xs, gt[:, dim].numpy(), label="gt", linewidth=2)
        ax.plot(xs, pred[:, dim].numpy(), label="pred", linewidth=2, linestyle="--")
        ax.set_ylabel(f"dim {dim}")
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend()
    axes[-1].set_xlabel("chunk step")
    fig.suptitle(f"Sample {sample_id:03d} action curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    job_name: str,
    cfg: PreTrainedConfig,
    checkpoint_dir: Path,
    dataset_root: Path,
    sample_count: int,
    mean_mae: float,
    mean_rmse: float,
    sample_rows: list[dict[str, Any]],
) -> None:
    report_path = output_dir / "report.md"
    lines = [
        "# PI05 / PI05-Full action prediction report",
        "",
        f"- job: `{job_name}`",
        f"- policy_type: `{cfg.type}`",
        f"- checkpoint: `{checkpoint_dir}`",
        f"- dataset_root: `{dataset_root}`",
        f"- sample_count: `{sample_count}`",
        f"- mean_mae: `{mean_mae:.6f}`",
        f"- mean_rmse: `{mean_rmse:.6f}`",
        "",
        "## Artifacts",
        "",
        "- `summary.json`",
        "- `samples.csv`",
        "- `sample_step_metrics.csv`",
        "- `summary_plots.png`",
        "- `images/`",
        "- `action_curves/`",
        "",
    ]
    for row in sample_rows:
        lines.extend(
            [
                f"## Sample {row['sample_id']:03d}",
                "",
                f"- dataset_index: `{row['dataset_index']}`",
                f"- task: {row['task']}",
                f"- subtask: {row['subtask']}",
                f"- mae: `{row['mae']:.6f}`",
                f"- rmse: `{row['rmse']:.6f}`",
                f"- image: `{row['image_path']}`",
                f"- action curves: `{row['curve_path']}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    curve_dir = output_dir / "action_curves"
    image_dir.mkdir(parents=True, exist_ok=True)
    curve_dir.mkdir(parents=True, exist_ok=True)

    cfg, policy, preprocessor, postprocessor = load_policy_and_processors(args.checkpoint_dir, args.device)
    dataset = LeRobotDataset(
        repo_id="offline_action_eval",
        root=str(args.dataset_root),
        sidecar_path=str(args.sidecar_path) if args.sidecar_path else None,
        video_backend=args.video_backend,
        download_videos=False,
    )

    indices = build_sample_indices(
        dataset_len=len(dataset),
        num_samples=args.num_samples,
        start_index=args.start_index,
        sample_stride=args.sample_stride,
        seed=args.seed,
    )
    if not indices:
        raise SystemExit("No dataset indices selected for evaluation.")

    sample_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    per_dim_mae_all = []
    per_step_mae_all = []

    with torch.inference_mode():
        for sample_id, dataset_index in enumerate(indices):
            raw_item = coerce_sample_tensors(dataset[dataset_index])
            processed = preprocessor(raw_item)

            pred_chunk = predict_action_chunk_absolute(policy, postprocessor, processed)[0]
            gt_chunk = raw_item["action"]
            if not torch.is_tensor(gt_chunk):
                gt_chunk = torch.as_tensor(gt_chunk)
            if gt_chunk.ndim == 1:
                gt_chunk = gt_chunk.unsqueeze(0)

            action_mask = raw_item.get("action_valid_mask", raw_item.get("observation.action_mask"))
            if action_mask is not None and not torch.is_tensor(action_mask):
                action_mask = torch.as_tensor(action_mask)
            if torch.is_tensor(action_mask) and action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0).expand(gt_chunk.shape[0], -1)

            metrics = compute_action_metrics(pred_chunk, gt_chunk, action_mask)
            per_dim_mae_all.append(metrics["per_dim_mae"])
            per_step_mae_all.append(metrics["per_step_mae"])

            image_path = image_dir / f"{sample_id:03d}.png"
            save_image_strip(choose_rgb_images(raw_item), image_path)
            curve_path = curve_dir / f"{sample_id:03d}.png"
            plot_sample_action_curves(sample_id, metrics["pred"], metrics["gt"], curve_path)

            task_text = _clean_text(raw_item.get("task"))
            subtask_text = _clean_text(raw_item.get("subtask", raw_item.get("next_subtask", "")))
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_index": dataset_index,
                    "task": task_text,
                    "subtask": subtask_text,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "image_path": str(image_path.relative_to(output_dir)),
                    "curve_path": str(curve_path.relative_to(output_dir)),
                }
            )

            for step_idx, step_mae in enumerate(metrics["per_step_mae"]):
                step_rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_index": dataset_index,
                        "chunk_step": step_idx,
                        "mae": float(step_mae),
                    }
                )

    max_dim = max(len(x) for x in per_dim_mae_all)
    max_step = max(len(x) for x in per_step_mae_all)
    dim_matrix = np.full((len(per_dim_mae_all), max_dim), np.nan, dtype=np.float32)
    step_matrix = np.full((len(per_step_mae_all), max_step), np.nan, dtype=np.float32)
    for row_idx, values in enumerate(per_dim_mae_all):
        dim_matrix[row_idx, : len(values)] = values
    for row_idx, values in enumerate(per_step_mae_all):
        step_matrix[row_idx, : len(values)] = values

    mean_per_dim_mae = np.nanmean(dim_matrix, axis=0)
    mean_per_step_mae = np.nanmean(step_matrix, axis=0)
    mean_mae = float(np.nanmean([row["mae"] for row in sample_rows]))
    mean_rmse = float(np.nanmean([row["rmse"] for row in sample_rows]))

    pd.DataFrame(sample_rows).to_csv(output_dir / "samples.csv", index=False)
    pd.DataFrame(step_rows).to_csv(output_dir / "sample_step_metrics.csv", index=False)
    summary = {
        "job_name": args.job_name or args.checkpoint_dir.name,
        "policy_type": cfg.type,
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(args.dataset_root),
        "sidecar_path": str(args.sidecar_path) if args.sidecar_path else None,
        "sample_count": len(sample_rows),
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_per_dim_mae": mean_per_dim_mae.tolist(),
        "mean_per_step_mae": mean_per_step_mae.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_summary(mean_per_dim_mae, mean_per_step_mae, output_dir)
    write_report(
        output_dir=output_dir,
        job_name=args.job_name or args.checkpoint_dir.name,
        cfg=cfg,
        checkpoint_dir=args.checkpoint_dir,
        dataset_root=args.dataset_root,
        sample_count=len(sample_rows),
        mean_mae=mean_mae,
        mean_rmse=mean_rmse,
        sample_rows=sample_rows,
    )


if __name__ == "__main__":
    main()
