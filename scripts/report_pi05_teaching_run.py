#!/usr/bin/env python3

"""Generate beginner-friendly plots and a markdown report from a PI05/PI05-Full training log."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

KV_RE = re.compile(r"([A-Za-z0-9_./-]+):([-+0-9.eE]+)")
STEP_TOKEN_RE = re.compile(r"step:([A-Za-z0-9.+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Training log path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and reports.")
    parser.add_argument("--output-md", type=Path, required=True, help="Markdown report path.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Parsed CSV path.")
    parser.add_argument("--title", type=str, default="PI05 Teaching Run Report")
    parser.add_argument("--source-name", action="append", default=[], help="Mapping '<index>:<name>'.")
    parser.add_argument("--action-source", action="append", type=int, default=[], help="Source index with action loss.")
    parser.add_argument(
        "--language-only-source",
        action="append",
        type=int,
        default=[],
        help="Source index that contributes only language loss.",
    )
    parser.add_argument("--base-probe-json", type=Path, default=None, help="Optional base-checkpoint probe JSON.")
    parser.add_argument("--final-probe-json", type=Path, default=None, help="Optional final-checkpoint probe JSON.")
    return parser.parse_args()


def parse_step_token(token: str) -> float:
    token = token.strip()
    multiplier = 1.0
    if token.endswith(("K", "k")):
        multiplier = 1e3
        token = token[:-1]
    elif token.endswith(("M", "m")):
        multiplier = 1e6
        token = token[:-1]
    elif token.endswith(("B", "b")):
        multiplier = 1e9
        token = token[:-1]
    return float(token) * multiplier


def parse_source_names(raw_pairs: list[str]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for pair in raw_pairs:
        if ":" not in pair:
            continue
        idx_str, name = pair.split(":", 1)
        try:
            mapping[int(idx_str.strip())] = name.strip()
        except ValueError:
            continue
    return mapping


def parse_log(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    log_index = 0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "step:" in line and "loss:" in line:
            if current is not None:
                rows.append(current)
            log_index += 1
            current = {"log_index": float(log_index)}
            step_match = STEP_TOKEN_RE.search(line)
            if step_match:
                current["step"] = parse_step_token(step_match.group(1))
            for key, value in KV_RE.findall(line):
                if key == "step":
                    continue
                current[key] = float(value)
            continue

        if current is None or "diag " not in line:
            continue

        for key, value in KV_RE.findall(line):
            current[key] = float(value)

    if current is not None:
        rows.append(current)

    if not rows:
        return pd.DataFrame(columns=["log_index", "step"])

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("log_index").reset_index(drop=True)
    return frame


def moving_average(series: pd.Series, window: int = 5) -> pd.Series:
    if len(series) < 2:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def make_plot(frame: pd.DataFrame, columns: list[str], title: str, ylabel: str, output_path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    existing = [column for column in columns if column in frame.columns and frame[column].notna().any()]
    if not existing:
        return False

    x_column = "step"
    x_label = "Step"
    if "log_index" in frame.columns and (
        "step" not in frame.columns or frame["step"].duplicated().any()
    ):
        x_column = "log_index"
        x_label = "Log event"

    plt.figure(figsize=(10, 5))
    for column in existing:
        plt.plot(frame[x_column], moving_average(frame[column]), label=column)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def safe_mean(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def trend_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame[column].dropna().empty:
        return "not logged"
    values = frame[column].dropna()
    window = max(1, int(len(values) * 0.1))
    start_value = safe_mean(values.iloc[:window])
    end_value = safe_mean(values.iloc[-window:])
    if math.isnan(start_value) or math.isnan(end_value):
        return "insufficient data"
    if end_value < start_value * 0.7:
        return f"strong drop ({start_value:.4f} -> {end_value:.4f})"
    if end_value < start_value:
        return f"moderate drop ({start_value:.4f} -> {end_value:.4f})"
    if abs(end_value - start_value) / max(abs(start_value), 1e-8) < 0.05:
        return f"mostly flat ({start_value:.4f} -> {end_value:.4f})"
    return f"rising / unstable ({start_value:.4f} -> {end_value:.4f})"


def format_sample_block(sample: dict[str, Any], header: str) -> str:
    def trim(text: str, limit: int = 650) -> str:
        text = str(text).replace("\n", "\\n")
        return text if len(text) <= limit else text[: limit - 3] + "..."

    topk_lines = "\n".join(
        f"- `{item['token']}` prob={item['prob']:.4f} token_id={item['token_id']}"
        for item in sample.get("topk_first_token", [])[:6]
    )

    return "\n".join(
        [
            f"### {header}",
            f"- dataset: `{sample.get('dataset', 'unknown')}` row `{sample.get('row_index', 'unknown')}`",
            f"- task: `{trim(sample.get('task', ''), 220)}`",
            f"- scalar losses: `loss={sample.get('loss', float('nan')):.4f}`, `flow={sample.get('flow_mse_loss', float('nan')):.4f}`, `action_ce={sample.get('action_ce_loss', float('nan')):.4f}`, `subtask_ce={sample.get('subtask_ce_loss', float('nan')):.4f}`",
            "",
            "```text",
            f"PROMPT: {trim(sample.get('prompt', ''))}",
            f"TARGET: {trim(sample.get('target', ''))}",
            f"TEACHER_FORCED: {trim(sample.get('teacher_forced_prediction', ''))}",
            f"GENERATED: {trim(sample.get('generated_text', ''))}",
            "```",
            "",
            "Top-k first generated token candidates:",
            topk_lines if topk_lines else "- not available",
            "",
        ]
    )


def load_probe(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {sample["sample_id"]: sample for sample in payload.get("samples", [])}


def build_markdown(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    source_names: dict[int, str],
    plot_paths: dict[str, Path],
    base_probe: dict[str, dict[str, Any]],
    final_probe: dict[str, dict[str, Any]],
) -> str:
    source_overview = [
        f"- source `{idx}` = `{source_names.get(idx, f'source_{idx}')}`"
        for idx in sorted(source_names.keys())
    ]
    action_sources = ", ".join(source_names.get(idx, str(idx)) for idx in args.action_source) or "none declared"
    language_only_sources = (
        ", ".join(source_names.get(idx, str(idx)) for idx in args.language_only_source) or "none declared"
    )

    lines = [
        f"# {args.title}",
        "",
        "## Run Setup",
        f"- log: `{args.log}`",
        f"- parsed steps: `{len(frame)}`",
        f"- action-enabled sources: `{action_sources}`",
        f"- language-only sources: `{language_only_sources}`",
        *source_overview,
        "",
        "## How To Read The Curves",
        "- `loss` is the mixed total objective, so it can bounce even when one component improves.",
        "- `subtask_ce_loss` is the text/subtask token loss; it gets signal from every source with language supervision.",
        "- `flow_mse_loss` and `action_ce_loss` only move on robot/action sources, so they are often noisier.",
        "- `loss_mask/action_active_ratio` tells how often the current batch can update action terms.",
        "- `loss_mask/language_active_ratio` tells how often the current batch can update language terms.",
        "- If the trainer logs coarse buckets like `2K` / `3K`, the plots switch to `log event` on the x-axis so every logged point stays visible.",
        "",
        "## Beginner Interpretation",
        f"- total loss trend: `{trend_text(frame, 'loss')}`",
        f"- subtask/text loss trend: `{trend_text(frame, 'subtask_ce_loss')}`",
        f"- action token CE trend: `{trend_text(frame, 'action_ce_loss')}`",
        f"- flow loss trend: `{trend_text(frame, 'flow_mse_loss')}`",
        f"- action mask trend: `{trend_text(frame, 'loss_mask/action_active_ratio')}`",
        f"- language mask trend: `{trend_text(frame, 'loss_mask/language_active_ratio')}`",
        "- If text loss drops faster than action losses, that usually means dense language supervision is easier than robot control.",
        "- If action losses look spikier, that is expected because only robot batches update them.",
        "- If per-source losses separate widely, that usually means the sources differ in difficulty or objective type.",
        "",
        "## Plots",
    ]

    for label, path in plot_paths.items():
        lines.extend([f"### {label}", f"![]({path.name})", ""])

    if base_probe or final_probe:
        lines.extend(["## Probe Samples", ""])
        shared_ids = sorted(set(base_probe.keys()) | set(final_probe.keys()))
        for sample_id in shared_ids:
            if sample_id in base_probe:
                lines.append(format_sample_block(base_probe[sample_id], f"Base Probe — {sample_id}"))
            if sample_id in final_probe:
                lines.append(format_sample_block(final_probe[sample_id], f"Final Probe — {sample_id}"))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    frame = parse_log(args.log)
    source_names = parse_source_names(args.source_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = {
        "overall_loss.png": (["loss", "loss_global/per_sample_mean"], "Overall Loss", "loss"),
        "action_and_flow.png": (["action_ce_loss", "flow_mse_loss"], "Action / Flow Loss", "loss"),
        "subtask_and_tactile.png": (
            ["subtask_ce_loss", "loss_tactile/present", "loss_tactile/missing"],
            "Subtask / Tactile Loss",
            "loss",
        ),
        "source_losses.png": (
            sorted([c for c in frame.columns if c.startswith("loss_source/") and "norm" not in c]),
            "Per-Source Losses",
            "loss",
        ),
        "mask_and_conditioning.png": (
            [
                "loss_mask/action_active_ratio",
                "loss_mask/language_active_ratio",
                "conditioning/state_present_ratio",
                "conditioning/tactile_gate_value",
            ],
            "Mask / Conditioning Signals",
            "value",
        ),
    }

    plot_paths: dict[str, Path] = {}
    for filename, (columns, title, ylabel) in plot_specs.items():
        output_path = args.output_dir / filename
        if make_plot(frame, columns, title, ylabel, output_path):
            plot_paths[title] = output_path

    frame.to_csv(args.output_csv, index=False)
    base_probe = load_probe(args.base_probe_json)
    final_probe = load_probe(args.final_probe_json)
    markdown = build_markdown(args, frame, source_names, plot_paths, base_probe, final_probe)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote markdown to {args.output_md}")
    print(f"Wrote CSV to {args.output_csv}")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
