#!/usr/bin/env python3

"""Parse a PI05/PI05-Full training log and emit a concise co-training report."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


STEP_RE = re.compile(r"step:([A-Za-z0-9.+-]+).*?\bloss:([-+0-9.eE]+)")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+):([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Path to train log file.")
    parser.add_argument(
        "--source-name",
        action="append",
        default=[],
        help="Source mapping in format '<index>:<name>' (example: 0:libero).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-step parsed values.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="PI05 Co-Training Report",
        help="Report title.",
    )
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


def mean(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return sum(values) / len(values)


def trend_text(start_v: float, end_v: float) -> str:
    if not (start_v == start_v and end_v == end_v):
        return "insufficient data"
    if end_v < start_v * 0.7:
        return "strong decrease (good)"
    if end_v < start_v:
        return "moderate decrease (good)"
    if abs(end_v - start_v) / max(abs(start_v), 1e-9) < 0.05:
        return "mostly flat"
    return "increasing (watch for instability)"


@dataclass
class StepRow:
    step: float
    loss: float
    loss_source: dict[int, float] = field(default_factory=dict)
    loss_source_norm: dict[int, float] = field(default_factory=dict)
    sampling_ratio: dict[int, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def parse_log(path: Path) -> list[StepRow]:
    rows: list[StepRow] = []
    current: StepRow | None = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            current = StepRow(
                step=parse_step_token(step_match.group(1)),
                loss=float(step_match.group(2)),
            )
            rows.append(current)
            continue

        if current is None or "diag " not in line:
            continue

        diag_payload = line.split("diag ", 1)[1]
        for key, value_s in KV_RE.findall(diag_payload):
            value = float(value_s)
            current.metrics[key] = value
            if key.startswith("loss_source/"):
                try:
                    idx = int(key.split("/", 1)[1])
                except ValueError:
                    continue
                current.loss_source[idx] = value
            if key.startswith("loss_source_norm/"):
                try:
                    idx = int(key.split("/", 1)[1])
                except ValueError:
                    continue
                current.loss_source_norm[idx] = value
            if key.startswith("sampling_ratio/source_"):
                idx_str = key.rsplit("_", 1)[-1]
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                current.sampling_ratio[idx] = value

    return rows


def summarize(rows: list[StepRow], source_names: dict[int, str]) -> str:
    if len(rows) == 0:
        return "No parsable training rows were found in the log."

    losses = [r.loss for r in rows]
    window = max(1, int(len(rows) * 0.1))
    loss_start = mean(losses[:window])
    loss_end = mean(losses[-window:])
    loss_best = min(losses)

    all_source_ids = sorted(
        {idx for r in rows for idx in r.loss_source.keys()} | {idx for idx in source_names.keys()}
    )

    source_lines: list[str] = []
    turbulence_rows: list[tuple[str, float]] = []
    for source_id in all_source_ids:
        source_loss = [r.loss_source[source_id] for r in rows if source_id in r.loss_source]
        source_loss_norm = [r.loss_source_norm[source_id] for r in rows if source_id in r.loss_source_norm]
        source_ratio = [r.sampling_ratio[source_id] for r in rows if source_id in r.sampling_ratio]
        source_label = source_names.get(source_id, f"source_{source_id}")
        if len(source_loss) == 0:
            source_lines.append(f"| {source_label} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue

        s_win = max(1, int(len(source_loss) * 0.1))
        s_start = mean(source_loss[:s_win])
        s_end = mean(source_loss[-s_win:])
        s_ratio = mean(source_ratio) if len(source_ratio) > 0 else float("nan")
        s_std = mean([(value - mean(source_loss)) ** 2 for value in source_loss]) ** 0.5
        mean_abs_delta = (
            mean([abs(curr - prev) for prev, curr in zip(source_loss[:-1], source_loss[1:])])
            if len(source_loss) >= 2
            else 0.0
        )
        _ = mean(source_loss_norm) if len(source_loss_norm) > 0 else None
        turbulence_rows.append((source_label, float(mean_abs_delta)))
        source_lines.append(
            f"| {source_label} | {s_start:.6f} | {s_end:.6f} | {s_ratio:.4f} | {s_std:.6f} | {mean_abs_delta:.6f} | {trend_text(s_start, s_end)} |"
        )

    turbulence_rows = sorted(turbulence_rows, key=lambda item: item[1], reverse=True)
    turbulence_bullets = [
        f"- `{label}` mean |Δloss|: `{value:.6f}`" for label, value in turbulence_rows[:5]
    ]

    focus_metric_keys: list[str] = []
    for exact_key in (
        "sampling_ratio/tactile_present",
        "sampling_ratio/prev_action_present",
        "loss_mask/action_active_ratio",
        "loss_mask/language_active_ratio",
        "loss_tactile/present",
        "loss_tactile/missing",
        "tactile_gate_value",
    ):
        if any(exact_key in row.metrics for row in rows):
            focus_metric_keys.append(exact_key)
    focus_metric_keys.extend(
        sorted({key for row in rows for key in row.metrics.keys() if key.startswith("conditioning/")})
    )

    conditioning_lines: list[str] = []
    for key in focus_metric_keys:
        values = [row.metrics[key] for row in rows if key in row.metrics]
        if not values:
            continue
        metric_window = max(1, int(len(values) * 0.1))
        start_v = mean(values[:metric_window])
        end_v = mean(values[-metric_window:])
        conditioning_lines.append(
            f"| `{key}` | {start_v:.6f} | {end_v:.6f} | {mean(values):.6f} | {trend_text(start_v, end_v)} |"
        )

    lines = [
        f"# {args.title}",
        "",
        "## Overall",
        f"- Parsed steps: `{len(rows)}`",
        f"- Step range: `{int(rows[0].step)} -> {int(rows[-1].step)}`",
        f"- Total loss start(10%): `{loss_start:.6f}`",
        f"- Total loss end(10%): `{loss_end:.6f}`",
        f"- Total loss best: `{loss_best:.6f}`",
        f"- Trend: `{trend_text(loss_start, loss_end)}`",
        "",
        "## Per Source",
        "| Source | Start(10%) | End(10%) | Mean Sampling Ratio | Std | Mean |Δloss| | Trend |",
        "|---|---:|---:|---:|---:|---:|---|",
        *source_lines,
        "",
        "## Source Turbulence Watchlist",
        *(turbulence_bullets if turbulence_bullets else ["- No per-source turbulence values were available."]),
        "",
    ]
    if conditioning_lines:
        lines.extend(
            [
                "## Conditioning And Modality Signals",
                "| Metric | Start(10%) | End(10%) | Mean | Trend |",
                "|---|---:|---:|---:|---|",
                *conditioning_lines,
                "",
            ]
        )
    lines.extend(
        [
            "## Rookie Interpretation",
            "- `loss` gives the overall optimization direction.",
            "- `loss_source/*` helps us see whether one dataset/source is lagging or being ignored.",
            "- `Mean |Δloss|` is a simple turbulence signal: high values often mean the source is noisy or under-fit.",
            "- `sampling_ratio/*` tells whether a source was actually seen as often as expected.",
            "- Tactile and conditioning metrics help explain why total loss may move even if action behavior does not.",
        ]
    )
    return "\n".join(lines)


def maybe_write_csv(rows: list[StepRow], output_csv: Path) -> None:
    all_source_ids = sorted({idx for r in rows for idx in r.loss_source.keys()})
    all_metric_keys = sorted(
        {
            key
            for row in rows
            for key in row.metrics.keys()
            if not key.startswith("loss_source/")
            and not key.startswith("loss_source_norm/")
            and not key.startswith("sampling_ratio/source_")
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "step",
            "loss",
            *[f"loss_source/{idx}" for idx in all_source_ids],
            *[f"loss_source_norm/{idx}" for idx in all_source_ids],
            *[f"sampling_ratio/source_{idx}" for idx in all_source_ids],
            *all_metric_keys,
        ]
        writer.writerow(header)
        for row in rows:
            out = [row.step, row.loss]
            out.extend(row.loss_source.get(idx, "") for idx in all_source_ids)
            out.extend(row.loss_source_norm.get(idx, "") for idx in all_source_ids)
            out.extend(row.sampling_ratio.get(idx, "") for idx in all_source_ids)
            out.extend(row.metrics.get(key, "") for key in all_metric_keys)
            writer.writerow(out)


def parse_source_names(raw: list[str]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for pair in raw:
        if ":" not in pair:
            continue
        idx_s, name = pair.split(":", 1)
        try:
            mapping[int(idx_s.strip())] = name.strip()
        except ValueError:
            continue
    return mapping


if __name__ == "__main__":
    args = parse_args()
    rows = parse_log(args.log)
    source_names = parse_source_names(args.source_name)
    report = summarize(rows, source_names)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    if args.output_csv is not None:
        maybe_write_csv(rows, args.output_csv)
    print(f"Wrote report to {args.output_md}")
    if args.output_csv is not None:
        print(f"Wrote CSV to {args.output_csv}")
