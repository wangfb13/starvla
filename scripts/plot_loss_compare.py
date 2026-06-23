#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_RE = re.compile(r"step:(\d+).*?loss:([0-9.+-eE]+)")
PROGRESS_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


def parse_log(path: Path):
    steps = []
    losses = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if not m:
                continue
            progress = PROGRESS_RE.search(line)
            if progress:
                steps.append(int(progress.group(1)))
            else:
                raw_step = m.group(1)
                if raw_step.endswith("K"):
                    steps.append(int(float(raw_step[:-1]) * 1000))
                else:
                    steps.append(int(raw_step))
            losses.append(float(m.group(2)))
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from two training logs.")
    parser.add_argument("log1", type=Path)
    parser.add_argument("log2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    s1, l1 = parse_log(args.log1)
    s2, l2 = parse_log(args.log2)

    if not s1 or not s2:
        raise SystemExit("No loss entries found in one or both logs")

    plt.figure(figsize=(10, 5.5), dpi=160)
    plt.plot(s1, l1, label=args.log1.stem, linewidth=2)
    plt.plot(s2, l2, label=args.log2.stem, linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    print(f"saved: {args.output}")
    print(f"{args.log1.name}: {len(l1)} points, step {s1[0]} -> {s1[-1]}")
    print(f"{args.log2.name}: {len(l2)} points, step {s2[0]} -> {s2[-1]}")


if __name__ == "__main__":
    main()
