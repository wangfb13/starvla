#!/usr/bin/env python3

"""Build a manifest.json for PI05-Full subtask validation from dataset roots and sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-roots-file",
        type=Path,
        required=True,
        help="Text file with one dataset root per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        required=True,
        help="Canonical sidecar root. Expects {sidecar_root}/{dataset_root.name}/summary.json and framewise_guidance.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest.json path.",
    )
    return parser.parse_args()


def load_dataset_roots(path: Path) -> list[Path]:
    roots: list[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            root = Path(line).expanduser()
            if not root.is_absolute():
                root = (path.parent / root).resolve()
            roots.append(root)
    return roots


def main() -> None:
    args = parse_args()
    roots = load_dataset_roots(args.dataset_roots_file)

    records = []
    skipped = []
    for dataset_root in roots:
        summary_path = args.sidecar_root / dataset_root.name / "summary.json"
        sidecar_path = args.sidecar_root / dataset_root.name / "framewise_guidance.parquet"
        if not summary_path.exists() or not sidecar_path.exists():
            skipped.append(
                {
                    "dataset_root": str(dataset_root),
                    "summary_exists": summary_path.exists(),
                    "sidecar_exists": sidecar_path.exists(),
                }
            )
            continue

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        records.append(
            {
                "rel_key": dataset_root.name,
                "dataset_root": str(dataset_root),
                "twindelta_sidecar": str(sidecar_path),
                "robot_type": summary.get("robot_type", "unknown"),
            }
        )

    payload = {
        "records": records,
        "record_count": len(records),
        "dataset_roots_file": str(args.dataset_roots_file),
        "sidecar_root": str(args.sidecar_root),
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output": str(args.output), "record_count": len(records), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
