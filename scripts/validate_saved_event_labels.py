#!/usr/bin/env python3
"""Independently match saved clean chunks against original sidecar frame rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.evaluation_dirs:
        summary = json.loads((directory / "summary.json").read_text())
        labels = pd.read_parquet(
            summary["sidecar"],
            columns=["episode_index", "frame_index", "action"],
            filters=[("episode_index", "==", summary["episode_index"])],
        ).set_index("frame_index")
        arrays = np.load(directory / "event_label_predictions.npz")
        results = []
        for sample in summary["samples"]:
            start, end = sample["chunk_original_frames"]
            expected = np.stack(labels.loc[np.arange(start, end + 1), "action"].to_numpy())
            actual = arrays[f'frame_{sample["current_frame"]:04d}_target_absolute']
            maximum = float(np.max(np.abs(actual - expected)))
            position_max_mm = float(np.max(np.linalg.norm(actual[:, :3] - expected[:, :3], axis=-1)) * 1000)
            assert maximum <= 2e-5, (directory, sample["current_frame"], maximum)
            assert sample["inference_batch_contains_action"] is False
            results.append({
                "current_frame": sample["current_frame"],
                "verified_original_frame_range": [start, end],
                "max_abs_all_action_components": maximum,
                "max_xyz_l2_mm": position_max_mm,
            })
        result = {"all_checks_passed": True, "samples": results}
        (directory / "label_frame_validation.json").write_text(json.dumps(result, indent=2))
        print(directory.name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
