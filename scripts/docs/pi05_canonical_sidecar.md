# PI05 Canonical Sidecar Builder

`scripts/build_pi05_canonical_sidecar.py` converts LeRobot v3 datasets into a VTLA-style
framewise sidecar. The sidecar keeps the original dataset untouched and writes canonical
`observation.state` / `action` vectors into `framewise_guidance.parquet`.

## Outputs

For each input dataset root, the script writes:

- `framewise_guidance.parquet`: framewise canonical sidecar.
- `stats.json`: normalization stats for canonical `observation.state` and framewise canonical `action`.
- `summary.json`: schema, dimensions, validity ratios, and recommended policy flags.
- `stats_relative_action_chunk{N}_gap{G}.json`: optional stats for `use_relative_actions=true` training.
- `build_manifest.json`: top-level map from dataset name to sidecar/stats paths.

The canonical schema follows the current repo's `pi05_full` contract:

- `observation.state`: 22D = left 10D + right 10D + `left_valid/right_valid`.
- `action`: 20D = left 10D + right 10D.
- Per arm 10D = xyz + rot6d(6) + gripper.
- `action_valid_mask`: 20D per-dimension loss mask, useful for single-arm and mixed-arm data.
- `state_mask`: 20D mask for the pose/gripper part of state.
- `arm_present_mask`: 2D left/right arm presence.
- `state_present`: scalar frame-level state availability.

## Dataset Type

By default, use `--dataset-type auto`. The script reads each dataset's `meta/info.json`
`robot_type` and maps:

- `spark3_pose_gripper` -> real-robot mode.
- `ugripper_left`, `ugripper_right`, `dual_ugripper` -> UMI mode.

You can still force the behavior with `--dataset-type umi` or `--dataset-type real`.

In UMI mode, pose state and action are transformed into each
episode's first valid state pose frame:

- position is expressed in the episode anchor coordinate frame.
- quaternion is converted to relative rotation and stored as rot6d.
- gripper is stored as a framewise continuous value; training can convert it to chunk-relative.

In real-robot mode, pose state and action stay in the dataset
coordinate frame, but quaternion is still converted to rot6d and packed into the same
22D/20D canonical schema.

## Examples

Build a UMI sidecar and relative-action stats:

```bash
/root/miniconda3/envs/lerobot_v0.5.1/bin/python \
  scripts/build_pi05_canonical_sidecar.py \
  --dataset-root /mnt/cpfs/datasets/pi05_groupA_train_roots_symlink/000__dag911262r00025a_lerobot_0410_233405_train_ready \
  --output-root /mnt/cpfs/repo/pi05_canonical_sidecars_umi \
  --relative-stats-chunk-size 50 \
  --relative-stats-action-gap-frames 6 \
  --overwrite
```

Build a real-robot sidecar:

```bash
/root/miniconda3/envs/lerobot_v0.5.1/bin/python \
  scripts/build_pi05_canonical_sidecar.py \
  --dataset-root /mnt/sparky_data/ugripper_peg_in_hole_v1gripper_3/sparky_ugripper_peg_in_hole_200_v30_unified \
  --output-root /mnt/cpfs/repo/pi05_canonical_sidecars_real \
  --relative-stats-chunk-size 50 \
  --relative-stats-action-gap-frames 6 \
  --overwrite
```

Process many roots from a txt file, one dataset root per line:

```bash
/root/miniconda3/envs/lerobot_v0.5.1/bin/python \
  scripts/build_pi05_canonical_sidecar.py \
  --dataset-roots-file /mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/groupA_roots.txt \
  --output-root /mnt/cpfs/repo/pi05_canonical_sidecars_umi \
  --relative-stats-chunk-size 50 \
  --relative-stats-action-gap-frames 6 \
  --overwrite
```

You can also repeat `--dataset-root`, repeat `--dataset-roots-file`, or mix both.

## Training Notes

If you train from this sidecar, the dataset already provides canonical 22D/20D vectors.
Do not run the online canonicalization step again:

```bash
--policy.canonicalize_bimanual_rot6d=false
```

Train directly from a roots txt and sidecar root:

```bash
--dataset.roots_file=/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/scripts/groupA_roots.txt \
--dataset.sidecar_root=/mnt/cpfs/repo/pi05_canonical_sidecars_groupA \
--policy.max_state_dim=22 \
--policy.max_action_dim=20 \
--policy.canonicalize_bimanual_rot6d=false
```

`dataset.roots_file` accepts one local dataset root per line. The listed roots may live under
different parent directories, but their final directory names must be unique because sidecars
are resolved as:

```text
{dataset.sidecar_root}/{dataset_root.name}/framewise_guidance.parquet
```

For a single dataset debug run, you can also point directly to one sidecar parquet:

```bash
--dataset.root=/path/to/one_lerobot_dataset \
--dataset.sidecar_path=/path/to/sidecar/framewise_guidance.parquet
```

In this mode, stats are resolved from the same directory as the parquet. With
`use_relative_actions=true`, the loader first tries
`/path/to/sidecar/stats_relative_action_chunk{N}_gap{G}.json`; otherwise it expects
`/path/to/sidecar/stats.json`. You can override paths with `dataset.stats_path` or
`dataset.relative_stats_path`.

If you still train with:

```bash
--policy.use_relative_actions=true
```

then the framewise `stats.json` is not enough for action normalization. The training loader
prefers `stats_relative_action_chunk{N}_gap{G}.json`, where `N` and `G` match your training
`chunk_size` and the effective action gap for that dataset. If this file is missing, training
computes the relative action stats dynamically from `framewise_guidance.parquet` instead of
falling back to incompatible framewise action stats.

Relative-action stats are configuration-specific. Precomputing them is still recommended for
large datasets, but it is no longer mandatory if the sidecar parquet contains
`observation.state`, `action`, `episode_index`, and an optional `action_valid_mask`.

For mixed UMI + real-robot training, set the global gap for UMI with
`--policy.action_gap_frames=<gap>`. Datasets with `robot_type=spark3_pose_gripper` are
automatically sampled and normalized with `gap=0` because their recorded state/action pairs
already include the control delay. Explicit `policy.action_gap_dataset_overrides` entries
still take precedence over this automatic inference.

The default `--relative-action-keyframes 0:3,10:13` matches canonical left/right
`xyz + rot6d` blocks. Training uses an SE(3)-aware transform for those keyframes:
`rel_pos = R_state^T @ (target_pos - state_pos)` and `rel_rot = R_state^T @ R_target`.
This replaces the old scalar subtraction behavior for rot6d. Gripper is outside the
keyframes and, with an empty exclude list, is trained as a continuous chunk-relative
value (`action_gripper - state_gripper`). Use the matching training arguments:

```bash
--policy.relative_action_keyframes='["0:3","10:13"]'
--policy.relative_exclude_joints='[]'
```

If you need absolute gripper targets for an ablation, override
`RELATIVE_EXCLUDE_JOINTS='["gripper"]'`.

When `relative_action_keyframes` is non-empty, the training loader dynamically computes
relative stats from `framewise_guidance.parquet` so it cannot silently reuse older
`stats_relative_action_chunk{N}_gap{G}.json` files produced with scalar rot6d subtraction.

When sidecar stats are loaded, `pi05`/`pi05_full` skip the slow dynamic
`build_canonical_bimanual_stats` pass and use the precomputed sidecar stats for
`observation.state` and `action`.
