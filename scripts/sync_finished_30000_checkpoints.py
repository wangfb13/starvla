#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_OUTPUTS_ROOT = Path("/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/outputs")
DEFAULT_DEST_HOST = "10.30.120.12"
DEFAULT_DEST_USER = "dm"
DEFAULT_DEST_PORT = 22
DEFAULT_DEST_ROOT = Path("/home/dm/DM_das/lerobot-v0.5.1/outputs")
DEFAULT_TIMEZONE = "Asia/Shanghai"
TARGET_STEP = "030000"


@dataclass(frozen=True)
class SyncTarget:
    output_dir: Path
    run_dir: Path
    checkpoint_dir: Path

    @property
    def relative_checkpoint_parent(self) -> Path:
        return self.run_dir.relative_to(self.output_dir.parent) / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan yesterday's output directories and sync completed 030000 checkpoints "
            "to a local machine via rsync."
        )
    )
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--dest-host", default=DEFAULT_DEST_HOST)
    parser.add_argument("--dest-user", default=DEFAULT_DEST_USER)
    parser.add_argument("--dest-port", type=int, default=DEFAULT_DEST_PORT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument(
        "--date-suffix",
        default=None,
        help="Override the suffix to scan, for example 0624. Defaults to yesterday in the configured timezone.",
    )
    parser.add_argument(
        "--step",
        default=TARGET_STEP,
        help="Checkpoint step directory to sync. Defaults to 030000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned rsync commands without executing them.",
    )
    return parser.parse_args()


def resolve_date_suffix(raw_suffix: str | None, timezone_name: str) -> str:
    if raw_suffix:
        suffix = raw_suffix.strip()
        if len(suffix) != 4 or not suffix.isdigit():
            raise ValueError(f"Invalid --date-suffix={raw_suffix!r}. Expected MMDD, for example 0624.")
        return suffix

    now = datetime.now(ZoneInfo(timezone_name))
    yesterday = now - timedelta(days=1)
    return yesterday.strftime("%m%d")


def find_sync_targets(outputs_root: Path, suffix: str, step: str) -> list[SyncTarget]:
    targets: list[SyncTarget] = []
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Outputs root does not exist: {outputs_root}")

    for output_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        if not output_dir.name.endswith(f"_{suffix}"):
            continue

        for run_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
            checkpoint_dir = run_dir / "checkpoints" / step
            if checkpoint_dir.is_dir():
                targets.append(SyncTarget(output_dir=output_dir, run_dir=run_dir, checkpoint_dir=checkpoint_dir))
    return targets


def ensure_remote_parent(dest_user: str, dest_host: str, dest_port: int, remote_dir: Path, dry_run: bool) -> None:
    command = [
        "ssh",
        "-p",
        str(dest_port),
        "-o",
        "BatchMode=yes",
        f"{dest_user}@{dest_host}",
        f"mkdir -p {remote_dir}",
    ]
    print("[remote-mkdir]", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def rsync_checkpoint(
    target: SyncTarget,
    *,
    dest_user: str,
    dest_host: str,
    dest_port: int,
    dest_root: Path,
    dry_run: bool,
) -> None:
    remote_parent = dest_root / target.relative_checkpoint_parent
    ensure_remote_parent(dest_user, dest_host, dest_port, remote_parent, dry_run)

    command = [
        "rsync",
        "-avP",
        "-e",
        f"ssh -p {dest_port} -o BatchMode=yes",
        str(target.checkpoint_dir),
        f"{dest_user}@{dest_host}:{remote_parent}/",
    ]
    print("[rsync]", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()

    try:
        suffix = resolve_date_suffix(args.date_suffix, args.timezone)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    print(f"[info] outputs_root={args.outputs_root}")
    print(f"[info] dest={args.dest_user}@{args.dest_host}:{args.dest_root}")
    print(f"[info] dest_port={args.dest_port}")
    print(f"[info] timezone={args.timezone}")
    print(f"[info] scan_suffix={suffix}")
    print(f"[info] step={args.step}")
    print(f"[info] dry_run={args.dry_run}")

    try:
        targets = find_sync_targets(args.outputs_root, suffix, args.step)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not targets:
        print(f"[info] No completed {args.step} checkpoints found for suffix _{suffix}.")
        return 0

    print(f"[info] Found {len(targets)} completed checkpoint(s) to sync.")
    for idx, target in enumerate(targets, start=1):
        print(f"[info] [{idx}/{len(targets)}] {target.checkpoint_dir}")
        try:
            rsync_checkpoint(
                target,
                dest_user=args.dest_user,
                dest_host=args.dest_host,
                dest_port=args.dest_port,
                dest_root=args.dest_root,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[error] Sync failed for {target.checkpoint_dir}: {exc}", file=sys.stderr)
            return exc.returncode or 1

    print("[info] Sync finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
