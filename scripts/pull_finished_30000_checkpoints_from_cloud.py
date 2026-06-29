#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import PurePosixPath, Path
from zoneinfo import ZoneInfo


DEFAULT_REMOTE_HOST = "121.89.91.224"
DEFAULT_REMOTE_USER = "root"
DEFAULT_REMOTE_PORT = 1022
DEFAULT_REMOTE_OUTPUTS_ROOT = PurePosixPath("/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5/outputs")
DEFAULT_LOCAL_OUTPUTS_ROOT = Path("/home/dm/DM_das/lerobot-v0.5.1/outputs")
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_STEP = "030000"
DEFAULT_STRICT_HOST_KEY_CHECKING = "accept-new"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run on the local machine. Check yesterday's cloud training outputs and "
            "pull completed 030000 checkpoints with rsync."
        )
    )
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote-outputs-root", type=PurePosixPath, default=DEFAULT_REMOTE_OUTPUTS_ROOT)
    parser.add_argument("--local-outputs-root", type=Path, default=DEFAULT_LOCAL_OUTPUTS_ROOT)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument(
        "--strict-host-key-checking",
        default=DEFAULT_STRICT_HOST_KEY_CHECKING,
        help="Value passed to ssh -o StrictHostKeyChecking=..., for example yes/no/accept-new.",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Enable ssh BatchMode=yes. Leave off if your local pull still needs password/passphrase/host-key prompts.",
    )
    parser.add_argument(
        "--date-suffix",
        default=None,
        help="Override the date suffix to scan, for example 0624. Defaults to yesterday in the configured timezone.",
    )
    parser.add_argument("--step", default=DEFAULT_STEP)
    parser.add_argument("--dry-run", action="store_true")
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


def build_ssh_base_command(
    host: str,
    user: str,
    port: int,
    *,
    strict_host_key_checking: str,
    batch_mode: bool,
) -> list[str]:
    command = [
        "ssh",
        "-p",
        str(port),
        "-o",
        f"StrictHostKeyChecking={strict_host_key_checking}",
    ]
    if batch_mode:
        command.extend(["-o", "BatchMode=yes"])
    command.append(f"{user}@{host}")
    return command


def run_ssh_capture(
    host: str,
    user: str,
    port: int,
    remote_command: str,
    *,
    strict_host_key_checking: str,
    batch_mode: bool,
) -> str:
    command = build_ssh_base_command(
        host,
        user,
        port,
        strict_host_key_checking=strict_host_key_checking,
        batch_mode=batch_mode,
    )
    command.append(remote_command)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def list_remote_targets(
    host: str,
    user: str,
    port: int,
    remote_outputs_root: PurePosixPath,
    suffix: str,
    step: str,
    *,
    strict_host_key_checking: str,
    batch_mode: bool,
) -> list[PurePosixPath]:
    pattern = f"*_{suffix}"
    remote_command = (
        f"find {shlex.quote(str(remote_outputs_root))} "
        f"-mindepth 4 -maxdepth 4 -type d "
        f"-path '*/{pattern}/*/checkpoints/{step}' | sort"
    )
    print(f"[debug] remote_find_command={remote_command}")
    stdout = run_ssh_capture(
        host,
        user,
        port,
        remote_command,
        strict_host_key_checking=strict_host_key_checking,
        batch_mode=batch_mode,
    )
    targets = []
    for line in stdout.splitlines():
        line = line.strip()
        if line:
            targets.append(PurePosixPath(line))
    return targets


def pull_remote_checkpoint(
    checkpoint_dir: PurePosixPath,
    *,
    remote_host: str,
    remote_user: str,
    remote_port: int,
    remote_outputs_root: PurePosixPath,
    local_outputs_root: Path,
    strict_host_key_checking: str,
    batch_mode: bool,
    dry_run: bool,
) -> None:
    output_dir_name = checkpoint_dir.relative_to(remote_outputs_root).parts[0]
    local_parent = local_outputs_root / output_dir_name / checkpoint_dir.name
    local_parent.mkdir(parents=True, exist_ok=True)
    remote_source = checkpoint_dir / "pretrained_model"

    command = [
        "rsync",
        "-avP",
        "-e",
        (
            f"ssh -p {remote_port} "
            f"-o StrictHostKeyChecking={strict_host_key_checking}"
            + (" -o BatchMode=yes" if batch_mode else "")
        ),
        f"{remote_user}@{remote_host}:{remote_source}",
        f"{local_parent}/",
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

    print(f"[info] remote={args.remote_user}@{args.remote_host}:{args.remote_outputs_root}")
    print(f"[info] remote_port={args.remote_port}")
    print(f"[info] local_outputs_root={args.local_outputs_root}")
    print(f"[info] timezone={args.timezone}")
    print(f"[info] scan_suffix={suffix}")
    print(f"[info] step={args.step}")
    print(f"[info] strict_host_key_checking={args.strict_host_key_checking}")
    print(f"[info] batch_mode={args.batch_mode}")
    print(f"[info] dry_run={args.dry_run}")

    try:
        targets = list_remote_targets(
            args.remote_host,
            args.remote_user,
            args.remote_port,
            args.remote_outputs_root,
            suffix,
            args.step,
            strict_host_key_checking=args.strict_host_key_checking,
            batch_mode=args.batch_mode,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[error] Failed to query remote outputs: {exc}", file=sys.stderr)
        print(
            "[hint] Try first: "
            f"ssh -p {args.remote_port} -o StrictHostKeyChecking={args.strict_host_key_checking} "
            f"{args.remote_user}@{args.remote_host} 'echo ok'",
            file=sys.stderr,
        )
        print(
            "[hint] If your manual rsync/ssh still needs password or host-key confirmation, "
            "do not enable --batch-mode.",
            file=sys.stderr,
        )
        return exc.returncode or 1

    if not targets:
        print(f"[info] No completed {args.step} checkpoints found for suffix _{suffix}.")
        return 0

    print(f"[info] Found {len(targets)} completed checkpoint(s) to pull.")
    for idx, checkpoint_dir in enumerate(targets, start=1):
        print(f"[info] [{idx}/{len(targets)}] {checkpoint_dir}")
        try:
            pull_remote_checkpoint(
                checkpoint_dir,
                remote_host=args.remote_host,
                remote_user=args.remote_user,
                remote_port=args.remote_port,
                remote_outputs_root=args.remote_outputs_root,
                local_outputs_root=args.local_outputs_root,
                strict_host_key_checking=args.strict_host_key_checking,
                batch_mode=args.batch_mode,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[error] Pull failed for {checkpoint_dir}: {exc}", file=sys.stderr)
            return exc.returncode or 1

    print("[info] Pull finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
