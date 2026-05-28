#!/usr/bin/env bash

set -euo pipefail

ROOTS_FILE="${1:-${ROOTS_FILE:-/mnt/base/user/zfc/lerobot_vtla/repo/pi05_data_sidecar/groupA_roots.txt}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-/mnt/cpfs/datasets/pi05_groupA_train_roots_symlink}}"

usage() {
    cat <<'EOF'
Usage:
  bash shfile/create_dataset_symlinks_from_txt.sh [ROOTS_FILE] [OUTPUT_DIR]

Environment overrides:
  ROOTS_FILE=/path/to/groupA_roots.txt
  OUTPUT_DIR=/path/to/output_parent_dir

After creating links, train with:
  --dataset.root=<OUTPUT_DIR>

Notes:
  - ROOTS_FILE should contain one LeRobot dataset root per line.
  - Empty lines and lines starting with # are ignored.
  - Existing symlinks with the same generated names are updated.
  - Existing non-symlink paths are left untouched and cause an error.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "${ROOTS_FILE}" ]]; then
    echo "[error] ROOTS_FILE does not exist: ${ROOTS_FILE}" >&2
    usage >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

count=0
skipped=0
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%%#*}"
    line="$(printf '%s' "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [[ -z "${line}" ]]; then
        continue
    fi

    if [[ ! -d "${line}" ]]; then
        echo "[warn] skip missing dataset root: ${line}" >&2
        skipped=$((skipped + 1))
        continue
    fi

    if [[ ! -f "${line}/meta/info.json" || ! -d "${line}/data" ]]; then
        echo "[warn] skip non-LeRobot dataset root: ${line}" >&2
        skipped=$((skipped + 1))
        continue
    fi

    basename="$(basename "${line}")"
    link_name="$(printf '%03d__%s' "${count}" "${basename}")"
    link_path="${OUTPUT_DIR}/${link_name}"

    if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
        echo "[error] refusing to replace non-symlink path: ${link_path}" >&2
        exit 1
    fi

    ln -sfn "${line}" "${link_path}"
    echo "[link] ${link_path} -> ${line}"
    count=$((count + 1))
done < "${ROOTS_FILE}"

echo "[done] created ${count} dataset symlinks under ${OUTPUT_DIR}"
if [[ "${skipped}" -gt 0 ]]; then
    echo "[done] skipped ${skipped} invalid or missing roots"
fi
echo "[hint] use: --dataset.root=${OUTPUT_DIR}"
