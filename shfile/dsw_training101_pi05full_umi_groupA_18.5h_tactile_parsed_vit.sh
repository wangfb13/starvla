#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/shfile/dsw_training101_pi05full_umi_groupA_18.5h.sh"

export ENABLE_TACTILE="${ENABLE_TACTILE:-true}"
export TACTILE_ENCODER_TYPE="${TACTILE_ENCODER_TYPE:-parsed_vit}"
export TACTILE_FEATURE_KEYS="${TACTILE_FEATURE_KEYS:-[\"observation.depth_deformation.tactile_left_left\",\"observation.depth_deformation.tactile_left_right\",\"observation.depth_deformation.tactile_right_left\",\"observation.depth_deformation.tactile_right_right\"]}"
export JOB_NAME="${JOB_NAME:-pi05full_umi_groupA_18.5h_tactile_parsed_vit}"

exec bash "${BASE_SCRIPT}" "$@"
