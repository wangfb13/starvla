#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/shfile/dlc_torchrun_train_pi05full_only_color_fruitPacking_sparky_withstate.sh"

export ENABLE_TACTILE="${ENABLE_TACTILE:-true}"
export ENABLE_SUBTASK_BRANCH="${ENABLE_SUBTASK_BRANCH:-true}"
export TACTILE_ENCODER_TYPE="${TACTILE_ENCODER_TYPE:-parsed_vit}"
export TACTILE_FEATURE_KEYS="${TACTILE_FEATURE_KEYS:-[\"observation.depth_deformation.tactile_left_left\",\"observation.depth_deformation.tactile_left_right\",\"observation.depth_deformation.tactile_right_left\",\"observation.depth_deformation.tactile_right_right\"]}"
export RESET_ACTION_EXPERT_AFTER_PRETRAINED_LOAD="${RESET_ACTION_EXPERT_AFTER_PRETRAINED_LOAD:-false}"
# export TACTILE_PATCH_ENCODER_DIM="${TACTILE_PATCH_ENCODER_DIM:-128}"
export JOB_NAME="${JOB_NAME:-pi05full_sparky_fruitPacking_only_color_dlc_torchrun_tactile_parsed_vit}"

exec bash "${BASE_SCRIPT}" "$@"
