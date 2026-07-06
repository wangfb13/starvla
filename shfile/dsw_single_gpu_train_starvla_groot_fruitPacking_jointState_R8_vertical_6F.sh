#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-starvla}"
  conda activate "${CONDA_ENV_NAME}"
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-/mnt/tacumi_data/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets/dsw_single_gpu_starvla}"
mkdir -p "${HF_DATASETS_CACHE}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

DATASET_ROOTS_FILE="${DATASET_ROOTS_FILE:-${ROOT_DIR}/scripts/sparky_fruit_packing_R8_vertical6Fundist_roots.txt}"
CANONICAL_SIDECAR_ROOT="${CANONICAL_SIDECAR_ROOT:-/mnt/cpfs/datasets/R8_vertical6Fundist_joint_state_sidecars}"
INFO_PATH="${INFO_PATH:-${ROOT_DIR}/shfile/info_groupA_fisheye_only.json}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

BASE_VLM="${BASE_VLM:-/mnt/base/user/xidong/tac_infra/playground/pretrained_models/Qwen3.5-0.8B}"
if [[ -z "${BASE_VLM}" ]]; then
  echo "[error] BASE_VLM is required."
  echo "        Example:"
  echo "        export BASE_VLM=/path/to/your/base_vlm_checkpoint"
  exit 1
fi

PRETRAINED_PATH="${PRETRAINED_PATH:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
RESUME="${RESUME:-false}"

JOB_NAME="${JOB_NAME:-starvla_groot_sparky_fruitPacking_jointState_R8_vertical_6F_dsw}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${ROOT_DIR}/outputs/${JOB_NAME}}"
OUT_DIR="${OUT_DIR:-${RUN_OUTPUT_ROOT}/${RUN_TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/train/${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TIMESTAMP}.log}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-30000}"
LOG_FREQ="${LOG_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_FREQ="${EVAL_FREQ:-0}"

CHUNK_SIZE="${CHUNK_SIZE:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
ACTION_GAP_FRAMES="${ACTION_GAP_FRAMES:-6}"
APPLY_ACTION_GAP_TO_REAL_DATASETS="${APPLY_ACTION_GAP_TO_REAL_DATASETS:-true}"
STATE_DIM="${STATE_DIM:-18}"
ACTION_DIM="${ACTION_DIM:-16}"
USE_STATE_INPUT="${USE_STATE_INPUT:-true}"
STATE_MODE="${STATE_MODE:-raw}"
USE_RELATIVE_ACTIONS="${USE_RELATIVE_ACTIONS:-false}"
RELATIVE_ACTION_KEYFRAMES="${RELATIVE_ACTION_KEYFRAMES:-[\"0:3\",\"10:13\"]}"
RELATIVE_EXCLUDE_KEYS="${RELATIVE_EXCLUDE_KEYS:-[]}"
RELATIVE_EXCLUDE_JOINTS="${RELATIVE_EXCLUDE_JOINTS:-[]}"
CANONICALIZE_BIMANUAL_ROT6D="${CANONICALIZE_BIMANUAL_ROT6D:-false}"
CANONICAL_MISSING_VALUE="${CANONICAL_MISSING_VALUE:-9930.0}"

ACTION_MODEL_TYPE="${ACTION_MODEL_TYPE:-DiT-B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
REPEATED_DIFFUSION_STEPS="${REPEATED_DIFFUSION_STEPS:-8}"
NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-4}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"

WRIST_ONLY="${WRIST_ONLY:-true}"
TOP_CAMERA_KEYS="${TOP_CAMERA_KEYS:-[]}"
WRIST_CAMERA_KEYS="${WRIST_CAMERA_KEYS:-[\"observation.images.cam_left\",\"observation.images.cam_right\"]}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-[224,224]}"
TACTILE_MODE="${TACTILE_MODE:-none}"
TACTILE_KEYS="${TACTILE_KEYS:-[\"observation.depth_deformation.tactile_left_left\",\"observation.depth_deformation.tactile_left_right\",\"observation.depth_deformation.tactile_right_left\",\"observation.depth_deformation.tactile_right_right\"]}"
TACTILE_ENCODER_PATH="${TACTILE_ENCODER_PATH:-}"
TACTILE_INSERT_LOCATION="${TACTILE_INSERT_LOCATION:-decoder}"

mkdir -p "${RUN_OUTPUT_ROOT}" "${LOG_DIR}"

POLICY_PATH_ARGS=()
if [[ -n "${PRETRAINED_PATH}" ]]; then
  POLICY_PATH_ARGS=(--policy.pretrained_path="${PRETRAINED_PATH}")
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "true" ]]; then
  if [[ -z "${CONFIG_PATH}" ]]; then
    echo "[error] RESUME=true requires CONFIG_PATH."
    exit 1
  fi
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[error] CONFIG_PATH does not exist: ${CONFIG_PATH}"
    exit 1
  fi
  RESUME_ARGS=(--resume=true "--config_path=${CONFIG_PATH}")
  POLICY_PATH_ARGS=()
fi

TACTILE_ARGS=()
if [[ "${TACTILE_MODE}" != "none" ]]; then
  TACTILE_ARGS=(
    --policy.tactile_mode="${TACTILE_MODE}"
    --policy.tactile_keys="${TACTILE_KEYS}"
    --policy.tactile_insert_location="${TACTILE_INSERT_LOCATION}"
  )
  if [[ -n "${TACTILE_ENCODER_PATH}" ]]; then
    TACTILE_ARGS+=(--policy.tactile_encoder_path="${TACTILE_ENCODER_PATH}")
  fi
fi

echo "[info] dataset_roots_file=${DATASET_ROOTS_FILE}"
echo "[info] sidecar_root=${CANONICAL_SIDECAR_ROOT}"
echo "[info] info_path=${INFO_PATH}"
echo "[info] base_vlm=${BASE_VLM}"
echo "[info] pretrained_path=${PRETRAINED_PATH:-<none>}"
echo "[info] batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[info] steps=${MAX_STEPS} save_freq=${SAVE_FREQ}"
echo "[info] state_dim=${STATE_DIM} action_dim=${ACTION_DIM}"
echo "[info] output_dir=${OUT_DIR}"
echo "[info] log_file=${LOG_FILE}"
if [[ "${RESUME}" == "true" ]]; then
  echo "[info] resume=true config_path=${CONFIG_PATH}"
fi

set +e
python3 src/lerobot/scripts/lerobot_train.py \
  --dataset.roots_file="${DATASET_ROOTS_FILE}" \
  --dataset.sidecar_root="${CANONICAL_SIDECAR_ROOT}" \
  --dataset.info_path="${INFO_PATH}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --policy.type=starvla_groot \
  "${POLICY_PATH_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --policy.repo_id="${JOB_NAME}" \
  --policy.base_vlm="${BASE_VLM}" \
  --policy.attn_implementation="${ATTN_IMPLEMENTATION}" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.action_gap_frames="${ACTION_GAP_FRAMES}" \
  --policy.apply_action_gap_to_real_datasets="${APPLY_ACTION_GAP_TO_REAL_DATASETS}" \
  --policy.action_model_type="${ACTION_MODEL_TYPE}" \
  --policy.repeated_diffusion_steps="${REPEATED_DIFFUSION_STEPS}" \
  --policy.num_inference_timesteps="${NUM_INFERENCE_TIMESTEPS}" \
  --policy.state_mode="${STATE_MODE}" \
  --policy.use_state_input="${USE_STATE_INPUT}" \
  --policy.use_relative_actions="${USE_RELATIVE_ACTIONS}" \
  --policy.relative_action_keyframes="${RELATIVE_ACTION_KEYFRAMES}" \
  --policy.relative_exclude_keys="${RELATIVE_EXCLUDE_KEYS}" \
  --policy.relative_exclude_joints="${RELATIVE_EXCLUDE_JOINTS}" \
  --policy.state_dim="${STATE_DIM}" \
  --policy.action_dim="${ACTION_DIM}" \
  --policy.canonicalize_bimanual_rot6d="${CANONICALIZE_BIMANUAL_ROT6D}" \
  --policy.canonical_missing_value="${CANONICAL_MISSING_VALUE}" \
  --policy.wrist_only="${WRIST_ONLY}" \
  --policy.top_camera_keys="${TOP_CAMERA_KEYS}" \
  --policy.wrist_camera_keys="${WRIST_CAMERA_KEYS}" \
  --policy.image_resolution="${IMAGE_RESOLUTION}" \
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
  --policy.push_to_hub=false \
  "${TACTILE_ARGS[@]}" \
  --steps="${MAX_STEPS}" \
  --log_freq="${LOG_FREQ}" \
  --save_freq="${SAVE_FREQ}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --eval_freq="${EVAL_FREQ}" \
  --wandb.enable=false \
  --output_dir="${OUT_DIR}" \
  --job_name="${JOB_NAME}" \
  2>&1 | tee "${LOG_FILE}"
train_exit="${PIPESTATUS[0]}"
set -e

echo "[done] train_exit=${train_exit} log=${LOG_FILE}"
exit "${train_exit}"
