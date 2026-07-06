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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets/${JOB_NAME}/node${RANK}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
mkdir -p "${HF_DATASETS_CACHE}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

: "${NPROC_PER_NODE:?DLC must set NPROC_PER_NODE}"
: "${WORLD_SIZE:?DLC must set WORLD_SIZE}"
: "${RANK:?DLC must set RANK}"
: "${MASTER_ADDR:?DLC must set MASTER_ADDR}"
: "${MASTER_PORT:?DLC must set MASTER_PORT}"

CONFIG_PATH="${CONFIG_PATH:-}"
RESUME_IF_POSSIBLE="${RESUME_IF_POSSIBLE:-0}"

DATASET_ROOTS_FILE="${DATASET_ROOTS_FILE:-${ROOT_DIR}/scripts/sparky_fruit_packing_R8_vertical6Fundist_roots.txt}"
CANONICAL_SIDECAR_ROOT="${CANONICAL_SIDECAR_ROOT:-/mnt/cpfs/datasets/R8_vertical6Fundist_joint_state_sidecars}"
INFO_PATH="${INFO_PATH:-${ROOT_DIR}/shfile/info_groupA_fisheye_only.json}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

BASE_VLM="${BASE_VLM:-/mnt/base/user/xidong/tac_infra/playground/pretrained_models/Qwen3.5-0.8B}"
if [[ -z "${BASE_VLM}" ]]; then
  echo "[error] BASE_VLM is required. Example: export BASE_VLM=/path/to/Qwen-VL-or-Qwen3.5-checkpoint"
  exit 1
fi

PRETRAINED_PATH="${PRETRAINED_PATH:-}"
JOB_NAME="${JOB_NAME:-starvla_groot_sparky_fruitPacking_only_color_R8_vertical_6F}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${ROOT_DIR}/outputs/${JOB_NAME}}"
OUT_DIR="${OUT_DIR:-${RUN_OUTPUT_ROOT}/${RUN_TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/train/${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/node${RANK}_${RUN_TIMESTAMP}.log}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
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

find_latest_checkpoint_dir() {
  local run_root_dir="$1"
  local checkpoints_dir=""

  if [[ -d "${run_root_dir}/checkpoints" ]]; then
    checkpoints_dir="${run_root_dir}/checkpoints"
  else
    local latest_run_dir=""
    latest_run_dir="$(
      find "${run_root_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | grep -E '^[0-9]{8}_[0-9]{6}$|^[0-9]+$' \
        | sort \
        | tail -n 1
    )"
    if [[ -n "${latest_run_dir}" && -d "${run_root_dir}/${latest_run_dir}/checkpoints" ]]; then
      checkpoints_dir="${run_root_dir}/${latest_run_dir}/checkpoints"
    fi
  fi

  if [[ ! -d "${checkpoints_dir}" ]]; then
    return 1
  fi

  local latest_step=""
  latest_step="$(
    find "${checkpoints_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]+$' \
      | sort -n \
      | tail -n 1
  )"
  if [[ -z "${latest_step}" ]]; then
    return 1
  fi
  printf '%s/%s\n' "${checkpoints_dir}" "${latest_step}"
}

POLICY_PATH_ARGS=()
if [[ -n "${PRETRAINED_PATH}" ]]; then
  POLICY_PATH_ARGS=(--policy.pretrained_path="${PRETRAINED_PATH}")
fi

RESUME_ARGS=()
if [[ "${RESUME_IF_POSSIBLE}" == "1" && -z "${CONFIG_PATH}" ]]; then
  latest_checkpoint_dir="$(find_latest_checkpoint_dir "${RUN_OUTPUT_ROOT}" || true)"
  if [[ -n "${latest_checkpoint_dir}" ]]; then
    candidate_config="${latest_checkpoint_dir}/pretrained_model/train_config.json"
    if [[ -f "${candidate_config}" ]]; then
      CONFIG_PATH="${candidate_config}"
    fi
  fi
fi
if [[ -n "${CONFIG_PATH}" ]]; then
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

TEE_ARGS=("${LOG_FILE}")
if [[ -f "${LOG_FILE}" && ( "${RESUME_IF_POSSIBLE}" == "1" || -n "${CONFIG_PATH}" ) ]]; then
  TEE_ARGS=(-a "${LOG_FILE}")
fi

echo "[info] job_name=${JOB_NAME}"
echo "[info] rank=${RANK}/${WORLD_SIZE} nproc_per_node=${NPROC_PER_NODE}"
echo "[info] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[info] roots_file=${DATASET_ROOTS_FILE}"
echo "[info] sidecar_root=${CANONICAL_SIDECAR_ROOT}"
echo "[info] info_path=${INFO_PATH}"
echo "[info] base_vlm=${BASE_VLM}"
echo "[info] pretrained_path=${PRETRAINED_PATH:-<none>}"
echo "[info] hf_datasets_cache=${HF_DATASETS_CACHE}"
echo "[info] batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[info] steps=${MAX_STEPS} save_freq=${SAVE_FREQ} eval_freq=${EVAL_FREQ}"
echo "[info] chunk_size=${CHUNK_SIZE} n_action_steps=${N_ACTION_STEPS}"
echo "[info] action_gap_frames=${ACTION_GAP_FRAMES}"
echo "[info] apply_action_gap_to_real_datasets=${APPLY_ACTION_GAP_TO_REAL_DATASETS}"
echo "[info] state_dim=${STATE_DIM} action_dim=${ACTION_DIM}"
echo "[info] use_state_input=${USE_STATE_INPUT} state_mode=${STATE_MODE}"
echo "[info] use_relative_actions=${USE_RELATIVE_ACTIONS}"
echo "[info] relative_action_keyframes=${RELATIVE_ACTION_KEYFRAMES}"
echo "[info] relative_exclude_keys=${RELATIVE_EXCLUDE_KEYS}"
echo "[info] relative_exclude_joints=${RELATIVE_EXCLUDE_JOINTS}"
echo "[info] wrist_only=${WRIST_ONLY}"
echo "[info] top_camera_keys=${TOP_CAMERA_KEYS}"
echo "[info] wrist_camera_keys=${WRIST_CAMERA_KEYS}"
echo "[info] tactile_mode=${TACTILE_MODE}"
echo "[info] action_model_type=${ACTION_MODEL_TYPE}"
echo "[info] attn_implementation=${ATTN_IMPLEMENTATION}"
echo "[info] repeated_diffusion_steps=${REPEATED_DIFFUSION_STEPS}"
echo "[info] num_inference_timesteps=${NUM_INFERENCE_TIMESTEPS}"
echo "[info] freeze_vision_encoder=${FREEZE_VISION_ENCODER}"
echo "[info] train_expert_only=${TRAIN_EXPERT_ONLY}"
echo "[info] output_dir=${OUT_DIR}"
echo "[info] log_file=${LOG_FILE}"
if [[ -n "${CONFIG_PATH}" ]]; then
  echo "[info] resume=true config_path=${CONFIG_PATH}"
fi

set +e
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/lerobot/scripts/lerobot_train.py \
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
    2>&1 | tee "${TEE_ARGS[@]}"
train_exit="${PIPESTATUS[0]}"
set -e

echo "[done] rank=${RANK} train_exit=${train_exit} log=${LOG_FILE}"
exit "${train_exit}"
