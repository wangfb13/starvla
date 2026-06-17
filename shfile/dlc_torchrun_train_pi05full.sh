#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot_v0.5.1}"
  conda activate "${CONDA_ENV_NAME}"
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-/mnt/tacumi_data/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# DLC injects these distributed variables. Do not set local fallbacks here,
# otherwise a missing cloud variable could silently turn a multi-node job into
# a single-node launch.
: "${NPROC_PER_NODE:?DLC must set NPROC_PER_NODE}"
: "${WORLD_SIZE:?DLC must set WORLD_SIZE}"
: "${RANK:?DLC must set RANK}"
: "${MASTER_ADDR:?DLC must set MASTER_ADDR}"
: "${MASTER_PORT:?DLC must set MASTER_PORT}"

CONFIG_PATH="${CONFIG_PATH:-}"
RESUME_IF_POSSIBLE="${RESUME_IF_POSSIBLE:-0}"

DATASET_ROOTS_FILE="${DATASET_ROOTS_FILE:-${ROOT_DIR}/scripts/groupA_roots.txt}"
CANONICAL_SIDECAR_ROOT="${CANONICAL_SIDECAR_ROOT:-/mnt/cpfs/datasets/pi05_canonical_sidecars_umi}"
INFO_PATH="${INFO_PATH:-${ROOT_DIR}/shfile/info_groupA_fisheye_only.json}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"
TOKENIZER_NAME="${TOKENIZER_NAME:-/mnt/tacumi_data/huggingface/paligemma-3b-pt-224}"

JOB_NAME="${JOB_NAME:-pi05full_umi_groupA_dlc_torchrun}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${ROOT_DIR}/outputs/${JOB_NAME}}"
OUT_DIR="${OUT_DIR:-${RUN_OUTPUT_ROOT}/${RUN_TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/train/${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/node${RANK}_${RUN_TIMESTAMP}.log}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-30000}"
LOG_FREQ="${LOG_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_FREQ="${EVAL_FREQ:-0}"

ACTION_GAP_FRAMES="${ACTION_GAP_FRAMES:-6}"
RELATIVE_ACTION_KEYFRAMES="${RELATIVE_ACTION_KEYFRAMES:-[\"0:3\",\"10:13\"]}"
RELATIVE_EXCLUDE_KEYS="${RELATIVE_EXCLUDE_KEYS:-[]}"
RELATIVE_EXCLUDE_JOINTS="${RELATIVE_EXCLUDE_JOINTS:-[]}"
USE_STATE_INPUT="${USE_STATE_INPUT:-false}"
ENABLE_SUBTASK_BRANCH="${ENABLE_SUBTASK_BRANCH:-true}"
LOSS_WEIGHT_SUBTASK_CE="${LOSS_WEIGHT_SUBTASK_CE:-1.0}"
RESET_ACTION_EXPERT_AFTER_PRETRAINED_LOAD="${RESET_ACTION_EXPERT_AFTER_PRETRAINED_LOAD:-false}"
NORMALIZATION_MAPPING="${NORMALIZATION_MAPPING:-}"
ENABLE_TACTILE="${ENABLE_TACTILE:-false}"
TACTILE_ENCODER_TYPE="${TACTILE_ENCODER_TYPE:-parsed_vit}"
TACTILE_FEATURE_KEYS="${TACTILE_FEATURE_KEYS:-[\"observation.depth_deformation.tactile_left_left\",\"observation.depth_deformation.tactile_left_right\",\"observation.depth_deformation.tactile_right_left\",\"observation.depth_deformation.tactile_right_right\"]}"

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

POLICY_PATH_ARGS=(--policy.pretrained_path="${PRETRAINED_PATH}")
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

NORMALIZATION_ARGS=()
if [[ -n "${NORMALIZATION_MAPPING}" ]]; then
  NORMALIZATION_ARGS=(--policy.normalization_mapping="${NORMALIZATION_MAPPING}")
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
echo "[info] pretrained_path=${PRETRAINED_PATH}"
echo "[info] tokenizer_name=${TOKENIZER_NAME}"
echo "[info] batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[info] steps=${MAX_STEPS} save_freq=${SAVE_FREQ} eval_freq=${EVAL_FREQ}"
echo "[info] action_gap_frames=${ACTION_GAP_FRAMES}"
echo "[info] relative_action_keyframes=${RELATIVE_ACTION_KEYFRAMES}"
echo "[info] relative_exclude_keys=${RELATIVE_EXCLUDE_KEYS}"
echo "[info] relative_exclude_joints=${RELATIVE_EXCLUDE_JOINTS}"
echo "[info] use_state_input=${USE_STATE_INPUT}"
echo "[info] enable_subtask_branch=${ENABLE_SUBTASK_BRANCH}"
echo "[info] loss_weight_subtask_ce=${LOSS_WEIGHT_SUBTASK_CE}"
echo "[info] reset_action_expert_after_pretrained_load=${RESET_ACTION_EXPERT_AFTER_PRETRAINED_LOAD}"
echo "[info] normalization_mapping=${NORMALIZATION_MAPPING:-<default>}"
echo "[info] enable_tactile=${ENABLE_TACTILE}"
echo "[info] tactile_encoder_type=${TACTILE_ENCODER_TYPE}"
echo "[info] tactile_feature_keys=${TACTILE_FEATURE_KEYS}"
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
    --policy.type=pi05_full \
    "${POLICY_PATH_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --policy.repo_id="${JOB_NAME}" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.tokenizer_name="${TOKENIZER_NAME}" \
    --policy.use_relative_actions=true \
    --policy.use_state_input="${USE_STATE_INPUT}" \
    --policy.enable_subtask_branch="${ENABLE_SUBTASK_BRANCH}" \
    --policy.reset_action_expert_after_pretrained_load="${RESET_ACTION_EXPERT_AFTER_PRETRAINED_LOAD}" \
    --policy.relative_action_keyframes="${RELATIVE_ACTION_KEYFRAMES}" \
    --policy.relative_exclude_keys="${RELATIVE_EXCLUDE_KEYS}" \
    --policy.relative_exclude_joints="${RELATIVE_EXCLUDE_JOINTS}" \
    --policy.action_gap_frames="${ACTION_GAP_FRAMES}" \
    --policy.loss_weight_subtask_ce="${LOSS_WEIGHT_SUBTASK_CE}" \
    --policy.canonicalize_bimanual_rot6d=false \
    --policy.max_state_dim=22 \
    --policy.max_action_dim=20 \
    --policy.canonical_missing_value=9930.0 \
    --policy.push_to_hub=false \
    --policy.enable_tactile="${ENABLE_TACTILE}" \
    --policy.tactile_encoder_type="${TACTILE_ENCODER_TYPE}" \
    --policy.tactile_feature_keys="${TACTILE_FEATURE_KEYS}" \
    "${NORMALIZATION_ARGS[@]}" \
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
