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
export LEROBOT_EVAL_LEFT_ARM_PRESENT="${LEROBOT_EVAL_LEFT_ARM_PRESENT:-1}"
export LEROBOT_EVAL_RIGHT_ARM_PRESENT="${LEROBOT_EVAL_RIGHT_ARM_PRESENT:-1}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

find_latest_pretrained_model() {
  local outputs_root="$1"
  if [[ ! -d "${outputs_root}" ]]; then
    return 1
  fi

  local latest_run=""
  latest_run="$(
    find "${outputs_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]{8}_[0-9]{6}$|^[0-9]+$' \
      | sort \
      | tail -n 1
  )"
  if [[ -z "${latest_run}" ]]; then
    return 1
  fi

  local run_dir="${outputs_root}/${latest_run}"
  local latest_step=""
  latest_step="$(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]+$' \
      | sort -n \
      | tail -n 1
  )"
  if [[ -z "${latest_step}" ]]; then
    return 1
  fi

  local candidate="${run_dir}/${latest_step}/pretrained_model"
  if [[ ! -d "${candidate}" ]]; then
    return 1
  fi
  printf '%s\n' "${candidate}"
}

JOB_NAME="${JOB_NAME:-starvla_groot_sparky_fruitPacking_only_color_R8_vertical_6F}"
POLICY_PATH="${1:-${POLICY_PATH:-/home/dm/DM_das/outputs/starvla/starvla_sparky_fruitPacking_only_color_R8_vf6Easyundist_0705gap6/020000/pretrained_model}}"
if [[ -z "${POLICY_PATH}" ]]; then
  POLICY_PATH="$(find_latest_pretrained_model "${ROOT_DIR}/outputs/${JOB_NAME}" || true)"
fi

COMMAND_MODE="${COMMAND_MODE:-joint}"
JOINT_COMMAND_TOPIC="${JOINT_COMMAND_TOPIC:-/joint_command_stream}"
JOINT_STATE_TOPIC="${JOINT_STATE_TOPIC:-/joint_states}"
ACTION_JOINT_NAMES="${ACTION_JOINT_NAMES:-[\"main_arm_left_1\",\"main_arm_left_2\",\"main_arm_left_3\",\"main_arm_left_4\",\"main_arm_left_5\",\"main_arm_left_6\",\"main_arm_left_7\",\"main_gripper_left\",\"main_arm_right_1\",\"main_arm_right_2\",\"main_arm_right_3\",\"main_arm_right_4\",\"main_arm_right_5\",\"main_arm_right_6\",\"main_arm_right_7\",\"main_gripper_right\"]}"
JOINT_COMMAND_NAMES="${JOINT_COMMAND_NAMES:-[\"main_arm_right_1\",\"main_arm_right_2\",\"main_arm_right_3\",\"main_arm_right_4\",\"main_arm_right_5\",\"main_arm_right_6\",\"main_arm_right_7\",\"main_arm_left_1\",\"main_arm_left_2\",\"main_arm_left_3\",\"main_arm_left_4\",\"main_arm_left_5\",\"main_arm_left_6\",\"main_arm_left_7\",\"main_hip_yaw\",\"main_hip_pitch\",\"main_knee\",\"main_head_pitch\",\"main_head_yaw\"]}"

LEFT_X5_IP="${LEFT_X5_IP:-192.168.9.10}"
RIGHT_X5_IP="${RIGHT_X5_IP:-192.168.9.11}"
ENABLE_UNDISTORT_CROP="${ENABLE_UNDISTORT_CROP:-true}"
UNDISTORT_CROP_SIZE="${UNDISTORT_CROP_SIZE:-896}"
INTRINSICS_PATH_LEFT="${INTRINSICS_PATH_LEFT:-/home/dm/DM_das/intrinsics_data/wrist_fishCamera_192_168_9_10.yaml}"
INTRINSICS_PATH_RIGHT="${INTRINSICS_PATH_RIGHT:-/home/dm/DM_das/intrinsics_data/wrist_fishCamera_192_168_9_11.yaml}"

RUN_TS="${RUN_TS:-$(date '+%Y%m%d_%H%M%S')}"
POLICY_TAG="${POLICY_TAG:-$(basename "$(dirname "${POLICY_PATH:-unknown}")" 2>/dev/null || echo unknown)}"
EVAL_NAME="${EVAL_NAME:-eval_${JOB_NAME}_joint_${POLICY_TAG}_${RUN_TS}}"
DATASET_ROOT_BASE="${DATASET_ROOT_BASE:-/home/dm/DM_das/datasets}"
TASK_TEXT="${TASK_TEXT:-fruit packing}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-600}"
RESET_TIME_S="${RESET_TIME_S:-5}"
LEFT_GRIPPER_OFFSET="${LEFT_GRIPPER_OFFSET:-0.0}"
RIGHT_GRIPPER_OFFSET="${RIGHT_GRIPPER_OFFSET:-0.0}"
INTERPOLATION_MULTIPLIER="${INTERPOLATION_MULTIPLIER:-1}"
ACTION_SKIP_START_STEPS="${ACTION_SKIP_START_STEPS:-0}"
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"

if [[ ! -f "${INTRINSICS_PATH_LEFT}" ]]; then
  echo "[error] Missing intrinsics file: ${INTRINSICS_PATH_LEFT}"
  exit 1
fi

if [[ ! -f "${INTRINSICS_PATH_RIGHT}" ]]; then
  echo "[error] Missing intrinsics file: ${INTRINSICS_PATH_RIGHT}"
  exit 1
fi

if [[ -z "${POLICY_PATH}" || ! -d "${POLICY_PATH}" ]]; then
  echo "[error] Missing policy checkpoint dir: ${POLICY_PATH:-<empty>}"
  echo "[hint] Set POLICY_PATH=/path/to/pretrained_model or pass it as the first argument."
  exit 1
fi

echo "[info] root_dir=${ROOT_DIR}"
echo "[info] job_name=${JOB_NAME}"
echo "[info] policy_path=${POLICY_PATH}"
echo "[info] eval_name=${EVAL_NAME}"
echo "[info] dataset_root=${DATASET_ROOT_BASE}/${EVAL_NAME}"
echo "[info] command_mode=${COMMAND_MODE}"
echo "[info] joint_command_topic=${JOINT_COMMAND_TOPIC}"
echo "[info] joint_state_topic=${JOINT_STATE_TOPIC}"
echo "[info] action_joint_names=${ACTION_JOINT_NAMES}"
echo "[info] joint_command_names=${JOINT_COMMAND_NAMES}"
echo "[info] enable_undistort_crop=${ENABLE_UNDISTORT_CROP}"
echo "[info] undistort_crop_size=${UNDISTORT_CROP_SIZE}"
echo "[info] task_text=${TASK_TEXT}"
echo "[info] action_skip_start_steps=${ACTION_SKIP_START_STEPS}"
echo "[info] python3=$(command -v python3)"

EXPECTED_LEROBOT_SRC="${ROOT_DIR}/src"
export EXPECTED_LEROBOT_SRC
python3 - <<'PY'
import inspect
import os
import sys

expected = os.path.realpath(os.environ["EXPECTED_LEROBOT_SRC"])
import lerobot

lerobot_path = os.path.realpath(inspect.getsourcefile(lerobot) or "")
print(f"[info] imported_lerobot={lerobot_path}")
if not lerobot_path.startswith(expected + os.sep):
    print(
        f"[error] Imported lerobot is not from current repo.\n"
        f"        expected prefix: {expected}\n"
        f"        actual path:     {lerobot_path}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

python3 "${ROOT_DIR}/src/lerobot/scripts/lerobot_record.py" \
  --robot.type=spark3_pose_gripper \
  --robot.command_mode="${COMMAND_MODE}" \
  --robot.joint_command_topic="${JOINT_COMMAND_TOPIC}" \
  --robot.joint_state_topic="${JOINT_STATE_TOPIC}" \
  --robot.action_joint_names="${ACTION_JOINT_NAMES}" \
  --robot.joint_command_names="${JOINT_COMMAND_NAMES}" \
  --robot.cameras='{
    "cam_right_wrist": {
      "type": "wirstFisheyeX5",
      "x5_ip": "'"${RIGHT_X5_IP}"'",
      "width": '"${UNDISTORT_CROP_SIZE}"',
      "height": '"${UNDISTORT_CROP_SIZE}"',
      "fps": 30,
      "intrinsics_file": "'"${INTRINSICS_PATH_RIGHT}"'",
      "enable_undistort_crop": '"${ENABLE_UNDISTORT_CROP}"',
      "undistort_crop_size": '"${UNDISTORT_CROP_SIZE}"'
    },
    "cam_left_wrist": {
      "type": "wirstFisheyeX5",
      "x5_ip": "'"${LEFT_X5_IP}"'",
      "width": '"${UNDISTORT_CROP_SIZE}"',
      "height": '"${UNDISTORT_CROP_SIZE}"',
      "fps": 30,
      "intrinsics_file": "'"${INTRINSICS_PATH_LEFT}"'",
      "enable_undistort_crop": '"${ENABLE_UNDISTORT_CROP}"',
      "undistort_crop_size": '"${UNDISTORT_CROP_SIZE}"'
    }
  }' \
  --display_data=false \
  --play_sounds="${PLAY_SOUNDS}" \
  --dataset.repo_id="/${EVAL_NAME}" \
  --dataset.root="${DATASET_ROOT_BASE}/${EVAL_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.single_task="${TASK_TEXT}" \
  --dataset.streaming_encoding=true \
  --dataset.vcodec=libsvtav1 \
  --dataset.encoder_threads=2 \
  --dataset.rename_map='{"observation.images.cam_right_wrist":"observation.images.cam_right","observation.images.cam_left_wrist":"observation.images.cam_left"}' \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.action_execution_skip_start_steps="${ACTION_SKIP_START_STEPS}" \
  --gripper_offsets.left="${LEFT_GRIPPER_OFFSET}" \
  --gripper_offsets.right="${RIGHT_GRIPPER_OFFSET}" \
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER}"
