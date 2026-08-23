#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-starvla}"
  conda activate "${CONDA_ENV_NAME}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
POLICY_PATH="${1:-${POLICY_PATH:-/home/dm/DM_das/starvla-feat-sparky_x5_starvla/outputs/umi_5pz_starvla/50000_resolute/pretrained_model}}"
BASE_VLM_PATH="${BASE_VLM_PATH:-${BASE_VLM:-/home/dm/DM_das/checkpoints/Qwen3.5-0.8B}}"
TASK_TEXT="${TASK_TEXT:-Stack the red cup on top of another cup.}"

COMMAND_MODE="${COMMAND_MODE:-pose}"
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
LEFT_GRIPPER_OFFSET="${LEFT_GRIPPER_OFFSET:-0.0}"
RIGHT_GRIPPER_OFFSET="${RIGHT_GRIPPER_OFFSET:-0.0}"
ACTION_SKIP_START_STEPS="${ACTION_SKIP_START_STEPS:-0}"
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"

if [[ -z "${POLICY_PATH}" ]]; then
  echo "Usage: POLICY_PATH=/path/to/checkpoint/pretrained_model $0" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${POLICY_PATH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
    if not (p / name).is_file():
        raise SystemExit(f"Missing checkpoint file: {p / name}")
cfg = json.loads((p / "config.json").read_text())
expected = {"type":"starvla_groot", "chunk_size":32, "n_action_steps":16,
            "action_gap_frames":4, "use_state_input":True, "state_dim":22, "action_dim":20,
            "use_relative_actions":True}
bad = {k:(cfg.get(k), v) for k,v in expected.items() if cfg.get(k) != v}
if bad:
    raise SystemExit(f"Relative checkpoint config mismatch: {bad}")
if cfg.get("relative_action_keyframes") != ["0:3", "10:13"]:
    raise SystemExit(f"Unexpected relative keyframes: {cfg.get('relative_action_keyframes')}")
print("[PASS] relative checkpoint: local-TCP SE(3), chunk32/execute16/delay4, state22")
PY

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LEROBOT_EVAL_LEFT_ARM_PRESENT=1
export LEROBOT_EVAL_RIGHT_ARM_PRESENT=1
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

for required in "${INTRINSICS_PATH_LEFT}" "${INTRINSICS_PATH_RIGHT}"; do
  [[ -f "${required}" ]] || { echo "[error] Missing required file: ${required}" >&2; exit 1; }
done
[[ -d "${BASE_VLM_PATH}" ]] || { echo "[error] Missing Qwen base model: ${BASE_VLM_PATH}" >&2; exit 1; }
[[ "${COMMAND_MODE}" == pose ]] || { echo "[error] COMMAND_MODE must be pose" >&2; exit 1; }

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_NAME="eval_starvla_groot_relative_${RUN_ID}"
DATASET_ROOT_BASE="${DATASET_ROOT_BASE:-${ROOT_DIR}/datasets/eval}"

echo "[info] mode=relative policy_path=${POLICY_PATH}"
echo "[info] task=${TASK_TEXT} chunk=32 execute=16 delay=4 skip=${ACTION_SKIP_START_STEPS}"
echo "[info] DiT state is relative to the first TCP state after episode reset"
echo "[info] relative TCP action is decoded against the current raw robot base-frame TCP; no QR calibration"

"${PYTHON_BIN}" src/lerobot/scripts/lerobot_record.py \
  --robot.type=spark3_pose_gripper \
  --robot.command_mode="${COMMAND_MODE}" \
  --robot.joint_command_topic="${JOINT_COMMAND_TOPIC}" \
  --robot.joint_state_topic="${JOINT_STATE_TOPIC}" \
  --robot.action_joint_names="${ACTION_JOINT_NAMES}" \
  --robot.joint_command_names="${JOINT_COMMAND_NAMES}" \
  --robot.cameras='{"cam_right_wrist":{"type":"wirstFisheyeX5","x5_ip":"'"${RIGHT_X5_IP}"'","width":'"${UNDISTORT_CROP_SIZE}"',"height":'"${UNDISTORT_CROP_SIZE}"',"fps":30,"intrinsics_file":"'"${INTRINSICS_PATH_RIGHT}"'","enable_undistort_crop":'"${ENABLE_UNDISTORT_CROP}"',"undistort_crop_size":'"${UNDISTORT_CROP_SIZE}"'},"cam_left_wrist":{"type":"wirstFisheyeX5","x5_ip":"'"${LEFT_X5_IP}"'","width":'"${UNDISTORT_CROP_SIZE}"',"height":'"${UNDISTORT_CROP_SIZE}"',"fps":30,"intrinsics_file":"'"${INTRINSICS_PATH_LEFT}"'","enable_undistort_crop":'"${ENABLE_UNDISTORT_CROP}"',"undistort_crop_size":'"${UNDISTORT_CROP_SIZE}"'}}' \
  --dataset.repo_id="/${EVAL_NAME}" \
  --dataset.root="${DATASET_ROOT_BASE}/${EVAL_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes="${NUM_EPISODES:-1}" \
  --dataset.episode_time_s="${EPISODE_TIME_S:-30}" \
  --dataset.reset_time_s="${RESET_TIME_S:-10}" \
  --dataset.single_task="${TASK_TEXT}" \
  --dataset.streaming_encoding=true \
  --dataset.vcodec=libsvtav1 \
  --dataset.encoder_threads=2 \
  --dataset.rename_map='{"observation.images.cam_right_wrist":"observation.images.cam_right_undist","observation.images.cam_left_wrist":"observation.images.cam_left_undist"}' \
  --policy.path="${POLICY_PATH}" \
  --policy.base_vlm="${BASE_VLM_PATH}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.online_anchor_state_to_episode_start=true \
  --policy.action_execution_skip_start_steps="${ACTION_SKIP_START_STEPS}" \
  --gripper_offsets.left="${LEFT_GRIPPER_OFFSET}" \
  --gripper_offsets.right="${RIGHT_GRIPPER_OFFSET}" \
  --interpolation_multiplier="${INTERPOLATION_MULTIPLIER:-1}" \
  --display_data=false \
  --play_sounds="${PLAY_SOUNDS}"
