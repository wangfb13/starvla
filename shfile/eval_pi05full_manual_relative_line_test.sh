export TOKENIZERS_PARALLELISM=false
# export HF_HOME=/mnt/tacumi_data/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LEROBOT_EVAL_LEFT_ARM_PRESENT=0
export LEROBOT_EVAL_RIGHT_ARM_PRESENT=1
COMMAND_MODE="${COMMAND_MODE:-pose}"

REPO_ROOT="/home/dm/DM_das/lerobot-v0.5.1"
# PALIGEMMA_PATH="/home/dm/DM_das/checkpoints/paligemma/paligemma-3b-pt-224"
INTRINSICS_PATH="${REPO_ROOT}/scripts/intrinsics_data/right_wirstFisheye_calibration.yaml"
POLICY_PATH="${REPO_ROOT}/outputs/pi05full_sparky_pegSorting_only_color_nostate_maskLeftArm_pi05_20260608T213500Z/005000/pretrained_model"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
EVAL_NAME="eval_pi05full_pegSorting_only_color_nostate_maskLeftArm_pi05_005000_${RUN_TS}"
DATASET_ROOT_BASE="/home/dm/DM_das/datasets"
LEFT_GRIPPER_OFFSET="${LEFT_GRIPPER_OFFSET:-0.0}"
RIGHT_GRIPPER_OFFSET="${RIGHT_GRIPPER_OFFSET:-0.0}"

if [ ! -f "${INTRINSICS_PATH}" ]; then
  echo "Missing intrinsics file: ${INTRINSICS_PATH}"
  exit 1
fi

if [ ! -d "${POLICY_PATH}" ]; then
  echo "Missing policy checkpoint dir: ${POLICY_PATH}"
  exit 1
fi

python "${REPO_ROOT}/src/lerobot/scripts/lerobot_record.py" \
  --robot.type=spark3_pose_gripper \
  --robot.command_mode="${COMMAND_MODE}" \
  --robot.cameras='{
    "cam_right_wrist": {
      "type": "wirstFisheyeX5",
      "x5_ip": "192.168.9.11",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "intrinsics_file": "'"${INTRINSICS_PATH}"'"
    }
  }' \
  --display_data=false \
  --dataset.repo_id="/${EVAL_NAME}" \
  --dataset.root="${DATASET_ROOT_BASE}/${EVAL_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="manual relative line test for pi05_full Spark inference path" \
  --dataset.streaming_encoding=false \
  --dataset.rename_map='{"observation.images.cam_right_wrist":"observation.images.cam_right"}' \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --manual_action_test.enabled=true \
  --manual_action_test.arm=right \
  --manual_action_test.axis=x \
  --manual_action_test.step_size=0.01 \
  --manual_action_test.forward_steps=15 \
  --manual_action_test.hold_steps=0 \
  --manual_action_test.return_to_start=true \
  --manual_action_test.gripper_delta=0.0 \
  --gripper_offsets.left="${LEFT_GRIPPER_OFFSET}" \
  --gripper_offsets.right="${RIGHT_GRIPPER_OFFSET}" \
  --interpolation_multiplier=1
