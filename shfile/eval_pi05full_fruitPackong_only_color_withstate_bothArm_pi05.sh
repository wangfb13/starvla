export TOKENIZERS_PARALLELISM=false
# export HF_HOME=/mnt/tacumi_data/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LEROBOT_EVAL_LEFT_ARM_PRESENT=1
export LEROBOT_EVAL_RIGHT_ARM_PRESENT=1

REPO_ROOT="/home/dm/DM_das/lerobot-v0.5.1"
# PALIGEMMA_PATH="/home/dm/DM_das/checkpoints/paligemma/paligemma-3b-pt-224"
INTRINSICS_PATH_LEFT="/home/dm/DM_das/intrinsics_data/wrist_fishCamera_192_168_9_10.yaml"
INTRINSICS_PATH_RIGHT="/home/dm/DM_das/intrinsics_data/wrist_fishCamera_192_168_9_11.yaml"
POLICY_PATH="/home/dm/DM_das/outputs/pi05full_sparky_fruitPacking_only_color_R8_vf6contact_stats_0701/pretrained_model"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
EVAL_NAME="eval_R8_vf6contact_stats_0701_3w_${RUN_TS}"
DATASET_ROOT_BASE="/home/dm/DM_das/datasets"
LEFT_GRIPPER_OFFSET="${LEFT_GRIPPER_OFFSET:-0.0}"
RIGHT_GRIPPER_OFFSET="${RIGHT_GRIPPER_OFFSET:-0.0}"
ACTION_SKIP_START_STEPS="${ACTION_SKIP_START_STEPS:-0}"

if [ ! -f "${INTRINSICS_PATH_RIGHT}" ]; then
  echo "Missing intrinsics file: ${INTRINSICS_PATH_RIGHT}"
  exit 1
fi

if [ ! -f "${INTRINSICS_PATH_LEFT}" ]; then
  echo "Missing intrinsics file: ${INTRINSICS_PATH_LEFT}"
  exit 1
fi

if [ ! -d "${POLICY_PATH}" ]; then
  echo "Missing policy checkpoint dir: ${POLICY_PATH}"
  exit 1
fi

python "${REPO_ROOT}/src/lerobot/scripts/lerobot_record.py" \
  --robot.type=spark3_pose_gripper \
  --robot.cameras='{
    "cam_right_wrist": {
      "type": "wirstFisheyeX5",
      "x5_ip": "192.168.9.11",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "intrinsics_file": "'"${INTRINSICS_PATH_RIGHT}"'"
    },
    "cam_left_wrist": {
      "type": "wirstFisheyeX5",
      "x5_ip": "192.168.9.10",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "intrinsics_file": "'"${INTRINSICS_PATH_LEFT}"'"
    },
    "cam_right_gripper_tactile": {
      type: dmTactileX5,
      x5_ip: 192.168.9.11,
      left_serial_number: X26040505,
      right_serial_number: X26040305,
      fps: 30,
      width: 128,
      height: 96,
      inference_mode: true,
    },
    "cam_left_gripper_tactile": {
      type: dmTactileX5,
      x5_ip: 192.168.9.10,
      left_serial_number: X26040573,
      right_serial_number: X26040560,
      fps: 30,
      width: 128,
      height: 96,
      inference_mode: true,
    },

  }' \
  --display_data=false \
  --play_sounds=false \
  --dataset.repo_id="/${EVAL_NAME}" \
  --dataset.root="${DATASET_ROOT_BASE}/${EVAL_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=1200 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="fruit packing" \
  --dataset.streaming_encoding=true \
  --dataset.vcodec=libsvtav1 \
  --dataset.encoder_threads=2 \
  --dataset.rename_map='{"observation.images.cam_right_wrist":"observation.images.cam_right", "observation.images.cam_left_wrist":"observation.images.cam_left"}' \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.action_execution_skip_start_steps="${ACTION_SKIP_START_STEPS}" \
  --inference_trace.enabled=true \
  --inference_trace.store_io_summaries=true \
  --inference_trace.store_image_keys=true \
  --gripper_offsets.left="${LEFT_GRIPPER_OFFSET}" \
  --gripper_offsets.right="${RIGHT_GRIPPER_OFFSET}" \
  --interpolation_multiplier=1
