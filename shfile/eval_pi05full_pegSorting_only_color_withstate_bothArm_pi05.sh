export TOKENIZERS_PARALLELISM=false
# export HF_HOME=/mnt/tacumi_data/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LEROBOT_EVAL_LEFT_ARM_PRESENT=1
export LEROBOT_EVAL_RIGHT_ARM_PRESENT=1

REPO_ROOT="/home/dm/DM_das/lerobot-v0.5.1"
# PALIGEMMA_PATH="/home/dm/DM_das/checkpoints/paligemma/paligemma-3b-pt-224"
INTRINSICS_PATH_LEFT="${REPO_ROOT}/scripts/intrinsics_data/left_wirstFisheye_calibration.yaml"
INTRINSICS_PATH_RIGHT="${REPO_ROOT}/scripts/intrinsics_data/right_wirstFisheye_calibration.yaml"
POLICY_PATH="${REPO_ROOT}/outputs/pi05full_sparky_pegSorting_only_color_withstate_maskLeftArm_pi05_20260608T213500Z/030000/pretrained_model"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
EVAL_NAME="eval_pi05full_pegSorting_only_color_withstate_maskLeftArm_pi05_030000_${RUN_TS}"
DATASET_ROOT_BASE="/home/dm/DM_das/datasets"

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
    }
  }' \
  --display_data=false \
  --dataset.repo_id="/${EVAL_NAME}" \
  --dataset.root="${DATASET_ROOT_BASE}/${EVAL_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=600 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="pick up the peg and place it into the correct box" \
  --dataset.streaming_encoding=true \
  --dataset.rename_map='{"observation.images.cam_right_wrist":"observation.images.cam_right", "observation.images.cam_left_wrist":"observation.images.cam_left"}' \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --interpolation_multiplier=2
