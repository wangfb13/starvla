export TOKENIZERS_PARALLELISM=false
# export HF_HOME=/mnt/tacumi_data/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export LEROBOT_EVAL_LEFT_ARM_PRESENT=0
export LEROBOT_EVAL_RIGHT_ARM_PRESENT=1

REPO_ROOT="/mnt/cpfs/projects/lerobot-v0.5.1-feat-sparky_x5"
INTRINSICS_PATH="${REPO_ROOT}/src/lerobot/camera_intrinsic/camera_calibration_right_wrist.json"
POLICY_PATH="${REPO_ROOT}/outputs/pi05full_sparky_pegSorting_only_color_nostate_maskLeftArm_pi05_20260608T213500Z/20260609_155604/checkpoints/030000/pretrained_model"

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
  --dataset.repo_id=/eval_pi05full_pegSorting_only_color_nostate_maskLeftArm_pi05_030000 \
  --dataset.root=/home/cxzk/DM_das/datasets/eval_pi05full_pegSorting_only_color_nostate_maskLeftArm_pi05_030000 \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=600 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="sort the peg into the correct hole" \
  --dataset.streaming_encoding=false \
  --dataset.rename_map='{"observation.images.cam_right_wrist":"observation.images.cam_right"}' \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --interpolation_multiplier=2
