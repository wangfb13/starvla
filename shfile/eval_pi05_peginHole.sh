export TOKENIZERS_PARALLELISM=false
# export HF_HOME=/mnt/tacumi_data/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
LEFT_GRIPPER_OFFSET="${LEFT_GRIPPER_OFFSET:-0.0}"
RIGHT_GRIPPER_OFFSET="${RIGHT_GRIPPER_OFFSET:-0.0}"

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=spark3_pose_gripper \
  --robot.cameras='{
    "cam_left_wrist": {
      "type": "wirstFisheyeX5",
      "x5_ip": "192.168.9.10",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "intrinsics_file": "/home/cxzk/DM_das/lerobot_spark_x5_armjoints/run_scripts/intrinsics_data/left_rgb_video_ros-camchain.yaml",
      "enable_undistort_crop": false,
      "undistort_crop_size": null
    },
    "cam_right_wrist": {
      "type": "wirstFisheyeX5",
      "x5_ip": "192.168.9.11",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "intrinsics_file": "/home/cxzk/DM_das/lerobot_spark_x5_armjoints/run_scripts/intrinsics_data/right_rgb_video_ros-camchain.yaml",
      "enable_undistort_crop": false,
      "undistort_crop_size": null
    }
  }' \
  --display_data=false \
  --dataset.repo_id=/eval_pi05_peg_in_hole_200_0520 \
  --dataset.root=/home/cxzk/DM_das/datasets/eval_pi05_peg_in_hole_200_0520 \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=600 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="pick up the peg and place it in the hole" \
  --dataset.streaming_encoding=false \
  --policy.path=./outputs/pi05_peg_in_hole_200/30000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=true \
  --gripper_offsets.left="${LEFT_GRIPPER_OFFSET}" \
  --gripper_offsets.right="${RIGHT_GRIPPER_OFFSET}" \
  --interpolation_multiplier=1

#   --policy.tokenizer_name=/home/lunar/Desktop/paligemma-3b-pt-224 \
