#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_v0.5.1

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/mnt/tacumi_data/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"


torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${WORLD_SIZE} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
    src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=dag911262r00025a_lerobot_0429_193253_6_deformation_cam_left_only \
    --dataset.root=/mnt/tacumi_data/lerobot_datasetV30_train_ready/test/dag911262r00025a_lerobot_0429_193253_6_deformation_cam_left_only \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_umi_0512 \
    --job_name=pi05_umi_0512 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.repo_id=pi05_umi_0512 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.tokenizer_name=/mnt/tacumi_data/huggingface/paligemma-3b-pt-224 \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --policy.action_gap_frames=6 \
    --policy.canonicalize_bimanual_rot6d=true \
    --policy.canonical_missing_value=9930.0 \
    --policy.push_to_hub=false \
    --steps=10 \
    --save_freq=10000 \
    --batch_size=8 \
    --eval_freq=0



    # --dataset.repo_id=train_0512 \
    # --dataset.root=/mnt/data/lerobot_datasetV30_train/extract_dataset/train_0512 \
    # --dataset.video_backend=pyav \
    # --policy.type=pi05 \
    # --output_dir=./outputs/pi05_umi_0512 \
    # --job_name=pi05_umi_0512 \
    # --policy.pretrained_path=lerobot/pi05_base \
    # --policy.repo_id=pi05_umi_0512 \
    # --policy.compile_model=false \
    # --policy.gradient_checkpointing=true \
    # --policy.dtype=bfloat16 \
    # --policy.device=cuda \
    # --policy.tokenizer_name=/mnt/tacumi_data/huggingface/paligemma-3b-pt-224 \
    # --policy.use_relative_actions=true \
    # --policy.relative_exclude_joints='["gripper"]' \
    # --policy.canonicalize_bimanual_rot6d=true \
    # --policy.canonical_missing_value=9930.0 \
    # --policy.push_to_hub=false \
    # --steps=30000 \
    # --save_freq=10000 \
    # --batch_size=32 \
    # --eval_freq=0
