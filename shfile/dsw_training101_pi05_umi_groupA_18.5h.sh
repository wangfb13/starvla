#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/mnt/tacumi_data/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

nohup python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=/mnt/cpfs/datasets/pi05_groupA_train_roots_symlink \
    --dataset.info_path=meta/info_fisheye.json \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_umi_groupA_18.5h \
    --job_name=pi05_umi_groupA_18.5h \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.repo_id=pi05_umi_groupA_18.5h \
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
    --steps=30000 \
    --save_freq=10000 \
    --batch_size=8 \
    --eval_freq=0   > pi05_umi_groupA_18.5h_training101_0522.log 2>&1 &

    # --dataset.video_backend=pyav \
    # --dataset.repo_id=dag911262r00025a_lerobot_0429_193253_6_deformation_cam_left_only \
