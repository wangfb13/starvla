#!/usr/bin/env bash
#SBATCH --job-name=qwenpi_260902_s32drop_abs_g4
#SBATCH --partition=NV_RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=30-00:00:00
#SBATCH --output=/public/home/eng_ywang/wangfubo/model_train/starvla/logs/%x_%j.out

set -euo pipefail

REPO=/public/home/eng_ywang/wangfubo/model_train/starvla
ENV_DIR=/public/home/eng_ywang/miniconda3/envs/wfb_starvla_env
DATASET_ROOT=/public/home/eng_ywang/wangfubo/umi_data/260902_umi_stack_cups3_sparky_trainready_224
SIDECAR_PATH=/public/home/eng_ywang/wangfubo/umi_data/starvla_groot_260902_interarm32_sidecar/260902_umi_stack_cups3_sparky_trainready_224/framewise_guidance.parquet
INFO_PATH=${REPO}/shfile/info_umi_260812_stack_cups_undist_wrist.json
BASE_VLM=/public/home/eng_ywang/wangfubo/pretrained_models/Qwen3.5-0.8B

GPU_COUNT=4
MICRO_BATCH_SIZE=1
GRAD_ACCUM_STEPS=6
MAX_STEPS=50000
SAVE_FREQ=10000
JOB_NAME=starvla_qwenpi_260902_state32drop5p5_absgrip_h2_c32_a16_g6_s50000
RUN_ID=${SLURM_JOB_ID}
OUTPUT_DIR=${REPO}/outputs/${JOB_NAME}/${RUN_ID}

export PATH=${ENV_DIR}/bin:${PATH}
export PYTHONPATH=${REPO}/src:${REPO}:${PYTHONPATH:-}
export LD_LIBRARY_PATH=${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=${HF_HOME:-/public/home/eng_ywang/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets/${JOB_NAME}/${RUN_ID}}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd ${REPO}
mkdir -p $(dirname ${OUTPUT_DIR}) ${HF_HOME} ${HF_DATASETS_CACHE}

echo "[info] node=$(hostname) job=${SLURM_JOB_ID} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
echo "[info] StarVLA QwenPI: 32D state history=2, full/inter-arm dropout=0.05/0.05"
echo "[info] action=20D current-relative SE3 + absolute grippers; gap=6 chunk=32 execute=16 skip=0"
echo "[info] micro_batch=1 gpu=4 grad_accum=6 effective_batch=24"
echo "[info] checkpoint every 10000 steps; retain latest only"
echo "[info] output=${OUTPUT_DIR}"

FRAMEWORK_CONFIG='{"framework":{"action_model":{"action_model_type":"LayerwiseFM","repeated_diffusion_steps":1,"num_inference_timesteps":4,"add_pos_embed":true,"max_seq_len":1024,"num_target_vision_tokens":32,"noise_beta_alpha":1.5,"noise_beta_beta":1.0,"noise_s":0.999,"num_timestep_buckets":1000,"diffusion_model_cfg":{"dropout":0.2,"final_dropout":true,"interleave_self_attention":true,"norm_type":"ada_norm","positional_embeddings":null,"attention_head_dim":64}}},"datasets":{"vla_data":{"include_state":true,"obs_image_size":[224,224]}}}'
ACTION_NAMES='["left_x","left_y","left_z","left_rot6d_0","left_rot6d_1","left_rot6d_2","left_rot6d_3","left_rot6d_4","left_rot6d_5","gripper_left","right_x","right_y","right_z","right_rot6d_0","right_rot6d_1","right_rot6d_2","right_rot6d_3","right_rot6d_4","right_rot6d_5","gripper_right"]'
STATE_NAMES='["left_x","left_y","left_z","left_rot6d_0","left_rot6d_1","left_rot6d_2","left_rot6d_3","left_rot6d_4","left_rot6d_5","gripper_left","right_x","right_y","right_z","right_rot6d_0","right_rot6d_1","right_rot6d_2","right_rot6d_3","right_rot6d_4","right_rot6d_5","gripper_right","left_valid","right_valid","left_wrt_right_x","left_wrt_right_y","left_wrt_right_z","left_wrt_right_rot6d_0","left_wrt_right_rot6d_1","left_wrt_right_rot6d_2","left_wrt_right_rot6d_3","left_wrt_right_rot6d_4","left_wrt_right_rot6d_5","state_available"]'

exec ${ENV_DIR}/bin/torchrun --standalone --nnodes=1 --nproc_per_node=${GPU_COUNT} \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.root=${DATASET_ROOT} \
  --dataset.sidecar_path=${SIDECAR_PATH} \
  --dataset.info_path=${INFO_PATH} \
  --dataset.video_backend=pyav \
  --policy.type=starvla \
  --policy.repo_id=${JOB_NAME} \
  --policy.push_to_hub=false \
  --policy.framework_name=QwenPI \
  --policy.framework_config="${FRAMEWORK_CONFIG}" \
  --policy.base_vlm=${BASE_VLM} \
  --policy.attn_implementation=sdpa \
  --policy.freeze_vlm=false \
  --policy.gradient_checkpointing=true \
  --policy.action_dim=20 \
  --policy.state_dim=32 \
  --policy.use_state_input=true \
  --policy.n_obs_steps=2 \
  --policy.state_history_horizon=2 \
  --policy.relative_state_history_to_current=true \
  --policy.relative_scalar_state_history_indices='[9,19]' \
  --policy.inter_arm_relative_state=true \
  --policy.inter_arm_state_start=22 \
  --policy.state_full_dropout_prob=0.05 \
  --policy.inter_arm_state_dropout_prob=0.05 \
  --policy.state_valid_indices='[20,21]' \
  --policy.state_available_index=31 \
  --policy.drop_n_last_frames=31 \
  --policy.chunk_size=32 \
  --policy.n_action_steps=16 \
  --policy.action_gap_frames=6 \
  --policy.action_execution_skip_start_steps=0 \
  --policy.apply_action_gap_to_real_datasets=false \
  --policy.use_relative_actions=true \
  --policy.relative_action_keyframes='["0:3","10:13"]' \
  --policy.relative_exclude_keys='[]' \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.action_feature_names="${ACTION_NAMES}" \
  --policy.state_feature_names="${STATE_NAMES}" \
  --policy.image_keys='["observation.images.cam_left_undist","observation.images.cam_right_undist"]' \
  --policy.task_override='Stack the red cup on top of another cup' \
  --policy.optimizer_lr=1e-4 \
  --policy.vlm_optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=50000 \
  --policy.scheduler_decay_lr=1e-5 \
  --steps=${MAX_STEPS} \
  --batch_size=${MICRO_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
  --num_workers=6 \
  --log_freq=10 \
  --save_freq=${SAVE_FREQ} \
  --save_total_limit=1 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME}
