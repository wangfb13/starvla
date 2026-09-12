#!/usr/bin/env bash
#SBATCH --job-name=qwenpi_cups3ds_ns_abs_c50_g4
#SBATCH --partition=NV_RTX4090
#SBATCH --exclude=gpur03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=5-00:00:00
#SBATCH --output=/public/home/eng_ywang/wangfubo/model_train/starvla/logs/%x_%j.out

set -euo pipefail

REPO=/public/home/eng_ywang/wangfubo/model_train/starvla
ENV_DIR=/public/home/eng_ywang/miniconda3/envs/wfb_starvla_env
ROOTS_FILE=${REPO}/shfile/roots_stack_cups_multids3.txt
SIDECAR_ROOT=/public/home/eng_ywang/wangfubo/umi_data/starvla_stack_cups_multids3_nostate_sidecar
INFO_PATH=${REPO}/shfile/info_umi_260812_stack_cups_undist_wrist.json
BASE_VLM=/public/home/eng_ywang/wangfubo/pretrained_models/Qwen3.5-0.8B

GPU_COUNT=4
MICRO_BATCH_SIZE=1
GRAD_ACCUM_STEPS=6
MAX_STEPS=100000
SAVE_FREQ=10000
JOB_NAME=starvla_qwenpi_stack_cups_multids3_nostate_absgrip_c50_a25_g6_s100000
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
mkdir -p ${SIDECAR_ROOT} $(dirname ${OUTPUT_DIR}) ${HF_HOME} ${HF_DATASETS_CACHE}

echo "[info] node=$(hostname) job=${SLURM_JOB_ID} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# Build one 32D sidecar per dataset on the allocated compute node.  This uses
# the same shared lock as the GR00T job, so concurrent startup is safe.
exec 9>"${SIDECAR_ROOT}/.build.lock"
flock 9
while IFS= read -r DATASET_ROOT || [[ -n "${DATASET_ROOT}" ]]; do
  [[ -n "${DATASET_ROOT}" ]] || continue
  SIDECAR_PATH="${SIDECAR_ROOT}/$(basename "${DATASET_ROOT}")/framewise_guidance.parquet"
  if [[ ! -s "${SIDECAR_PATH}" ]]; then
    "${ENV_DIR}/bin/python" scripts/build_pi05_canonical_sidecar.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-type umi \
      --umi-pose-frame absolute \
      --umi-gripper-normalization data_min \
      --add-inter-arm-relative-state \
      --relative-stats-chunk-size 50 \
      --relative-stats-action-gap-frames 6 \
      --relative-action-keyframes 0:3,10:13 \
      --output-root "${SIDECAR_ROOT}"
  fi
done < "${ROOTS_FILE}"
flock -u 9

echo "[info] StarVLA QwenPI: no model-state input; state is preprocessing-only"
echo "[info] action=20D current-relative SE3 + absolute grippers; gap=6 chunk=50 execute=25 skip=0"
echo "[info] micro_batch=1 gpu=4 grad_accum=6 effective_batch=24"
echo "[info] datasets=3 episodes=297 frames=227791 effective_samples~=213238"
echo "[info] estimated_epochs=(100000*24)/213238 ~= 11.25"
echo "[info] checkpoint every 10000 steps; retain latest only"
echo "[info] output=${OUTPUT_DIR}"

FRAMEWORK_CONFIG='{"framework":{"action_model":{"action_model_type":"LayerwiseFM","repeated_diffusion_steps":1,"num_inference_timesteps":4,"add_pos_embed":true,"max_seq_len":1024,"num_target_vision_tokens":32,"noise_beta_alpha":1.5,"noise_beta_beta":1.0,"noise_s":0.999,"num_timestep_buckets":1000,"diffusion_model_cfg":{"dropout":0.2,"final_dropout":true,"interleave_self_attention":true,"norm_type":"ada_norm","positional_embeddings":null,"attention_head_dim":64}}},"datasets":{"vla_data":{"include_state":false,"obs_image_size":[224,224]}}}'
ACTION_NAMES='["left_x","left_y","left_z","left_rot6d_0","left_rot6d_1","left_rot6d_2","left_rot6d_3","left_rot6d_4","left_rot6d_5","gripper_left","right_x","right_y","right_z","right_rot6d_0","right_rot6d_1","right_rot6d_2","right_rot6d_3","right_rot6d_4","right_rot6d_5","gripper_right"]'

exec ${ENV_DIR}/bin/torchrun --standalone --nnodes=1 --nproc_per_node=${GPU_COUNT} \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.roots_file=${ROOTS_FILE} \
  --dataset.sidecar_root=${SIDECAR_ROOT} \
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
  --policy.use_state_input=false \
  --policy.n_obs_steps=1 \
  --policy.state_history_horizon=1 \
  --policy.relative_state_history_to_current=false \
  --policy.relative_scalar_state_history_indices='[]' \
  --policy.inter_arm_relative_state=false \
  --policy.state_full_dropout_prob=0.0 \
  --policy.inter_arm_state_dropout_prob=0.0 \
  --policy.drop_n_last_frames=49 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=25 \
  --policy.action_gap_frames=6 \
  --policy.action_execution_skip_start_steps=0 \
  --policy.apply_action_gap_to_real_datasets=false \
  --policy.use_relative_actions=true \
  --policy.relative_action_keyframes='["0:3","10:13"]' \
  --policy.relative_exclude_keys='[]' \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.action_feature_names="${ACTION_NAMES}" \
  --policy.image_keys='["observation.images.cam_left_undist","observation.images.cam_right_undist"]' \
  --policy.task_override='Stack the red cup on top of another cup' \
  --policy.optimizer_lr=1e-4 \
  --policy.vlm_optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=1e-5 \
  --steps=${MAX_STEPS} \
  --batch_size=${MICRO_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
  --num_workers=8 \
  --log_freq=10 \
  --save_freq=${SAVE_FREQ} \
  --save_total_limit=1 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME}
