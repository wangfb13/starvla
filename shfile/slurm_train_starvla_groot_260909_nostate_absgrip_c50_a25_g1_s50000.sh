#!/usr/bin/env bash
#SBATCH --job-name=groot_260909_ns_abs_c50a25_g1
#SBATCH --partition=NV_RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --output=/public/home/eng_ywang/wangfubo/model_train/starvla/logs/%x_%j.out

set -euo pipefail

REPO=/public/home/eng_ywang/wangfubo/model_train/starvla
ENV_DIR=/public/home/eng_ywang/miniconda3/envs/wfb_starvla_env
DATASET_ROOT=/public/home/eng_ywang/wangfubo/umi_data/260909_umi_stack_cups_bucai_pz5_trainready_224_small_tag
SIDECAR_ROOT=/public/home/eng_ywang/wangfubo/umi_data/starvla_groot_260909_nostate_sidecar
SIDECAR_DIR="${SIDECAR_ROOT}/$(basename "${DATASET_ROOT}")"
SIDECAR_PATH="${SIDECAR_DIR}/framewise_guidance.parquet"
INFO_PATH=${REPO}/shfile/info_umi_260812_stack_cups_undist_wrist.json
BASE_VLM=/public/home/eng_ywang/wangfubo/pretrained_models/Qwen3.5-0.8B

GPU_COUNT=1
MICRO_BATCH_SIZE=4
GRAD_ACCUM_STEPS=6
MAX_STEPS=50000
SAVE_FREQ=10000
JOB_NAME=starvla_groot_260909_nostate_absgrip_c50_a25_g6_s50000
RUN_ID="${SLURM_JOB_ID}"
OUTPUT_DIR=${REPO}/outputs/${JOB_NAME}/${RUN_ID}

required_paths=(
  "${ENV_DIR}/bin/python"
  "${ENV_DIR}/bin/torchrun"
  "${DATASET_ROOT}/meta/info.json"
  "${INFO_PATH}"
  "${BASE_VLM}/config.json"
  "${BASE_VLM}/model.safetensors-00001-of-00001.safetensors"
)
for path in "${required_paths[@]}"; do
  [[ -e "${path}" ]] || { echo "[error] missing required path: ${path}" >&2; exit 1; }
done

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/public/home/eng_ywang/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets/${JOB_NAME}/${RUN_ID}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "${REPO}"
mkdir -p "${SIDECAR_ROOT}" "$(dirname "${OUTPUT_DIR}")" "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "[info] node=$(hostname) job=${SLURM_JOB_ID} cuda=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# Build the 32D sidecar on the allocated compute node. Never run Python on login.
if [[ ! -s "${SIDECAR_PATH}" ]]; then
  exec 9>"${SIDECAR_ROOT}/.build.lock"
  flock 9
  if [[ ! -s "${SIDECAR_PATH}" ]]; then
    "${ENV_DIR}/bin/python" scripts/build_pi05_canonical_sidecar.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-type umi \
      --umi-pose-frame absolute \
      --umi-gripper-normalization data_min \
      --add-inter-arm-relative-state \
      --relative-stats-chunk-size 32 \
      --relative-stats-action-gap-frames 6 \
      --relative-action-keyframes 0:3,10:13 \
      --output-root "${SIDECAR_ROOT}"
  fi
  flock -u 9
fi

"${ENV_DIR}/bin/python" -m py_compile \
  scripts/build_pi05_canonical_sidecar.py \
  src/lerobot/processor/relative_action_processor.py \
  src/lerobot/datasets/sidecar_utils.py \
  src/lerobot/datasets/factory.py \
  src/lerobot/policies/starvla_groot/configuration_starvla_groot.py \
  src/lerobot/policies/starvla_groot/processor_starvla_groot.py \
  src/lerobot/policies/starvla_groot/modeling_starvla_groot.py

echo "[info] no-state policy: state is retained only inside the processor for current-relative TCP decoding"
echo "[info] history=1 chunk=50 execute=25 execution_skip=0 gap=6 stride=1 steps=${MAX_STEPS}"
echo "[info] gripper: absolute action targets; data-min normalization left=0.3719903528690338 right=0.3978762626647949"
echo "[info] task override: Stack the red cup on top of another cup"
echo "[info] micro_batch=${MICRO_BATCH_SIZE} gpu=${GPU_COUNT} grad_accum=${GRAD_ACCUM_STEPS} effective_batch=24"
echo "[info] estimated_epochs=(50000*24)/105825 ~= 11.34"
echo "[info] save_freq=${SAVE_FREQ}; rolling retention keeps only the latest checkpoint"
echo "[info] output=${OUTPUT_DIR}"

exec "${ENV_DIR}/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.sidecar_path="${SIDECAR_PATH}" \
  --dataset.info_path="${INFO_PATH}" \
  --dataset.video_backend=pyav \
  --policy.type=starvla_groot \
  --policy.repo_id="${JOB_NAME}" \
  --policy.base_vlm="${BASE_VLM}" \
  --policy.attn_implementation=sdpa \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.n_obs_steps=1 \
  --policy.state_history_horizon=1 \
  --policy.relative_state_history_to_current=false \
  --policy.relative_scalar_state_history_indices='[]' \
  --policy.inter_arm_relative_state=false \
  --policy.inter_arm_state_start=22 \
  --policy.state_full_dropout_prob=0.0 \
  --policy.inter_arm_state_dropout_prob=0.0 \
  --policy.state_input_mask_indices='[]' \
  --policy.state_valid_indices='[20,21]' \
  --policy.state_available_index=31 \
  --policy.drop_n_last_frames=49 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=25 \
  --policy.action_execution_skip_start_steps=0 \
  --policy.action_gap_frames=6 \
  --policy.apply_action_gap_to_real_datasets=false \
  --policy.action_model_type=DiT-B \
  --policy.repeated_diffusion_steps=8 \
  --policy.num_inference_timesteps=4 \
  --policy.state_mode=raw \
  --policy.use_state_input=false \
  --policy.task_override='Stack the red cup on top of another cup' \
  --policy.use_relative_actions=true \
  --policy.relative_action_keyframes='["0:3","10:13"]' \
  --policy.relative_exclude_keys='[]' \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.state_dim=32 \
  --policy.action_dim=20 \
  --policy.canonicalize_bimanual_rot6d=false \
  --policy.wrist_only=true \
  --policy.top_camera_keys='[]' \
  --policy.wrist_camera_keys='["observation.images.cam_left_undist","observation.images.cam_right_undist"]' \
  --policy.image_resolution='[224,224]' \
  --policy.tactile_mode=none \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=50000 \
  --policy.scheduler_decay_lr=1e-6 \
  --policy.push_to_hub=false \
  --steps="${MAX_STEPS}" \
  --batch_size="${MICRO_BATCH_SIZE}" \
  --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  --num_workers=8 \
  --log_freq=10 \
  --save_freq="${SAVE_FREQ}" \
  --save_total_limit=1 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}"
