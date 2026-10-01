#!/usr/bin/env bash
# Parameter-search run: 1000 guided molecules for one property at a given (mu, reward_scale).
# Fresh process each call => kernel cleared between runs.
# Usage: run_prop.sh <PROPERTY> <MU> <REWARD_SCALE> <TAG> [GUIDE_MAX_T]
set -euo pipefail
PROPERTY="$1"; MU="$2"; REWARD_SCALE="$3"; TAG="$4"; GUIDE_MAX_T="${5:-0.95}"

REPO=/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco
PYTHON="${REPO}/.envs/tabasco-gpu/bin/python"
OUTPUT_DIR="${REPO}/codex_scratch/check/search/${TAG}"
cd "${REPO}"
mkdir -p cache/huggingface cache/matplotlib cache/torch cache/tmp cache/xdg "${OUTPUT_DIR}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME="${REPO}/cache/huggingface"
export HF_DATASETS_CACHE="${REPO}/cache/huggingface/datasets"
export HF_HUB_CACHE="${REPO}/cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${REPO}/cache/matplotlib"
export XDG_CACHE_HOME="${REPO}/cache/xdg"
export TORCH_HOME="${REPO}/cache/torch"
export TMPDIR="${REPO}/cache/tmp"
NVIDIA_SITE="${REPO}/.envs/tabasco-gpu/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_SITE}/cublas/lib:${NVIDIA_SITE}/cuda_cupti/lib:${NVIDIA_SITE}/cuda_nvrtc/lib:${NVIDIA_SITE}/cuda_runtime/lib:${NVIDIA_SITE}/cudnn/lib:${NVIDIA_SITE}/cufft/lib:${NVIDIA_SITE}/curand/lib:${NVIDIA_SITE}/cusolver/lib:${NVIDIA_SITE}/cusparse/lib:${NVIDIA_SITE}/nccl/lib:${NVIDIA_SITE}/nvjitlink/lib:${NVIDIA_SITE}/nvtx/lib:${LD_LIBRARY_PATH:-}"

DFM_CKPT=outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt
MFM_CKPT=outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt

echo "START tag=${TAG} host=$(hostname) gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) prop=${PROPERTY} mu=${MU} rs=${REWARD_SCALE} gmax=${GUIDE_MAX_T}"

"${PYTHON}" src/qm9_length_conditioned_guidance_experiment.py \
  --checkpoint "${DFM_CKPT}" --device cuda \
  --seed 8 --target-seed 2026 --num-samples 1000 \
  --sample-steps 128 --mu "${MU}" --reward-scale "${REWARD_SCALE}" \
  --value-samples 32 --value-batch-size 0 --value-glass-steps 4 \
  --value-sampler mfm --mfm-checkpoint "${MFM_CKPT}" --mfm-key student --mfm-diagonal \
  --guide-every 1 --guide-min-t 0.05 --guide-max-t "${GUIDE_MAX_T}" \
  --eps 1e-6 --soft-atom-temperature 0.25 \
  --property-name "${PROPERTY}" --partition train_flow --hist-bins 100 \
  --guide-atomics --compute-pb \
  --output-dir "${OUTPUT_DIR}"

echo "DONE tag=${TAG}"
