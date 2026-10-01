#!/usr/bin/env bash
# PB-valid best-of-lookahead run: one property, given (mu, rs, select_min_t) [+ guide window].
# Usage: run_bol.sh <PROPERTY> <MU> <RS> <SELMINT> <TAG> [GUIDE_MIN_T] [GUIDE_MAX_T]
set -euo pipefail
PROPERTY="$1"; MU="$2"; RS="$3"; SELMINT="$4"; TAG="$5"; GMIN="${6:-0.05}"; GMAX="${7:-0.95}"; NUM="${8:-100}"

REPO=/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco
PYTHON="${REPO}/.envs/tabasco-gpu/bin/python"
OUTPUT_DIR="${REPO}/codex_scratch/memory/search/${TAG}"
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

echo "START tag=${TAG} host=$(hostname) gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) prop=${PROPERTY} mu=${MU} rs=${RS} selmint=${SELMINT} gwin=[${GMIN},${GMAX}]"

"${PYTHON}" codex_scratch/memory/best_of_lookahead.py \
  --checkpoint "${DFM_CKPT}" --mfm-checkpoint "${MFM_CKPT}" --mfm-key student --mfm-diagonal \
  --device cuda --seed 8 --target-seed 2026 --num-samples "${NUM}" \
  --sample-steps 128 --mu "${MU}" --reward-scale "${RS}" \
  --value-samples 32 --value-batch-size 0 --value-glass-steps 4 \
  --guide-every 1 --guide-min-t "${GMIN}" --guide-max-t "${GMAX}" \
  --eps 1e-6 --soft-atom-temperature 0.25 \
  --property-name "${PROPERTY}" --partition train_flow --hist-bins 100 \
  --select-min-t "${SELMINT}" --pb-candidates 0 --pb-workers 16 \
  --output-dir "${OUTPUT_DIR}"

echo "DONE tag=${TAG}"
