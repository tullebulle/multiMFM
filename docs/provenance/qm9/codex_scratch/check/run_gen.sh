#!/usr/bin/env bash
# Determinism check: run the real guided-generation driver with a FIXED config.
# Usage: run_gen.sh <OUTPUT_DIR>
# Everything (seed, hyperparameters, checkpoints) is identical between runs;
# only the output dir differs. Run once per GPU and diff the outputs.
set -euo pipefail

OUTPUT_DIR="$1"

REPO=/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco
PYTHON="${REPO}/.envs/tabasco-gpu/bin/python"
cd "${REPO}"
mkdir -p cache/huggingface cache/matplotlib cache/torch cache/tmp cache/xdg "${OUTPUT_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${REPO}/cache/huggingface"
export HF_DATASETS_CACHE="${REPO}/cache/huggingface/datasets"
export HF_HUB_CACHE="${REPO}/cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${REPO}/cache/matplotlib"
export XDG_CACHE_HOME="${REPO}/cache/xdg"
export TORCH_HOME="${REPO}/cache/torch"
export TMPDIR="${REPO}/cache/tmp"

NVIDIA_SITE="${REPO}/.envs/tabasco-gpu/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_SITE}/cublas/lib:${NVIDIA_SITE}/cuda_cupti/lib:${NVIDIA_SITE}/cuda_nvrtc/lib:${NVIDIA_SITE}/cuda_runtime/lib:${NVIDIA_SITE}/cudnn/lib:${NVIDIA_SITE}/cufft/lib:${NVIDIA_SITE}/curand/lib:${NVIDIA_SITE}/cusolver/lib:${NVIDIA_SITE}/cusparse/lib:${NVIDIA_SITE}/nccl/lib:${NVIDIA_SITE}/nvjitlink/lib:${NVIDIA_SITE}/nvtx/lib:${LD_LIBRARY_PATH:-}"

DFM_CKPT=outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt
MFM_CKPT=outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt

echo "host=$(hostname) gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) output_dir=${OUTPUT_DIR}"

"${PYTHON}" src/qm9_length_conditioned_guidance_experiment.py \
  --checkpoint "${DFM_CKPT}" \
  --device cuda \
  --seed 8 \
  --target-seed 2026 \
  --num-samples 32 \
  --sample-steps 128 \
  --mu 0.45 \
  --reward-scale 0.3 \
  --value-samples 32 \
  --value-batch-size 0 \
  --value-glass-steps 4 \
  --value-sampler mfm \
  --mfm-checkpoint "${MFM_CKPT}" \
  --mfm-key student \
  --mfm-diagonal \
  --guide-every 1 \
  --guide-min-t 0.05 \
  --guide-max-t 0.95 \
  --eps 1e-6 \
  --soft-atom-temperature 0.25 \
  --property-name alpha \
  --partition train_flow \
  --hist-bins 100 \
  --guide-atomics \
  --output-dir "${OUTPUT_DIR}"

echo "DONE ${OUTPUT_DIR}"
