#!/bin/bash
#SBATCH --job-name=repro_from_ckpt
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Executes notebooks/reproduce_from_checkpoints.ipynb end to end on one GPU (all GPU sections
# on, small n). The executed copy goes to outputs/reproduce_from_checkpoints/gpu_<jobid>/; the
# notebook in notebooks/ is not modified. Submit from the repo root:
#   sbatch notebooks/_run_reproduce_gpu.sbatch
# `kempner` is A100-only (no Blackwell, which torch 2.5.1 does not support).
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
[ -f "${REPO_ROOT}/scripts/env.sh" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
cd "${REPO_ROOT}"
mkdir -p logs
# shellcheck source=/dev/null
source scripts/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) start=$(date)"
export REPRO_OUT="${REPO_ROOT}/outputs/reproduce_from_checkpoints/gpu_${SLURM_JOB_ID:-local}"
mkdir -p "${REPRO_OUT}"
# Execute a snapshot, so editing or re-executing the notebook in place meanwhile cannot race this job.
# The kernel then runs in ${REPRO_OUT}; notebooks/ goes on PYTHONPATH for _common / _repro.
cp notebooks/reproduce_from_checkpoints.ipynb "${REPRO_OUT}/reproduce_from_checkpoints_input.ipynb"
export PYTHONPATH="${REPO_ROOT}/notebooks${PYTHONPATH:+:${PYTHONPATH}}"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=multimfm \
  --ExecutePreprocessor.timeout=3600 \
  --output "${REPRO_OUT}/reproduce_from_checkpoints_executed.ipynb" "${REPRO_OUT}/reproduce_from_checkpoints_input.ipynb"
echo "done $(date): ${REPRO_OUT}/reproduce_from_checkpoints_executed.ipynb"
