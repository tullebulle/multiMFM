#!/bin/bash
#SBATCH --job-name=glass_grad_mc
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
REWARD="${REWARD:-gc}"
OUTPUT_DIR="${OUTPUT_DIR:-workdir/rebuttal_gradient_accuracy_${REWARD}}"

"${PYTHON}" -u scripts/ablate_glass_gradient_mc.py \
  --output_dir "${OUTPUT_DIR}" \
  --reward "${REWARD}" \
  --lengths 50 100 200 400 \
  --t_eval 0.50 \
  --nfe_probe 32 \
  --nfe_calibration 64 \
  --calibration_samples 1024 \
  --sample_batch_size 32 \
  --nfe_value 8 \
  --n_probes 32 \
  --n_repeats 8 \
  --reference_mc 2048 \
  --mc_values 1 2 4 8 16 32 64 128 256 \
  --mc_chunk 16 \
  --z_target 1.0 \
  --reward_beta 1.0 \
  --reward_scale 1.0 \
  --thresholds 0.05 0.10 0.20 \
  --bootstrap 2000
