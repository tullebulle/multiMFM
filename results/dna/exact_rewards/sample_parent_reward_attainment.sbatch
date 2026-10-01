#!/bin/bash
#SBATCH --job-name=parent_reward
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
LENGTH="${LENGTH:?Set LENGTH to one of 50, 100, 200, or 400}"
REWARD="${REWARD:?Set REWARD to motif or conjunction}"
OUT_ROOT="${OUT_ROOT:-workdir/rebuttal_reward_attainment_${REWARD}_20260731}"
SEED="${SEED:-20260731}"

"${PYTHON}" -u scripts/sample_parent_reward_attainment.py \
  --lengths "${LENGTH}" \
  --reward "${REWARD}" \
  --output_dir "${OUT_ROOT}" \
  --seed "${SEED}" \
  --n_samples 100 \
  --batch_size 8 \
  --nfe_traj 64 \
  --t_end 0.999 \
  --guide_t_start 0.50 \
  --guide_t_end 0.95 \
  --mc 8 \
  --mc_chunk 4 \
  --guidance_frac 8.0 \
  --grad_clip 10.0 \
  --coeff_cap 10.0 \
  --calibration_samples 1024 \
  --nfe_calibration 64 \
  --calibration_batch_size 32 \
  --target_percentile 0.90 \
  --reward_beta 1.0 \
  --reward_scale 1.0 \
  --motif TTTTTC \
  --motif2 AAAATT \
  --motif_tau 0.10 \
  --conjunction_tau 0.10 \
  --bootstrap 2000 \
  --device cuda:0
