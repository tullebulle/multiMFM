#!/bin/bash
#SBATCH --job-name=parent_postdiv
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

# Reward-free conditional posterior diversity across L=50,100,200,400.
# Checkpoints are frozen inside the evaluator. dMFM composes learned two-time
# flow maps (so DMFM_STEPS=1 is the direct F(0,1, eps | t, x_t) prediction),
# while GLASS uses 100 Euler steps as the numerical posterior reference.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
OUT_DIR="${OUT_DIR:-workdir/parent_posterior_diversity_glass32_20260726}"
DMFM_STEPS="${DMFM_STEPS:-1}"
DMFM_SAMPLER="${DMFM_SAMPLER:-flow_map}"
GLASS_STEPS="${GLASS_STEPS:-100}"

"${PYTHON}" -u scripts/eval_parent_posterior_diversity.py \
  --lengths 50 100 200 400 \
  --times 0.0 0.25 0.5 0.75 0.9 0.99 1.0 \
  --n_sources 32 \
  --n_futures 16 \
  --source_batch 4 \
  --dmfm_steps "${DMFM_STEPS}" \
  --dmfm_sampler "${DMFM_SAMPLER}" \
  --glass_steps "${GLASS_STEPS}" \
  --bootstrap 2000 \
  --seed 20260730 \
  --out_dir "${OUT_DIR}"
