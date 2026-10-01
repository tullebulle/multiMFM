#!/bin/bash
#SBATCH --job-name=yeast_L50_dfm
#SBATCH --partition=seas_gpu
#SBATCH -A albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-workdir/yeast_parent_a_L50_dfm_dit_h192_b4}"
STEPS="${STEPS:-200000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
VAL_EVERY="${VAL_EVERY:-5000}"

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
"${PYTHON}" -u scripts/train_dna_dfm_dit.py \
  --run_name yeast_parent_a_L50_dfm_dit_h192_b4 \
  --output_dir "${OUTPUT_DIR}" \
  --yeast_data_pt data/yeast_parent_disjoint/yeast_parent_L50.pt \
  --yeast_split_pt data/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt \
  --yeast_split a_train \
  --yeast_val_split a_val \
  --max_train_seqs 0 \
  --steps "${STEPS}" \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --hidden_dim 192 \
  --transformer_heads 6 \
  --transformer_ff_mult 4 \
  --dit_blocks 4 \
  --dropout 0.0 \
  --gaussian_beta_schedule linear \
  --save_every "${SAVE_EVERY}" \
  --val_every "${VAL_EVERY}" \
  --val_batches 0 \
  --eval_samples 0 \
  --num_workers 4
