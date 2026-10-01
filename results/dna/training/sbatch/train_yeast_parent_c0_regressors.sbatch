#!/bin/bash
#SBATCH --job-name=yeast_parent_c0
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
DATA_PT="${DATA_PT:-data/yeast_parent_disjoint/yeast_parent_L50.pt}"
SPLIT_PT="${SPLIT_PT:-data/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt}"
RUN_TAG="${RUN_TAG:-2026-07-26}"
GUIDE_OUT="${GUIDE_OUT:-workdir/yeast_parent_c0_guide_${RUN_TAG}}"
ORACLE_OUT="${ORACLE_OUT:-workdir/yeast_parent_c0_oracle_${RUN_TAG}}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

common_args=(
  --data_pt "${DATA_PT}"
  --split_pt "${SPLIT_PT}"
  --model_type park_cnn
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --batch_size "${BATCH_SIZE}"
  --lr 0.001
  --weight_decay 0.0001
  --reverse_complement_augmentation
  --reverse_complement_average
  --device cuda:0
)

"${PYTHON}" -u scripts/train_parent_c0_regressor.py \
  "${common_args[@]}" \
  --role guide \
  --train_split a_train \
  --validation_split a_val \
  --test_split test \
  --seed 0 \
  --out_dir "${GUIDE_OUT}"

"${PYTHON}" -u scripts/train_parent_c0_regressor.py \
  "${common_args[@]}" \
  --role oracle \
  --train_split b_train \
  --validation_split b_val \
  --test_split test \
  --seed 1 \
  --out_dir "${ORACLE_OUT}"
