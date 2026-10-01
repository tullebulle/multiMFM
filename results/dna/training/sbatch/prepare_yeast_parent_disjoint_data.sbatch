#!/bin/bash
#SBATCH --job-name=yeast_parent_data
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
"${PYTHON}" -u data/prepare_yeast_parent_splits.py \
  --input data/yeast_sequences.txt \
  --out_dir data/yeast_parent_disjoint \
  --seed 0 \
  --lengths 50 100 200 400
