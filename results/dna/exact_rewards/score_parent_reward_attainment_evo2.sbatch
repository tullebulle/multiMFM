#!/bin/bash
#SBATCH --job-name=parent_reward_evo2
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01

PYTHON="${PYTHON:-.envs/evo2/bin/python}"
SAMPLE_DIR="${SAMPLE_DIR:?Set SAMPLE_DIR to the length-specific sampling directory}"
OUT_DIR="${OUT_DIR:-${SAMPLE_DIR}/evo2}"
SEED="${SEED:-20260731}"
HF_HOME="${HF_HOME:-${SLURM_SUBMIT_DIR}/.cache/huggingface}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Evo2 Python executable: ${PYTHON}" >&2
  exit 2
fi

export HF_HOME
export OMP_NUM_THREADS=4

"${PYTHON}" -u scripts/score_parent_reward_attainment_evo2.py \
  --sample_dir "${SAMPLE_DIR}" \
  --out_dir "${OUT_DIR}" \
  --model_name evo2_7b_base \
  --batch_size 1 \
  --bootstrap 2000 \
  --seed "${SEED}" \
  --device cuda:0
