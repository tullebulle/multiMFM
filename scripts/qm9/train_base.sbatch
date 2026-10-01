#!/usr/bin/env bash
#SBATCH --job-name=qm9_train_base
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Retrain the QM9 base flow-matching model (Table 19, QM9 column).
#
#   sbatch scripts/qm9/train_base.sbatch                 # from the repo root
#   TRAIN_STEPS=200 PB_EVERY=0 sbatch scripts/qm9/train_base.sbatch   # smoke test
#
# Replays the arguments stored in checkpoints/base_flow/model_step_100000.pt
# (field "args"; original run outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531
# in the purged tabasco repo). The trainer is multimfm.qm9_data (originally
# qm9_dfm_smoke.py), NOT multimfm.train_base_flow. The checkpoint used for all
# QM9 results is the step-100000 checkpoint written at the end.
# Training is not bit-reproducible (random init/order on GPU): a retrained model is
# a new sample from the same recipe, not the shipped weights.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
# shellcheck source=/dev/null
source scripts/env.sh

TRAIN_STEPS="${TRAIN_STEPS:-100000}"
PB_EVERY="${PB_EVERY:-5000}"
OUT="${OUT:-${REPO_ROOT}/outputs/qm9/base_flow/qm9_dfm_trainflow_long${TRAIN_STEPS}}"
PB_FLAGS=(--pb-every "${PB_EVERY}" --pb-samples 256 --pb-batch-size 64 --pb-sample-steps 64)
if [ "${PB_EVERY}" != "0" ]; then PB_FLAGS+=(--pb-eval-step-zero); fi

python -m multimfm.qm9_data \
  --partition train_flow --limit-data 50000 --max-len 9 \
  --batch-size 128 --train-steps "${TRAIN_STEPS}" \
  --sample-batch-size 8 --sample-steps 16 \
  --hidden-dim 64 --num-layers 4 --num-heads 4 \
  --lr 0.002 --seed 0 --device cuda \
  --log-every 100 --train-csv "${OUT}/train_metrics.csv" \
  --checkpoint-every 5000 --checkpoint-dir "${OUT}/checkpoints" --save-final-checkpoint \
  --check-regressor --regressor-kind guide --property alpha \
  --output-json "${OUT}/summary.json" \
  "${PB_FLAGS[@]}" \
  --pb-csv "${OUT}/pb_curve.csv"
