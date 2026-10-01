#!/usr/bin/env bash
#SBATCH --job-name=qm9_fk_fixed
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Feynman-Kac with adaptive resampling, and the sanity check it has to pass.
#
# The unconditional version resampled K particles with replacement every 8 steps. That
# loses diversity even when the weights are uniform, so over ~16 events the population
# coalesces to one ancestor whatever lambda is. Measured on QM9: at lambda = 0, where FK
# must be identical to best-of-N with the same K and the same final selection, it scored
# 8.851 against best-of-N's 3.592 -- almost exactly the base model (8.96), i.e. best-of-1.
#
# With resampling gated on the effective sample size (the standard SMC fix), lambda = 0
# cannot resample at all and FK must reproduce best-of-N. Task 0 checks precisely that;
# the rest re-tune lambda, which the QM9 sweep never did -- lambda is an inverse
# temperature on a reward whose scale differs ~100x between the two domains, and only the
# DNA betas were ever chosen on a pilot.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP=alpha
N="${N:-512}"
W="${SLURM_CPUS_PER_TASK:-16}"
SEEDS=(1 2 3); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
TSEED=$((2026 + SEED))
OUT="outputs/qm9/fk_fixed/seed${SEED}"

for LAM in 0 0.01 0.03 0.1 0.3 1.0 3.0; do
  d="${OUT}/fk_k4_lam${LAM}"
  [ -f "${d}/fk_${PROP}_seed${SEED}.json" ] && { echo "skip lam=${LAM}"; continue; }
  echo "----- lambda=${LAM} seed=${SEED} (ESS-gated) -----"
  python -m multimfm.baselines --method fk --property-name "${PROP}" \
    --num-samples "${N}" --sample-steps 128 --seed "${SEED}" --target-seed "${TSEED}" \
    --n-particles 4 --fk-lambda "${LAM}" --fk-resample-every 8 --fk-ess-threshold 0.5 \
    --pb-workers "${W}" --output-dir "${d}" 2>&1 \
    | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
done
echo done
