#!/usr/bin/env bash
#SBATCH --job-name=qm9_fk_lambda
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
# Is Feynman-Kac really this much weaker on QM9 than on DNA, or is its temperature
# mis-set there?
#
# FK resamples K particles by exp(lambda * delta r). lambda is an inverse temperature
# whose right value depends entirely on the scale of r, and the two domains differ by
# about two orders of magnitude: the DNA reward is -(f_cyc - y*)^2 with f_cyc = O(1),
# so r ~ -1; the QM9 reward is -(pred - target)^2 in Bohr^3, so r ~ -100. The DNA
# betas were chosen on a pilot at matched NFE. The QM9 lambdas were not -- the sweep
# fixed {0.003, 0.01, 0.03} a priori -- so FK may simply never have been tuned there.
#
# A decisive check comes for free: at lambda = 0 the resampling weights are uniform and
# FK must reduce to best-of-N with the same K and the same final selection (both draw K
# candidates with the same integrator and both call select_best). If it does not, the
# gap is a defect, not tuning.
#
# K = 4 throughout (572 forward-equivalents), three seeds, same n as the comparison.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP=alpha
N="${N:-512}"
W="${SLURM_CPUS_PER_TASK:-16}"
SEEDS=(1 2 3); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
TSEED=$((2026 + SEED))
OUT="outputs/qm9/fk_lambda/seed${SEED}"

for LAM in 0 0.0001 0.001 0.003 0.01 0.03 0.1 0.3; do
  d="${OUT}/fk_k4_lam${LAM}"
  [ -f "${d}/fk_${PROP}_seed${SEED}.json" ] && { echo "skip lam=${LAM}"; continue; }
  echo "----- lambda=${LAM} seed=${SEED} -----"
  python -m multimfm.baselines --method fk --property-name "${PROP}" \
    --num-samples "${N}" --sample-steps 128 --seed "${SEED}" --target-seed "${TSEED}" \
    --n-particles 4 --fk-lambda "${LAM}" --fk-resample-every 8 \
    --pb-workers "${W}" --output-dir "${d}" 2>&1 \
    | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
done
echo done
