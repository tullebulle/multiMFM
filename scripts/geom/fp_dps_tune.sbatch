#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_dps_tune
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Tune DPS on its own, because the NFE sweep does not.
#
# lambda = 20 and gamma = 2 were selected for multiMFM, whose value gradient is a
# log-mean-exp over flow-map posterior samples. DPS differentiates a single deterministic
# endpoint, so its gradient has a different scale and variance; carrying multiMFM's strength
# across to it is exactly the transport error that made guided arms look dead elsewhere in
# this work, and here it would make the baseline look weak in our favour.
#
# Each baseline gets its own strength, selected on its own best score, and the comparison
# then uses DPS at its best rather than DPS at ours.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

SEEDS=(301 302 303); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
N="${N:-256}"
OUT="results/geom/runs/fp_dps_tune"
mkdir -p "${OUT}"

for MU in 0.5 1.0 4.0 8.0; do
  python -u -m multimfm.geom.dps_steering \
    --num-mols "$N" --reward-kind fingerprint_tanimoto --reward-lambda 20 --mu "$MU" \
    --guide-min-t 0.1 --guide-max-t 0.9 --guide-every 2 --sample-steps 100 \
    --target-selection filter --device cuda --pb \
    --output-dir "${OUT}" --tag "dps_mu${MU}_seed${SEED}" --seed "${SEED}" \
    || echo "FAILED dps mu${MU}"
done
echo "done seed ${SEED}"
