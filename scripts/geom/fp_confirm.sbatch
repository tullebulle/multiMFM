#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_confirm
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Held-out confirmation of the configuration that beat TFG-Flow's relative improvement.
#
# The sweep put `fingerprint_tanimoto, lambda 20, mu 2, mfm_steps 2, window [0.1,0.9]` at
# +26.02% relative (0.2117 -> 0.2668) with PoseBusters validity 0.984 against the unguided
# 0.875 -- past TFG-Flow's +22.35%. That was n = 64 on seed 0, and it was the best of
# fifteen configurations, so it is exactly the kind of number that shrinks on replication:
# the landscape at that sample size is not monotone in either knob, which is itself a
# warning. This re-runs it on three UNSEEN seeds at n = 256, and the paper quotes these.
#
# The published configuration runs alongside on the same seeds, so the comparison between
# them is paired rather than read across sweeps.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

SEEDS=(1 2 3); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
N="${N:-256}"
OUT="results/geom/runs/fp_confirm"

run() {  # reward lambda mu mfm_steps tag
  python -u -m multimfm.geom.mfm_steering \
    --num-mols "$N" --reward-kind "$1" --reward-lambda "$2" --mu "$3" \
    --mfm-steps "$4" --guide-min-t 0.1 --guide-max-t 0.9 --guide-every 2 \
    --value-samples 4 --value-batch-size 8 --sample-steps 100 \
    --target-selection filter --device cuda --pb \
    --output-dir "${OUT}" --tag "$5_seed${SEED}" --seed "${SEED}" || echo "FAILED $5 seed ${SEED}"
}

run fingerprint_tanimoto 20 2.0 2 "best_tan_lam20_mu2_s2"
run fingerprint          0.02 0.5 10 "published_l2_mu0.5_s10"
echo "done seed ${SEED}"
