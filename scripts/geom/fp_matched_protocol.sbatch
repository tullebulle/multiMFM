#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_matched
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
# Our model under TFG-Flow's protocol, because neither of the two obvious comparisons is fair.
#
# Relative improvement favours whoever starts lower: their unconditional run is 0.170 and
# ours 0.2347, so the same absolute gain is a larger percentage for them. Absolute score
# favours whoever has the easier protocol. Quoting either alone picks a winner by choosing a
# denominator.
#
# Two of the three protocol differences turn out not to matter. Their convention averages
# over all samples and scores invalid ones 0; our conversion rate is 1.0, so our
# valid-only and all-molecule means are identical (0.2543 both). And our generated molecules
# already average 25 heavy atoms, inside their cap. The third is real: their generator is
# capped at 30 heavy atoms and ours at 71, so our target pool contains molecules theirs
# cannot represent. This run caps ours at 30 to match.
#
# Under matched protocol the comparison is absolute, against their 0.208 -- and against the
# random-pair floor for that regime, which is 0.2099, i.e. ABOVE their reported guided score.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

SEEDS=(1 2 3); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
N="${N:-256}"
OUT="results/geom/runs/fp_matched30"

run() {  # reward lambda mu mfm_steps tag
  python -u -m multimfm.geom.mfm_steering \
    --num-mols "$N" --reward-kind "$1" --reward-lambda "$2" --mu "$3" \
    --mfm-steps "$4" --max-atoms 30 \
    --guide-min-t 0.1 --guide-max-t 0.9 --guide-every 2 \
    --value-samples 4 --value-batch-size 8 --sample-steps 100 \
    --target-selection filter --device cuda --pb \
    --output-dir "${OUT}" --tag "$5_seed${SEED}" --seed "${SEED}" || echo "FAILED $5 seed ${SEED}"
}

run fingerprint_tanimoto 20 2.0 2 "ours_max30"
run fingerprint          0.02 0.5 10 "published_max30"
echo "done seed ${SEED}"
