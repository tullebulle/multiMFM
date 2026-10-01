#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_confirm2
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Confirm the refinement candidates properly, because n = 64 on one seed does not measure
# what it appears to.
#
# The first winner read +26.02% at n = 64 seed 0 and came back at **+8.38% +- 1.32** on
# three unseen seeds at n = 256. The refinement sweep then produced four configurations
# above +20% at that same unreliable setting, so they are treated as directions to test, not
# as results. Expect them to shrink too.
#
# The two worth testing are principled rather than incidental: guiding at every step instead
# of every second (`ge1`), and four times the posterior samples (`mc16`), which is the knob
# that controls value-estimate quality and is the one our method is supposed to exploit.
# Both cost more compute, so they are reported with their NFE rather than as free wins.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

SEEDS=(1 2 3); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
N="${N:-256}"
OUT="results/geom/runs/fp_confirm2"

run() {  # lambda mu mfm_steps guide_every value_samples tag
  python -u -m multimfm.geom.mfm_steering \
    --num-mols "$N" --reward-kind fingerprint_tanimoto --reward-lambda "$1" --mu "$2" \
    --mfm-steps "$3" --guide-min-t 0.1 --guide-max-t 0.9 --guide-every "$4" \
    --value-samples "$5" --value-batch-size 8 --sample-steps 100 \
    --target-selection filter --device cuda --pb \
    --output-dir "${OUT}" --tag "$6_seed${SEED}" --seed "${SEED}" || echo "FAILED $6 seed ${SEED}"
}

run 20 2.0 2 1 4  "ge1"
run 20 2.0 2 2 16 "mc16"
run 20 2.0 2 1 8  "ge1_mc8"
run 20 1.5 2 2 4  "mu1.5"
echo "done seed ${SEED}"
