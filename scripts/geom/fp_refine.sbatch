#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_refine
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Refine around the configuration that beat TFG-Flow, on the window that actually won.
#
# The winner is lambda 20, mu 2, mfm_steps 2, guide_every 2, value_samples 4 over the FULL
# window [0.1, 0.9] -- +26.02% relative at PB 0.984. The earlier sweep moved several knobs
# at once and mostly varied the early window, which lost; these are the neighbours of the
# winner that were never tried at the window that won.
#
# Four directions, one per task:
#   mu     -- the single biggest factor so far (0.5 -> 2 took the old objective from +1.4%
#             to +16.6%), and 2 was simply the larger of the two values tried at this window
#   lambda -- 20 beat 5 and 50, so the optimum is bracketed but not located
#   MC     -- the posterior sample count has never been varied here at all, and it is the
#             knob that controls value-estimate quality
#   S, guide_every -- inner steps and how often guidance is applied
#
# Same seed and n as the sweep that found the winner, so the numbers are comparable; any
# improvement goes to the held-out confirmation before it is quoted.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

N="${N:-64}"
OUT="results/geom/runs/fp_refine"
I="${SLURM_ARRAY_TASK_ID:-0}"

run() {  # lambda mu mfm_steps guide_every value_samples tag
  python -u -m multimfm.geom.mfm_steering \
    --num-mols "$N" --reward-kind fingerprint_tanimoto --reward-lambda "$1" --mu "$2" \
    --mfm-steps "$3" --guide-min-t 0.1 --guide-max-t 0.9 --guide-every "$4" \
    --value-samples "$5" --value-batch-size 8 --sample-steps 100 \
    --target-selection filter --device cuda --pb \
    --output-dir "${OUT}" --tag "$6" --seed 0 || echo "FAILED $6"
}

case "${I}" in
  0) for MU in 1.0 1.5 3.0; do run 20 "$MU" 2 2 4 "mu${MU}"; done ;;
  1) for LAM in 10 30; do run "$LAM" 2.0 2 2 4 "lam${LAM}"; done
     run 20 2.0 2 1 4 "ge1" ;;
  2) for MC in 8 16; do run 20 2.0 2 2 "$MC" "mc${MC}"; done ;;
  3) for S in 1 4; do run 20 2.0 "$S" 2 4 "s${S}"; done
     run 20 2.0 2 1 8 "ge1_mc8" ;;
esac
echo "done task ${I}"
