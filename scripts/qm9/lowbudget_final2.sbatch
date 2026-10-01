#!/usr/bin/env bash
#SBATCH --job-name=qm9_lowbudget_final2
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Finish the comparison for PLAIN multiMFM, and extend both arms past the crossover.
#
# The first full-n pass ran the configurations that minimise oracle MAE outright.
# For steer-search those also keep PoseBusters validity at 0.96-1.00, because it
# returns a valid look-ahead; for plain multiMFM they do not (PB 0.61-0.70), so
# plain multiMFM was being dropped by the validity floor that every method in the
# figure has to clear. The grid does contain configurations where it clears the
# floor comfortably -- they use a smaller gamma -- and those are what this runs at
# full sample size:
#
#     budget 212  gamma 1.5   PB 0.77      budget 470  gamma 2.2   PB 0.80
#     budget 299  gamma 1.5   PB 0.79      budget 818  gamma 1.0   PB 0.89
#
# Tasks 4-5 add two points past the crossover (1,508 calls) so the figure shows
# where best-of-N catches up rather than stopping while we are still ahead.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP=alpha
N="${N:-512}"
W="${SLURM_CPUS_PER_TASK:-16}"
I="${SLURM_ARRAY_TASK_ID:-0}"

run() { # S ge MC gamma seed diag(0|1)
  local S="$1" GE="$2" MC="$3" G="$4" SD="$5" DIAG="$6"
  local tag="s${S}_ge${GE}_mc${MC}_g${G}_$([ "${DIAG}" = 1 ] && echo diag || echo twotime)"
  local out="outputs/qm9/lowbudget_final2/seed${SD}/${tag}"
  [ -f "${out}/summary.json" ] && { echo "skip ${tag} seed${SD}"; return; }
  local extra=(); [ "${DIAG}" = 1 ] && extra+=(--mfm-diagonal)
  echo "----- ${tag} seed${SD} -----"
  python -m multimfm.steer_search --property-name "${PROP}" --num-samples "${N}" \
    --checkpoint checkpoints/base_flow/model_step_100000.pt \
    --mfm-checkpoint checkpoints/mfm_student/student_step_1000.pt --mfm-key student \
    --device cuda --sample-steps 128 --seed "${SD}" --target-seed $((2026 + SD)) \
    --mu "${G}" --reward-scale 0.3 --select-min-t 0.3 \
    --lookahead-source mfm --lookahead-topk 0 \
    --guide-every "${GE}" --guide-min-t 0.05 --guide-max-t 0.95 \
    --value-samples "${MC}" --value-glass-steps "${S}" --value-batch-size 0 \
    --eps 1e-6 --soft-atom-temperature 0.25 --partition train_flow \
    --pb-candidates 0 --pb-workers "${W}" "${extra[@]}" \
    --output-dir "${out}" 2>&1 | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
}

SD=$(( (I % 3) + 1 ))
case $(( I / 3 )) in
  0) # validity-passing plain-multiMFM configurations, both query modes
     for D in 1 0; do
       run 1 4 1 1.5 "${SD}" "${D}"
       run 1 2 1 1.5 "${SD}" "${D}"
       run 2 2 1 2.2 "${SD}" "${D}"
       run 2 1 1 1.0 "${SD}" "${D}"
     done ;;
  1) # past the crossover
     for D in 1 0; do
       run 2 1 1 2.2 "${SD}" "${D}"
       run 4 1 1 1.5 "${SD}" "${D}"
     done ;;
esac
echo "done task ${I}"
