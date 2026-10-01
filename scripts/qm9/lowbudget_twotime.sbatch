#!/usr/bin/env bash
#SBATCH --job-name=qm9_lowbudget_twotime
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# One more lever for plain multiMFM, whose weak spot at low budget is validity
# (PB 0.61-0.70 where steer-search sits at 0.96-0.98, because steer-search can
# simply return a valid look-ahead instead).
#
# Every QM9 run so far queries the student with --mfm-diagonal, i.e. as a
# one-time map. The same checkpoint can be queried as a genuine two-time flow
# map. On the published configuration that was measured once as PB 0.91 against
# 0.86 at an MAE cost inside noise, and it has never been tried at low budget --
# which is exactly where plain multiMFM needs the validity.
#
# Runs both query modes at the three low-budget configurations, same seeds, so
# the only difference is the flag.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP=alpha
N="${N:-512}"
SEEDS=(1 2 3)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
TSEED=$((2026 + SEED))
W="${SLURM_CPUS_PER_TASK:-16}"
ROOT="outputs/qm9/lowbudget_twotime/seed${SEED}"

run() { # S guide_every MC gamma diag(0|1)
  local S="$1" GE="$2" MC="$3" G="$4" DIAG="$5"
  local tag="s${S}_ge${GE}_mc${MC}_g${G}_$([ "${DIAG}" = 1 ] && echo diag || echo twotime)"
  local out="${ROOT}/${tag}"
  [ -f "${out}/summary.json" ] && { echo "skip ${tag}"; return; }
  local extra=(); [ "${DIAG}" = 1 ] && extra+=(--mfm-diagonal)
  echo "----- ${tag} -----"
  python -m multimfm.steer_search --property-name "${PROP}" --num-samples "${N}" \
    --checkpoint checkpoints/base_flow/model_step_100000.pt \
    --mfm-checkpoint checkpoints/mfm_student/student_step_1000.pt --mfm-key student \
    --device cuda --sample-steps 128 --seed "${SEED}" --target-seed "${TSEED}" \
    --mu "${G}" --reward-scale 0.3 --select-min-t 0.3 \
    --lookahead-source mfm --lookahead-topk 0 \
    --guide-every "${GE}" --guide-min-t 0.05 --guide-max-t 0.95 \
    --value-samples "${MC}" --value-glass-steps "${S}" --value-batch-size 0 \
    --eps 1e-6 --soft-atom-temperature 0.25 --partition train_flow \
    --pb-candidates 0 --pb-workers "${W}" "${extra[@]}" \
    --output-dir "${out}" 2>&1 | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
}

for DIAG in 1 0; do
  run 1 4 1 3.4 "${DIAG}"
  run 2 2 1 3.4 "${DIAG}"
  run 2 1 1 1.5 "${DIAG}"
done
echo "done seed ${SEED}"
