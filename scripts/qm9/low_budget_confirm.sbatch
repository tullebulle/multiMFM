#!/usr/bin/env bash
#SBATCH --job-name=qm9_lowbudget_confirm
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Held-out confirmation of the low-budget configurations.
#
# The configurations plotted in the steering figure were picked as the best of a
# grid on seed 1. The baselines they are compared against were tuned on a separate
# pilot, so selecting ours on the reported numbers would be the one asymmetry left
# in that comparison. This re-runs only the selected configurations on two unseen
# evaluation seeds; the figure quotes these, not the grid.
#
#   sbatch scripts/qm9/low_budget_confirm.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP="${PROP:-alpha}"
N="${N:-256}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}" VALUE_BATCH_SIZE=0
export TOPK=0 LOOKAHEAD_SOURCE=mfm

# the grid winners: (value_glass_steps, guide_every, value_samples, gamma)
SEL=( "1 4 1 3.4" "1 2 1 6.0" "2 2 1 3.4" "2 1 1 1.5" )
read -r S GE MC GAMMA <<< "${SEL[${SLURM_ARRAY_TASK_ID:-0}]}"

for SD in 2 3; do
  export SEED="${SD}" TARGET_SEED=$((2026 + SD))
  out="outputs/qm9/low_budget_confirm/${PROP}_s${S}_ge${GE}_mc${MC}_g${GAMMA}_seed${SD}_n${N}"
  [ -f "${out}/summary.json" ] && { echo "skip seed ${SD}"; continue; }
  echo "===== S=${S} ge=${GE} MC=${MC} gamma=${GAMMA} seed=${SD}"
  OUTPUT_DIR="${out}" \
  EXTRA_ARGS="--value-glass-steps ${S} --guide-every ${GE} --value-samples ${MC}" \
    bash scripts/run_steer_search.sh "${PROP}" "${N}" "${GAMMA}" || echo "FAILED seed ${SD}" >&2
done
echo done
