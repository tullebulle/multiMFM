#!/usr/bin/env bash
#SBATCH --job-name=qm9_lowbudget_grid
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Find a QM9 budget regime where multiMFM beats every baseline.
#
# Cost model (forward-equivalent, k=2):
#     128  outer trajectory steps       <- identical for every method compared
#   + 3 x G x S x MC                    <- the value estimator, ours alone
# with G guided outer steps, S = --value-glass-steps, MC = --value-samples.
#
# Every published multiMFM / steer-search number uses S=4, MC=32, G=115, i.e.
# 1,508 of its 1,544 forward-equivalents are the value estimator. S has never
# been varied, yet it is the flow-map lever: the DNA operating point that wins
# its comparison uses ONE inner step, not four.
#
# gamma is swept inside every configuration rather than carried over, because a
# strength tuned at S=4, MC=32 is tuned to a different estimator: changing S or
# MC changes the gradient's noise and scale, and on DNA transporting a strength
# across such a change was repeatedly what made a guided arm look dead.
#
# One steer_search run reports BOTH arms, so each costs one run:
#   final_endpoint          = plain multiMFM at this configuration
#   best_lookahead_pbvalid  = multiMFM-SS
#
#   sbatch scripts/qm9/low_budget_grid.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP="${PROP:-alpha}"
N="${N:-256}"
export SEED="${SEED:-1}" TARGET_SEED="${TARGET_SEED:-2027}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}" VALUE_BATCH_SIZE=0
export TOPK=0 LOOKAHEAD_SOURCE=mfm

# (value_glass_steps, guide_every, value_samples) -> target forward-equivalent budget
CFGS=( "1 4 1" "1 2 1" "1 1 1" "2 2 1" "1 1 2" "2 1 1" )
read -r S GE MC <<< "${CFGS[${SLURM_ARRAY_TASK_ID:-0}]}"

for GAMMA in 1.5 3.4 6.0 11.0; do
  tag="${PROP}_s${S}_ge${GE}_mc${MC}_g${GAMMA}"
  out="outputs/qm9/low_budget_grid/${tag}_n${N}"
  if [ -f "${out}/summary.json" ]; then echo "skip ${tag}"; continue; fi
  echo "===== ${tag}  (S=${S} guide_every=${GE} MC=${MC} gamma=${GAMMA})"
  OUTPUT_DIR="${out}" \
  EXTRA_ARGS="--value-glass-steps ${S} --guide-every ${GE} --value-samples ${MC}" \
    bash scripts/run_steer_search.sh "${PROP}" "${N}" "${GAMMA}" || echo "FAILED ${tag}" >&2
done
echo "done task ${SLURM_ARRAY_TASK_ID:-0}"
