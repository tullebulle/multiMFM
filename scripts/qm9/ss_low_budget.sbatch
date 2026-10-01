#!/usr/bin/env bash
#SBATCH --job-name=qm9_ss_low_budget
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# The QM9 steering sweep never entered the budget regime where the DNA result wins.
#
# On DNA the dMFM leads every baseline between 287 and 577 forward-equivalent
# network calls, which is 3.0-6.0x the cost of one unguided trajectory (96 steps).
# On QM9 the cheapest multiMFM-SS configuration measured so far costs 1,544
# forward-equivalents, 12.1x the unguided cost (128 steps) -- already past the
# crossover with best-of-N. The low-budget regime is unmeasured on QM9, not lost.
#
# This fills it in by guiding less often (--guide-every) at MC=1, which is the
# knob DPS already uses in the same sweep, so the points land on the same axis
# and in the same results file.
#
#   sbatch scripts/qm9/ss_low_budget.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP="${PROP:-alpha}"
N="${N:-256}"
export SEED="${SEED:-1}" TARGET_SEED="${TARGET_SEED:-2027}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}" VALUE_BATCH_SIZE=0
export TOPK=0 LOOKAHEAD_SOURCE=mfm

for GE in 8 4 2; do
  out="outputs/qm9/ss_low_budget/${PROP}_mc1_ge${GE}_s${SEED}_n${N}"
  if [ -f "${out}/summary.json" ]; then echo "skip ge=${GE} (done)"; continue; fi
  echo "== ${PROP} MC=1 guide_every=${GE}"
  OUTPUT_DIR="${out}" EXTRA_ARGS="--value-samples 1 --guide-every ${GE}" \
    bash scripts/run_steer_search.sh "${PROP}" "${N}" || echo "FAILED ge=${GE}" >&2
done
echo done
