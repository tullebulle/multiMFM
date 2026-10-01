#!/usr/bin/env bash
#SBATCH --job-name=dna_fig4_wconfirm
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Extend the dMFM curve of Figure 4 (right) to budgets below every baseline.
#
# The benchmark guides over t in [0.01, 0.95], i.e. 91 of the 96 outer steps, and
# the value estimator is one map evaluation per guided step -- so 91 of the dMFM's
# 187 generative calls, and all 91 of its backward passes, are spent guiding.
# Every other DNA experiment in this paper (Table 1, Fig 3, Fig 7, Fig 8) guides
# over [0.5, 0.95] instead, which is roughly half as many guided steps. That
# setting is already tuned for this task; it has simply never been used here.
#
# Narrowing the window is the same lever as QM9's --guide-every: it buys budget
# without touching the outer trajectory, which stays at 96 steps for every method
# compared. At t_start = 0.5 the MC=1 arm should cost about 139 generative calls
# and 43 backward passes, i.e. ~186 forward-equivalents at the measured k = 1.1 --
# below the cheapest point any baseline reaches (192).
#
# Each array task is one guidance window; shards are the 5 dMFM jobs x 3 repeats.
#
#   sbatch scripts/dna/fig4_window_sweep.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
PY="python"

WINDOWS=( 0.5 0.35 )
TS="${WINDOWS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT_DIR="outputs/dna/fig4/panelB/window_confirm_t${TS}"
N_OUTPUTS="${N_OUTPUTS:-100}"
REPEATS="${REPEATS:-3}"

echo "guidance window [${TS}, 0.95] -> ${OUT_DIR}"
for IDX in 0 1 2 3 4; do           # the five dMFM jobs of the sweep plan
  for REP in $(seq 0 $((REPEATS-1))); do
    ${PY} -u -m dmfm.benchmarks.run_steering \
      --plan sweep --index "${IDX}" --repeat "${REP}" \
      --out-dir "${OUT_DIR}" --n-outputs "${N_OUTPUTS}" --seed 7 \
      --guide-t-start "${TS}" \
      || echo "shard idx=${IDX} rep=${REP} FAILED"
  done
done
echo "done window ${TS}"
