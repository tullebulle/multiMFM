#!/usr/bin/env bash
#SBATCH --job-name=dna_lowbudget_final
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Final, low-noise DNA steering comparison over the budget range that matters.
#
# The plotted curves use 100 samples x 3 repeats. This raises that to 300 x 5 for
# the dMFM arms at the two guidance windows that win (t_start 0.5 and 0.35) and,
# equally, for the four search baselines, so neither side is smoother than the
# other. Everything else -- the base generator, guide, target, seeds and the 96
# outer steps -- is unchanged.
#
# Tasks 0-1: dMFM at the two windows. Tasks 2-5: one baseline family each.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
PY="python"
N_OUTPUTS="${N_OUTPUTS:-300}"
REPEATS="${REPEATS:-5}"
I="${SLURM_ARRAY_TASK_ID:-0}"

run_idx() { # out_dir index extra...
  local out="$1" idx="$2"; shift 2
  for REP in $(seq 0 $((REPEATS-1))); do
    ${PY} -u -m dmfm.benchmarks.run_steering --plan sweep --index "${idx}" --repeat "${REP}" \
      --out-dir "${out}" --n-outputs "${N_OUTPUTS}" --seed 0 "$@" \
      || echo "FAILED idx=${idx} rep=${REP} in ${out}"
  done
}

case "${I}" in
  0) for IDX in 0 1 2 3 4; do run_idx "outputs/dna/fig4/final/window_t0.5"  "${IDX}" --guide-t-start 0.5;  done ;;
  1) for IDX in 0 1 2 3 4; do run_idx "outputs/dna/fig4/final/window_t0.35" "${IDX}" --guide-t-start 0.35; done ;;
  *) # baselines: the sweep plan's non-dMFM jobs, split four ways by family
     ${PY} - "$I" <<'PYEOF' > /tmp/idx_$$.txt
import sys
from dmfm.benchmarks import plan
fam = ["best_of_n", "fk", "beam", "mcts"][int(sys.argv[1]) - 2]
print(" ".join(str(i) for i, j in enumerate(plan.PLANS["sweep"]) if j.method == fam))
PYEOF
     for IDX in $(cat /tmp/idx_$$.txt); do run_idx "outputs/dna/fig4/final/baselines" "${IDX}"; done
     rm -f /tmp/idx_$$.txt ;;
esac
echo "done task ${I}"
