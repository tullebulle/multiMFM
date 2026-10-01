#!/usr/bin/env bash
#SBATCH --job-name=dna_matched_grid
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Put the dMFM on the SAME budget grid as the search baselines.
#
# best-of-N and Feynman-Kac are quantised to N x 96 network calls, so they sit at
# 192, 288, 480 and 864 forward-equivalents. The dMFM's own grid (188, 232, 320,
# 496, 848) came from sweeping MC at one guidance window, which lands near those
# points but not on them, and a figure where each method is measured somewhere
# else is harder to read than it needs to be.
#
# The dMFM's cost is exactly
#       96 + G (MC + 1.1),      G = guided steps = 92 - ceil(t_start x 96),
# verified against the measured runs, so choosing (t_start, MC) places a point on
# any budget. These four land on the baselines' grid to within 0.5%:
#
#   192 -> MC=1 t_start=0.4792     480 -> MC=4 t_start=0.1771
#   288 -> MC=2 t_start=0.3125     864 -> MC=8 t_start=0.0833
#
# Same 300 samples x 5 repeats as the rest of the comparison.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

N_OUTPUTS="${N_OUTPUTS:-300}"
REPEATS="${REPEATS:-5}"
# (plan index for that MC, t_start, label)
IDX=(0 1 2 3)                 # sweep-plan indices for MC = 1, 2, 4, 8
TS=(0.4792 0.3125 0.1771 0.0833)
I="${SLURM_ARRAY_TASK_ID:-0}"
idx="${IDX[$I]}"; ts="${TS[$I]}"
OUT="outputs/dna/fig4/matched/b$(printf '%d' $((192 * (I + 1))))"

echo "matched point: plan index ${idx} (MC) at t_start=${ts} -> ${OUT}"
for REP in $(seq 0 $((REPEATS-1))); do
  python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${idx}" --repeat "${REP}" \
    --out-dir "${OUT}" --n-outputs "${N_OUTPUTS}" --seed 0 --guide-t-start "${ts}" \
    || echo "FAILED idx=${idx} rep=${REP}"
done
echo done
