#!/usr/bin/env bash
#SBATCH --job-name=qm9_matched_grid
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
# Put both multiMFM arms on the SAME budget grid as best-of-N.
#
# best-of-N costs N x 128 network calls, so it sits at 256, 512, 1,024 and 2,048
# forward-equivalents; Feynman-Kac lands within 12% of the same ladder. Our own
# points came from sweeping inner steps and guidance frequency, which gave 212,
# 296, 470 and 818 -- near those budgets but not on them.
#
# Our cost is exactly
#       128 + 3 G S MC,     G = guided steps = 128 (0.95 - guide_min_t),
# verified against the measured runs, so guide_min_t places a point on any budget:
#
#   256 -> S=1 MC=1 guide_min_t=0.614      1024 -> S=4 MC=1 guide_min_t=0.364
#   512 -> S=2 MC=1 guide_min_t=0.450      2048 -> S=4 MC=2 guide_min_t=0.325
#
# gamma is swept at each point rather than carried over, for the usual reason:
# changing S, MC or the window changes the gradient's scale. Query mode is the
# two-time flow map, which was the better of the two at both extremes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

PROP=alpha
N="${N:-512}"
W="${SLURM_CPUS_PER_TASK:-16}"
I="${SLURM_ARRAY_TASK_ID:-0}"
SD=$(( (I % 3) + 1 ))
HALF=$(( I / 3 ))

# budget: S MC guide_min_t
B=(256 512 1024 2048); S=(1 2 4 4); MC=(1 1 1 2); GMIN=(0.614 0.450 0.364 0.325)

run() { # i gamma
  local i="$1" G="$2"
  local out="outputs/qm9/matched_grid/seed${SD}/b${B[$i]}_g${G}"
  [ -f "${out}/summary.json" ] && { echo "skip b${B[$i]} g${G} seed${SD}"; return; }
  echo "----- budget ${B[$i]}: S=${S[$i]} MC=${MC[$i]} gmin=${GMIN[$i]} gamma=${G} seed=${SD} -----"
  python -m multimfm.steer_search --property-name "${PROP}" --num-samples "${N}" \
    --checkpoint checkpoints/base_flow/model_step_100000.pt \
    --mfm-checkpoint checkpoints/mfm_student/student_step_1000.pt --mfm-key student \
    --device cuda --sample-steps 128 --seed "${SD}" --target-seed $((2026 + SD)) \
    --mu "${G}" --reward-scale 0.3 --select-min-t 0.3 \
    --lookahead-source mfm --lookahead-topk 0 \
    --guide-every 1 --guide-min-t "${GMIN[$i]}" --guide-max-t 0.95 \
    --value-samples "${MC[$i]}" --value-glass-steps "${S[$i]}" --value-batch-size 0 \
    --eps 1e-6 --soft-atom-temperature 0.25 --partition train_flow \
    --pb-candidates 0 --pb-workers "${W}" \
    --output-dir "${out}" 2>&1 | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
}

for i in $(seq $((HALF * 2)) $((HALF * 2 + 1))); do
  for G in 1.5 2.2 3.4; do run "$i" "$G"; done
done
echo "done task ${I}"
