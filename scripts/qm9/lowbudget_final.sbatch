#!/usr/bin/env bash
#SBATCH --job-name=qm9_lowbudget_final
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Final, low-noise QM9 steering comparison over the budget range that matters.
#
# The published baseline sweep uses n = 100 molecules per seed over 3 seeds, which
# is where the visible scatter in the comparison comes from -- DPS in particular is
# non-monotone at that sample size. This re-runs OURS AND THE BASELINES at n = 512
# over 3 seeds (5x the molecules) and only over 128-4,096 forward-equivalent calls,
# the range the comparison is about. Both sides get the same treatment; raising n
# for our arms alone would be a smooth curve against a noisy one.
#
# Our configurations are the four held-out-confirmed winners plus two neighbours,
# all at gamma = 3.4 or 1.5 (the gamma = 6 pick did not survive its held-out seeds
# and is dropped).
#
# Array: task / 4 -> seed (1,2,3); task % 4 -> family (ours-cheap, ours-mid, bon, fk+dps).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

SEEDS=(1 2 3)
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
SEED="${SEEDS[$((I / 4))]}"
FAM=$((I % 4))
export SEED TARGET_SEED=$((2026 + SEED))
N="${N:-512}"
PROP=alpha
W="${SLURM_CPUS_PER_TASK:-16}"
OUT="outputs/qm9/lowbudget_final/seed${SEED}"
export PB_WORKERS="${W}" VALUE_BATCH_SIZE=0 TOPK=0 LOOKAHEAD_SOURCE=mfm
echo "task=${I} seed=${SEED} family=${FAM} n=${N}"

bl() { local dir="$1"; shift; local method="$1"; shift
  [ -f "${OUT}/${dir}/${method}_${PROP}_seed${SEED}.json" ] && { echo "skip ${dir}"; return; }
  python -m multimfm.baselines --method "${method}" --property-name "${PROP}" \
    --num-samples "${N}" --sample-steps 128 --seed "${SEED}" --target-seed "${TARGET_SEED}" \
    --pb-workers "${W}" --output-dir "${OUT}/${dir}" "$@" 2>&1 | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
}
ours() { # S guide_every MC gamma
  local d="${OUT}/ours_s$1_ge$2_mc$3_g$4"
  [ -f "${d}/summary.json" ] && { echo "skip ours s$1 ge$2 mc$3 g$4"; return; }
  OUTPUT_DIR="${d}" EXTRA_ARGS="--value-glass-steps $1 --guide-every $2 --value-samples $3" \
    bash scripts/run_steer_search.sh "${PROP}" "${N}" "$4" || echo "FAILED ours $1 $2 $3 $4" >&2
}

case "${FAM}" in
  0) ours 1 4 1 3.4; ours 1 2 1 3.4; ours 2 4 1 3.4 ;;
  1) ours 2 2 1 3.4; ours 2 1 1 1.5; ours 2 1 1 3.4 ;;
  2) bl base base; for K in 2 4 8 16 32; do bl "bon_n${K}" best_of_n --n-candidates "${K}"; done ;;
  3) for K in 4 8 16; do bl "fk_k${K}" fk --n-particles "${K}" --fk-lambda 0.01 --fk-resample-every 8; done
     for MU in 0.1 0.3; do bl "dps_mu${MU}" dps --mu "${MU}"; done
     for GE in 2 4 8; do bl "dps_mu0.3_ge${GE}" dps --mu 0.3 --guide-every "${GE}"; done ;;
esac
echo "done $(date)"
