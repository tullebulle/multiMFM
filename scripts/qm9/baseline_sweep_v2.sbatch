#!/usr/bin/env bash
#SBATCH --job-name=qm9_bl_v2
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --array=0-14%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Same-backbone QM9 steering comparison at matched NFE (authors' decision 8), the
# fixed version of scripts/sweep_nfe_benchmark.sbatch: every point is MEASURED (no
# extrapolation), NFE is counted by forward hooks for every method including multiMFM,
# best-of-N is extended to N = 256, 512 so that it reaches multiMFM's budget, and the
# sweep is repeated on the three evaluation seeds (1-3, target seed 2026 + seed; all
# method hyperparameters were chosen on seed 8). alpha, n = 100 targets per seed.
#
#   sbatch scripts/qm9/baseline_sweep_v2.sbatch       # 15 tasks = 3 seeds x 5 families
#
# task i: seed = (1, 2, 3)[i / 5], family = i % 5:
#   0 base + best-of-N, N in {4, 8, 16, 32, 64, 128, 256, 512}
#   1 Feynman-Kac, K in {4, 8, 16, 32} x lambda in {0.003, 0.01, 0.03}, resample every 8
#   2 DPS, mu in {0.03, 0.1, 0.3, 1.0}; guide_every in {2, 4, 8} at mu = 0.1
#   3 multiMFM at the Table-2 multiMFM config (mu 3.4, rs 0.3), MC in {1,2,4,8,16,32},
#     steer_search --lookahead-source both (endpoint = multiMFM; SS on the same ODE)
#   4 multiMFM-SS at the Table-2 SS config (mu 11, rs 0.3, select_min_t 0.3), same MC grid
# Outputs: outputs/qm9/baselines_v2/seed<s>/<config>/ (runs whose .json exists are
# skipped). Collect + figure: bash scripts/qm9/collect_table2_3seed.sh (or
# python -m multimfm.qm9_results baselines-v2; python -m multimfm.qm9_figures baselines).
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
# shellcheck source=/dev/null
source scripts/env.sh

SEEDS=(1 2 3)
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
SEED="${SEEDS[$((I / 5))]}"
FAM=$((I % 5))
TSEED=$((2026 + SEED))
N="${N:-100}"
PROP=alpha
W="${SLURM_CPUS_PER_TASK:-16}"
OUT="${OUT_ROOT:-${REPO_ROOT}/outputs/qm9/baselines_v2}/seed${SEED}"
echo "task=${I} seed=${SEED}/${TSEED} family=${FAM} host=$(hostname) start=$(date)"

bl() {  # dir method args...
  local dir="$1"; shift; local method="$1"; shift
  if [ -f "${OUT}/${dir}/${method}_${PROP}_seed${SEED}.json" ]; then echo "skip ${dir}"; return; fi
  echo "----- ${dir} -----"
  python -m multimfm.baselines --method "${method}" --property-name "${PROP}" \
    --num-samples "${N}" --sample-steps 128 --seed "${SEED}" --target-seed "${TSEED}" \
    --pb-workers "${W}" --output-dir "${OUT}/${dir}" "$@" 2>&1 | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
}
ss() {  # dir mu rs selmint mc
  if [ -f "${OUT}/$1/summary.json" ]; then echo "skip $1"; return; fi
  echo "----- $1 -----"
  python -m multimfm.steer_search --property-name "${PROP}" --num-samples "${N}" \
    --sample-steps 128 --seed "${SEED}" --target-seed "${TSEED}" \
    --mu "$2" --reward-scale "$3" --select-min-t "$4" --lookahead-source both \
    --mfm-diagonal --guide-every 1 --guide-min-t 0.05 --guide-max-t 0.95 \
    --value-samples "$5" --value-glass-steps 4 --pb-workers "${W}" \
    --output-dir "${OUT}/$1" 2>&1 | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
}

case "${FAM}" in
  0) bl base base
     for K in 4 8 16 32 64 128 256 512; do bl "bon_n${K}" best_of_n --n-candidates "${K}"; done ;;
  1) for K in 4 8 16 32; do for LAM in 0.003 0.01 0.03; do
       bl "fk_k${K}_lam${LAM}" fk --n-particles "${K}" --fk-lambda "${LAM}" --fk-resample-every 8
     done; done ;;
  2) for MU in 0.03 0.1 0.3 1.0; do bl "dps_mu${MU}" dps --mu "${MU}"; done
     for GE in 2 4 8; do bl "dps_mu0.1_ge${GE}" dps --mu 0.1 --guide-every "${GE}"; done ;;
  3) for MC in 1 2 4 8 16 32; do ss "mmfm_mu3.4_mc${MC}" 3.4 0.3 0.3 "${MC}"; done ;;
  4) for MC in 1 2 4 8 16 32; do ss "mmfmss_mu11_mc${MC}" 11.0 0.3 0.3 "${MC}"; done ;;
esac
echo "done $(date)"
