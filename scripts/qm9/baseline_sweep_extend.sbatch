#!/usr/bin/env bash
#SBATCH --job-name=qm9_sweep_ext
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_requeue
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# kempner_requeue contains 24 RTX PRO 6000 Blackwell nodes (sm_120). The pinned
# torch 2.5.1+cu124 builds sm_50-sm_90 only, so a job landing there dies ~40 s in
# with "CUDA error: no kernel image is available for execution on the device".
# Those nodes carry the feature rtx6000pro and no cc8.0/cc9.0; constrain to the
# supported cards instead.
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Extends the QM9 steering-vs-NFE comparison so Feynman-Kac and DPS reach the same
# budgets as multiMFM and best-of-N. In baseline_sweep_v2 they stopped early:
# FK at K=32 (4,576 NFE) and DPS at 243 NFE, against a ladder running to ~16,000.
#
#   Feynman-Kac   K in {64, 128}, lambda = 0.01 fixed (best of the re-bracketed grid)
#   DPS           sample_steps in {512, 1024, 2048, 4096, 8192}, mu = 0.1, guide_every = 1
#
# Why sample_steps is the right budget axis for DPS, and not a confound: DPS applies
# its guidance as `mu * grad * dt` (glass_guidance.production_step_with_optional_guidance),
# so the integral of the guidance over the trajectory is invariant to the number of
# steps. Refining the schedule buys a more accurate solution of the SAME guided ODE at
# the SAME effective strength, rather than quietly turning the strength up.
# `guide_every` is already at its floor of 1, so it offers no further budget.
# NFE is then roughly 1.9 * sample_steps (one forward per step, plus one guidance
# gradient per step inside the t-window), giving ~970 / 1,940 / 3,890 / 7,780 / 15,550.
#
# Everything else is held at the v2 settings so the new points drop onto the existing
# curves: alpha, n=100 per seed, seeds {1,2,3}, target seed 2026+seed, same base model.
# Each configuration skips itself if its summary already exists, so a requeued task
# resumes rather than repeating work.
#
#   sbatch --array=0-14 scripts/qm9/baseline_sweep_extend.sbatch
set -euo pipefail

REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "submit from the repo root, or set MULTIMFM_ROOT" >&2
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

run() {  # dir method steps args...
  local dir="$1"; shift; local method="$1"; shift; local steps="$1"; shift
  if [ -f "${OUT}/${dir}/${method}_${PROP}_seed${SEED}.json" ]; then echo "skip ${dir}"; return; fi
  echo "----- ${dir} (sample_steps=${steps}) -----"
  python -m multimfm.baselines --method "${method}" --property-name "${PROP}" \
    --num-samples "${N}" --sample-steps "${steps}" --seed "${SEED}" --target-seed "${TSEED}" \
    --pb-workers "${W}" --output-dir "${OUT}/${dir}" "$@" 2>&1 \
    | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
}

case "${FAM}" in
  0) run "fk_k64_lam0.01"  fk  128 --n-particles 64  --fk-lambda 0.01 --fk-resample-every 8 ;;
  1) run "fk_k128_lam0.01" fk  128 --n-particles 128 --fk-lambda 0.01 --fk-resample-every 8 ;;
  2) for S in 512 1024; do run "dps_mu0.1_s${S}" dps "${S}" --mu 0.1 --guide-every 1; done ;;
  3) for S in 2048 4096; do run "dps_mu0.1_s${S}" dps "${S}" --mu 0.1 --guide-every 1; done ;;
  4) run "dps_mu0.1_s8192" dps 8192 --mu 0.1 --guide-every 1 ;;
esac
echo "done $(date)"
