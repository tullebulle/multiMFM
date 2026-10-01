#!/usr/bin/env bash
#SBATCH --job-name=qm9_dps_dose
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_requeue
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# DPS at matched DELIVERED guidance, to separate "more compute does not help DPS"
# from "our sampler delivers less guidance when you add steps".
#
# Why this is needed. CenteredMetricInterpolant.step is a contraction:
#     x_new = x_t (1 - dt/(1-t)) + x1_pred (dt/(1-t))
# so a perturbation injected at time t reaches the endpoint scaled by ~dt/(1-t).
# DPS injects mu*grad*dt, so one injection contributes ~dt^2 and the whole window
# contributes ~N*dt^2 = 1/N. Verified numerically: relative to 128 steps the
# delivered guidance is 0.250 / 0.125 / 0.0625 / 0.0312 at 512 / 1024 / 2048 / 4096.
# The earlier sweep therefore did NOT measure DPS with more compute; it measured DPS
# with up to 32x less guidance, and its monotone degradation is that dose reduction.
#
# Here mu is scaled proportionally to sample_steps so the delivered dose matches the
# 128-step mu=0.1 reference. If DPS still degrades under this, the degradation is
# real; if it flattens or improves, the earlier reading was an artifact.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source scripts/env.sh
mkdir -p logs

SEEDS=(1 2 3)
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
SEED="${SEEDS[$((I / 4))]}"
FAM=$((I % 4))
TSEED=$((2026 + SEED))
N="${N:-100}"
PROP=alpha
W="${SLURM_CPUS_PER_TASK:-16}"
OUT="${OUT_ROOT:-${REPO_ROOT}/outputs/qm9/baselines_v2}/seed${SEED}"

case "${FAM}" in
  0) S=512;  MU=0.4 ;;
  1) S=1024; MU=0.8 ;;
  2) S=2048; MU=1.6 ;;
  3) S=4096; MU=3.2 ;;
esac
DIR="dps_dosematched_s${S}"
if [ -f "${OUT}/${DIR}/dps_${PROP}_seed${SEED}.json" ]; then echo "skip ${DIR}"; exit 0; fi
echo "task=${I} seed=${SEED} steps=${S} mu=${MU} (mu scaled so delivered dose matches mu=0.1 at 128 steps)"
python -m multimfm.baselines --method dps --property-name "${PROP}" \
  --num-samples "${N}" --sample-steps "${S}" --seed "${SEED}" --target-seed "${TSEED}" \
  --pb-workers "${W}" --output-dir "${OUT}/${DIR}" --mu "${MU}" --guide-every 1 2>&1 \
  | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER" || true; }
echo "done $(date)"
