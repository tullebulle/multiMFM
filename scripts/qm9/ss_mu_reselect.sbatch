#!/usr/bin/env bash
#SBATCH --job-name=qm9_ss_mu
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Guidance-strength check for the Algorithm-3 steer-search, on the TUNING seed only
# (seed 8 / target seed 2026, the first 300 of the submitted n=1000 target set;
# disjoint from the evaluation seeds 1-3). The submitted SS mu were tuned for the old
# base-endpoint harvesting; this asks whether the MFM-look-ahead selection wants a
# different mu. Grid per property: the multiMFM mu (literally "the same guided ODE as
# multiMFM", as §3.2 words it), the geometric midpoint, the submitted SS mu, 2 x SS mu;
# reward_scale is the same in both submitted configs.
#
# Pre-registered rule (applied in notebook 07 / qm9_results): keep the submitted SS mu
# unless another grid mu gives a lower Algorithm-3 oracle MAE by more than 2 standard
# errors of the difference (per-molecule errors, n = 300 each) with PB-valid >= 0.95.
#
#   sbatch scripts/qm9/ss_mu_reselect.sbatch
# Outputs: outputs/qm9/ss_mfm_reselect/<prop>_mu<mu>_n300/ (skipped if present).
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
OUT="${REPO_ROOT}/outputs/qm9/ss_mfm_reselect"
N="${N:-300}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-12}" SEED=8 TARGET_SEED=2026

declare -A GRID=(
  [cv]="2.8 3.7 5.0 10.0"
  [mu]="0.78 1.5 3.0 6.0"
  [alpha]="3.4 6.1 11.0 22.0"
  [gap]="0.17 0.41 1.0 2.0"
  [homo]="0.5 0.87 1.5 3.0"
  [lumo]="0.44 0.81 1.5 3.0"
)
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) PROPS=(alpha cv) ;;
  1) PROPS=(gap homo) ;;
  2) PROPS=(lumo mu) ;;
esac
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) props=${PROPS[*]} start=$(date)"
for P in "${PROPS[@]}"; do
  for MU in ${GRID[$P]}; do
    D="${OUT}/${P}_mu${MU}_n${N}"
    if [ -f "${D}/summary.json" ]; then echo "skip ${D}"; continue; fi
    OUTPUT_DIR="${D}" LOOKAHEAD_SOURCE=both EXTRA_ARGS=--diagnostic-variants \
      bash scripts/run_steer_search.sh "${P}" "${N}" "${MU}"
  done
done
echo "done $(date)"
