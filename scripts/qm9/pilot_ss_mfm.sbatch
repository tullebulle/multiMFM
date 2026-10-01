#!/usr/bin/env bash
#SBATCH --job-name=qm9_ss_pilot
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=h100|h200
#SBATCH --requeue
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
# Small-n test of the Algorithm-3 steer-search (MFM look-aheads) before the 3-seed
# n=1000 runs, on the TUNING seed/targets (seed 8, target seed 2026: the set the
# submitted SS configs were chosen on, disjoint from the evaluation seeds 1-3).
# Submit from the repo root:   sbatch scripts/qm9/pilot_ss_mfm.sbatch
# (three array tasks = three lanes of runs; ran on kempner_requeue because gpu_test was
# congested; gpu_test works too: sbatch -p gpu_test --constraint= ...). Preempted lanes
# rerun only the runs whose summary.json is missing.
#
# Writes outputs/qm9/ss_mfm_pilot/:
#   <prop>_n100/          all six properties, n=100, submitted SS configs,
#                         LOOKAHEAD_SOURCE=both + --diagnostic-variants: on ONE guided
#                         trajectory, the paper selection (mfm, every guided step, value
#                         score), the old base_endpoint harvest (t >= select_min_t),
#                         mfm with guide_hard re-scoring, mfm restricted to t >= select_min_t
#   alpha_n100_topk0/     alpha with every look-ahead kept (exactness check of top-K)
#   multimfm_alpha_n100/  property_targets (non-SS) alpha, n=100, per-sample PB + NFE
# Array task i runs lane i sequentially on one GPU.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
OUT="${REPO_ROOT}/outputs/qm9/ss_mfm_pilot"
N="${N:-100}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-8}" SEED=8 TARGET_SEED=2026

skip_or() {  # run "$2" unless $1/summary.json exists
  if [ -f "$1/summary.json" ]; then echo "skip $1"; else eval "$2"; fi
}
ss() {  # prop -> command for the SS run of that property
  echo "OUTPUT_DIR=${OUT}/$1_n${N} LOOKAHEAD_SOURCE=both EXTRA_ARGS=--diagnostic-variants bash scripts/run_steer_search.sh $1 ${N}"
}
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) task=${SLURM_ARRAY_TASK_ID:-0} start=$(date)"
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) skip_or "${OUT}/alpha_n${N}" "$(ss alpha)"
     skip_or "${OUT}/cv_n${N}" "$(ss cv)"
     skip_or "${OUT}/multimfm_alpha_n${N}" "OUTPUT_DIR=${OUT}/multimfm_alpha_n${N} PER_SAMPLE_PB=1 bash scripts/qm9/run_multimfm.sh alpha ${N}" ;;
  1) skip_or "${OUT}/gap_n${N}" "$(ss gap)"
     skip_or "${OUT}/homo_n${N}" "$(ss homo)"
     skip_or "${OUT}/alpha_n${N}_topk0" "OUTPUT_DIR=${OUT}/alpha_n${N}_topk0 LOOKAHEAD_SOURCE=mfm TOPK=0 bash scripts/run_steer_search.sh alpha ${N}" ;;
  2) skip_or "${OUT}/lumo_n${N}" "$(ss lumo)"
     skip_or "${OUT}/mu_n${N}" "$(ss mu)" ;;
esac
echo "done $(date)"
