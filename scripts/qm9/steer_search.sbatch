#!/usr/bin/env bash
#SBATCH --job-name=qm9_steer_search
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# multiMFM-SS (steer_search) row of Table 2 / Table 21, n=1000 per property.
# Submit from the repo root:
#
#   sbatch --array=0-5 scripts/qm9/steer_search.sbatch               # all six, replicate "a"
#   REPLICATE=b sbatch --array=0-5 scripts/qm9/steer_search.sbatch   # second replicate
#   sbatch scripts/qm9/steer_search.sbatch lumo                      # one property
#   NUM=100 sbatch scripts/qm9/steer_search.sbatch alpha             # smaller n
#
# Array index -> property: 0 cv, 1 mu, 2 alpha, 3 gap, 4 homo, 5 lumo.
# The paper replicates a/b used identical arguments (same seeds); REPLICATE only
# changes the output directory. The n=1000 paper runs took ~1.5-3 h each on one
# H100 with 16 PoseBusters worker processes (2026-07-25); the PB check of every
# harvested look-ahead (65-90 per molecule) dominates.
# Partition/account lines are site-specific; override them on the sbatch command line.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs

PROPS=(cv mu alpha gap homo lumo)
PROP="${1:-${PROPS[${SLURM_ARRAY_TASK_ID:-2}]}}"
NUM="${NUM:-1000}"
REPLICATE="${REPLICATE:-a}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qm9/multimfm_ss/k_${PROP}_${REPLICATE}_n${NUM}}"
export PB_WORKERS="${PB_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null || echo none)"
bash "${REPO_ROOT}/scripts/run_steer_search.sh" "${PROP}" "${NUM}"
