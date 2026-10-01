#!/usr/bin/env bash
#SBATCH --job-name=qm9_multimfm
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# multiMFM (non-SS) row of Table 2 / Tables 21-22, n=1000 per property.
# Submit from the repo root (Slurm copies this script elsewhere, so the repo is
# located via $SLURM_SUBMIT_DIR, or $MULTIMFM_ROOT if set):
#
#   sbatch --array=0-5 scripts/qm9/run_multimfm.sbatch      # all six properties
#   sbatch scripts/qm9/run_multimfm.sbatch alpha            # one property
#   NUM=100 sbatch scripts/qm9/run_multimfm.sbatch alpha    # smaller n
#
# Array index -> property: 0 cv, 1 mu, 2 alpha, 3 gap, 4 homo, 5 lumo.
# One n=1000 run took ~7 min on an H100 (2026-07-25, two runs sharing a GPU).
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
TAG="${TAG:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qm9/multimfm/${PROP}_n${NUM}${TAG:+_${TAG}}}"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null || echo none)"
bash "${REPO_ROOT}/scripts/qm9/run_multimfm.sh" "${PROP}" "${NUM}"
