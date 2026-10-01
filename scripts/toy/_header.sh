# Shared preamble for the toy (Fig. 1) scripts.  Source it from the repo root:
#   source scripts/toy/_header.sh
# Resolves the repo root (MULTIMFM_ROOT, else the Slurm submit dir, else $PWD),
# activates the multimfm env and cds there.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; run from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX}")" != "${MULTIMFM_ENV:-multimfm}" ]; then
  # shellcheck disable=SC1091
  source scripts/env.sh
fi
export PYTHONUNBUFFERED=1
PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
CKPT_DIR="${CKPT_DIR:-checkpoints/toy}"
echo "repo=${REPO_ROOT} host=$(hostname) job=${SLURM_JOB_ID:-none} device=${DEVICE}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi
