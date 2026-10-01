# Shared preamble for the GEOM-Drugs (Table 3 / Tables 19-21) Slurm wrappers.
#   source "${REPO_ROOT}/scripts/geom/_header.sh"
#
# Slurm runs a *copy* of the submitted script, so the repository is located via
# $MULTIMFM_ROOT, else $SLURM_SUBMIT_DIR (submit from the repo root), else $PWD.
# No absolute paths anywhere.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
# shellcheck disable=SC1091
source scripts/env.sh
export MULTIMFM_ROOT="${REPO_ROOT}"
export PYTHONUNBUFFERED=1
PY="${PY:-python}"
echo "repo=${REPO_ROOT} host=$(hostname) job=${SLURM_JOB_ID:-none}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi
