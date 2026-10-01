#!/bin/bash
# Single collection command for the regenerated DNA items (Fig 6 + Table 16, Fig 7, Fig 8,
# Table 6). Reduces the sharded/per-strength outputs into results/dna/regen, rewrites
# results/dna/regen/PROVENANCE.md and re-executes notebooks/09_dna_regenerated.ipynb.
# Safe to run repeatedly and while some jobs are still queued: unfinished items are listed
# as "pending" and the notebook still executes.
#
#   bash scripts/dna/regen/collect.sh
#   N=10000 NUM_SHARDS=20 bash scripts/dna/regen/collect.sh      # must match the Fig 6 run
#   SKIP_NOTEBOOK=1 bash scripts/dna/regen/collect.sh
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "run from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

N="${N:-10000}"
NUM_SHARDS="${NUM_SHARDS:-20}"   # matches the live CPU route; auto-detected from filenames once shards exist
SEED="${SEED:-0}"

${PY} -u -m dmfm.regen.collect --n "${N}" --num_shards "${NUM_SHARDS}" --seed "${SEED}" "$@"

if [ "${SKIP_NOTEBOOK:-0}" = "1" ]; then
  echo "SKIP_NOTEBOOK=1: not executing the notebook"
  exit 0
fi

cd "${REPO}/notebooks"
"${HOME}"/micromamba/envs/multimfm/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=multimfm --ExecutePreprocessor.timeout=1800 \
  09_dna_regenerated.ipynb
echo "re-executed notebooks/09_dna_regenerated.ipynb"
