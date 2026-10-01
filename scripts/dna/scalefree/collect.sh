#!/usr/bin/env bash
# Aggregate the dMFM scale-free gradient shards into results/dna/scale_free_mc/dmfm/.
# CPU only, seconds to a minute, and safe to re-run at any time: complete (reward x
# student) run directories are rescored from their stored gradient pairs, incomplete ones
# are listed and skipped.
#
#   bash scripts/dna/scalefree/collect.sh
#   OUTPUT_ROOT=outputs/dna/scale_free_mc_dmfm bash scripts/dna/scalefree/collect.sh
#
# Optional: verify the recomputed GLASS-2048 reference against the published Fig 5
# tensors (they live outside this repo, so give their parent directory):
#   PUBLISHED_ROOT=/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir \
#     bash scripts/dna/scalefree/collect.sh
set -euo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
cd "${REPO}"
source scripts/env.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dna/scale_free_mc_dmfm}"
RESULTS_DIR="${RESULTS_DIR:-results/dna/scale_free_mc/dmfm}"

python -u -m dmfm.scalefree.collect --output_root "${OUTPUT_ROOT}" --results_dir "${RESULTS_DIR}"

if [ -n "${PUBLISHED_ROOT:-}" ] && [ -d "${PUBLISHED_ROOT}" ]; then
  echo
  echo "== verifying the recomputed GLASS-2048 reference against the published tensors"
  python -u -m dmfm.scalefree.verify_reference \
    --output_root "${OUTPUT_ROOT}" \
    --published-root "${PUBLISHED_ROOT}" \
    --out "${RESULTS_DIR}/reference_verification.json"
fi
