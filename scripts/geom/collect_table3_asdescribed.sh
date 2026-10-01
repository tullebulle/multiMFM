#!/usr/bin/env bash
# Single collection command for the as-described Table 3 rerun (authors' decision 1).
#
#   bash scripts/geom/collect_table3_asdescribed.sh            # from the repo root
#   JOB=<array job id> bash scripts/geom/collect_table3_asdescribed.sh
#   SHARDS=4 bash scripts/geom/collect_table3_asdescribed.sh   # if the eval used 4 shards/seed
#
# It (1) aggregates the held-out (lambda, mu) sweep and records the selected point,
# (2) aggregates the 3-seed x 4-shard 4,096-molecule evaluation into results/geom/
#     (table3_asdescribed{_tuning,_selection,_shards,_seeds}.* and table3_asdescribed.json,
#     and adds the run to table21_geom_row.json), and
# (3) re-executes notebooks/08_geom_fingerprint.ipynb so the notebook shows the submitted
#     value, the earlier 4,096x3 rerun and this run side by side.
#
# Safe to run while shards are still finishing: partial results are aggregated and
# table3_asdescribed.json reports "complete": false.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source scripts/env.sh
export MULTIMFM_ROOT="${REPO_ROOT}"

TUNE_DIR="results/geom/runs/asdescribed_tuning"
EVAL_DIR="results/geom/runs/asdescribed_eval"

if compgen -G "${TUNE_DIR}/*fingerprint_steering_*.json" > /dev/null; then
  echo "=== (lambda, mu) selection on the held-out pool ==="
  python scripts/geom/collect_asdescribed.py tune
else
  echo "no tuning results yet in ${TUNE_DIR} (skipping selection)"
fi

if compgen -G "${EVAL_DIR}/*fingerprint_steering_*.json" > /dev/null; then
  echo
  echo "=== as-described evaluation (4,096 molecules x 3 seeds) ==="
  python scripts/geom/collect_asdescribed.py eval \
    --shards "${SHARDS:-8}" --seed-groups "${SEED_GROUPS:-3}" ${JOB:+--job "${JOB}"}
else
  echo "no evaluation results yet in ${EVAL_DIR} (skipping)"
fi

echo
echo "=== re-executing notebooks/08_geom_fingerprint.ipynb ==="
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=multimfm 08_geom_fingerprint.ipynb
echo "done."
