#!/usr/bin/env bash
# Submit the as-described Table 3 evaluation with the (lambda, mu) and guidance schedule
# that were selected on the held-out target pool.
#
#   bash scripts/geom/submit_table3_asdescribed.sh              # from the repo root
#   ARRAY=0-23%8 SHARDS=8 bash scripts/geom/submit_table3_asdescribed.sh
#
# Reads results/geom/table3_asdescribed_selection.json (written by
# scripts/geom/collect_asdescribed.py tune, i.e. by collect_table3_asdescribed.sh), prints
# what it is about to run, and submits scripts/geom/table3_asdescribed.sbatch. Finished
# shards are skipped, so this is also the way to resubmit after preemption.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

SEL="results/geom/table3_asdescribed_selection.json"
if [ ! -s "${SEL}" ]; then
  echo "missing ${SEL}: run the tuning sweep first" >&2
  echo "  sbatch scripts/geom/tune_lambda_mu.sbatch" >&2
  echo "  bash scripts/geom/collect_table3_asdescribed.sh" >&2
  exit 1
fi

read -r LAMBDA MU EVERY TMIN TMAX MUSCHED < <(
  "${HOME}/micromamba/envs/multimfm/bin/python" - "${SEL}" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["selected"]
print(s["reward_lambda"], s["mu"], s.get("guide_every", 2),
      s.get("guide_min_t", 0.1), s.get("guide_max_t", 0.9),
      s.get("mu_schedule", "constant"))
PY
)

echo "selected lambda=${LAMBDA} mu=${MU} [${MUSCHED}] guide_every=${EVERY} t in [${TMIN}, ${TMAX}]"
echo "submitting ${SEED_GROUPS:-3} seeds x ${SHARDS:-8} shards = 4,096 molecules per seed"
LAMBDA="${LAMBDA}" MU="${MU}" MU_SCHEDULE="${MUSCHED}" \
  GUIDE_EVERY="${EVERY}" GUIDE_MIN_T="${TMIN}" GUIDE_MAX_T="${TMAX}" \
  SHARDS="${SHARDS:-8}" SEED_GROUPS="${SEED_GROUPS:-3}" \
  sbatch --array="${ARRAY:-0-23%8}" scripts/geom/table3_asdescribed.sbatch
echo
echo "when it finishes: bash scripts/geom/collect_table3_asdescribed.sh"
