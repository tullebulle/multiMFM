#!/usr/bin/env bash
# Collect the QM9 reruns into results/qm9 and re-execute notebook 07 (login node, CPU).
#
#   bash scripts/qm9/collect_table2_3seed.sh
#
# 1. outputs/qm9/table2_3seed/       -> results/qm9/rerun_3seed/   (Tables 2, 21, 22; 3 seeds)
# 2. outputs/qm9/ss_mfm_pilot/       -> results/qm9/ss_mfm_pilot/  (seed-8 n=100 SS pilot)
# 3. outputs/qm9/baselines_v2/       -> results/qm9/baselines/baseline_sweep_v2.csv + figure
# 4. notebooks/07_qm9_property_guidance.ipynb executed in place (RERUN = False)
# Safe to run while jobs are still going: missing runs show up as "pending".
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )}"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source scripts/env.sh
PY=python

if [ -d outputs/qm9/table2_3seed ]; then $PY -m multimfm.qm9_results rerun3; fi
if [ -d outputs/qm9/ss_mfm_pilot ]; then $PY -m multimfm.qm9_results pilot-ss; fi
if [ -d outputs/qm9/ss_mfm_reselect ]; then $PY -m multimfm.qm9_results ss-reselect; fi
if [ -d outputs/qm9/baselines_v2 ]; then
  $PY -m multimfm.qm9_results baselines-v2
  $PY -m multimfm.qm9_figures baselines
fi
squeue -u "${USER}" -h -o "%i %j %T" 2>/dev/null | grep "qm9_" || echo "no qm9 jobs in the queue"

cd notebooks
"$(dirname "$(command -v python)")/jupyter" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=multimfm --ExecutePreprocessor.timeout=1800 \
  07_qm9_property_guidance.ipynb
echo "notebook 07 re-executed"
