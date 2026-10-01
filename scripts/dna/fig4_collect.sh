#!/bin/bash
# Collect the Figure 4 shards into results/dna/fig4 and re-execute notebook 06.
# CPU only, runs on the login node in ~1 minute. Safe to run repeatedly and
# while shards are still arriving (it aggregates whatever is on disk).
#
#   bash scripts/dna/fig4_collect.sh
set -euo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "${REPO}"
# shellcheck disable=SC1091
source scripts/env.sh

# Shards live in one tree per device type: outputs/dna/fig4 (GPU, TF32) and
# outputs/dna/fig4_cpu (CPU, fp32). A plotted panel must come from ONE of them, so for
# each part we take whichever tree holds more shards rather than merging them. Override
# any of these to force a choice, e.g. PANEL_A_DIR=outputs/dna/fig4/panelA.
pick() {  # pick <relative path under each tree>
  local g="outputs/dna/fig4/$1" c="outputs/dna/fig4_cpu/$1"
  local ng nc
  ng=$(find "${g}" -name '*.json' 2>/dev/null | wc -l)
  nc=$(find "${c}" -name '*.json' 2>/dev/null | wc -l)
  if [ "${nc}" -gt "${ng}" ]; then echo "${c}"; else echo "${g}"; fi
}
PANEL_A_DIR="${PANEL_A_DIR:-$(pick panelA)}"
PANEL_A_C0_DIR="${PANEL_A_C0_DIR:-$(pick panelA_c0)}"
PANEL_B_DIR="${PANEL_B_DIR:-$(pick panelB/sweep)}"
PILOT_DIR="${PILOT_DIR:-$(pick panelB/pilot)}"
echo "panelA=${PANEL_A_DIR}"
echo "panelB=${PANEL_B_DIR}"
echo "pilot=${PILOT_DIR}"

python -m dmfm.benchmarks.collect \
  --panelA-dir "${PANEL_A_DIR}" \
  --panelA-c0-dir "${PANEL_A_C0_DIR}" \
  --panelB-dir "${PANEL_B_DIR}" \
  --pilot-dir "${PILOT_DIR}" \
  --out results/dna/fig4

python scripts/dna/fig4_freeze_pilot.py
python scripts/dna/fig4_write_provenance.py

cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=multimfm \
  --ExecutePreprocessor.timeout=1800 \
  06_dna_benchmarks.ipynb
echo "results/dna/fig4 updated and notebooks/06_dna_benchmarks.ipynb re-executed"
