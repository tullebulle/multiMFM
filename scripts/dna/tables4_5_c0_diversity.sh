#!/usr/bin/env bash
# Tables 4-5: sequence diversity of the C0-guided sets (CPU only, ~10 min).
#
# Original: workdir/rebuttal_parent_c0_diversity_20260925/analyze_parent_c0_diversity.py
# run in the multimfm env against the two n=1000 sample directories; the vendored
# DNA-MFM@187fe7b metric functions it used are dmfm.experiments._yeast_c0_diversity_metrics.
# All other defaults (--baseline_n 1000 --seed 0 --max_pairwise_n 3000
# --k_nearest_train 5 --pair_chunk 128 --size_matched_draws 1000) are unchanged.
#
# CAVEAT: the Tables 4-5 numbers printed in the paper come from the purged
# split65k pipeline; this reproduces the parent-disjoint recompute, which differs.
# See docs/provenance/dna/README.md.
#
#   bash scripts/dna/tables4_5_c0_diversity.sh <run_dir_target_m1> <run_dir_target_p1> [out_dir]
set -euo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "${REPO}"
source scripts/env.sh

RUN_M1="${1:-results/dna/c0_guidance/n1000/target_m1p0}"
RUN_P1="${2:-results/dna/c0_guidance/n1000/target_1p0}"
OUT_DIR="${3:-outputs/dna/c0_diversity}"

python -W ignore::FutureWarning -u -m dmfm.experiments.analyze_c0_diversity \
  --repo_root "${REPO}" \
  --run_dirs "${RUN_M1}" "${RUN_P1}" \
  --out_dir "${OUT_DIR}"
