#!/usr/bin/env bash
# Tables 4-5 (+ the App. B.1 clustering statistic) recomputed on the **dMFM**-guided
# n = 1,000 C0 sets (REPRODUCIBILITY_PLAN.md S7 item 8). CPU only, ~10 min.
#
# WHY. Tables 4-6 score *generated sequences*. The sets they were computed from came from
# GLASS guidance on the base DFM (the purged split65k pipeline for the published numbers, and
# `results/dna/c0_guidance/n1000/` for the parent-disjoint recompute in
# `results/dna/diversity/regen/`), so as they stand the tables describe GLASS output. This
# rebuilds them from the dMFM sets that Table 1 / Fig 3 now cite, produced by
# `scripts/dna/table1_c0_guidance_dmfm.sbatch` with `--posterior dmfm --grad_normalize
# --nfe_value 4` (the two S7-item-8 conventions) -- Slurm 49352242.
#
# Table 4 is real-data baselines only and is posterior-independent; it is recomputed here so
# the ratios in Table 5 divide by the same denominators, and it should agree with
# `results/dna/diversity/regen/table4_real_baselines.csv` to the last digit.
#
# Nothing under results/dna/diversity/ is touched: output goes to
# outputs/dna/dmfm_demos/diversity and is collected into results/dna/dmfm_demos/diversity.
#
#   bash scripts/dna/dmfm_demos/tables4_5_diversity_dmfm.sh                  # dMFM sets
#   ARM=glass_norm bash scripts/dna/dmfm_demos/tables4_5_diversity_dmfm.sh   # matched control
#
# All analyser defaults are the published ones (--baseline_n 1000 --seed 0
# --max_pairwise_n 3000 --k_nearest_train 5 --pair_chunk 128 --size_matched_draws 1000).
set -euo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
cd "${REPO}"
source scripts/env.sh

ARM="${ARM:-dmfm4_norm}"
SET_ROOT="${SET_ROOT:-outputs/dna/table1_${ARM}_n1000}"
RUN_M1="${RUN_M1:-${SET_ROOT}/target_m1p0/merged}"
RUN_P1="${RUN_P1:-${SET_ROOT}/target_1p0/merged}"
OUT_DIR="${OUT_DIR:-outputs/dna/dmfm_demos/diversity/${ARM}}"

for d in "${RUN_M1}" "${RUN_P1}"; do
  [ -f "${d}/sample_scores.csv" ] || { echo "missing ${d}/sample_scores.csv" >&2; exit 2; }
  n=$(( $(wc -l < "${d}/sample_scores.csv") - 1 ))
  [ "${n}" -eq "${EXPECT_N:-1000}" ] || { echo "${d} has ${n} rows, expected ${EXPECT_N:-1000}" >&2; exit 2; }
done

mkdir -p "${OUT_DIR}"
echo "arm=${ARM} m1=${RUN_M1} p1=${RUN_P1} -> ${OUT_DIR}"
python -W ignore::FutureWarning -u -m dmfm.experiments.analyze_c0_diversity \
  --repo_root "${REPO}" \
  --run_dirs "${RUN_M1}" "${RUN_P1}" \
  --out_dir "${OUT_DIR}"
