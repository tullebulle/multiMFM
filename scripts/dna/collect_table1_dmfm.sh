#!/bin/bash
# Collect the dMFM Table 1 run and its matched GLASS baseline (both from
# scripts/dna/table1_c0_guidance_dmfm.sbatch, run with GRAD_NORMALIZE=1): merge the shards,
# score with the parent-disjoint oracle (CPU is fine), copy the small results to
# results/dna/c0_guidance/dmfm_n1000/ and write results/dna/c0_guidance/table1_dmfm_n1000_summary.json.
# Fails (listing the missing ids) if a shard is not finished yet; resubmitting the sbatch reruns
# only the missing shards.
#
#   bash scripts/dna/collect_table1_dmfm.sh [job ids ...]
#   bash scripts/dna/collect_table1_dmfm.sh 49352242 49352244                  # both arms (default)
#   GLASS_RUN_ROOT= bash scripts/dna/collect_table1_dmfm.sh 49352242            # dMFM arm only
#   PARTIAL=1 bash scripts/dna/collect_table1_dmfm.sh ...   # collect the finished shards only (n < 1000)
set -euo pipefail
cd "$( dirname "${BASH_SOURCE[0]}" )/../.."
# shellcheck disable=SC1091
source scripts/env.sh
# dMFM arm: the 4-step ESD student at 4 composed steps, scale-free rule (job 49352242).
RUN_ROOT="${RUN_ROOT:-outputs/dna/table1_dmfm4_norm_n1000}"
# Matched baseline: GLASS-4 under the identical rule (job 49352244). Set GLASS_RUN_ROOT= to skip.
GLASS_RUN_ROOT="${GLASS_RUN_ROOT-outputs/dna/table1_glass_norm_n1000}"
python -u -m dmfm.experiments.collect_table1_c0 \
  --run_root "${RUN_ROOT}" \
  --results_dir results/dna/c0_guidance/dmfm_n1000 \
  ${GLASS_RUN_ROOT:+--glass_run_root "${GLASS_RUN_ROOT}"} \
  ${GLASS_RUN_ROOT:+--glass_results_dir results/dna/c0_guidance/glass_matched_n1000} \
  --withdrawn_run_root outputs/dna/table1_dmfm_n1000 \
  --summary_json results/dna/c0_guidance/table1_dmfm_n1000_summary.json \
  --n_samples "${N_SAMPLES:-1000}" \
  ${PARTIAL:+--allow_partial} \
  --jobs "$@"
