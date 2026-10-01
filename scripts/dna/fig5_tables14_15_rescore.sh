#!/usr/bin/env bash
# Tables 14-15 (+ the Fig 5 inputs): scale-free rescore and summary of the
# gradient-MC runs. CPU only.
#
# Original (2026-08-02): scripts/rescore_gradient_pairs_scale_free.py over each
# workdir/rebuttal_gradient_accuracy_{gc,motif,conjunction} run, then
# scripts/summarize_gradient_accuracy_metrics.py, then
# scripts/plot_gradient_accuracy_required_mc.py for the figure inputs.
# The published Tables 14-15 are this rescore of those runs.
#
#   bash scripts/dna/fig5_tables14_15_rescore.sh outputs/dna/gradient_accuracy_gc [...]
set -euo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "${REPO}"
source scripts/env.sh

if [[ $# -eq 0 ]]; then
  set -- outputs/dna/gradient_accuracy_gc outputs/dna/gradient_accuracy_motif \
      outputs/dna/gradient_accuracy_conjunction
fi

for run in "$@"; do
  echo "== rescore ${run}"
  python -u -m dmfm.experiments.rescore_gradient_pairs_scale_free --input-dir "${run}" --bootstrap 10000 --seed 20260802
  echo "== summarize ${run}"
  python -u -m dmfm.experiments.summarize_gradient_accuracy_metrics --run_dir "${run}"
done

echo "== Fig 5 inputs"
python -u -m dmfm.experiments.plot_gradient_accuracy_required_mc \
  --motif_dir outputs/dna/gradient_accuracy_motif \
  --conjunction_dir outputs/dna/gradient_accuracy_conjunction \
  --output_dir outputs/dna/gradient_accuracy_figures
