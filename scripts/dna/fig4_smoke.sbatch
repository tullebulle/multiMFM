#!/bin/bash
#SBATCH --job-name=dna_fig4_smoke
#SBATCH --partition=gpu_test
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Small-n GPU test of the Figure 4 rerun: one job of every method plus a short
# panel-A shard, and a timing line for each so the full arrays can be sized.
# Nothing here is a paper number.
#
#   sbatch scripts/dna/fig4_smoke.sbatch
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

OUT="${OUT:-outputs/dna/fig4_smoke}"
N="${N:-16}"
rm -rf "${OUT}"

echo "=== sampler calibration (unguided C0 vs eta) ==="
${PY} -u -m dmfm.benchmarks.run_steering --mode calibrate --n-outputs 512 \
  --out-dir "${OUT}/panelB/sweep"

echo "=== panel B: one job per method ==="
# sweep indices: 0 dmfm MC=1, 4 dmfm MC=16, 5 best_of_n N=2, 10 best_of_n N=32,
#                11 fk K=2, 16 fk K=32, 17 beam K=48, 24 beam K=4, 25 mcts I=2, 31 mcts I=64
for i in 0 4 5 10 11 16 17 24 25 31; do
  ${PY} -u -m dmfm.benchmarks.run_steering --plan sweep --index "${i}" --repeat 0 \
    --n-outputs "${N}" --out-dir "${OUT}/panelB/sweep"
done

echo "=== panel A: 1 condition, small reference (port of the recovered original producer) ==="
${PY} -u -m dmfm.benchmarks.value_error \
  --out "${OUT}/panelA/smoke.json" \
  --cond-start 0 --cond-end 1 --n-conditions 2 \
  --n-list 1 2 4 8 --sde-ref-n 64 --sde-steps 128 --fmap-nfe 32

echo "=== collect ==="
${PY} -u -m dmfm.benchmarks.collect \
  --panelA-dir "${OUT}/panelA" --panelB-dir "${OUT}/panelB/sweep" \
  --pilot-dir "${OUT}/panelB/pilot" --panelA-c0-dir "${OUT}/panelA_c0" --out "${OUT}/results"
cat "${OUT}/results/panelB_curves.csv"
echo "smoke done"
