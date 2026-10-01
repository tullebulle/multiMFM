#!/bin/bash
#SBATCH --job-name=dna_fig4_cpucost
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Cost out the Figure 4 shards on CPU before moving any of them off the GPU queue.
# Measures, on the real serial_requeue hardware with the real settings:
#   1. one *real* pilot shard (n=32) for the most expensive pilot configuration;
#   2. one *real* panel-A condition at the original's true --sde-ref-n 512 /
#      --sde-steps 1024 (the reference is the expensive part, and reducing it would
#      change a published-facing number, so it is not reduced);
#   3. one *real* sweep shard (n=100) for the most expensive dMFM setting, whose
#      backward passes are the part CPU handles worst.
# Nothing here is a paper number: it writes to a throwaway directory.
#
#   sbatch scripts/dna/fig4_cpu_cost.sbatch
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

OUT="${OUT:-outputs/dna/fig4_cpucost}"
rm -rf "${OUT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
echo "threads=${OMP_NUM_THREADS} account=${SLURM_JOB_ACCOUNT:-?} partition=${SLURM_JOB_PARTITION:-?}"

t0=$(date +%s)
echo "=== [1/3] pilot shard, n=32, beam W=2,L=2,K=8 (most expensive pilot config) ==="
${PY} -u -m dmfm.benchmarks.run_steering --plan pilot --index "${PILOT_INDEX:-21}" --repeat 0 \
  --n-outputs 32 --seed 12345 --device cpu --out-dir "${OUT}/pilot" || echo "PILOT FAILED"
t1=$(date +%s); echo "### pilot shard wall seconds: $((t1 - t0))"

echo "=== [2/3] panel-A: ONE condition at the original's true reference settings ==="
${PY} -u -m dmfm.benchmarks.value_error --out "${OUT}/panelA/one.json" \
  --n-conditions 32 --cond-start 0 --cond-end 1 --device cpu || echo "PANELA FAILED"
t2=$(date +%s); echo "### panel-A one condition wall seconds: $((t2 - t1))"

echo "=== [3/3] sweep shard, n=100, dMFM MC=16 (backward-pass heavy) ==="
${PY} -u -m dmfm.benchmarks.run_steering --plan sweep --index 4 --repeat 0 \
  --n-outputs 100 --seed 0 --device cpu --out-dir "${OUT}/sweep" || echo "SWEEP FAILED"
t3=$(date +%s); echo "### sweep dMFM MC=16 shard wall seconds: $((t3 - t2))"

echo "=== totals ==="
echo "pilot_shard_s=$((t1 - t0)) panelA_condition_s=$((t2 - t1)) sweep_dmfm_shard_s=$((t3 - t2))"
echo "extrapolation: pilot = 49 shards, panel A = 32 conditions, sweep = 96 shards"
