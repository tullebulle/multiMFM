#!/bin/bash
#SBATCH --job-name=dna_fig4_panelB
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Figure 4 (right): target error vs NFE, rerun on the parent-disjoint L=50 models
# with real best-of-N / Feynman-Kac / beam / MCTS baselines (authors' decision 3).
#
# Work unit = one (method, budget, repeat) shard, written as one JSON; a shard whose
# JSON exists is skipped, so a preempted/failed task just gets resubmitted. Shard s
# is (job index s % n_jobs, repeat s // n_jobs). Array task a of an array of size W
# runs shards a, a+W, a+2W, ... in one allocation (each shard takes ~1-3 min, so
# packing them avoids paying the queue wait per shard).
#
#   PLAN=pilot N_OUTPUTS=32 SEED=12345 REPEATS=1 sbatch --array=0-3 scripts/dna/fig4_panelB.sbatch
#   PLAN=sweep                          REPEATS=3 sbatch --array=0-5 scripts/dna/fig4_panelB.sbatch
# The sweep reads the pilot's winners from outputs/dna/fig4/panelB/pilot at start-up
# (dmfm.benchmarks.plan.frozen_from_pilot), so launch it after the pilot has finished
# (e.g. --dependency=afterany:<pilot job id>).
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

PLAN="${PLAN:-sweep}"
OUT_DIR="${OUT_DIR:-outputs/dna/fig4/panelB/${PLAN}}"
N_OUTPUTS="${N_OUTPUTS:-100}"
SEED="${SEED:-0}"
REPEATS="${REPEATS:-3}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
WORKERS="${WORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

N_JOBS=$(${PY} -c "from dmfm.benchmarks import plan; print(len(plan.PLANS['${PLAN}']))")
N_SHARDS=$(( N_JOBS * REPEATS ))
echo "plan=${PLAN} n_jobs=${N_JOBS} repeats=${REPEATS} shards=${N_SHARDS} worker=${TASK}/${WORKERS}"
${PY} -c "from dmfm.benchmarks import plan; print('frozen hyperparameters:', plan.frozen_from_pilot())"

# calibration: run_steering takes an O_EXCL lock, writes atomically and refuses a
# calibration.json made with other settings, so concurrent tasks cannot clash.
if [ "${PLAN}" = "pilot" ] && [ "${TASK}" = "0" ]; then
  ${PY} -u -m dmfm.benchmarks.run_steering --mode calibrate --n-outputs 1024 --seed 0 \
    --out-dir "outputs/dna/fig4/panelB/sweep" || echo "calibration FAILED or refused"
fi

for (( s = TASK; s < N_SHARDS; s += WORKERS )); do
  INDEX=$(( s % N_JOBS ))
  REPEAT=$(( s / N_JOBS ))
  ${PY} -u -m dmfm.benchmarks.run_steering \
    --plan "${PLAN}" \
    --index "${INDEX}" \
    --repeat "${REPEAT}" \
    --out-dir "${OUT_DIR}" \
    --n-outputs "${N_OUTPUTS}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS:-} || echo "shard ${s} (index ${INDEX} repeat ${REPEAT}) FAILED"
done
echo "worker ${TASK} done"
