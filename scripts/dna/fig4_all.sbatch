#!/bin/bash
#SBATCH --job-name=dna_fig4
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Figure 4, both panels, all shards: sampler calibration -> baseline pilot (n=32,
# seed 12345) -> panel A (flow states, then held-out data states) -> sweep (n=100 x
# 3 repeats, seeds 0/1/2, baseline hyperparameters frozen from the pilot).
#
# Every array task is an identical work-stealing worker (dmfm.benchmarks.worker):
# a unit whose output exists is skipped, a unit is claimed with an atomic lock, so
# one running task does everything (~1-2 GPU-hours) and more tasks split it. A
# preempted task resumes where it stopped. Resubmitting is always safe.
#
#   sbatch --array=0-5 scripts/dna/fig4_all.sbatch                                  # kempner_requeue (H100)
#   sbatch --partition=gpu_requeue --account=albergo_lab --array=0-3 \
#          scripts/dna/fig4_all.sbatch   # general GPU pool; kempner_albergo_lab is NOT a
#                                        # valid association there, and the pool is mixed
#                                        # (A100/H100/H200/RTX Pro). Same out-root, so the
#                                        # two routes skip and lock each other's shards.
#   bash scripts/dna/fig4_collect.sh          # afterwards (login node, CPU)
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

${PY} -u -m dmfm.benchmarks.worker --stages "${STAGES:-calibrate,pilot,panelA,panelA_c0,sweep}"
