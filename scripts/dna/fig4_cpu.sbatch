#!/bin/bash
#SBATCH --job-name=dna_fig4_cpu
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# CPU workers for Figure 4 on serial_requeue (account albergo_lab; serial_requeue
# denies kempner_albergo_lab). Same work-stealing driver as scripts/dna/fig4_all.sbatch,
# but into a SEPARATE output tree, so what produced each number is obvious.
#
#     outputs/dna/fig4_cpu/...     CPU, true fp32, CPU RNG      (this script)
#     outputs/dna/fig4/...         GPU, TF32 matmuls, CUDA RNG  (fig4_all.sbatch)
#
# THE RULE: a *plotted* panel is produced entirely on ONE device type. CUDA runs the
# matmuls in TF32 and draws from a CUDA RNG stream; CPU runs true fp32 and a CPU RNG.
# The two therefore do not produce the same samples, and with no interactive GPU there
# is no A/B run that would show the difference sits below the reported error bars. So a
# panel is either all-GPU or all-CPU, never mixed. `dmfm.benchmarks.collect` warns if a
# shard directory contains more than one device type, `plan.pilot_scores` takes the
# pilot from one tree only, and every shard records its device, Slurm account and
# partition in its own metadata.
#
# STAGES (default `pilot`) selects what may run here:
#   pilot   SAFE. 49 shards at n=32 that only *rank* baseline hyperparameter families;
#           no pilot shard is ever plotted, and the winner is then measured on whatever
#           device produces the reported curve. Running it here also unblocks the GPU
#           sweep, which waits on the pilot.
#   panelA / panelA_c0 / sweep  only if the whole panel runs here (see fig4_cpu_cost).
#
#   sbatch --array=0-7 scripts/dna/fig4_cpu.sbatch                   # pilot on CPU
#   STAGES=pilot,panelA sbatch --array=0-15 scripts/dna/fig4_cpu.sbatch
#
# Afterwards `bash scripts/dna/fig4_collect.sh` reads both trees.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
echo "device=cpu threads=${OMP_NUM_THREADS} account=${SLURM_JOB_ACCOUNT:-?} partition=${SLURM_JOB_PARTITION:-?}"

${PY} -u -m dmfm.benchmarks.worker \
  --stages "${STAGES:-pilot}" \
  --out-root "${OUT_ROOT:-outputs/dna/fig4_cpu}" \
  --device cpu \
  --max-wait-min "${MAX_WAIT_MIN:-1}"
