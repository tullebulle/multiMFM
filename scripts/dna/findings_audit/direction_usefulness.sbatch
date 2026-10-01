#!/bin/bash
#SBATCH --job-name=dna_fa_dir
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Item (c), single-step version: is the accurate posterior's value-gradient direction useful?
# CPU because the GPU pool is capped out at the account level (QOSMaxGRESPerUser) and this needs only
# one gradient per (probe, posterior) plus base-flow continuations.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

${PY} -u scripts/dna/findings_audit/direction_usefulness.py --device cpu \
  --n_probes 8 --mc 8 --probe_times 0.5 0.7 --targets 1.0 -1.0 \
  --deltas 0.1484 1.484 14.84 --n_random 2
