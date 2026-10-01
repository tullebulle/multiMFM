#!/bin/bash
#SBATCH --job-name=dna_fa_smooth
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Item (c): is the accurate posterior's value function differentiable, or piecewise constant?
# CPU only (forward passes, small batches, no autodiff); the login node is cgroup-limited to one
# core with a load average near 50, hence Slurm.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

${PY} -u scripts/dna/findings_audit/value_smoothness.py --device cpu \
  --n_probes 4 --mc 8 --probe_times 0.5 0.7 --targets 1.0 -1.0 --n_dirs 3 \
  --rel_h 0.000001 0.00001 0.0001 0.001 0.01 0.03 0.1 0.3
