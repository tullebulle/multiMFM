#!/bin/bash
#SBATCH --job-name=dna_fa_cpuhedge
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CPU hedge for item (c) in case the GPU queue never clears: the same end-to-end test at n = 8 and
# y* = -1 only, which is enough to separate "guides" (oracle MAE ~0.13-0.2) from "does not guide at
# all" (~0.81, the unguided value). RK4-25 is left out because 100 posterior evaluations per MC draw
# is too slow on CPU; `glass_euler32_e999` is the accurate non-degenerate posterior here.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

${PY} -u scripts/dna/findings_audit/posterior_accuracy_probe.py \
  --mode sample --n_samples 8 --batch_size 8 --targets -1.0 --device cpu \
  --configs glass_euler4 glass_euler32 glass_euler32@norm \
            glass_euler32_e999 glass_euler32_e999@norm \
  --out_dir results/dna/findings_audit/sample_posterior_accuracy_cpu_n8
