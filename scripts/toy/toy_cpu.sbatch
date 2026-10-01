#!/usr/bin/env bash
#SBATCH --job-name=toy_cpu
#SBATCH --account=albergo_lab
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CPU fallback for the toy stages (the models are ~1.8M parameters).  Each stage
# writes its own output file and skips itself if that file already exists, so a
# requeued job resumes rather than restarting.
#   sbatch --export=ALL,STAGES="probe_windows posterior_sweep" scripts/toy/toy_cpu.sbatch
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export DEVICE=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
for stage in ${STAGES:-probe_windows posterior_sweep}; do
  echo "=== stage ${stage} ($(date)) ==="
  bash "${REPO_ROOT}/scripts/toy/${stage}.sh"
done
echo "=== done ($(date)) ==="
