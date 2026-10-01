#!/usr/bin/env bash
#SBATCH --job-name=toy
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Run one or more toy stages in sequence on one GPU (submit from the repo root):
#   sbatch --export=ALL,STAGES="train_base train_mfm_diagonal" scripts/toy/toy.sbatch
#   sbatch --export=ALL,STAGES="train_mfm_consistency" scripts/toy/toy.sbatch
#   sbatch --export=ALL,STAGES="steer_fig1" scripts/toy/toy.sbatch
# Stages are the scripts/toy/<stage>.sh files.  Original runs were on a single
# GPU; the ESD stage took 1085 s there (terminal log).
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
for stage in ${STAGES:-train_base train_mfm_diagonal train_mfm_consistency steer_fig1}; do
  echo "=== stage ${stage} ($(date)) ==="
  bash "${REPO_ROOT}/scripts/toy/${stage}.sh"
done
echo "=== done ($(date)) ==="
