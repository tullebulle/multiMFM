#!/bin/bash
#SBATCH --job-name=nfe_audit_timing
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nfe_audit_timing-%j.out
#SBATCH --error=logs/nfe_audit_timing-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python results/shared/nfe_audit/measure_backward_cost.py --part both
