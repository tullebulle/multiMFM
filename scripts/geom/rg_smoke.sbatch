#!/usr/bin/env bash
#SBATCH --job-name=geom_rg_smoke
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Smallest run that exercises all four methods on the R_g target, to catch wiring
# errors before any of them is run at a size worth reporting.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
R="scripts/geom/rg_benchmark.py"
N="${N:-16}"
python -u "$R" --method base       --n "$N" --batch 16 --tag smoke_base      || echo "FAILED base"
python -u "$R" --method best_of_n  --n "$N" --batch 8 --n-candidates 4 --tag smoke_bon || echo "FAILED bon"
python -u "$R" --method dps        --n "$N" --batch 16 --mu 0.5 --tag smoke_dps || echo "FAILED dps"
python -u "$R" --method multimfm   --n "$N" --batch 8 --value-samples 4 --mu 0.5 --tag smoke_mmfm || echo "FAILED mmfm"
echo done
