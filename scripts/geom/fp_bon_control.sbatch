#!/usr/bin/env bash
#SBATCH --job-name=geom_bon_ctl
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Control: best-of-1 must reproduce the steering runs' unguided arm.
#
# The sweep says best-of-N beats both guided methods at matched cost. That conclusion is only
# worth anything if the two pipelines are drawing the same molecules against the same targets.
# best-of-1 performs no selection, so it IS the unguided generator: at the same seed and the
# same count it must land on the same Tanimoto as the unguided arm of the guided runs
# (~0.2275 at these seeds). If it comes back materially higher, the two pipelines differ --
# easier targets, a different molecule-size draw, a different scorer convention -- and the
# comparison is invalid rather than negative.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

SEEDS=(301 302 303); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
python -u scripts/geom/fp_best_of_n.py --num-mols 256 --n-candidates 1 --batch 32 \
  --seed "${SEED}" --pb --device cuda \
  --out-dir results/geom/runs/fp_bon_control --tag "bon_k1_seed${SEED}"
echo "done seed ${SEED}"
