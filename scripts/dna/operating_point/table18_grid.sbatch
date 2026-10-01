#!/bin/bash
#SBATCH --job-name=dna_op_t18
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=50:00
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Table 18's (posterior steps, MC) grid at matched value-gradient NFE. One array task per
# reward at L=50. The published run pinned the dMFM posterior at 4 composed steps and swept
# MC in {1,2,4,8}; this sweeps steps in {1,2,4,8} x MC in {1,...,32}, and gives GLASS the same
# grid, on the SAME probe states and the SAME GLASS-128 RK4-50 reference (identical seeds), so
# the steps=4 rows are comparable with the published table.
#
#   sbatch scripts/dna/operating_point/table18_grid.sbatch
#   REWARD=gc LENGTH=400 sbatch --array=0 scripts/dna/operating_point/table18_grid.sbatch
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

REWARDS=(gc motif conjunction)
REWARD="${REWARD:-${REWARDS[${SLURM_ARRAY_TASK_ID:-0}]}}"
LENGTH="${LENGTH:-50}"
OUT="${OUT:-results/dna/operating_point/table18_grid/${REWARD}}"

if [ -f "${OUT}/L${LENGTH}/run_metadata.json" ]; then
  echo "done already: ${OUT}/L${LENGTH}"; exit 0
fi
${PY} -u -m dmfm.experiments.operating_point_gradient_grid \
  --length "${LENGTH}" --reward "${REWARD}" --output_dir "${OUT}" \
  --n_probes "${N_PROBES:-32}" --n_repeats "${N_REPEATS:-4}" \
  --mc_chunk "${MC_CHUNK:-16}"
