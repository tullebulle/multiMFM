#!/bin/bash
#SBATCH --job-name=dna_op_t13
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=50:00
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Table 13's (posterior steps, MC) grid at fixed value-gradient NFE, L=50, motif reward,
# n=100 (the published n). Task 0 = NFE 8 (the published budget, whose 1x8 @clip row
# reproduces the published configuration); task 1 = NFE 32 (the budget at which the C0
# operating point was chosen). Both rules are run: the published clip-down rule and the
# scale-free rule of convention 2.
#
#   sbatch scripts/dna/operating_point/table13_grid.sbatch
#   REWARD=conjunction sbatch --array=0 scripts/dna/operating_point/table13_grid.sbatch
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

NFES=(8 32)
NFE="${NFE:-${NFES[${SLURM_ARRAY_TASK_ID:-0}]}}"
REWARD="${REWARD:-motif}"
LENGTH="${LENGTH:-50}"
OUT="${OUT:-results/dna/operating_point/table13_grid/${REWARD}_nfe${NFE}}"
if [ -f "${OUT}/L${LENGTH}/run_metadata.json" ]; then echo "done: ${OUT}"; exit 0; fi

${PY} -u -m dmfm.experiments.operating_point_reward_grid \
  --length "${LENGTH}" --reward "${REWARD}" --out_dir "${OUT}" \
  --nfe "${NFE}" --n_samples "${N_SAMPLES:-100}" --mc_chunk "${MC_CHUNK:-8}"
