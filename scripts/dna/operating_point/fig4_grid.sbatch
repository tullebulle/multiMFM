#!/bin/bash
#SBATCH --job-name=dna_op_fig4
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=50:00
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Figure 4 (right): does any (steps, MC) split other than the plotted 1 x MC move the dMFM
# curve at matched NFE per output? Two budgets (the curve's top point MC=16 and MC=8),
# n_outputs 32, one repeat. A shard whose JSON exists is skipped, so a requeue resumes.
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

${PY} -u "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/operating_point/fig4_grid.py" \
  --nfe ${NFE:-16 8} --steps ${STEPS:-1 2 4 8 16} \
  --n-outputs "${N_OUTPUTS:-32}" --repeats "${REPEATS:-1}" \
  --out-dir "${OUT:-results/dna/operating_point/fig4_grid}" \
  --student "${STUDENT:-convention}"
