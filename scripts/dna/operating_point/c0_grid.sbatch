#!/bin/bash
#SBATCH --job-name=dna_op_c0
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=40:00
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# The (steps, MC) operating-point grid at matched value-gradient NFE for a C0-guided demo.
# Both arms get the same grid and the same scale-free rule, so neither is pinned to the
# other's estimator split (REPRODUCIBILITY_PLAN.md S7 item 8, convention 3).
#
# NFE=32 grid: 1x32, 2x16, 4x8, 8x4 on each arm. dMFM configs follow convention 1
# (diagonal student at one composed step, 4-step ESD student above that).
#
#   TARGETS=0.30 OUT=results/dna/operating_point/fig7_grid_n64 \
#     sbatch scripts/dna/operating_point/c0_grid.sbatch
#   NFE=8 ... for the Table 13 budget level
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

TARGETS="${TARGETS:-0.30}"
N_SAMPLES="${N_SAMPLES:-64}"
GF="${GF:-8.0}"
NFE="${NFE:-32}"
OUT="${OUT:-results/dna/operating_point/c0_grid_nfe${NFE}_n${N_SAMPLES}}"

# (steps, mc) pairs at the requested NFE; dMFM student per convention 1.
CONFIGS=""
for STEPS in 1 2 4 8; do
  MC=$(( NFE / STEPS ))
  [ "${MC}" -ge 1 ] || continue
  if [ "${STEPS}" = "1" ]; then DM="diag_fm1"; else DM="dmfm4_fm${STEPS}"; fi
  CONFIGS="${CONFIGS} ${DM}@norm@gf=${GF}@mc=${MC} glass_euler${STEPS}@norm@gf=${GF}@mc=${MC}"
done
echo "configs:${CONFIGS}"

mkdir -p "${OUT}"
${PY} -u -m dmfm.experiments.operating_point_c0 \
  --configs ${CONFIGS} \
  --targets ${TARGETS} \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --mc_chunk "${MC_CHUNK:-8}" \
  --guidance_frac "${GF}" \
  --out_dir "${OUT}/target_$(printf '%s' "${TARGETS}" | tr -- ' -.' '_mp')" \
  --seed "${SEED:-0}"
