#!/bin/bash
#SBATCH --job-name=dna_op_f7n
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=40:00
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Fig 7's `guidance_frac 8` point at each arm's own operating point, at the published n = 1000:
# task 0 = dMFM 1 composed step x MC 32 (the diagonal student, which convention 1 selects at one
# step, resolved by sample_c0_guidance_dmfm itself); task 1 = GLASS 2 Euler steps x MC 16. Both
# NFE 32, both under the scale-free rule, everything else identical to
# scripts/dna/dmfm_demos/fig7_tilt_ladder.sbatch at gf = 8 (that script is imported by reference,
# not edited; OUT_DIR is redirected into results/dna/operating_point/).
#
# Submit this only once a grid has shown a real move at small n -- it is the full-n rerun, not
# the measurement that chooses anything.
#
#   sbatch scripts/dna/operating_point/fig7_at_operating_point.sbatch
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

ARMS=(dmfm glass)
STEPSS=(1 2)
MCS=(32 16)
I="${SLURM_ARRAY_TASK_ID:-0}"
ARM="${ARM:-${ARMS[$I]}}"
NFE_VALUE="${NFE_VALUE:-${STEPSS[$I]}}"
MC="${MC:-${MCS[$I]}}"
N_SAMPLES="${N_SAMPLES:-1000}"
OUT_ROOT="${OUT_ROOT:-results/dna/operating_point/fig7_operating_point_n${N_SAMPLES}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${ARM}_s${NFE_VALUE}mc${MC}}"
if [ -f "${OUT_DIR}/summary.json" ]; then echo "exists: ${OUT_DIR}"; exit 0; fi

echo "arm=${ARM} steps=${NFE_VALUE} mc=${MC} n=${N_SAMPLES} -> ${OUT_DIR}"
${PY} -u -m dmfm.experiments.sample_c0_guidance_dmfm \
  --paired --posterior "${ARM}" --grad_normalize \
  --out_dir "${OUT_DIR}" --run_name "fig7_op_${ARM}_s${NFE_VALUE}mc${MC}" \
  --seed "${SEED:-0}" --n_samples "${N_SAMPLES}" --batch_size 8 \
  --target_c0 "${TARGET_C0:-0.30}" --reward_sigma 0.15 --reward_scale 0.5 \
  --mc "${MC}" --mc_chunk "${MC_CHUNK:-8}" \
  --nfe_sample 64 --nfe_value "${NFE_VALUE}" --t_max 0.95 \
  --guide_t_start 0.50 --guide_t_end 0.95 \
  --guidance_frac "${GF:-8.0}" --coeff_cap 10.0 --grad_clip 10.0 --hist_bins 50
