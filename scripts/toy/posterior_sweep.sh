#!/usr/bin/env bash
# Why is the multiMFM Fig. 1 panel weaker than the GLASS one?  Runs the published
# panel's guidance configuration with different posterior samplers (GLASS with k
# Euler steps vs the MFM with k composed flow-map steps) and reports per-region
# colour accuracy.  See src/toy/posterior_sweep.py.
#   bash scripts/toy/posterior_sweep.sh
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
MFM_CKPT="${MFM_CKPT:-${CKPT_DIR}/mfm_consistency_10000.pt}"
TAG="${TAG:-t095}"
mkdir -p results/toy
${PY} -u -m toy.posterior_sweep --task norway \
  --base_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" --mfm_ckpt "${MFM_CKPT}" \
  --glass_steps ${GLASS_STEPS:-8 32} --mfm_steps ${MFM_STEPS:-1 2 4 8} \
  --guidance_ends ${GUIDANCE_ENDS:-0.95 0.98} \
  ${RATIO:+--ratio ${RATIO}} \
  --n "${N_NORWAY:-4000}" --device "${DEVICE}" \
  --out "results/toy/posterior_sweep_norway_${TAG}.json"
