#!/usr/bin/env bash
# Same comparison on the smiley task (position + colour steering).
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
MFM_CKPT="${MFM_CKPT:-${CKPT_DIR}/mfm_consistency_10000.pt}"
TAG="${TAG:-t095}"
mkdir -p results/toy
${PY} -u -m toy.posterior_sweep --task smiley \
  --base_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" --mfm_ckpt "${MFM_CKPT}" \
  --glass_steps ${GLASS_STEPS:-8} --mfm_steps ${MFM_STEPS:-1 2 4 8} \
  --n "${N_SMILEY:-2000}" --device "${DEVICE}" \
  --out "results/toy/posterior_sweep_smiley_${TAG}.json"
