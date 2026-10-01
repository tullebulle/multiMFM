#!/usr/bin/env bash
# Where is the distilled toy MFM accurate?  Diagonal-velocity error on an explicit
# (s, t_cond) grid, including outside the training window, plus the conditional
# posterior error of the composed map against a 128-step GLASS reference started
# from the same noise.  See src/toy/probe_windows.py.
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
MFM_CKPT="${MFM_CKPT:-${CKPT_DIR}/mfm_consistency_10000.pt}"
TAG="${TAG:-t095}"
mkdir -p results/toy
${PY} -u -m toy.probe_windows \
  --ckpt "${MFM_CKPT}" --base_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --n "${N_PROBE:-8192}" --n_endpoint "${N_COND:-256}" --n_post "${N_POST:-128}" \
  --device "${DEVICE}" --out "results/toy/probe_windows_${TAG}.json"
