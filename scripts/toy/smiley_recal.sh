#!/usr/bin/env bash
# Stage 4 (optional): calibrate the smiley reward scale beta against the published
# Fig. 1 and re-run the two full-size smiley panels with it.  See
# src/toy/smiley_recal.py for why beta is not recoverable from the transcripts.
#   bash scripts/toy/smiley_recal.sh              (sweep + rerun + merge into results/toy)
#   SWEEP_ONLY=1 bash scripts/toy/smiley_recal.sh (sweep only, ~5 min)
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
${PY} -u -m toy.smiley_recal \
  --base_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --mfm_ckpt "${CKPT_DIR}/mfm_consistency_10000.pt" \
  --sweep_n "${SWEEP_N:-2000}" \
  --panel_n "${PANEL_N:-10000}" \
  ${SWEEP_ONLY:+--sweep_only} \
  --out_dir "${OUT_DIR:-results/toy}" \
  --device "${DEVICE}"
