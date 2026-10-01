#!/usr/bin/env bash
# Stage 3: the Fig. 1 panels and the numbers behind them (results/toy/).
# Runs both guidance backends: plain GLASS posterior sampling (what the
# published 2026-05-06 figure used, see docs/provenance/toy/README.md) and the
# multiMFM posterior sampler (the June rerun).  Settings live in toy.guidance.
#   bash scripts/toy/steer_fig1.sh                  (full size, ~10 min on one GPU)
#   N_PANEL=1000 QUICK=1 bash scripts/toy/steer_fig1.sh   (smoke)
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
mkdir -p results/toy
${PY} -u -m toy.fig1 \
  --base_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --mfm_ckpt "${CKPT_DIR}/mfm_consistency_10000.pt" \
  --backends ${BACKENDS:-glass mfm} \
  --n_panel "${N_PANEL:-10000}" \
  ${QUICK:+--quick} \
  --out_dir "${OUT_DIR:-results/toy}" \
  --device "${DEVICE}"

# Table 12: what the rebuilt training actually used, read back off the checkpoints.
${PY} -u -m toy.table12 --out results/toy/table12_actual.json
