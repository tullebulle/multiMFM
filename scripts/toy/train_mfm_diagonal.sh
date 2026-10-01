#!/usr/bin/env bash
# Stage 1: diagonal GLASS distillation of the multimodal MFM.
# The original command was not recorded; the output name mfm_diagonal_10000.pt
# (terminal log: "Init: checkpoints/mfm_diagonal_10000.pt") implies --steps 10000.
# All other flags are the recovered train_mfm_diagonal.py defaults
# (batch 2048, lr 3e-4, warmup 500, EMA 0.999, s, t_cond in [1e-3, 0.95]).
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
${PY} -u -m toy.train_mfm_diagonal \
  --teacher_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --ckpt "${CKPT_DIR}/mfm_diagonal_10000.pt" \
  --steps "${STEPS:-10000}" \
  --seed "${SEED:-0}" \
  --device "${DEVICE}"
${PY} -u -m toy.probe_mfm_diagonal \
  --ckpt "${CKPT_DIR}/mfm_diagonal_10000.pt" \
  --teacher_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --device "${DEVICE}" \
  --loss_plot mfm_diagonal_10000_loss.png \
  --scatter_plot mfm_diagonal_10000_glass_scatter.png \
  --metrics_out results/toy/probe_mfm_diagonal_10000.json
