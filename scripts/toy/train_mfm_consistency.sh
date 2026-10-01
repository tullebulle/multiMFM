#!/usr/bin/env bash
# Stage 2: ESD consistency (off-diagonal) from the 10k diagonal checkpoint.
# Exact command from the terminal log (cursor terminal 1.txt, lines 30-40):
#   python -u train_mfm_consistency.py --teacher_ckpt ckpt_uniform_color_square.pt \
#     --init_ckpt checkpoints/mfm_diagonal_10000.pt --ckpt checkpoints/mfm_consistency_10000.pt \
#     --steps 10000 --batch_size 1024 --lr 1e-4 --warmup 500 --gap_anneal_steps 5000 \
#     --ema_decay 0.995 --device cuda
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
${PY} -u -m toy.train_mfm_consistency \
  --teacher_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --init_ckpt "${CKPT_DIR}/mfm_diagonal_10000.pt" \
  --ckpt "${CKPT_DIR}/mfm_consistency_10000.pt" \
  --steps "${STEPS:-10000}" \
  --batch_size 1024 \
  --lr 1e-4 \
  --warmup 500 \
  --gap_anneal_steps "${GAP_ANNEAL_STEPS:-5000}" \
  --ema_decay 0.995 \
  --seed "${SEED:-0}" \
  --device "${DEVICE}"
${PY} -u -m toy.probe_mfm_diagonal \
  --ckpt "${CKPT_DIR}/mfm_consistency_10000.pt" \
  --teacher_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --device "${DEVICE}" \
  --loss_plot mfm_consistency_10000_loss.png \
  --scatter_plot mfm_consistency_10000_diag_glass_scatter.png \
  --metrics_out results/toy/probe_mfm_consistency_10000.json
