#!/usr/bin/env bash
# Stage 0: base multimodal flow model on U(colours) x U([-1,1]^2)  (Table 12).
# The original command was not recorded; flags = Table 12 values (also the
# defaults of toy.train).  Original output name: ckpt_uniform_color_square.pt.
#   bash scripts/toy/train_base.sh            (STEPS=500 for a smoke test)
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
${PY} -u -m toy.train \
  --dataset uniform_color_square \
  --ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --steps "${STEPS:-20000}" \
  --batch_size 4096 \
  --lr 3e-4 \
  --warmup 500 \
  --ema_decay 0.999 \
  --d_model 128 --n_heads 8 --n_layers 6 --mlp_ratio 4 \
  --seed "${SEED:-0}" \
  --device "${DEVICE}"
