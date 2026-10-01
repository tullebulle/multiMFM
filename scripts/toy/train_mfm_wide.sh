#!/usr/bin/env bash
# A second student trained over the whole guided range.  The recovered student
# used s_max = t_cond_max = time_max = 0.95 in both stages, but the published
# Fig. 1 guides out to t = 0.98 and the posterior map is always integrated to
# u = 1.0, so the shipped student is extrapolating exactly where it is used.
# This retrains diagonal + ESD with the window opened to 0.99; everything else
# is unchanged.  ~20 min on one A100.
source "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/toy/_header.sh"
TMAX="${TMAX:-0.99}"
${PY} -u -m toy.train_mfm_diagonal \
  --teacher_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --ckpt "${CKPT_DIR}/mfm_diagonal_10000_t099.pt" \
  --steps "${STEPS:-10000}" --s_max "${TMAX}" --t_cond_max "${TMAX}" \
  --seed "${SEED:-0}" --device "${DEVICE}"
${PY} -u -m toy.train_mfm_consistency \
  --teacher_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --init_ckpt "${CKPT_DIR}/mfm_diagonal_10000_t099.pt" \
  --ckpt "${CKPT_DIR}/mfm_consistency_10000_t099.pt" \
  --steps "${STEPS:-10000}" --batch_size 1024 --lr 1e-4 --warmup 500 \
  --gap_anneal_steps "${GAP_ANNEAL_STEPS:-5000}" --ema_decay 0.995 \
  --time_max "${TMAX}" --t_cond_max "${TMAX}" \
  --seed "${SEED:-0}" --device "${DEVICE}"
${PY} -u -m toy.probe_mfm_diagonal \
  --ckpt "${CKPT_DIR}/mfm_consistency_10000_t099.pt" \
  --teacher_ckpt "${CKPT_DIR}/ckpt_uniform_color_square.pt" \
  --device "${DEVICE}" \
  --loss_plot mfm_consistency_10000_t099_loss.png \
  --scatter_plot mfm_consistency_10000_t099_diag_glass_scatter.png \
  --metrics_out results/toy/probe_mfm_consistency_10000_t099.json
