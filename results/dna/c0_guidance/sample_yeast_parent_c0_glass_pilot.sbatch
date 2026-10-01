#!/bin/bash
#SBATCH --job-name=yeast_parent_glass
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
DFM_CKPT="${DFM_CKPT:-workdir/yeast_parent_a_L50_dfm_dit_h192_b4/best.pt}"
GUIDE_CKPT="${GUIDE_CKPT:-workdir/yeast_parent_c0_guide_2026-07-26/best_state.pt}"
ORACLE_CKPT="${ORACLE_CKPT:-workdir/yeast_parent_c0_oracle_2026-07-26/best_state.pt}"
TARGET_C0="${TARGET_C0:-1.0}"
GUIDANCE_FRAC="${GUIDANCE_FRAC:-8.0}"
N_SAMPLES="${N_SAMPLES:-32}"
SEED="${SEED:-0}"
TARGET_TAG="$(printf '%s' "${TARGET_C0}" | tr -- '-.' 'mp')"
RUN_TAG="${RUN_TAG:-pilot_target${TARGET_TAG}_g${GUIDANCE_FRAC}_seed${SEED}}"
OUT_DIR="${OUT_DIR:-workdir/yeast_parent_glass_${RUN_TAG}}"
ORACLE_OUT="${ORACLE_OUT:-${OUT_DIR}/oracle}"

"${PYTHON}" -u scripts/sample_yeast_c0_guidance_hist.py \
  --ckpt "${DFM_CKPT}" \
  --c0_ckpt "${GUIDE_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size 8 \
  --target_c0 "${TARGET_C0}" \
  --reward_sigma 0.15 \
  --reward_scale 0.5 \
  --mc 8 \
  --mc_chunk 8 \
  --nfe_traj 64 \
  --nfe_value 4 \
  --t_max 0.95 \
  --guide_t_start 0.50 \
  --guide_t_end 0.95 \
  --guidance_frac "${GUIDANCE_FRAC}" \
  --coeff_cap 10.0 \
  --grad_clip 10.0

"${PYTHON}" -u scripts/score_yeast_c0_oracle.py \
  --oracle_ckpt "${ORACLE_CKPT}" \
  --oracle_model_type park_cnn \
  --run_glob "${OUT_DIR}/sample_scores.csv" \
  --out_dir "${ORACLE_OUT}" \
  --batch_size 512 \
  --bootstrap 2000 \
  --seed "${SEED}" \
  --reverse_complement_average \
  --device cuda:0
