#!/bin/bash
#SBATCH --job-name=dna_table13
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Table 13: exact motif / conjunction reward attainment, one (length, reward) cell.
# Guidance posterior = one direct dMFM map F(0,1,eps | t, x_t) of the DIAGONAL
# dMFM students, MC=8; trajectory + 1024-sample target calibration from the base
# DFM. Original: slurm/sample_parent_reward_attainment.sbatch, jobs
# 35383601/624/648/653 (motif L50..400) and 35383663/669/675/680 (conjunction),
# kempner_h100, 2 h.
#
# --dmfm_ckpt is passed explicitly: the shared DEFAULT_DMFMS map was later changed
# to the 4-step students, which is NOT what Table 13 used.
#
#   LENGTH=50 REWARD=motif sbatch scripts/dna/table13_reward_attainment.sbatch
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

L="${LENGTH:?Set LENGTH to one of 50, 100, 200, 400}"
REWARD="${REWARD:?Set REWARD to motif or conjunction}"
OUT_ROOT="${OUT_ROOT:-outputs/dna/reward_attainment_${REWARD}}"
SEED="${SEED:-20260731}"
DMFM_CKPT="${DMFM_CKPT:-$(${PY} -c "from dmfm import paths; print(paths.dmfm_ckpt(${L}))")}"

${PY} -u -m dmfm.experiments.sample_reward_attainment \
  --lengths "${L}" \
  --reward "${REWARD}" \
  --dmfm_ckpt "${L}=${DMFM_CKPT}" \
  --output_dir "${OUT_ROOT}" \
  --seed "${SEED}" \
  --n_samples 100 \
  --batch_size 8 \
  --nfe_traj 64 \
  --t_end 0.999 \
  --guide_t_start 0.50 \
  --guide_t_end 0.95 \
  --mc 8 \
  --mc_chunk 4 \
  --guidance_frac 8.0 \
  --grad_clip 10.0 \
  --coeff_cap 10.0 \
  --calibration_samples 1024 \
  --nfe_calibration 64 \
  --calibration_batch_size 32 \
  --target_percentile 0.90 \
  --reward_beta 1.0 \
  --reward_scale 1.0 \
  --motif TTTTTC \
  --motif2 AAAATT \
  --motif_tau 0.10 \
  --conjunction_tau 0.10 \
  --bootstrap 2000 \
  --device cuda:0
