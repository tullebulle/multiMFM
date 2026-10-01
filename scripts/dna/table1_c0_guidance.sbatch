#!/bin/bash
#SBATCH --job-name=dna_table1_c0
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Table 1 / Fig 3: paired unguided-vs-guided C0 sampling at L=50 with GLASS
# guidance (MC=8, 4-step value rollout) on the BASE DFM, then independent-oracle
# scoring. Note: despite the paper text, no dMFM checkpoint is involved here
# (see docs/provenance/dna/original_runs.md section 5 and section 13).
#
# Original: slurm/sample_yeast_parent_c0_glass_pilot.sbatch (n=32 pilot, jobs
# 35242278/9, which is what Table 1 prints) and the identical n=1000 rerun
# (scratch_dfm_recon_20260925/run_n1000.sbatch, jobs 48651189/90). Flags are the
# same in both; only --n_samples differs. N_SAMPLES defaults to 1000 here.
#
#   TARGET_C0=1.0  sbatch scripts/dna/table1_c0_guidance.sbatch
#   TARGET_C0=-1.0 sbatch scripts/dna/table1_c0_guidance.sbatch
#   N_SAMPLES=32 TARGET_C0=1.0 sbatch scripts/dna/table1_c0_guidance.sbatch   # paper pilot
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

DFM_CKPT="${DFM_CKPT:-checkpoints/dna/base/L50/best.pt}"
GUIDE_CKPT="${GUIDE_CKPT:-checkpoints/dna/c0/guide/best_state.pt}"
ORACLE_CKPT="${ORACLE_CKPT:-checkpoints/dna/c0/oracle/best_state.pt}"
TARGET_C0="${TARGET_C0:-1.0}"
GUIDANCE_FRAC="${GUIDANCE_FRAC:-8.0}"
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-0}"
# batch_size 8 is part of the RNG stream: changing it changes the samples.
BATCH_SIZE="${BATCH_SIZE:-8}"
TARGET_TAG="$(printf '%s' "${TARGET_C0}" | tr -- '-.' 'mp')"
RUN_TAG="${RUN_TAG:-n${N_SAMPLES}_target${TARGET_TAG}_g${GUIDANCE_FRAC}_seed${SEED}}"
OUT_DIR="${OUT_DIR:-outputs/dna/yeast_parent_glass_${RUN_TAG}}"
ORACLE_OUT="${ORACLE_OUT:-${OUT_DIR}/oracle}"

${PY} -u -m dmfm.experiments.sample_c0_guidance \
  --ckpt "${DFM_CKPT}" \
  --c0_ckpt "${GUIDE_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
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

${PY} -u -m dmfm.experiments.score_c0_oracle \
  --oracle_ckpt "${ORACLE_CKPT}" \
  --oracle_model_type park_cnn \
  --run_glob "${OUT_DIR}/sample_scores.csv" \
  --out_dir "${ORACLE_OUT}" \
  --batch_size 512 \
  --bootstrap 2000 \
  --seed "${SEED}" \
  --reverse_complement_average \
  --device cuda:0
