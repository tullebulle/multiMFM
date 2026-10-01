#!/bin/bash
#SBATCH --job-name=dna_regen_fig7
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Fig 7 (REGENERATED DATA, surviving producer): "degrees of tilt of the cyclizability
# distribution as the strength of tilt varies". The producer survived as
# dirichlet-flow-matching/scripts/sample_yeast_c0_guidance_hist.py (ported to
# dmfm.experiments.sample_c0_guidance, which writes exactly the published histogram.png) and
# so did its launcher (slurm/sample_yeast_c0_guidance_hist.sbatch, DNA-MFM@d40b0f6; copy in
# results/dna/original_scripts/slurm/). Only the sampled data was lost; the published panels
# used an early May-2026 base flow and the purged Keras guide, at guidance strengths that were
# never recorded. This reruns the ORIGINAL LAUNCHER'S SETTINGS (MC=128, NFE_TRAJ=128,
# NFE_VALUE=12, batch 16, guidance on t in [0.5, 0.95], cap 10, clip 10, n=1000) on the
# parent-disjoint L=50 base DFM and guide at the Fig 7/8 target C0 = 0.30, sweeping
# --guidance_frac. ESTIMATOR=table1 switches to the cheaper Table 1 estimator
# (MC=8, NFE_TRAJ=64, NFE_VALUE=4, batch 64).
#
#   sbatch scripts/dna/regen/fig7_guidance_sweep.sbatch         # 6 strengths x n=1000
#   N_SAMPLES=32 sbatch --array=0,5 --partition=gpu_test scripts/dna/regen/fig7_guidance_sweep.sbatch
#
# GUIDANCE_FRAC=0 reproduces Fig 7(a): the guidance coefficient is exactly zero, so the
# guided branch is the unguided branch and the two histograms coincide, as published.
# Each task skips itself if its summary.json exists (--requeue safe).
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

STRENGTHS=(${STRENGTHS:-0.0 0.5 1.0 2.0 4.0 8.0})
IDX="${SLURM_ARRAY_TASK_ID:-0}"
GUIDANCE_FRAC="${GUIDANCE_FRAC:-${STRENGTHS[${IDX}]}}"

DFM_CKPT="${DFM_CKPT:-checkpoints/dna/base/L50/best.pt}"
GUIDE_CKPT="${GUIDE_CKPT:-checkpoints/dna/c0/guide/best_state.pt}"
TARGET_C0="${TARGET_C0:-0.30}"
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-0}"
if [ "${ESTIMATOR:-launcher}" = "table1" ]; then
  MC="${MC:-8}"; NFE_TRAJ="${NFE_TRAJ:-64}"; NFE_VALUE="${NFE_VALUE:-4}"; BATCH_SIZE="${BATCH_SIZE:-64}"
else  # original sample_yeast_c0_guidance_hist.sbatch defaults
  MC="${MC:-128}"; NFE_TRAJ="${NFE_TRAJ:-128}"; NFE_VALUE="${NFE_VALUE:-12}"; BATCH_SIZE="${BATCH_SIZE:-16}"
fi
# The per-sample noise is seeded by sample id, so the batch size only changes throughput.
TAG="$(printf 'g%s' "${GUIDANCE_FRAC}" | tr -- '-.' 'mp')"
# FIG7_BASE lets a second submission (e.g. to gpu_requeue) write to its own tree while
# keeping one directory per guidance strength; collect.sh scans every fig7* tree.
FIG7_BASE="${FIG7_BASE:-outputs/dna/regen/fig7}"
OUT_DIR="${OUT_DIR:-${FIG7_BASE}/${TAG}_mc${MC}_n${N_SAMPLES}_seed${SEED}}"

# Record the job for results/dna/regen/PROVENANCE.md (one TSV line per array task).
mkdir -p outputs/dna/regen
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_NAME:-local}" \
  "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}" "${SLURM_ARRAY_TASK_ID:-}" "guidance_frac=${GUIDANCE_FRAC} target=${TARGET_C0} n=${N_SAMPLES} mc=${MC} nfe_traj=${NFE_TRAJ} nfe_value=${NFE_VALUE} seed=${SEED} (Fig 7)" \
  >> outputs/dna/regen/jobs.tsv

if [ -f "${OUT_DIR}/summary.json" ]; then
  echo "${OUT_DIR}/summary.json exists; skipping guidance_frac=${GUIDANCE_FRAC}"
  exit 0
fi

echo "guidance_frac=${GUIDANCE_FRAC} target=${TARGET_C0} n=${N_SAMPLES} -> ${OUT_DIR}"
${PY} -u -m dmfm.experiments.sample_c0_guidance \
  --ckpt "${DFM_CKPT}" \
  --c0_ckpt "${GUIDE_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --run_name "regen_fig7_${TAG}" \
  --seed "${SEED}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --target_c0 "${TARGET_C0}" \
  --reward_sigma 0.15 \
  --reward_scale 0.5 \
  --mc "${MC}" \
  --nfe_traj "${NFE_TRAJ}" \
  --nfe_value "${NFE_VALUE}" \
  --t_max 0.95 \
  --guide_t_start 0.50 \
  --guide_t_end 0.95 \
  --guidance_frac "${GUIDANCE_FRAC}" \
  --coeff_cap 10.0 \
  --grad_clip 10.0 \
  --hist_bins 50
