#!/bin/bash
#SBATCH --job-name=dna_fig7_dmfm
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=40:00
#SBATCH --array=0-13%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Fig 7 rerun with the dMFM posterior (REPRODUCIBILITY_PLAN.md S7 item 8: the DNA demos are
# presented as dMFM but were run with GLASS on the base DFM, and the authors chose to rerun
# them with dMFM rather than relabel the paper).
#
# WHAT THE PUBLISHED FIGURE RAN. `sample_yeast_c0_guidance_hist.py` (ported to
# `dmfm.experiments.sample_c0_guidance`) loads only the base DFM and the C0 guide and has no
# dMFM code path: the value gradient is Euler steps of the GLASS posterior velocity. The
# sampled data was lost and the guidance strengths were never recorded; the surviving launcher
# defaults are MC=128, nfe_traj=128, nfe_value=12, batch 16, t in [0.5, 0.95], n=1000,
# target C0 = 0.30. Its faithful regeneration (scripts/dna/regen/fig7_guidance_sweep.sbatch,
# Slurm 49298146, cancelled 2026-09-29) is a GLASS experiment and is superseded by this one.
#
# WHAT THIS RUNS. `dmfm.experiments.sample_c0_guidance_dmfm --paired`, whose GLASS mode is
# numerically identical to the published sampler (tests/dna/test_api.py::
# test_paired_glass_equals_table1_sampler), at the Table 1 / Fig 3 operating point:
#   * posterior dmfm, nfe_value 4 composed flow-map steps, student resolved by
#     paths.dmfm_ckpt_for_steps(50, 4) = checkpoints/dna/dmfm_4step/L50 (convention 1);
#   * --grad_normalize, i.e. the value gradient rescaled to exactly --grad_clip, on BOTH arms
#     (convention 2);
#   * MC = 8, nfe_sample 64, t_max 0.95, guidance window [0.5, 0.95], coeff_cap 10,
#     grad_clip 10, reward_sigma 0.15, reward_scale 0.5, seed 0, n = 1000, target C0 = 0.30.
# The published launcher's MC=128/nfe_value=12 estimator is NOT used: the dMFM conventions fix
# the posterior at 4 composed steps, and MC=8 is the paper's DNA operating point (Table 1),
# which is also what makes the ladder affordable (see COST below).
#
# STRENGTH LADDER, chosen before any output was looked at: guidance_frac in
# {0, 1, 2, 4, 8, 16, 32}. Rationale: 0 reproduces Fig 7(a) exactly (the coefficient is zero,
# so the guided branch *is* the unguided branch); the rest is a geometric ladder in powers of
# two that brackets both arms' optima under the scale-free rule as measured in
# docs/dmfm_glass_parity.md section 6 (GLASS 4-8, dMFM 16), with one point above dMFM's. The
# whole ladder is reported. For the three-panel figure the panels are, fixed in advance:
# gf = 0 (none), gf = 8 (moderate; the published Table 1 strength), gf = 32 (strong).
#
# BOTH ARMS. Tasks 0-6 are the dMFM arm, tasks 7-13 the GLASS arm at the SAME estimator
# budget (MC=8, nfe_value=4) and the same scale-free rule, so the comparison changes only the
# posterior sampler. The GLASS arm here is therefore not the published Fig 7 estimator either;
# it is the matched control that convention 2 requires.
#
# COST. Fig 7's GLASS "faithful estimator" estimate was ~46 GPU-h for 6 strengths at n=1000
# (MC=128 x nfe_value=12 = 1536 posterior model evaluations per guided step per sample).
# At MC=8 x nfe_value=4 = 32 the value gradient is 48x cheaper, so the recomputed estimate is
# ~1 GPU-h for 7 strengths x n=1000 per arm, ~2 GPU-h for both arms. Measured: see
# results/dna/dmfm_demos/PROVENANCE.md.
#
#   sbatch scripts/dna/dmfm_demos/fig7_tilt_ladder.sbatch              # both arms, 14 tasks
#   sbatch --array=0-6 scripts/dna/dmfm_demos/fig7_tilt_ladder.sbatch  # dMFM arm only
#   N_SAMPLES=32 sbatch --array=0,6 --time=20:00 scripts/dna/dmfm_demos/fig7_tilt_ladder.sbatch
#
# Each task skips itself if its summary.json exists, so a requeued task resumes the ladder.
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

STRENGTHS=(${STRENGTHS:-0 1 2 4 8 16 32})
ARMS=(${ARMS:-dmfm glass})
N_STRENGTH="${#STRENGTHS[@]}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARM="${ARM:-${ARMS[$(( IDX / N_STRENGTH ))]}}"
GUIDANCE_FRAC="${GUIDANCE_FRAC:-${STRENGTHS[$(( IDX % N_STRENGTH ))]}}"

TARGET_C0="${TARGET_C0:-0.30}"
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-0}"
MC="${MC:-8}"
NFE_VALUE="${NFE_VALUE:-4}"
NFE_SAMPLE="${NFE_SAMPLE:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"

TAG="$(printf 'gf%s' "${GUIDANCE_FRAC}" | tr -- '-.' 'mp')"
OUT_ROOT="${OUT_ROOT:-outputs/dna/dmfm_demos/fig7}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${ARM}/${TAG}_mc${MC}_n${N_SAMPLES}_seed${SEED}}"

mkdir -p "${OUT_ROOT}"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_NAME:-local}" \
  "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}" "${SLURM_ARRAY_TASK_ID:-}" \
  "Fig7 arm=${ARM} guidance_frac=${GUIDANCE_FRAC} target=${TARGET_C0} n=${N_SAMPLES} mc=${MC} nfe_value=${NFE_VALUE} nfe_sample=${NFE_SAMPLE} seed=${SEED} grad_normalize=1" \
  >> "${OUT_ROOT}/jobs.tsv"

if [ -f "${OUT_DIR}/summary.json" ]; then
  echo "${OUT_DIR}/summary.json exists; skipping arm=${ARM} guidance_frac=${GUIDANCE_FRAC}"
  exit 0
fi

echo "arm=${ARM} guidance_frac=${GUIDANCE_FRAC} target=${TARGET_C0} n=${N_SAMPLES} -> ${OUT_DIR}"
${PY} -u -m dmfm.experiments.sample_c0_guidance_dmfm \
  --paired \
  --posterior "${ARM}" \
  --grad_normalize \
  --out_dir "${OUT_DIR}" \
  --run_name "fig7_dmfm_${ARM}_${TAG}" \
  --seed "${SEED}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --target_c0 "${TARGET_C0}" \
  --reward_sigma 0.15 \
  --reward_scale 0.5 \
  --mc "${MC}" \
  --mc_chunk "${MC}" \
  --nfe_sample "${NFE_SAMPLE}" \
  --nfe_value "${NFE_VALUE}" \
  --t_max 0.95 \
  --guide_t_start 0.50 \
  --guide_t_end 0.95 \
  --guidance_frac "${GUIDANCE_FRAC}" \
  --coeff_cap 10.0 \
  --grad_clip "${GRAD_CLIP}" \
  --hist_bins 50
