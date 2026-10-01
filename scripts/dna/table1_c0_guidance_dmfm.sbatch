#!/bin/bash
#SBATCH --job-name=dna_table1_dmfm
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Table 1 / Fig 3 with the dMFM as the posterior sampler (authors' decision 2026-09-29: Table 1
# reports this run; the GLASS numbers of scripts/dna/table1_c0_guidance.sbatch are provenance only).
#
# Identical protocol to the GLASS Table 1 run -- same reward (sigma 0.15, scale 0.5),
# guidance_frac 8, cap 10, clip 10, t-window [0.5, 0.95], 64 trajectory steps to t=0.95,
# MC=8 frozen per-sample pool (seed+1e6+i), 4 posterior steps, per-sample initial noise
# (seed+i) shared by each guided/unguided pair, batches of 8, seed 0, n=1000 per target --
# with the 4 GLASS Euler steps on the base DFM replaced by 4 composed flow-map steps of the
# L=50 dMFM. See dmfm.experiments.sample_c0_guidance_dmfm.
#
# WHICH STUDENT (changed 2026-09-29, docs/dmfm_glass_parity.md). --nfe_value 4 composes jumps of
# 0.25, so the student must be the 4-step ESD one, checkpoints/dna/dmfm_4step/L50 (ESD on gaps in
# [0, 0.25], selected on val_mfm_probe_esd_gap_0.25 -- the artifact Table 18 uses at 4 steps).
# The first n=1000 dMFM run (job 49277977) instead composed 4 steps of the *diagonal* student
# checkpoints/dna/dmfm/L50 (ESD on random gaps, used as a ONE-step map by Tables 13/17) and guided
# ~3x worse than GLASS. DMFM_CKPT now defaults to the step-matched artifact and OUT_ROOT to a
# separate tree, so the legacy shards in outputs/dna/table1_dmfm_n1000 are not mixed in. To
# reproduce job 49277977 exactly:
#   LEGACY_DIAG_STUDENT=1 sbatch scripts/dna/table1_c0_guidance_dmfm.sbatch
#
# Array: task = target * N_SHARDS + shard; 2 targets x 5 shards of 200 pairs (shard bounds are
# multiples of the batch size, so batches equal those of an unsharded run). A shard whose
# sample_scores.csv exists is skipped, so resubmitting reruns only missing/preempted shards.
# Then collect (CPU, login node; merges shards, scores with the oracle, writes results/):
#   bash scripts/dna/collect_table1_dmfm.sh
#
#   sbatch scripts/dna/table1_c0_guidance_dmfm.sbatch                            # 4-step ESD student
#   RUN_ROOT=outputs/dna/table1_dmfm4_n1000 bash scripts/dna/collect_table1_dmfm.sh <jobid>
#   POSTERIOR=glass sbatch ...                                    # matched GLASS arm (same scale rule)
#   LEGACY_DIAG_STUDENT=1 sbatch ...                              # reproduce withdrawn job 49277977 exactly
#   GRAD_NORMALIZE=0 sbatch ...                                   # superseded clip-down rule, on purpose
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

DFM_CKPT="${DFM_CKPT:-checkpoints/dna/base/L50/best.pt}"
NFE_VALUE="${NFE_VALUE:-4}"
# MC is part of the method, not just the budget: the value gradient costs
# NFE_VALUE x MC network calls, so the same budget can be split many ways and the
# split must be stated (plan S7 item 8, convention 3). Pinning both arms to 4x8 --
# GLASS's estimator -- is what produced the apparent dMFM deficit; at matched NFE 32
# a one-step flow map affords MC 32 and ties GLASS. Defaults reproduce the 4x8 runs.
MC="${MC:-8}"
MC_CHUNK="${MC_CHUNK:-${MC}}"
N_SAMPLES="${N_SAMPLES:-1000}"
POSTERIOR="${POSTERIOR:-dmfm}"
# GRAD_NORMALIZE=1 (the DEFAULT, and mandatory for every dMFM demo -- REPRODUCIBILITY_PLAN.md
# section 7 item 8) rescales the value gradient to exactly --grad_clip instead of only clipping it
# down. That makes the guidance step scale-free, which is the only way to compare posterior samplers
# whose gradient norms differ by 5-12x (docs/dmfm_glass_parity.md). Apply it to BOTH arms or not at
# all. GRAD_NORMALIZE=0 is the superseded clip-down rule and must be requested explicitly; it is the
# default only under LEGACY_DIAG_STUDENT=1, which reproduces the withdrawn job 49277977 in full.
if [[ "${LEGACY_DIAG_STUDENT:-0}" == "1" ]]; then GRAD_NORMALIZE="${GRAD_NORMALIZE:-0}"; else GRAD_NORMALIZE="${GRAD_NORMALIZE:-1}"; fi
case "${GRAD_NORMALIZE}" in 0|1) ;; *) echo "GRAD_NORMALIZE must be 0 or 1, got '${GRAD_NORMALIZE}'" >&2; exit 2 ;; esac
if [[ "${GRAD_NORMALIZE}" == "0" && "${LEGACY_DIAG_STUDENT:-0}" != "1" ]]; then
  echo "WARNING: GRAD_NORMALIZE=0 selects the superseded clip-down guidance rule, which plan section 7" >&2
  echo "         item 8 forbids for dMFM demos; the run will look plausible but is not comparable." >&2
fi
if [[ "${GRAD_NORMALIZE}" == "1" ]]; then NORM_FLAG="--grad_normalize"; NORM_TAG="_norm"; else NORM_FLAG=""; NORM_TAG=""; fi
if [[ "${LEGACY_DIAG_STUDENT:-0}" == "1" ]]; then
  # Job 49277977: the diagonal student composed in --nfe_value steps it was not distilled for.
  DMFM_CKPT="${DMFM_CKPT:-$(${PY} -c "from dmfm import paths; print(paths.rel(paths.dmfm_ckpt(50)))")}"
  GAP_FLAG="--allow_gap_mismatch"
  ARM_TAG="dmfm"
else
  DMFM_CKPT="${DMFM_CKPT:-$(${PY} -c "from dmfm import paths; print(paths.rel(paths.dmfm_ckpt_for_steps(50, ${NFE_VALUE})))")}"
  GAP_FLAG=""
  ARM_TAG="dmfm4"
fi
# The GLASS arm loads no student; keep each (posterior, scale rule) in its own tree.
[[ "${POSTERIOR}" == "dmfm" ]] || ARM_TAG="${POSTERIOR}"
CFG_TAG=""
if [ "${NFE_VALUE}" != "4" ] || [ "${MC}" != "8" ]; then CFG_TAG="_s${NFE_VALUE}mc${MC}"; fi
DEFAULT_OUT_ROOT="outputs/dna/table1_${ARM_TAG}${NORM_TAG}${CFG_TAG}_n${N_SAMPLES}"
GUIDE_CKPT="${GUIDE_CKPT:-checkpoints/dna/c0/guide/best_state.pt}"
N_SHARDS="${N_SHARDS:-5}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
TARGETS=(1.0 -1.0)

TASK="${SLURM_ARRAY_TASK_ID:-0}"
TARGET_C0="${TARGET_C0:-${TARGETS[$(( TASK / N_SHARDS ))]}}"
SHARD="${SHARD:-$(( TASK % N_SHARDS ))}"
SHARD_SIZE=$(( (N_SAMPLES / N_SHARDS + 7) / 8 * 8 ))
START=$(( SHARD * SHARD_SIZE ))
STOP=$(( START + SHARD_SIZE < N_SAMPLES ? START + SHARD_SIZE : N_SAMPLES ))
TARGET_TAG="$(printf '%s' "${TARGET_C0}" | tr -- '-.' 'mp')"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/target_${TARGET_TAG}/shards/$(printf '%04d_%04d' "${START}" "${STOP}")}"
echo "target=${TARGET_C0} shard=${SHARD} ids=[${START}, ${STOP}) posterior=${POSTERIOR} steps=${NFE_VALUE} mc=${MC} student=${DMFM_CKPT} grad_normalize=${GRAD_NORMALIZE} -> ${OUT_DIR}"

if [[ -f "${OUT_DIR}/sample_scores.csv" ]]; then
  echo "shard already done: ${OUT_DIR}"; exit 0
fi

${PY} -u -m dmfm.experiments.sample_c0_guidance_dmfm \
  --paired --posterior "${POSTERIOR}" ${GAP_FLAG} ${NORM_FLAG} \
  --sample_start "${START}" --sample_stop "${STOP}" \
  --teacher_ckpt "${DFM_CKPT}" \
  --student_ckpt "${DMFM_CKPT}" \
  --c0_ckpt "${GUIDE_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size 8 \
  --target_c0 "${TARGET_C0}" \
  --reward_sigma 0.15 \
  --reward_scale 0.5 \
  --mc "${MC}" \
  --mc_chunk "${MC_CHUNK}" \
  --nfe_sample 64 \
  --nfe_value "${NFE_VALUE}" \
  --t_max 0.95 \
  --guide_t_start 0.50 \
  --guide_t_end 0.95 \
  --guidance_frac 8.0 \
  --coeff_cap 10.0 \
  --grad_clip 10.0
