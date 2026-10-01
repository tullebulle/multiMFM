#!/usr/bin/env bash
#SBATCH --job-name=geom_table3_asdesc
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_requeue
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%6
#SBATCH --constraint="cc8.0|cc9.0"
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# NOTE: torch 2.5.1+cu124 has no sm_120 kernels, so the Blackwell (rtx6000pro) nodes in
# kempner_requeue fail with "no kernel image is available for execution on the device".
# The constraint above restricts the job to compute capability 8.0/9.0 (A100/H100/H200).
#
# Table 3 (GEOM-Drugs structural fingerprint guidance) rerun EXACTLY AS THE PAPER
# DESCRIBES IT -- authors' decision 1 of 2026-09-29.
#
# What the paper states (Sec. 3.2, App. F.8/F.9) and this run does:
#   * 4,096 molecules per seed (App. F.9), here 4 shards x 1,024;
#   * 10 flow-map integration steps, 4 posterior samples, 100-step outer ODE (App. F.8);
#   * reward R(G)=exp(-lambda*||E(G)-c||^2) with E the fingerprint head, targets drawn
#     from the GEOM-Drugs val split, molecules up to 71 heavy atoms;
#   * Tanimoto of the DIRECTLY COUNTED FP2 fingerprint of the generated molecule against
#     the target (no oracle), plus PoseBusters validity;
#   * the unguided arm runs on the same noise and the same targets, giving the base
#     generator's Tanimoto and PB validity on the same 4,096 molecules (Table 21's
#     "Unsteered" column, for which no record exists).
# TFG-Flow reports +- values, so 3 seeds are run (SEED_GROUPS): different target draws
# and different noise. lambda/mu are not stated in the paper; they are selected on a
# held-out target pool (scripts/geom/tune_lambda_mu.sbatch) that this run excludes.
#
# One array task per (seed group, shard): 3 x 4 = 12 tasks. Each writes its own JSON and
# exits immediately if that file exists, so a preempted task reruns alone.
#
#   LAMBDA=<selected> MU=<selected> sbatch scripts/geom/table3_asdescribed.sbatch
#   NUM_MOLS=64 sbatch --array=0-1 --partition=gpu_test --time=00:40:00 \
#       scripts/geom/table3_asdescribed.sbatch                       # small-n test
#
# Collect with: bash scripts/geom/collect_table3_asdescribed.sh
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

NUM_MOLS="${NUM_MOLS:-4096}"            # molecules per seed (App. F.9)
SHARDS="${SHARDS:-4}"                   # shards per seed -> NUM_MOLS/SHARDS per task
SEED_GROUPS="${SEED_GROUPS:-3}"         # number of seeds (mean +- SE over these)
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
MFM_STEPS="${MFM_STEPS:-10}"
VALUE_SAMPLES="${VALUE_SAMPLES:-4}"
VALUE_BATCH="${VALUE_BATCH:-16}"
GUIDE_EVERY="${GUIDE_EVERY:-2}"
GUIDE_MIN_T="${GUIDE_MIN_T:-0.1}"
GUIDE_MAX_T="${GUIDE_MAX_T:-0.9}"
MU_SCHEDULE="${MU_SCHEDULE:-constant}"   # 'paper' = sigma_t^2/2 coefficient (multimfm.geom.mu_schedule)
LAMBDA="${LAMBDA:?set LAMBDA (selected on the held-out pool; see scripts/geom/tune_lambda_mu.sbatch)}"
MU="${MU:?set MU (selected on the held-out pool)}"
HOLDOUT_N="${HOLDOUT_N:-512}"
HOLDOUT_SEED="${HOLDOUT_SEED:-20260929}"
OUT_DIR="${OUT_DIR:-results/geom/runs/asdescribed_eval}"
TAG_PREFIX="${TAG_PREFIX:-asdesc}"

PER_SHARD=$(( NUM_MOLS / SHARDS ))
if [ $(( PER_SHARD * SHARDS )) -ne "${NUM_MOLS}" ]; then
  echo "NUM_MOLS=${NUM_MOLS} is not divisible by SHARDS=${SHARDS}" >&2; exit 1
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL=$(( SEED_GROUPS * SHARDS ))
if [ "${IDX}" -ge "${TOTAL}" ]; then
  echo "array index ${IDX} beyond ${TOTAL} tasks"; exit 0
fi
GROUP=$(( IDX / SHARDS ))
SHARD=$(( IDX % SHARDS ))
SEED=$(( 10 * GROUP + SHARD ))          # seeds 0..3, 10..13, 20..23 (as in job 47833756)
MUS_SUFFIX=""; [ "${MU_SCHEDULE}" != "constant" ] && MUS_SUFFIX="_mus${MU_SCHEDULE}"
TAG="${TAG_PREFIX}${MUS_SUFFIX}_r${GROUP}_s${SEED}_n${PER_SHARD}"
OUT_JSON="${OUT_DIR}/mfm_fingerprint_steering_${TAG}.json"

mkdir -p "${OUT_DIR}"
if [ -s "${OUT_JSON}" ]; then
  echo "already done: ${OUT_JSON}"; exit 0
fi

echo "task ${IDX}: seed group ${GROUP}, shard ${SHARD}, seed ${SEED}, ${PER_SHARD} molecules"
echo "lambda=${LAMBDA} mu=${MU} mu_schedule=${MU_SCHEDULE} sample_steps=${SAMPLE_STEPS} mfm_steps=${MFM_STEPS} value_samples=${VALUE_SAMPLES}"
"${PY}" -m multimfm.geom.mfm_steering \
  --num-mols "${PER_SHARD}" --sample-steps "${SAMPLE_STEPS}" --mfm-steps "${MFM_STEPS}" \
  --value-samples "${VALUE_SAMPLES}" --value-batch-size "${VALUE_BATCH}" \
  --guide-every "${GUIDE_EVERY}" --guide-min-t "${GUIDE_MIN_T}" --guide-max-t "${GUIDE_MAX_T}" \
  --mu "${MU}" --reward-lambda "${LAMBDA}" --seed "${SEED}" \
  --holdout-n "${HOLDOUT_N}" --holdout-seed "${HOLDOUT_SEED}" \
  --mu-schedule "${MU_SCHEDULE}" \
  --device cuda --tag "${TAG}" --output-dir "${OUT_DIR}"
echo "wrote ${OUT_JSON}"
