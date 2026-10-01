#!/usr/bin/env bash
#SBATCH --job-name=geom_tune_lammu
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_requeue
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14%5
#SBATCH --constraint="cc8.0|cc9.0"
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# NOTE: torch 2.5.1+cu124 has no sm_120 kernels, so the Blackwell (rtx6000pro) nodes in
# kempner_requeue fail with "no kernel image is available for execution on the device".
# The constraint above restricts the job to compute capability 8.0/9.0 (A100/H100/H200).
#
# Select (lambda, mu) for the as-described Table 3 rerun on a HELD-OUT target set.
#
# The paper states the reward shape R(G)=exp(-||E(G)-c||^2) but no numeric guidance
# strengths for GEOM-Drugs, and the recorded runs used lambda in [0.02, 0.1], mu in
# [0.25, 0.5] with no documented selection. This array sweeps a grid on the held-out
# pool (`--tune-on-holdout`, HOLDOUT_N targets defined by HOLDOUT_SEED) which the
# evaluation run then excludes, so selection and evaluation share no target.
# lambda=1.0 is the paper's literal reward (no extra scale) and is included.
#
# MU_SCHEDULE=paper switches the guidance coefficient from the code's constant mu to the
# paper's sigma_t^2/2 = (1-t)/t (mu becomes an overall scale); see multimfm.geom.mu_schedule.
# Those runs get a '_muspaper' tag so they never collide with the constant-mu sweep.
#
# One array task per grid point; each writes its own JSON and is skipped if that file
# already exists, so preempted tasks rerun alone.
#
#   sbatch scripts/geom/tune_lambda_mu.sbatch                      # from the repo root
#   TUNE_N=32 sbatch --array=0-14%5 --partition=gpu_test --time=00:40:00 \
#       scripts/geom/tune_lambda_mu.sbatch                         # quick test
#
# The paper-coefficient sweep that tests the constant-mu substitution (Slurm 49281926):
#   MU_SCHEDULE=paper TUNE_N=256 \
#     GRID_OVERRIDE="0.01:0.05 0.01:0.15 0.01:0.5 0.02:0.05 0.02:0.15 0.02:0.5" \
#     sbatch --array=0-5%3 --time=00:25:00 scripts/geom/tune_lambda_mu.sbatch
# The mu grid is smaller there because (1-t)/t multiplies it by up to 9x inside t in [0.1,0.9].
#
# Collect with: bash scripts/geom/collect_table3_asdescribed.sh
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

TUNE_N="${TUNE_N:-256}"
HOLDOUT_N="${HOLDOUT_N:-512}"
HOLDOUT_SEED="${HOLDOUT_SEED:-20260929}"
SEED="${SEED:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"     # App. F.8: 100-step outer ODE
MFM_STEPS="${MFM_STEPS:-10}"            # App. F.8: 10 flow-map inner steps
VALUE_SAMPLES="${VALUE_SAMPLES:-4}"     # App. F.8: four posterior samples
VALUE_BATCH="${VALUE_BATCH:-8}"
GUIDE_EVERY="${GUIDE_EVERY:-2}"
GUIDE_MIN_T="${GUIDE_MIN_T:-0.1}"
GUIDE_MAX_T="${GUIDE_MAX_T:-0.9}"
OUT_DIR="${OUT_DIR:-results/geom/runs/asdescribed_tuning}"

# lambda x mu grid; index = SLURM_ARRAY_TASK_ID.
GRID=(
  "0.01 0.25" "0.01 0.5" "0.01 1.0"
  "0.02 0.25" "0.02 0.5" "0.02 1.0"
  "0.05 0.25" "0.05 0.5" "0.05 1.0"
  "0.10 0.25" "0.10 0.5" "0.10 1.0"
  "1.00 0.25" "1.00 0.5" "1.00 1.0"
)
# GRID_OVERRIDE="lam:mu lam:mu ..." replaces the grid (used for the second, literal-schedule
# pass: Algorithm 2 guides at EVERY step over the whole trajectory, while the recorded runs
# guide every 2nd step inside t in [0.1, 0.9]; the paper states neither).
if [ -n "${GRID_OVERRIDE:-}" ]; then
  GRID=()
  for pair in ${GRID_OVERRIDE}; do GRID+=("${pair%%:*} ${pair##*:}"); done
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#GRID[@]}" ]; then
  echo "array index ${IDX} beyond the grid (${#GRID[@]} points)"; exit 0
fi
read -r LAMBDA MU <<< "${GRID[$IDX]}"
# Non-default guidance schedules get a tag suffix so their outputs never collide with the
# recorded-schedule sweep.
SCHED_SUFFIX=""
if [ "${GUIDE_EVERY}" != "2" ] || [ "${GUIDE_MIN_T}" != "0.1" ] || [ "${GUIDE_MAX_T}" != "0.9" ]; then
  SCHED_SUFFIX="_ge${GUIDE_EVERY}_t${GUIDE_MIN_T}-${GUIDE_MAX_T}"
fi
# TAG_SUFFIX/EXTRA_ARGS let the same script run diagnostic probes of a negative gain on the
# held-out pool without a new file, e.g.
#   VALUE_SAMPLES=16 TAG_SUFFIX=_vs16 GRID_OVERRIDE="0.01:0.25" ...
#   EXTRA_ARGS="--no-guide-atomics" TAG_SUFFIX=_coordsonly GRID_OVERRIDE="0.01:0.25" ...
MU_SCHEDULE="${MU_SCHEDULE:-constant}"
if [ "${MU_SCHEDULE}" != "constant" ]; then
  SCHED_SUFFIX="${SCHED_SUFFIX}_mus${MU_SCHEDULE}"
fi
SCHED_SUFFIX="${SCHED_SUFFIX}${TAG_SUFFIX:-}"
TAG="tune_lam${LAMBDA}_mu${MU}_n${TUNE_N}${SCHED_SUFFIX}"
OUT_JSON="${OUT_DIR}/mfm_fingerprint_steering_${TAG}.json"

mkdir -p "${OUT_DIR}"
if [ -s "${OUT_JSON}" ]; then
  echo "already done: ${OUT_JSON}"; exit 0
fi

echo "tuning point ${IDX}: lambda=${LAMBDA} mu=${MU} on ${TUNE_N} held-out targets"
"${PY}" -m multimfm.geom.mfm_steering \
  --num-mols "${TUNE_N}" --sample-steps "${SAMPLE_STEPS}" --mfm-steps "${MFM_STEPS}" \
  --value-samples "${VALUE_SAMPLES}" --value-batch-size "${VALUE_BATCH}" \
  --guide-every "${GUIDE_EVERY}" --guide-min-t "${GUIDE_MIN_T}" --guide-max-t "${GUIDE_MAX_T}" \
  --mu "${MU}" --reward-lambda "${LAMBDA}" --seed "${SEED}" \
  --holdout-n "${HOLDOUT_N}" --holdout-seed "${HOLDOUT_SEED}" --tune-on-holdout \
  --mu-schedule "${MU_SCHEDULE}" \
  --device cuda --tag "${TAG}" --output-dir "${OUT_DIR}" ${EXTRA_ARGS:-}
echo "wrote ${OUT_JSON}"
