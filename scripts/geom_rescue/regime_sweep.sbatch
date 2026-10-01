#!/usr/bin/env bash
#SBATCH --job-name=geom_rescue_regime
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --array=1-7%3
#SBATCH --constraint="cc8.0|cc9.0"
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Regime sweep for docs/geom_rescue_attempt.md H4: the 17-point grid in
# results/geom/diagnostics/guidance_grid.csv varied only (lambda, mu, mu-schedule).
# It never varied the guided t-window, the guided-step density, the look-ahead
# sample count, the look-ahead step count, or coords-only vs coords+types.
# Every arm is at the selected lambda=0.01, mu=0.25, constant schedule, n=256 on
# the held-out target pool, so it is directly comparable to that grid's R0 point.
#
#   sbatch --partition=kempner_requeue --requeue scripts/geom_rescue/regime_sweep.sbatch
#
# Arms (see the pre-registration in docs/geom_rescue_attempt.md 0.2).
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

LAMBDA="${LAMBDA:-0.01}"
MU="${MU:-0.25}"
NUM_MOLS="${NUM_MOLS:-256}"
OUT_DIR="${OUT_DIR:-results/geom/rescue/regime}"
IDX="${SLURM_ARRAY_TASK_ID:-1}"

# defaults = the recorded configuration
EXTRA=()
GUIDE_EVERY=2 ; GMIN=0.1 ; GMAX=0.9 ; VS=4 ; MFM_STEPS=10
case "${IDX}" in
  1) NAME=R1_late_t0.70    ; GMIN=0.70 ; GMAX=1.0 ; GUIDE_EVERY=1 ;;
  2) NAME=R2_late_t0.85    ; GMIN=0.85 ; GMAX=1.0 ; GUIDE_EVERY=1 ;;
  3) NAME=R3_early_t0.1_0.5; GMIN=0.10 ; GMAX=0.5 ; GUIDE_EVERY=1 ;;
  4) NAME=R4_every1        ; GUIDE_EVERY=1 ;;
  5) NAME=R5_vs16          ; VS=16 ;;
  6) NAME=R6_coords_only   ; EXTRA+=(--no-guide-atomics) ;;
  7) NAME=R7_mfmsteps20    ; MFM_STEPS=20 ;;
  *) echo "no arm ${IDX}"; exit 0 ;;
esac

TAG="rescue_${NAME}_lam${LAMBDA}_mu${MU}_n${NUM_MOLS}"
OUT_JSON="${OUT_DIR}/mfm_fingerprint_steering_${TAG}.json"
mkdir -p "${OUT_DIR}"
if [ -s "${OUT_JSON}" ]; then echo "already done: ${OUT_JSON}"; exit 0; fi

echo "arm ${IDX} ${NAME}: gmin=${GMIN} gmax=${GMAX} every=${GUIDE_EVERY} vs=${VS} mfm=${MFM_STEPS} extra=${EXTRA[*]:-none}"
"${PY}" -m multimfm.geom.mfm_steering \
  --num-mols "${NUM_MOLS}" --sample-steps 100 --mfm-steps "${MFM_STEPS}" \
  --value-samples "${VS}" --value-batch-size 8 \
  --guide-every "${GUIDE_EVERY}" --guide-min-t "${GMIN}" --guide-max-t "${GMAX}" \
  --mu "${MU}" --reward-lambda "${LAMBDA}" --seed 0 \
  --holdout-n 512 --holdout-seed 20260929 --tune-on-holdout \
  --mu-schedule constant "${EXTRA[@]}" \
  --device cuda --tag "${TAG}" --output-dir "${OUT_DIR}"
echo "wrote ${OUT_JSON}"
