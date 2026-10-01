#!/usr/bin/env bash
#SBATCH --job-name=geom_rescue_dose
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --array=0-11%3
#SBATCH --constraint="cc8.0|cc9.0"
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Displacement-dose sweep with a scale-free guidance step
# (docs/geom_rescue_attempt.md H4 / §0.3). The recorded grid swept (lambda, mu),
# which couples the tilt direction to the step magnitude; here the per-molecule
# gradient is renormalised to RMS 1.0 so the coherent coordinate displacement over
# the guided window is exactly  mu * 1.0 * sum(dt) * 2.0 A = 0.8 * mu  A
# (window t in [0.1,0.9], guide_every 2, log schedule: sum(dt) = 0.4).
#
# Doses: 0.01, 0.03, 0.06, 0.12, 0.30, 0.60 A, at lambda = 0.01 and lambda = 1.0.
# The 0.02-0.06 A band is the only one in which FP2 responds without being
# randomised, and no recorded point sits in it.
#
#   sbatch --partition=kempner_h200 scripts/geom_rescue/dose_sweep.sbatch
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

NUM_MOLS="${NUM_MOLS:-256}"
OUT_DIR="${OUT_DIR:-results/geom/rescue/dose}"
RMS="${RMS:-1.0}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

DOSES=(0.01 0.03 0.06 0.12 0.30 0.60)
LAMBDAS=(0.01 1.0)
NL=${#LAMBDAS[@]}
D_I=$(( IDX / NL )); L_I=$(( IDX % NL ))
if [ "${D_I}" -ge "${#DOSES[@]}" ]; then echo "no arm ${IDX}"; exit 0; fi
DOSE="${DOSES[$D_I]}"; LAMBDA="${LAMBDAS[$L_I]}"
MU=$(awk -v d="${DOSE}" 'BEGIN{printf "%.6g", d/0.8}')

TAG="rescue_dose${DOSE}A_lam${LAMBDA}_rms${RMS}_n${NUM_MOLS}"
OUT_JSON="${OUT_DIR}/mfm_fingerprint_steering_${TAG}.json"
mkdir -p "${OUT_DIR}"
if [ -s "${OUT_JSON}" ]; then echo "already done: ${OUT_JSON}"; exit 0; fi

echo "arm ${IDX}: target dose ${DOSE} A -> mu=${MU}, lambda=${LAMBDA}, grad RMS fixed to ${RMS}"
"${PY}" scripts/geom_rescue/dose_steering.py \
  --num-mols "${NUM_MOLS}" --sample-steps 100 --mfm-steps 10 \
  --value-samples 4 --value-batch-size 8 \
  --guide-every 2 --guide-min-t 0.1 --guide-max-t 0.9 \
  --mu "${MU}" --reward-lambda "${LAMBDA}" \
  --guidance-max-coord-rms "${RMS}" --guidance-max-atom-rms "${RMS}" \
  --seed 0 --holdout-n 512 --holdout-seed 20260929 --tune-on-holdout \
  --mu-schedule constant --device cuda --tag "${TAG}" --output-dir "${OUT_DIR}"
echo "wrote ${OUT_JSON}"
