#!/usr/bin/env bash
#SBATCH --job-name=geom_pb_valid
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# PoseBusters validity of the GEOM-Drugs base model -- the "Unsteered (%)" column of
# paper Table 21. No surviving record measures this on the 4,096 molecules Table 21
# claims (see results/geom/PROVENANCE.md), so this script is also how that number gets
# established: run it with NUM_SAMPLES=4096.
#
#   sbatch scripts/geom/eval_pb_validity.sbatch                      # 200 mols, EMA
#   NUM_SAMPLES=4096 TIME=08:00:00 sbatch scripts/geom/eval_pb_validity.sbatch
#   PB_CONFIG=src/tabasco/utils/posebusters_no_strain.yaml \
#       sbatch scripts/geom/eval_pb_validity.sbatch                  # training-curve config
#   NUM_SAMPLES=20 sbatch --partition=gpu_test --time=00:20:00 \
#       scripts/geom/eval_pb_validity.sbatch                         # smoke test
#
# The default PB configuration is the GEOM experimental one (configs/geom/posebusters_geom.yaml),
# i.e. the one Table 3 and Table 21 use. The historical training/validation curve in
# results/geom/teacher_pb_curve.csv used posebusters_no_strain.yaml instead.
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

CHECKPOINT="${CHECKPOINT:-checkpoints/geom/teacher/epoch=38-step=89310.ckpt}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
NUM_STEPS="${NUM_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-100}"
PB_CONFIG="${PB_CONFIG:-configs/geom/posebusters_geom.yaml}"
SEED="${SEED:-0}"
WEIGHTS="${WEIGHTS:-ema}"            # 'ema' (recommended) or 'raw'
OUT="${OUT:-results/geom/runs/pb_validity_${SLURM_JOB_ID:-manual}.json}"

extra=()
[ "${WEIGHTS}" = "ema" ] && extra+=(--ema)

echo "checkpoint=${CHECKPOINT} n=${NUM_SAMPLES} steps=${NUM_STEPS} weights=${WEIGHTS} pb=${PB_CONFIG}"
"${PY}" -m multimfm.geom.check_pb_validity \
  --checkpoint "${CHECKPOINT}" --num-samples "${NUM_SAMPLES}" --num-steps "${NUM_STEPS}" \
  --batch-size "${BATCH_SIZE}" --pb-config "${PB_CONFIG}" --seed "${SEED}" \
  --device cuda --out "${OUT}" "${extra[@]}"

echo "wrote ${OUT}"
