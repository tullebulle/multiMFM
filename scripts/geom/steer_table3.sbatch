#!/usr/bin/env bash
#SBATCH --job-name=geom_steer_table3
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
# Table 3, GEOM-Drugs structural-fingerprint steering with multiMFM.
#
# Two modes, because the paper's text and its number disagree (see
# results/geom/PROVENANCE.md and notebooks/08_geom_fingerprint.ipynb):
#
#   MODE=grid   the four-point (lambda, mu) grid at n=64, one seed -- what actually
#               produced the printed 0.227 / 87.5% (Slurm job 35670937).
#   MODE=paper  the same configuration at the n=4,096 the text claims, repeated over
#               REPLICATES seeds, sharded over the GPUs of the job (job 47833756).
#               Needs --gres=gpu:4 to match the original wall clock.
#   MODE=single one (lambda, mu, seed, n) point; everything is overridable.
#
# Examples (submit from the repo root):
#   sbatch scripts/geom/steer_table3.sbatch                                  # grid, n=64
#   MODE=paper NUM_MOLS=4096 REPLICATES=3 sbatch --gres=gpu:4 --cpus-per-task=64 \
#       --mem=256G scripts/geom/steer_table3.sbatch
#   MODE=single NUM_MOLS=16 LAMBDA=0.02 MU=0.5 SEED=0 sbatch --partition=gpu_test \
#       --time=00:40:00 scripts/geom/steer_table3.sbatch                     # smoke
#
# Everything else keeps the recorded configuration: 100 outer ODE steps, 10 flow-map
# inner steps, 4 posterior samples, guide every 2 steps in t in [0.1, 0.9], EMA teacher
# weights, PoseBusters on (configs/geom/posebusters_geom.yaml).
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

MODE="${MODE:-grid}"
NUM_MOLS="${NUM_MOLS:-64}"
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
MFM_STEPS="${MFM_STEPS:-10}"
VALUE_SAMPLES="${VALUE_SAMPLES:-4}"
VALUE_BATCH="${VALUE_BATCH:-8}"
GUIDE_EVERY="${GUIDE_EVERY:-2}"
GUIDE_MIN_T="${GUIDE_MIN_T:-0.1}"
GUIDE_MAX_T="${GUIDE_MAX_T:-0.9}"
LAMBDA="${LAMBDA:-0.02}"
MU="${MU:-0.5}"
SEED="${SEED:-0}"
REPLICATES="${REPLICATES:-3}"
TARGET_SELECTION="${TARGET_SELECTION:-filter}"   # 'legacy' reproduces the 71-vs-82 crash
OUT_DIR="${OUT_DIR:-results/geom/runs/${MODE}_${SLURM_JOB_ID:-manual}}"
TAG_PREFIX="${TAG_PREFIX:-table3}"

common_args=(
  --sample-steps "${SAMPLE_STEPS}" --mfm-steps "${MFM_STEPS}"
  --value-samples "${VALUE_SAMPLES}" --value-batch-size "${VALUE_BATCH}"
  --guide-every "${GUIDE_EVERY}" --guide-min-t "${GUIDE_MIN_T}" --guide-max-t "${GUIDE_MAX_T}"
  --target-selection "${TARGET_SELECTION}"
  --device cuda --output-dir "${OUT_DIR}"
)

run_point() {  # $1=lambda $2=mu $3=seed $4=n $5=tag [$6=cuda device]
  echo "======== lambda=$1 mu=$2 seed=$3 n=$4 (mfm_steps=${MFM_STEPS}) ========"
  CUDA_VISIBLE_DEVICES="${6:-0}" "${PY}" -m multimfm.geom.mfm_steering \
    "${common_args[@]}" --num-mols "$4" --mu "$2" --reward-lambda "$1" --seed "$3" \
    --tag "$5"
}

case "${MODE}" in
  grid)
    # The recorded four points, in the order job 35670937 ran them.
    for point in "0.05 0.5" "0.02 0.5" "0.05 0.25" "0.1 0.5"; do
      set -- ${point}
      run_point "$1" "$2" "${SEED}" "${NUM_MOLS}" "${TAG_PREFIX}_n${NUM_MOLS}_lam$1_mu$2"
    done
    ;;
  paper)
    GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
    PER_SHARD=$(( NUM_MOLS / GPUS ))
    if [ $(( PER_SHARD * GPUS )) -ne "${NUM_MOLS}" ]; then
      echo "NUM_MOLS=${NUM_MOLS} is not divisible by GPUS=${GPUS}" >&2; exit 1
    fi
    echo "n=${NUM_MOLS} over ${GPUS} GPU(s) => ${PER_SHARD} molecules/shard, ${REPLICATES} replicate(s)"
    for r in $(seq 0 $(( REPLICATES - 1 ))); do
      echo "======== REPLICATE ${r} ========"
      for i in $(seq 0 $(( GPUS - 1 ))); do
        s=$(( 10 * r + i ))
        run_point "${LAMBDA}" "${MU}" "${s}" "${PER_SHARD}" \
          "${TAG_PREFIX}_n${NUM_MOLS}_r${r}_s${s}" "${i}" \
          > "logs/${SLURM_JOB_ID:-manual}_r${r}_s${s}.log" 2>&1 &
      done
      wait
      echo "  replicate ${r} shards done."
    done
    ;;
  single)
    run_point "${LAMBDA}" "${MU}" "${SEED}" "${NUM_MOLS}" \
      "${TAG_PREFIX}_n${NUM_MOLS}_lam${LAMBDA}_mu${MU}_s${SEED}"
    ;;
  *)
    echo "unknown MODE=${MODE} (grid|paper|single)" >&2; exit 1
    ;;
esac

echo "======== AGGREGATE ========"
"${PY}" scripts/geom/aggregate_steering.py "${OUT_DIR}"
echo "======== DONE: ${OUT_DIR} ========"
