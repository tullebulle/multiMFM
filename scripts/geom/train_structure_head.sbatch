#!/usr/bin/env bash
#SBATCH --job-name=geom_structure_head
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Train the GEOM-Drugs fingerprint guide head E(G) used by the Table 3 steering runs
# (TFG-Flow EGNN Predictor -> 1024-bit FP2, BCE loss, val Tanimoto for selection).
#
# The released head (checkpoints/geom/structure_head/best.pth, val Tanimoto 0.852) was
# trained in four legs; this script runs one leg. The final, released leg was:
#   3 GPUs x 256 (global 768), lr 1e-3 cosine, warmup 500, 40 epochs, warm-started from
#   the previous leg. See checkpoints/geom/structure_head/train_args.json.
#
#   sbatch scripts/geom/train_structure_head.sbatch                       # leg 1 (no warm start)
#   INIT_CKPT=checkpoints/geom/structure_head_rerun/best.pth LR=1e-3 LR_SCHEDULE=cosine \
#       EPOCHS=40 sbatch scripts/geom/train_structure_head.sbatch         # final leg
#   EPOCHS=1 LIMIT_TRAIN=2000 LIMIT_VAL=500 NPROC=1 sbatch --gres=gpu:1 \
#       --partition=gpu_test --time=00:30:00 scripts/geom/train_structure_head.sbatch
#
# Needs the TRAIN split of the prepared fingerprint LMDB, which is NOT shipped (only
# val is). Rebuild it first -- see docs/provenance/geom/data.md.
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

NPROC="${NPROC:-3}"
DATA_DIR="${DATA_DIR:-data/geom/geom_fp}"
LOG_DIR="${LOG_DIR:-checkpoints/geom/structure_head_rerun}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"       # per GPU
LR="${LR:-3e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-none}"    # 'cosine' for the final leg
WARMUP_STEPS="${WARMUP_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-6}"
COORD_NORMALIZER="${COORD_NORMALIZER:-2.0}"
INIT_CKPT="${INIT_CKPT:-}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"

if [ ! -f "${DATA_DIR}/train.lmdb" ]; then
  echo "missing ${DATA_DIR}/train.lmdb -- the 1.3 GB train split is not shipped;" >&2
  echo "rebuild it as described in docs/provenance/geom/data.md" >&2
  exit 1
fi

extra=()
[ -n "${INIT_CKPT}" ] && extra+=(--init_ckpt "${INIT_CKPT}")
[ -n "${LIMIT_TRAIN}" ] && extra+=(--limit_train "${LIMIT_TRAIN}")
[ -n "${LIMIT_VAL}" ] && extra+=(--limit_val "${LIMIT_VAL}")
[ "${WANDB:-0}" = "1" ] && extra+=(--wandb --wandb_name "${WANDB_NAME:-structure-head}")

echo "nproc=${NPROC} batch/gpu=${BATCH_SIZE} global=$(( BATCH_SIZE * NPROC )) lr=${LR} epochs=${EPOCHS}"
"${PY}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m multimfm.geom.train_structure_head \
  --data_dir "${DATA_DIR}" --log_dir "${LOG_DIR}" \
  --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --lr "${LR}" \
  --lr_schedule "${LR_SCHEDULE}" --warmup_steps "${WARMUP_STEPS}" \
  --num_workers "${NUM_WORKERS}" --coord_normalizer "${COORD_NORMALIZER}" \
  --h_embed_size 128 --e_embed_size 128 --cls_embed_size 64 --num_layers 6 \
  "${extra[@]}"

echo "head checkpoints in ${LOG_DIR}"
