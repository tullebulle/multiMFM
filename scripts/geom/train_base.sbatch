#!/usr/bin/env bash
#SBATCH --job-name=geom_train_base
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Train the GEOM-Drugs base flow model (paper Table 19, GEOM-Drugs column).
# Reproduces the run that produced checkpoints/geom/teacher/epoch=38-step=89310.ckpt:
# 1 GPU, batch 512 (x8 rotation copies = 4,096 effective), lr 2e-3, EMA 0.999,
# ~89k steps / 39 epochs, which took ~17 h on an H200.
#
#   sbatch scripts/geom/train_base.sbatch                    # from the repo root
#   EXPERIMENT=geom_dfm_xlarge sbatch scripts/geom/train_base.sbatch
#   MAX_STEPS=200 BATCH_SIZE=8 sbatch --partition=gpu_test --time=00:30:00 \
#       scripts/geom/train_base.sbatch                       # smoke test
#
# Requires the processed GEOM-Drugs splits in data/geom/ (public on Hugging Face,
# carlosinator/tabasco-geom-drugs -- see docs/provenance/geom/data.md). The LMDB is
# built into data/geom/lmdb_geom on the first run; staging it on node-local storage
# (LMDB_DIR=/tmp/...) is faster but then it is rebuilt every job.
# Training is not bit-reproducible on GPU, so a retrained model is a new sample from
# the same recipe, not the shipped weights.
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

EXPERIMENT="${EXPERIMENT:-geom_dfm}"
TRAINER="${TRAINER:-gpu}"
DEVICES="${DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_STEPS="${MAX_STEPS:-200000}"
LMDB_DIR="${LMDB_DIR:-}"
CKPT_PATH="${CKPT_PATH:-}"          # resume from a Lightning checkpoint

overrides=(
  "experiment=${EXPERIMENT}"
  "trainer=${TRAINER}"
  "trainer.devices=${DEVICES}"
  "trainer.max_steps=${MAX_STEPS}"
  "datamodule.batch_size=${BATCH_SIZE}"
)
[ -n "${LMDB_DIR}" ] && overrides+=("datamodule.lmdb_dir=${LMDB_DIR}")
[ -n "${CKPT_PATH}" ] && overrides+=("ckpt_path=${CKPT_PATH}")

echo "hydra overrides: ${overrides[*]}"
"${PY}" -m multimfm.geom.train_base "${overrides[@]}"

echo "checkpoints in outputs/geom/base/${SLURM_JOB_ID:-manual}/"
