#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_nfe
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# GEOM-Drugs fingerprint steering against the same-backbone baselines, on the NFE axis.
#
# The published comparison is to TFG-Flow, which fixes the external scale but shares no code
# with us and so cannot say whether the gain comes from the value estimator or from anything
# else about our generator. These three arms share the base model, the guide head, the
# targets, the metric and the validity treatment, and differ only in how the budget is spent:
#
#   multiMFM    log-mean-exp over flow-map posterior samples   -- has an N to spend
#   DPS         the reward at the one-step denoiser endpoint   -- has no N to spend
#   best-of-N   N independent draws, picked by the guide head  -- spends it on sampling
#
# Cost, forward-equivalent, with the backward pass billed at twice a forward:
#   guided     100 + 3 x G x S x MC      (G guided steps, S flow-map steps, MC samples)
#   best-of-N  100 x N
# The window t in [0.1,0.9] over 100 outer steps gives G = 80 / guide_every.
#
# Rungs targeted: ~200, ~400, ~1000, ~2000, ~4000.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

SEEDS=(301 302 303)
# Tasks 0-2 run the guided arms, tasks 3-5 the search baseline, so the two halves run in
# parallel rather than queueing behind each other: best-of-40 alone is 10,240 trajectories
# plus a PoseBusters check on every candidate, which is the long pole of the whole sweep.
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$(( TASK % 3 ))]}"
ARM=$(( TASK / 3 ))
N="${N:-256}"
OUT="results/geom/runs/fp_nfe_sweep"
mkdir -p "${OUT}"

COMMON=(--num-mols "$N" --reward-kind fingerprint_tanimoto --reward-lambda 20 --mu 2.0
        --guide-min-t 0.1 --guide-max-t 0.9 --sample-steps 100
        --target-selection filter --device cuda --pb
        --output-dir "${OUT}" --seed "${SEED}")

if [ "${ARM}" -eq 0 ]; then
  # ---- DPS: one denoiser call per guided step; only the guiding frequency sets its cost.
  for GE in 4 2 1; do
    python -u -m multimfm.geom.dps_steering "${COMMON[@]}" \
      --guide-every "$GE" --tag "dps_ge${GE}_seed${SEED}" || echo "FAILED dps ge${GE}"
  done

  # ---- multiMFM: sweep the posterior budget, the lever DPS does not have.
  for cfg in "4 2 1" "2 2 1" "2 2 4" "1 2 4" "1 2 8"; do
    set -- $cfg
    python -u -m multimfm.geom.mfm_steering "${COMMON[@]}" \
      --guide-every "$1" --mfm-steps "$2" --value-samples "$3" --value-batch-size 8 \
      --tag "mfm_ge$1_s$2_mc$3_seed${SEED}" || echo "FAILED mfm $cfg"
  done
else
  # ---- best-of-N: the search baseline, selecting on the guide head only.
  for K in 2 4 10 20 40; do
    python -u scripts/geom/fp_best_of_n.py --num-mols "$N" --n-candidates "$K" \
      --batch 32 --seed "${SEED}" --pb --device cuda \
      --out-dir "${OUT}" --tag "bon_k${K}_seed${SEED}" || echo "FAILED bon k${K}"
  done
fi

echo "done task ${TASK} (arm ${ARM}, seed ${SEED})"
