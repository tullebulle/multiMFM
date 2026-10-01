#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_t3_4k
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Table 3 at the sample size the superseded negative result had.
#
# The confirmed result stands on 256 molecules x 3 seeds. The negative result it replaces
# stood on 4,096 x 3. A reader comparing the two will notice the sample size fell by 16x at
# the same moment the sign flipped, and that is a fair thing to notice. This run removes the
# objection rather than answering it: same 4,096 x 3 design, same sharding (4 shards of 1,024
# pooled into one replicate), on twelve seeds that have never been used for anything.
#
# Both blocks of the table are produced here:
#   ours_n4096       -- molecules up to 71 heavy atoms, the headline configuration
#   ours_max30_n4096 -- the same model under TFG-Flow's 30-heavy-atom cap
# Each run emits its own unguided arm on identical noise and identical targets, so the
# baseline rows come from the same runs and are paired, not borrowed.
#
# Targets are drawn per seed (verified: seeds 1/2/3 of the n=256 run touch disjoint index
# ranges), so the four shards of a replicate score different targets rather than repeating one.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

# 12 tasks = 3 replicates x 4 shards. Replicate r pools shards 4r..4r+3.
TASK="${SLURM_ARRAY_TASK_ID:-0}"
REP=$(( TASK / 4 ))
SHARD=$(( TASK % 4 ))
SEED=$(( 101 + TASK ))
N="${N:-1024}"
OUT="results/geom/runs/fp_table3_n4096"
mkdir -p "${OUT}"

echo "replicate ${REP} shard ${SHARD} seed ${SEED} n=${N}"

# ---- block 1: up to 71 heavy atoms, headline config (lambda 20, gamma 2, 2 steps, MC 8, every step)
python -u -m multimfm.geom.mfm_steering \
  --num-mols "$N" --reward-kind fingerprint_tanimoto --reward-lambda 20 --mu 2.0 \
  --mfm-steps 2 --guide-min-t 0.1 --guide-max-t 0.9 --guide-every 1 \
  --value-samples 8 --value-batch-size 8 --sample-steps 100 \
  --target-selection filter --device cuda --pb \
  --output-dir "${OUT}" --tag "ours_n4096_rep${REP}_shard${SHARD}" --seed "${SEED}" \
  || echo "FAILED ours_n4096 rep${REP} shard${SHARD}"

# ---- block 2: same model under TFG-Flow's cap (MC 4, every second step -- the cheaper setting)
python -u -m multimfm.geom.mfm_steering \
  --num-mols "$N" --reward-kind fingerprint_tanimoto --reward-lambda 20 --mu 2.0 \
  --mfm-steps 2 --max-atoms 30 \
  --guide-min-t 0.1 --guide-max-t 0.9 --guide-every 2 \
  --value-samples 4 --value-batch-size 8 --sample-steps 100 \
  --target-selection filter --device cuda --pb \
  --output-dir "${OUT}" --tag "ours_max30_n4096_rep${REP}_shard${SHARD}" --seed "${SEED}" \
  || echo "FAILED ours_max30_n4096 rep${REP} shard${SHARD}"

echo "done replicate ${REP} shard ${SHARD} seed ${SEED}"
