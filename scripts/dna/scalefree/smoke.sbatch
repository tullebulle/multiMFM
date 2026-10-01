#!/bin/bash
#SBATCH --job-name=dna_sf_dmfm_smoke
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Timing/sanity smoke for the dMFM scale-free rerun: the full per-probe protocol
# (GLASS-2048 reference, the whole N grid, 8 repeats) on SMOKE_PROBES conditioning
# states only, at the worst-case length. Multiply the per-probe time by 32 to size the
# real shards. Writes to a throwaway output root.
#
#   sbatch scripts/dna/scalefree/smoke.sbatch
#   SMOKE_LENGTH=50 SMOKE_REWARD=gc sbatch scripts/dna/scalefree/smoke.sbatch
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

SMOKE_LENGTH="${SMOKE_LENGTH:-400}"
SMOKE_REWARD="${SMOKE_REWARD:-conjunction}"
SMOKE_PROBES="${SMOKE_PROBES:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dna/scale_free_mc_dmfm_smoke}"

for STUDENT in dmfm dmfm4; do
  echo "=== ${STUDENT} L=${SMOKE_LENGTH} ${SMOKE_REWARD} probes=${SMOKE_PROBES}"
  ${PY} -u -m dmfm.scalefree.run_shard \
    --length "${SMOKE_LENGTH}" \
    --reward "${SMOKE_REWARD}" \
    --student "${STUDENT}" \
    --output_root "${OUTPUT_ROOT}" \
    --n_probes "${SMOKE_PROBES}" \
    --bootstrap 200
done
