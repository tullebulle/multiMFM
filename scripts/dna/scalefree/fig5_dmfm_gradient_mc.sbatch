#!/bin/bash
#SBATCH --job-name=dna_sf_dmfm
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Fig 5 and Tables 14-15, rerun with finite-N **dMFM** value-gradient estimates against
# the published independent GLASS-2048 reference (dmfm.scalefree.run_shard). The
# published runs used the GLASS posterior on the base DFM on both sides, so they measured
# GLASS Monte Carlo error only; the Fig 5 caption and App F.2 describe this rerun.
#
# One array task per (reward, length); a task whose L{L}/metadata.json exists exits
# immediately, so a preempted or failed shard reruns alone:
#
#   sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch                  # all 12
#   sbatch --array=3,7 scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch      # two shards
#   STUDENT=dmfm4 REFERENCE_FROM=outputs/dna/scale_free_mc_dmfm/diag1step \
#     sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch                # 4-step students
#
# STUDENT=dmfm (default) uses the diagonal+ESD students of Tables 13/17 with one composed
# flow-map step, as App F.2 describes. STUDENT=dmfm4 uses the 4-step ESD students of
# Table 18 (same one-step map, for comparison). REFERENCE_FROM points at a *student tag*
# directory whose GLASS-2048 reference gradients are reused (they do not depend on the
# student), which makes the second student nearly free.
#
# Then: bash scripts/dna/scalefree/collect.sh
#
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

STUDENT="${STUDENT:-dmfm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dna/scale_free_mc_dmfm}"
MC_CHUNK="${MC_CHUNK:-16}"

# Longest lengths first so they start while the array is throttled.
REWARDS=(gc motif conjunction)
LENGTHS=(400 200 100 50)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
REWARD="${REWARDS[$((TASK / 4))]}"
LENGTH="${LENGTHS[$((TASK % 4))]}"
echo "task=${TASK} reward=${REWARD} length=${LENGTH} student=${STUDENT}"

set -- --length "${LENGTH}" --reward "${REWARD}" --student "${STUDENT}" \
       --output_root "${OUTPUT_ROOT}" --mc_chunk "${MC_CHUNK}" --skip_existing
if [ -n "${REFERENCE_FROM:-}" ]; then
  set -- "$@" --reference_from "${REFERENCE_FROM}/${REWARD}"
fi

${PY} -u -m dmfm.scalefree.run_shard "$@"
