#!/bin/bash
#SBATCH --job-name=dna_sf_dmfm_cpu
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-95%96
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# CPU route for Fig 5 / Tables 14-15 with dMFM finite-N gradients, for when
# kempner_requeue is backed up. serial_requeue denies kempner_albergo_lab, hence
# --account=albergo_lab.
#
# WHAT IS AND IS NOT COMPUTED HERE. Measured CPU cost per shard (8 threads):
#   GLASS-2048 reference   ~7 h at L=50, ~43 h at L=400   -> NOT viable on CPU
#   dMFM finite-N estimates ~1.8 h at L=50, ~11 h at L=400 -> viable, split by probe
# So this route computes only the dMFM estimates and REUSES the published run's
# GLASS-2048 reference gradients and reward calibration (--reference_from,
# --calibration_from). That is not a shortcut: the reference then *is* the tensor the
# published cells were scored against, and the calibration is identical by construction
# rather than to 6e-6. Only the finite-N dMFM gradients are produced here, in fp32 on CPU
# instead of TF32 on an H100; the precision bound on the resulting cells is stated in
# results/dna/scale_free_mc/dmfm/PROVENANCE.md and measured by
# `python -m dmfm.scalefree.precision_bound`.
#
# 96 tasks = 3 rewards x 4 lengths x 8 probe ranges of 4 states each, so no task is longer
# than ~80 min and preemption costs at most one range. A task whose part JSON exists exits
# immediately. Merge + score with: bash scripts/dna/scalefree/collect.sh
#
#   PUBLISHED_ROOT=/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir \
#     sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc_cpu.sbatch
#
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

STUDENT="${STUDENT:-dmfm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dna/scale_free_mc_dmfm_cpu}"
STUDENT_TAG="${STUDENT_TAG:-}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
PROBES_PER_TASK="${PROBES_PER_TASK:-4}"
N_PROBES="${N_PROBES:-32}"

if [ -z "${PUBLISHED_ROOT:-}" ]; then
  echo "PUBLISHED_ROOT must point at the directory holding the published run dirs" >&2
  exit 1
fi
# Published run directory names, per reward (the gc run is the _pairmetrics one).
declare -A PUB=( [gc]=rebuttal_gradient_accuracy_gc_pairmetrics \
                 [motif]=rebuttal_gradient_accuracy_motif \
                 [conjunction]=rebuttal_gradient_accuracy_conjunction )

REWARDS=(gc motif conjunction)
LENGTHS=(400 200 100 50)
RANGES=$(( N_PROBES / PROBES_PER_TASK ))
TASK="${SLURM_ARRAY_TASK_ID:-0}"
PER_REWARD=$(( 4 * RANGES ))
REWARD="${REWARDS[$(( TASK / PER_REWARD ))]}"
REST=$(( TASK % PER_REWARD ))
LENGTH="${LENGTHS[$(( REST / RANGES ))]}"
RANGE=$(( REST % RANGES ))
START=$(( RANGE * PROBES_PER_TASK ))
END=$(( START + PROBES_PER_TASK ))
PUB_DIR="${PUBLISHED_ROOT}/${PUB[$REWARD]}"

echo "task=${TASK} reward=${REWARD} L=${LENGTH} probes=[${START},${END}) threads=${THREADS}"
echo "reference+calibration reused from ${PUB_DIR}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"

set -- --length "${LENGTH}" --reward "${REWARD}" --student "${STUDENT}" \
       --output_root "${OUTPUT_ROOT}" --device cpu --torch_threads "${THREADS}" \
       --probe_start "${START}" --probe_end "${END}" --n_probes "${N_PROBES}" \
       --reference_from "${PUB_DIR}" --calibration_from "${PUB_DIR}" --skip_existing
[ -n "${STUDENT_TAG}" ] && set -- "$@" --student_tag "${STUDENT_TAG}"

${PY} -u -m dmfm.scalefree.run_shard "$@"
