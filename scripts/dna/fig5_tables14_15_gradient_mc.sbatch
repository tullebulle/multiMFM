#!/bin/bash
#SBATCH --job-name=dna_fig5_gradmc
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Fig 5 and Tables 14-15: finite-MC GLASS value-gradient accuracy against a
# GLASS-2048 reference, on the BASE DFM (not the dMFM; see
# docs/provenance/dna/original_runs.md section 8 and section 13). 32 probe states per
# length are obtained by integrating the base flow to t=0.5, 8 noise pools each,
# N in {1..256}.
# Original: slurm/ablate_yeast_glass_gradient_mc.sbatch with REWARD in
# {gc, motif, conjunction} -> workdir/rebuttal_gradient_accuracy_{gc,motif,conjunction}
# (kempner_h100). Tables 14-15 are the 2026-08-02 scale-free rescore of those runs.
#
# GLASS_END_TIME: the surviving script gained --glass_end_time (default 0.999) after the
# runs, which integrated the Euler GLASS posterior flow to t=1.0. Verified 2026-09-29: the
# stored L=50 reference gradients (motif, gc) are 0.3-0.6 % from a recomputation with 1.0 and
# 1.2-1.3 % from one with 0.999, for every probe (TF32 precision noise alone is 0.2-0.4 %);
# see tests/dna/check_gradient_mc_reference.py. Hence the default 1.0.
#
#   REWARD=gc sbatch scripts/dna/fig5_tables14_15_gradient_mc.sbatch
#   then: bash scripts/dna/fig5_tables14_15_rescore.sh <output_dir> ...
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

REWARD="${REWARD:-gc}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dna/gradient_accuracy_${REWARD}}"
GLASS_END_TIME="${GLASS_END_TIME:-1.0}"

${PY} -u -m dmfm.experiments.ablate_glass_gradient_mc \
  --output_dir "${OUTPUT_DIR}" \
  --reward "${REWARD}" \
  --lengths 50 100 200 400 \
  --t_eval 0.50 \
  --nfe_probe 32 \
  --nfe_calibration 64 \
  --calibration_samples 1024 \
  --sample_batch_size 32 \
  --nfe_value 8 \
  --glass_end_time "${GLASS_END_TIME}" \
  --n_probes 32 \
  --n_repeats 8 \
  --reference_mc 2048 \
  --mc_values 1 2 4 8 16 32 64 128 256 \
  --mc_chunk 16 \
  --z_target 1.0 \
  --reward_beta 1.0 \
  --reward_scale 1.0 \
  --thresholds 0.05 0.10 0.20 \
  --bootstrap 2000
