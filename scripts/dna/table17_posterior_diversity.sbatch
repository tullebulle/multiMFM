#!/bin/bash
#SBATCH --job-name=dna_table17
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Table 17: reward-free conditional posterior diversity, dMFM one-step flow map
# vs a 100-step GLASS Euler reference, L=50..400. The dMFM checkpoints are the
# frozen selection in dmfm.paths (checkpoints/dna/dmfm/L*/).
# Original: slurm/eval_parent_posterior_diversity.sbatch, job 35378081
# (kempner_h100) -> workdir/parent_posterior_diversity_mapselected_flowmap1_glass100_20260726
# (launched with that OUT_DIR; DMFM_STEPS=1 DMFM_SAMPLER=flow_map GLASS_STEPS=100).
# Arguments identical.
#
#   sbatch scripts/dna/table17_posterior_diversity.sbatch
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

OUT_DIR="${OUT_DIR:-outputs/dna/posterior_diversity}"
DMFM_STEPS="${DMFM_STEPS:-1}"
DMFM_SAMPLER="${DMFM_SAMPLER:-flow_map}"
GLASS_STEPS="${GLASS_STEPS:-100}"

${PY} -u -m dmfm.experiments.eval_posterior_diversity \
  --lengths 50 100 200 400 \
  --times 0.0 0.25 0.5 0.75 0.9 0.99 1.0 \
  --n_sources 32 \
  --n_futures 16 \
  --source_batch 4 \
  --dmfm_steps "${DMFM_STEPS}" \
  --dmfm_sampler "${DMFM_SAMPLER}" \
  --glass_steps "${GLASS_STEPS}" \
  --bootstrap 2000 \
  --seed 20260730 \
  --out_dir "${OUT_DIR}"
