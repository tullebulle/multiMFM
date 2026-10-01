#!/bin/bash
#SBATCH --job-name=dna_prepare_data
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CPU only. The kempner_albergo_lab account is valid only on the Kempner GPU partitions,
# so this job uses albergo_lab on serial_requeue (the original ran on serial_requeue).
#
# Parent-disjoint yeast windows and splits (Table 10; input to everything else).
# Original: slurm/prepare_yeast_parent_disjoint_data.sbatch, job 35116368 (rerun
# 35241776 added the C0 labels to the L=50 file). Needs the raw loop-seq table
# data/dna/raw/yeast_sequences.txt, which is NOT redistributed -- see
# docs/provenance/dna/README.md for the Basu et al. (2021) download and the
# sha256 recorded in parent_split_seed0.json.
#
#   sbatch scripts/dna/prepare_data.sbatch
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

INPUT="${INPUT:-data/dna/raw/yeast_sequences.txt}"
OUT_DIR="${OUT_DIR:-data/dna/yeast_parent_disjoint}"

${PY} -u -m dmfm.experiments.prepare_yeast_parent_splits \
  --input "${INPUT}" \
  --out_dir "${OUT_DIR}" \
  --seed 0 \
  --lengths 50 100 200 400
