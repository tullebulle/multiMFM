#!/bin/bash
#SBATCH --job-name=dna_regen_fig6_cpu
#SBATCH --partition=serial_requeue
#SBATCH --account=albergo_lab
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# CPU twin of fig6_table16_marginals.sbatch (Fig 6 + Table 16), authorised 2026-09-29 to
# get past a saturated kempner_requeue. serial_requeue denies kempner_albergo_lab, hence
# --account=albergo_lab. Writes to its OWN sample directory (outputs/dna/regen/fig6_cpu)
# so CPU and GPU runs are never mixed, and every shard records its device, torch version,
# Slurm account/partition/node and decode-margin statistics inside its .npz.
#
#   sbatch scripts/dna/regen/fig6_table16_marginals_cpu.sbatch
#   bash scripts/dna/regen/collect.sh          # auto-detects whichever run is complete
#
# Why CPU is safe here, and how the run proves it rather than asserting it:
#   * The initial noise is drawn by a seeded CPU generator and only then moved to the
#     sampling device (dmfm.regen.base_marginals._noise), so a CPU run and a GPU run share
#     IDENTICAL randomness. They can differ only in float arithmetic (fp32 here; fp32/TF32
#     on GPU).
#   * Sampling is deterministic given that noise, and the decoded token is argmax of the
#     final state. Each shard reports the top-1 minus top-2 margin of every token: the flow
#     is integrated to t_max = 1.0, which drives the state onto a simplex vertex, so the
#     margin is ~1.0 against a float32 noise floor of ~1e-7. No token's decode is close
#     enough to be flipped by a change of device.
#   * Fig 6 and Table 16 are distributional summaries over thousands of samples
#     (position-wise base frequencies, global marginals, a C0 KDE and Pearson r), the case
#     where a device change matters least even before the margin argument.
#
# Cost, measured on this cluster: ~4.9 ms per (sample, integration step) on 4 threads for
# the 2.9M-parameter L=50 base DFM, i.e. ~105 min for n=10,000 at nfe=128 in one process.
# 20 shards of 500 samples => ~6 min per task.
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

N="${N:-10000}"
NUM_SHARDS="${NUM_SHARDS:-20}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-125}"
NFE="${NFE:-128}"        # June recipe (make_yeast_split_sample_figures.py)
T_MAX="${T_MAX:-1.0}"
SEED="${SEED:-0}"
SAMPLE_DIR="${SAMPLE_DIR:-outputs/dna/regen/fig6_cpu}"

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"

# Record the job for results/dna/regen/PROVENANCE.md (one TSV line per array task).
mkdir -p outputs/dna/regen
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_NAME:-local}" \
  "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}" "${SLURM_ARRAY_TASK_ID:-}" \
  "n=${N} shards=${NUM_SHARDS} nfe=${NFE} seed=${SEED} CPU/${THREADS}thr account=${SLURM_JOB_ACCOUNT:-?} partition=${SLURM_JOB_PARTITION:-?} (Fig 6 / Table 16)" \
  >> outputs/dna/regen/jobs.tsv

${PY} -u -m dmfm.regen.base_marginals \
  --n "${N}" \
  --num_shards "${NUM_SHARDS}" \
  --shard "${SHARD}" \
  --batch_size "${BATCH_SIZE}" \
  --nfe "${NFE}" \
  --t_max "${T_MAX}" \
  --seed "${SEED}" \
  --device cpu \
  --sample_dir "${SAMPLE_DIR}" \
  --skip_existing

# The last shard to finish also reduces (idempotent; refuses until all shards are present).
if ${PY} -u -m dmfm.regen.base_marginals --n "${N}" --num_shards "${NUM_SHARDS}" --seed "${SEED}" \
      --device cpu --sample_dir "${SAMPLE_DIR}" --reduce; then
  echo "reduced Fig 6 / Table 16 into results/dna/regen/fig6_table16"
else
  echo "shards still missing; run 'bash scripts/dna/regen/collect.sh' once the array finishes"
fi
