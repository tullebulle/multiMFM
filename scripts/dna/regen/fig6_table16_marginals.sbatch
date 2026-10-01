#!/bin/bash
#SBATCH --job-name=dna_regen_fig6
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Fig 6 + Table 16 (REGENERATED; no surviving producer, see src/dmfm/regen/base_marginals.py).
# Draws N unguided samples from the parent-disjoint base DFM and writes the position-wise /
# global base frequencies, the guide-CNN cyclizability distributions and the Table 16
# Pearson correlations.
#
#   sbatch scripts/dna/regen/fig6_table16_marginals.sbatch                # 2-shard array
#   N=2000 NUM_SHARDS=1 sbatch --array=0 --partition=gpu_test scripts/dna/regen/fig6_table16_marginals.sbatch
#   bash scripts/dna/regen/collect.sh                                     # reduce + notebook
#
# Each array task writes outputs/dna/regen/fig6/generated_tokens_shard<k>of<K>.npz and skips
# its own output if it already exists, so a preempted shard reruns alone (--requeue safe).
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

N="${N:-10000}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-500}"
NFE="${NFE:-128}"      # June recipe (make_yeast_split_sample_figures.py)
T_MAX="${T_MAX:-1.0}"
SEED="${SEED:-0}"

# Record the job for results/dna/regen/PROVENANCE.md (one TSV line per array task).
mkdir -p outputs/dna/regen
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_NAME:-local}" \
  "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}" "${SLURM_ARRAY_TASK_ID:-}" "n=${N} shards=${NUM_SHARDS} nfe=${NFE} seed=${SEED} (Fig 6 / Table 16)" \
  >> outputs/dna/regen/jobs.tsv

${PY} -u -m dmfm.regen.base_marginals \
  --n "${N}" \
  --num_shards "${NUM_SHARDS}" \
  --shard "${SHARD}" \
  --batch_size "${BATCH_SIZE}" \
  --nfe "${NFE}" \
  --t_max "${T_MAX}" \
  --seed "${SEED}" \
  --skip_existing

# The last shard to finish also reduces (harmless if it races: reduce is idempotent and
# refuses unless all shard files are present).
if ${PY} -u -m dmfm.regen.base_marginals --n "${N}" --num_shards "${NUM_SHARDS}" --seed "${SEED}" --reduce; then
  echo "reduced Fig 6 / Table 16 into results/dna/regen/fig6_table16"
else
  echo "shards still missing; run 'bash scripts/dna/regen/collect.sh' once the array finishes"
fi
