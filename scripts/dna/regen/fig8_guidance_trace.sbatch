#!/bin/bash
#SBATCH --job-name=dna_regen_fig8
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Fig 8 (REGENERATED; producer unknown, see src/dmfm/regen/guidance_trace.py). Records the
# three published panels along one guided and one unguided trajectory from the same noise:
# value-gradient norm, GLASS value, and the C0 score of the argmax of the model's predicted
# endpoint, with target C0 = 0.30 (the published figure's dotted line).
#
#   sbatch scripts/dna/regen/fig8_guidance_trace.sbatch
#   MC=8 NFE_TRAJ=32 sbatch --partition=gpu_test scripts/dna/regen/fig8_guidance_trace.sbatch   # smoke
#
# Writes results/dna/regen/fig8/{trace.csv,metadata.json}; skips if trace.csv exists.
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

# Record the job for results/dna/regen/PROVENANCE.md (one TSV line per array task).
mkdir -p outputs/dna/regen
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_NAME:-local}" \
  "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}" "${SLURM_ARRAY_TASK_ID:-}" "trace mc=${MC:-128} nfe_traj=${NFE_TRAJ:-128} nfe_value=${NFE_VALUE:-12} target=${TARGET_C0:-0.30} (Fig 8)" \
  >> outputs/dna/regen/jobs.tsv

${PY} -u -m dmfm.regen.guidance_trace \
  --sample_ids ${SAMPLE_IDS:-0 1 2 3} \
  --target_c0 "${TARGET_C0:-0.30}" \
  --seed "${SEED:-0}" \
  --mc "${MC:-128}" \
  --mc_chunk "${MC_CHUNK:-128}" \
  --nfe_traj "${NFE_TRAJ:-128}" \
  --nfe_value "${NFE_VALUE:-12}" \
  --t_max 0.95 \
  --guide_t_start "${GUIDE_T_START:-0.0}" \
  --guide_t_end "${GUIDE_T_END:-0.95}" \
  --guidance_frac "${GUIDANCE_FRAC:-1.0}" \
  --coeff_cap 10.0 \
  --grad_clip 10.0 \
  --out_dir "${OUT_DIR:-results/dna/regen/fig8}" \
  --skip_existing
