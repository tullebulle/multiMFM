#!/bin/bash
#SBATCH --job-name=dna_table18
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Table 18: raw value-gradient MAE of the 4-step dMFM vs a GLASS-128 (RK4, 50
# steps) reference, for 3 rewards x 4 lengths (one array task each).
# Original: slurm/evaluate_dmfm4_glass_mae_precision.sbatch, job array 35642548
# -> workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/<reward>/.
# Arguments identical; the 4-step checkpoints come from dmfm.paths
# (checkpoints/dna/dmfm_4step/L*/), which are exactly the files the original
# launcher hard-coded.
#
# NOTE: App. F.6 / Table 18 text says t=0.3; every run used --t_eval 0.50.
#
# The Table 18 numbers are the per-coordinate MAE files written afterwards by
# dmfm.experiments.rescore_gradient_pairs_mae (original: scripts/rescore_gradient_pairs_mae.py,
# run by hand on 2026-07-27 23:33 with its defaults --bootstrap 10000 --seed 20260802):
# <OUT>/L<L>/gradient_mae_{per_condition,summary}.csv and <OUT>/gradient_mae_summary_all_lengths.csv.
# Each task rescores its own length at the end; the all-lengths file is rebuilt from the
# per-length summaries present, so it is complete once all 4 lengths of a reward are done.
#
#   sbatch scripts/dna/table18_dmfm4_gradient_mc.sbatch
#   REWARD=gc LENGTH=50 sbatch --array=0 scripts/dna/table18_dmfm4_gradient_mc.sbatch  # single cell
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

SPECS=(
  "gc 50" "gc 100" "gc 200" "gc 400"
  "motif 50" "motif 100" "motif 200" "motif 400"
  "conjunction 50" "conjunction 100" "conjunction 200" "conjunction 400"
)
if [[ -n "${REWARD:-}" && -n "${LENGTH:-}" ]]; then
  read -r REWARD LENGTH <<< "${REWARD} ${LENGTH}"
else
  read -r REWARD LENGTH <<< "${SPECS[${SLURM_ARRAY_TASK_ID:-0}]}"
fi

case "${LENGTH}" in
  50|100|200) MC_CHUNK="${MC_CHUNK:-16}" ;;
  400)        MC_CHUNK="${MC_CHUNK:-8}" ;;
  *) echo "Unsupported length: ${LENGTH}" >&2; exit 2 ;;
esac
DMFM="${DMFM:-$(${PY} -c "from dmfm import paths; print(paths.dmfm4_ckpt(${LENGTH}))")}"
OUT="${OUT:-outputs/dna/dmfm4_glass128_mae_precision/${REWARD}}"

${PY} -u -m dmfm.experiments.ablate_dmfm_one_step_gradient_mc \
  --output_dir "${OUT}" \
  --lengths "${LENGTH}" \
  --dmfm_ckpt "${LENGTH}=${DMFM}" \
  --reward "${REWARD}" \
  --reward_objective target \
  --seed 20260802 \
  --t_eval 0.50 \
  --nfe_calibration 32 \
  --calibration_samples 128 \
  --sample_batch_size 16 \
  --n_probes 32 \
  --n_repeats 8 \
  --reference_mc 128 \
  --reference_sampler glass \
  --reference_glass_steps 50 \
  --reference_glass_solver rk4 \
  --glass_end_time 0.999 \
  --dmfm_sampler flow_map \
  --dmfm_steps 4 \
  --dmfm_end_time 0.999 \
  --mc_values 1 2 4 8 \
  --nested_mc_pools \
  --mc_chunk "${MC_CHUNK}" \
  --z_target 1.0 \
  --reward_beta 1.0 \
  --reward_scale 1.0 \
  --thresholds 0.05 0.10 0.20 \
  --bootstrap 500

# Table 18 itself: per-coordinate gradient MAE (original rescore defaults).
${PY} -u -m dmfm.experiments.rescore_gradient_pairs_mae \
  --input-dir "${OUT}" \
  --lengths "${LENGTH}" \
  --bootstrap 10000 \
  --seed 20260802
