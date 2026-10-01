#!/bin/bash
# Higher-precision MC MAE curve: 32 held-out states, 8 paired noise pools each.
#SBATCH --job-name=dmfm4_glass_mae_precision
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%6
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

SPECS=(
  "gc 50" "gc 100" "gc 200" "gc 400"
  "motif 50" "motif 100" "motif 200" "motif 400"
  "conjunction 50" "conjunction 100" "conjunction 200" "conjunction 400"
)
read -r REWARD LENGTH <<< "${SPECS[${SLURM_ARRAY_TASK_ID}]}"

case "${LENGTH}" in
  50) DMFM="workdir/yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=29-step=15000.ckpt"; MC_CHUNK=16 ;;
  100) DMFM="workdir/yeast_parent_a_L100_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=306-step=149000.ckpt"; MC_CHUNK=16 ;;
  200) DMFM="workdir/yeast_parent_a_L200_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=39-step=34000.ckpt"; MC_CHUNK=16 ;;
  400) DMFM="workdir/yeast_parent_a_L400_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=96-step=129000.ckpt"; MC_CHUNK=8 ;;
  *) echo "Unsupported length: ${LENGTH}" >&2; exit 2 ;;
esac

PYTHON="/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python"
OUT="workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/${REWARD}"

"${PYTHON}" -u scripts/ablate_dmfm_one_step_gradient_mc.py \
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
