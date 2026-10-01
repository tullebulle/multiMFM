#!/bin/bash
#SBATCH --job-name=dna_dmfm_l400_ft
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# L=400 diagonal-only finetune: the 750 extra steps (batch 8, consistency OFF,
# resumed from the L400 diagonal+ESD run at step 45000) that produced the
# shipped L400 dMFM, checkpoints/dna/dmfm/L400/epoch=33-step=45750-v1.ckpt.
# Original: run interactively on 2026-07-26 (no sbatch file survives); the flags
# below are checkpoints/dna/dmfm/L400/args.json verbatim. It was launched twice,
# which is where the "-v1" duplicates come from; 45750 and 45750-v1 have
# bitwise-identical weights.
#
#   sbatch scripts/dna/train_dmfm_l400_finetune.sbatch
# CKPT defaults to the shipped resume point checkpoints/dna/dmfm_lineage/L400_diagonal/
# epoch=33-step=45000.ckpt (the L400 diagonal+ESD run's step-45000 file, byte-identical to the
# original); set CKPT to use the output of your own scripts/dna/train_dmfm.sbatch LENGTH=400 run.
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

L=400
CKPT="${CKPT:-checkpoints/dna/dmfm_lineage/L400_diagonal/epoch=33-step=45000.ckpt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing ${CKPT}: python scripts/manifest.py verify --group dna --fetch-missing," >&2
  echo "or set CKPT to the step-45000 checkpoint of scripts/dna/train_dmfm.sbatch LENGTH=400." >&2
  exit 2
fi
RUN_NAME="${RUN_NAME:-yeast_parent_a_L400_dmfm_diagonly_finetune_20260726}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dna/${RUN_NAME}}"

${PY} -u -m dmfm.experiments.train_dmfm \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --ckpt "${CKPT}" \
  --no-wandb \
  --dataset_type yeast \
  --yeast_data_pt "data/dna/yeast_parent_disjoint/yeast_parent_L${L}.pt" \
  --yeast_split_pt "data/dna/yeast_parent_disjoint/yeast_parent_L${L}_split_seed0.pt" \
  --yeast_train_split a_train \
  --yeast_val_split a_val \
  --mode gaussian \
  --gaussian_beta_schedule linear \
  --gaussian_mfm_diag_distill \
  --gaussian_adaptive_loss \
  --gaussian_adaptive_loss_c 0.01 \
  --gaussian_adaptive_loss_r 0.5 \
  --mfm_teacher_ckpt "checkpoints/dna/base/L${L}/best.pt" \
  --mfm_teacher_ckpt_hparams "checkpoints/dna/base/L${L}/args.json" \
  --mfm_t_cond_warmup_steps 0 \
  --mfm_t_cond_power 1.0 \
  --mfm_t_cond_0_rate 0.0 \
  --mfm_diag_loss_weight 1.0 \
  --mfm_lsd_loss_weight 0.0 \
  --mfm_diag_adaptive_p 0.5 \
  --mfm_lsd_adaptive_p 1.0 \
  --mfm_val_gaps 0.01 0.1 0.5 \
  --mfm_fixed_probe_batch_size 128 \
  --mfm_fixed_probe_seed 12345 \
  --mfm_checkpoint_metric val_mfm_probe_diag_mse \
  --model dit \
  --hidden_dim 192 \
  --transformer_heads 6 \
  --transformer_ff_mult 4 \
  --dit_blocks 4 \
  --dropout 0.0 \
  --optimizer adam \
  --weight_decay 0.0 \
  --lr 1e-4 \
  --grad_clip 1.0 \
  --batch_size 8 \
  --num_workers 0 \
  --limit_val_batches 1 \
  --val_check_interval 250 \
  --max_steps 46000 \
  --print_freq 100 \
  --log_every_n_steps 50 \
  --no_tqdm
