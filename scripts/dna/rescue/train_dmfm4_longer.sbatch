#!/bin/bash
#SBATCH --job-name=dna_rescue_train
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Re-run the L=50 four-step ESD distillation, keeping checkpoints by val_loss instead of by
# val_mfm_probe_esd_gap_0.25 (docs/dmfm_deficit_rescue.md).
#
# Why: the shipped 4-step student is epoch=29-step=15000 of a 150,000-step run, selected as the
# argmin of val_mfm_probe_esd_gap_0.25. Reading that run's 150 validation records
# (backup .../workdir/yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727/val_*.csv)
# shows that metric is the ONLY one that gets worse with training -- its block median rises from
# 0.0022 (0-15k) to 0.0495 (135-150k), 22x -- while val_loss falls 0.367 -> 0.264, the diagonal
# probe MSE falls 3.4x and the gap-0.0625/0.125 ESD probes fall 10-12x. The selected step ranks
# 127th of 150 on val_loss. So the shipped student may simply be an early one, kept by a
# criterion anti-correlated with every other measure of the student.
#
# TRAINING IS OTHERWISE UNCHANGED from scripts/dna/train_dmfm_4step.sbatch: same init_ckpt (the
# diagonal student), teacher, data, split, objective, gap range (0, 0.25], lr, batch, grad clip,
# validation cadence. Only --mfm_checkpoint_metric and the output dir differ, so nothing the
# paper reports is touched; the shipped artifact stays where it is.
#
#   sbatch scripts/dna/rescue/train_dmfm4_longer.sbatch                  # 60k steps
#   MAX_STEPS=150000 sbatch scripts/dna/rescue/train_dmfm4_longer.sbatch # the full original run
# Resubmitting resumes from last.ckpt, so a preempted job continues where it stopped.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

L="${LENGTH:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_PT="data/dna/yeast_parent_disjoint/yeast_parent_L${L}.pt"
SPLIT_PT="data/dna/yeast_parent_disjoint/yeast_parent_L${L}_split_seed0.pt"
TEACHER_CKPT="checkpoints/dna/base/L${L}/best.pt"
TEACHER_ARGS="checkpoints/dna/base/L${L}/args.json"
INIT_CKPT="${INIT_CKPT:-$(${PY} -c "from dmfm import paths; print(paths.dmfm_ckpt(${L}))")}"
MAX_STEPS="${MAX_STEPS:-60000}"
METRIC="${MFM_CHECKPOINT_METRIC:-val_loss}"
RUN_NAME="${RUN_NAME:-rescue_L${L}_dmfm_4step_esd_upto025_by_${METRIC}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dna/deficit_rescue/${RUN_NAME}}"

for path in "${DATA_PT}" "${SPLIT_PT}" "${TEACHER_CKPT}" "${TEACHER_ARGS}" "${INIT_CKPT}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done
RESUME=()
if [[ -f "${OUTPUT_DIR}/last.ckpt" ]]; then RESUME=(--ckpt "${OUTPUT_DIR}/last.ckpt"); fi

echo "L=${L} steps=${MAX_STEPS} metric=${METRIC} init=${INIT_CKPT} resume=${RESUME[*]:-none}"
${PY} -u -m dmfm.experiments.train_dmfm \
  --run_name "${RUN_NAME}" --output_dir "${OUTPUT_DIR}" "${RESUME[@]}" \
  --init_ckpt "${INIT_CKPT}" --no-wandb --dataset_type yeast \
  --yeast_data_pt "${DATA_PT}" --yeast_split_pt "${SPLIT_PT}" \
  --yeast_train_split a_train --yeast_val_split a_val \
  --mode gaussian --gaussian_beta_schedule linear --gaussian_mfm_diag_distill \
  --gaussian_adaptive_loss --gaussian_adaptive_loss_c 0.01 --gaussian_adaptive_loss_r 0.5 \
  --mfm_teacher_ckpt "${TEACHER_CKPT}" --mfm_teacher_ckpt_hparams "${TEACHER_ARGS}" \
  --mfm_t_cond_warmup_steps 0 --mfm_t_cond_power 1.0 --mfm_t_cond_0_rate 0.0 \
  --mfm_consistency --mfm_consistency_type esd_teacher --mfm_consistency_warmup_steps 0 \
  --mfm_consistency_gap_mode long --mfm_consistency_min_gap 0.0 --mfm_consistency_max_gap 0.25 \
  --mfm_diag_loss_weight 1.0 --mfm_lsd_loss_weight 1.0 \
  --mfm_diag_adaptive_p 0.5 --mfm_lsd_adaptive_p 1.0 \
  --mfm_val_gaps 0.0625 0.125 0.25 \
  --mfm_fixed_probe_batch_size "${BATCH_SIZE}" --mfm_fixed_probe_seed 12345 \
  --mfm_checkpoint_metric "${METRIC}" \
  --model dit --hidden_dim 192 --transformer_heads 6 --transformer_ff_mult 4 --dit_blocks 4 \
  --dropout 0.0 --optimizer adam --weight_decay 0.0 --lr "${LR:-2e-5}" --grad_clip 1.0 \
  --batch_size "${BATCH_SIZE}" --num_workers 4 \
  --limit_val_batches 1 --val_check_interval 1000 --max_steps "${MAX_STEPS}" \
  --print_freq 200 --log_every_n_steps 50 --no_tqdm
