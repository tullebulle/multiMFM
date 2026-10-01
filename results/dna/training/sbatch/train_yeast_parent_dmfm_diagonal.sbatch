#!/bin/bash
#SBATCH --job-name=yeast_parent_dmfm_diag
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

# Match the previous successful DNA dMFM recipe: a short diagonal GLASS warmup
# followed by teacher-anchored ESD consistency, while retaining diagonal loss.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

L="${LENGTH:?Set LENGTH to one of 50, 100, 200, or 400}"
case "${L}" in
  50) BATCH_SIZE="${BATCH_SIZE:-64}" ;;
  100) BATCH_SIZE="${BATCH_SIZE:-64}" ;;
  200) BATCH_SIZE="${BATCH_SIZE:-32}" ;;
  400) BATCH_SIZE="${BATCH_SIZE:-16}" ;;
  *) echo "Unsupported LENGTH=${L}" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python}"
DATA_PT="${DATA_PT:-data/yeast_parent_disjoint/yeast_parent_L${L}.pt}"
SPLIT_PT="${SPLIT_PT:-data/yeast_parent_disjoint/yeast_parent_L${L}_split_seed0.pt}"
TEACHER_DIR="${TEACHER_DIR:-workdir/yeast_parent_a_L${L}_dfm_dit_h192_b4}"
TEACHER_CKPT="${TEACHER_CKPT:-${TEACHER_DIR}/best.pt}"
TEACHER_ARGS="${TEACHER_ARGS:-${TEACHER_DIR}/args.json}"
MAX_STEPS="${MAX_STEPS:-50000}"
LR="${LR:-1e-4}"
DIAG_WARMUP_STEPS="${DIAG_WARMUP_STEPS:-1000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1}"
MFM_CHECKPOINT_METRIC="${MFM_CHECKPOINT_METRIC:-val_loss}"
RUN_NAME="${RUN_NAME:-yeast_parent_a_L${L}_dmfm_diagonal_h192_b4}"
OUTPUT_DIR="${OUTPUT_DIR:-workdir/${RUN_NAME}}"

for path in "${DATA_PT}" "${SPLIT_PT}" "${TEACHER_CKPT}" "${TEACHER_ARGS}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 2; }
done

echo "L=${L}; batch=${BATCH_SIZE}; steps=${MAX_STEPS}; diag_warmup=${DIAG_WARMUP_STEPS}; teacher=${TEACHER_CKPT}"
"${PYTHON}" -u scripts/train_dna.py \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --no-wandb \
  --dataset_type yeast \
  --yeast_data_pt "${DATA_PT}" \
  --yeast_split_pt "${SPLIT_PT}" \
  --yeast_train_split a_train \
  --yeast_val_split a_val \
  --mode gaussian \
  --gaussian_beta_schedule linear \
  --gaussian_mfm_diag_distill \
  --gaussian_adaptive_loss \
  --gaussian_adaptive_loss_c 0.01 \
  --gaussian_adaptive_loss_r 0.5 \
  --mfm_teacher_ckpt "${TEACHER_CKPT}" \
  --mfm_teacher_ckpt_hparams "${TEACHER_ARGS}" \
  --mfm_t_cond_warmup_steps 0 \
  --mfm_t_cond_power 1.0 \
  --mfm_t_cond_0_rate 0.0 \
  --mfm_consistency \
  --mfm_consistency_type esd_teacher \
  --mfm_consistency_warmup_steps "${DIAG_WARMUP_STEPS}" \
  --mfm_consistency_anneal_end_step 1 \
  --mfm_consistency_step_offset 0 \
  --mfm_diag_loss_weight 1.0 \
  --mfm_lsd_loss_weight 1.0 \
  --mfm_diag_adaptive_p 0.5 \
  --mfm_lsd_adaptive_p 1.0 \
  --mfm_val_gaps 0.01 0.05 0.1 0.25 0.5 0.75 \
  --mfm_fixed_probe_batch_size "${BATCH_SIZE}" \
  --mfm_fixed_probe_seed 12345 \
  --mfm_checkpoint_metric "${MFM_CHECKPOINT_METRIC}" \
  --model dit \
  --hidden_dim 192 \
  --transformer_heads 6 \
  --transformer_ff_mult 4 \
  --dit_blocks 4 \
  --dropout 0.0 \
  --optimizer adam \
  --weight_decay 0.0 \
  --lr "${LR}" \
  --grad_clip 1.0 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 4 \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  --max_steps "${MAX_STEPS}" \
  --print_freq 200 \
  --log_every_n_steps 50 \
  --no_tqdm
