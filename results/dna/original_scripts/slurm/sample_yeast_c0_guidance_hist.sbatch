#!/bin/bash
#SBATCH --job-name=yeast_c0_hist
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err


cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

CKPT="${CKPT:-workdir/yeast_mid50_dfm_dit_h192_b4_lr5e-5_2026-05-02_13-06-17/last.pt}"
C0_CKPT="${C0_CKPT:-dinko/C0free_torch.pt}"
RUN_NAME="${RUN_NAME:-yeast_c0_guidance_hist}"
OUTPUT_DIR="${OUTPUT_DIR:-workdir/${RUN_NAME}_$(date +%Y-%m-%d_%H-%M-%S)}"

N_SAMPLES="${N_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-0}"

TARGET_C0="${TARGET_C0:-1.00}"
TARGET_C0_SECOND="${TARGET_C0_SECOND:-}"
REWARD_SIGMA="${REWARD_SIGMA:-0.15}"
REWARD_SCALE="${REWARD_SCALE:-0.5}"
MC="${MC:-128}"
NFE_TRAJ="${NFE_TRAJ:-128}"
NFE_VALUE="${NFE_VALUE:-12}"
T_MAX="${T_MAX:-0.95}"

GUIDE_T_START="${GUIDE_T_START:-0.50}"
GUIDE_T_END="${GUIDE_T_END:-0.95}"
GUIDANCE_FRAC="${GUIDANCE_FRAC:-8.0}"
COEFF_CAP="${COEFF_CAP:-10.0}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"
TAPER="${TAPER:-false}"

TAPER_FLAG=""
if [[ "${TAPER}" == "true" || "${TAPER}" == "1" ]]; then
  TAPER_FLAG="--taper"
fi

SECOND_TARGET_FLAG=()
if [[ -n "${TARGET_C0_SECOND}" ]]; then
  SECOND_TARGET_FLAG=(--target_c0_second "${TARGET_C0_SECOND}")
fi

echo "Running on host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Samples: ${N_SAMPLES}"
echo "Targets: ${TARGET_C0}${TARGET_C0_SECOND:+, ${TARGET_C0_SECOND}}"

micromamba run -n seq_h100 python -c "import torch; print('cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(), 'name0', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))"

export OMP_NUM_THREADS=4

micromamba run -n seq_h100 python scripts/sample_yeast_c0_guidance_hist.py \
  --ckpt "${CKPT}" \
  --c0_ckpt "${C0_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --target_c0 "${TARGET_C0}" \
  "${SECOND_TARGET_FLAG[@]}" \
  --reward_sigma "${REWARD_SIGMA}" \
  --reward_scale "${REWARD_SCALE}" \
  --mc "${MC}" \
  --nfe_traj "${NFE_TRAJ}" \
  --nfe_value "${NFE_VALUE}" \
  --t_max "${T_MAX}" \
  --guide_t_start "${GUIDE_T_START}" \
  --guide_t_end "${GUIDE_T_END}" \
  --guidance_frac "${GUIDANCE_FRAC}" \
  --coeff_cap "${COEFF_CAP}" \
  --grad_clip "${GRAD_CLIP}" \
  ${TAPER_FLAG}
