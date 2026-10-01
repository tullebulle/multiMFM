#!/bin/bash
#SBATCH --job-name=yeast_split_figs
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

CKPT="${CKPT:-workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt}"
DATA_PT="${DATA_PT:-data/yeast_mid50.pt}"
SPLIT_PT="${SPLIT_PT:-data/yeast_mid50_split_seed0_train65000_val8000.pt}"
REAL_SPLIT="${REAL_SPLIT:-test}"
N_SAMPLES="${N_SAMPLES:-4096}"
NFE="${NFE:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUT_DIR="${OUT_DIR:-workdir/yeast_split_sample_figures_$(date +%Y-%m-%d_%H-%M-%S)}"
FIG_DIR="${FIG_DIR:-figures/split_clean}"
SEED="${SEED:-0}"

echo "host=$(hostname)"
echo "ckpt=${CKPT}"
echo "out_dir=${OUT_DIR}"
echo "fig_dir=${FIG_DIR}"

export OMP_NUM_THREADS=4

micromamba run -n seq_h100 python scripts/make_yeast_split_sample_figures.py \
  --ckpt "${CKPT}" \
  --data_pt "${DATA_PT}" \
  --split_pt "${SPLIT_PT}" \
  --real_split "${REAL_SPLIT}" \
  --n_samples "${N_SAMPLES}" \
  --nfe "${NFE}" \
  --batch_size "${BATCH_SIZE}" \
  --out_dir "${OUT_DIR}" \
  --fig_dir "${FIG_DIR}" \
  --seed "${SEED}"
