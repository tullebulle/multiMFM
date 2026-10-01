#!/bin/bash
#SBATCH --job-name=yeast_value_mc
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/slurm-%x-%j.out
#SBATCH --error=slurm/logs/slurm-%x-%j.err

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

TEACHER_CKPT="${TEACHER_CKPT:-workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt}"
STUDENT_CKPT="${STUDENT_CKPT:-workdir/yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35/epoch=622-step=632000.ckpt}"
OUT_DIR="${OUT_DIR:-workdir/yeast_split_value_mc_$(date +%Y-%m-%d_%H-%M-%S)}"
FIG_DIR="${FIG_DIR:-figures/split_clean}"

N_CONDITIONS="${N_CONDITIONS:-8}"
SDE_REF_N="${SDE_REF_N:-512}"
SDE_STEPS="${SDE_STEPS:-1024}"
N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256}"

echo "host=$(hostname)"
echo "teacher=${TEACHER_CKPT}"
echo "student=${STUDENT_CKPT}"
echo "out_dir=${OUT_DIR}"

export OMP_NUM_THREADS=4

micromamba run -n seq_h100 python scripts/benchmark_yeast_split_value_mc.py \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --student_ckpt "${STUDENT_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --fig_dir "${FIG_DIR}" \
  --n_conditions "${N_CONDITIONS}" \
  --sde_ref_n "${SDE_REF_N}" \
  --sde_steps "${SDE_STEPS}" \
  --n_list ${N_LIST}
