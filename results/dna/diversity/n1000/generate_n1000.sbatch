#!/bin/bash
#SBATCH --job-name=dfm_n1000
#SBATCH --partition=kempner_h100
#SBATCH -A kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/n/netscratch/kozinsky_lab/Lab/uunneberg/scratch_dfm_recon_20260925/n1000_logs/%x-%j.out
#SBATCH --error=/n/netscratch/kozinsky_lab/Lab/uunneberg/scratch_dfm_recon_20260925/n1000_logs/%x-%j.err

set -euo pipefail

T="${1:?target_c0 required}"
TAG="${2:?tag required}"

W="/n/netscratch/kozinsky_lab/Lab/uunneberg/scratch_dfm_recon_20260925"
REPO=/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dirichlet-flow-matching
PYTHON=/n/home04/uunneberg/micromamba/envs/seq_h100/bin/python

export PYTHONNOUSERSITE=1
export PYTHONPATH="${W}:${W}/pydeps"
export LD_LIBRARY_PATH=/n/home04/uunneberg/micromamba/envs/seq_h100/lib:${LD_LIBRARY_PATH:-}

cd "${REPO}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"${PYTHON}" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'dev',torch.cuda.get_device_name(0))"

OUT="workdir/yeast_parent_glass_n1000_target${TAG}_g8.0_seed0"
echo "===== target=${T} -> ${OUT}"
/usr/bin/time -f "WALL_SECONDS generate target=${T} %e" "${PYTHON}" -u "${W}/scripts/sample_yeast_c0_guidance_hist.py" \
  --ckpt workdir/yeast_parent_a_L50_dfm_dit_h192_b4/best.pt \
  --c0_ckpt workdir/yeast_parent_c0_guide_2026-07-26/best_state.pt \
  --output_dir "${OUT}" \
  --seed 0 \
  --n_samples 1000 \
  --batch_size 8 \
  --target_c0 "${T}" \
  --reward_sigma 0.15 \
  --reward_scale 0.5 \
  --mc 8 \
  --mc_chunk 8 \
  --nfe_traj 64 \
  --nfe_value 4 \
  --t_max 0.95 \
  --guide_t_start 0.50 \
  --guide_t_end 0.95 \
  --guidance_frac 8.0 \
  --coeff_cap 10.0 \
  --grad_clip 10.0

echo "===== GENERATE_DONE ${OUT}"

/usr/bin/time -f "WALL_SECONDS oracle target=${T} %e" "${PYTHON}" -u scripts/score_yeast_c0_oracle.py \
  --oracle_ckpt workdir/yeast_parent_c0_oracle_2026-07-26/best_state.pt \
  --oracle_model_type park_cnn \
  --run_glob "${OUT}/sample_scores.csv" \
  --out_dir "${OUT}/oracle" \
  --batch_size 512 \
  --bootstrap 2000 \
  --seed 0 \
  --reverse_complement_average

echo "===== ORACLE_DONE ${OUT}"
echo "N1000_JOB_DONE target=${T}"
