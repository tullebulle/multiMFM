set -euo pipefail
REPO=/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco
PYTHON="${REPO}/.envs/tabasco-gpu/bin/python"; cd "${REPO}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME="${REPO}/cache/huggingface" HF_DATASETS_CACHE="${REPO}/cache/huggingface/datasets" HF_HUB_CACHE="${REPO}/cache/huggingface/hub" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${REPO}/cache/matplotlib" XDG_CACHE_HOME="${REPO}/cache/xdg" TORCH_HOME="${REPO}/cache/torch" TMPDIR="${REPO}/cache/tmp"
NV="${REPO}/.envs/tabasco-gpu/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV}/cublas/lib:${NV}/cuda_cupti/lib:${NV}/cuda_nvrtc/lib:${NV}/cuda_runtime/lib:${NV}/cudnn/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nccl/lib:${NV}/nvjitlink/lib:${NV}/nvtx/lib:${LD_LIBRARY_PATH:-}"
"${PYTHON}" codex_scratch/memory/best_of_lookahead.py \
  --checkpoint outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt \
  --mfm-checkpoint outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt \
  --mfm-key student --mfm-diagonal --device cuda --seed 8 --target-seed 2026 --num-samples 100 \
  --sample-steps 128 --mu 1.2 --reward-scale 0.3 --value-samples 32 --value-glass-steps 4 \
  --guide-every 1 --property-name alpha --select-min-t 0.5 --pb-candidates 0 --pb-workers 16 \
  --output-dir codex_scratch/memory/search/smoke2_alpha
