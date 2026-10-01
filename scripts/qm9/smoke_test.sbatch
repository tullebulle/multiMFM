#!/usr/bin/env bash
#SBATCH --job-name=qm9_smoke
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Small-n GPU checks of the QM9 pipeline (< 1 h on one GPU). Submit from the repo root:
#
#   sbatch scripts/qm9/smoke_test.sbatch sampling   # non-SS + SS at small n
#   sbatch scripts/qm9/smoke_test.sbatch training   # a few updates of both trainers
#
# sampling writes to outputs/qm9/smoke/:
#   rungen_alpha_n32     the original determinism-check config (codex_scratch/check/
#                        run_gen.sh: alpha, mu=0.45, rs=0.3, n=32, no PB); stored
#                        references: results/qm9/verification/run_gen_gpu_{A,B}
#   multimfm_alpha_n100  scripts/qm9/run_multimfm.sh alpha 100 (paper config, first
#   multimfm_cv_n100     100 of the n=1000 target set) -> compare with results/qm9/multimfm
#   ss_alpha_n100        scripts/run_steer_search.sh alpha 100 -> compare with the four
#                        n=100 replicates in results/qm9/verification/ss_alpha_n100_*
# The comparison is done in notebooks/07_qm9_property_guidance.ipynb.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
MODE="${1:-sampling}"
SMOKE="${REPO_ROOT}/outputs/qm9/smoke"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null || echo none) mode=${MODE}"

if [ "${MODE}" = "sampling" ]; then
  # shellcheck source=/dev/null
  source scripts/env.sh
  echo "=== [1] run_gen.sh determinism config (n=32, no PB) ==="
  python -m multimfm.property_targets \
    --device cuda --seed 8 --target-seed 2026 --num-samples 32 \
    --sample-steps 128 --mu 0.45 --reward-scale 0.3 \
    --value-samples 32 --value-batch-size 0 --value-glass-steps 4 \
    --value-sampler mfm --mfm-key student --mfm-diagonal \
    --guide-every 1 --guide-min-t 0.05 --guide-max-t 0.95 \
    --eps 1e-6 --soft-atom-temperature 0.25 \
    --property-name alpha --partition train_flow --hist-bins 100 \
    --guide-atomics --output-dir "${SMOKE}/rungen_alpha_n32"
  echo "=== [2] multiMFM alpha n=100 ==="
  OUTPUT_DIR="${SMOKE}/multimfm_alpha_n100" bash scripts/qm9/run_multimfm.sh alpha 100
  echo "=== [3] multiMFM cv n=100 ==="
  OUTPUT_DIR="${SMOKE}/multimfm_cv_n100" bash scripts/qm9/run_multimfm.sh cv 100
  echo "=== [4] multiMFM-SS alpha n=100 ==="
  OUTPUT_DIR="${SMOKE}/ss_alpha_n100" PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}" \
    bash scripts/run_steer_search.sh alpha 100
elif [ "${MODE}" = "training" ]; then
  echo "=== [1] base trainer, 200 steps, no PB curve ==="
  TRAIN_STEPS=200 PB_EVERY=0 OUT="${SMOKE}/train_base_200" bash scripts/qm9/train_base.sbatch
  echo "=== [2] MFM trainer (final-phase recipe), 20 updates from the SHIPPED student ==="
  INIT_CKPT="${REPO_ROOT}/checkpoints/mfm_student/student_step_1000.pt" \
    MAX_UPDATES=20 EVAL_EVERY=20 N_EVAL=8 OUT="${SMOKE}/train_mfm_20" \
    bash scripts/qm9/train_mfm.sbatch
else
  echo "unknown mode ${MODE} (sampling|training)" >&2; exit 1
fi
echo "=== done ==="
