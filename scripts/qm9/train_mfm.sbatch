#!/usr/bin/env bash
#SBATCH --job-name=qm9_train_mfm
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Re-run the FINAL training phase of the QM9 meta flow map (Table 20, QM9 column).
#
#   INIT_CKPT=/path/to/student_step_3000.pt sbatch scripts/qm9/train_mfm.sbatch
#
# Replays the arguments stored in checkpoints/mfm_student/student_step_1000.pt
# (field "args"; original run outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/
# stage1_20260612-112012, Slurm job 21926337): stage 1 = diagonal GLASS
# distillation on teacher-generated conditioning states (128-step unguided teacher
# trajectories, 16-step GLASS posterior targets), with direction and
# condition-sensitivity terms. No ESD term is active in this phase. Cosine LR
# horizon 4000 steps; the shipped checkpoint is the EMA student at step 1000, so by
# default the run stops after 1000 updates (MAX_UPDATES; the schedule is unchanged).
#
# NOT END-TO-END REPRODUCIBLE: this phase was initialised from
#   outputs/mfm_diag_rollout8_forced_20260611_long_spr4/stage4_20260611-171509/student_step_3000.pt
# (stage-4 forced 8-step diagonal-rollout training, job 21603965), which was itself
# the end of an earlier chain of QM9 MFM runs. All of those checkpoints were purged
# and are not in the backup, so INIT_CKPT must be supplied. For a pipeline smoke test
# only, INIT_CKPT=checkpoints/mfm_student/student_step_1000.pt continues training
# from the shipped student (this is not a reproduction).
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs
# shellcheck source=/dev/null
source scripts/env.sh

INIT_CKPT="${INIT_CKPT:-}"
if [ -z "${INIT_CKPT}" ] || [ ! -f "${INIT_CKPT}" ]; then
  echo "INIT_CKPT is not set or missing (${INIT_CKPT:-unset})." >&2
  echo "The original init checkpoint (spr4 stage4 step 3000) was purged; see the header." >&2
  exit 2
fi
TEACHER_CKPT="${TEACHER_CKPT:-${REPO_ROOT}/checkpoints/base_flow/model_step_100000.pt}"
MAX_UPDATES="${MAX_UPDATES:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
N_EVAL="${N_EVAL:-32}"
OUT="${OUT:-${REPO_ROOT}/outputs/qm9/mfm_student/mfm_gencond_diag_stage1}"

python -m multimfm.train_mfm_student \
  --device cuda --teacher-checkpoint "${TEACHER_CKPT}" \
  --partition train_flow --limit-data 40000 --max-len 9 \
  --batch-size 4 --steps 4000 --max-updates "${MAX_UPDATES}" --lr 1e-6 --t-max 0.95 \
  --diag-loss-type mse --diag-adaptive-p 0.5 --diag-adaptive-c 0.01 \
  --diag-direction-weight 5.0 \
  --diag-condition-jvp-weight 0.1 --diag-condition-jvp-direction-weight 0.25 \
  --diag-condition-jvp-eps 0.03 \
  --stage 1 --init-ckpt "${INIT_CKPT}" \
  --esd-weight 1.0 --diag-weight 1.0 --coord-loss-weight 8.0 --atom-loss-weight 0.5 \
  --esd-target-clip 0.0 --esd-delta-clip 0.0 \
  --esd-loss-type adaptive --esd-adaptive-p 0.5 --esd-adaptive-c 0.01 \
  --warmup 200 --anneal-frac 0.5 --max-jump 0.5 \
  --t-cond-power 1.0 --t-cond-zero-rate 0.0 \
  --t-cond-grid 0.1,0.3,0.5,0.7,0.9 --t-cond-jitter 0.04 \
  --transition-steps 1,2,4 --transition-teacher-steps 16 \
  --rollout-steps 4 --rollout-final-weight 8.0 --rollout-velocity-weight 2.0 \
  --rollout-teacher-forced-velocity-weight 0.0 \
  --rollout-spread-weight 0.0 --rollout-spread-samples 1 \
  --rollout-diagonal --rollout-condition-source generated \
  --generated-condition-steps 128 \
  --eval-diagonal --ema 0.999 --log-every 50 --eval-every "${EVAL_EVERY}" --n-eval "${N_EVAL}" \
  --mfm-steps 4 --glass-steps 4 --seed 44 \
  --block-conditioning additive --velocity-parametrization endpoint \
  --out-dir "${OUT}"
