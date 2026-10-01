#!/usr/bin/env bash
#SBATCH --job-name=geom_train_student
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Distil the GEOM-Drugs multiMFM student (paper Table 20, GEOM-Drugs column).
#
# The released student is the end of a four-leg chain; LEG selects one leg with the
# arguments recorded in that leg's checkpoint (docs/provenance/geom/checkpoints.md):
#
#   LEG=1a  stage 1, from the teacher, 3 GPUs x 256, lr 2e-4, 20k of 40k steps
#   LEG=1b  stage 1, resume 1a,        3 GPUs x 256, lr 1e-3, 40k steps
#   LEG=2a  stage 2, resume 1b,        3 GPUs x 128, lr 1e-4, max_jump 0.25, 20k of 30k
#   LEG=2b  stage 2, resume 2a,        4 GPUs x  96, lr 1e-4, max_jump 0.15, esd_delta_clip 8
#           -> its step-10,000 checkpoint is checkpoints/geom/mfm_student/student_step_10000.pt
#
# Each leg needs the previous leg's checkpoint in INIT_CKPT (legs 1b/2a/2b); only the
# final leg's checkpoint ships with the repo, so a full rebuild runs 1a -> 2b in order.
#
#   LEG=1a sbatch --gres=gpu:3 scripts/geom/train_student.sbatch
#   LEG=2b INIT_CKPT=outputs/geom/mfm_student/stage2a/student_step_20000.pt \
#       sbatch scripts/geom/train_student.sbatch
#   LEG=2b STEPS=40 EVAL_EVERY=0 NPROC=1 BATCH=4 LIMIT=256 NO_WANDB=1 \
#       INIT_CKPT=checkpoints/geom/mfm_student/student_step_10000.pt \
#       sbatch --partition=gpu_test --gres=gpu:1 --time=00:30:00 \
#       scripts/geom/train_student.sbatch                              # smoke test
#
# Needs data/geom/processed_geom_train.pt (see docs/provenance/geom/data.md).
# Distillation is not bit-reproducible on GPU.
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

LEG="${LEG:-2b}"
TEACHER="${TEACHER:-checkpoints/geom/teacher/epoch=38-step=89310.ckpt}"
DATA="${DATA:-data/geom/processed_geom_train.pt}"
LIMIT="${LIMIT:-150000}"
OUT_DIR="${OUT_DIR:-outputs/geom/mfm_student/leg${LEG}_${SLURM_JOB_ID:-manual}}"
NO_WANDB="${NO_WANDB:-1}"
EVAL_EVERY="${EVAL_EVERY:-}"
INIT_CKPT="${INIT_CKPT:-}"

# --- per-leg recipe, exactly as recorded in each leg's checkpoint ---
case "${LEG}" in
  1a) STAGE=1; NPROC_D=3; BATCH_D=256; STEPS_D=40000;  LR_D=2e-4; EMA_D=0.999;
      TCP_D=1.0; TCZ_D=0.0; EVAL_D=5000; EXTRA=(--endpoint-space-loss --init-from-teacher);;
  1b) STAGE=1; NPROC_D=3; BATCH_D=256; STEPS_D=40000;  LR_D=1e-3; EMA_D=0.999;
      TCP_D=1.0; TCZ_D=0.0; EVAL_D=5000; EXTRA=(--endpoint-space-loss);;
  2a) STAGE=2; NPROC_D=3; BATCH_D=128; STEPS_D=30000;  LR_D=1e-4; EMA_D=0.995;
      TCP_D=2.0; TCZ_D=0.1; EVAL_D=5000;
      EXTRA=(--max-jump 0.25 --anneal-frac 1.0 --esd-weight 1.0 --diag-weight 2.0
             --esd-target-clip 10.0 --esd-delta-clip 0.0 --esd-loss-type mse);;
  2b) STAGE=2; NPROC_D=4; BATCH_D=96;  STEPS_D=30000;  LR_D=1e-4; EMA_D=0.995;
      TCP_D=2.0; TCZ_D=0.1; EVAL_D=2000;
      EXTRA=(--max-jump 0.15 --anneal-frac 0.2 --esd-weight 1.0 --diag-weight 2.0
             --esd-target-clip 10.0 --esd-delta-clip 8.0 --esd-loss-type mse);;
  *) echo "unknown LEG=${LEG} (1a|1b|2a|2b)" >&2; exit 1;;
esac

NPROC="${NPROC:-${NPROC_D}}"
BATCH="${BATCH:-${BATCH_D}}"
STEPS="${STEPS:-${STEPS_D}}"
LR="${LR:-${LR_D}}"
EMA="${EMA:-${EMA_D}}"
EVAL_EVERY="${EVAL_EVERY:-${EVAL_D}}"

if [ "${LEG}" != "1a" ] && [ -z "${INIT_CKPT}" ]; then
  echo "LEG=${LEG} resumes the previous leg: set INIT_CKPT=<student_step_*.pt>" >&2
  exit 1
fi
[ -n "${INIT_CKPT}" ] && EXTRA+=(--init-ckpt "${INIT_CKPT}")
[ "${NO_WANDB}" = "1" ] && EXTRA+=(--no-wandb)

echo "leg=${LEG} stage=${STAGE} nproc=${NPROC} batch/dev=${BATCH} global=$(( BATCH * NPROC )) steps=${STEPS} lr=${LR}"
"${PY}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m multimfm.geom.train_student \
  --stage "${STAGE}" --teacher-checkpoint "${TEACHER}" --data-path "${DATA}" \
  --steps "${STEPS}" --lr "${LR}" --warmup "${WARMUP:-500}" --ema "${EMA}" \
  --batch-size "${BATCH}" --limit-data "${LIMIT}" --t-max "${T_MAX:-0.95}" \
  --t-cond-power "${TCP_D}" --t-cond-zero-rate "${TCZ_D}" \
  --coord-loss-weight 1.0 --atom-loss-weight 1.0 \
  --velocity-parametrization endpoint --block-conditioning additive \
  --eval-every "${EVAL_EVERY}" --log-every "${LOG_EVERY:-200}" \
  --n-eval "${N_EVAL:-64}" --glass-steps "${GLASS_STEPS:-64}" --mfm-steps "${MFM_STEPS:-4}" \
  --seed "${SEED:-0}" --out-dir "${OUT_DIR}" "${EXTRA[@]}"

echo "checkpoints in ${OUT_DIR}"
