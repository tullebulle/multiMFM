#!/usr/bin/env bash
#SBATCH --job-name=qm9_op_t2
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --requeue
#SBATCH --array=0-35%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Stage B of docs/qm9_operating_point.md: Table 2 (and Tables 21, 22) re-run at ONE
# (posterior steps x MC) split chosen on alpha at seed 8 -- 3 seeds x 6 properties x
# n = 1000, the same grid, seeds, target seeds and drivers as
# scripts/qm9/table2_3seed.sbatch, so every arm is paired per sample_idx with the
# published 4 x 32 reference in outputs/qm9/table2_3seed/.
#
#   STEPS / MC are required and are the SAME for all six properties and both rows.
#   There is no per-property choice: that is the whole point of the pre-registration.
#
#   STEPS=1 MC=128 sbatch scripts/qm9/operating_point/table2_split.sbatch
#
#   task i: row = (multimfm, ss)[i / 18], seed = (1, 2, 3)[(i % 18) / 6],
#           property = (cv, mu, alpha, gap, homo, lumo)[i % 6], target seed = 2026 + seed.
#
# ROWS_PER_CHUNK=8000 reproduces the reference run's value-gradient footprint
# (VALUE_BATCH_SIZE=250 at MC 32) at every split, so peak GPU memory does not depend on
# the split.  --mem=180G: with --lookahead-topk 0 the SS row keeps every look-ahead on the
# CPU, and the store is guided_steps x MC x 1000 molecules x 108 floats per variant,
# i.e. ~6 GB per variant at MC 128 against ~1.6 GB at MC 32.
# A task whose summary.json exists is skipped, so a requeued task resumes.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "submit from the repo root, or set MULTIMFM_ROOT" >&2; exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs

# GPU diagnostics.  Four tasks died at the first .to(device) with "CUDA error: CUDA-capable
# device(s) is/are busy or unavailable" -- 49464797_1, 49464799_1, 49465288_0, 49474230_1 --
# and all four were on ONE node, holygpu8a19102, each time while another task of mine
# already held a card there.  It is node-specific, not a co-location problem in general:
# two tasks of mine co-located on holygpu8a19202 ran fine, and the echoes below show why
# (Slurm sets CUDA_VISIBLE_DEVICES=0, the device cgroup constrains, and nvidia-smi -L
# reports a DIFFERENT card UUID in each task).  Remedy: resubmit with
# --exclude=<that node>.  Every configuration skips itself when its summary.json exists,
# so a resubmission resumes instead of repeating work, and no result comes from a partial
# run.  The fallback below only fires if the cgroup is ever found unconstrained.
echo "gpu env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" \
     "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
# Measured cause (jobs 49474230_1, 49465288_0): Slurm sets CUDA_VISIBLE_DEVICES=0 -- the
# step-local index -- but does NOT constrain the device cgroup on these nodes, so every
# co-located task actually opens global GPU 0.  nvidia-smi ignores CUDA_VISIBLE_DEVICES and
# lists what the cgroup permits, so "more than one device listed" is exactly the test for
# an unconstrained cgroup; in that case use the global index Slurm allocated instead.
nvidia-smi -L 2>&1 | sed 's/^/  nvidia-smi -L: /' || true
_NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
if [ "${_NGPU:-0}" -gt 1 ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  echo "cgroup not constraining (${_NGPU} devices visible): CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}"
fi

STEPS="${STEPS:?set STEPS=<posterior steps>}"
MC="${MC:?set MC=<value samples>}"
ROWS=(multimfm ss)
PROPS=(cv mu alpha gap homo lumo)
SEEDS=(1 2 3)
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
ROW="${ROWS[$((I / 18))]}"
SEED="${SEEDS[$(((I % 18) / 6))]}"
PROP="${PROPS[$((I % 6))]}"
N="${N:-1000}"
ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/qm9/operating_point/table2_s${STEPS}_mc${MC}_n${N}}"
OUT="${ROOT}/${ROW}/${PROP}_s${SEED}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}"
export ROWS_PER_CHUNK="${ROWS_PER_CHUNK:-8000}"
echo "task=${I} row=${ROW} prop=${PROP} seed=${SEED} split=${STEPS}x${MC} n=${N}" \
     "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" \
     "restart=${SLURM_RESTART_COUNT:-0} start=$(date)"
if [ -f "${OUT}/summary.json" ]; then echo "skip: ${OUT}/summary.json exists"; exit 0; fi

bash scripts/qm9/operating_point/run_split.sh \
  "${ROW}" "${PROP}" "${STEPS}" "${MC}" "${N}" "${SEED}" "${OUT}" 2>&1 \
  | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER|kekulize|^=+$|^$" || true; }
echo "done $(date)"
