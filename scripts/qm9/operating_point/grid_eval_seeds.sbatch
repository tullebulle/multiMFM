#!/usr/bin/env bash
#SBATCH --job-name=qm9_op_seeds
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --requeue
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# See grid_alpha.sbatch on the partition choice (A100 everywhere, matching the reference).
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Stage A2 of docs/qm9_operating_point.md: the same four NFE-matched splits on the
# evaluation seeds 1-3 at n = 100, target seeds 2027-2029, property alpha.
#
# Purpose is the baseline comparison, not the selection: seeds 1-3 / n = 100 / alpha is
# exactly the set of results/qm9/baselines/baseline_sweep_v2.csv, so best-of-N, DPS,
# Feynman-Kac and the existing MC ladder at 4 posterior steps can be compared with the
# split axis on shared sample ids at matched generative NFE.  The operating point itself
# is selected on seed 8 (grid_alpha.sbatch), never here.
#
#   task i: row = (multimfm, ss)[i / 3], seed = (1, 2, 3)[i % 3]
#   sbatch scripts/qm9/operating_point/grid_eval_seeds.sbatch
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

ROWS=(multimfm ss)
SEEDS=(1 2 3)
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
ROW="${ROWS[$((I / 3))]}"
SEED="${SEEDS[$((I % 3))]}"
PROP="${PROP:-alpha}"
N="${N:-100}"
ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/qm9/operating_point/evalseeds_${PROP}_n${N}}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}"
export ROWS_PER_CHUNK="${ROWS_PER_CHUNK:-3200}"
echo "task=${I} row=${ROW} seed=${SEED} prop=${PROP} n=${N} host=$(hostname)" \
     "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) start=$(date)"

for SPLIT in "1 128" "2 64" "4 32" "8 16"; do
  set -- ${SPLIT}
  STEPS="$1"; MC="$2"
  OUT="${ROOT}/${ROW}/s${STEPS}_mc${MC}_seed${SEED}"
  if [ -f "${OUT}/summary.json" ]; then echo "skip ${ROW}/s${STEPS}_mc${MC}_seed${SEED}"; continue; fi
  echo "----- ${ROW} ${STEPS}x${MC} seed ${SEED} -----"
  bash scripts/qm9/operating_point/run_split.sh \
    "${ROW}" "${PROP}" "${STEPS}" "${MC}" "${N}" "${SEED}" "${OUT}" 2>&1 \
    | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER|kekulize|^=+$|^$" || true; }
done
echo "done $(date)"
