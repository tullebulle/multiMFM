#!/usr/bin/env bash
#SBATCH --job-name=qm9_op_grid
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# --partition=kempner: every node there is an A100, which is also the card the reference
# 4 x 32 runs in outputs/qm9/table2_3seed used (scripts/qm9/table2_3seed.sbatch), so the
# grid and its reference are on one card generation.  On kempner_requeue instead, add
# --constraint="a100|a100-mig|h100|h200": that partition holds RTX PRO 6000 Blackwell
# nodes (sm_120) which the pinned torch 2.5.1+cu124 cannot run.
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Stage A of docs/qm9_operating_point.md: the NFE-matched (steps x MC) grid on the
# pre-registered selection property (alpha), seed 8 / target seed 2026, n = 100 --
# the set on which every published QM9 guidance hyperparameter was already selected,
# disjoint from Table 2's evaluation seeds 1-3.
#
#   task 0 = the multiMFM row, task 1 = the multiMFM-SS row
#   splits 1x128, 2x64, 4x32 (published), 8x16 -- all 14,848 generative NFE/molecule
#   plus two labelled diagnostics outside the pre-registered selection: the same
#   checkpoint queried as a genuine two-time flow map (no --mfm-diagonal) at 2x64 and
#   4x32, where the jump gaps (0.5, 0.25) are inside the student's max_jump 0.5.
#
#   sbatch scripts/qm9/operating_point/grid_alpha.sbatch
#
# A configuration whose summary.json exists is skipped, so a requeued task resumes.
# If a task dies at once with "CUDA-capable device(s) is/are busy or unavailable", see the
# GPU note below and just resubmit it; the skip-if-summary.json logic makes that free.
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
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
ROW="${ROWS[$I]}"
PROP="${PROP:-alpha}"
N="${N:-100}"
SEED="${SEED:-8}"
ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/qm9/operating_point/grid_${PROP}_n${N}_s${SEED}}"
export PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}"
export ROWS_PER_CHUNK="${ROWS_PER_CHUNK:-3200}"
echo "task=${I} row=${ROW} prop=${PROP} n=${N} seed=${SEED} host=$(hostname)" \
     "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" \
     "restart=${SLURM_RESTART_COUNT:-0} start=$(date)"

one() {  # steps mc diagonal tag
  local steps="$1" mc="$2" diag="$3" tag="$4"
  local out="${ROOT}/${ROW}/${tag}"
  if [ -f "${out}/summary.json" ]; then echo "skip ${ROW}/${tag}"; return; fi
  echo "----- ${ROW}/${tag} -----"
  MFM_DIAGONAL="${diag}" bash scripts/qm9/operating_point/run_split.sh \
    "${ROW}" "${PROP}" "${steps}" "${mc}" "${N}" "${SEED}" "${out}" 2>&1 \
    | { grep -viE "warning|pkg_resources|no bonds|UFFTYPER|kekulize|^=+$|^$" || true; }
}

# SPLITS is a space-separated list of steps:mc:diagonal:tag.  The default is the
# pre-registered grid -- the published diagonal usage at four splits of one NFE budget --
# plus two diagnostics that are NOT part of the selection (doc section 1a(b)): the same
# checkpoint queried as a genuine two-time flow map instead of a diagonal velocity field.
# Override it to add off-budget reference points, e.g.
#   SPLITS="4:64:1:s4_mc64 4:128:1:s4_mc128" sbatch ... grid_alpha.sbatch
SPLITS="${SPLITS:-1:128:1:s1_mc128 2:64:1:s2_mc64 4:32:1:s4_mc32 8:16:1:s8_mc16 2:64:0:fm2_mc64_nodiag 4:32:0:fm4_mc32_nodiag}"
for SPEC in ${SPLITS}; do
  IFS=: read -r STEPS MC DIAG TAG <<< "${SPEC}"
  one "${STEPS}" "${MC}" "${DIAG}" "${TAG}"
done
echo "done $(date)"
