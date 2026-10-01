#!/usr/bin/env bash
#SBATCH --job-name=qm9_t2_3seed
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --requeue
#SBATCH --array=0-35%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Table 2 (+ Tables 21, 22) QM9 rows, 3 seeds x 6 properties x n = 1000 (authors'
# decisions 4 and 5). One array task = one (row, seed, property) run; a task whose
# summary.json exists is skipped, so a preempted/failed task is simply resubmitted:
#
#   sbatch scripts/qm9/table2_3seed.sbatch                  # all 36 tasks
#   sbatch --array=18-35 scripts/qm9/table2_3seed.sbatch    # only the SS row
#
# task i: row = (multimfm, ss)[i / 18], seed = (1, 2, 3)[(i % 18) / 6],
#         property = (cv, mu, alpha, gap, homo, lumo)[i % 6], target seed = 2026 + seed.
# Evaluation seeds 1-3 are disjoint from the seed-8 / target-seed-2026 set on which
# every guidance config was selected (submitted runs, SS pilot).
#   multimfm: scripts/qm9/run_multimfm.sh with the submitted per-property (mu, rs),
#             unguided + guided, per-sample PB flags and SMILES.
#   ss:       scripts/run_steer_search.sh with the submitted per-property SS configs
#             (lumo mu 3.0 instead of 1.5, see below),
#             LOOKAHEAD_SOURCE=both --diagnostic-variants: the Algorithm-3 selection
#             (MFM look-aheads, the reported row) and, on the same trajectory, the
#             original base-endpoint harvest and two MFM diagnostics.
# Outputs: outputs/qm9/table2_3seed/<row>/<prop>_s<seed>/. Collect with
#   bash scripts/qm9/collect_table2_3seed.sh
# Partition/account lines are site-specific. VALUE_BATCH_SIZE=250 (default here) keeps
# the value-gradient batch within a 40 GB A100 (one batch of 32,000 posterior samples
# needs ~64 GB); it only changes how the MFM prior noise is partitioned. Do not use
# Blackwell GPUs (unsupported by torch 2.5.1).
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO_ROOT}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO_ROOT}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO_ROOT}"
mkdir -p logs

ROWS=(multimfm ss)
PROPS=(cv mu alpha gap homo lumo)
SEEDS=(1 2 3)
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
ROW="${ROWS[$((I / 18))]}"
SEED="${SEEDS[$(((I % 18) / 6))]}"
PROP="${PROPS[$((I % 6))]}"
TARGET_SEED=$((2026 + SEED))
NUM="${NUM:-1000}"
ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/qm9/table2_3seed}"
OUT="${ROOT}/${ROW}/${PROP}_s${SEED}"
echo "task=${I} row=${ROW} prop=${PROP} seed=${SEED}/${TARGET_SEED} n=${NUM} host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) restart=${SLURM_RESTART_COUNT:-0} start=$(date)"
if [ -f "${OUT}/summary.json" ]; then
  echo "skip: ${OUT}/summary.json exists"; exit 0
fi
export SEED TARGET_SEED OUTPUT_DIR="${OUT}" PB_WORKERS="${SLURM_CPUS_PER_TASK:-16}"
export VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-250}"
if [ "${ROW}" = "multimfm" ]; then
  PER_SAMPLE_PB=1 bash scripts/qm9/run_multimfm.sh "${PROP}" "${NUM}"
else
  # SS mu: the submitted per-property value, except lumo 1.5 -> 3.0, chosen by the
  # pre-registered rule on the seed-8 mu grid (scripts/qm9/ss_mu_reselect.sbatch,
  # results/qm9/ss_mfm_reselect/ss_mu_decision.csv). TOPK=0 keeps every MFM look-ahead
  # (~1.5 GB CPU per variant at n = 1000), so the selection is exactly Algorithm 3's.
  SS_MU=""
  if [ "${PROP}" = "lumo" ]; then SS_MU=3.0; fi
  TOPK="${TOPK:-0}" LOOKAHEAD_SOURCE=both EXTRA_ARGS="--diagnostic-variants" \
    bash scripts/run_steer_search.sh "${PROP}" "${NUM}" ${SS_MU}
fi
echo "done $(date)"
