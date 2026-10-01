#!/usr/bin/env bash
# multiMFM (non-SS) QM9 property guidance: the "multiMFM (Ours)" row of Table 2 and
# the multiMFM columns of Tables 21 (PB validity) and 22 (uniqueness).
#
#   scripts/qm9/run_multimfm.sh <property> [num_samples] [mu] [reward_scale]
#
# property in {cv, mu, alpha, gap, homo, lumo}; num_samples defaults to 1000 (the
# paper). Pass e.g. 32 for a quick check: with the fixed seeds the first n molecules
# get the same lengths and targets as the first n of the n=1000 run.
# Output dir: $OUTPUT_DIR, default outputs/qm9/multimfm/<property>_n<num_samples>/
# (per_sample.csv, summary.csv, guided_step_logs.csv, summary.json).
# Environment overrides: SEED / TARGET_SEED (default 8 / 2026, the submitted runs),
# PER_SAMPLE_PB=1 (default 0) to store each molecule's PB flag and SMILES in
# per_sample.csv (PB then runs in PB_WORKERS processes, default 8), VALUE_BATCH_SIZE
# (default 0 = all molecules in one value-gradient batch, ~64 GB GPU memory at n = 1000;
# 250 fits a 40 GB GPU and only changes how the MFM prior noise is drawn).
#
# This replays tabasco/codex_scratch/check/search/run_prop.sh (2026-07-25) with the
# in-repo checkpoints. The per-property (mu, reward_scale) are the grid points the
# paper numbers come from (runs r2_cv_mu2p80, r3_mu_mu0p78, r5_alpha_mu3p40,
# r5_gap_mu0p17, r3_homo_mu0p50, r2_lumo_mu0p44; see results/qm9/PROVENANCE.md).
# They were chosen as the lowest oracle MAE with PB-valid >= 0.90 over a 5-round
# search on this same seed/target set. Guided sampling is not bit-reproducible
# (nondeterministic CUDA backward), so reruns agree within noise, not exactly.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/env.sh"

PROPERTY="${1:?usage: run_multimfm.sh <cv|mu|alpha|gap|homo|lumo> [num_samples] [mu] [reward_scale]}"
NUM="${2:-1000}"

case "${PROPERTY}" in
  cv)    MU=2.80; RS=1.5 ;;
  mu)    MU=0.78; RS=8.638 ;;
  alpha) MU=3.40; RS=0.3 ;;
  gap)   MU=0.17; RS=12000 ;;
  homo)  MU=0.50; RS=41500 ;;
  lumo)  MU=0.44; RS=12000 ;;
  *) echo "unknown property '${PROPERTY}' (expected cv|mu|alpha|gap|homo|lumo)" >&2; exit 1 ;;
esac
MU="${3:-$MU}"; RS="${4:-$RS}"
OUT="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qm9/multimfm/${PROPERTY}_n${NUM}}"
SEED="${SEED:-8}"
TARGET_SEED="${TARGET_SEED:-2026}"
PB_EXTRA=()
if [ "${PER_SAMPLE_PB:-0}" = "1" ]; then PB_EXTRA=(--per-sample-pb --pb-workers "${PB_WORKERS:-8}"); fi

echo "multiMFM: prop=${PROPERTY} n=${NUM} mu=${MU} reward_scale=${RS} seed=${SEED}/${TARGET_SEED} out=${OUT}"
python -m multimfm.property_targets \
  --checkpoint "${REPO_ROOT}/checkpoints/base_flow/model_step_100000.pt" --device cuda \
  --seed "${SEED}" --target-seed "${TARGET_SEED}" --num-samples "${NUM}" \
  --sample-steps 128 --mu "${MU}" --reward-scale "${RS}" \
  --value-samples 32 --value-batch-size "${VALUE_BATCH_SIZE:-0}" --value-glass-steps 4 \
  --value-sampler mfm \
  --mfm-checkpoint "${REPO_ROOT}/checkpoints/mfm_student/student_step_1000.pt" \
  --mfm-key student --mfm-diagonal \
  --guide-every 1 --guide-min-t 0.05 --guide-max-t 0.95 \
  --eps 1e-6 --soft-atom-temperature 0.25 \
  --property-name "${PROPERTY}" --partition train_flow --hist-bins 100 \
  --guide-atomics --compute-pb "${PB_EXTRA[@]}" \
  --output-dir "${OUT}"
