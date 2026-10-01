#!/usr/bin/env bash
# multiMFM-SS (steer_search) for one QM9 property: the "multiMFM-SS (Ours)" row of
# Table 2 and the multiMFM-SS column of Table 21.
#
#   scripts/run_steer_search.sh <property> [num_samples] [mu] [reward_scale] [select_min_t]
#
# property in {alpha, cv, gap, homo, lumo, mu}; num_samples defaults to 1000 (the
# paper). Pass a small n (e.g. 32 or 100) for a quick check. The per-property
# mu / reward_scale / select_min_t below are the configs of the submitted runs
# (tabasco/codex_scratch/memory/bol1k.manifest, runs k_<prop>_{a,b}, 2026-07-25);
# pass positional overrides to sweep them.
#
# Environment overrides:
#   LOOKAHEAD_SOURCE  mfm (default; Algorithm 3: store the value estimator's MFM
#                     look-aheads, no extra generative evaluations) | base_endpoint (the
#                     code behind the submitted row: harvest the base model's one-step
#                     endpoint prediction at t >= select_min_t, one extra base forward
#                     per harvested step) | both (both selections on one trajectory)
#   SEED, TARGET_SEED sampling / target seeds (default 8 / 2026, the submitted runs)
#   TOPK              MFM look-aheads kept per molecule on CPU (default 1024)
#   PB_WORKERS        PoseBusters worker processes (default 16)
#   EXTRA_ARGS        appended to the python command (e.g. "--diagnostic-variants")
#   VALUE_BATCH_SIZE  value-gradient chunk (default 0 = one batch, ~64 GB at n = 1000;
#                     250 fits a 40 GB GPU; only changes how the MFM prior noise is drawn)
# Output: $OUTPUT_DIR, default outputs/steer_search/<property>_n<num_samples>/
# (summary.json, per_sample.csv).
#
# select_min_t only applies to the base_endpoint harvest (and the mfm_window
# diagnostic); the mfm source uses every guided step, as Algorithm 3 does.
# The submitted "replicates" k_<prop>_a / k_<prop>_b used IDENTICAL arguments (same
# seed and targets); they differ only through nondeterministic CUDA kernels.
# steer_search also reports final_endpoint (the guided trajectory's last state at
# this larger SS guidance strength); that is not the non-SS multiMFM row, which is
# scripts/qm9/run_multimfm.sh.
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/env.sh"

PROPERTY="${1:-alpha}"
NUM="${2:-1000}"

# Per-property steering config (mu, reward_scale, select_min_t) of the submitted runs.
case "${PROPERTY}" in
  alpha) MU=11.0;  RS=0.3;     SELMINT=0.3 ;;
  cv)    MU=5.0;   RS=1.5;     SELMINT=0.5 ;;
  gap)   MU=1.0;   RS=12000.0; SELMINT=0.3 ;;
  homo)  MU=1.5;   RS=41500.0; SELMINT=0.3 ;;
  lumo)  MU=1.5;   RS=12000.0; SELMINT=0.3 ;;
  mu)    MU=3.0;   RS=8.638;   SELMINT=0.3 ;;
  *) echo "unknown property '${PROPERTY}' (expected alpha|cv|gap|homo|lumo|mu)" >&2; exit 1 ;;
esac
MU="${3:-$MU}"; RS="${4:-$RS}"; SELMINT="${5:-$SELMINT}"
OUT="${OUTPUT_DIR:-${REPO_ROOT}/outputs/steer_search/${PROPERTY}_n${NUM}}"
PB_WORKERS="${PB_WORKERS:-16}"
SEED="${SEED:-8}"
TARGET_SEED="${TARGET_SEED:-2026}"
LOOKAHEAD_SOURCE="${LOOKAHEAD_SOURCE:-mfm}"
TOPK="${TOPK:-1024}"
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})

echo "steer_search: prop=${PROPERTY} n=${NUM} mu=${MU} reward_scale=${RS} select_min_t=${SELMINT} source=${LOOKAHEAD_SOURCE} seed=${SEED}/${TARGET_SEED} out=${OUT}"
python -m multimfm.steer_search \
  --checkpoint "${REPO_ROOT}/checkpoints/base_flow/model_step_100000.pt" \
  --mfm-checkpoint "${REPO_ROOT}/checkpoints/mfm_student/student_step_1000.pt" \
  --mfm-key student --mfm-diagonal --device cuda \
  --seed "${SEED}" --target-seed "${TARGET_SEED}" --num-samples "${NUM}" \
  --sample-steps 128 --mu "${MU}" --reward-scale "${RS}" \
  --value-samples 32 --value-batch-size "${VALUE_BATCH_SIZE:-0}" --value-glass-steps 4 \
  --guide-every 1 --guide-min-t 0.05 --guide-max-t 0.95 \
  --eps 1e-6 --soft-atom-temperature 0.25 \
  --property-name "${PROPERTY}" --partition train_flow --hist-bins 100 \
  --lookahead-source "${LOOKAHEAD_SOURCE}" --lookahead-topk "${TOPK}" \
  --select-min-t "${SELMINT}" --pb-candidates 0 --pb-workers "${PB_WORKERS}" \
  --output-dir "${OUT}" "${EXTRA[@]}"
