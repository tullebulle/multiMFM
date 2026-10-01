#!/usr/bin/env bash
# One QM9 guided run at an explicit (posterior steps x MC) split of the value-gradient
# budget. See docs/qm9_operating_point.md.
#
#   scripts/qm9/operating_point/run_split.sh <row> <property> <steps> <mc> <n> <seed> <out_dir>
#
#   row       multimfm | ss   (property_targets / steer_search, i.e. the two Table 2 rows)
#   steps     --value-glass-steps  (posterior Euler steps of the diagonal MFM velocity)
#   mc        --value-samples      (Monte-Carlo posterior draws per guided step)
#   n         --num-samples
#   seed      sampling seed; the target seed is TARGET_SEED, default 2026 for seed 8,
#             else 2026 + seed (the table2_3seed convention)
#
# Every other argument is held at the value the published Table 2 run used, taken
# verbatim from scripts/qm9/run_multimfm.sh and scripts/run_steer_search.sh, which this
# script deliberately does not modify. The only variables are `steps` and `mc`, whose
# product is the value-gradient NFE per guided step: 4 x 32 is the published point, and
# 1 x 128 / 2 x 64 / 8 x 16 are the other splits of the same budget.
#
# Environment:
#   TARGET_SEED     override the target seed
#   MFM_DIAGONAL    1 (default, the published usage: the student as a diagonal velocity
#                   field, Euler-integrated) | 0 (query it as a genuine two-time flow map
#                   over gaps of 1/steps -- a labelled diagnostic, see the doc section 1a)
#   ROWS_PER_CHUNK  target rows per value-gradient chunk (default 3200); the chunk size
#                   in molecules is ROWS_PER_CHUNK / mc, so peak GPU memory is the same
#                   at every split. 3200 reproduces the n=100 pilot's footprint (6.4 GB),
#                   8000 the n=1000 table2_3seed footprint (VALUE_BATCH_SIZE=250 at MC 32).
#   SS_SOURCE       --lookahead-source for the ss row (default both, as table2_3seed)
#   SS_DIAG         1 (default) adds --diagnostic-variants, as table2_3seed
#   TOPK            --lookahead-topk (default 0 = keep every look-ahead, exactly Alg. 3)
#   PB_WORKERS      PoseBusters workers (default 16)
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/env.sh"

ROW="${1:?row}"; PROP="${2:?property}"; STEPS="${3:?steps}"; MC="${4:?mc}"
N="${5:?n}"; SEED="${6:?seed}"; OUT="${7:?out_dir}"
if [ "${SEED}" = "8" ]; then DEF_TSEED=2026; else DEF_TSEED=$((2026 + SEED)); fi
TSEED="${TARGET_SEED:-$DEF_TSEED}"
W="${PB_WORKERS:-16}"

# Peak-memory-matched chunking: a constant number of posterior rows per chunk.
RPC="${ROWS_PER_CHUNK:-3200}"
VBS=$(( RPC / MC ))
if [ "${VBS}" -lt 1 ]; then VBS=1; fi
if [ "${VBS}" -ge "${N}" ]; then VBS=0; fi

DIAG_FLAG=()
if [ "${MFM_DIAGONAL:-1}" = "1" ]; then DIAG_FLAG=(--mfm-diagonal); fi

BASE_CKPT="${REPO_ROOT}/checkpoints/base_flow/model_step_100000.pt"
MFM_CKPT="${REPO_ROOT}/checkpoints/mfm_student/student_step_1000.pt"

case "${ROW}:${PROP}" in
  # multiMFM: the (mu, reward_scale) of scripts/qm9/run_multimfm.sh
  multimfm:cv)    MU=2.80; RS=1.5 ;;
  multimfm:mu)    MU=0.78; RS=8.638 ;;
  multimfm:alpha) MU=3.40; RS=0.3 ;;
  multimfm:gap)   MU=0.17; RS=12000 ;;
  multimfm:homo)  MU=0.50; RS=41500 ;;
  multimfm:lumo)  MU=0.44; RS=12000 ;;
  # multiMFM-SS: the (mu, reward_scale, select_min_t) of scripts/run_steer_search.sh,
  # with lumo mu 3.0 -- the value table2_3seed used (ss_mu_reselect decision), not the
  # submitted 1.5, so that this grid is paired with the reference 3-seed run.
  ss:alpha) MU=11.0; RS=0.3;     SELMINT=0.3 ;;
  ss:cv)    MU=5.0;  RS=1.5;     SELMINT=0.5 ;;
  ss:gap)   MU=1.0;  RS=12000.0; SELMINT=0.3 ;;
  ss:homo)  MU=1.5;  RS=41500.0; SELMINT=0.3 ;;
  ss:lumo)  MU=3.0;  RS=12000.0; SELMINT=0.3 ;;
  ss:mu)    MU=3.0;  RS=8.638;   SELMINT=0.3 ;;
  *) echo "unknown row:property '${ROW}:${PROP}'" >&2; exit 1 ;;
esac

mkdir -p "${OUT}"
echo "run_split: row=${ROW} prop=${PROP} steps=${STEPS} mc=${MC} nfe_value=$((STEPS * MC))" \
     "n=${N} seed=${SEED}/${TSEED} mu=${MU} rs=${RS} vbs=${VBS}" \
     "diagonal=${MFM_DIAGONAL:-1} out=${OUT}"

if [ "${ROW}" = "multimfm" ]; then
  python -m multimfm.property_targets \
    --checkpoint "${BASE_CKPT}" --device cuda \
    --seed "${SEED}" --target-seed "${TSEED}" --num-samples "${N}" \
    --sample-steps 128 --mu "${MU}" --reward-scale "${RS}" \
    --value-samples "${MC}" --value-batch-size "${VBS}" --value-glass-steps "${STEPS}" \
    --value-sampler mfm \
    --mfm-checkpoint "${MFM_CKPT}" --mfm-key student "${DIAG_FLAG[@]}" \
    --guide-every 1 --guide-min-t 0.05 --guide-max-t 0.95 \
    --eps 1e-6 --soft-atom-temperature 0.25 \
    --property-name "${PROP}" --partition train_flow --hist-bins 100 \
    --guide-atomics --compute-pb --per-sample-pb --pb-workers "${W}" \
    --output-dir "${OUT}"
else
  SS_EXTRA=()
  if [ "${SS_DIAG:-1}" = "1" ]; then SS_EXTRA=(--diagnostic-variants); fi
  python -m multimfm.steer_search \
    --checkpoint "${BASE_CKPT}" \
    --mfm-checkpoint "${MFM_CKPT}" --mfm-key student "${DIAG_FLAG[@]}" --device cuda \
    --seed "${SEED}" --target-seed "${TSEED}" --num-samples "${N}" \
    --sample-steps 128 --mu "${MU}" --reward-scale "${RS}" \
    --value-samples "${MC}" --value-batch-size "${VBS}" --value-glass-steps "${STEPS}" \
    --guide-every 1 --guide-min-t 0.05 --guide-max-t 0.95 \
    --eps 1e-6 --soft-atom-temperature 0.25 \
    --property-name "${PROP}" --partition train_flow --hist-bins 100 \
    --lookahead-source "${SS_SOURCE:-both}" --lookahead-topk "${TOPK:-0}" \
    --select-min-t "${SELMINT}" --pb-candidates 0 --pb-workers "${W}" \
    --output-dir "${OUT}" "${SS_EXTRA[@]}"
fi
