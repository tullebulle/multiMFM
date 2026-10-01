#!/usr/bin/env bash
# Dispatch best-of-lookahead runs across interactive GPUs (one srun each, parallel).
# Manifest lines: JOBID PROPERTY MU RS SELMINT TAG   (#comments / blanks ignored)
set -uo pipefail
MANIFEST="$1"
REPO=/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco
LOGDIR="${REPO}/codex_scratch/memory/search/logs"
mkdir -p "${LOGDIR}"
declare -a PIDS TAGS
while read -r JOBID PROPERTY MU RS SELMINT TAG GMIN GMAX NUM _rest; do
  [[ -z "${JOBID:-}" || "${JOBID:0:1}" == "#" ]] && continue
  GMIN="${GMIN:-0.05}"; GMAX="${GMAX:-0.95}"; NUM="${NUM:-100}"
  echo "dispatch: jobid=${JOBID} prop=${PROPERTY} mu=${MU} rs=${RS} selmint=${SELMINT} gwin=[${GMIN},${GMAX}] n=${NUM} tag=${TAG}"
  srun --jobid="${JOBID}" --overlap bash "${REPO}/codex_scratch/memory/run_bol.sh" \
       "${PROPERTY}" "${MU}" "${RS}" "${SELMINT}" "${TAG}" "${GMIN}" "${GMAX}" "${NUM}" > "${LOGDIR}/${TAG}.log" 2>&1 &
  PIDS+=($!); TAGS+=("${TAG}")
done < "${MANIFEST}"
echo "launched ${#PIDS[@]} runs; waiting..."
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "OK   ${TAGS[$i]}"; else echo "FAIL ${TAGS[$i]}"; FAIL=$((FAIL+1)); fi
done
echo "=== all done; failures=${FAIL} ==="
