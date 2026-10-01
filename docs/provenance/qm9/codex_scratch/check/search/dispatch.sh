#!/usr/bin/env bash
# Dispatch a manifest of runs across interactive GPU allocations, one srun each, in parallel.
# Manifest lines: JOBID PROPERTY MU REWARD_SCALE TAG   (blank lines / #comments ignored)
# Usage: dispatch.sh <manifest_file>
set -uo pipefail
MANIFEST="$1"
REPO=/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco
LOGDIR="${REPO}/codex_scratch/check/search/logs"
mkdir -p "${LOGDIR}"

declare -a PIDS TAGS
while read -r JOBID PROPERTY MU RS TAG GMAX _rest; do
  [[ -z "${JOBID:-}" || "${JOBID:0:1}" == "#" ]] && continue
  GMAX="${GMAX:-0.95}"
  echo "dispatch: jobid=${JOBID} prop=${PROPERTY} mu=${MU} rs=${RS} tag=${TAG} gmax=${GMAX}"
  srun --jobid="${JOBID}" --overlap bash "${REPO}/codex_scratch/check/search/run_prop.sh" \
       "${PROPERTY}" "${MU}" "${RS}" "${TAG}" "${GMAX}" > "${LOGDIR}/${TAG}.log" 2>&1 &
  PIDS+=($!); TAGS+=("${TAG}")
done < "${MANIFEST}"

echo "launched ${#PIDS[@]} runs; waiting..."
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "OK   ${TAGS[$i]}"; else echo "FAIL ${TAGS[$i]} (exit $?)"; FAIL=$((FAIL+1)); fi
done
echo "=== all done; failures=${FAIL} ==="
