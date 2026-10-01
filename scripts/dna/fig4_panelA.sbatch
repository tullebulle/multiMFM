#!/bin/bash
#SBATCH --job-name=dna_fig4_panelA
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Figure 4 (left) only: |V_hat_t(x) - V_t(x)| vs the number of posterior samples N.
# `scripts/dna/fig4_all.sbatch` runs this together with panel B and is the normal
# entry point; this wrapper exists to rerun panel A on its own.
#
# Port of the recovered original producer
# (results/dna/original_scripts/benchmark_yeast_split_value_mc.py) on the
# parent-disjoint L=50 models: synthetic SmoothDNAReward (seed 1234, gain 4.0),
# held-out test sequences noised to t ~ U(0.05, 0.35), teacher-SDE reference
# (512 samples x 1024 steps), dMFM NFE=1, FMAP ODE NFE=256, plus the sde_mc floor
# series that the published panel omits.
#
# Work unit = a block of BLOCK conditions, one JSON each; existing JSONs are skipped.
# Array task a of W runs blocks a, a+W, ...
#
#   sbatch --array=0-1 scripts/dna/fig4_panelA.sbatch                          # original protocol
#   VARIANT=c0 sbatch --array=0-1 scripts/dna/fig4_panelA.sbatch               # robustness variant
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

N_CONDITIONS="${N_CONDITIONS:-32}"   # the original used 8; extra conditions only add error bars
BLOCK="${BLOCK:-4}"
VARIANT="${VARIANT:-original}"
EXTRA=""
STAGE="panelA"
if [ "${VARIANT}" != "original" ]; then
  STAGE="panelA_c0"
  EXTRA="--reward c0 --reference glass"
fi
OUT_DIR="${OUT_DIR:-outputs/dna/fig4/${STAGE}}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
WORKERS="${WORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
N_BLOCKS=$(( (N_CONDITIONS + BLOCK - 1) / BLOCK ))

for (( b = TASK; b < N_BLOCKS; b += WORKERS )); do
  START=$(( b * BLOCK ))
  END=$(( START + BLOCK ))
  ${PY} -u -m dmfm.benchmarks.value_error \
    --out "${OUT_DIR}/conds_$(printf '%02d' ${START})_$(printf '%02d' ${END}).json" \
    --n-conditions "${N_CONDITIONS}" \
    --cond-start "${START}" \
    --cond-end "${END}" \
    --sde-ref-n "${SDE_REF_N:-512}" \
    --sde-steps "${SDE_STEPS:-1024}" \
    --dmfm-nfe "${DMFM_NFE:-1}" \
    --fmap-nfe "${FMAP_NFE:-256}" \
    --chunk "${CHUNK:-512}" \
    ${EXTRA} ${EXTRA_ARGS:-} || echo "block ${b} FAILED"
done
echo "worker ${TASK} done"
