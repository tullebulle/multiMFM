#!/bin/bash
#SBATCH --job-name=dna_op_cpu
#SBATCH --partition=kozinsky_requeue
#SBATCH --account=kozinsky_lab
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# CPU hedge for the operating-point grids, for when the GPU partitions are backed up behind the
# account's shared 16-GPU cap. Defaults to kozinsky_requeue/kozinsky_lab, which on 2026-09-30 had an
# empty queue while serial_requeue had 8,260 jobs ahead of ours; override with
# --partition/--account for serial_requeue (account albergo_lab).
#
# account's shared 16-GPU cap. CPU is true fp32 while the GPU runs are TF32, and the CPU and GPU
# RNG streams differ, so absolute numbers from a CPU run are NOT comparable with a GPU run's;
# every comparison drawn from one of these runs is between configurations inside it, which is
# what the operating-point question needs. Smaller n than the GPU route, so wider CIs.
#
#   WHAT=c0  sbatch scripts/dna/operating_point/cpu_hedge.sbatch
#   WHAT=t18 sbatch scripts/dna/operating_point/cpu_hedge.sbatch
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

WHAT="${WHAT:-c0}"
NFE="${NFE:-32}"
case "${WHAT}" in
  c0)
    N_SAMPLES="${N_SAMPLES:-32}"
    OUT="${OUT:-results/dna/operating_point/fig7_grid_cpu_nfe${NFE}_n${N_SAMPLES}}"
    CONFIGS=""
    for STEPS in 1 2 4 8; do
      MC=$(( NFE / STEPS ))
      if [ "${STEPS}" = "1" ]; then DM="diag_fm1"; else DM="dmfm4_fm${STEPS}"; fi
      CONFIGS="${CONFIGS} ${DM}@norm@mc=${MC} glass_euler${STEPS}@norm@mc=${MC}"
    done
    ${PY} -u -m dmfm.experiments.operating_point_c0 --configs ${CONFIGS} \
      --targets "${TARGETS:-0.30}" --n_samples "${N_SAMPLES}" --batch_size 8 --mc_chunk 8 \
      --guidance_frac "${GF:-8.0}" --device cpu \
      --out_dir "${OUT}/target_$(printf '%s' "${TARGETS:-0.30}" | tr -- ' -.' '_mp')"
    ;;
  t18)
    OUT="${OUT:-results/dna/operating_point/table18_grid_cpu/${REWARD:-gc}}"
    ${PY} -u -m dmfm.experiments.operating_point_gradient_grid \
      --length "${LENGTH:-50}" --reward "${REWARD:-gc}" --output_dir "${OUT}" \
      --device cpu --n_probes "${N_PROBES:-8}" --n_repeats "${N_REPEATS:-2}" --mc_chunk 16
    ;;
  t13)
    OUT="${OUT:-results/dna/operating_point/table13_grid_cpu/${REWARD:-motif}_nfe${NFE}}"
    ${PY} -u -m dmfm.experiments.operating_point_reward_grid \
      --length "${LENGTH:-50}" --reward "${REWARD:-motif}" --out_dir "${OUT}" \
      --nfe "${NFE}" --n_samples "${N_SAMPLES:-32}" --mc_chunk 8 --device cpu \
      --student "${STUDENT:-convention}"
    ;;
  fig4)
    OUT="${OUT:-results/dna/operating_point/fig4_grid_cpu}"
    ${PY} -u "${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/operating_point/fig4_grid.py" \
      --nfe ${NFE} --steps ${STEPS:-1 2 4 8} --n-outputs "${N_OUTPUTS:-8}" --repeats 1 \
      --out-dir "${OUT}" --student "${STUDENT:-convention}" --device cpu
    ;;
  *) echo "unknown WHAT=${WHAT}" >&2; exit 2 ;;
esac
