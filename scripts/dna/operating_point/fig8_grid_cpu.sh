#!/bin/bash
# Fig 8's single-trajectory trace over the (steps, MC) grid at matched value-gradient NFE 32.
# CPU and deterministic (the initial noise and the MC pool come from seeded CPU generators),
# which is the route docs/dmfm_demo_reruns.md used for the recorded Fig 8 run. Same settings as
# scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch except for the (nfe_value, mc) split.
set -euo pipefail
cd "${MULTIMFM_ROOT:-$(dirname "$0")/../../..}"
PY="${PY:-$HOME/micromamba/envs/multimfm/bin/python}"
NFE="${NFE:-32}"
OUT_ROOT="${OUT_ROOT:-results/dna/operating_point/fig8_grid_nfe${NFE}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
for ARM in dmfm glass; do
  for STEPS in 1 2 4 8; do
    MC=$(( NFE / STEPS ))
    OUT="${OUT_ROOT}/${ARM}_s${STEPS}mc${MC}"
    echo "=== ${ARM} steps=${STEPS} mc=${MC} -> ${OUT}"
    ${PY} -u -m dmfm.regen.guidance_trace \
      --posterior "${ARM}" --grad_normalize --skip_existing \
      --sample_ids ${SAMPLE_IDS:-0 1 2 3} --target_c0 "${TARGET_C0:-0.30}" --seed "${SEED:-0}" \
      --mc "${MC}" --mc_chunk "${MC}" --nfe_traj 64 --nfe_value "${STEPS}" \
      --t_max 0.95 --guide_t_start 0.50 --guide_t_end 0.95 \
      --guidance_frac 8.0 --coeff_cap 10.0 --grad_clip 10.0 \
      --reward_sigma 0.15 --reward_scale 0.5 \
      --device cpu --out_dir "${OUT}"
  done
done
