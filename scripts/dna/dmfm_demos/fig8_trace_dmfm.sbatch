#!/bin/bash
#SBATCH --job-name=dna_fig8_dmfm
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Fig 8 rerun with the dMFM posterior (REPRODUCIBILITY_PLAN.md S7 item 8).
#
# WHAT THE PUBLISHED FIGURE RAN. The producer of figures/steering.png is lost; the
# regeneration `dmfm.regen.guidance_trace` (Slurm 49298115, cancelled 2026-09-29) had no dMFM
# code path at all -- it called `api.glass_posterior_fn`, so its three panels described GLASS
# on the base DFM at the recovered launcher settings MC=128, nfe_value=12, nfe_traj=128,
# guidance window [0.0, 0.95], guidance_frac 1.0.
#
# WHAT THIS RUNS. The same three panels with `--posterior dmfm` at the campaign's operating
# point, so Fig 8 illustrates the trajectory that Table 1 / Fig 3 / Fig 7 actually sample:
#   * 4 composed flow-map steps of paths.dmfm_ckpt_for_steps(50, 4) (convention 1);
#   * --grad_normalize, the scale-free gradient rule (convention 2);
#   * MC = 8, nfe_traj 64, t_max 0.95, guidance window [0.5, 0.95], guidance_frac 8,
#     coeff_cap 10, grad_clip 10, reward_sigma 0.15, reward_scale 0.5, target C0 = 0.30.
# The matched GLASS control (ARM=glass) runs the same settings with the GLASS posterior, as
# convention 2 requires wherever a GLASS comparison is shown.
#
# SAMPLE SELECTION, fixed before any output was inspected: seed 0, sample_ids 0 1 2 3 are all
# recorded, and the figure shows **sample_idx 0**. Nothing is chosen on appearance.
#
# DEVICE. The initial noise and the MC pool are drawn by seeded CPU generators, so a CPU run
# and a GPU run share identical randomness; a CPU run is additionally deterministic (the GPU
# value-gradient backward is not). Fig 8 illustrates one trajectory and no published number
# depends on it, so DEVICE=cpu is the default route here and is what the recorded run used.
#
#   DEVICE=cpu bash scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch     # ~6 min, deterministic
#   sbatch scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch              # GPU, ~1 min
#   ARM=glass sbatch scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch    # matched GLASS control
set -euo pipefail
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

ARM="${ARM:-dmfm}"
POSTERIOR="${POSTERIOR:-${ARM}}"   # ARM names the output subdir; POSTERIOR the sampler
DEVICE="${DEVICE:-$(command -v nvidia-smi >/dev/null 2>&1 && echo cuda || echo cpu)}"
MC="${MC:-8}"
NFE_VALUE="${NFE_VALUE:-4}"
NFE_TRAJ="${NFE_TRAJ:-64}"
OUT_ROOT="${OUT_ROOT:-results/dna/dmfm_demos/fig8}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${ARM}}"

mkdir -p "${OUT_ROOT}"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_NAME:-local}" \
  "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}" "" \
  "Fig8 arm=${ARM} device=${DEVICE} mc=${MC} nfe_value=${NFE_VALUE} nfe_traj=${NFE_TRAJ} target=${TARGET_C0:-0.30} gf=${GUIDANCE_FRAC:-8} grad_normalize=1" \
  >> "${OUT_ROOT}/jobs.tsv"

set -- --posterior "${POSTERIOR}" \
       --sample_ids ${SAMPLE_IDS:-0 1 2 3} \
       --target_c0 "${TARGET_C0:-0.30}" \
       --seed "${SEED:-0}" \
       --mc "${MC}" --mc_chunk "${MC_CHUNK:-${MC}}" \
       --nfe_traj "${NFE_TRAJ}" --nfe_value "${NFE_VALUE}" \
       --t_max 0.95 \
       --guide_t_start "${GUIDE_T_START:-0.50}" --guide_t_end "${GUIDE_T_END:-0.95}" \
       --guidance_frac "${GUIDANCE_FRAC:-8.0}" \
       --coeff_cap 10.0 --grad_clip 10.0 \
       --reward_sigma 0.15 --reward_scale 0.5 \
       --device "${DEVICE}" \
       --out_dir "${OUT_DIR}" --skip_existing
# GRAD_NORMALIZE=0 restores the stock clip-down rule, used only for the
# "what the published regeneration ran" reference row:
#   ARM=glass_launcher POSTERIOR=glass GRAD_NORMALIZE=0 MC=128 NFE_VALUE=12 NFE_TRAJ=128 \
#     GUIDE_T_START=0.0 GUIDANCE_FRAC=1.0 sbatch scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch
if [ "${GRAD_NORMALIZE:-1}" = "1" ]; then set -- "$@" --grad_normalize; fi

echo "arm=${ARM} device=${DEVICE} -> ${OUT_DIR}"
${PY} -u -m dmfm.regen.guidance_trace "$@"
