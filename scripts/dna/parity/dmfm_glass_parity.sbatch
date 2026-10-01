#!/bin/bash
#SBATCH --job-name=dna_parity
#SBATCH --partition=gpu_test
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# dMFM-vs-GLASS parity for the Table 1 C0-guidance experiment
# (docs/dmfm_glass_parity.md). Two modes, selected by MODE:
#
#   MODE=grad   sbatch scripts/dna/parity/dmfm_glass_parity.sbatch
#   MODE=sample N_SAMPLES=32 sbatch scripts/dna/parity/dmfm_glass_parity.sbatch
#
# Everything is matched across configurations: same initial noise (seed+i), same frozen
# MC pool (seed+1e6+i), same guide, same sampler, same Table 1 guidance settings; only the
# posterior sampler (and, where a token says so, one named knob) changes.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

MODE="${MODE:-grad}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-results/dna/parity/${MODE}_L50}"

if [[ "${MODE}" == "grad" ]]; then
  ${PY} -u -m dmfm.experiments.parity_dmfm_glass_c0 \
    --mode grad --seed "${SEED}" --out_dir "${OUT_DIR}" \
    --n_probes "${N_PROBES:-8}" --probe_times ${PROBE_TIMES:-0.5 0.7 0.9} \
    --reference_steps "${REFERENCE_STEPS:-100}" --targets 1.0 -1.0
else
  ${PY} -u -m dmfm.experiments.parity_dmfm_glass_c0 \
    --mode sample --seed "${SEED}" --out_dir "${OUT_DIR}" \
    --n_samples "${N_SAMPLES:-32}" --batch_size 8 --targets 1.0 -1.0 \
    --configs ${CONFIGS:-glass_euler4 diag_fm4 diag_fm1 dmfm4_fm4 dmfm4_fm1 dmfm4_fm4_e999 dmfm4_fm4@norm diag_fm4@norm glass_euler4@norm}
fi
