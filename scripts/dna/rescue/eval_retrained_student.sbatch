#!/bin/bash
#SBATCH --job-name=dna_rescue_evalckpt
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --array=0-1%1
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Score a re-distilled 4-step student against the SAME GLASS arm, same rule, same ids
# (docs/dmfm_deficit_rescue.md section 2). STUDENT is the checkpoint to test; the configuration
# names are unchanged, so `dmfm4_fm*` rows here are directly comparable with the stock run in
# results/dna/deficit_rescue/nfe_L50_n200/.
#
#   STUDENT=outputs/dna/deficit_rescue/rescue_L50_dmfm_4step_esd_upto025_by_val_loss/last.ckpt \
#     TAG=val_loss_60k sbatch scripts/dna/rescue/eval_retrained_student.sbatch
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

STUDENT="${STUDENT:?set STUDENT to the checkpoint to score}"
[[ -f "${STUDENT}" ]] || { echo "no such checkpoint: ${STUDENT}" >&2; exit 2; }
TAG="${TAG:-retrained}"
N_SAMPLES="${N_SAMPLES:-200}"
TARGETS=(1.0 -1.0)
TARGET="${TARGET:-${TARGETS[${SLURM_ARRAY_TASK_ID:-0}]}}"
T_TAG="$(printf '%s' "${TARGET}" | tr -- '-.' 'mp')"
# Both arms, both rules, and the step counts that matter: the 4 steps the shipped student is used
# at, plus 1 and 2 steps, plus the NFE-matched MC spend.
# diag_fm1 rows use the *diagonal* student, which --student_ckpt_dmfm4 does not touch, so they
# also fill the one cell the nfe sweep left out: the convention-compliant one-step dMFM artifact
# (paths.dmfm_ckpt_for_steps(50, 1)) at the NFE-matched MC budget.
CONFIGS="${CONFIGS:-glass_euler4@norm glass_euler2@norm dmfm4_fm4@norm dmfm4_fm2@norm \
  dmfm4_fm1@norm dmfm4_fm1@norm@mc=32 diag_fm1@norm@mc=32}"
OUT_DIR="${OUT_DIR:-results/dna/deficit_rescue/student_${TAG}_L50_n${N_SAMPLES}/target_${T_TAG}}"
echo "student=${STUDENT} target=${TARGET} -> ${OUT_DIR}"
if [[ -f "${OUT_DIR}/sample_scores.csv" ]]; then echo "already done"; exit 0; fi

${PY} -u -m dmfm.experiments.rescue_dmfm_glass_c0 \
  --out_dir "${OUT_DIR}" --seed "${SEED:-0}" \
  --n_samples "${N_SAMPLES}" --batch_size 8 --targets "${TARGET}" \
  --student_ckpt_dmfm4 "${STUDENT}" \
  --configs ${CONFIGS}
