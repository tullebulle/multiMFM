#!/bin/bash
#SBATCH --job-name=dna_rescue
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# "Is the dMFM-vs-GLASS deficit our error?" -- docs/dmfm_deficit_rescue.md.
# Two symmetric experiments; EXP picks one, the array index picks the target
# (so a preempted task reruns one target only). Every knob is given to BOTH arms.
#
#   EXP=rule   sbatch scripts/dna/rescue/deficit_rescue.sbatch   # guidance rule x strength
#   EXP=nfe    sbatch scripts/dna/rescue/deficit_rescue.sbatch   # posterior steps x MC at matched NFE
#   EXP=matched N_SAMPLES=600 sbatch scripts/dna/rescue/deficit_rescue.sbatch  # headline, larger n
#
# A task whose sample_scores.csv exists is skipped, so resubmitting fills gaps.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

EXP="${EXP:-rule}"
N_SAMPLES="${N_SAMPLES:-200}"
SEED="${SEED:-0}"
TARGETS=(1.0 -1.0)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
TARGET="${TARGET:-${TARGETS[${TASK}]}}"
TAG="$(printf '%s' "${TARGET}" | tr -- '-.' 'mp')"

case "${EXP}" in
  rule)
    # Guidance-rule and guidance-strength grid, identical for both arms.
    #   @norm    = the scale-free rule of the n=1000 runs (every step rescaled to grad_clip)
    #   @relnorm = scale-free ACROSS arms but magnitude-preserving WITHIN an arm
    # glass_euler4 with no modifier is the published Table 1 clip-down rule, for reference.
    CONFIGS="${CONFIGS:-glass_euler4 \
      glass_euler4@norm@gf=8 glass_euler4@norm@gf=16 glass_euler4@norm@gf=32 \
      dmfm4_fm4@norm@gf=8 dmfm4_fm4@norm@gf=16 dmfm4_fm4@norm@gf=32 \
      glass_euler4@relnorm@gf=8 glass_euler4@relnorm@gf=16 \
      dmfm4_fm4@relnorm@gf=8 dmfm4_fm4@relnorm@gf=16 \
      diag_fm4@norm@gf=8}"
    ;;
  nfe)
    # Posterior step count (hence posterior accuracy) and MC budget, under the one rule both
    # arms share. The value gradient costs mc x n_steps network calls, so the @mc rows spend a
    # cheaper posterior's saved NFE on more MC draws at the SAME total NFE as glass_euler4@mc=8.
    CONFIGS="${CONFIGS:-glass_euler2@norm glass_euler4@norm glass_euler8@norm \
      dmfm4_fm1@norm dmfm4_fm2@norm dmfm4_fm4@norm dmfm4_fm8@norm \
      diag_fm1@norm diag_fm4@norm \
      dmfm4_fm1@norm@mc=32 dmfm4_fm2@norm@mc=16 glass_euler2@norm@mc=16}"
    ;;
  matched)
    # The headline comparison only, at larger n: each arm at its own best split of the SAME
    # value-gradient NFE (32 network calls), plus the GLASS arm of the n=1000 run and the
    # convention-compliant one-step dMFM artifact (the diagonal student).
    CONFIGS="${CONFIGS:-glass_euler4@norm glass_euler2@norm@mc=16 \
      dmfm4_fm1@norm@mc=32 diag_fm1@norm@mc=32}"
    ;;
  *) echo "EXP must be rule|nfe|matched, got '${EXP}'" >&2; exit 2 ;;
esac

OUT_DIR="${OUT_DIR:-results/dna/deficit_rescue/${EXP}_L50_n${N_SAMPLES}/target_${TAG}}"
echo "EXP=${EXP} target=${TARGET} n=${N_SAMPLES} -> ${OUT_DIR}"
if [[ -f "${OUT_DIR}/sample_scores.csv" ]]; then echo "already done"; exit 0; fi

${PY} -u -m dmfm.experiments.rescue_dmfm_glass_c0 \
  --out_dir "${OUT_DIR}" --seed "${SEED}" \
  --n_samples "${N_SAMPLES}" --batch_size 8 --targets "${TARGET}" \
  --configs ${CONFIGS}
