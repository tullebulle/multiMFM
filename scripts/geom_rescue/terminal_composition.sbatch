#!/usr/bin/env bash
#SBATCH --job-name=geom_rescue_terminal
#SBATCH --account=kempner_albergo_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --constraint="cc8.0|cc9.0"
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-3%2
#
# Terminal composition guidance on generated molecules
# (docs/geom_rescue_attempt.md H2). On real GEOM val molecules a single
# element substitution chosen by the guide head with a metric-aligned
# (differentiable-Tanimoto) reward raises the counted FP2 Tanimoto by
# +0.0447 +- 0.0098 (n=24, 4.6 sigma); the oracle ceiling for one flip is
# +0.0746 +- 0.0076 and the paper's L2 reward gets only +0.0243 +- 0.0106.
# This job asks the load-bearing question: does that survive on the molecules the
# *generator* makes (off the head's training distribution), and does PoseBusters
# validity survive the retyping?
#
#   sbatch --partition=kempner_h200 scripts/geom_rescue/terminal_composition.sbatch
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

NUM_MOLS="${NUM_MOLS:-256}"
ORACLE_MOLS="${ORACLE_MOLS:-32}"
ROUNDS="${ROUNDS:-4}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "${IDX}" in
  0) SEED=0 ; FLIP=cnof ;;
  1) SEED=0 ; FLIP=all  ;;
  2) SEED=1 ; FLIP=cnof ;;
  3) SEED=1 ; FLIP=all  ;;
  *) echo "no arm"; exit 0 ;;
esac
OUT="results/geom/rescue/terminal_generated_seed${SEED}_${FLIP}_n${NUM_MOLS}.json"
mkdir -p results/geom/rescue
if [ -s "${OUT}" ]; then echo "already done: ${OUT}"; exit 0; fi

"${PY}" scripts/geom_rescue/steer_terminal_composition.py \
  --num-mols "${NUM_MOLS}" --oracle-mols "${ORACLE_MOLS}" --rounds "${ROUNDS}" \
  --sample-steps 100 --objective l2 tanimoto --eta 0.3 1.0 3.0 \
  --flip-set "${FLIP}" --seed "${SEED}" --device cuda --out "${OUT}"
echo "wrote ${OUT}"
