#!/usr/bin/env bash
#SBATCH --job-name=geom_fp_tanimoto
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# The GEOM fingerprint task, with the two fixes that have never been tried.
#
# The comparison that matters is the one TFG-Flow itself claims: the RELATIVE improvement of
# guided over the same method's own unconditional baseline. Theirs is 0.170 -> 0.208,
# +22.35%. Ours is currently 0.2287 -> 0.2241, -2.0%. Absolute Tanimoto is not comparable
# across the two papers -- their generator caps at 30 heavy atoms against our 71, and they
# score invalid molecules 0 and map unresolved MASK atoms to hydrogen -- but the ratio is,
# because both arms of each ratio share whatever protocol produced them.
#
# Fix 1: the objective. We have been optimising -||p - c||^2, whose Spearman correlation
# with the counted FP2 Tanimoto is 0.375 over composition candidates, against 0.708 for the
# differentiable Tanimoto. Optimising the better-ranked surrogate is the obvious thing to
# try and it has never been run in-trajectory.
#
# Fix 2: the operating point. mfm_steps has never been swept here, exactly as it had not
# been on QM9 or on the R_g task, where the inherited value was several times the optimum.
#
# The reward scales differ by orders of magnitude -- the L2 form sums over 1,024 bits while
# the Tanimoto is in [0, 1] -- so reward_lambda is swept per reward rather than carried over.
#
# Every run reports its own unguided arm from the same noise and targets, so the relative
# improvement is paired, and PoseBusters validity, because the composition rescue showed
# this metric can be moved by destroying molecules.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

N="${N:-64}"
OUT="results/geom/runs/fp_tanimoto_sweep"
I="${SLURM_ARRAY_TASK_ID:-0}"
run() {  # reward lambda mu mfm_steps guide_max_t tag
  python -u -m multimfm.geom.mfm_steering \
    --num-mols "$N" --reward-kind "$1" --reward-lambda "$2" --mu "$3" \
    --mfm-steps "$4" --guide-min-t 0.1 --guide-max-t "$5" --guide-every 2 \
    --value-samples 4 --value-batch-size 8 --sample-steps 100 \
    --target-selection filter --device cuda --pb \
    --output-dir "${OUT}" --tag "$6" --seed 0 || echo "FAILED $6"
}

case "${I}" in
  0) for LAM in 1 5 20; do run fingerprint_tanimoto "$LAM" 0.5 2 0.9 "tan_lam${LAM}_mu0.5_s2"; done ;;
  1) for LAM in 1 5 20; do run fingerprint_tanimoto "$LAM" 2.0 2 0.9 "tan_lam${LAM}_mu2_s2"; done ;;
  2) for LAM in 5 20; do run fingerprint_tanimoto "$LAM" 2.0 2 0.6 "tan_lam${LAM}_mu2_s2_early"; done
     run fingerprint_tanimoto 5 0.5 1 0.9 "tan_lam5_mu0.5_s1" ;;
  3) run fingerprint 0.02 0.5 2 0.9 "l2_inherited_s2"        # the old objective, cheaper point
     run fingerprint 0.02 2.0 2 0.9 "l2_mu2_s2"
     run fingerprint 0.02 0.5 10 0.9 "l2_inherited_s10" ;;   # the published configuration
esac
echo "done task ${I}"
