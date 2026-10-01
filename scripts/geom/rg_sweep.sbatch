#!/usr/bin/env bash
#SBATCH --job-name=geom_rg_sweep
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Find the operating point for R_g steering on GEOM, and measure the baselines on the same
# budget ladder.
#
# The inherited GEOM configuration is 10 inner flow-map steps with 4 posterior samples. Its
# cost is 100 + 3 G S MC per molecule with G = 80 guided steps, i.e. **9,700**
# forward-equivalents, of which 9,600 are the value estimator -- so it competes against
# best-of-64 (R_g error 0.090 at 6,400) and loses badly at 0.227. The same configuration at
# S = 1, MC = 1 costs **340** and competes against best-of-4 (0.325).
#
# Inner steps have never been swept on GEOM, exactly as they had never been swept on QM9,
# where the inherited 4 turned out to be 2-4x more than the optimum. Guidance strength has
# likewise only ever been tuned for the fingerprint reward, which is a different objective.
#
# Targets are +3 sd into each molecule's OWN size-conditional R_g distribution; a global
# target would select on molecule size, which carries half the pooled spread.
#
# PoseBusters validity is NOT measured here -- this sweep only locates the operating point.
# Nothing is claimed from it until the chosen configuration is re-run with validity, since
# stretching a molecule is the obvious way to raise R_g and exactly what PB penalises.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
R="scripts/geom/rg_benchmark.py"
N="${N:-64}"
I="${SLURM_ARRAY_TASK_ID:-0}"

case "${I}" in
  0) for MU in 0.5 1 2 4; do python -u "$R" --method multimfm --n "$N" --batch 16 \
        --mfm-steps 1 --value-samples 1 --mu "$MU" --tag "mmfm_s1_mc1_mu${MU}" || true; done ;;
  1) for MU in 0.5 1 2 4; do python -u "$R" --method multimfm --n "$N" --batch 16 \
        --mfm-steps 1 --value-samples 2 --mu "$MU" --tag "mmfm_s1_mc2_mu${MU}" || true; done ;;
  2) for MU in 0.5 1 2 4; do python -u "$R" --method multimfm --n "$N" --batch 8 \
        --mfm-steps 2 --value-samples 1 --mu "$MU" --tag "mmfm_s2_mc1_mu${MU}" || true; done ;;
  3) for MU in 0.5 1 2 4; do python -u "$R" --method multimfm --n "$N" --batch 8 \
        --mfm-steps 2 --value-samples 2 --mu "$MU" --tag "mmfm_s2_mc2_mu${MU}" || true; done ;;
  4) for MU in 0.5 1 2 4; do python -u "$R" --method dps --n "$N" --batch 16 \
        --mu "$MU" --tag "dps_mu${MU}" || true; done ;;
  5) python -u "$R" --method base --n "$N" --batch 16 --tag base
     for K in 4 16 64; do python -u "$R" --method best_of_n --n "$N" --batch 4 \
        --n-candidates "$K" --tag "bon_n${K}" || true; done ;;
esac
echo "done task ${I}"
