#!/usr/bin/env bash
#SBATCH --job-name=nfe_sweep
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# QM9 analogue of the DNA "Inference-time Steering Comparison" figure: every method
# swept ACROSS its compute budget so each one is a curve, not a point.
#
#   multiMFM / multiMFM-SS   MC = value_samples in {1,2,4,8,16,32}  -> the MC=k curve
#   best-of-N                N in {4,8,16,32,64,128}
#   Feynman-Kac              K in {4,8,16,32}, lambda RE-BRACKETED (see below)
#   DPS                      mu swept for quality, guide_every swept for budget
#
# FK lambda: the pilot swept {0.3,1,3} and everything collapsed -- for alpha the
# reward -(pred-target)^2 is O(10-100), so lambda>=0.3 saturates the softmax, one
# particle takes all the weight and the particle set loses the diversity that the
# final selection needs. Re-bracketed two orders of magnitude down.
# DPS mu: the pilot was still improving at its top value (0.1), so it is extended.
# Ran as Slurm job 48559075 (2026-09-25) from the earlier repo location; summaries are
# collected in results/qm9/baselines/ (not part of the paper as of 2026-09-29).
# SUPERSEDED (2026-09-29): this is the historical script behind Slurm job 48559075; its
# figures (outputs/pilot/figures/make_figs.py) plotted power-law-EXTRAPOLATED best-of-N
# points and hardcoded numbers. The fixed, fully measured sweep (hook-counted NFE for
# every method incl. multiMFM, best-of-N up to N=512, 3 held-out seeds) is
# scripts/qm9/baseline_sweep_v2.sbatch; its figure is built from results/qm9/baselines/
# only (python -m multimfm.qm9_figures baselines). multimfm.baselines itself changed
# (lazy PB selection, hook NFE); the steer_search calls below pin
# --lookahead-source base_endpoint, the harvesting these runs used.
set -euo pipefail

# Submit from the repo root (Slurm runs a copy of this script, so the repo is
# located via $SLURM_SUBMIT_DIR, or $MULTIMFM_ROOT if set).
REPO="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "${REPO}/scripts/env.sh" ]; then
  echo "cannot find the repo at ${REPO}; submit from the repo root or set MULTIMFM_ROOT" >&2
  exit 1
fi
cd "${REPO}"
# shellcheck source=/dev/null
source scripts/env.sh
mkdir -p logs outputs/sweep

N=100
STEPS=128
PROP=alpha
SEED=8
OUT="${REPO}/outputs/sweep"

bl() { python -m multimfm.baselines --property-name ${PROP} --num-samples ${N} \
       --sample-steps ${STEPS} --seed ${SEED} --pb-workers 16 "$@" 2>&1 \
       | grep -viE "warning|pkg_resources"; }

echo "################ BASE ################"
bl --method base --output-dir "${OUT}/base"

echo "################ BEST-OF-N ################"
for K in 4 8 16 32 64 128; do
  echo "----- best_of_n N=${K} -----"
  bl --method best_of_n --n-candidates ${K} --output-dir "${OUT}/bon_n${K}"
done

echo "################ FEYNMAN-KAC (re-bracketed lambda) ################"
for K in 4 8 16 32; do
  for LAM in 0.003 0.01 0.03; do
    echo "----- fk K=${K} lam=${LAM} -----"
    bl --method fk --n-particles ${K} --fk-lambda ${LAM} --fk-resample-every 8 \
       --output-dir "${OUT}/fk_k${K}_lam${LAM}"
  done
done

echo "################ DPS (mu for quality) ################"
for MU in 0.03 0.1 0.3 1.0; do
  echo "----- dps mu=${MU} -----"
  bl --method dps --mu ${MU} --output-dir "${OUT}/dps_mu${MU}"
done
echo "################ DPS (guide_every for budget, at mu=0.1) ################"
for GE in 2 4 8; do
  echo "----- dps guide_every=${GE} -----"
  bl --method dps --mu 0.1 --guide-every ${GE} --output-dir "${OUT}/dps_ge${GE}"
done

echo "################ multiMFM / multiMFM-SS (MC sweep) ################"
# steer_search emits final_endpoint (= multiMFM) and best_lookahead_pbvalid (= SS)
# from one run, on the identical target set.
for MC in 1 2 4 8 16 32; do
  echo "----- multiMFM MC=${MC} -----"
  python -m multimfm.steer_search --property-name ${PROP} --num-samples ${N} \
    --sample-steps ${STEPS} --seed ${SEED} \
    --mu 11.0 --reward-scale 0.3 --select-min-t 0.3 --lookahead-source base_endpoint \
    --mfm-diagonal --guide-every 1 --value-samples ${MC} --value-glass-steps 4 \
    --pb-workers 16 --output-dir "${OUT}/mmfm_mc${MC}" 2>&1 \
    | grep -viE "warning|pkg_resources"
done

echo "################ DONE ################"
ls -R "${OUT}" | head -40
