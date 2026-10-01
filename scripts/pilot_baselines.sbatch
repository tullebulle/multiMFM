#!/usr/bin/env bash
#SBATCH --job-name=bl_pilot
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Pilot for the same-backbone QM9 steering table. One property (alpha), one seed,
# n=100, real sampler settings (128 steps). Purpose: (a) confirm each method runs at
# scale, (b) find each method's own operating point, (c) measure NFE so the final
# table can be read at matched budget.
#
# Each method gets its OWN sweep -- the reviewer asks for an identical BUDGET and an
# identical evaluation, not identical hyperparameters. Handicapping a baseline with
# another method's mu would be the same mistake we are trying to correct.
# Ran as Slurm job 48045913 (2026-09-23) from the earlier repo location; summaries are
# collected in results/qm9/baselines/ (not part of the paper as of 2026-09-29).
# SUPERSEDED (2026-09-29): this is the historical script behind Slurm job 48045913; its
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
mkdir -p logs outputs/pilot

N=100
STEPS=128
PROP=alpha
SEED=8
OUT="${REPO}/outputs/pilot"

run() { echo "----- $* -----"; python -m multimfm.baselines --property-name ${PROP} \
        --num-samples ${N} --sample-steps ${STEPS} --seed ${SEED} \
        --pb-workers 16 --output-dir "${OUT}" "$@" 2>&1 | grep -viE "warning|pkg_resources"; }

echo "================ BASE ================"
run --method base

echo "================ DPS (mu sweep) ================"
for MU in 0.003 0.01 0.03 0.1; do
  run --method dps --mu ${MU} --output-dir "${OUT}/dps_mu${MU}"
done

echo "================ BEST-OF-N (N sweep) ================"
for K in 4 8 16; do
  run --method best_of_n --n-candidates ${K} --output-dir "${OUT}/bon_n${K}"
done

echo "================ FEYNMAN-KAC (K x lambda) ================"
for K in 8 16; do
  for LAM in 0.3 1.0 3.0; do
    run --method fk --n-particles ${K} --fk-lambda ${LAM} --fk-resample-every 8 \
        --output-dir "${OUT}/fk_k${K}_lam${LAM}"
  done
done

echo "================ multiMFM + multiMFM-SS (reference) ================"
# steer_search reports final_endpoint (= multiMFM) and best_lookahead_pbvalid (= SS)
# in one run, with the identical target set.
python -m multimfm.steer_search --property-name ${PROP} --num-samples ${N} \
  --sample-steps ${STEPS} --seed ${SEED} \
  --mu 11.0 --reward-scale 0.3 --select-min-t 0.3 --lookahead-source base_endpoint \
  --mfm-diagonal --guide-every 1 --value-samples 32 --value-glass-steps 4 \
  --pb-workers 16 --output-dir "${OUT}/steer_search" 2>&1 | grep -viE "warning|pkg_resources"

echo "================ PILOT TABLE ================"
python - "${OUT}" <<'PY'
import json, sys, glob, os
root = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
    d = json.load(open(f))
    if "summary" not in d:
        continue
    nfe = d.get("nfe") or {}
    a = d.get("args", {})
    for s in d["summary"]:
        knob = ""
        if "dps" in s["run"]:       knob = f"mu={a.get('mu')}"
        elif "best_of_n" in s["run"]: knob = f"N={a.get('n_candidates')}"
        elif "fk" in s["run"]:      knob = f"K={a.get('n_particles')},lam={a.get('fk_lambda')}"
        rows.append((s["run"], knob, s.get("oracle_mae"), s.get("pb_pb_valid"),
                     nfe.get("nfe_base_per_mol"), nfe.get("nfe_regressor_per_mol")))
print(f"{'method':<26}{'config':<18}{'oracleMAE':>11}{'PB':>7}{'baseNFE':>10}{'regNFE':>9}")
for r in sorted(rows, key=lambda x: (x[0], str(x[1]))):
    mae = f"{r[2]:.3f}" if r[2] is not None else "n/a"
    pb  = f"{r[3]:.3f}" if r[3] is not None else "n/a"
    bn  = f"{r[4]:.0f}" if r[4] is not None else "-"
    rn  = f"{r[5]:.0f}" if r[5] is not None else "-"
    print(f"{r[0]:<26}{r[1]:<18}{mae:>11}{pb:>7}{bn:>10}{rn:>9}")
print("\nsteer_search rows: final_endpoint = multiMFM, best_lookahead_pbvalid = multiMFM-SS")
print("NOTE n=100 is NOISY (memory: n=100 was optimistic vs n=1000 for homo/dipole).")
PY
