#!/usr/bin/env bash
#SBATCH --job-name=dna_fig4_hard
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# The steering comparison at the target the rest of the DNA work actually uses.
#
# At C0 = 0.30 the comparison is guide-limited and cannot separate the methods. Every
# arm steers with the same guide and is scored by a held-out oracle, and DPS drives the
# guide to 0.039 -- essentially perfect -- landing at oracle 0.144. That 0.144 is the
# guide-oracle disagreement: no method that only sees the guide can go below it, so it
# is a ceiling on guidance rather than a floor anyone beats. DPS reaches it at the
# cheapest budget and is then flat across a 4.5x budget range, while the dMFM walks down
# to it from above. Nothing can be distinguished there.
#
# 0.30 is also a mild target: the base model's C0 has mean -0.177 and sd 0.438, so 0.30
# is 1.1 sd out. Table 1, Figure 3 and Figure 7 all use y* = +-1, which is 2.7 sd out --
# rare enough that a one-step denoiser approximation of the endpoint should be a poor
# guide to where the posterior actually has mass, which is the regime a Monte-Carlo
# value estimate exists for.
#
# Same ladder, same protocol, same held-out oracle; only the target changes.
# Tasks: 0 = dMFM (scale-free), 1 = DPS, 2 = best-of-N, 3 = Feynman-Kac.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

TGT="${TGT:-1.0}"
NOUT="${NOUT:-200}"
REPS="${REPS:-3}"
I="${SLURM_ARRAY_TASK_ID:-0}"
B=(192 288 480 864); IDX=(0 1 2 3); TS=(0.4792 0.3125 0.1771 0.0833)
ROOT="outputs/dna/fig4/hard_target_t${TGT}"

pilot_pick() {  # dir -> best guidance_frac by GUIDE metric
  python - "$1" <<'PY'
import json, sys, pathlib
best, lab = None, None
for f in pathlib.Path(sys.argv[1]).rglob("*.json"):
    if f.name == "calibration.json":
        continue
    d = json.loads(f.read_text())
    m = (d.get("metrics") or {}).get("mae")
    gf = (d.get("config_params") or {}).get("guidance_frac")
    if m is None:
        continue
    if gf is None:
        gf = float(f.parent.name.replace("gf", ""))
    if best is None or m < best:
        best, lab = m, gf
print(lab if lab is not None else 8.0)
PY
}

case "${I}" in
  0)  # dMFM, scale-free gradient rule, strength pilot then ladder
    for j in 0 1 2 3; do
      b="${B[$j]}"; P="${ROOT}/dmfm_pilot/b${b}"
      for GF in 2 4 8 16 32 64; do
        python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${IDX[$j]}" --repeat 0 \
          --out-dir "${P}/gf${GF}" --n-outputs 100 --seed 777 --target-c0 "${TGT}" \
          --guide-t-start "${TS[$j]}" --guidance-frac "${GF}" --grad-normalize || true
      done
      BEST=$(pilot_pick "${P}"); echo "dmfm budget ${b}: gf=${BEST}"
      for REP in $(seq 0 $((REPS-1))); do
        python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${IDX[$j]}" --repeat "${REP}" \
          --out-dir "${ROOT}/dmfm/b${b}" --n-outputs "${NOUT}" --seed 0 --target-c0 "${TGT}" \
          --guide-t-start "${TS[$j]}" --guidance-frac "${BEST}" --grad-normalize || true
      done
    done ;;
  1)  # DPS, same protocol
    S=(96 96 160 288); T2=(0.479 0.01 0.01 0.01)
    for j in 0 1 2 3; do
      b="${B[$j]}"; P="${ROOT}/dps_pilot/b${b}"
      for GF in 2 4 8 16 32 64; do
        python -u scripts/dna/run_dps_steering.py --n-steps "${S[$j]}" --guide-t-start "${T2[$j]}" \
          --guidance-frac "${GF}" --n-outputs 100 --batch-size 50 --repeat 0 --seed 777 \
          --target-c0 "${TGT}" --label "b${b}_gf${GF}" --out-dir "${P}" || true
      done
      BEST=$(pilot_pick "${P}"); echo "dps budget ${b}: gf=${BEST}"
      for REP in $(seq 0 $((REPS-1))); do
        python -u scripts/dna/run_dps_steering.py --n-steps "${S[$j]}" --guide-t-start "${T2[$j]}" \
          --guidance-frac "${BEST}" --n-outputs "${NOUT}" --batch-size 50 --repeat "${REP}" --seed 0 \
          --target-c0 "${TGT}" --label "b${b}" --out-dir "${ROOT}/dps/b${b}" || true
      done
    done ;;
  2|3)
    FAM=$([ "${I}" = 2 ] && echo best_of_n || echo fk)
    mapfile -t IDXS < <(python - "${FAM}" <<'PY'
import sys
from dmfm.benchmarks import plan
print("\n".join(str(i) for i, j in enumerate(plan.PLANS["sweep"]) if j.method == sys.argv[1]))
PY
)
    for idx in "${IDXS[@]}"; do
      for REP in $(seq 0 $((REPS-1))); do
        python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${idx}" --repeat "${REP}" \
          --out-dir "${ROOT}/baselines" --n-outputs "${NOUT}" --seed 0 --target-c0 "${TGT}" || true
      done
    done ;;
esac
echo "done task ${I}"
