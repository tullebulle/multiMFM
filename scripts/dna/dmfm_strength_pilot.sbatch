#!/usr/bin/env bash
#SBATCH --job-name=dna_dmfm_strength
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
# Tune the dMFM's guidance strength the same way every baseline's was tuned.
#
# The baselines each got a pilot over their own hyperparameters at matched NFE; the dMFM
# arm has been run at the harness default (guidance_frac = 8) throughout. That asymmetry
# was invisible while the dMFM was the only guided method here, but DPS has now been added
# and its strength WAS pilot-tuned, so the comparison is no longer symmetric.
#
# Same protocol as scripts/dna/dps_ladder.sbatch: pilot over the strength at n = 100 on a
# separate seed, select on the GUIDE (what the sampler optimises and all that is available
# at deployment), then run the winner at n = 300 x 5 repeats and report the held-out
# ORACLE. Selecting on the oracle would be using the held-out evaluator to tune.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

I="${SLURM_ARRAY_TASK_ID:-0}"
# the four ladder budgets, as (plan index = MC setting, guidance window)
B=(192 288 480 864); IDX=(0 1 2 3); TS=(0.4792 0.3125 0.1771 0.0833)
b="${B[$I]}"; idx="${IDX[$I]}"; ts="${TS[$I]}"
PILOT="outputs/dna/fig4/dmfm_pilot/b${b}"
FINAL="outputs/dna/fig4/dmfm_tuned/b${b}"

echo "===== budget ${b}: plan index ${idx}, window from ${ts} -- strength pilot"
for GF in 2 4 8 16 32; do
  python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${idx}" --repeat 0 \
    --out-dir "${PILOT}/gf${GF}" --n-outputs 100 --seed 777 \
    --guide-t-start "${ts}" --guidance-frac "${GF}" || echo "pilot gf=${GF} FAILED"
done

BEST=$(python - "${PILOT}" <<'PY'
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
)
echo "budget ${b}: pilot chose guidance_frac=${BEST}"

for REP in 0 1 2 3 4; do
  python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${idx}" --repeat "${REP}" \
    --out-dir "${FINAL}" --n-outputs 300 --seed 0 \
    --guide-t-start "${ts}" --guidance-frac "${BEST}" || echo "final rep=${REP} FAILED"
done
echo "done budget ${b}"
