#!/usr/bin/env bash
#SBATCH --job-name=dna_dps_ladder
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
# DPS as a steering baseline on DNA, so both panels compare against the same methods.
#
# The DNA comparison ran best-of-N, Feynman-Kac, beam and MCTS; the QM9 one ran
# best-of-N, Feynman-Kac and DPS. The two sets differ for no reason beyond which sweep
# each was written in, and DPS is the one that matters most -- it is the standard
# gradient-guidance baseline, and it is the arm multiMFM most directly replaces. The
# DNA value-estimation panel already scores the same DPS estimator, so this only turns
# it into a sampler (src/dmfm/benchmarks/dps_steering.py).
#
# DPS has no sampling knob, so its budget is set by the integration steps and the
# guidance window -- the same two knobs the QM9 sweep gives it. These four land on the
# ladder the other DNA methods sit on:
#
#   192 -> 96 steps,  window from 0.479      480 -> 160 steps, full window
#   288 -> 96 steps,  full window            864 -> 288 steps, full window
#
# Each task pilots the guidance strength at n = 100 (one repeat, scored on the GUIDE,
# which is what the sampler optimises) and then runs the winner at n = 300 x 5 repeats,
# matching how the other baselines were tuned and the sample size of the comparison.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

I="${SLURM_ARRAY_TASK_ID:-0}"
B=(192 288 480 864); STEPS=(96 96 160 288); TS=(0.479 0.01 0.01 0.01)
b="${B[$I]}"; steps="${STEPS[$I]}"; ts="${TS[$I]}"
PILOT="outputs/dna/fig4/dps_pilot/b${b}"
FINAL="outputs/dna/fig4/dps/b${b}"
R="scripts/dna/run_dps_steering.py"

echo "===== budget ${b}: steps=${steps} t_start=${ts}  -- pilot over guidance strength"
for GF in 2 4 8 16 32; do
  python -u "${R}" --n-steps "${steps}" --guide-t-start "${ts}" --guidance-frac "${GF}" \
    --n-outputs 100 --batch-size 50 --repeat 0 --seed 777 \
    --label "b${b}_gf${GF}" --out-dir "${PILOT}" || echo "pilot gf=${GF} FAILED"
done

BEST=$(python - "${PILOT}" <<'PY'
import json, sys, pathlib
best, lab = None, None
for f in pathlib.Path(sys.argv[1]).glob("*.json"):
    d = json.loads(f.read_text())
    m = d["metrics"]["mae"]                      # guide metric: what the sampler optimises
    if best is None or m < best:
        best, lab = m, d["config_params"]["guidance_frac"]
print(lab if lab is not None else 8.0)
PY
)
echo "budget ${b}: pilot chose guidance_frac=${BEST}"

for REP in 0 1 2 3 4; do
  python -u "${R}" --n-steps "${steps}" --guide-t-start "${ts}" --guidance-frac "${BEST}" \
    --n-outputs 300 --batch-size 50 --repeat "${REP}" --seed 0 \
    --label "b${b}" --out-dir "${FINAL}" || echo "final rep=${REP} FAILED"
done
echo "done budget ${b}"
