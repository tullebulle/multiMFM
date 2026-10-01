#!/usr/bin/env bash
#SBATCH --job-name=dna_fig4_split
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Sweep the (guidance window, MC) split at each budget on the rare DNA target.
#
# The dMFM's cost is 96 + G (MC + 1.1), so a budget can be spent many ways: a wide
# guidance window with few posterior samples, or a narrow one with many. The hard-target
# run used a single split per budget, chosen only so the budget landed on the ladder.
#
# That is the one knob this paper has repeatedly found to matter most and the sweep here
# never touched: on Table 1 moving 4 steps x MC 8 to 1 step x MC 32 at the SAME cost moved
# oracle MAE from 0.193 to 0.153, and the rule that came out of it was to match the split
# to the estimator, not to inherit it. DPS has no such knob -- it is one denoiser call per
# guided step -- so if the dMFM is being run at the wrong split this comparison is not
# measuring the method.
#
# Target y* = +1, held-out oracle, strength pilot per split as before.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

TGT=1.0
NOUT="${NOUT:-200}"
REPS="${REPS:-3}"
I="${SLURM_ARRAY_TASK_ID:-0}"
ROOT="outputs/dna/fig4/split_sweep_t${TGT}"

# per budget: "planIndex:t_start" alternatives (plan index 0..4 = MC 1,2,4,8,16)
case "${I}" in
  0) B=192; SPLITS=("1:0.635" "2:0.760" "3:0.844") ;;
  1) B=288; SPLITS=("0:0.010" "2:0.562" "3:0.740") ;;
  2) B=480; SPLITS=("3:0.521" "4:0.729") ;;
  3) B=864; SPLITS=("4:0.490") ;;
esac

for spec in "${SPLITS[@]}"; do
  idx="${spec%%:*}"; ts="${spec##*:}"
  tag="idx${idx}_t${ts}"
  P="${ROOT}/pilot/b${B}_${tag}"
  for GF in 2 4 8 16 32; do
    python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${idx}" --repeat 0 \
      --out-dir "${P}/gf${GF}" --n-outputs 100 --seed 777 --target-c0 "${TGT}" \
      --guide-t-start "${ts}" --guidance-frac "${GF}" --grad-normalize || true
  done
  BEST=$(python - "${P}" <<'PY'
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
  echo "budget ${B} split ${tag}: gf=${BEST}"
  for REP in $(seq 0 $((REPS-1))); do
    python -u -m dmfm.benchmarks.run_steering --plan sweep --index "${idx}" --repeat "${REP}" \
      --out-dir "${ROOT}/final/b${B}_${tag}" --n-outputs "${NOUT}" --seed 0 --target-c0 "${TGT}" \
      --guide-t-start "${ts}" --guidance-frac "${BEST}" --grad-normalize || true
  done
done
echo "done budget ${B}"
