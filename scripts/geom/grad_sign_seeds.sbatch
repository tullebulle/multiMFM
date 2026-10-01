#!/usr/bin/env bash
#SBATCH --job-name=geom_gradsign_seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Put error bars on the `grad-sign` step scan.
#
# A single `grad-sign` run reports one number per probe step, with no uncertainty, from
# n=4 molecules x 16 look-ahead samples. Its verdict string is then decided by the sign of
# one of those differences. This runs the identical scan over SEEDS independent draws
# (different molecules, targets and trajectory each time) and reports, per step, the mean
# and SEM of the *paired* delta (probe minus the step-0 value of the same run). That is the
# quantity the verdict is really about, and it is the only way to tell whether the scan's
# shape is signal or sampling noise.
#
#   sbatch --partition=gpu_test --account=kempner_albergo_lab --gres=gpu:1 \
#          --export=ALL,DEVICE=cuda scripts/geom/grad_sign_seeds.sbatch
#
# Writes results/geom/diagnostics/grad_sign_seeds/grad_sign_seed<S>.json plus an aggregate
# results/geom/diagnostics/grad_sign_seed_spread.json. See docs/geom_guidance_diagnosis.md.
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_MOLS="${NUM_MOLS:-4}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-4}"
MFM_STEPS="${MFM_STEPS:-10}"
OUTER_STEPS="${OUTER_STEPS:-100}"
PROBE_SAMPLES="${PROBE_SAMPLES:-16}"
T_PROBE="${T_PROBE:-0.5}"
LAMBDA="${LAMBDA:-0.01}"
STEPS="${STEPS:-0.001 0.01 0.1 0.3 1.0}"
OUTDIR="${OUTDIR:-results/geom/diagnostics/grad_sign_seeds}"
AGG="${AGG:-results/geom/diagnostics/grad_sign_seed_spread.json}"

mkdir -p "${OUTDIR}"
for S in ${SEEDS}; do
  OUT="${OUTDIR}/grad_sign_seed${S}.json"
  if [ -s "${OUT}" ]; then echo "already done: ${OUT}"; continue; fi
  echo "=== grad-sign seed ${S} ==="
  "${PY}" -m multimfm.geom.diagnostics grad-sign \
    --device "${DEVICE}" --num-mols "${NUM_MOLS}" --t "${T_PROBE}" \
    --outer-steps "${OUTER_STEPS}" --posterior-samples "${POSTERIOR_SAMPLES}" \
    --mfm-steps "${MFM_STEPS}" --probe-samples "${PROBE_SAMPLES}" \
    --reward-lambda "${LAMBDA}" --steps ${STEPS} --seed "${S}" --out "${OUT}"
done

echo
echo "=== aggregate over seeds: paired delta per step, mean +- SEM ==="
"${PY}" - "${OUTDIR}" "${AGG}" <<'PYEOF'
import json, math, sys, glob, os
outdir, agg = sys.argv[1], sys.argv[2]
recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(outdir, "*.json")))]
if not recs:
    raise SystemExit("no grad-sign records found")
steps = sorted({float(k) for r in recs for k in r["guide_reward_by_step"]})
rows = []
for s in steps:
    d = []
    for r in recs:
        by = r["guide_reward_by_step"]
        key = next((k for k in by if float(k) == s), None)
        base = next((k for k in by if float(k) == 0.0), None)
        if key is not None and base is not None:
            d.append(by[key] - by[base])
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    rows.append({"step": s, "n_seeds": n, "mean_delta": m, "sd": sd, "sem": sem,
                 "t_stat": (m / sem) if sem and sem == sem and sem > 0 else float("nan"),
                 "per_seed": d})
print(f"{'step':>8}{'mean delta':>13}{'sd':>10}{'SEM':>9}{'t':>8}{'seeds':>7}")
for r in rows:
    print(f"{r['step']:>8.3f}{r['mean_delta']:>+13.4f}{r['sd']:>10.4f}"
          f"{r['sem']:>9.4f}{r['t_stat']:>+8.2f}{r['n_seeds']:>7d}")
pos = [r for r in rows if r["step"] > 0]
neg = [r for r in rows if r["step"] < 0]
print("\nPaired +/- comparison at matched |step| (positive minus negative; a correctly")
print("signed gradient makes this POSITIVE, and it must clear its own SEM):")
print(f"{'|step|':>8}{'delta(+) - delta(-)':>22}{'combined SEM':>15}{'t':>8}")
verdict_rows = []
for p in pos:
    m = next((q for q in neg if abs(q["step"]) == p["step"]), None)
    if m is None:
        continue
    diff = p["mean_delta"] - m["mean_delta"]
    cse = math.sqrt(p["sem"] ** 2 + m["sem"] ** 2) if p["sem"] == p["sem"] else float("nan")
    t = diff / cse if cse and cse == cse and cse > 0 else float("nan")
    verdict_rows.append({"abs_step": p["step"], "diff": diff, "combined_sem": cse, "t": t})
    print(f"{p['step']:>8.3f}{diff:>+22.4f}{cse:>15.4f}{t:>+8.2f}")
rec = {"_what": "grad-sign step scan repeated over seeds; paired delta per step",
       "n_seeds": len(recs), "rows": rows, "plus_minus_comparison": verdict_rows}
os.makedirs(os.path.dirname(agg), exist_ok=True)
with open(agg, "w") as fh:
    json.dump(rec, fh, indent=2)
print(f"\nwrote {agg}")
PYEOF
