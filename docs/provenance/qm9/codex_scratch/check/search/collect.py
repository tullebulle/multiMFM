#!/usr/bin/env python3
"""Collect search results: for every summary.json under the search dir, print
property, mu, reward_scale, guided oracle MAE (meV for energies), PB validity,
and whether it beats the per-property baseline while keeping PB >= 0.5."""
import json, glob, os, sys

HARTREE_TO_MEV = 27211.386245988
ENERGY = {"gap", "homo", "lumo"}
# baseline oracle MAE to beat (native units for alpha/mu/cv; meV for energies)
BASELINE = {"cv": 0.739, "mu": 0.959, "alpha": 1.983,
            "gap": 678.0, "homo": 284.0, "lumo": 450.0}

root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
rows = []
for path in glob.glob(os.path.join(root, "**", "summary.json"), recursive=True):
    try:
        d = json.load(open(path))
    except Exception:
        continue
    a = d.get("args", {})
    prop = a.get("property_name")
    if prop not in BASELINE:
        continue
    mu = float(a.get("mu", "nan")); rs = float(a.get("reward_scale", "nan"))
    gmax = float(a.get("guide_max_t", 0.95))
    guided = next((r for r in d["summary"] if r.get("run") == "guided"), None)
    ung = next((r for r in d["summary"] if r.get("run") == "unguided"), None)
    if guided is None:
        continue
    om = guided.get("oracle_mae")
    pb = guided.get("pb_pb_valid")
    if om is None:
        continue
    disp = om * HARTREE_TO_MEV if prop in ENERGY else om
    base = BASELINE[prop]
    unit = "meV" if prop in ENERGY else ""
    improve = (base - disp) / base * 100.0
    tag = os.path.basename(os.path.dirname(path))
    rows.append((prop, mu, rs, disp, pb, base, improve, unit, tag, gmax))

PB_FLOOR = 0.90
order = {p: i for i, p in enumerate(["cv", "mu", "alpha", "gap", "homo", "lumo"])}
rows.sort(key=lambda r: (order.get(r[0], 9), r[3]))
rows.sort(key=lambda r: (order.get(r[0], 9), r[3]))
print(f"{'prop':6} {'mu':>7} {'rs':>9} {'gmax':>5} {'oracleMAE':>11} {'PB':>6} {'base':>8} {'improv%':>8}  {'PB>=.90':>7}  tag")
print("-" * 100)
for prop, mu, rs, disp, pb, base, improve, unit, tag, gmax in rows:
    pbs = "" if pb is None else f"{pb:.3f}"
    ok = "OK" if (pb is not None and pb >= PB_FLOOR) else "LOW"
    flag = "  <BEAT" if (disp < base and pb is not None and pb >= PB_FLOOR) else ""
    print(f"{prop:6} {mu:7.3f} {rs:9.3f} {gmax:5.2f} {disp:9.3f}{unit:>2} {pbs:>6} {base:8.3f} {improve:7.1f}% {ok:>7}{flag}  {tag}")
