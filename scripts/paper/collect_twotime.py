#!/usr/bin/env python
"""Collect the two-time vs diagonal flow-map query test into results/qm9/steering_twotime.csv."""
from __future__ import annotations

import csv, glob, json, re, statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def fwd_equiv(summary):
    """Forward-equivalent generative calls per molecule, from the run's own counters.

    The value estimator and the ODE steps both live in the `trajectory` phase; the
    selection and evaluation phases are guide/oracle calls, which this axis excludes.
    """
    n = summary.get("nfe") or {}
    v = n.get("nfe_generative_fwd_equiv")
    if v:
        return float(v)
    tr = (n.get("by_phase") or {}).get("trajectory") or {}
    v = tr.get("nfe_generative_fwd_equiv")
    return float(v) if v else None

g = defaultdict(list)
pats = ["outputs/qm9/lowbudget_twotime/seed*/*/summary.json",
        "outputs/qm9/lowbudget_final2/seed*/*/summary.json",
        "outputs/qm9/matched_grid/seed*/*/summary.json"]
for f in sorted(f for pat in pats for f in glob.glob(str(REPO / pat))):
    m = re.match(r".*/s(\d+)_ge(\d+)_mc(\d+)_g([\d.]+)_(diag|twotime)/summary.json", f)
    if m:
        S, ge, mc, gam, mode = m.groups()
        cfg = f"S{S}/ge{ge}/mc{mc}/g{gam}"
    else:
        mb = re.match(r".*/b(\d+)_g([\d.]+)/summary.json", f)
        if not mb:
            continue
        budget, gam = mb.groups()
        S = ge = mc = "0"
        mode, cfg = "twotime", f"matched b{budget}/g{gam}"
    s = json.loads(Path(f).read_text())
    runs = {r["run"]: r for r in s["summary"]}
    nfe = fwd_equiv(s)
    if nfe is None:
        nfe = (128 + 3 * (115 // int(ge)) * int(S) * int(mc)) if int(ge) else 0
    for arm, key in (("multiMFM", "final_endpoint"), ("SS", "best_lookahead_pbvalid")):
        r = runs.get(key)
        if r and r.get("oracle_mae") is not None:
            g[(cfg, arm, mode, round(float(nfe)))].append(
                (r["oracle_mae"], r.get("pb_pb_valid")))

rows = []
for (cfg, arm, mode, nfe), v in sorted(g.items(), key=lambda kv: kv[0][3]):
    maes = [x[0] for x in v]
    pbs = [x[1] for x in v if x[1] is not None]
    rows.append({"config": cfg, "arm": arm, "query": mode, "nfe_fwd_equiv": nfe,
                 "oracle_mae": round(st.mean(maes), 4),
                 "oracle_mae_sem": round(st.stdev(maes) / len(maes) ** 0.5, 4) if len(maes) > 1 else 0.0,
                 "pb_valid": round(st.mean(pbs), 4) if pbs else "", "n_seeds": len(v)})
out = REPO / "results/qm9/steering_twotime.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {out.relative_to(REPO)}")
