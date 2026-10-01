#!/usr/bin/env python
"""Collect the QM9 low-budget grid and the DNA guidance-window sweep.

    python scripts/paper/collect_low_budget.py

Writes results/qm9/low_budget_grid.csv and results/dna/fig4/panelB_window_curves.csv.
Both carry the *measured* forward-equivalent budget, so the two domains and every
method sit on one axis: QM9 uses the run's own `nfe_generative_fwd_equiv`
(k = 2, measured), DNA uses gen + 1.1 x backward (k = 1.1, measured).
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
K_DNA = 1.1


def qm9() -> None:
    root = REPO / "outputs/qm9/low_budget_grid"
    rows = []
    for d in sorted(root.glob("*/")):
        f = d / "summary.json"
        if not f.exists():
            continue
        m = re.match(r"(\w+?)_s(\d+)_ge(\d+)_mc(\d+)_g([\d.]+)_n(\d+)", d.name)
        if not m:
            continue
        prop, S, ge, mc, gamma, n = m.groups()
        s = json.loads(f.read_text())
        runs = {r["run"]: r for r in s["summary"]}
        nfe = s.get("nfe", {}) or {}
        fwd = nfe.get("nfe_generative_fwd_equiv")
        if fwd is None:  # fall back to the cost model the sweep was designed around
            fwd = 128 + 3 * (115 // int(ge)) * int(S) * int(mc)
        for arm, key in (("multimfm", "final_endpoint"), ("ss", "best_lookahead_pbvalid")):
            r = runs.get(key)
            if not r or r.get("oracle_mae") is None:
                continue
            rows.append({"property": prop, "arm": arm, "value_glass_steps": S,
                         "guide_every": ge, "value_samples": mc, "gamma": gamma,
                         "n": n, "nfe_fwd_equiv": round(float(fwd)),
                         "oracle_mae": r["oracle_mae"], "pb_valid": r.get("pb_pb_valid"),
                         "dir": str(d.relative_to(REPO))})
    if not rows:
        print("qm9: nothing collected yet")
        return
    out = REPO / "results/qm9/low_budget_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"qm9: {len(rows)} rows -> {out.relative_to(REPO)}")


def dna() -> None:
    rows = []
    for d in sorted((REPO / "outputs/dna/fig4/panelB").glob("window_t*")):
        ts = d.name.replace("window_t", "")
        per = defaultdict(list)
        for f in sorted(d.rglob("*.json")):
            if f.name == "calibration.json":
                continue
            try:
                s = json.loads(f.read_text())
            except Exception:
                continue
            lab, met, nfe = s.get("config"), s.get("metrics") or {}, s.get("nfe") or {}
            if lab is None or "mae" not in met:
                continue
            per[lab].append((float(nfe.get("nfe_gen_per_output", 0)),
                             float(nfe.get("nfe_backward_per_output", 0)),
                             float(met["mae"]), float(met.get("frac10", "nan"))))
        for lab, v in sorted(per.items()):
            gen = st.mean(x[0] for x in v)
            bwd = st.mean(x[1] for x in v)
            mae = st.mean(x[2] for x in v)
            sem = st.stdev([x[2] for x in v]) / len(v) ** 0.5 if len(v) > 1 else 0.0
            rows.append({"guide_t_start": ts, "config": lab, "n_repeats": len(v),
                         "nfe_gen_per_output": round(gen), "nfe_backward_per_output": round(bwd),
                         "nfe_fwd_equiv": round(gen + K_DNA * bwd),
                         "mae_mean": mae, "mae_sem": sem,
                         "frac10_mean": st.mean(x[3] for x in v)})
    if not rows:
        print("dna: nothing collected yet")
        return
    out = REPO / "results/dna/fig4/panelB_window_curves.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"dna: {len(rows)} rows -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    qm9()
    dna()
