#!/usr/bin/env python
"""Collect the final low-noise steering runs (QM9 n=512 x 3 seeds, DNA 300 x 5).

    python scripts/paper/collect_final_steering.py

Ours and the baselines are collected by the same code path and at the same sample
size, so neither side is smoother than the other. Writes
results/qm9/steering_final.csv and results/dna/fig4/steering_final.csv.
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



def _agg(vals):
    m = st.mean(vals)
    s = st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    return m, s


def qm9():
    per = defaultdict(list)
    for seed_dir in sorted((REPO / "outputs/qm9/lowbudget_final").glob("seed*")):
        seed = seed_dir.name.replace("seed", "")
        for d in sorted(seed_dir.glob("*/")):
            name = d.name
            if name.startswith("ours_"):
                m = re.match(r"ours_s(\d+)_ge(\d+)_mc(\d+)_g([\d.]+)", name)
                f = d / "summary.json"
                if not (m and f.exists()):
                    continue
                s = json.loads(f.read_text())
                runs = {r["run"]: r for r in s["summary"]}
                nfe = fwd_equiv(s)
                S, ge, mc, gam = m.groups()
                if nfe is None:
                    nfe = 128 + 3 * (115 // int(ge)) * int(S) * int(mc)
                for arm, key in (("multimfm", "final_endpoint"), ("ss", "best_lookahead_pbvalid")):
                    r = runs.get(key)
                    if r and r.get("oracle_mae") is not None:
                        per[(arm, f"S{S}/ge{ge}/mc{mc}/g{gam}", round(float(nfe)))].append(
                            (r["oracle_mae"], r.get("pb_pb_valid")))
            else:
                for f in d.glob("*.json"):
                    try:
                        s = json.loads(f.read_text())
                    except Exception:
                        continue
                    summ = s.get("summary") or []
                    row = next((r for r in summ if r.get("run") in ("guided", "final", "output")), None)
                    row = row or (summ[0] if summ else None)
                    nfe = fwd_equiv(s)
                    if not row or row.get("oracle_mae") is None or nfe is None:
                        continue
                    meth = ({"bon": "best_of_n", "fk": "fk", "dps": "dps", "base": "base"}
                            .get(d.name.split("_")[0], d.name.split("_")[0]))
                    per[(meth, d.name, round(float(nfe)))].append(
                        (row["oracle_mae"], row.get("pb_pb_valid")))
    rows = []
    for (meth, cfg, nfe), v in sorted(per.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        mae, sem = _agg([x[0] for x in v])
        pbs = [x[1] for x in v if x[1] is not None]
        rows.append({"method": meth, "config": cfg, "nfe_fwd_equiv": nfe,
                     "oracle_mae": round(mae, 4), "oracle_mae_sem": round(sem, 4),
                     "pb_valid": round(st.mean(pbs), 4) if pbs else "", "n_seeds": len(v)})
    if not rows:
        print("qm9 final: nothing yet")
        return
    out = REPO / "results/qm9/steering_final.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"qm9 final: {len(rows)} points -> {out.relative_to(REPO)}")


def dna():
    per = defaultdict(list)
    roots = [REPO / "outputs/dna/fig4/final", REPO / "outputs/dna/fig4/matched"]
    for d in sorted(x for r in roots if r.exists() for x in r.glob("*/")):
        tag = ("matched/" if d.parent.name == "matched" else "") + d.name
        for f in sorted(d.rglob("*.json")):
            if f.name == "calibration.json":
                continue
            try:
                s = json.loads(f.read_text())
            except Exception:
                continue
            met, nfe = s.get("metrics") or {}, s.get("nfe") or {}
            if "mae" not in met:
                continue
            key = (s.get("method"), s.get("config"), tag)
            per[key].append((float(nfe.get("nfe_gen_per_output", 0)),
                             float(nfe.get("nfe_backward_per_output", 0)), float(met["mae"])))
    rows = []
    for (meth, cfg, tag), v in sorted(per.items()):
        gen, bwd = st.mean(x[0] for x in v), st.mean(x[1] for x in v)
        mae, sem = _agg([x[2] for x in v])
        rows.append({"method": meth, "config": cfg, "arm": tag,
                     "nfe_fwd_equiv": round(gen + K_DNA * bwd),
                     "mae_mean": round(mae, 5), "mae_sem": round(sem, 5), "n_repeats": len(v)})
    if not rows:
        print("dna final: nothing yet")
        return
    out = REPO / "results/dna/fig4/steering_final.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"dna final: {len(rows)} points -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    qm9()
    dna()
