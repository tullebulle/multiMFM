#!/usr/bin/env python
"""Collect the corrected Feynman-Kac ladder into results/qm9/fk_ladder.csv.

Run with ESS-gated resampling at lambda = 0, which is where its own pilot put it: with
the resampling defect fixed, every positive lambda measured was worse (3.63 at 0 rising
monotonically to 8.84 at 3.0), so FK's tuned setting on QM9 is no resampling at all.
"""
from __future__ import annotations

import csv, glob, json, re, statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
g = defaultdict(list)
for f in sorted(glob.glob(str(REPO / "outputs/qm9/fk_ladder/seed*/fk_k*_lam0/fk_alpha_seed*.json"))):
    K = int(re.search(r"fk_k(\d+)_lam0", f).group(1))
    d = json.loads(Path(f).read_text())
    row = next((r for r in d["summary"] if r.get("run", "").endswith("pbvalid")), d["summary"][0])
    nfe = d.get("nfe", {})
    fwd = nfe.get("nfe_generative_fwd_equiv") or \
        (nfe.get("by_phase", {}).get("sampling", {}) or {}).get("nfe_generative_fwd_equiv")
    if row.get("oracle_mae") is None or not fwd:
        continue
    g[(K, round(float(fwd)))].append((row["oracle_mae"], row.get("pb_pb_valid")))

rows = []
for (K, fwd), v in sorted(g.items(), key=lambda kv: kv[0][1]):
    maes = [x[0] for x in v]
    pbs = [x[1] for x in v if x[1] is not None]
    rows.append({"config": f"K={K},lambda=0", "nfe_fwd_equiv": fwd,
                 "oracle_mae": round(st.mean(maes), 4),
                 "oracle_mae_sem": round(st.stdev(maes) / len(maes) ** 0.5, 4) if len(maes) > 1 else 0.0,
                 "pb_valid": round(st.mean(pbs), 4) if pbs else "", "n_seeds": len(v)})
out = REPO / "results/qm9/fk_ladder.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} points -> {out.relative_to(REPO)}")
for r in rows:
    print(f"  {r['config']:16} {r['nfe_fwd_equiv']:>6} calls  oracle {r['oracle_mae']:.3f} "
          f"+-{r['oracle_mae_sem']:.3f}  PB {r['pb_valid']}  n={r['n_seeds']}")
