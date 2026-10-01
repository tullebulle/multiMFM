#!/usr/bin/env python
"""QM9 inference-time steering at matched inference cost.

    python scripts/paper/qm9_steering_figure.py

Axis. Forward-equivalent network calls per returned molecule: the guided methods are
billed for their backward pass at its measured cost (k = 2), the pure-forward search
baselines are not. Every method integrates the same 128-step outer trajectory, so the
whole budget difference is the value estimator. Our points were placed on the baselines'
own budget grid by solving the cost for the guidance window, not interpolated onto it.

Reading. The like-for-like guidance comparison is **multiMFM against DPS**: neither
selects among candidates, both spend their budget steering one trajectory. multiMFM-SS
adds selection on top and is shown as the further gain, against best-of-N which is the
selection baseline. Every method must clear the same PoseBusters validity floor.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _qm9style as style

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures"
PB_FLOOR = 0.75
LADDER = [256, 512, 1024, 2048]
TOL = 0.15
XLIM = (215, 2650)

style.use()



def snap(points):
    """At each rung, the best error reachable with at most that budget (same rule for all)."""
    out = []
    for rung in LADDER:
        avail = [(x, y, e) for x, y, e in points if x <= rung * (1 + TOL)]
        if avail:
            _, y, e = min(avail, key=lambda p: p[1])
            out.append((rung, y, e))
    return out


def load():
    d = defaultdict(list)
    for r in csv.DictReader(open(REPO / "results/qm9/steering_final.csv")):
        if r["method"] in ("base", "fk"):       # fk re-runs after the resampling fix
            continue
        if float(r["pb_valid"] or 0) < PB_FLOOR:
            continue
        key = {"ss": "ours_ss", "multimfm": "ours"}.get(r["method"], r["method"])
        d[key].append((float(r["nfe_fwd_equiv"]), float(r["oracle_mae"]),
                       float(r["oracle_mae_sem"])))
    for r in csv.DictReader(open(REPO / "results/qm9/multimfm_search_grid.csv")):
        if r["property"] == "alpha" and float(r["guided_pb_valid"] or 0) >= PB_FLOOR:
            d["ours"].append((1508.0, float(r["guided_oracle_mae"]), 0.0))
    f = REPO / "results/qm9/steering_twotime.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            if float(r["pb_valid"] or 0) < PB_FLOOR:
                continue
            d["ours_ss" if r["arm"] == "SS" else "ours"].append(
                (float(r["nfe_fwd_equiv"]), float(r["oracle_mae"]), float(r["oracle_mae_sem"])))
    fk = REPO / "results/qm9/fk_ladder.csv"
    if fk.exists():
        for r in csv.DictReader(open(fk)):
            d["fk"].append((float(r["nfe_fwd_equiv"]), float(r["oracle_mae"]),
                            float(r["oracle_mae_sem"])))
    return {k: snap(v) for k, v in d.items()}


def main():
    d = load()
    fig, ax = plt.subplots(figsize=(4.6, 3.5))

    for m in ("best_of_n", "fk", "dps", "ours", "ours_ss"):
        if m not in d or not d[m]:
            continue
        x, y, e = np.array(d[m]).T
        style.draw(ax, m, x, y, yerr=e)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks(LADDER)
    ax.set_xticklabels([f"{b:,}" for b in LADDER])
    ax.minorticks_off()
    yt = [1.0, 1.5, 2, 3, 4, 6]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.set_xlabel(r"NFE")
    ax.set_ylabel(r"Oracle MAE")
    ax.set_title(r"QM9 Property Steering", pad=8)
    ax.grid(alpha=0.25, which="major", lw=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"qm9_steering.{ext}"
        fig.savefig(p, dpi=220)
        print("wrote", p.relative_to(REPO))
    plt.close(fig)

    print(f"\n{'budget':>7} | {'multiMFM':>9} {'DPS':>9} {'ratio':>7} | "
          f"{'multiMFM-SS':>12} {'best-of-N':>10}")
    at = lambda m, r: {x: y for x, y, _ in d.get(m, [])}.get(r)
    for r in LADDER:
        m, p, s, b = at("ours", r), at("dps", r), at("ours_ss", r), at("best_of_n", r)
        fm = lambda v: f"{v:9.3f}" if v else f"{'--':>9}"
        print(f"{r:>7} | {fm(m)} {fm(p)} {(f'{p/m:6.2f}x' if m and p else '    --'):>7} | "
              f"{fm(s):>12} {fm(b):>10}")


if __name__ == "__main__":
    main()
