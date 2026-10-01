#!/usr/bin/env python
"""QM9 property steering: value estimation and steering performance, one figure.

    python scripts/paper/qm9_figure.py

Both panels are QM9 and both put NFE on the x axis.

(a) Value-function estimation. |V_hat - V| against the posterior samples the estimate is
    built from; each sample is one network call, so N is the NFE the estimator spends.
    DPS is one call and independent of N. The reference is a large GLASS pool, itself a
    finite-sample estimate, so its own self-split spread is drawn as the resolution floor.

(b) Steering. Held-out oracle MAE against forward-equivalent network calls per molecule.
    The guided methods are billed for their backward pass at its measured cost; the
    pure-forward baselines are not. Every method integrates the same 128-step trajectory,
    so the budget difference is the value estimator alone, and each method is shown at the
    best error it reaches with at most that budget.

The two panels are the same claim measured twice: the estimator that recovers the value
function (a) is the one that steers best per network call (b).
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

style.use()



def snap(points):
    out = []
    for rung in LADDER:
        avail = [(x, y, e) for x, y, e in points if x <= rung * (1 + TOL)]
        if avail:
            _, y, e = min(avail, key=lambda p: p[1])
            out.append((rung, y, e))
    return out


def steering():
    d = defaultdict(list)
    for r in csv.DictReader(open(REPO / "results/qm9/steering_final.csv")):
        if r["method"] in ("base", "fk") or float(r["pb_valid"] or 0) < PB_FLOOR:
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
            if float(r["pb_valid"] or 0) >= PB_FLOOR:
                d["ours_ss" if r["arm"] == "SS" else "ours"].append(
                    (float(r["nfe_fwd_equiv"]), float(r["oracle_mae"]),
                     float(r["oracle_mae_sem"])))
    f = REPO / "results/qm9/fk_ladder.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            d["fk"].append((float(r["nfe_fwd_equiv"]), float(r["oracle_mae"]),
                            float(r["oracle_mae_sem"])))
    return {k: snap(v) for k, v in d.items()}


def value_error():
    f = REPO / "results/qm9/value_error.csv"
    if not f.exists():
        return None
    d = defaultdict(list)
    for r in csv.DictReader(open(f)):
        d[r["estimator"]].append((int(r["n"]), float(r["abs_value_error"])))
    return {k: sorted(v) for k, v in d.items()}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ---------------------------------------------------------------- (a) value error
    ax = axes[0]
    ve = value_error()
    if ve:
        for key, skey in (("FMAP", "fmap"), ("DPS", "dps"), ("multiMFM", "mfm")):
            if key in ve:
                x, y = np.array(ve[key]).T
                style.draw(ax, skey, x, y, zorder=5 if key == "multiMFM" else 2)
        if "reference self-split" in ve:
            x, y = np.array(ve["reference self-split"]).T
            ax.plot(x, y, color=style.C_FLOOR, ls=(0, (1, 2)), lw=1.1,
                    label=r"reference noise floor")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "value-error run pending", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
    ax.set_xlabel(r"NFE")
    ax.set_ylabel(r"Value error")
    ax.grid(alpha=0.25, which="major", lw=0.4)
    ax.legend(frameon=False, loc="lower left")

    # ---------------------------------------------------------------- (b) steering
    ax = axes[1]
    d = steering()
    for m in ("best_of_n", "fk", "dps", "ours", "ours_ss"):
        if m not in d or not d[m]:
            continue
        x, y, e = np.array(d[m]).T
        style.draw(ax, m, x, y, yerr=e)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(215, 2650)
    ax.set_xticks(LADDER)
    ax.set_xticklabels([f"{b:,}" for b in LADDER])
    yt = [1.0, 1.5, 2, 3, 4, 6]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()
    ax.set_xlabel(r"NFE")
    ax.set_ylabel(r"Oracle MAE")
    ax.grid(alpha=0.25, which="major", lw=0.4)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(r"QM9 Property Steering", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        p = OUT / f"qm9_property_steering.{ext}"
        fig.savefig(p, dpi=220)
        print("wrote", p.relative_to(REPO))
    plt.close(fig)


if __name__ == "__main__":
    main()
