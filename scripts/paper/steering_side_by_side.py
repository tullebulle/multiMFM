#!/usr/bin/env python
"""DNA and QM9 inference-time steering, side by side, on one fair axis.

    python scripts/paper/steering_side_by_side.py

Both panels plot target error against **forward-equivalent** network calls per
returned sample, so the guided methods are billed for their backward pass and the
pure-forward search baselines are not. The DNA conversion uses the measured
k = 1.1 (results/shared/nfe_audit/backward_cost_h200.json); QM9's
`nfe_generative_fwd_equiv` column already carries the measured k = 2.

Left:  DNA L=50, |f_cyc(x) - y*|, 100 samples per point, 3 repeats.
Right: QM9 alpha, oracle MAE in Bohr^3; each point is the best configuration of
       that method at that budget.
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures"
K_DNA = 1.1

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

# One identity per method across both panels: colour + marker + line style.
STYLE = {
    "ours":      ("#d62728", "o", "-",  2.0),
    "best_of_n": ("#1f77b4", "s", "--", 1.2),
    "fk":        ("#2ca02c", "^", "--", 1.2),
    "mcts":      ("#9467bd", "v", ":",  1.2),
    "beam":      ("#8c564b", "D", ":",  1.2),
    "dps":       ("#ff7f0e", "P", "-.", 1.2),
}
LABEL = {"best_of_n": "best-of-$N$", "fk": "Feynman--Kac", "mcts": "MCTS",
         "beam": "beam search", "dps": "DPS"}


def dna():
    rows = list(csv.DictReader(open(REPO / "results/dna/fig4/panelB_curves.csv")))
    out = defaultdict(list)
    for r in rows:
        x = float(r["nfe_gen_per_output"]) + K_DNA * float(r["nfe_backward_per_output"])
        out["ours" if r["method"] == "dmfm" else r["method"]].append(
            (x, float(r["mae_mean"]), float(r["mae_sem"])))
    return {k: sorted(v) for k, v in out.items()}


def qm9():
    rows = [r for r in csv.DictReader(open(REPO / "results/qm9/baselines/baseline_sweep_v2.csv"))
            if r["oracle_mae"] not in ("", "nan")]
    per = defaultdict(list)
    for r in rows:
        if r["method"] == "base":
            continue
        per[(r["method"], r["config"])].append(
            (float(r["nfe_generative_fwd_equiv"] or 0), float(r["oracle_mae"])))
    # one point per (method, budget): the best configuration measured there
    best = defaultdict(dict)
    for (m, _c), v in per.items():
        nfe = round(st.mean(a for a, _ in v))
        mae = st.mean(b for _, b in v)
        sem = st.stdev([b for _, b in v]) / len(v) ** 0.5 if len(v) > 1 else 0.0
        key = "ours" if m == "steer_search" else m
        if nfe not in best[key] or mae < best[key][nfe][0]:
            best[key][nfe] = (mae, sem)
    return {k: sorted((n, m, s) for n, (m, s) in d.items()) for k, d in best.items()}


def draw(ax, data, order, title, ylabel, ours_label):
    for m in order:
        if m not in data:
            continue
        x, y, e = np.array(data[m]).T
        c, mk, ls, lw = STYLE[m]
        ax.errorbar(x, y, yerr=e, color=c, marker=mk, ls=ls, lw=lw, ms=4,
                    capsize=2, label=ours_label if m == "ours" else LABEL[m],
                    zorder=3 if m == "ours" else 2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forward-equivalent network calls per sample")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", borderaxespad=0.4)


def main():
    d, q = dna(), qm9()
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9))
    draw(axes[0], d, ["best_of_n", "fk", "mcts", "beam", "ours"],
         "DNA ($L{=}50$, cyclizability)", r"$|f_{\mathrm{cyc}}(x)-y^\star|$", "dMFM (ours)")
    draw(axes[1], q, ["best_of_n", "fk", "dps", "ours"],
         r"QM9 ($\alpha$)", r"oracle MAE [Bohr$^3$]", "multiMFM-SS (ours)")

    # Shade where our curve is below the lower envelope of every baseline, found on a
    # fine log grid over the budgets where our curve and at least one baseline both
    # exist. Shading beyond that range would claim budgets we never ran.
    for ax, data in zip(axes, (d, q)):
        ours = np.array(data["ours"]).T
        others = [np.array(v).T for k, v in data.items() if k != "ours"]
        lo_x = max(ours[0].min(), min(o[0].min() for o in others))
        hi_x = min(ours[0].max(), max(o[0].max() for o in others))
        grid = np.geomspace(lo_x, hi_x, 400)
        mine = np.interp(grid, ours[0], ours[1], left=np.nan, right=np.nan)
        env = np.nanmin(np.vstack([
            np.interp(grid, o[0], o[1], left=np.nan, right=np.nan) for o in others]), axis=0)
        lead = np.isfinite(mine) & np.isfinite(env) & (mine < env)
        if lead.any():
            lo, hi = grid[lead].min(), grid[lead].max()
            ax.axvspan(lo, hi, color="#d62728", alpha=0.08, zorder=0)
            ax.annotate(f"ours ahead of every\nbaseline below {hi:,.0f}",
                        xy=(np.sqrt(lo * hi), 0.97), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6, color="#b02020")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"steering_dna_qm9_side_by_side.{ext}"
        fig.savefig(p, dpi=200)
        print("wrote", p.relative_to(REPO))
    plt.close(fig)


if __name__ == "__main__":
    main()
