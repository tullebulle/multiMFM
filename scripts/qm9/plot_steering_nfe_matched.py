#!/usr/bin/env python
"""QM9 steering-vs-NFE comparison, drawn in the style of Figure 4 (right).

The first version of this plot put every method on its own budget ladder, so
best-of-N ran out to 65,536 NFE while multiMFM stopped near 15,000 and DPS
clustered below 250. The curves then spanned different ranges and could not be
read against each other. Here each method is held at ONE hyperparameter setting
and swept over a shared budget ladder, so the evaluation points line up:

    multiMFM, multiMFM-SS   mu = 3.4 fixed,      MC in {1, 2, 4, 8, 16, 32}
    best-of-N                                     N in {4, 8, 16, 32, 64, 128}
    Feynman-Kac             lambda = 0.01 fixed,  K in {4, 8, 16, 32}
    DPS                     mu = 0.1 fixed,       guide_every in {8, 4, 2, 1}

best-of-N is trimmed at N=128 so its range matches the others rather than
running four times further right. DPS is shown where it lies: its budget knob
saturates at 243 NFE, below the shared ladder, so it gets an open marker and no
line to the rest. Points are plotted at their true NFE, not snapped to the
nominal budget; the dotted verticals mark the nominal ladder so the near-matches
are visible.

Property alpha, n = 100 molecules per seed, mean +- SE over seeds {1, 2, 3},
same base model and same target set for every method.

    python scripts/qm9/plot_steering_nfe_matched.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "results" / "qm9" / "baselines" / "baseline_sweep_v2.csv"
OUT = REPO / "notebooks" / "figures" / "07_qm9_steering_nfe_matched.png"

# label, marker, colour, linewidth -- the Figure 4 (right) palette, so the two
# figures read as one pair: the headline method in red and heavier than the rest.
STYLE = {
    "multiMFM-SS":  ("multiMFM-SS", "o", "#d62728", 2.6),
    "multiMFM":     ("multiMFM",    "v", "#8c564b", 1.8),
    "best-of-N":    ("Best-of-$N$", "s", "#1f77b4", 1.8),
    "Feynman-Kac":  ("Feynman-Kac", "D", "#ff7f0e", 1.8),
    "DPS":          ("DPS",         "P", "#9467bd", 1.8),
}
RUN_TO_METHOD = {
    "best_lookahead_pbvalid": "multiMFM-SS",
    "final_endpoint": "multiMFM",
    "best_of_n_pbvalid": "best-of-N",
    "fk_pbvalid": "Feynman-Kac",
    "dps": "DPS",
}
NOMINAL = [512, 1024, 2048, 4096, 8192]
# multiMFM stops at MC=16 (7,488 NFE), so every other method is cut at the same
# budget rather than running further right and stretching the axis.
MAX_MC = 16
MAX_NFE = 10_000


def _cfg(pattern: str, config: str) -> float | None:
    m = re.search(pattern, config)
    return float(m.group(1)) if m else None


def select(df: pd.DataFrame) -> pd.DataFrame:
    """One fixed hyperparameter setting per method, swept over the budget knob."""
    df = df[df.run.isin(RUN_TO_METHOD)].copy()
    df["method"] = df.run.map(RUN_TO_METHOD)

    keep = []
    for _, r in df.iterrows():
        m, c = r.method, str(r.config)
        if m in ("multiMFM", "multiMFM-SS"):
            mc = _cfg(r"MC=([0-9]+)", c)
            ok = _cfg(r"mu=([0-9.]+)", c) == 3.4 and mc is not None and mc <= MAX_MC
        elif m == "best-of-N":
            ok = _cfg(r"N=([0-9]+)", c) is not None
        elif m == "Feynman-Kac":
            ok = _cfg(r"lambda=([0-9.]+)", c) == 0.01
        elif m == "DPS":
            # mu fixed; budget comes from guide_every (down to 1) and then from
            # sample_steps, which the extension sweep adds.
            ok = _cfg(r"mu=([0-9.]+)", c) == 0.1
        else:
            ok = False
        keep.append(ok)
    df = df[pd.Series(keep, index=df.index)]
    df = df[df.nfe_generative <= MAX_NFE]              # one shared budget range

    # DPS's sample_steps extension is not encoded in `config` (every row reads
    # "mu=0.1,guide_every=1"), only in the output directory and the NFE. Group on a
    # key that carries it, or the 128-step run and all the refined ones collapse
    # into a single point.
    df["config"] = [
        f"{c},sample_steps={m.group(1)}" if (m := re.search(r"_s(\d+)/", str(f))) else c
        for c, f in zip(df.config.astype(str), df.file.astype(str))
    ]
    g = (df.groupby(["method", "config"])
           .agg(nfe=("nfe_generative", "mean"),
                mae=("oracle_mae", "mean"),
                se=("oracle_mae", lambda s: s.std(ddof=1) / np.sqrt(max(len(s), 1))),
                pb=("pb_valid", "mean"),
                seeds=("seed", "nunique"))
           .reset_index())
    return g.sort_values(["method", "nfe"])


def budget_label(method: str, config: str) -> str:
    for pat, fmt in ((r"MC=([0-9]+)", "MC={}"), (r"N=([0-9]+)", "N={}"),
                     (r"K=([0-9]+)", "K={}"), (r"sample_steps=([0-9]+)", "steps={}"),
                     (r"_s([0-9]+)", "steps={}"), (r"guide_every=([0-9]+)", "every={}")):
        m = re.search(pat, str(config))
        if m:
            return fmt.format(m.group(1))
    return ""


def main() -> None:
    if not SWEEP.exists():
        sys.exit(f"missing {SWEEP}")
    g = select(pd.read_csv(SWEEP))

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for b in NOMINAL:
        ax.axvline(b, color="0.85", ls=":", lw=0.8, zorder=0)

    for method, (lab, mk, colour, lw) in STYLE.items():
        s = g[g.method == method]
        if not len(s):
            continue
        # DPS is drawn detached only while its budget knob cannot reach the ladder;
        # once the sample_steps extension lands it spans the range and joins as a line.
        detached = method == "DPS" and s.nfe.max() < NOMINAL[0]
        line = "none" if detached else "-"
        face = "none" if detached else colour
        ax.errorbar(s.nfe, s.mae, yerr=s.se, marker=mk, color=colour, ls=line,
                    mfc=face, lw=lw, ms=5, capsize=3, label=lab, zorder=3)

    head = g[g.method == "multiMFM-SS"].reset_index(drop=True)
    for i, r in head.iterrows():
        # The budgets double, so on a log axis the last two labels are close enough
        # to overlap horizontally; drop the final one onto a second row instead.
        last = i == len(head) - 1
        ax.annotate(budget_label(r.method, r.config), (r.nfe, r.mae),
                    textcoords="offset points",
                    xytext=(0, -22 if last else -12),
                    fontsize=6.5, color="#d62728", ha="center")

    ax.set_xscale("log")
    # One deviation from the Figure 4 (right) style, which is linear in y: here the
    # spread runs from 0.92 to 8.1, so a linear axis squashes the region where the
    # methods actually cross (0.9-1.4) into the bottom eighth of the panel.
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 8])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("NFE (generative model, per returned molecule)")
    ax.set_ylabel(r"oracle MAE, $\alpha$ [Bohr$^3$]")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.45)   # room for the MC=32 annotation
    ax.set_title("Inference-time steering comparison (QM9)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")

    print("\nmatched-budget table (mean +- SE over seeds):")
    for _, r in g.iterrows():
        print(f"  {r.method:<12} {budget_label(r.method, r.config):<8} "
              f"NFE={r.nfe:>7.0f}  MAE={r.mae:.3f}+-{r.se:.3f}  PB={r.pb:.3f}")


if __name__ == "__main__":
    main()
