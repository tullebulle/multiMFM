#!/usr/bin/env python
"""Paper figures that are regenerated from dMFM results (Fig 5, Fig 7).

    python scripts/paper/dmfm_figures.py            # writes figures/*.pdf

Fig 5: finite-N dMFM value-gradient accuracy against the GLASS-2048 reference,
       from results/dna/scale_free_mc/dmfm/diag1step/<reward>/.
Fig 7: C0 histograms, unguided vs dMFM-guided, target 0.30, n=1000, from
       results/dna/dmfm_demos/fig7 (4x8 ladder) and
       results/dna/operating_point/fig7_operating_point_n1000 (1x32).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

REWARDS = [("gc", "GC target"), ("motif", "Motif"), ("conjunction", "Conjunction")]
LCOL = {50: "#1f77b4", 100: "#ff7f0e", 200: "#2ca02c", 400: "#d62728"}
LMRK = {50: "o", 100: "s", 200: "^", 400: "D"}


def read_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def fig5() -> None:
    spec = [("relative", "global_relative_l1", "relative_l1_ci95_low", "relative_l1_ci95_high",
             r"relative $\ell_1$ error"),
            ("cosine", "mean_cosine", "cosine_ci95_low", "cosine_ci95_high", "mean cosine similarity")]
    for tag, col, lo, hi, ylab in spec:
        fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.1), sharey=True)
        for ax, (key, title) in zip(axes, REWARDS):
            rows = read_csv(REPO / "results/dna/scale_free_mc/dmfm/diag1step" / key
                            / "gradient_scale_free_summary_all_lengths.csv")
            for L in (50, 100, 200, 400):
                g = sorted((r for r in rows if int(r["length"]) == L), key=lambda r: int(r["mc"]))
                x = np.array([int(r["mc"]) for r in g])
                y = np.array([float(r[col]) for r in g])
                err = np.vstack([y - np.array([float(r[lo]) for r in g]),
                                 np.array([float(r[hi]) for r in g]) - y])
                ax.errorbar(x, y, yerr=np.abs(err), color=LCOL[L], marker=LMRK[L], ms=3, lw=1,
                            capsize=1.5, label=f"{L} bp")
            ax.set_xscale("log", base=2)
            ax.set_title(title)
            ax.set_xlabel("posterior samples $N$")
            ax.set_ylim(0, 1 if tag == "cosine" else None)
        axes[0].set_ylabel(ylab)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        p = OUT / f"dna_scaling_dmfm_{tag}.pdf"
        fig.savefig(p)
        plt.close(fig)
        print("wrote", p.relative_to(REPO))


def _scores(p: Path, col: str) -> np.ndarray:
    key = {"unguided": ["unguided", "unguided_score"], "guided": ["guided", "guided_score"]}[col]
    rows = read_csv(p)
    k = next(c for c in key if c in rows[0])
    return np.array([float(r[k]) for r in rows])


def fig7() -> None:
    ladder = REPO / "results/dna/dmfm_demos"
    op = REPO / "results/dna/operating_point/fig7_operating_point_n1000/dmfm_s1mc32/sample_scores.csv"
    panels = [("a", ladder / "fig7/dmfm/gf0_mc8_n1000_seed0/scores.csv"),
              ("b", ladder / "fig7/dmfm/gf2_mc8_n1000_seed0/scores.csv"),
              ("c", op)]
    bins = np.linspace(-1.6, 1.2, 57)
    for tag, p in panels:
        u, g = _scores(p, "unguided"), _scores(p, "guided")
        fig, ax = plt.subplots(figsize=(2.2, 1.7))
        ax.hist(u, bins=bins, color="0.6", alpha=0.7, label="unguided")
        ax.hist(g, bins=bins, color="#1f77b4", alpha=0.6, label="dMFM guided")
        ax.axvline(0.30, color="k", ls="--", lw=0.8)
        ax.set_xlabel(r"predicted C0 ($f_{\mathrm{guide}}$)")
        ax.set_ylabel("count")
        ax.text(0.02, 0.97, f"MAE to 0.30: {np.abs(g - 0.30).mean():.3f}", transform=ax.transAxes,
                va="top", fontsize=6)
        if tag == "a":
            ax.legend(frameon=False, fontsize=6, loc="upper right")
        fig.tight_layout()
        out = OUT / f"fig7_dmfm_{tag}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print("wrote", out.relative_to(REPO), "n =", len(g))




def fig6() -> None:
    """Fig 6: real vs generated vs uniform base composition and C0, parent-disjoint."""
    src = REPO / "results/dna/regen/fig6_table16"
    pos = read_csv(src / "position_freqs.csv")
    glob = read_csv(src / "global_marginals.csv")
    c0 = read_csv(src / "c0_scores.csv")
    sets = ["real", "generated", "uniform"]
    bases = ["A", "C", "G", "T"]
    fig = plt.figure(figsize=(6.85, 5.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1], hspace=0.55, wspace=0.3)
    mats = {}
    for s in sets:
        rows = sorted((r for r in pos if r["source"] == s), key=lambda r: int(r["position"]))
        mats[s] = np.array([[float(r[b]) for b in bases] for r in rows]).T
    vmax = max(m.max() for m in mats.values())
    for j, s in enumerate(sets):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(mats[s], aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_yticks(range(4), bases)
        ax.set_title(s)
        if j == 0:
            ax.set_ylabel("base")
        if j == 2:
            fig.colorbar(im, ax=ax, fraction=0.04)
    for j, s in enumerate(sets):
        ax = fig.add_subplot(gs[1, j])
        d = mats[s] - mats["real"]
        m = np.abs(mats["generated"] - mats["real"]).max()
        im = ax.imshow(d, aspect="auto", vmin=-m, vmax=m, cmap="RdBu_r")
        ax.set_yticks(range(4), bases)
        ax.set_title(f"{s} $-$ real")
        ax.set_xlabel("position")
        if j == 2:
            fig.colorbar(im, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[2, :2])
    w = 0.26
    for i, s in enumerate(sets):
        f = [float(next(r for r in glob if r["source"] == s and r["base"] == b)["freq"]) for b in bases]
        ax.bar(np.arange(4) + (i - 1) * w, f, width=w, label=s)
    ax.set_xticks(range(4), bases)
    ax.set_ylabel("global frequency")
    ax.legend(frameon=False, fontsize=6)
    ax = fig.add_subplot(gs[2, 2])
    bins = np.linspace(-2, 1.5, 45)
    for s in sets:
        v = [float(r["c0_pred_forward"]) for r in c0 if r["source"] == s and r["c0_pred_forward"]]
        if v:
            ax.hist(v, bins=bins, histtype="step", lw=1.2, density=True, label=s)
    ax.set_xlabel("predicted C0")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=6)
    out = OUT / "fig6_dmfm.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(REPO))


def fig8() -> None:
    """Fig 8: one guided vs unguided dMFM trajectory."""
    rows = read_csv(REPO / "results/dna/dmfm_demos/fig8/dmfm/trace.csv")
    idx = "0"
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 1.9))
    panels = [("grad_norm", r"$\|\nabla V_t\|$", True),
              ("value", r"estimated $V_t$", False),
              ("endpoint_hard_score", "guide score of endpoint", False)]
    for ax, (col, ylab, logy) in zip(axes, panels):
        for arm, c in (("unguided", "0.5"), ("guided", "#1f77b4")):
            g = sorted((r for r in rows if r["sample_idx"] == idx and r["arm"] == arm),
                       key=lambda r: float(r["t"]))
            x = [float(r["t"]) for r in g]
            y = [float(r[col]) for r in g]
            if logy:
                x = [a for a, b in zip(x, y) if b > 0]
                y = [b for b in y if b > 0]
            ax.plot(x, y, color=c, lw=1.2, label=arm)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("$t$")
        ax.set_ylabel(ylab)
    axes[2].axhline(0.3, color="k", ls="--", lw=0.8)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = OUT / "fig8_dmfm.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    fig5()
    fig7()
    fig6()
    fig8()
