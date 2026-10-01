"""QM9 figures built only from files under ``results/qm9``.

``python -m multimfm.qm9_figures baselines`` draws the same-backbone steering
comparison (authors' decision 8) from ``results/qm9/baselines/baseline_sweep_v2.csv``
(written by ``python -m multimfm.qm9_results baselines-v2``). Every point is a
measured run (mean +/- SE over the evaluation seeds); nothing is extrapolated or
hardcoded. NFE on the x-axis is the hook-measured number of generative-model forward
passes (base flow model + meta flow map) per generated molecule; ``--x fwd_equiv``
counts each back-propagated forward as 3 (forward + ~2x backward) instead.

Series (all alpha, n = 100 targets per seed, 128 outer steps):
  multiMFM-SS    Algorithm 3 steer-search at the Table-2 SS config (mu 11), MC = 1..32
  multiMFM       guided endpoint at the Table-2 multiMFM config (mu 3.4), MC = 1..32
  best-of-N      PB-valid best by the guide, N = 4..512
  Feynman-Kac    K = 4..32 particles, lambda = 0.01 (best on the seed-8 tuning sweep)
  DPS            mu = 0.1, guide every 8/4/2/1 steps
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from multimfm import paths

# Reference categorical palette (dataviz skill, light mode), assigned in fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e0dfda"

SERIES = [
    # label, method, run, filter(row) -> bool, knob column, marker
    ("multiMFM-SS (Alg. 3)", "steer_search", "best_lookahead_pbvalid",
     lambda r: _f(r["mu"]) == 11.0, "value_samples", "o"),
    ("multiMFM", "steer_search", "final_endpoint",
     lambda r: _f(r["mu"]) == 3.4, "value_samples", "s"),
    ("best-of-N", "best_of_n", "best_of_n_pbvalid", lambda r: True, "n_candidates", "^"),
    ("Feynman-Kac", "fk", "fk_pbvalid", lambda r: _f(r["fk_lambda"]) == 0.01, "n_particles", "D"),
    ("DPS", "dps", "dps", lambda r: _f(r["mu"]) == 0.1, "guide_every", "v"),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _mean_se(vals):
    vals = [v for v in vals if not math.isnan(v)]
    n = len(vals)
    if not n:
        return float("nan"), float("nan"), 0
    m = sum(vals) / n
    se = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1) / n) if n > 1 else float("nan")
    return m, se, n


def aggregate(rows: list[dict], x: str = "generative") -> dict:
    """{label: [dict(knob, nfe, mae, mae_se, pb, pb_se, n_seeds)] sorted by nfe}."""
    out = {}
    xcol = "nfe_generative" if x == "generative" else "nfe_generative_fwd_equiv"
    for label, method, run, keep, knob, _ in SERIES:
        groups = defaultdict(list)
        for r in rows:
            if r["method"] == method and r["run"] == run and keep(r):
                groups[_f(r[knob])].append(r)
        pts = []
        for k, g in groups.items():
            mae, mae_se, n = _mean_se([_f(r["oracle_mae"]) for r in g])
            pb, pb_se, _ = _mean_se([_f(r["pb_valid"]) for r in g])
            nfe, _, _ = _mean_se([_f(r[xcol]) for r in g])
            pts.append(dict(knob=k, nfe=nfe, mae=mae, mae_se=mae_se, pb=pb, pb_se=pb_se,
                            n_seeds=n, seeds=sorted({int(_f(r["seed"])) for r in g})))
        out[label] = sorted(pts, key=lambda d: d["nfe"])
    base = [r for r in rows if r["method"] == "base"]
    if base:
        m, se, n = _mean_se([_f(r["oracle_mae"]) for r in base])
        pb, _, _ = _mean_se([_f(r["pb_valid"]) for r in base])
        out["_unguided"] = dict(mae=m, mae_se=se, pb=pb, n_seeds=n)
    return out


def plot_baselines(csv_path: Path, out_stem: Path, *, x: str = "generative"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate(read_rows(csv_path), x=x)
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "text.color": INK,
        "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"width_ratios": [1.45, 1]})
    knob_name = {"best-of-N": "N", "Feynman-Kac": "K", "DPS": "every", "multiMFM": "MC",
                 "multiMFM-SS (Alg. 3)": "MC"}
    for (label, *_rest), color in zip(SERIES, PALETTE):
        pts = agg.get(label) or []
        if not pts:
            continue
        marker = _rest[-1]
        xs = [p["nfe"] for p in pts]
        ys = [p["mae"] for p in pts]
        es = [0.0 if math.isnan(p["mae_se"]) else p["mae_se"] for p in pts]
        ax.errorbar(xs, ys, yerr=es, color=color, lw=2, marker=marker, ms=6, mec="white",
                    mew=1.2, capsize=0, elinewidth=1, label=label, zorder=3)
        last = pts[-1]
        dy = {"multiMFM": -11, "multiMFM-SS (Alg. 3)": 6}.get(label, 4)
        ax.annotate(f"{knob_name[label]}={int(last['knob'])}", (last["nfe"], last["mae"]),
                    textcoords="offset points", xytext=(6, dy), fontsize=7.5, color=INK2)
        # panel b: accuracy vs validity at every budget, largest budget emphasised
        bx.plot([p["pb"] for p in pts], ys, color=color, lw=1, alpha=0.5, zorder=2)
        bx.plot([last["pb"]], [last["mae"]], marker=marker, color=color, ms=7, mec="white",
                mew=1.2, zorder=3, ls="none")
    ung = agg.get("_unguided")
    if ung:
        ax.axhline(ung["mae"], color=INK2, lw=1, ls=(0, (1, 3)))
        ax.text(0.01, ung["mae"] * 1.05, "unguided", color=INK2, fontsize=7.5, va="bottom",
                transform=ax.get_yaxis_transform())
        bx.plot([ung["pb"]], [ung["mae"]], marker="x", color=INK2, ms=6, ls="none")
        bx.annotate("unguided", (ung["pb"], ung["mae"]), textcoords="offset points",
                    xytext=(-4, 5), ha="right", fontsize=7.5, color=INK2)
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    yt = [0.5, 0.7, 1, 1.5, 2, 3, 5, 8]
    for a in (ax, bx):
        a.set_yscale("log")
        a.yaxis.set_major_locator(FixedLocator(yt))
        a.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in yt]))
        a.yaxis.set_minor_locator(NullLocator())
        a.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xscale("log")
    ax.set_xlabel("generative-model forward passes per molecule" if x == "generative"
                  else "forward-equivalents per molecule (fwd + 2 x bwd)")
    ax.set_ylabel(r"oracle MAE, $\alpha$ [Bohr$^3$]")
    ax.legend(loc="lower left")
    bx.set_xlabel("PoseBusters validity")
    bx.set_ylabel(r"oracle MAE, $\alpha$ [Bohr$^3$]")
    bx.set_title("each method across its budget; marker = largest budget", fontsize=8, color=INK2)
    seeds = sorted({s for pts in agg.values() if isinstance(pts, list) for p in pts for s in p["seeds"]})
    fig.suptitle(f"QM9 $\\alpha$ steering at matched compute (same base model and targets; "
                 f"n = 100 per seed, mean $\\pm$ SE over seeds {seeds})", fontsize=9, color=INK)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    return fig, agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--csv", type=Path,
                   default=paths.RESULTS / "qm9" / "baselines" / "baseline_sweep_v2.csv")
    b.add_argument("--out", type=Path,
                   default=paths.RESULTS / "qm9" / "baselines" / "fig_qm9_steering_nfe")
    b.add_argument("--x", choices=["generative", "fwd_equiv"], default="generative")
    args = p.parse_args()
    if args.cmd == "baselines":
        _, agg = plot_baselines(args.csv, args.out, x=args.x)
        for label, pts in agg.items():
            if isinstance(pts, list):
                print(label, [(int(q["knob"]), round(q["nfe"]), round(q["mae"], 3)) for q in pts])
        print("wrote", args.out)


if __name__ == "__main__":
    main()
