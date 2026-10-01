"""Shared look for the QM9 figures: LaTeX typography and one palette.

Both of our arms are red and differ by line style, so a reader sees "ours" as one family
at a glance; the baselines take the cool half of a diverging RdBu scale, with DPS -- the
other *guided* method, and so the one our arms are read against -- in neutral grey rather
than a third blue.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

OURS_SS = "#a01228"       # deep red
OURS = "#e34a33"          # lighter red (unused: both arms share OURS_SS)
C_DPS = "#595959"         # neutral grey: the other guided method
C_BON = "#2166ac"         # deep blue
C_FK = "#6baed6"          # light blue
C_FLOOR = "#9e9e9e"

# (colour, marker, linestyle, linewidth, markersize, label)
SERIES = {
    # Both arms are the same red: multiMFM solid and bold (it is also the method the
    # value-estimation panel measures, drawn there identically), multiMFM-SS dashed.
    "ours":      (OURS_SS, "o", "-",              2.5, 5.6, r"multiMFM"),
    "ours_ss":   (OURS_SS, "^", (0, (4.5, 1.6)),  2.1, 5.2, r"multiMFM-SS"),
    "dps":       (C_DPS,   "X", (0, (1, 1.4)),    1.5, 4.6, r"DPS"),
    "best_of_n": (C_BON,   "s", (0, (6, 2)),      1.5, 4.0, r"best-of-$N$"),
    "fk":        (C_FK,    "D", (0, (3, 1.3)),    1.5, 3.8, r"Feynman--Kac"),
    # value-estimation panel: the two deterministic point estimates
    "fmap":      (C_BON,   "s", (0, (6, 2)),      1.5, 4.0, r"FMAP"),
    "mfm":       (OURS_SS, "o", "-",              2.4, 5.6, r"multiMFM"),
}


def use(scale: float = 1.0) -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 9 * scale,
        "axes.labelsize": 10 * scale,
        "axes.titlesize": 10.5 * scale,
        "legend.fontsize": 8.2 * scale,
        "xtick.labelsize": 9 * scale,
        "ytick.labelsize": 9 * scale,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.handlelength": 2.3,
        "legend.labelspacing": 0.32,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def draw(ax, key, x, y, yerr=None, zorder=None):
    c, mk, ls, lw, ms, lab = SERIES[key]
    return ax.errorbar(x, y, yerr=yerr, color=c, marker=mk, ls=ls, lw=lw, ms=ms,
                       capsize=2.2, elinewidth=0.9, label=lab,
                       zorder=zorder if zorder is not None else (5 if key.startswith("ours") else 2))
