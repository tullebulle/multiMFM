"""One house style for every figure in the paper.

Before this module the paper's figures were written by five agents in five styles:
three different font sizes, four figure sizes, two palettes, markers that meant one
method in one panel and another method in the next, and error bars drawn in some
panels but not others even where the CSV carried them. This module is the single
place those decisions live, so a reader comparing two panels is comparing the
results and not the plotting.

Use it as::

    from multimfm import figstyle as fs

    fs.use_paper_style()
    fig, ax = fs.new_figure("panel")
    fs.plot_series(ax, "dmfm", x, y, yerr=sem)
    fs.plot_series(ax, "fk", x2, y2, yerr=sem2)
    fs.finish(ax, xlabel="...", ylabel="...", logx=True)
    fs.save(fig, "rerun_dna_fig4_right_nfe", sources=[...], script="...")

What the module guarantees
--------------------------
* **Identity is never carried by colour alone.** Every series in `SERIES` has its
  own marker *and* its own line style, and `plot_series` always draws both. This
  is not decoration: see "Palette" below.
* **Error bars are not optional.** `plot_series` refuses to draw a series whose
  `yerr` is `None` unless the caller passes `yerr_absent="none measured"` (or
  another reason), which is then recorded in the figure manifest. A tie has to
  look like a tie.
* **Linear axes are not truncated.** `finish(..., include_zero=True)` is the
  default for a linear y-axis of a non-negative error metric, so a small gap
  cannot be inflated by cropping the baseline away. When a log axis is used,
  `save` records it in the manifest so the caption can say so.
* **Two colours can never collide inside one panel.** `plot_series` tracks the
  colours already used on an axis and raises if a second series asks for one.

Palette
-------
The five hues are the ones Figure 4 already used (matplotlib `tab10`), kept
deliberately: two published panels use them, and readers of the submitted PDF
recognise them.

They are **not** colour-blind safe on their own. Checked with the `dataviz`
skill's `validate_palette.js` (OKLab dE x100, light surface, all pairs):

    red/blue      protan 21.1   normal 31.7   ok
    red/orange    deutan 16.8   normal 18.6   ok
    red/green     deutan  3.9   normal 33.8   FAIL under deuteranopia
    red/purple    deutan 21.3   normal 23.0   ok
    blue/orange   protan 24.6   normal 35.7   ok
    blue/green    tritan  6.8   normal 25.2   weak under tritanopia
    blue/purple   protan  1.7   normal 14.2   FAIL, and below the normal-vision floor
    orange/green  protan  0.7   normal 28.6   FAIL under protanopia (the worst pair)
    orange/purple protan 26.2   normal 28.9   ok
    green/purple  tritan 15.3   normal 31.6   ok

No five-hue categorical palette containing both a red and a green passes; the
fix the validator asks for in the failing band is mandatory secondary encoding,
which is why marker shape and line style are part of `SERIES` and why
`plot_series` will not let a caller drop them. The three failing pairs
(dMFM/beam, best-of-N/MCTS, Feynman-Kac/beam) are all distinguished here by
marker, by dash pattern and by line weight. `docs/figures.md` records this.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator  # noqa: E402

__all__ = [
    "RED", "BLUE", "ORANGE", "GREEN", "PURPLE", "BROWN", "GREY", "INK", "INK2",
    "SERIES", "Series", "SIZES", "FIGURES",
    "use_paper_style", "new_figure", "plot_series", "published_series", "finish",
    "save", "placeholder", "annotate_points", "reference_line", "log_ticks",
    "manifest_path", "read_manifest",
]

# --------------------------------------------------------------------------- palette
RED = "#d62728"      # the headline method, always
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
PURPLE = "#9467bd"
BROWN = "#8c564b"    # the un-searched variant of the headline method
GREY = "#7f7f7f"     # references, floors, "not a method"

INK = "#222222"      # primary text
INK2 = "#555555"     # secondary text, axis furniture
GRID = "#d9d9d9"

# Type sizes. The smallest text on any figure is ANNOT = 7.0 pt, which is the
# floor a two-column ICLR layout can carry at 1:1 inclusion. Nothing here scales
# below that; if a panel needs smaller text it needs to be a wider panel.
BASE_PT = 8.0
TITLE_PT = 8.5
LABEL_PT = 8.0
TICK_PT = 7.0
LEGEND_PT = 7.0
ANNOT_PT = 7.0

# Figure sizes, in inches, for the ICLR two-column layout.
#   column width  ~3.31 in   text width ~6.79 in
#
# Each figure is emitted at the size it is meant to print at, so it goes into the
# document with \includegraphics[width=\columnwidth] (or \textwidth) and is NOT
# rescaled. That is what keeps the 7 pt floor a real floor: a 4-inch figure
# squeezed into a 3.35-inch column would put every 7 pt label at 5.9 pt.
SIZES: dict[str, tuple[float, float]] = {
    "col": (3.35, 3.35),          # one column, legend below the axes
    "panel": (3.35, 2.70),        # one column, legend inside
    "panel_tall": (3.35, 3.20),
    "pair": (6.85, 3.35),         # full text width, two column-width panels
    "wide": (6.85, 2.85),         # full text width, one row
    "wide_tall": (6.85, 4.10),
    "grid23": (6.85, 4.60),       # 2 x 3 small multiples
    "strip4": (6.85, 2.05),       # 1 x 4 image panels (Figure 1)
}


@dataclass(frozen=True)
class Series:
    """How one method is drawn, everywhere it appears."""

    label: str
    color: str
    marker: str
    linestyle: Any
    linewidth: float = 1.5
    markersize: float = 4.5
    zorder: int = 3

    def kwargs(self) -> dict[str, Any]:
        return dict(color=self.color, marker=self.marker, ls=self.linestyle,
                    lw=self.linewidth, ms=self.markersize, zorder=self.zorder)


#: The registry. A method has one appearance in the whole paper.
#:
#: Colours are reused across *disjoint* figures (DPS and MCTS never share a panel,
#: nor do beam search and FMAP); `plot_series` enforces that at draw time rather
#: than trusting the author to remember.
SERIES: dict[str, Series] = {
    # headline methods -- heavier line, filled circle, solid
    "dmfm":        Series("dMFM",          RED,    "o", "-",                 2.2, 5.0, 5),
    "multimfm_ss": Series("multiMFM-SS",   RED,    "o", "-",                 2.2, 5.0, 5),
    "multimfm":    Series("multiMFM",      BROWN,  "v", (0, (5, 1.6)),       1.5, 4.5, 4),
    # search / guidance baselines
    "best_of_n":   Series(r"Best-of-$N$",  BLUE,   "s", (0, (4.5, 1.8)),     1.5, 4.2, 3),
    "fk":          Series("Feynman-Kac",   ORANGE, "D", (0, (5, 1.5, 1, 1.5)), 1.5, 4.0, 3),
    "beam":        Series("Beam search",   GREEN,  "^", (0, (3, 1, 1, 1, 1, 1)), 1.5, 4.5, 3),
    "mcts":        Series("MCTS",          PURPLE, "P", (0, (1, 1.4)),       1.5, 4.8, 3),
    "dps":         Series("DPS",           PURPLE, "P", (0, (1, 1.4)),       1.5, 4.8, 3),
    # value-estimator references (Figure 4 left)
    "fmap":        Series("FMAP ODE",      GREEN,  "^", (0, (3, 1, 1, 1, 1, 1)), 1.5, 4.5, 3),
    "sde_mc":      Series("SDE MC floor",  GREY,   "v", "-",                 1.3, 4.0, 2),
    # not a method: a measured reference level
    "reference":   Series("reference",     GREY,   "",  (0, (1, 2.5)),       1.0, 0.0, 1),
}

#: A *sequential* ramp, for a series index that is ordered rather than categorical
#: (sequence length, budget, temperature). One hue, light to dark, so the reader
#: reads the order off the ink. Validated with the `dataviz` skill's
#: `validate_palette.js --ordinal`: monotone lightness, adjacent dL >= 0.06,
#: light end 2.46:1 against the surface, hue spread 18 deg -- all pass.
ORDINAL = ["#74a9cf", "#4292c6", "#2171b5", "#084594"]
ORDINAL_MARKERS = ["o", "s", "^", "D"]

#: Where figures are written. Owned by the figure agent; see the README there.
FIGURES = Path(__file__).resolve().parents[2] / "notebooks" / "figures"


# --------------------------------------------------------------------------- style
def use_paper_style() -> None:
    """Install the paper rcParams. Idempotent; call once per script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,          # embed TrueType, not Type3 -- ICLR wants this
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": BASE_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        "figure.titlesize": TITLE_PT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.35,
        "legend.columnspacing": 1.2,
        "lines.solid_capstyle": "round",
        "errorbar.capsize": 2.0,
    })


def new_figure(size: str | tuple[float, float] = "panel", nrows: int = 1, ncols: int = 1,
               **kw: Any):
    """A figure at one of the layout sizes in :data:`SIZES`."""
    figsize = SIZES[size] if isinstance(size, str) else size
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


# --------------------------------------------------------------------------- drawing
_USED_COLORS: dict[int, dict[str, str]] = {}


def plot_series(ax, key: str, x: Sequence[float], y: Sequence[float], *,
                yerr: Sequence[float] | None = None,
                yerr_absent: str | None = None,
                label: str | None = None,
                detached: bool = False,
                muted: bool = False,
                **kw: Any):
    """Draw one registered series, with its error bars.

    Parameters
    ----------
    yerr
        Required. Pass the standard error / half-CI the results carry. Several of
        this paper's comparisons are ties within one SEM and must *look* like ties.
    yerr_absent
        The reason there is no `yerr`, when there genuinely is none (for example
        the submitted figure's baseline rows, whose `mae_sem` is 0.000000 in every
        row because no producing code ever wrote one). Recorded in the manifest.
    detached
        Draw markers only, no connecting line -- for a series whose budget knob
        cannot reach the shared ladder, so that a line would imply measurements
        that were not made.
    muted
        Draw at reduced weight and opacity: a "for reference" overlay, not a result.
    """
    if key not in SERIES:
        raise KeyError(f"unregistered series {key!r}; add it to figstyle.SERIES")
    s = SERIES[key]

    used = _USED_COLORS.setdefault(id(ax), {})
    if used.get(s.color, key) != key:
        raise ValueError(
            f"colour {s.color} is already used on this axis by {used[s.color]!r}; "
            f"{key!r} cannot share it -- give one of them its own entry in SERIES")
    used[s.color] = key

    if yerr is None and yerr_absent is None:
        raise ValueError(
            f"{key!r}: no yerr given. Draw the error bars the results carry, or say "
            f"why there are none via yerr_absent=...")

    opts = s.kwargs()
    if detached:
        opts["ls"] = "none"
        opts["mfc"] = "none"
    if muted:
        opts["lw"] = min(opts["lw"], 1.1)
        opts["ms"] = max(opts["ms"] - 1.3, 2.5)
        opts["alpha"] = 0.75
        opts["zorder"] = 2
    opts.update(kw)

    return ax.errorbar(x, y, yerr=yerr, label=label if label is not None else s.label,
                       capsize=2.0, elinewidth=0.9, capthick=0.9, **opts)


def published_series(ax, key: str, x: Sequence[float], y: Sequence[float], *,
                     yerr: Sequence[float] | None = None, suffix: str = "published",
                     **kw: Any):
    """Overlay a *published* curve: same colour and marker, dotted and light.

    The distinction between what was published and what we re-measured is the one
    a reader must never get wrong, so it is carried by weight and by the label,
    not only by the file name.
    """
    s = SERIES[key]
    _USED_COLORS.setdefault(id(ax), {}).setdefault(s.color, key)
    opts = dict(color=s.color, marker=s.marker, ls=(0, (1, 2)), lw=1.0,
                ms=max(s.markersize - 1.5, 2.5), mfc="none", alpha=0.8, zorder=2)
    opts.update(kw)
    return ax.errorbar(x, y, yerr=yerr, label=f"{s.label} ({suffix})",
                       capsize=1.5, elinewidth=0.7, **opts)


def plot_ordinal(ax, index: int, x: Sequence[float], y: Sequence[float], *,
                 label: str, yerr: Sequence[float] | None = None,
                 yerr_absent: str | None = None, n: int = 4, **kw: Any):
    """Draw one step of an *ordered* family (lengths, budgets) from :data:`ORDINAL`.

    Same error-bar rule as :func:`plot_series`: give `yerr`, or say why there is none.
    """
    if yerr is None and yerr_absent is None:
        raise ValueError(f"{label!r}: no yerr given; pass it or set yerr_absent=...")
    step = ORDINAL[round(index * (len(ORDINAL) - 1) / max(n - 1, 1))]
    marker = ORDINAL_MARKERS[index % len(ORDINAL_MARKERS)]
    opts = dict(color=step, marker=marker, ls="-", lw=1.4, ms=3.8, zorder=3)
    opts.update(kw)
    return ax.errorbar(x, y, yerr=yerr, label=label, capsize=1.8, elinewidth=0.8,
                       capthick=0.8, **opts)


def reference_line(ax, value: float, text: str, *, orient: str = "h",
                   band: tuple[float, float] | None = None, at: float = 0.985,
                   ha: str = "right") -> None:
    """A measured reference level (a noise floor, an unguided baseline)."""
    if orient == "h":
        ax.axhline(value, color=GREY, lw=0.9, ls=(0, (1, 2.5)), zorder=1)
        ax.text(at, value, text, transform=ax.get_yaxis_transform(), ha=ha,
                va="bottom", fontsize=ANNOT_PT - 0.5, color=INK2, zorder=6)
    else:
        ax.axvline(value, color=GREY, lw=0.9, ls=(0, (1, 2.5)), zorder=1)
        ax.text(value, 0.985, text, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=ANNOT_PT - 0.5, color=INK2, zorder=6)
    if band is not None:
        ax.axhspan(band[0], band[1], color=GREY, alpha=0.11, lw=0, zorder=0)


def annotate_points(ax, xs: Iterable[float], ys: Iterable[float], texts: Iterable[str], *,
                    color: str = INK2, dy: float = -11.0, dx: float = 0.0,
                    ha: str = "center") -> None:
    """Direct labels on a series -- the secondary encoding the palette needs."""
    for x, y, t in zip(xs, ys, texts):
        if not t:
            continue
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=ANNOT_PT, color=color, ha=ha, zorder=6)


def log_ticks(ax, axis: str, values: Sequence[float]) -> None:
    """Explicit, readable decade-free ticks on a log axis (1, 2, 3, 5, 8 ...)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(list(values)))
    a.set_major_formatter(FixedFormatter([f"{v:g}" for v in values]))
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def placeholder(ax, title: str, reason: str) -> None:
    """A clearly-marked pending panel. Never a guess at what the data will say."""
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                               facecolor="#f7f7f7", edgecolor=RED, lw=1.0,
                               hatch="////", alpha=0.65, zorder=0))
    ax.text(0.5, 0.60, "DATA PENDING", transform=ax.transAxes, ha="center", va="center",
            fontsize=TITLE_PT + 2, color=RED, weight="bold", zorder=3)
    ax.text(0.5, 0.42, reason, transform=ax.transAxes, ha="center", va="center",
            fontsize=ANNOT_PT, color=INK2, wrap=True, zorder=3)
    if title:
        ax.set_title(title, fontsize=TITLE_PT, color=INK)


def finish(ax, *, xlabel: str = "", ylabel: str = "", title: str = "",
           logx: bool = False, logy: bool = False, logx_base: float = 10.0,
           include_zero: bool = True, legend: bool | dict[str, Any] = True,
           grid_alpha: float = 0.55) -> None:
    """Axis furniture, applied the same way on every panel.

    `include_zero` keeps a *linear* y-axis anchored at zero, so a small difference
    can never be made to look large by cropping the baseline off the bottom. It is
    ignored on a log axis (where zero does not exist) and when the data go
    negative.
    """
    if logx:
        ax.set_xscale("log", base=logx_base)
    if logy:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=TITLE_PT, color=INK)
    ax.grid(True, alpha=grid_alpha, which="major")
    if logy:
        ax.grid(True, alpha=grid_alpha * 0.45, which="minor")
    if include_zero and not logy:
        lo, hi = ax.get_ylim()
        if lo > 0 and hi > 0:
            ax.set_ylim(bottom=0.0)
    if legend is not False:
        kw = dict(loc="best", fontsize=LEGEND_PT, frameon=False, handlelength=2.4)
        if isinstance(legend, dict):
            kw.update(legend)
        ax.legend(**kw)


# --------------------------------------------------------------------------- output
def manifest_path() -> Path:
    return FIGURES / "_manifest.json"


def read_manifest() -> dict[str, Any]:
    p = manifest_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _axis_note(fig) -> list[str]:
    notes = []
    for i, ax in enumerate(fig.axes):
        if not ax.has_data():
            continue
        xs, ys = ax.get_xscale(), ax.get_yscale()
        if xs != "linear" or ys != "linear":
            notes.append(f"axes[{i}]: x {xs}, y {ys}")
    return notes


def save(fig, name: str, *, paper_item: str = "", sources: Sequence[str] = (),
         script: str = "", caption: str = "", diagnostic: bool = False,
         notes: Sequence[str] = (), formats: Sequence[str] = ("png", "pdf"),
         dpi: int = 200) -> list[Path]:
    """Write the figure as PNG (200 dpi) and PDF (vector), and record provenance.

    Every paper figure gets both: the PNG for review and notebooks, the PDF for the
    submission. The manifest that `notebooks/figures/README.md` is generated from
    records which results files the figure was built from, which script rebuilds
    it, and whether either axis is logarithmic -- so a caption can say so.
    """
    out_dir = FIGURES / "diagnostics" if diagnostic else FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)

    man = read_manifest()
    man[name] = {
        "kind": "diagnostic" if diagnostic else "paper",
        "paper_item": paper_item,
        "files": [str(p.relative_to(FIGURES.parents[1])) for p in written],
        "sources": list(sources),
        "script": script,
        "caption": caption,
        "log_axes": _axis_note(fig),
        "notes": list(notes),
    }
    manifest_path().write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    for p in written:
        print(f"wrote {p}")
    return written


# --------------------------------------------------------------------------- helpers
def loglog_crossing(x_a: Sequence[float], y_a: Sequence[float],
                    x_b: Sequence[float], y_b: Sequence[float]) -> float | None:
    """Where curve A stops being below curve B, interpolated linearly in log-log.

    Returned as an x value, or None if they do not cross inside the range both
    curves cover. Used to annotate a crossover instead of asserting one.
    """
    lo = max(min(x_a), min(x_b))
    hi = min(max(x_a), max(x_b))
    if not (hi > lo):
        return None

    def interp(xs, ys, x):
        lx = [math.log(v) for v in xs]
        ly = [math.log(v) for v in ys]
        for i in range(len(lx) - 1):
            if lx[i] <= math.log(x) <= lx[i + 1]:
                f = (math.log(x) - lx[i]) / (lx[i + 1] - lx[i])
                return math.exp(ly[i] + f * (ly[i + 1] - ly[i]))
        return None

    n = 400
    prev = None
    for i in range(n + 1):
        x = math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / n)
        a, b = interp(x_a, y_a, x), interp(x_b, y_b, x)
        if a is None or b is None:
            continue
        d = a - b
        if prev is not None and prev[1] < 0 <= d:
            return x
        prev = (x, d)
    return None
