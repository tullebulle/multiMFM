"""Fig. 1 four-panel plot (p0, p1, p1 e^{r_Norway}, p1 e^{r_Smiley}).

The plotting code (knobs, ``plot_panel``, layout) is RECOVERED near-verbatim from
the editor history of ``norway/norway_base_smiley_four_panel.ipynb`` (Cursor
local history, last save 2026-05-06 15:20 UTC, i.e. the notebook that drew the
paper figure; copy in docs/provenance/toy/recovered/cursor_history/).  Changes:
wrapped in functions, panels passed in instead of loaded from
``figures/norway_base_smiley_four_panel_data.pt``, and the unused
``panel_target`` helper dropped.  Note the notebook draws the non-smiley panels
colour by colour (blue, white, red), so red is always on top; that is why red
dominates the p0 / p1 panels of the paper figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from .data import COLOR_NAMES, NUM_CLASSES
from .reward_smiley import SMILEY_SHAPE, smiley_scores

PLOT_COLORS = {
    "blue": "#006DCC",
    "white": "#F8F8F8",
    "red": "#FF2D20",
}
TITLES = [
    r"$p_0(x)$",
    r"$p_1(x)$",
    r"$p_1(x)\,e^{r_{\mathrm{Norway}}(x)}$",
    r"$p_1(x)\,e^{r_{\mathrm{Smiley}}(x)}$",
]
PANEL_KEYS = ["prior", "base", "norway", "smiley"]
PANEL_KINDS = ["prior", "base", "norway", "smiley"]
LIMITS = {
    "prior": (-3.0, 3.0),
    "base": (-1.08, 1.08),
    "norway": (-1.08, 1.08),
    "smiley": (-1.08, 1.08),
}
MARKER_SIZE = {
    "prior": 3.0,
    "base": 3.0,
    "norway": 3.0,
    "smiley": 3.0,
}
ALPHA = 0.90
FIGSIZE = (10.3, 3.35)
WSPACE = 0.0
TITLE_SIZE = 25
LABEL_FONTSIZE = 25
DPI = 220

# Margins.  ``notebook`` is what the last saved state of the figure notebook has
# (left=0.001, right=0.999, wspace=0) -- it clips the y_2 label, and the
# published figure does not, so the notebook was edited once more before the
# final save.  ``published`` reproduces the geometry measured off
# ``figures/norway.pdf`` rendered at 140 dpi (1442 x 469 px): four 269 x 269 px
# panels starting at x = 83, 433, 782, 1131 and spanning y = 140..408.
LAYOUTS = {
    "notebook": dict(left=0.001, right=0.999, bottom=0.13, top=0.88, wspace=0.0),
    "published": dict(left=0.0576, right=0.9709, bottom=0.1301, top=0.7037, wspace=0.2986),
}
DEFAULT_LAYOUT = "published"

RC_PARAMS = {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    # Embed TrueType rather than Type 3. Matplotlib's PDF default is Type 3, which
    # arXiv and the ICLR submission system reject, so without this the Fig 1 PDFs
    # cannot be submitted.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def plot_panel(ax, x, title, kind, smiley_score_kwargs=None):
    x = torch.as_tensor(x).float().cpu()
    c = x[:, :NUM_CLASSES]
    y = x[:, NUM_CLASSES:]
    pred = c.argmax(dim=-1)

    if kind == "smiley":
        kw = dict(SMILEY_SHAPE if smiley_score_kwargs is None else smiley_score_kwargs)
        kw.pop("color_temp", None)
        scores = smiley_scores(y[:, 0], y[:, 1], **kw)
        target = scores.argmax(dim=-1)
        correct = pred == target

        blue_right_eye = correct & (target == 0)
        white_mouth = correct & (target == 1)
        red_left_eye = correct & (target == 2)

        # Base smiley points only, faint.
        for color_idx, name in enumerate(COLOR_NAMES):
            mask = pred == color_idx
            ax.scatter(
                y[mask, 0], y[mask, 1],
                s=MARKER_SIZE["smiley"],
                c=PLOT_COLORS[name],
                alpha=0.35 if name == "red" else 0.55,
                edgecolors="#AFAFAF" if name == "white" else "none",
                linewidths=0.08 if name == "white" else 0.0,
                rasterized=True,
                zorder=1,
            )

        ax.scatter(
            y[red_left_eye, 0], y[red_left_eye, 1],
            s=MARKER_SIZE["smiley"],
            c="#C94A42",
            alpha=0.65,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )

        ax.scatter(
            y[blue_right_eye, 0], y[blue_right_eye, 1],
            s=MARKER_SIZE["smiley"] * 1.15,
            c=PLOT_COLORS["blue"],
            alpha=0.95,
            edgecolors="none",
            rasterized=True,
            zorder=5,
        )

        ax.scatter(
            y[white_mouth, 0], y[white_mouth, 1],
            s=MARKER_SIZE["smiley"] * 1.15,
            c=PLOT_COLORS["white"],
            alpha=0.95,
            edgecolors="#AFAFAF",
            linewidths=0.08,
            rasterized=True,
            zorder=6,
        )

    else:
        # Other plots stay exactly color-by-color, no accuracy/layering logic.
        y_np = y.numpy()
        pred_np = pred.numpy()
        for color_idx, name in enumerate(COLOR_NAMES):
            mask = pred_np == color_idx
            ax.scatter(
                y_np[mask, 0], y_np[mask, 1],
                s=MARKER_SIZE[kind],
                c=PLOT_COLORS[name],
                alpha=ALPHA,
                edgecolors="#AFAFAF" if name == "white" else "none",
                linewidths=0.03 if name == "white" else 0.0,
                rasterized=True,
            )

    ax.set_facecolor("#050505")
    ax.set_xlim(*LIMITS[kind])
    ax.set_ylim(*LIMITS[kind])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=TITLE_SIZE, pad=5)
    ax.set_xlabel(r"$y_1$", fontsize=LABEL_FONTSIZE, labelpad=0)
    ax.set_ylabel(r"$y_2$", fontsize=LABEL_FONTSIZE, labelpad=0)
    ax.set_xticks([])
    ax.set_yticks([])


def four_panel(panels: dict, smiley_score_kwargs=None, out_path=None, titles=TITLES,
               layout: str = DEFAULT_LAYOUT):
    """panels: dict with keys prior, base, norway, smiley -> (N, 5) endpoints."""
    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, constrained_layout=False)
        for i, (ax, key, title, kind) in enumerate(zip(axes, PANEL_KEYS, titles, PANEL_KINDS)):
            plot_panel(ax, panels[key], title, kind, smiley_score_kwargs=smiley_score_kwargs)
            if i != 0:
                ax.set_ylabel("")
        fig.subplots_adjust(**LAYOUTS[layout])
        if out_path is not None:
            fig.savefig(out_path, dpi=DPI)
    return fig
