#!/usr/bin/env python
"""Rebuild every paper figure from `results/`, through the one house style.

    ~/micromamba/envs/multimfm/bin/python notebooks/figures/build.py          # everything
    ~/micromamba/envs/multimfm/bin/python notebooks/figures/build.py fig4r    # one figure
    ~/micromamba/envs/multimfm/bin/python notebooks/figures/build.py --list

CPU only, a few seconds, no checkpoints. Every panel is drawn by
`multimfm.figstyle`, so palette, markers, type sizes, error-bar conventions and
figure sizes are the same across the paper; see that module's docstring for what
the style guarantees and what it refuses to draw.

Naming, which is load-bearing
-----------------------------
    published_*   the values in the submitted PDF, digitized from its vector
                  figures. NOT a measurement of the current code.
    rerun_*       our re-measurement of a published experiment.
    new_*         an experiment with no counterpart in the submission.

Output: PNG at 200 dpi and PDF (vector) side by side. Analysis-only plots go to
`notebooks/figures/diagnostics/`; the top level is the paper set.

Provenance for each figure is written to `notebooks/figures/_manifest.json` and
rendered into `notebooks/figures/README.md` by `--readme` (run by default).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from multimfm import figstyle as fs  # noqa: E402

RESULTS = REPO / "results"
FIGS = REPO / "notebooks" / "figures"
DIAG = FIGS / "diagnostics"

# `results/...` path -> the method key it is drawn with
DNA_PANEL_B = {"dmfm": "dmfm", "best_of_n": "best_of_n", "fk": "fk",
               "beam": "beam", "mcts": "mcts"}
QM9_METHODS = {"multiMFM-SS": "multimfm_ss", "multiMFM": "multimfm",
               "best-of-N": "best_of_n", "Feynman-Kac": "fk", "DPS": "dps"}

BUILDERS: dict[str, callable] = {}


def builder(name: str):
    def deco(f):
        BUILDERS[name] = f
        return f
    return deco


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO))


# =============================================================================== Fig 4 panels
# Both panels are drawn by a function that takes an axis, so the same code makes
# the two column-width standalone files and the full-width combined figure. The
# paper's Figure 4 is the pair; the singles exist for a one-column placement.

def _panel_fig4_right(ax, d: pd.DataFrame, *, compact: bool = False) -> str:
    """Target error against the measured forward-equivalent compute axis.

    k = 1.1 backward-to-forward, measured on an H200 for this code path
    (`docs/nfe_accounting_audit.md` §2). The published axis bills a
    back-propagated forward at zero, which slides the dMFM curve left relative
    to the pure-forward baselines. The full measured budget ladder of every
    method is drawn: dMFM leads below the crossover, Feynman-Kac above it, and
    both regions are inside the panel.
    """
    curves = {}
    for m, key in DNA_PANEL_B.items():
        g = d[d.method == m].sort_values("axis_k1p1")
        if not len(g):
            continue
        fs.plot_series(ax, key, g.axis_k1p1.to_numpy(), g.mae.to_numpy(),
                       yerr=g.se.to_numpy())
        curves[key] = (g.axis_k1p1.to_numpy(), g.mae.to_numpy())

    dm = d[d.method == "dmfm"].sort_values("axis_k1p1")
    labels = [c.replace("MC=", "MC ") for c in dm.config]
    if compact:                      # only the ends, or the labels collide
        labels = [labels[0]] + [""] * (len(labels) - 2) + [labels[-1]]
    fs.annotate_points(ax, dm.axis_k1p1, dm.mae, labels, color=fs.RED, dy=8)

    note = ""
    if "dmfm" in curves and "fk" in curves:
        x = fs.loglog_crossing(*curves["dmfm"], *curves["fk"])
        if x is not None:
            ax.axvline(x, color=fs.INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.text(x * 0.94, 0.55, f"crossover {x:.0f}",
                    transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="center", fontsize=fs.ANNOT_PT - 0.5,
                    color=fs.INK2, zorder=6)
            ax.text(x * 0.93, 0.02, "dMFM\nleads", transform=ax.get_xaxis_transform(),
                    ha="right", va="bottom", fontsize=fs.ANNOT_PT - 0.5, color=fs.RED)
            ax.text(x * 1.08, 0.02, "Feynman-Kac\nleads", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=fs.ANNOT_PT - 0.5, color=fs.ORANGE)
            note = (f"dMFM leads below {x:.0f} forward-equivalent NFE and Feynman-Kac above "
                    f"it; both regions are inside the plotted range.")

    fs.finish(ax, xlabel="forward-equivalent NFE per returned sample",
              ylabel=r"$|f_{\mathrm{cyc}}(x) - y^\star|$",
              title="(right) target error against compute",
              logx=True, include_zero=True, legend=False)
    fs.log_ticks(ax, "x", [200, 500, 1000, 2000, 4000])
    return note


FIG4L_LABELS = {"dmfm": "dMFM (NFE 1)", "dps": "DPS denoiser",
                "fmap": "FMAP ODE (NFE 256)", "sde_mc": "SDE MC floor (omitted from the paper)"}


def _panel_fig4_left(ax, d: pd.DataFrame, floor: float, *, compact: bool = False) -> None:
    """Value-estimate error against the number of posterior samples."""
    for key in ("dmfm", "dps", "fmap", "sde_mc"):
        g = d[d.method == key]
        if len(g):
            fs.plot_series(ax, key, g.n_samples.to_numpy(), g.abs_V_err_mean.to_numpy(),
                           yerr=g.abs_V_err_sem.to_numpy(), label=FIG4L_LABELS[key])
    lo = min(d.abs_V_err_mean.min(), floor) * 0.55
    txt = (f"reference's own MC spread ({floor:.2f})" if compact else
           f"reference's own MC spread ({floor:.2f}): nothing\nbelow this line is resolved")
    fs.reference_line(ax, floor, txt, band=(lo, floor), at=0.02, ha="left")
    fs.finish(ax, xlabel="posterior samples $N$", ylabel=r"$|\hat V_t(x) - V_t(x)|$",
              title="(left) value-estimate accuracy",
              logx=True, logx_base=2, logy=True, include_zero=False, legend=False)
    ax.set_ylim(bottom=lo)
    fs.log_ticks(ax, "y", [0.2, 0.5, 1, 2, 5])


def _fig4_sources() -> tuple[pd.DataFrame, pd.DataFrame, float, list[str]]:
    a = RESULTS / "shared" / "nfe_audit" / "dna_axes.csv"
    b = RESULTS / "dna" / "fig4" / "panelA_value_error.csv"
    c = RESULTS / "dna" / "fig4" / "summary.json"
    fl = json.loads(c.read_text())["panelA"]["reference_floor"]["selfsplit_abs_diff_mean"]
    return (pd.read_csv(a),
            pd.read_csv(b).sort_values(["method", "n_samples"]),
            fl, [_rel(a), _rel(b), _rel(c)])


FIG4R_CAPTION = (
    "Target error against forward-equivalent compute for five inference-time steering methods "
    "on the parent-disjoint L=50 DNA flow. The x axis counts generative forward passes plus "
    "1.1 back-propagated forward passes per returned sample (the 1.1 is measured on an H200 for "
    "this code path) and is logarithmic; y is linear and anchored at zero. Error bars are the "
    "standard error over 3 repeats of 100 outputs. ")
FIG4L_CAPTION = (
    "Absolute error of the value estimate against the number of posterior samples, over 32 "
    "conditioning states. Both axes are logarithmic (x base 2). Error bars are the standard "
    "error over conditions. The shaded band is the reference's own Monte-Carlo spread: no curve "
    "can be resolved inside it, so the N >= 256 points sit at or below the noise floor of the "
    "comparison and must not be read as resolved.")
FIG4R_NOTES = ["x axis is LOG; y axis is LINEAR and includes zero",
               "k = 1.1 is measured, not assumed (docs/nfe_accounting_audit.md)",
               "on the published axis (backward billed at zero) the crossover is 582 instead, "
               "and dMFM appears to win at two settings rather than one"]
FIG4L_NOTES = ["x axis LOG (base 2), y axis LOG",
               "the shaded band is the comparison's own noise floor, not a result",
               "the SDE MC floor series is omitted from the published panel"]


@builder("fig4")
def fig4_pair() -> None:
    """DNA Figure 4, both panels, at full text width -- the form the paper uses."""
    axes_df, panel_a, fl, srcs = _fig4_sources()
    fig, (axl, axr) = fs.new_figure("pair", 1, 2)
    _panel_fig4_left(axl, panel_a, fl, compact=True)
    note = _panel_fig4_right(axr, axes_df, compact=True)
    for ax, ncol in ((axl, 2), (axr, 3)):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=ncol,
                  fontsize=fs.LEGEND_PT, frameon=False, handlelength=2.2, columnspacing=0.9)
    fig.tight_layout(w_pad=2.4)
    fs.save(fig, "rerun_dna_fig4", paper_item="Figure 4 (both panels)",
            sources=srcs, script="notebooks/figures/build.py fig4",
            caption="Left: " + FIG4L_CAPTION + " Right: " + FIG4R_CAPTION + note,
            notes=["full text width (6.85 in): include at width=\\textwidth, unscaled"]
                  + FIG4L_NOTES + FIG4R_NOTES)


@builder("fig4r")
def fig4_right() -> None:
    """DNA Figure 4 (right) alone, at one column width."""
    axes_df, _, _, srcs = _fig4_sources()
    fig, ax = fs.new_figure("col")
    note = _panel_fig4_right(ax, axes_df)
    ax.set_title("DNA steering: target error against compute")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=2,
              fontsize=fs.LEGEND_PT, frameon=False, handlelength=2.2, columnspacing=0.9)
    fs.save(fig, "rerun_dna_fig4_right_nfe", paper_item="Figure 4 (right)",
            sources=srcs, script="notebooks/figures/build.py fig4r",
            caption=FIG4R_CAPTION + note,
            notes=["one column wide (3.35 in): include at width=\\columnwidth, unscaled"]
                  + FIG4R_NOTES)

    # --- diagnostic: what the accounting convention alone does to the picture
    fig, axes = fs.new_figure("wide", 1, 2, sharey=True)
    for ax, col, ttl in zip(axes, ("axis_k0", "axis_k1p1"),
                            ("published axis: backward billed at 0",
                             "measured axis: backward billed at 1.1 forwards")):
        for m, key in DNA_PANEL_B.items():
            g = axes_df[axes_df.method == m].sort_values(col)
            if len(g):
                fs.plot_series(ax, key, g[col].to_numpy(), g.mae.to_numpy(),
                               yerr=g.se.to_numpy())
        fs.finish(ax, xlabel="NFE per returned sample", title=ttl, logx=True,
                  legend=(ax is axes[0]))
    axes[0].set_ylabel(r"$|f_{\mathrm{cyc}}(x) - y^\star|$")
    fig.tight_layout()
    fs.save(fig, "dna_fig4_right_axis_convention", diagnostic=True,
            paper_item="Figure 4 (right), accounting sensitivity",
            sources=[srcs[0]], script="notebooks/figures/build.py fig4r",
            caption="The same measurements on the published and the measured compute axis.")


@builder("fig4l")
def fig4_left() -> None:
    """DNA Figure 4 (left) alone, at one column width."""
    _, panel_a, fl, srcs = _fig4_sources()
    fig, ax = fs.new_figure("col")
    _panel_fig4_left(ax, panel_a, fl)
    ax.set_title("Value-estimate accuracy vs $N$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=1,
              fontsize=fs.LEGEND_PT, frameon=False, handlelength=2.2)
    fs.save(fig, "rerun_dna_fig4_left_value_error", paper_item="Figure 4 (left)",
            sources=srcs, script="notebooks/figures/build.py fig4l",
            caption=FIG4L_CAPTION,
            notes=["one column wide (3.35 in): include at width=\\columnwidth, unscaled"]
                  + FIG4L_NOTES)

    # --- diagnostic: robustness variant (different reward and reference)
    rob = RESULTS / "dna" / "fig4" / "panelA_value_error_c0_reward.csv"
    if rob.exists():
        r = pd.read_csv(rob).sort_values(["method", "n_samples"])
        fl2 = json.loads((RESULTS / "dna" / "fig4" / "summary.json").read_text()) \
            .get("panelA_c0_reward", {}).get("reference_floor", {}).get(
                "selfsplit_abs_diff_mean", fl)
        fig, ax = fs.new_figure("col")
        _panel_fig4_left(ax, r, fl2)
        ax.set_title("robustness: cyclizability reward, GLASS-2048 reference")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=1,
                  fontsize=fs.LEGEND_PT, frameon=False)
        fs.save(fig, "dna_fig4_left_robustness", diagnostic=True,
                paper_item="Figure 4 (left), robustness variant",
                sources=[_rel(rob)], script="notebooks/figures/build.py fig4l",
                caption="Panel A's protocol with the right panel's reward and a GLASS reference.")

# =============================================================================== published Fig 4
@builder("published4")
def published_fig4() -> None:
    """The two panels of the SUBMITTED Figure 4, digitized from its vector PDFs.

    These are the published *values*. They are not a measurement of the code in
    this repository, and the models and guide they were produced on are purged.
    Drawn here so the two can be put side by side without anyone having to guess
    which file is which -- hence the `published_` prefix.
    """
    dig = RESULTS / "dna" / "figure_digitized"

    # ---- left
    src = dig / "fig4_left_MC_value_digitized.csv"
    d = pd.read_csv(src)
    keymap = {"dMFM NFE=1": "dmfm", "DPS denoiser": "dps", "FMAP": "fmap"}
    fig, ax = fs.new_figure("panel_tall")
    for series, key in keymap.items():
        g = d[d.series == series].sort_values("N")
        if len(g):
            fs.plot_series(ax, key, g.N.to_numpy(), g.value_error.to_numpy(),
                           yerr=None, yerr_absent="the published panel has no error bars",
                           label=f"{series} (published)")
    fs.finish(ax, xlabel="posterior samples $N$", ylabel=r"$|\hat V_t(x) - V_t(x)|$",
              title="SUBMITTED Figure 4 (left), digitized",
              logx=True, logx_base=2, logy=True, include_zero=False,
              legend=dict(loc="lower left"))
    fs.log_ticks(ax, "y", [0.2, 0.5, 1, 2, 5])
    fs.save(fig, "published_dna_fig4_left_value_error",
            paper_item="Figure 4 (left) as submitted",
            sources=[_rel(src)], script="notebooks/figures/build.py published4",
            caption=("The submitted Figure 4 (left), re-drawn from values digitized off the "
                     "vector PDF (affine residual ~2e-5 pt). Both axes logarithmic. The "
                     "published panel carries no error bars, so none are drawn; it also omits "
                     "the SDE Monte-Carlo floor series that bounds what the comparison can "
                     "resolve."),
            notes=["PUBLISHED VALUES, not a rerun",
                   "no error bars exist in the source figure",
                   "x LOG base 2, y LOG"])

    # ---- right
    src = dig / "fig4_right_nfe_benchmark_digitized.csv"
    d = pd.read_csv(src)
    keymap = {"dMFM": "dmfm", "Best-of-N": "best_of_n", "Feynman-Kac": "fk",
              "Beam Search": "beam", "MCTS": "mcts"}
    fig, ax = fs.new_figure("panel_tall")
    for series, key in keymap.items():
        g = d[d.series == series].sort_values("nfe")
        if not len(g):
            continue
        err = g.errbar_half_width.to_numpy()
        has = bool(np.any(err > 0))
        fs.plot_series(ax, key, g.nfe.to_numpy(), g.target_error.to_numpy(),
                       yerr=err if has else None,
                       yerr_absent=None if has else
                       "source rows carry mae_sem = 0.000000 (no producing code)",
                       label=f"{series} (published)")
    fs.finish(ax, xlabel="NFE (generative model, per returned sample)",
              ylabel=r"$|f_{\mathrm{cyc}}(x) - y^\star|$",
              title="SUBMITTED Figure 4 (right), digitized",
              logx=True, include_zero=True, legend=dict(loc="upper right"))
    fs.log_ticks(ax, "x", [200, 300, 500, 1000, 2000])
    fs.save(fig, "published_dna_fig4_right_nfe",
            paper_item="Figure 4 (right) as submitted",
            sources=[_rel(src)], script="notebooks/figures/build.py published4",
            caption=("The submitted Figure 4 (right), re-drawn from values digitized off the "
                     "vector PDF. x is logarithmic, y is linear from zero. The four baseline "
                     "curves came from a CSV hard-coded in the benchmark script whose "
                     "producing code was never in the repository and whose mae_sem is "
                     "0.000000 in every row, so no error bars can honestly be drawn for them."),
            notes=["PUBLISHED VALUES, not a rerun",
                   "baseline error bars do not exist in the source",
                   "compute axis bills a back-propagated forward at zero"])

    # ---- diagnostic: published vs rerun, on the axis each was drawn with
    reruns = RESULTS / "dna" / "fig4" / "panelB_curves.csv"
    if reruns.exists():
        pb = pd.read_csv(reruns)
        old = pd.read_csv(src)
        fig, ax = fs.new_figure("wide")
        for series, key in keymap.items():
            g = old[old.series == series].sort_values("nfe")
            if len(g):
                fs.published_series(ax, key, g.nfe.to_numpy(), g.target_error.to_numpy(),
                                    suffix="submitted, split65k")
        for m, key in DNA_PANEL_B.items():
            g = pb[pb.method == m].sort_values("nfe_gen_per_output")
            if len(g):
                fs.plot_series(ax, key, g.nfe_gen_per_output.to_numpy(),
                               g.mae_mean.to_numpy(), yerr=g.mae_sem.to_numpy(),
                               label=f"{fs.SERIES[key].label} (rerun)")
        fs.finish(ax, xlabel="NFE (generative model, per returned sample)",
                  ylabel=r"$|f_{\mathrm{cyc}}(x) - y^\star|$",
                  title="submitted (dotted, purged split65k models) vs rerun (solid, parent-disjoint)",
                  logx=True, include_zero=True,
                  legend=dict(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1))
        fs.log_ticks(ax, "x", [200, 300, 500, 1000, 2000, 4000])
        fs.save(fig, "dna_fig4_right_submitted_vs_rerun", diagnostic=True,
                paper_item="Figure 4 (right), submitted vs rerun",
                sources=[_rel(src), _rel(reruns)],
                script="notebooks/figures/build.py published4",
                caption=("Different models, different guide and really-implemented baselines, "
                         "so the two sets are not numerically comparable; the overlay only "
                         "shows what changed."))


# =============================================================================== QM9
@builder("qm9")
def qm9_steering() -> None:
    """QM9 inference-time steering on a shared, measured compute axis.

    Not in the submission. Three things this panel deliberately does not do:

    * It does not trim best-of-N. Its measured ladder runs to N = 512, where it
      reaches 0.819 +- 0.048 against multiMFM-SS's 1.006 +- 0.045 -- i.e. it wins
      at the top of the range. Earlier versions of this plot cut the axis at
      ~10,000 NFE, which removed exactly that.
    * It does not draw the DPS `sample_steps` extension. That sweep delivers
      1/N of the guidance dose as N grows (`docs/STATE_OF_CLAIMS.md`), so its
      apparent degradation with compute is our handicap, not a property of DPS.
      The dose-matched rerun is Slurm job 49455395. Only DPS's `guide_every`
      ladder is shown, detached, with the confound stated on the panel.
    * It uses the measured forward-equivalent axis (QM9: a guided step costs
      2.89-3.00x the generative forwards it is billed for, so k = 2 is the right
      column), not the axis that bills back-propagated forwards at zero.
    """
    src = RESULTS / "shared" / "nfe_audit" / "qm9_axes_untrimmed.csv"
    d = pd.read_csv(src)

    fig, ax = fs.new_figure("col")
    for method, key in QM9_METHODS.items():
        g = d[d.method == method].sort_values("axis_k2")
        if not len(g):
            continue
        detached = key == "dps"
        fs.plot_series(ax, key, g.axis_k2.to_numpy(), g.mae.to_numpy(),
                       yerr=g.se.to_numpy(), detached=detached,
                       label=fs.SERIES[key].label + (" (dose-confounded*)" if detached else ""))

    # Selective direct labels only: the two ends of the headline ladder. Labelling
    # every point would put six numbers into the region where three curves cross.
    head = d[d.method == "multiMFM-SS"].sort_values("axis_k2")
    ends = head.iloc[[0, -1]]
    fs.annotate_points(ax, ends.axis_k2, ends.mae,
                       ["MC 1", "MC 32"], color=fs.RED, dy=10)

    fig.text(0.0, -0.135,
             "*DPS's budget knob saturates below the shared ladder and its dose rises\n"
             " with its budget, so those four points are drawn unjoined. Its sample-\n"
             " steps extension is withheld: our sampler delivers 1/N of the dose as N\n"
             " grows, so the degradation that sweep shows is our handicap, not DPS.\n"
             " Dose-matched rerun: Slurm job 49455395.",
             fontsize=fs.ANNOT_PT - 0.7, color=fs.INK2, va="top", ha="left")

    fs.finish(ax, xlabel="forward-equivalent NFE per returned molecule",
              ylabel=r"oracle MAE, $\alpha$ [Bohr$^3$]",
              title=r"QM9 $\alpha$ steering at matched compute",
              logx=True, logy=True, include_zero=False,
              legend=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=2,
              fontsize=fs.LEGEND_PT, frameon=False, handlelength=2.2, columnspacing=0.9)
    fs.log_ticks(ax, "y", [0.8, 1, 1.5, 2, 3, 5, 8])
    fs.log_ticks(ax, "x", [200, 500, 2000, 10000, 50000])
    ax.set_ylim(0.70, 9.8)
    fs.save(fig, "new_qm9_steering_nfe", paper_item="QM9 steering comparison (new)",
            sources=[_rel(src)], script="notebooks/figures/build.py qm9",
            caption=("Oracle MAE on alpha against forward-equivalent compute, n = 100 "
                     "molecules per seed, mean +- SE over seeds 1/2/3, same base model and "
                     "target set for every method. The x axis counts generative forward passes "
                     "plus twice the back-propagated ones, the measured cost of a guided step "
                     "here. Both axes are logarithmic: the y range "
                     "spans 0.82 to 8.1 Bohr^3 and a linear axis would compress the region "
                     "where the methods cross into the bottom eighth of the panel. Every "
                     "method is drawn over its full measured budget ladder; best-of-N "
                     "overtakes multiMFM-SS at the top of that range."),
            notes=["one column wide (3.35 in): include at width=\\columnwidth, unscaled",
                   "x LOG, y LOG -- state this in the caption",
                   "best-of-N is NOT trimmed: it wins at N = 512",
                   "DPS sample_steps extension withheld (1/N dose artifact, job 49455395)",
                   "multiMFM-SS's advantage is partly bought off this axis: 2x the "
                   "regressor calls and 4-13x the PoseBusters checks of best-of-N"])


# =============================================================================== Fig 5
@builder("fig5")
def fig5_gradient_mc() -> None:
    """DNA Figure 5 / Tables 14-15: gradient accuracy against posterior samples."""
    rewards = [("gc", "GC target"), ("motif", "Motif"), ("conjunction", "Conjunction")]
    frames = {}
    for key, _ in rewards:
        p = RESULTS / "dna" / "scale_free_mc" / key / "gradient_scale_free_summary_all_lengths.csv"
        if p.exists():
            frames[key] = (pd.read_csv(p), p)
    if not frames:
        print("fig5: no summary CSVs found; skipping")
        return

    lengths = sorted({int(v) for df, _ in frames.values() for v in df.length.unique()})
    fig, axes = fs.new_figure("grid23", 2, 3, sharex=True, sharey="row")
    for col, (key, title) in enumerate(rewards):
        if key not in frames:
            for row in (0, 1):
                fs.placeholder(axes[row][col], title, "run not collected")
            continue
        df = frames[key][0]
        for row, (ycol, lo, hi, ylab) in enumerate([
                ("global_relative_l1", "relative_l1_ci95_low", "relative_l1_ci95_high",
                 r"relative $L_1$ error"),
                ("mean_cosine", "cosine_ci95_low", "cosine_ci95_high",
                 "mean cosine similarity")]):
            ax = axes[row][col]
            for i, L in enumerate(lengths):
                g = df[df.length == L].sort_values("mc")
                if not len(g):
                    continue
                # CI95 from the results, drawn as an asymmetric error bar -- the
                # earlier version of this figure drew no uncertainty at all even
                # though the columns were right there.
                y = g[ycol].to_numpy()
                err = np.vstack([np.abs(y - g[lo].to_numpy()), np.abs(g[hi].to_numpy() - y)])
                fs.plot_ordinal(ax, i, g.mc.to_numpy(), y, yerr=err,
                                label=f"{L} bp", n=len(lengths))
            fs.finish(ax, xlabel="posterior samples $N$" if row == 1 else "",
                      ylabel=ylab if col == 0 else "",
                      title=title if row == 0 else "",
                      logx=True, logx_base=2, include_zero=True, legend=False)
            if row == 1:
                ax.set_ylim(0, 1)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(lengths), frameon=False,
               fontsize=fs.LEGEND_PT, bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fs.save(fig, "rerun_dna_fig5_gradient_mc", paper_item="Figure 5 / Tables 14-15",
            sources=[_rel(p) for _, p in frames.values()],
            script="notebooks/figures/build.py fig5",
            caption=("Accuracy of the finite-sample value gradient against the number of "
                     "posterior samples, for three rewards and four sequence lengths. x is "
                     "logarithmic (base 2); both y axes are linear and anchored at zero. "
                     "Error bars are the 95% CI over 8 noise pools x 32 states carried in the "
                     "results CSVs. Both the estimator and the reference here are GLASS; the "
                     "dMFM version is still queued."),
            notes=["x LOG base 2; y LINEAR from zero",
                   "95% CIs are drawn -- the previous version of this figure omitted them "
                   "although the columns existed",
                   "estimator AND reference are both GLASS (see docs/STATE_OF_CLAIMS.md)"])

    # --- diagnostic: the preliminary 4-state dMFM slice, explicitly not a result
    prelim = RESULTS / "dna" / "scale_free_mc" / "dmfm" / "preliminary" / "L50_motif_4states_cells.csv"
    if prelim.exists():
        c = pd.read_csv(prelim).sort_values("mc")
        fig, axes = fs.new_figure("wide", 1, 2)
        for ax, (a, b, ylab) in zip(axes, [
                ("dmfm_relative_l1", "published_glass_relative_l1", r"relative $L_1$ error"),
                ("dmfm_cosine", "published_glass_cosine", "mean cosine similarity")]):
            fs.plot_series(ax, "dmfm", c.mc, c[a], yerr=None,
                           yerr_absent="4-state slice, no repeats recorded", label="dMFM")
            fs.plot_series(ax, "sde_mc", c.mc, c[b], yerr=None,
                           yerr_absent="4-state slice, no repeats recorded",
                           label="GLASS (published cells, same 4 states)")
            fs.finish(ax, xlabel="posterior samples $N$", ylabel=ylab, logx=True,
                      logx_base=2, include_zero=True)
        fig.suptitle("PRELIMINARY 4-of-32-state CPU slice -- explicitly not the paper result",
                     fontsize=fs.TITLE_PT, color=fs.RED)
        fig.tight_layout()
        fs.save(fig, "dna_fig5_dmfm_preliminary", diagnostic=True,
                paper_item="Figure 5, preliminary dMFM slice",
                sources=[_rel(prelim)], script="notebooks/figures/build.py fig5",
                caption=("4 of 32 states, motif reward, L=50 only, fp32 CPU against a TF32-GPU "
                         "reference. results/dna/scale_free_mc/dmfm/preliminary/README.md says "
                         "in as many words: do not put these in the paper."))


# =============================================================================== Fig 1 (toy)
@builder("fig1")
def fig1_toy() -> None:
    """Paper Figure 1, the toy four-panel.

    Deliberately NOT restyled. `src/toy/figure.py` reproduces the geometry and
    typography measured off the published `figures/norway.pdf`; imposing the
    house style on it would throw away a recovery another agent did on purpose.

    It is re-emitted rather than copied for one reason: the copies in
    `results/toy/` embed **Type 3 fonts**, which arXiv and most conference
    checkers reject and which rasterise badly. `figstyle.use_paper_style()` sets
    `pdf.fonttype = 42`, and `toy.figure.four_panel` inherits it, so re-running
    that same function here produces the same drawing with subsetted TrueType.
    The producer script is not modified; see the report.
    """
    npz = RESULTS / "toy" / "fig1_panels.npz"
    if not npz.exists():
        print("fig1: results/toy/fig1_panels.npz missing; skipping")
        return
    import matplotlib.pyplot as plt
    from toy.figure import four_panel
    from toy.reward_smiley import SMILEY_SHAPE

    z = np.load(npz, allow_pickle=True)
    variants = [("mfm", "rerun_fig1_toy_multimfm", "Figure 1 (multiMFM-guided)"),
                ("glass", "rerun_fig1_toy_glass", "Figure 1 (GLASS-guided variant)")]
    for backend, out, item in variants:
        smiley_key = ("smiley_mfm_panel" if backend == "mfm" and "smiley_mfm_panel" in z
                      else f"smiley_{backend}")
        if f"norway_{backend}" not in z or smiley_key not in z:
            print(f"fig1: {backend} panels missing from the npz; skipping")
            continue
        fig = four_panel({"prior": z["prior"], "base": z["base"],
                          "norway": z[f"norway_{backend}"], "smiley": z[smiley_key]},
                         smiley_score_kwargs=SMILEY_SHAPE)
        fs.save(fig, out, paper_item=item, sources=[_rel(npz)],
                script="notebooks/figures/build.py fig1",
                caption=("Prior, base, Norway-reward and Smiley-reward panels of the toy "
                         "multimodal problem, 10,000 endpoints each. Scatter panels; no "
                         "axis scales, so nothing here can be truncated or log-compressed."),
                notes=["kept in the recovered published layout, NOT the house style",
                       "full text width; the PDF is 10.3 in wide, so include it with an "
                       "explicit width= -- its text is 25 pt and survives the reduction",
                       "the submitted Fig 1 PDF predates the toy MFM, so the published "
                       "figure may show plain GLASS (docs/REPRODUCIBILITY_PLAN.md 7.6)"],
                dpi=220)
        plt.close(fig)


# =============================================================================== moves
DIAGNOSTIC_MOVES = {
    # existing file in notebooks/figures      -> name under diagnostics/
    "07_qm9_multimfm_search_grid.png":        "qm9_multimfm_search_grid.png",
    "07_qm9_steering_nfe.png":                "superseded_qm9_steering_nfe.png",
    "07_qm9_steering_nfe.pdf":                "superseded_qm9_steering_nfe.pdf",
    "07_qm9_steering_nfe_fwd_equiv.png":      "superseded_qm9_steering_nfe_fwd_equiv.png",
    "07_qm9_steering_nfe_fwd_equiv.pdf":      "superseded_qm9_steering_nfe_fwd_equiv.pdf",
    "07_qm9_steering_nfe_matched.png":        "UNSAFE_qm9_steering_nfe_matched_dps_artifact.png",
    "dna_fig6_analog_parent_disjoint.png":    "dna_fig6_analog_parent_disjoint.png",
    "dna_fig6_from_paper_pdf.png":            "published_dna_fig6_render.png",
    "dna_fig7_analog_n1000.png":              "dna_fig7_analog_n1000.png",
    "toy_fig1_unguided.png":                  "toy_fig1_unguided.png",
    "toy_fig1_glass.png":                     "toy_fig1_glass_titled.png",
    "toy_fig1_mfm.png":                       "toy_fig1_mfm_titled.png",
}
# Superseded by a `build.py` output at the top level; the old file is dropped so
# nobody can pick the wrong one out of the directory.
SUPERSEDED = [
    "dna_fig4_left.png", "dna_fig4_right.png",            # -> published_dna_fig4_*
    "dna_fig4_left_value_error.png",                      # -> rerun_dna_fig4_left_value_error
    "dna_fig4_right_nfe_benchmark.png",                   # -> rerun_dna_fig4_right_nfe
    "dna_fig4_left_robustness.png",                       # -> diagnostics/
    "dna_fig4_right_submitted_vs_rerun.png",              # -> diagnostics/
    "dna_fig5_scale_free_mc.png",                         # -> rerun_dna_fig5_gradient_mc
    "dna_fig5_scale_free_mc_dmfm.png",                    # -> diagnostics/
]


@builder("tidy")
def tidy() -> None:
    """Move analysis-only plots to diagnostics/ and drop what build.py replaces."""
    DIAG.mkdir(parents=True, exist_ok=True)
    for src, dst in DIAGNOSTIC_MOVES.items():
        p = FIGS / src
        if p.exists():
            shutil.move(str(p), str(DIAG / dst))
            print(f"moved  {src} -> diagnostics/{dst}")
    for src in SUPERSEDED:
        p = FIGS / src
        if p.exists():
            p.unlink()
            print(f"removed {src} (superseded by a build.py output)")


# =============================================================================== README
README_HEAD = """\
# Paper figures

Everything at this level is a **paper figure**. Everything in `diagnostics/` is
analysis: useful, but not for the submission.

Rebuild all of it, on CPU, in seconds:

    ~/micromamba/envs/multimfm/bin/python notebooks/figures/build.py

Every figure is drawn through `src/multimfm/figstyle.py`, which holds the one
palette, marker set, type scale, figure size and error-bar convention the paper
uses. Each is written as **PNG (200 dpi)** and **PDF (vector)**.

## The file-name prefix says where the numbers came from

| prefix | means |
|---|---|
| `published_` | the values in the submitted PDF, digitized from its vector figures. **Not** a measurement of the code in this repository. |
| `rerun_` | our re-measurement of a published experiment, on the current models. |
| `new_` | an experiment with no counterpart in the submission. |

A `published_` and a `rerun_` file of the same panel show **different numbers on
different models**. Read the prefix before putting one in the paper.

## Paper figures
"""


PENDING = """\
## Paper items with no figure here yet

These are not missing by accident and no placeholder is drawn for them, because a
placeholder in the paper set would be worse than an absence. Their runs are
queue-bound; see `docs/RERUN_STATUS.md` for the job and the collect command.

| paper item | what it needs | job |
|---|---|---|
| Figure 3 (DNA C0 guidance) | the n=1000 dMFM Table-1 shards | see `RERUN_STATUS.md` |
| Figure 5 with dMFM | the dMFM scale-free array; only the GLASS version is drawn above, and only a 4-state CPU slice of the dMFM one exists (in `diagnostics/`) | 49298891 / 49297229 |
| Figure 6 (base marginals) + Table 16 | the CPU marginals array | 49294651 |
| Figure 7 (tilt vs guidance strength) | 5 of 14 sweep points, incl. dMFM `gf` 8 | 49298146 |
| Figure 8 (single guided trace) | the dMFM trace | 49298115 |

The notebooks write their own figures into this directory as they execute
(`savefig` in `notebooks/_common.py` targets the top level), so files may appear
here that this README does not list. **Anything at the top level that is not in
the table above and not in `_manifest.json` is notebook output, not a vetted
paper figure.**
"""


@builder("readme")
def write_readme() -> None:
    man = fs.read_manifest()
    paper = {k: v for k, v in man.items() if v.get("kind") != "diagnostic"}
    diag = {k: v for k, v in man.items() if v.get("kind") == "diagnostic"}

    def table(items: dict) -> str:
        rows = ["| file | paper item | rebuilt by | built from |",
                "|---|---|---|---|"]
        for name in sorted(items):
            v = items[name]
            exts = sorted({Path(f).suffix.lstrip(".") for f in v.get("files", [])})
            srcs = "<br>".join(f"`{s}`" for s in v.get("sources", [])) or "--"
            rows.append(f"| `{name}.{{{','.join(exts)}}}` | {v.get('paper_item') or '--'} "
                        f"| `{v.get('script') or '--'}` | {srcs} |")
        return "\n".join(rows)

    parts = [README_HEAD, table(paper), "", PENDING, "",
             "### What each one says, and what it does not", ""]
    for name in sorted(paper):
        v = paper[name]
        parts.append(f"**`{name}`** -- {v.get('paper_item') or ''}\n")
        if v.get("caption"):
            parts.append(v["caption"] + "\n")
        seen = set()
        for n in list(v.get("notes", [])) + list(v.get("log_axes", [])):
            # several panels of one figure report the same axis scales; say it once
            n = n.split(": ", 1)[-1] if n.startswith("axes[") else n
            if n not in seen:
                seen.add(n)
                parts.append(f"- {n}")
        parts.append("")

    parts += ["## Diagnostics (`diagnostics/`) -- not for the paper", "", table(diag), ""]
    for name in sorted(diag):
        v = diag[name]
        if v.get("caption"):
            parts.append(f"**`{name}`** -- {v['caption']}\n")

    parts += [
        "",
        "### Diagnostics with no manifest entry",
        "",
        "These were written by the notebooks before the shared style existed and are kept",
        "as-is. They are analysis, not paper figures.",
        "",
        "| file | what it is |",
        "|---|---|",
        "| `qm9_multimfm_search_grid.png` | the multiMFM hyperparameter grid behind Table 2; scratch. |",
        "| `superseded_qm9_steering_nfe*.{png,pdf}` | earlier QM9 steering plots, each method on its own budget ladder. Superseded by `new_qm9_steering_nfe`. |",
        "| `UNSAFE_qm9_steering_nfe_matched_dps_artifact.png` | **do not publish.** Its DPS curve degrades with compute, which is our sampler delivering 1/N of the guidance dose, not a property of DPS. Dose-matched rerun: Slurm job 49455395. |",
        "| `dna_fig6_analog_parent_disjoint.png` | Fig 6 analog on the parent-disjoint model; its own title says it is not the paper figure. |",
        "| `published_dna_fig6_render.png` | a raster render of the submitted Fig 6. |",
        "| `dna_fig7_analog_n1000.png` | Fig 7 analog from surviving data; different target and model. |",
        "| `toy_fig1_unguided.png`, `toy_fig1_*_titled.png` | Fig 1 panels with an explanatory super-title added; the paper versions are `rerun_fig1_toy_*`. |",
        "",
        "---",
        "",
        "Provenance for every generated figure, machine-readable: `_manifest.json`.",
        "Style rules and the colour-blindness audit of the palette: `docs/figures.md`.",
    ]
    (FIGS / "README.md").write_text("\n".join(parts) + "\n")
    print(f"wrote {FIGS / 'README.md'}")



def check_unvetted() -> None:
    """Warn about top-level images this build did not produce.

    Notebooks 01, 05 and 06 call `savefig` from `notebooks/_common.py`, which writes
    into the top level of this directory -- the vetted paper set. Executing them
    drops un-prefixed reruns next to the paper files, which is exactly the
    confusion the naming convention exists to prevent. Until `_common.savefig`
    routes notebook output elsewhere (see docs/figures.md 8), this prints the list
    so it is at least visible.
    """
    known = set(fs.read_manifest())
    stray = sorted(p.name for p in FIGS.iterdir()
                   if p.suffix in (".png", ".pdf") and p.stem not in known)
    if stray:
        print("\nWARNING: top-level images this build did not produce, so they are NOT")
        print("vetted paper figures (most likely notebook output; see docs/figures.md 8):")
        for n in stray:
            print(f"  {n}")


# =============================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    order = ["tidy", "fig4", "fig4l", "fig4r", "published4", "qm9", "fig5", "fig1"]
    if a.list:
        for k in order:
            print(f"{k:<12} {(BUILDERS[k].__doc__ or '').strip().splitlines()[0]}")
        return

    fs.use_paper_style()
    todo = a.which or order
    for k in todo:
        if k not in BUILDERS:
            sys.exit(f"unknown figure {k!r}; --list to see them")
        print(f"\n=== {k} ===")
        BUILDERS[k]()
    if "readme" not in todo:
        write_readme()
    check_unvetted()


if __name__ == "__main__":
    main()
