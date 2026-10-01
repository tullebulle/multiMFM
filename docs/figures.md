# Figure style contract

Owner: figures agent, 2026-09-30.
Code: `src/multimfm/figstyle.py`. Builder: `notebooks/figures/build.py`.
File list and per-figure captions: `notebooks/figures/README.md`.

Before this, the paper's figures were written by five agents in five styles — three type
scales, four figure sizes, two palettes, markers that meant one method in one panel and a
different method in the next, and error bars drawn in some panels but not others *even
where the CSV carried them*. A reader comparing two panels was comparing the plotting as
much as the results. This document is the contract that stops that.

---

## 1. Rebuild

    ~/micromamba/envs/multimfm/bin/python notebooks/figures/build.py         # all
    ~/micromamba/envs/multimfm/bin/python notebooks/figures/build.py fig4    # one
    ~/micromamba/envs/multimfm/bin/python notebooks/figures/build.py --list

CPU only, seconds, no checkpoints. Everything comes out of `results/`.

## 2. Naming, which is load-bearing

| prefix | means |
|---|---|
| `published_` | values in the submitted PDF, digitized from its vector figures. **Not** a measurement of this repository's code. |
| `rerun_` | our re-measurement of a published experiment, on the current models. |
| `new_` | an experiment with no counterpart in the submission. |

This replaces the previous naming, under which `dna_fig4_left.png` was the *digitized
published* panel and `dna_fig4_left_value_error.png` was the *rerun* — two files one word
apart, showing different numbers from different models, with nothing in either name to say
which was which. Both old names are gone; nothing in the repository still refers to them.

Diagnostics live in `notebooks/figures/diagnostics/` and keep plain descriptive names. The
top level is the paper set.

## 3. Output

Every figure is written as **PNG at 200 dpi** and **PDF (vector)**. `pdf.fonttype = 42`, so
fonts embed as subsetted CID TrueType; Type 3 fonts are rejected by arXiv and by most
conference checkers. (`results/toy/fig1_*.pdf` embedded Type 3 — see §8.)

**Figures are emitted at final print size and must be included unscaled.** A 4-inch figure
squeezed into a 3.35-inch column puts every 7 pt label at 5.9 pt. Sizes are in
`figstyle.SIZES`:

| key | inches | for |
|---|---|---|
| `col` | 3.35 × 3.35 | one column, legend under the axes |
| `panel` / `panel_tall` | 3.35 × 2.70 / 3.20 | one column, legend inside |
| `pair` | 6.85 × 3.35 | full text width, two column-width panels |
| `wide` / `wide_tall` | 6.85 × 2.85 / 4.10 | full text width, one row |
| `grid23` | 6.85 × 4.60 | 2 × 3 small multiples |

Type scale: body 8 pt, titles 8.5, axis labels 8, ticks 7, legend 7, annotations 7.
**7 pt is the floor** and nothing is drawn below it.

## 4. Palette, and the colour-blindness audit

Five hues, kept from the submitted Figure 4 (matplotlib `tab10`) so that readers of the PDF
recognise them:

| slot | hex | used for |
|---|---|---|
| headline | `#d62728` | dMFM, multiMFM-SS — always red, always the heaviest line |
| | `#1f77b4` | best-of-$N$ |
| | `#ff7f0e` | Feynman–Kac |
| | `#2ca02c` | beam search, FMAP |
| | `#9467bd` | MCTS, DPS |
| aux | `#8c564b` | multiMFM (the un-searched variant) |
| aux | `#7f7f7f` | references, floors, "not a method" |

Checked with the `dataviz` skill's `validate_palette.js` (OKLab ΔE × 100, light surface,
**all pairs**). It does **not** pass:

```
pair            worst CVD      normal vision
red/blue        protan 21.1    31.7   ok
red/orange      deutan 16.8    18.6   ok
red/green       deutan  3.9    33.8   FAIL (deuteranopia)
red/purple      deutan 21.3    23.0   ok
blue/orange     protan 24.6    35.7   ok
blue/green      tritan  6.8    25.2   weak (tritanopia)
blue/purple     protan  1.7    14.2   FAIL, and below the 15 normal-vision floor
orange/green    protan  0.7    28.6   FAIL (protanopia) — the worst pair
orange/purple   protan 26.2    28.9   ok
green/purple    tritan 15.3    31.6   ok
```

Also: `#ff7f0e` is 2.47:1 against a white surface, below the 3:1 relief threshold.

**This is not fixable by re-stepping.** No five-hue categorical palette containing both a
red and a green passes an all-pairs CVD check; the confusion lines are the whole point.
Substituting a darker green moves the failure from orange/green to red/green, and
substituting the purple moves it to blue/purple. Dropping red or green would change the
headline colour two published panels already use.

**So the palette is kept and the relief the validator asks for is made mandatory instead.**
`figstyle.SERIES` gives every method its own **marker shape** *and* its own **dash
pattern**, and `plot_series` always draws both — a caller cannot opt out. The three failing
pairs are:

| failing pair | methods | separated by |
|---|---|---|
| orange/green | Feynman–Kac / beam search | ◆ dash-dot-dot vs ▲ dash-dot-dot-dot, different dash periods |
| blue/purple | best-of-$N$ / MCTS | ■ long dash vs ✚ fine dot |
| red/green | dMFM / beam search | ● solid at lw 2.2 vs ▲ dashed at lw 1.5, plus direct labels on the red curve |

Two colours can never collide inside one panel: `plot_series` records the colours already
used on an axis and raises if a second series asks for one. (DPS and MCTS share purple, and
FMAP and beam search share green, because they never appear together; the check is what
guarantees that rather than the author's memory.)

Ordered families — sequence length, budget — use `figstyle.ORDINAL`, a single-hue blue ramp
`#74a9cf #4292c6 #2171b5 #084594` that **passes** `validate_palette.js --ordinal`:
monotone lightness, adjacent ΔL ≥ 0.06, light end 2.46:1 against the surface, hue spread 18°.

## 5. What the module refuses to draw

These are enforced in code, not left to discipline.

* **No silent missing error bars.** `plot_series` raises unless it is given `yerr`, or given
  `yerr_absent="<reason>"`, which is recorded in the manifest. Several of this paper's
  comparisons are ties within one SEM and must *look* like ties. The published Figure 4
  right panel's four baseline curves carry `mae_sem = 0.000000` in every row because their
  producing code was never in the repository — so `published_dna_fig4_right_nfe` states
  that rather than drawing a clean line and letting the reader assume precision.
* **No truncated linear axis.** `finish(include_zero=True)` is the default and anchors a
  linear y-axis of a non-negative error metric at zero, so a small gap cannot be inflated by
  cropping the baseline away.
* **Log axes are declared.** `save` records each axis's scale in `_manifest.json` and the
  README prints it, so the caption can say so. Where a log axis was chosen over a linear
  one, the reason is in the caption: on the QM9 panel the y range is 0.82–8.1 Bohr³ and a
  linear axis compresses the crossing region into the bottom eighth of the panel.
* **No cropped measured range.** Where a method was measured further than it is flattering
  to show it, it is shown. See §6.
* **Placeholders, not guesses.** `figstyle.placeholder` draws a hatched red-bordered "DATA
  PENDING" panel. No figure in the paper set currently uses one; items with no data are
  listed as absent in the README instead, because a placeholder inside the paper set is
  worse than an absence.

## 6. Three specific things a reader could otherwise misread

1. **DNA Figure 4 right is drawn on the measured forward-equivalent axis**, not the
   published one. `docs/nfe_accounting_audit.md` measured the backward on this code path at
   1.1 forwards (H200), and every generative forward the dMFM value estimator makes is
   inside a differentiated graph, so its curve moves right while the pure-forward baselines
   do not. The crossover with Feynman–Kac is **430** rather than the published axis's 582,
   and both sides of it are inside the plotted range and labelled on the panel
   ("dMFM leads" / "Feynman–Kac leads"). The published-axis version is kept side by side in
   `diagnostics/dna_fig4_right_axis_convention` so the shift is auditable.
   *This is a presentation decision on top of another agent's measurement; the authors
   should confirm which axis the paper carries.*
2. **The QM9 panel is not trimmed at 10,000 NFE.** best-of-$N$'s measured ladder runs to
   N = 512, where it reaches 0.819 ± 0.048 against multiMFM-SS's 1.006 ± 0.045 — it wins at
   the top of the range. Earlier versions of this plot cut the axis just short of that.
3. **The QM9 DPS `sample_steps` extension is withheld.** That sweep shows DPS degrading with
   more compute, which `docs/STATE_OF_CLAIMS.md` traces to our sampler delivering 1/N of the
   guidance dose as N grows — our handicap, not a property of DPS. Only DPS's `guide_every`
   ladder is drawn, unjoined, with the dose confound stated on the panel and the
   dose-matched rerun (Slurm job 49455395) named. The old unsafe file is preserved as
   `diagnostics/UNSAFE_qm9_steering_nfe_matched_dps_artifact.png`.

## 7. Adding a figure

1. `fs.use_paper_style()`, then `fs.new_figure(<size key>)`.
2. One `fs.plot_series(ax, <registered key>, x, y, yerr=...)` per method. If the method is
   not in `figstyle.SERIES`, add it there — do not pass an ad-hoc colour.
3. `fs.finish(ax, xlabel=..., ylabel=..., logx=..., logy=...)`.
4. `fs.save(fig, "<prefix>_<name>", paper_item=..., sources=[...], script=...,
   caption=..., notes=[...])`. `sources` is what makes the README honest; fill it in.
5. Rerun `build.py readme`.

## 8. Changes other agents' files need (not made here)

* **`src/toy/figure.py`** sets `RC_PARAMS` without `pdf.fonttype`, so
  `results/toy/fig1_*.pdf` embed **Type 3** fonts (confirmed with `pdffonts`). `build.py
  fig1` works around it by re-emitting the same `four_panel` call under the paper rcParams,
  but `python -m toy.fig1` still writes Type 3 into `results/toy/`. The fix is one line in
  `RC_PARAMS`: `"pdf.fonttype": 42`.
* **`notebooks/_common.py`'s `savefig`** writes into the top level of
  `notebooks/figures/`, which is now the vetted paper set. Notebook figures should go to
  `notebooks/figures/diagnostics/` (or carry a prefix) so an executed notebook cannot drop
  an unvetted PNG next to the paper files. Suggested: `savefig(fig, name, paper=False)`
  defaulting to the `diagnostics/` subdirectory.
* **`scripts/sync_paper_figures.sh`** copies into `figures/` from paths under `results/`
  that this build no longer writes. Its list should point at `notebooks/figures/*.pdf`.
* **`scripts/qm9/plot_steering_nfe_matched.py`** and **`src/multimfm/qm9_figures.py`** hold
  their own copies of the palette and type scale, and the former produces the unsafe DPS
  panel. Both should import `multimfm.figstyle`; `plot_steering_nfe_matched.py` is
  superseded by `build.py qm9`.
* **`scripts/dna/fig4_build_notebook.py`** embeds `PANEL_A_STYLE` / `PANEL_B_STYLE` in the
  notebook cells it generates. Those should become
  `from multimfm import figstyle as fs` plus `fs.plot_series(...)`, so the notebook and the
  paper figure cannot drift apart.
