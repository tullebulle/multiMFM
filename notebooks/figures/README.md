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

| file | paper item | rebuilt by | built from |
|---|---|---|---|
| `new_qm9_steering_nfe.{pdf,png}` | QM9 steering comparison (new) | `notebooks/figures/build.py qm9` | `results/shared/nfe_audit/qm9_axes_untrimmed.csv` |
| `published_dna_fig4_left_value_error.{pdf,png}` | Figure 4 (left) as submitted | `notebooks/figures/build.py published4` | `results/dna/figure_digitized/fig4_left_MC_value_digitized.csv` |
| `published_dna_fig4_right_nfe.{pdf,png}` | Figure 4 (right) as submitted | `notebooks/figures/build.py published4` | `results/dna/figure_digitized/fig4_right_nfe_benchmark_digitized.csv` |
| `rerun_dna_fig4.{pdf,png}` | Figure 4 (both panels) | `notebooks/figures/build.py fig4` | `results/shared/nfe_audit/dna_axes.csv`<br>`results/dna/fig4/panelA_value_error.csv`<br>`results/dna/fig4/summary.json` |
| `rerun_dna_fig4_left_value_error.{pdf,png}` | Figure 4 (left) | `notebooks/figures/build.py fig4l` | `results/shared/nfe_audit/dna_axes.csv`<br>`results/dna/fig4/panelA_value_error.csv`<br>`results/dna/fig4/summary.json` |
| `rerun_dna_fig4_right_nfe.{pdf,png}` | Figure 4 (right) | `notebooks/figures/build.py fig4r` | `results/shared/nfe_audit/dna_axes.csv`<br>`results/dna/fig4/panelA_value_error.csv`<br>`results/dna/fig4/summary.json` |
| `rerun_dna_fig5_gradient_mc.{pdf,png}` | Figure 5 / Tables 14-15 | `notebooks/figures/build.py fig5` | `results/dna/scale_free_mc/gc/gradient_scale_free_summary_all_lengths.csv`<br>`results/dna/scale_free_mc/motif/gradient_scale_free_summary_all_lengths.csv`<br>`results/dna/scale_free_mc/conjunction/gradient_scale_free_summary_all_lengths.csv` |
| `rerun_fig1_toy_glass.{pdf,png}` | Figure 1 (GLASS-guided variant) | `notebooks/figures/build.py fig1` | `results/toy/fig1_panels.npz` |
| `rerun_fig1_toy_multimfm.{pdf,png}` | Figure 1 (multiMFM-guided) | `notebooks/figures/build.py fig1` | `results/toy/fig1_panels.npz` |

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


### What each one says, and what it does not

**`new_qm9_steering_nfe`** -- QM9 steering comparison (new)

Oracle MAE on alpha against forward-equivalent compute, n = 100 molecules per seed, mean +- SE over seeds 1/2/3, same base model and target set for every method. The x axis counts generative forward passes plus twice the back-propagated ones, the measured cost of a guided step here. Both axes are logarithmic: the y range spans 0.82 to 8.1 Bohr^3 and a linear axis would compress the region where the methods cross into the bottom eighth of the panel. Every method is drawn over its full measured budget ladder; best-of-N overtakes multiMFM-SS at the top of that range.

- one column wide (3.35 in): include at width=\columnwidth, unscaled
- x LOG, y LOG -- state this in the caption
- best-of-N is NOT trimmed: it wins at N = 512
- DPS sample_steps extension withheld (1/N dose artifact, job 49455395)
- multiMFM-SS's advantage is partly bought off this axis: 2x the regressor calls and 4-13x the PoseBusters checks of best-of-N
- x log, y log

**`published_dna_fig4_left_value_error`** -- Figure 4 (left) as submitted

The submitted Figure 4 (left), re-drawn from values digitized off the vector PDF (affine residual ~2e-5 pt). Both axes logarithmic. The published panel carries no error bars, so none are drawn; it also omits the SDE Monte-Carlo floor series that bounds what the comparison can resolve.

- PUBLISHED VALUES, not a rerun
- no error bars exist in the source figure
- x LOG base 2, y LOG
- x log, y log

**`published_dna_fig4_right_nfe`** -- Figure 4 (right) as submitted

The submitted Figure 4 (right), re-drawn from values digitized off the vector PDF. x is logarithmic, y is linear from zero. The four baseline curves came from a CSV hard-coded in the benchmark script whose producing code was never in the repository and whose mae_sem is 0.000000 in every row, so no error bars can honestly be drawn for them.

- PUBLISHED VALUES, not a rerun
- baseline error bars do not exist in the source
- compute axis bills a back-propagated forward at zero
- x log, y linear

**`rerun_dna_fig4`** -- Figure 4 (both panels)

Left: Absolute error of the value estimate against the number of posterior samples, over 32 conditioning states. Both axes are logarithmic (x base 2). Error bars are the standard error over conditions. The shaded band is the reference's own Monte-Carlo spread: no curve can be resolved inside it, so the N >= 256 points sit at or below the noise floor of the comparison and must not be read as resolved. Right: Target error against forward-equivalent compute for five inference-time steering methods on the parent-disjoint L=50 DNA flow. The x axis counts generative forward passes plus 1.1 back-propagated forward passes per returned sample (the 1.1 is measured on an H200 for this code path) and is logarithmic; y is linear and anchored at zero. Error bars are the standard error over 3 repeats of 100 outputs. dMFM leads below 430 forward-equivalent NFE and Feynman-Kac above it; both regions are inside the plotted range.

- full text width (6.85 in): include at width=\textwidth, unscaled
- x axis LOG (base 2), y axis LOG
- the shaded band is the comparison's own noise floor, not a result
- the SDE MC floor series is omitted from the published panel
- x axis is LOG; y axis is LINEAR and includes zero
- k = 1.1 is measured, not assumed (docs/nfe_accounting_audit.md)
- on the published axis (backward billed at zero) the crossover is 582 instead, and dMFM appears to win at two settings rather than one
- x log, y log
- x log, y linear

**`rerun_dna_fig4_left_value_error`** -- Figure 4 (left)

Absolute error of the value estimate against the number of posterior samples, over 32 conditioning states. Both axes are logarithmic (x base 2). Error bars are the standard error over conditions. The shaded band is the reference's own Monte-Carlo spread: no curve can be resolved inside it, so the N >= 256 points sit at or below the noise floor of the comparison and must not be read as resolved.

- one column wide (3.35 in): include at width=\columnwidth, unscaled
- x axis LOG (base 2), y axis LOG
- the shaded band is the comparison's own noise floor, not a result
- the SDE MC floor series is omitted from the published panel
- x log, y log

**`rerun_dna_fig4_right_nfe`** -- Figure 4 (right)

Target error against forward-equivalent compute for five inference-time steering methods on the parent-disjoint L=50 DNA flow. The x axis counts generative forward passes plus 1.1 back-propagated forward passes per returned sample (the 1.1 is measured on an H200 for this code path) and is logarithmic; y is linear and anchored at zero. Error bars are the standard error over 3 repeats of 100 outputs. dMFM leads below 430 forward-equivalent NFE and Feynman-Kac above it; both regions are inside the plotted range.

- one column wide (3.35 in): include at width=\columnwidth, unscaled
- x axis is LOG; y axis is LINEAR and includes zero
- k = 1.1 is measured, not assumed (docs/nfe_accounting_audit.md)
- on the published axis (backward billed at zero) the crossover is 582 instead, and dMFM appears to win at two settings rather than one
- x log, y linear

**`rerun_dna_fig5_gradient_mc`** -- Figure 5 / Tables 14-15

Accuracy of the finite-sample value gradient against the number of posterior samples, for three rewards and four sequence lengths. x is logarithmic (base 2); both y axes are linear and anchored at zero. Error bars are the 95% CI over 8 noise pools x 32 states carried in the results CSVs. Both the estimator and the reference here are GLASS; the dMFM version is still queued.

- x LOG base 2; y LINEAR from zero
- 95% CIs are drawn -- the previous version of this figure omitted them although the columns existed
- estimator AND reference are both GLASS (see docs/STATE_OF_CLAIMS.md)
- x log, y linear

**`rerun_fig1_toy_glass`** -- Figure 1 (GLASS-guided variant)

Prior, base, Norway-reward and Smiley-reward panels of the toy multimodal problem, 10,000 endpoints each. Scatter panels; no axis scales, so nothing here can be truncated or log-compressed.

- kept in the recovered published layout, NOT the house style
- full text width; the PDF is 10.3 in wide, so include it with an explicit width= -- its text is 25 pt and survives the reduction
- the submitted Fig 1 PDF predates the toy MFM, so the published figure may show plain GLASS (docs/REPRODUCIBILITY_PLAN.md 7.6)

**`rerun_fig1_toy_multimfm`** -- Figure 1 (multiMFM-guided)

Prior, base, Norway-reward and Smiley-reward panels of the toy multimodal problem, 10,000 endpoints each. Scatter panels; no axis scales, so nothing here can be truncated or log-compressed.

- kept in the recovered published layout, NOT the house style
- full text width; the PDF is 10.3 in wide, so include it with an explicit width= -- its text is 25 pt and survives the reduction
- the submitted Fig 1 PDF predates the toy MFM, so the published figure may show plain GLASS (docs/REPRODUCIBILITY_PLAN.md 7.6)

## Diagnostics (`diagnostics/`) -- not for the paper

| file | paper item | rebuilt by | built from |
|---|---|---|---|
| `dna_fig4_left_robustness.{pdf,png}` | Figure 4 (left), robustness variant | `notebooks/figures/build.py fig4l` | `results/dna/fig4/panelA_value_error_c0_reward.csv` |
| `dna_fig4_right_axis_convention.{pdf,png}` | Figure 4 (right), accounting sensitivity | `notebooks/figures/build.py fig4r` | `results/shared/nfe_audit/dna_axes.csv` |
| `dna_fig4_right_submitted_vs_rerun.{pdf,png}` | Figure 4 (right), submitted vs rerun | `notebooks/figures/build.py published4` | `results/dna/figure_digitized/fig4_right_nfe_benchmark_digitized.csv`<br>`results/dna/fig4/panelB_curves.csv` |
| `dna_fig5_dmfm_preliminary.{pdf,png}` | Figure 5, preliminary dMFM slice | `notebooks/figures/build.py fig5` | `results/dna/scale_free_mc/dmfm/preliminary/L50_motif_4states_cells.csv` |

**`dna_fig4_left_robustness`** -- Panel A's protocol with the right panel's reward and a GLASS reference.

**`dna_fig4_right_axis_convention`** -- The same measurements on the published and the measured compute axis.

**`dna_fig4_right_submitted_vs_rerun`** -- Different models, different guide and really-implemented baselines, so the two sets are not numerically comparable; the overlay only shows what changed.

**`dna_fig5_dmfm_preliminary`** -- 4 of 32 states, motif reward, L=50 only, fp32 CPU against a TF32-GPU reference. results/dna/scale_free_mc/dmfm/preliminary/README.md says in as many words: do not put these in the paper.


### Diagnostics with no manifest entry

These were written by the notebooks before the shared style existed and are kept
as-is. They are analysis, not paper figures.

| file | what it is |
|---|---|
| `qm9_multimfm_search_grid.png` | the multiMFM hyperparameter grid behind Table 2; scratch. |
| `superseded_qm9_steering_nfe*.{png,pdf}` | earlier QM9 steering plots, each method on its own budget ladder. Superseded by `new_qm9_steering_nfe`. |
| `UNSAFE_qm9_steering_nfe_matched_dps_artifact.png` | **do not publish.** Its DPS curve degrades with compute, which is our sampler delivering 1/N of the guidance dose, not a property of DPS. Dose-matched rerun: Slurm job 49455395. |
| `dna_fig6_analog_parent_disjoint.png` | Fig 6 analog on the parent-disjoint model; its own title says it is not the paper figure. |
| `published_dna_fig6_render.png` | a raster render of the submitted Fig 6. |
| `dna_fig7_analog_n1000.png` | Fig 7 analog from surviving data; different target and model. |
| `toy_fig1_unguided.png`, `toy_fig1_*_titled.png` | Fig 1 panels with an explanatory super-title added; the paper versions are `rerun_fig1_toy_*`. |

---

Provenance for every generated figure, machine-readable: `_manifest.json`.
Style rules and the colour-blindness audit of the palette: `docs/figures.md`.
