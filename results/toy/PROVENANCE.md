# results/toy — provenance

Paper items: **Fig. 1**, **Table 12**, App. E.2 / F.7.

Everything here was produced on 2026-09-29 from `src/toy/`, which is itself a rebuild of the purged
`norway` repo. `docs/provenance/toy/README.md` records, per module, what is verbatim original
source and what is a reconstruction; read it before treating any number here as "the paper's".

The paper prints **no** quantitative results for the toy — Fig. 1 is a qualitative figure and
Table 12 is a hyperparameter table. The reference values the rebuild is checked against therefore
come from the original session transcript
(`docs/provenance/toy/recovered/cursor_terminal_1.txt`), and are stored in `paper_values.json`
under `original_runs`.

## Files

| file | produced by | job | date |
|---|---|---|---|
| `probe_mfm_diagonal_10000.json` | `bash scripts/toy/train_mfm_diagonal.sh` (`python -m toy.probe_mfm_diagonal --ckpt checkpoints/toy/mfm_diagonal_10000.pt --teacher_ckpt checkpoints/toy/ckpt_uniform_color_square.pt --metrics_out results/toy/probe_mfm_diagonal_10000.json`) | 49256183 | 2026-09-29 |
| `probe_mfm_consistency_10000.json` | `bash scripts/toy/train_mfm_consistency.sh` (same probe on the ESD checkpoint) | 49256184 | 2026-09-29 |
| `fig1_panels.npz` | `bash scripts/toy/steer_fig1.sh` → `python -m toy.fig1 --backends glass mfm --n_panel 10000 --out_dir results/toy` | 49269134 | 2026-09-29 |
| `steering_metrics.json` | idem (configs, per-panel metrics, per-step diagnostics, checkpoint sha256, runtimes) | 49269134 | 2026-09-29 |
| `fig1_glass.png` | idem — Fig. 1 with GLASS-guided Norway / smiley panels (**the variant the submitted figure shows**) | 49269134 | 2026-09-29 |
| `fig1_mfm.png` | idem — Fig. 1 with multiMFM-guided panels | 49269134 | 2026-09-29 |
| `fig1_glass.pdf`, `fig1_mfm.pdf` | `python -m toy.fig1 --redraw --out_dir results/toy` (re-renders from the npz; the paper includes Fig. 1 as a PDF) | — | 2026-09-29 |
| `table12_actual.json` | `python -m toy.table12 --out results/toy/table12_actual.json` (reads the three checkpoints) | 49269134 | 2026-09-29 |
| `smiley_beta_calibration.json` | `bash scripts/toy/smiley_recal.sh` → `python -m toy.smiley_recal --out_dir results/toy` | 49285374 | 2026-09-29 |
| `fig1_published_reference.png` | `pdftoppm -r 110 -png /n/netscratch/kozinsky_lab/.../dMFM/figures/norway.pdf` — the submitted figure, for side-by-side comparison in the notebook | — | 2026-09-29 |
| `paper_values.json` | hand-written from `dMFM (40).pdf` Table 12 and from `cursor_terminal_1.txt` | — | 2026-09-29 |

`fig1_panels.npz` keys: `prior`, `base`, `norway_unguided`, `norway_glass`, `norway_mfm`,
`smiley_unguided`, `smiley_glass`, `smiley_mfm`, `smiley_mfm_panel` — each `(N, 5)` float32,
columns `[c_B, c_W, c_R, y1, y2]` (one-hot colour for sampled endpoints, raw Gaussian for `prior`).

## Slurm jobs

| job | partition | stages | result |
|---|---|---|---|
| 49256183 `toy_base_diag` | `kempner_requeue` | `train_base`, `train_mfm_diagonal` | COMPLETED, 36:00, holygpu8a19602 |
| 49256184 `toy_esd` | `kempner_requeue` | `train_mfm_consistency` | COMPLETED, 10:37, holygpu8a19602 |
| 49269134 `toy_fig1` | `gpu_test` | `steer_fig1` | COMPLETED, 1:24:54, holygpu7c26105 |
| 49285374 `toy_recal` | `gpu_test` | `smiley_recal` | **queued 2026-09-29 16:12**; ~95 min once it starts. It overwrites `smiley_beta_calibration.json` with the full-fidelity sweep and regenerates `smiley_glass` / `smiley_mfm_panel` in `fig1_panels.npz`. Until then those panels use β = 16 and the calibration file holds the reduced-fidelity CPU sweep. |

Submitted as
`sbatch --partition=<p> --job-name=<n> --export=ALL,STAGES="<stages>" scripts/toy/toy.sbatch`
from the repo root. Neither training job was preempted, so no requeue was needed.

## Rebuilding from scratch

```bash
source scripts/env.sh
sbatch --partition=kempner_requeue --requeue --job-name=toy_base_diag \
    --export=ALL,STAGES="train_base train_mfm_diagonal" scripts/toy/toy.sbatch
sbatch --partition=kempner_requeue --requeue --job-name=toy_esd \
    --dependency=afterok:<id> --export=ALL,STAGES=train_mfm_consistency scripts/toy/toy.sbatch
sbatch --partition=gpu_test --time=4:00:00 --job-name=toy_fig1 \
    --dependency=afterok:<id> --export=ALL,STAGES=steer_fig1 scripts/toy/toy.sbatch
```

Optional, after the panels exist — calibrate the smiley reward scale against the published figure and
regenerate the two smiley panels with it:

```bash
sbatch --partition=gpu_test --time=4:00:00 --job-name=toy_recal \
    --export=ALL,STAGES=smiley_recal scripts/toy/toy.sbatch
```

Collection (checks the outputs, re-renders the figures with the current plotting code and re-executes the
notebook). This is the single command to run after any of the jobs above finish:

```bash
bash scripts/toy/collect_fig1.sh
```

The three stages are small: ~26 min (base, 20k × 4096), ~9 min (diagonal, 10k × 2048), ~10 min
(ESD, 10k × 1024) on one A100. They are not checkpoint-resumable; on a preemptible partition a
preempted stage simply restarts, which is cheaper than adding resume logic for a 10-minute job.

## Checkpoints

`checkpoints/toy/` (gitignored, listed in `checkpoints/MANIFEST.json`, group `toy`):

| file | stage | settings |
|---|---|---|
| `ckpt_uniform_color_square.pt` | base flow | 20,000 steps, batch 4096, lr 3e-4, warmup 500, cosine, EMA 0.999, grad clip 1.0, seed 0 |
| `mfm_diagonal_10000.pt` | diagonal GLASS distillation | 10,000 steps, batch 2048, lr 3e-4, warmup 500, EMA 0.999, s/t_cond ∈ [1e-3, 0.95], c/y weights 1.0, seed 0 |
| `mfm_consistency_10000.pt` | ESD consistency | 10,000 steps, batch 1024, lr 1e-4, warmup 500, `gap_anneal_steps` 5000, EMA 0.995, seed 0 |

The base and diagonal commands were not recorded and use the Table 12 values / the recovered
defaults; the ESD command is verbatim from the original session
(`cursor_terminal_1.txt` lines 30–40).

## Agreement with the original runs

Reproduced here vs. what the original session logged (`paper_values.json` → `original_runs`).

### ESD stage (training + held-out diagonal probe)

| quantity | original | rebuild |
|---|---|---|
| final `gap_scale` / mean gap / max gap | 1.0 / 0.309 / 0.915 | 1.0 / 0.3094 / 0.9152 |
| final total loss | 0.00211 | 0.002271 |
| final diagonal GLASS loss | 0.000119 | 0.000089 |
| final ESD consistency loss | 0.001992 | 0.002182 |
| probe EMA diagonal loss vs GLASS | 3.85e-4 | 3.59e-4 |
| probe EMA relative L2 | 0.0141 | 0.01383 |
| probe EMA cosine | 0.99988 | 0.999894 |
| probe baseline teacher loss | 0.3796 | 0.37149 |
| runtime | 1084.8 s | 608.8 s (different GPU) |

### Guidance

| run | original | rebuild |
|---|---|---|
| Norway unguided, N=10000 | reward 0.324, match 0.334 | reward 0.3235, match 0.3322 |
| Norway multiMFM 1-step, ratio 6, N=10000 | reward 0.471, match 0.768 | reward 0.4721, match 0.7711 |
| Norway GLASS, ratio 2 | reward 0.407, match 0.573 (N=1000 smoke, window to 0.95) | reward 0.4586, match 0.7462 (N=10000, window to 0.98 as in the figure run — not a like-for-like comparison) |
| Smiley multiMFM, β=64, ratio 2, late window, N=1000 | reward 0.563, match 0.961, coverage 1.000 | reward 0.5769, match 0.9710, coverage 1.000 |
| Smiley GLASS ("bigger smile / thick mouth"), N=10000 | reward 0.555, match 0.959 | reward 0.5870, match 0.9887 (β = 16; see below) |

Base-model sanity: unguided colour fractions B/W/R = 0.34/0.33/0.33, fraction inside the unit square
0.998, `‖v_base,c‖` ≈ 7.1 after the colour unfreeze at t = 0.8 (the original notebook's saved log prints
7.211 / 7.223 at t = 0.899 / 0.989, reward 0.324, match 0.334 — the rebuilt base model reproduces all of
these to three decimals).

### Smiley concentration (`smiley_beta_calibration.json`)

The regenerated smiley features are **tighter** than the published ones, and score better than the original
logged run. Measuring both rasters identically (classify panel-4 pixels by colour inside the feature windows,
map to data coordinates), the published eyes have a horizontal pixel std of 0.113 and the regenerated ones
0.051 (GLASS) / 0.062 (multiMFM); the published feature centres (−0.286, +0.332) and (+0.312, +0.334) are
reproduced well, so the geometry is right and only the guidance strength is off.

The reward scale β of the final N = 10,000 GLASS smiley was never recorded — 16 comes from the file name of
its N = 1,000 predecessor. β enters the position guidance linearly (the colour guidance is ratio-normalised),
and under the exact tilt the stationary spread around a Gaussian feature of width σ is σ/√β, so β *is* the blob
width. `toy.smiley_recal` sweeps β against the published measurement and re-runs the two full-size smiley
panels with the selected value. Treat that value as **calibrated to the published figure**, not recovered.

## Fig. 1: which guidance does the submitted figure show? (authors' decision 7)

**GLASS, not multiMFM.** `figures/norway.pdf` renders identically to
`norway_base_smiley_four_panel_tweak.pdf`, the output of
`norway/norway_base_smiley_four_panel.ipynb`. Saves of that notebook at 2026-05-06 02:30–02:39 Z
still contain the cell output that prints the stored panel sources:

* Norway: `uniform_norway_slow_color_guidance_N10000_freeze_t08_ratio2_endpoints.pt`
* Smiley: `uniform_smiley_guidance_N1000_lam0p2_beta16_mc16_steps200_glass8_chunk100_endpoints.pt`
  (`glass8` = 8 GLASS Euler steps), replaced in the final 15:20 Z save — the one whose plot knobs
  match the published figure — by
  `uniform_smiley_delayed_color_bigger_smile_thick_mouth_N10000_endpoints.pt`, the same GLASS
  sampler at N = 10,000 with the larger smiley geometry.

All of those predate the toy MFM (earliest save of the MFM guidance notebook: 2026-05-11).
`fig1_glass.png` is therefore the faithful regeneration of the submitted figure; `fig1_mfm.png` is
the multiMFM version the caption describes. Full argument: `docs/provenance/toy/README.md`.

## Table 12 (plan §7.5)

Table 12 prints a single column but the toy trains in three stages. Its Optimization / Training
block matches the **base** stage only:

| row | Table 12 | base | diagonal | ESD |
|---|---|---|---|---|
| Batch size | 4096 | 4096 | 2048 | 1024 |
| Training steps | 20,000 | 20,000 | 10,000 | 10,000 |
| Peak learning rate | 3e-4 | 3e-4 | 3e-4 | 1e-4 |
| EMA decay | 0.999 | 0.999 | 0.999 | 0.995 |
| Warmup steps | 500 | 500 | 500 | 500 |
| (architecture rows) | as printed | match | match | match |

The `gap_anneal_steps` = 5000 curriculum of the ESD stage has no row at all. The "Total loss
L = Ly + Lc + L_ESD" row is also not a loss any single stage optimises: the base stage uses
Ly + Lc, the diagonal stage a GLASS-velocity regression, and the ESD stage L_diag + L_ESD
(weights 1.0 / 1.0).
