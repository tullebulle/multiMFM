# Toy example (paper Fig. 1, Table 12, App. E.2 / F.7) — provenance

The toy lived in `/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/norway`, which was purged
from netscratch. Nothing of it survives as a file. Everything in `src/toy/`, `scripts/toy/`
and `checkpoints/toy/` was rebuilt from transcripts and editor history in September 2026.
This file records, per module, what is **verbatim** original source, what is a
**reconstruction** against a recorded interface, and what is **new**.

## Sources used

| tag | location |
|---|---|
| `codex` | `~/.codex/sessions/2026/06/03/rollout-2026-06-03T13-08-02-019e8e74-d7f5-7962-9f77-c063607ddbdd.jsonl` — the June Codex session that read the norway files; file bodies appear in `exec_command` outputs |
| `term1` | `~/.cursor/projects/n-netscratch-kozinsky-lab-Lab-uunneberg-repos-norway/terminals/1.txt` — copy at `recovered/cursor_terminal_1.txt`. The June Codex conversation, with the exact training commands, the guidance settings and the run metrics |
| `hist` | `~/.cursor-server/data/User/History/` — VS Code / Cursor local file history. Copies of the recovered entries are under `recovered/cursor_history/` |
| `paper` | `dMFM (40).pdf`, App. E.2 (architecture), F.7.1–F.7.3 (rewards), Table 12 |

`recovered/` holds the raw extracted files, `recovered/cursor_history/` the notebooks and
notes pulled out of the editor history, and `recovered/norway_file_listing_2026-06-04.txt` /
`recovered/rg_hits_norway_2026-06-11.txt` the file listing and ripgrep hits that tell us
which files existed and what their CLI defaults were.

## Per-module status

| `src/toy/…` | status | evidence |
|---|---|---|
| `model.py` | **verbatim** (byte-identical) | `hist` `2025436f/NDD4.py`, saved 2026-05-08T19:46:49Z, resource `repos/norway/model.py`. md5 `3e40688a…` matches `src/toy/model.py` |
| `glass.py` | **verbatim** | `codex` line 2291 |
| `mfm_model.py` | **verbatim** through line 280, tail reconstructed | `codex` line 2288; the transcript was cut at `sed -n 1,280p`, so the `__main__` self-test below it is reconstructed (marked in the file) |
| `train_mfm_diagonal.py` | **verbatim** through line 280, CLI reconstructed | `codex` line 2288; argparse block below it reconstructed from the flag names used in `term1` and in `probe_mfm_diagonal.py` |
| `probe_mfm_diagonal.py` | **verbatim**, imports/paths adapted | `codex` line 2289; only `from x import` → `from .x import` and `figures/` → `paths.FIGURES` |
| `train_mfm_consistency.py` | **partly recovered** | `codex` lines ~15669–15738 give L80–135 and L240–295 (`make_consistency_batch`, the JVP/ESD target, the logging and the checkpoint dict); the rest is reconstructed around them. Flag names and defaults are pinned by the `rg` hits in `recovered/rg_hits_norway_2026-06-11.txt` and by the full training command in `term1` |
| `figure.py` | **near-verbatim** | the plotting code of `norway_base_smiley_four_panel.ipynb`, `hist` `-491f92fd/hEXB.ipynb`, saved 2026-05-06T15:20:59Z — the notebook that produced the paper's Fig. 1. Wrapped in functions; panels are passed in instead of loaded from `figures/norway_base_smiley_four_panel_data.pt` |
| `data.py` | reconstructed | interface fixed by `train_mfm_diagonal.py` (`NUM_CLASSES, Y_DIM, sample_flag, sample_uniform_color_square`) and by the figure notebook (`COLOR_NAMES, NUM_CLASSES`); distribution from `paper` F.7.1 |
| `train.py` | reconstructed | Table 12 (batch 4096, 20k steps, lr 3e-4, warmup 500, EMA 0.999, cosine schedule) + the objective named in the base checkpoint that `train_mfm_diagonal.py` loads |
| `reward.py` | reconstructed | `paper` F.7.2, with the names used by the figure notebook (`norwegian_region_scores(..., order="model")`) and the MFM notebook (`norwegian_color_reward_for_model_state(..., k, color_score="log_softmax", detach_regions=True)`, `norwegian_color_reward_raw`) |
| `reward_smiley.py` | reconstructed | `paper` F.7.3 + the shape kwargs recorded in the MFM notebook (`eye_sigma 0.28, mouth_sigma 0.23, mouth_x_half 0.84, mouth_y_low −0.88, mouth_y_high 0.18, mouth_y0 −0.55, mouth_curve 0.65, k 18, color_temp 1`). The eye centre is **not** recorded; (0.30, 0.35) was measured off the published Fig. 1 raster |
| `value_fn.py` | reconstructed | `term1` states the estimator exactly: `X1 = glass_integrate(base_model, t_cond=t, x_cond=x_t, n_samples=N, n_steps=n_glass_steps); V_t(x_t) = logmeanexp(r(X1))`; `freeze_model` / `make_joint_teacher_v` names come from the verbatim `train_mfm_diagonal.py` imports |
| `mfm_value_fn.py` | reconstructed | `term1` gives the plan verbatim (`load_mfm`, `mfm_posterior_sample`, `mfm_value_function`, the `I = randn; for s0, s1 in posterior_grid: I = mfm.X(...)` loop) and the CLI default `--ckpt checkpoints/mfm_consistency_10000.pt` |
| `sample.py` | reconstructed | imported by the MFM notebook (`load_model(..., use_ema=True)`, `resolve_figure_path`); the sampler follows `paper` E.2.1 / Table 12, and its time grid reproduces the times printed in the notebook's saved output (t = 0.100, 0.599, 0.799, 0.899, 0.989 at steps 11/61/81/91/100) |
| `guidance.py` | reconstructed | the MFM notebook gives the two call signatures and the full final configs; `term1` gives the update rule (`dx/dt = base_velocity + guidance`, `guidance = ∇_x V_t(x)`), the fixed-ratio colour normalisation, the chunking flags and the metrics |
| `fig1.py`, `table12.py`, `smiley_recal.py`, `paths.py` | **new** | drivers written for this repo; no original counterpart |

`figure.py` carries one deliberate deviation: the last *saved* state of the figure notebook uses
`left=0.001, right=0.999, wspace=0`, which clips the `y_2` label, and the published figure does not
— so the notebook was edited once more before the figure was made. `LAYOUTS["published"]` restores
the geometry measured off `figures/norway.pdf` (four 269 × 269 px panels at x = 83, 433, 782, 1131
and y = 140…408 in a 1442 × 469 px render); `LAYOUTS["notebook"]` keeps the recovered values.

## Which guidance is in the published Fig. 1?  (authors' decision 7)

**Plain GLASS posterior guidance, not multiMFM.** The evidence is a stored provenance record
inside the figure notebook itself.

`figures/norway.pdf` in the paper source tree is byte-for-byte the same rendered image as
`figures/norway_base_smiley_four_panel_tweak.pdf`, i.e. the output of
`norway_base_smiley_four_panel.ipynb`. Five saves of that notebook survive in `hist`
(`-491f92fd/`): 2026-05-06 02:02, 02:06, 02:30, 02:38, 02:39 and 15:20 Z. The 15:20 Z save is
the one whose plot knobs (titles `$p_0(x)$ … $p_1(x)e^{r_{\mathrm{Smiley}}(x)}$`, FIGSIZE
10.3 × 3.35, TITLE_SIZE 25, y-label only on the first panel) match the published figure
exactly.

The 02:30–02:39 saves still carry their cell outputs, which print the panel bundle's
`sources` field:

```
{'prior' : 'torch.randn with seed=0, shape matching Norway endpoints',
 'norway': 'figures/uniform_norway_slow_color_guidance_N10000_freeze_t08_ratio2_endpoints.pt',
 'smiley': 'figures/uniform_smiley_guidance_N1000_lam0p2_beta16_mc16_steps200_glass8_chunk100_endpoints.pt'}
```

`glass8` is the number of GLASS Euler steps; `slow_color_guidance` is the GLASS
delayed-colour script (`dMFM_experiments/experiment_uniform_norway_slow_color_guidance.py`,
named as the GLASS predecessor in `term1`). The final 15:20 Z save replaces only the smiley
panel, with
`figures/uniform_smiley_delayed_color_bigger_smile_thick_mouth_N10000_endpoints.pt` — the
same GLASS sampler at N = 10,000 with the larger "bigger smile / thick mouth" geometry.

All of those files existed on 2026-05-06. The toy MFM did not: `term1` records the MFM being
distilled and then used for the first time in the June session, and the earliest save of
`mfm_guidance_norway_smiley.ipynb` (the notebook that swaps GLASS for the MFM) is
2026-05-11T21:27Z. `term1` also states the conclusion of that later comparison in the
authors' own words — "the MFM guidance is slightly weaker or noisier than the original GLASS
guidance", and "MFM guidance works, but needs about 3× stronger normalized color guidance
than GLASS for this setup" — which is consistent with the crisp flag boundaries in the
published panel.

So the published Fig. 1 shows GLASS posterior guidance of the base multimodal flow model.
The caption's claim that it is "posterior guidance using a multimodal Meta Flow Map" is not
supported by the figure's own provenance. `results/toy/` therefore contains both variants.

## Original-run metrics (the rebuild is checked against these)

The paper reports no numbers for the toy, so `term1` is the reference. See
`toy.guidance.ORIGINAL_RUN_METRICS` and `results/toy/paper_values.json`.

## How faithful is the rebuild?

Checked against the metrics the original session logged (no paper numbers exist for the toy):

* **Training** — reproduced. The retrained ESD stage lands on the original's final losses
  (total 0.00227 vs 0.00211, diagonal 8.9e-5 vs 1.19e-4, ESD 0.00218 vs 0.00199), gap statistics
  (mean 0.3094 vs 0.309, max 0.9152 vs 0.915) and held-out diagonal probe (EMA relative L2 0.01383
  vs 0.0141, cosine 0.999894 vs 0.99988). The retrained base model reproduces the original's
  unguided Norway reward 0.324 / colour match 0.334 and the printed `‖v_base,c‖ ≈ 7.2`.
* **Guidance** — reproduced for Norway (multiMFM ratio 6: reward 0.4721 / match 0.7711 vs the
  original's 0.471 / 0.768) and close for the multiMFM smiley (0.5769 / 0.9710 vs 0.563 / 0.961).
* **The published figure** — the panel geometry and style are reproduced exactly; the Norway panel
  is visually indistinguishable; the smiley features come out about 2× tighter than published,
  which is the one unrecorded knob (β; see `src/toy/smiley_recal.py` and
  `results/toy/smiley_beta_calibration.json`). The original endpoint tensors are gone, so the point
  clouds can never match sample for sample.

## What is still unknown

* the exact `data.py`, `train.py`, `reward*.py`, `value_fn.py`, `sample.py` source — only their
  interfaces and their maths are pinned;
* the smiley eye centre (x_e, y_e) — (0.30, 0.35) was estimated off the published raster, whose
  feature centroids are (−0.286, +0.332) and (+0.312, +0.334), so it is close;
* the colour-guidance ratio and β of the final N = 10,000 GLASS smiley run (its N = 1,000
  predecessor used β = 16, and the published blob widths imply a smaller effective value);
* whether `pos_lambda_base="sde_sigma_sq"` means σ_t² = 2(1/t − 1) — that is the formula in the same
  author's DNA code (`dmfm.models.dna_models.sde_sigma_sq`), which is why it was used here;
* the base-model and diagonal-stage training commands (only Table 12's values / the recovered
  defaults; the ESD command is verbatim);
* the panel data bundle `figures/norway_base_smiley_four_panel_data.pt` itself.
