# Changes to `iclr2027_conferencemod.tex`

Log of every edit made to the paper source during the reproducibility pass (2026-09-29), the
placeholders still open, the edits that are queued behind reruns, and an audit of every printed
number that could be checked against `results/`.

**On line numbers**: every line number in this document was correct when written and drifts as later
edits land. The stable keys are the LaTeX labels, the quoted old/new text, and for placeholders the
marker text itself (`grep -n "TODO" iclr2027_conferencemod.tex`). Section
[1.12](#112-placeholder-inventory) is regenerated from the file and is the one table to trust for
placeholder locations.

Scope: only `iclr2027_conferencemod.tex` (repo root) is edited here. Table/figure numbers below are
the numbers the compiled document assigns (verified against `paper.aux`; see
[Compilation](#compilation)). Line numbers are those of the current file after all edits listed in
[1. Edits applied](#1-edits-applied); where an edit moved later lines, the numbers in later sections
are already post-edit.

Paper table label -> number map (for anyone cross-referencing labels):

| # | label | # | label |
|---|---|---|---|
| 1 | `tab:dna-independent-oracle` | 12 | `tab:hyperparameters_multimfm` |
| 2 | `tab:full_property_comparison` | 13 | `tab:exact-dna-reward-improvement` |
| 3 | `tab:geom_drugs_multimfm` | 14 | `tab:scale-free-summary` |
| 4 | `tab:yeast_real_diversity_baselines` | 15 | `tab:scale-free-full` |
| 5 | `tab:yeast_generated_diversity` | 16 | `tab:pearson_corr` |
| 6 | `tab:yeast_evo2_plausibility` | 17 | `tab:dna_posterior_diversity_scaling` |
| 7 | `tab:hyperparameters_microft` | 18 | `tab:raw-gradient-mae` |
| 8 | (second `\caption` inside Table 7's float; no content) | 19 | `tab:tabasco-base-training` |
| 9 | `tab:dna-per-length` | 20 | `tab:tabasco-mfm-training` |
| 10 | `tab:parent-split` | 21 | `tab:pb-validity` |
| 11 | `tab:c0-regressor-accuracy` | 22 | `tab:qm9-uniqueness` |

Figures: 1 `fig:multimodal-toy`, 2 `fig:dMFM-inner-time`, 3 `fig:dMFM_cyclizability_schematic`,
4 `benchmark_dMFM`, 5 `fig:dna-scaling`, 6 `fig:yeast_generation_eval`, 7 `fig:histograms`,
8 `fig:guidance_demo`. Algorithms: 1 dMFM training, 2 multiMFM steering, 3 multiMFM + steer-search.

---

## 1. Edits applied

### 1.1 Table 7 - dMFM loss weights (authors' decision 6a)

| | |
|---|---|
| Lines | 1390-1391 |
| Old | `Diagonal weight            & $0.35$ \\` / `Consistency weight                 & $10.0$ \\` |
| New | `Diagonal weight            & $1.0$ \\` / `Consistency weight                 & $1.0$ \\` |

Justification: every surviving DNA training record uses 1.0/1.0. Checked all nine arg dumps in
`results/dna/training/`:

* `mfm_diag_loss_weight = 1.0` in `dmfm_L{50,100,200,400}_args.json`,
  `dmfm_4step_L{50,100,200,400}_args.json`, `dmfm_L400_parent_esd_run_args.json` (9/9).
* `mfm_lsd_loss_weight = 1.0` in 8/9. The exception is `dmfm_L400_args.json`
  (`yeast_parent_a_L400_dmfm_diagonly_finetune_20260726`), which has `mfm_consistency = false`,
  `mfm_consistency_type = "lsd"` and `mfm_lsd_loss_weight = 0.0`: that run is diagonal-only by
  design (see `results/dna/PROVENANCE.md`, "fine-tune from step 45,000 ..., no consistency loss").
* `mfm_lsd_loss_weight` is the weight actually applied to the consistency term:
  `src/dmfm/utils/mfm_diag_distill.py:437` (`w_lsd = float(getattr(self.args,
  "mfm_lsd_loss_weight", 1.0))`), with the diagonal weight at line 346.
* No record anywhere contains 0.35 or 10.0 as a loss weight.

### 1.2 Appendix F.6 - evaluation time (decision 6b)

| | |
|---|---|
| Line | 2068 |
| Old | `$t=0.3$, eight matched noise pools per state,` |
| New | `$t=0.5$, eight matched noise pools per state,` |

Justification: all twelve per-length metadata files of the run behind Table 18 record
`"t_eval": 0.5` (`results/dna/dmfm_vs_glass/{gc,motif,conjunction}/L{50,100,200,400}/metadata.json`
and the three `run_metadata.json`; 12/12, no other value present).
`results/dna/PROVENANCE.md` also states "**t_eval 0.5**" for job array 35642548.
The rest of the F.6 protocol sentence is confirmed by the same metadata: 32 states (`n_probes`),
8 pools (`n_repeats`), GLASS-128 reference (`reference_mc`), 50-step RK4
(`reference_glass_steps`/`reference_glass_solver`), four-map dMFM (`dmfm_steps: 4`).

### 1.3 Table 12 - toy training stages (decision 6c)

| | |
|---|---|
| Lines | 1648-1673 (was one `Optimization` + one `Training` block; now four blocks) |

Old:

```
\multicolumn{2}{l}{\emph{Optimization}} \\
Optimizer & AdamW \\ Peak learning rate & $3\times 10^{-4}$ \\
Learning-rate schedule & Cosine decay with warmup \\ Warmup steps & $500$ \\ EMA decay & $0.999$ \\
\multicolumn{2}{l}{\emph{Training}} \\
Batch size & $4096$ \\ Training steps & $20{,}000$ \\
```

New: `Optimization (all stages)` (optimizer, schedule, warmup only), then three per-stage blocks

| block | batch | steps | peak lr | EMA |
|---|---|---|---|---|
| Base flow model | 4096 | 20,000 | 3e-4 | 0.999 |
| Meta flow map, diagonal stage | 2048 | 10,000 | 3e-4 | 0.999 |
| Meta flow map, ESD stage | 1024 | 10,000 | 1e-4 | 0.995 |

Justification and **cross-check request to the toy agent**:

* The printed values (4096 / 20,000 / 3e-4 / 0.999) are the **base** stage, verified from the
  checkpoint: `checkpoints/toy/ckpt_uniform_color_square.pt` ->
  `training_config = {steps: 20000, batch_size: 4096, lr: 3e-4, warmup: 500, ema_decay: 0.999}`,
  `objective = "endpoint_mse_plus_ce"`.
* Diagonal stage verified from `checkpoints/toy/mfm_diagonal_10000.pt` ->
  `training_config = {steps: 10000, batch_size: 2048, lr: 3e-4, warmup: 500, ema_decay: 0.999}`,
  `objective = diagonal_glass_velocity`. Note this is **not** what the brief assumed ("the listed
  values describe the base/diagonal stage"): the diagonal stage used batch 2048 and 10k steps, so a
  two-stage split would still have been wrong. `src/toy/table12.py` (toy agent's own module) says
  the same in its `note`.
* ESD stage verified (cross-check closed) from `checkpoints/toy/mfm_consistency_10000.pt`, which
  appeared while this was being written ->
  `training_config = {steps: 10000, batch_size: 1024, lr: 1e-4, warmup: 500, ema_decay: 0.995,
  gap_anneal_steps: 5000, diagonal_weight: 1.0, consistency_weight: 1.0, target_clip: 10.0}`,
  `objective = esd_teacher_consistency_plus_diagonal_glass`. The toy agent independently records the
  same four values in `results/toy/paper_values.json` -> `table12_missing_esd_stage`, sourced to the
  original session transcript (`docs/provenance/toy/recovered/cursor_terminal_1.txt` lines 30-40),
  i.e. these are the original run's settings and not merely the reconstruction's defaults. They also
  match `src/toy/train_mfm_consistency.py`'s argparse defaults and
  `docs/REPRODUCIBILITY_PLAN.md` 7.5.
* Everything else in Table 12 (K=3, d=2, D=128, 6 blocks, 8 heads, MLP x4, adaLN-Zero, 2 tokens,
  AdamW, cosine+warmup, warmup 500) matches all three checkpoints' `config`. Recorded but not
  carried into the table: the ESD stage's `gap_anneal_steps = 5000` and `target_clip = 10.0`.
* Not changed, but see [3.9](#39-table-12-loss-block): the `Total loss` row is still wrong.

### 1.4 Appendix F.8 - MFM distillation data (decision 6d)

| | |
|---|---|
| Lines | 2270-2272 (one sentence inserted) |
| Added | `Distillation used a fixed subset of each training set rather than all of it: $150{,}000$ of the $\approx 1.17$M GEOM-Drugs molecules, and $40{,}000$ of the $50{,}000$ QM9 molecules on which the QM9 base generator was trained.` |

Justification:

* GEOM: `results/geom/table20_geom_hparams.json` ->
  `distillation_data = {limit_data: 150000, available_molecules: 1141716, note: "the GEOM MFM was
  distilled on 150k of the ~1.17M molecules"}` (read out of
  `checkpoints/geom/mfm_student/student_step_10000_args.json`).
* QM9: `results/qm9/table20_mfm_training.json` -> `args.limit_data = 40000`, against
  `results/qm9/table19_base_training.json` -> `training_set_molecules = 50000`,
  `limit_data = 50000`.
* "$\approx 1.17$M" is Table 19's own wording and is kept unchanged so no new count enters the
  paper; see [3.4](#34-geom-training-set-size) for the 1,141,716 vs 1,172,182 question.
* The QM9 half of the sentence is beyond the letter of decision 6d. It is equally well sourced and
  pre-empts the obvious follow-up question, but delete it if the authors prefer the minimal fix.

### 1.5 Table 20 - omitted GEOM phase settings (decision 6e)

| | |
|---|---|
| Lines | 2355-2358 (four rows inserted in the *Objective / conditioning* block) |

```
        Diagonal / ESD loss weights & $1.0\,/\,1.0$ & $2.0\,/\,1.0$ \\
        ESD delta / target clip & None & $8.0\,/\,10.0$ \\
        Maximum inner-time jump $u-s$ & $0.5$ & $0.15$ \\
        Inner-gap anneal fraction & $0.5$ & $0.2$ \\
```

Justification, GEOM column: `results/geom/table20_geom_hparams.json` ->
`omitted_from_table20 = {max_jump: 0.15, anneal_frac: 0.2, esd_delta_clip: 8.0,
esd_target_clip: 10.0, diag_weight: 2.0, esd_weight: 1.0, t_cond_power: 2.0,
lr_schedule: "cosine"}`.
QM9 column: `results/qm9/table20_mfm_training.json` -> `args = {diag_weight: 1.0, esd_weight: 1.0,
esd_target_clip: 0.0, esd_delta_clip: 0.0, max_jump: 0.5, anneal_frac: 0.5}`. "None" for the QM9
clips is the code's own semantics: the clamp is gated on `target_clip > 0.0`
(`src/tabasco/flow/mfm_losses.py:590`), so 0.0 means no clipping.
`t_cond_power` (1.0 QM9 / 2.0 GEOM) and `lr_schedule` were left out: the schedule is already
implied by the existing "Cosine schedule horizon" row, and `t_cond_power` has no natural
one-line name in the paper's notation. Add them if the authors want the block complete.

### 1.6 Table 1 and 3.1 - n=32 pilot -> n=1000 run (authors' decision 2)

| | |
|---|---|
| Lines | 355-366 (table body), 324-325, 328 (3.1 prose) |

| cell | old (n=32) | new (n=1000) |
|---|---|---|
| +1 mean oracle, unguided | -0.169 | -0.171 |
| +1 mean oracle, guided | 0.912 | 0.917 |
| +1 unguided MAE | 1.170 [1.038, 1.291] | 1.174 [1.149, 1.200] |
| +1 guided MAE | 0.157 [0.119, 0.196] | 0.148 [0.141, 0.156] |
| +1 paired improvement | 1.013 [0.881, 1.142] | 1.025 [0.999, 1.052] |
| -1 mean oracle, unguided | -0.169 | -0.171 |
| -1 mean oracle, guided | -0.916 | -0.949 |
| -1 unguided MAE | 0.831 [0.709, 0.964] | 0.833 [0.806, 0.857] |
| -1 guided MAE | 0.153 [0.118, 0.190] | 0.119 [0.114, 0.125] |
| -1 paired improvement | 0.678 [0.554, 0.816] | 0.713 [0.687, 0.740] |
| 3.1: "reduces oracle MAE from 1.17 to **0.16** ... from 0.83 to **0.15**" | | "... to **0.15** ... to **0.12**" |
| 3.1: "The guided errors of **0.16 and 0.15** ... to within **0.15**" | | "**0.15 and 0.12** ... within **0.12**" |

Justification: `results/dna/c0_guidance/table1_n1000_summary.json`
(`table1_n1000_latex_rows`, `in_text_sec3_1`), which reproduces the stored
`results/dna/c0_guidance/n1000/target_*/oracle/oracle_summary.csv` to 2e-16 with the original
percentile-bootstrap method (2,000 resamples, `default_rng(0)` per statistic, float32). Slurm jobs
48651189 (+1) / 48651190 (-1).
"1,000 guided--unguided pairs" (line 320) is now correct as printed and was left alone.
Not changed: the oracle's own held-out MAE quoted in the same sentence stays 0.16
(`results/dna/c0_regressors/oracle/metadata.json` -> `test_metrics.mae = 0.16198`).

### 1.7 Section 3.2 - the GEOM fingerprint guide is not off-the-shelf

| | |
|---|---|
| Line | 593 |
| Old | `Our $\mathcal E(G)$ is off-the-shelf from TFG-Flow.` |
| New | `TFG-Flow does not release a structural-fingerprint predictor, so we train $\mathcal E(G)$ ourselves on GEOM-Drugs, following its guide architecture (a six-layer discriminative EGNN with mean pooling and a linear head) and its fingerprint definition.` |

Justification: `results/geom/tfgflow_reference.json` (geom agent, 2026-09-29), which checked the
TFG-Flow paper, its README's "Pre-trained Models" section and a local clone of
`github.com/linhaowei1/tfg-flow@4e7ef16`: `structural_fingerprint_predictor_released: false` - the
only released predictors are the six QM9 property models (which *are* used off the shelf for our
QM9 experiments) plus the QM9 base flow. The head used for Table 3 is
`checkpoints/geom/structure_head/best.pth`, trained here (`third_party/tfg_flow/
train_structure_head.py`). The architecture wording follows TFG-Flow App. D.2 as quoted in that
record ("we train a 6-layer discriminative EGNN by adding linear head on the average pooling of
output node feature").

The 30-vs-71 heavy-atom caveat noted here (TFG-Flow's GEOM-Drugs dataloader caps molecules at 30
atoms on a 5.52M-conformer split, ours generates up to 71, so the two Tanimoto values are not
measured on the same molecule-size distribution) was flagged for the authors at this point and has
since been **written into both 3.2 and the Table 3 caption** as part of
[1.8](#18-geom-drugs-table-3-its-caption-app-f8-and-f9-table-21-decision-1-second-round).

### 1.8 GEOM-Drugs: Table 3, its caption, App. F.8 and F.9, Table 21 (decision 1, second round)

Applied from the geom agent's `docs/provenance/geom/paper_sentences.md`, items 1-7 (its item 1 was
already done in 1.7 and is here upgraded to its fuller wording). **All GEOM numbers below are
interim** and carry a visible `[TODO rerun: ...]` marker in the tex: they are the mean +/- SE over
the three seeds x 4,096 molecules of the rerun of the *submitted* `(lambda, mu) = (0.02, 0.5)`
configuration (Slurm 47833756, `results/geom/table3_n4096_replicates.csv`). The as-described run,
with `(lambda, mu) = (0.01, 0.25)` selected on 512 held-out target fingerprints
(`results/geom/table3_asdescribed_selection.json`), is Slurm **49274410** and still queued.

| # | line(s) | change |
|---|---|---|
| (a) | 599 | `Our $\mathcal E(G)$ is off-the-shelf from TFG-Flow.` -> `Our $\mathcal E(G)$ follows TFG-Flow's guide predictor for this task---a six-layer EGNN with mean pooling and a linear head, trained with binary cross-entropy against the same 1024-bit OpenBabel FP2 fingerprint---which we train ourselves on GEOM-DRUGS, since TFG-Flow releases only its QM9 property predictors.` (the 1024-bit FP2 and BCE details are in `docs/provenance/geom/checkpoints.md`) |
| (b) | 603 | Table 3 caption: the two reference values are now identified as TFG-Flow's GEOM-DRUGS rows averaged over three seeds, with the 30-vs-71 heavy-atom caveat; `87.5\%`/`92.0\%` -> `85.0\%`/`88.0\%` "on the same molecules"; plus the full interim marker |
| (c) | 609-612 | Table 3 body: first row relabelled `TFG-Flow unconditional (their "Upper bound")`; **new row** `Our base generator (unguided) & $0.2287 \pm 0.0005$`; `multiMFM & \textbf{0.227}` -> `multiMFM & $0.2241 \pm 0.0001$` (bold dropped: it is no longer the best value in the column) |
| (d) | 599 | `Our similarity improves on the baseline TFG-Flow, although still moderate.` -> a negative result, see below |
| (e) | 2380-2392 | App. F.8: added the guidance settings (`lambda=0.01`, `mu=0.25`, every second outer step, `t in [0.1,0.9]`, selected on 512 held-out targets) and the note that the samplers' `sigma_t^2/2 = (1-t)/t` is absorbed into the constant `mu` the implementation uses |
| (f) | 2400 | App. F.9: `generated $4{,}096$ molecules; its reported Tanimoto similarity and PB validity come from the same run` -> `generated $4{,}096$ molecules for each of three seeds; ... and the unsteered column is the unguided arm of the same runs (identical noise and identical targets)` |
| (g) | 2422, 2427 | Table 21 GEOM row `$4{,}096$ & $92.0$ & $87.5$` -> `$4{,}096\times3$ & $88.0$ & $85.0$`; the paragraph after it now gives both numbers "on the same molecules". Table 21's `\tabcolsep` went 4.5pt -> 3.0pt because `$4{,}096\times3$` made the table 12.3pt overfull |

The replacement for (d), as it now stands in the tex:

> TFG-Flow's GEOM-DRUGS pipeline caps molecules at 30 heavy atoms while ours generates up to 71, so
> its two reference values set the scale of the task rather than a matched baseline. Our own
> unconditional base generator already exceeds them on this metric (0.2287 versus 0.208), but
> steering does not improve the directly counted fingerprint similarity at this scale: the paired
> change over 4,096x3 molecules is -0.0046 +/- 0.0008, and PoseBusters validity falls from 88.0% to
> 85.0%. The guide-predicted similarity moves the same way (0.2338 to 0.2308), so the shortfall is
> not an artifact of scoring with directly counted fingerprints instead of the guide. We report this
> as a negative result for this benchmark.

**One deviation from the geom agent's suggested wording, and it mattered.** Its item 4 proposed
ending with *"The guide's own (head-predicted) similarity does improve, so the gap is a
guide-quality and not an estimator problem"*. That is **not true of the 4,096x3 run**. From
`results/geom/table3_n4096_shards.csv`, the head-predicted Tanimoto *falls* in every one of the
three replicates: 0.23500 -> 0.23129, 0.23259 -> 0.23031, 0.23376 -> 0.23078 (pooled 0.233785 ->
0.230795; 9 of 11 shards negative). It improved only in the n=64 grid point (0.2165 -> 0.2291). I
wrote the factual version above instead. **The geom agent has since retracted the claim** and
confirmed the direction at five further held-out sweep points; see
[3.14](#314-the-geom-guides-own-head-predicted-similarity-falls-too-retraction-recorded). The
paper states the measured numbers and offers no mechanism.

**Also for the authors, not edited:** the abstract's closing claim ("In both cases, our steering
algorithms improve target attainment") now covers a dataset where steering does not improve the
reported metric. The geom agent's item 10 sets out what GEOM-DRUGS does and does not support, and
every line of it is negative for attainment:

| quantity | base -> steered | source |
|---|---|---|
| counted FP2 Tanimoto (the reported metric) | 0.2287 -> 0.2241, paired -0.0046 +/- 0.0008 | `table3_n4096_replicates.csv` |
| the guide's own head-predicted Tanimoto | 0.2338 -> 0.2308, 9/11 shards negative | `head_tanimoto_direction.json` |
| PoseBusters validity | 88.0% -> 85.0% | same run |
| every held-out $(\lambda,\mu)$ tried | negative on both similarities | `table3_asdescribed_tuning.csv` |

What GEOM-DRUGS *does* support is the scaling claim ("We scale multiMFM to GEOM-DRUGS", molecules of
up to 71 heavy atoms against 9 for QM9), which is untouched, and that our base generator is strong in
absolute terms - but that is a base-model comparison across different molecule-size ranges, so it is
context rather than a win. The honest options are to restrict the improvement claim to QM9 and DNA,
or to keep "both cases" only if it is about scaling. **This is a framing decision for the authors and
I have not touched the abstract.** If the coefficient question in
[3.15](#315-the-guided-ode-as-implemented-is-not-the-guided-ode-as-written) turns out to be the cause,
the claim could be restored after a fix - but not on the strength of the present runs.

### 1.9 Table 1 and 3.1 reverted to placeholders (authors' decision: Table 1 must be dMFM)

| | |
|---|---|
| Lines | 355-372 (table body), 324-325, 328 (3.1 prose) |

The n=1000 values applied in 1.6 are **withdrawn**: both the submitted n=32 pilot and the n=1000
rerun used GLASS posterior guidance on the base discrete flow with no dMFM checkpoint loaded
(3.1 of this document), and the authors decided Table 1 must report a dMFM run. The ten cells are
now `\textbf{[TODO rerun]}`, with a spanning row in the table stating why, and the three 3.1
numbers plus the "to within X" figure are placeholders. dna-core is submitting a paired n=1000
dMFM run and will write `results/dna/c0_guidance/table1_dmfm_n1000_summary.json`.

Kept for the record - **what was in the table before the revert** (GLASS, n=1000, from
`results/dna/c0_guidance/table1_n1000_summary.json`; the submitted n=32 values are in 1.6):

| cell | submitted (n=32 GLASS) | n=1000 GLASS (briefly applied, now withdrawn) |
|---|---|---|
| +1 mean oracle unguided / guided | -0.169 / 0.912 | -0.171 / 0.917 |
| +1 unguided MAE | 1.170 [1.038, 1.291] | 1.174 [1.149, 1.200] |
| +1 guided MAE | 0.157 [0.119, 0.196] | 0.148 [0.141, 0.156] |
| +1 paired improvement | 1.013 [0.881, 1.142] | 1.025 [0.999, 1.052] |
| -1 mean oracle unguided / guided | -0.169 / -0.916 | -0.171 / -0.949 |
| -1 unguided MAE | 0.831 [0.709, 0.964] | 0.833 [0.806, 0.857] |
| -1 guided MAE | 0.153 [0.118, 0.190] | 0.119 [0.114, 0.125] |
| -1 paired improvement | 0.678 [0.554, 0.816] | 0.713 [0.687, 0.740] |
| 3.1 prose | 1.17 -> 0.16, 0.83 -> 0.15 | 1.17 -> 0.15, 0.83 -> 0.12 |
| fraction of pairs improved | 1.000 / 1.000 | 0.983 / 0.971 |

Unchanged and still correct: "we generate 1,000 guided--unguided pairs" (line 320), the oracle's own
held-out MAE of 0.16, and 3.1's "We evaluate dMFM for DNA design guided by predicted cyclizability",
which the new run will make true as written.

### 1.10 Figure 5, Tables 14-15, 3.3 and App. F.6 marked as awaiting the dMFM rerun

Same defect as Table 1: the finite-$N$ estimates were GLASS on the base discrete flow.
Independently confirmed here from `results/dna/scale_free_mc/*/run_metadata.json`, which lists
`ckpt = []` (no dMFM checkpoint), `per_length[0].checkpoint =
workdir/yeast_parent_a_L50_dfm_dit_h192_b4/best.pt` (the base generator) and
`reference = "independent finite-MC GLASS estimator with reference_mc samples"` - i.e. estimator
*and* reference are GLASS on the base flow.

| line(s) | change |
|---|---|
| 656 | Figure 5 caption: dMFM wording kept, marker added (both panels awaiting the rerun) |
| 1868 | Table 14 caption: marker added (every entry awaiting the rerun) |
| 1900 | Table 15 caption: same marker |
| 665 | 3.3: `relative errors are $1.40$--$2.55$ and cosine similarities are $0.18$--$0.32$` -> four `[TODO rerun]` placeholders, plus a marker saying the "$N=8$ is an effective operating point" claim itself is awaiting the rerun |
| 2086 | App. F.6: the sentence "...compares GLASS Monte Carlo error across lengths, without measuring dMFM approximation error" is the one true statement about the recorded runs and therefore **contradicts** App. F.2 (line 1839, "terminal samples from the dMFM sampler") and the Figure 5 caption. Marked, so that whoever lands the rerun resolves the three together rather than half of them |

The 108 Table 15 cells and the 18 Table 14 cells were not individually placeheld: that would make the
tables unreadable, and the caption markers are unmissable. Same choice for Tables 4-6 below.

### 1.11 Tables 4-6 and App. B.1 marked as awaiting parent-disjoint regeneration

| line(s) | change |
|---|---|
| 736, 754, 774 | captions of Tables 4, 5, 6: marker saying the entries come from the June 2026 `split65k` random-window split, not the parent-disjoint models this appendix describes; Table 6's adds that the per-sequence Evo 2 scores behind its CIs were lost, so it needs a new scoring run |
| 719 | App. B.1's opening: marker stating explicitly that Tables 4-6 as printed do not yet match the parent-disjoint description in that same sentence |
| 770 | B.1 diversity paragraph: nine numbers -> `[TODO rerun]`, plus a note that the parent-disjoint recomputation softens the -1 clustering substantially so the direction-specific conclusion may not survive |
| 794 | B.1 Evo 2 paragraph: six numbers -> `[TODO rerun]`, same note |
| 407 | 3.1: the "more local clustering for the negative target" sentence marked, since it rests on those tables |

**Superseded for Tables 4 and 5 and the diversity prose**: the regeneration landed and those are now
final, see [1.14](#114-tables-4-and-5-and-the-app-b1-diversity-paragraph-substituted-final). Only
Table 6 and the Evo 2 paragraph remain marked.

### 1.13 Figure 4 and the two 3.3 paragraphs (decision 3, text errors independent of the rerun)

Two of these are outright misdescriptions of what was run and are fixed now; the third is an
unsupported claim and is marked.

| line(s) | change |
|---|---|
| 631-643 | Figure 4 caption rewritten for both panels (was one sentence: *"Left: Value-function estimation error as the number of Monte Carlo samples $N$ increases. Right: Target error versus reported network function evaluations (NFEs)."*). It now states that the left panel's reward is a **synthetic smooth sequence reward and not the cyclizability guide**, that the conditioning states are held-out test sequences noised to their own $t\sim\mathcal U(0.05,0.35)$, that the reference $V_t$ is a teacher-SDE Monte Carlo estimate and is itself finite-sample **so every curve is bounded below by the reference's own sampling error**, that DPS uses the student's denoiser and FMAP the teacher ODE to $t=1$ (hence flat in $N$), and for the right panel that every method shares base generator, guide, target and evaluation with per-baseline hyperparameters chosen on a separate pilot at matched NFE. Two bracketed clauses remain as placeholders (lines 637, 641) |
| 647 | 3.3 "Posterior Recovery": added the reward, the conditioning-state construction, and the sentence that the reference is itself a Monte Carlo estimate so *"the comparison is therefore bounded below by the reference's own sampling error and does not resolve differences beneath it"*. The paragraph previously implied the cyclizability reward of 3.1 throughout |
| 651 | 3.3 "Steering": added *"Every method uses the same base generator, the same guide, the same target and the same evaluation, and each baseline's hyperparameters were chosen on a separate pilot at matched NFE."*, and marked the claim *"dMFM achieves low target error at low NFEs, the regime most relevant for practical sampling"* as awaiting the rerun, stating in the marker that the four baseline curves were never measured (hard-coded, with SEM 0 at three repeats) |

Sources: `results/dna/original_scripts/benchmark_yeast_split_value_mc.py` (recovered verbatim from
`tullebulle/DNA-MFM@187fe7b`), summarised in `results/dna/PROVENANCE.md` item 3 and
`results/dna/fig4/PROVENANCE.md`, and the protocol cells of `notebooks/06_dna_benchmarks.ipynb`
(cells 7 and 20, "Caveats to carry into the caption and text"). Those give: reward
`SmoothDNAReward(seed=1234, gain=4.0)` = random linear + nearest-neighbour pair terms + a fixed
8-base motif bonus + GC penalty + entropy bonus; `--reward_gain 4.0`; conditioning
`x_t = beta_t x_1 + (1-beta_t) eps` on held-out `test` sequences with
`t in [--tcond_low 0.05, --tcond_high 0.35]`; reference = teacher SDE, `--sde_ref_n 512`,
`--sde_steps 1024`; `V = logsumexp(r) - log N`; `--dmfm_nfe 1`, `--fmap_nfe 256`; and an `sde_mc`
"reference's own error" series that the published panel omits. For the right panel, the hard-coded
baseline block is `results/dna/original_scripts/benchmark_yeast_split_dmfm_steering.py` lines 28-48,
byte-identical to `results/dna/benchmarks/fig4_right_hardcoded_baselines.csv`.

**Note on provenance of the wording.** The dna-fig4 agent's replacement caption was not on disk
anywhere I could find it (searched `results/dna/fig4/`, `docs/provenance/dna/`, `notebooks/06_*`,
`src/dmfm/benchmarks/` and the shared scratchpad), so I wrote the caption from the sources above,
covering exactly the two points the coordinator specified plus the right panel's matched-setup
statement. **dna-fig4: please check it and send corrections**, in particular whether to name the
`sde_mc` floor series in the caption and what the final conditioning-state count is (the original's
default is 8, which `notebooks/06` records as an assumption since no log of the plotted run
survives; the rerun uses 32).

**Main-text length.** The caption was drafted longer and then cut back, because it moved the
Impact Statement from page 11 to page 12. It still does: with the substitute style file, the
untouched document's main text ends on page 11 and the edited one's on page 12. Stripping all 57
`[TODO rerun]` markers gives 34 pages against 38 - so the markers cost about four pages, **all of
them in the appendix**; the extra main-text page comes from the substantive additions (the GEOM
negative-result passage in 3.2, this caption, and the two 3.3 sentences), not from the scaffolding.
**Authors: if ICLR 2027 caps the main text you are now one page over the submission**, and the
natural trims are the GEOM passage (detail could move to F.9) and this caption. I have not made
that choice. Page counts are indicative only: `iclr2027_conference.sty` is missing and a 2026
substitute was used (see [Compilation](#compilation)).

### 1.14 Tables 4 and 5 and the App. B.1 diversity paragraph substituted (final)

The parent-disjoint regeneration is done, so the placeholders from
[1.11](#111-tables-4-6-and-app-b1-marked-as-awaiting-parent-disjoint-regeneration) are replaced with
final numbers. Source: `results/dna/diversity/regen/{table4_real_baselines.csv,
table5_generated.csv,tables.md}`, produced by `dmfm.experiments.analyze_c0_diversity` via
`bash scripts/dna/tables4_5_c0_diversity.sh` on the parent-disjoint n=1,000 C0 sets in
`results/dna/c0_guidance/n1000/` - the same sample sets Table 6's Evo 2 job will score, so Tables
4-6 end up internally consistent. The round-1 split65k recovery
(`results/dna/june_split65k_recovered/`) and the round-1 recompute
(`results/dna/diversity/n1000/`) are both preserved untouched.

**Table 4** (lines 744-755). Rows relabelled to the parent-disjoint split names, which is what the
appendix describes; the caption now says what each set is and the `[TODO rerun]` marker is gone.

| row (old -> new label) | nearest generated | pairwise generated |
|---|---|---|
| `train_sample` -> `full_sample` | 0.5219 (unchanged) | 0.7377 (unchanged) |
| `split_train_sample` -> `a_train_sample` | 0.5219 -> **0.5200** | 0.7373 -> **0.7361** |
| `split_val_sample` -> `a_val_sample` | 0.5184 -> **0.5194** | 0.7372 -> **0.7385** |
| `split_test_sample` -> `test_sample` | 0.5201 -> **0.5219** | 0.7374 -> **0.7381** |

Exact uniqueness (1.0000) and the pairwise median (0.7400) are unchanged in all four rows.

**Table 5** (lines 762-766). All 16 numeric cells move; exact uniqueness and both novelty columns
stay at 1.0000. Read from `table5_generated.csv`:

| target / set | guide score | nearest train | nearest test | nearest gen. | pairwise gen. | pairwise / held-out | nearest gen. / held-out |
|---|---|---|---|---|---|---|---|
| -1 guided | -0.9893 -> **-0.9932** | 0.3966 -> **0.4458** | 0.4361 -> **0.4737** | 0.4175 -> **0.4960** | 0.7010 -> **0.7287** | 0.9507 -> **0.9872** | 0.8027 -> **0.9504** |
| -1 unguided | -0.1692 -> **-0.1767** | 0.4449 -> **0.4583** | 0.4779 -> **0.4817** | 0.5210 -> **0.5221** | 0.7364 -> **0.7374** | 0.9987 -> **0.9991** | 1.0017 -> **1.0003** |
| +1 guided | 1.0010 -> **0.9986** | 0.4446 -> **0.4559** | 0.4769 -> **0.4804** | 0.4913 -> **0.5091** | 0.7372 -> **0.7348** | 0.9997 -> **0.9956** | 0.9447 -> **0.9755** |
| +1 unguided | -0.1766 -> **-0.1767** | 0.4466 -> **0.4583** | 0.4787 -> **0.4817** | 0.5215 -> **0.5221** | 0.7370 -> **0.7374** | 0.9995 -> **0.9991** | 1.0027 -> **1.0003** |

The two `unguided` rows are now **identical**, because the parent-disjoint runs share one unguided
set between the two targets (the split65k runs used a different seed per target). The caption says so,
and also spells out what "train", "test" and "held-out" mean here.

**App. B.1 prose** (lines 730, 736, 781). The clustering claim is softened as the numbers require:
`target_c0 = -1` went from `0.9507x` / `0.8027x` of held-out to `0.9872x` / `0.9504x`, i.e. the
direction survives but the effect is about four times smaller. "measurably more clustered" ->
"clusters slightly more", and the paragraph now closes with the magnitude stated plainly: *"Both
guided sets therefore sit below the held-out baseline on the nearest-neighbour statistic and the
negative target sits further below, but the gap is small: a 1.3% deficit in pairwise diversity and a
5.0% deficit in nearest-generated distance."* The two held-out reference figures in the same
paragraph move with Table 4 (pairwise 0.7374 -> 0.7381, nearest-gen 0.5201 -> 0.5219), and the
`+1` figures become 0.7348 / 0.9956x and 0.5091. The split names in the surrounding methods sentence
(line 736) were renamed to match Table 4 (`split-train`/`split-test`/`split_test_sample` ->
`a_train`/held-out `test`/`test_sample`).

I did **not** put the old split65k ratios into the paper as a comparison, although it is tempting to
show the improvement: the paper should report the parent-disjoint result, and the history belongs
here.

**Still placeheld in B.1**: Table 6 (line 785) and the Evo 2 paragraph (line 805), waiting on the
Evo 2 scoring job. Table 6's marker now states that the samples are regenerated and only the scoring
is pending, that the intervals printed today cannot be recomputed because the per-sequence scores
were lost, and that its three reference row labels become `a_train_sample`, `a_val_sample` and
`test_sample` to match Table 4. The App. B.1 opening marker (line 730) was narrowed from
"Tables 4-6" to Table 6 alone.

**Also added** (line 736, one sentence, not a placeholder): Evo 2 7B is run in bf16 without
FlashAttention or Transformer Engine, as in the original scoring runs, and since FlashAttention is
exact attention this affects only floating-point reduction order. Source:
`results/dna/regen/PROVENANCE.md` ("no Transformer Engine, no FlashAttention - no installable build
exists for this cluster's GLIBC 2.28") and `scripts/dna/regen/README.md`, which also records that the
original runs used no flash-attn either. This is material for the reproducibility package, which is
why it went in rather than only here.

### 1.15 Figure/table caveats now settled (recorded, not yet applied)

From dna-regen, for when Figures 6-8 and Table 16 land:

* **Figure replacement paths.** `notebooks/09_dna_regenerated.ipynb` writes
  `results/dna/regen/figures/fig{6,7,8}_regenerated.{pdf,png}`. These replace
  `figures/{base,gen_vs_real,marginals,cycl_dist}.pdf` (Figure 6, four files -> one),
  `figures/{histogram,histogram2,histogram4}.png` (Figure 7, three -> one) and
  `figures/steering.png` (Figure 8). **Each of those is a structural change to the figure
  environment, not a file swap**: Figure 6 currently has four `\includegraphics` in one float
  (lines 1961-1968) and Figure 7 three `subfigure` blocks with their own sub-captions
  (lines 1976-1999). Tell me whether the regenerated PDFs are single composite figures - if so I
  collapse each float to one `\includegraphics` and fold the sub-captions into the main caption.
* **Figure 7's guidance strengths** are a reconstruction of the *sequence* of strengths, not the
  published settings, which were never recorded. The sub-captions "No guidance / Moderate guidance /
  Strong guidance" must therefore stay qualitative, or say explicitly that the strengths are a
  reconstruction. Queued in [4.7](#47-figures-6-7-8-table-16-table-6-decision-9-dna-regen-agent).
* **Figure 8's caption must not claim the published trajectory**: no producer for it exists at either
  commit, so the regenerated figure is a new example, not the same sample.
* **Table 16 stays an approximation**: no correlation code exists anywhere, so the four Pearson
  values are recomputed from the caption's own description of the statistic. The caption should say
  so.

### 1.16 App. F.8 guide-predictor paragraph, and the $\sigma_t^2/2$ note tightened

**(a) New paragraph** (line 2404), applied verbatim from item 9 of
`docs/provenance/geom/paper_sentences.md` (sourced to
`checkpoints/geom/structure_head/train_args.json` and
`third_party/tfg_flow/dataloader/datasets/geom_fingerprint.py`): a `\paragraph{Fingerprint guide
predictor.}` giving the six-layer E(3)-equivariant graph network (128-dim nodes and edges,
attention, SiLU, masked mean pooling, 1024-logit linear head, 1.06M parameters), the 64-dimensional
atom-type embedding over the nine-class GEOM-DRUGS heavy-atom vocabulary, coordinates in the
generator's frame (A divided by 2.0), BCE against the OpenBabel FP2 path fingerprint with bonds
re-perceived from the 3-D coordinates, 1,172,182 training and 146,401 validation molecules,
selection by validation Tanimoto at threshold 0.5, AdamW with gradient clipping 1.0, the four
warm-started legs (1 GPU at 1e-4; two 3-GPU legs at per-GPU batch 256, global 768, 3e-4; a final
3-GPU leg of 40 epochs at peak 1e-3 with 500 warmup steps and cosine decay), the progression
0.764 -> 0.788 -> 0.817 -> 0.852, and the selected epoch 34 with validation Tanimoto 0.852 and BCE
0.0686. This closes the outstanding request from
[1.8](#18-geom-drugs-table-3-its-caption-app-f8-and-f9-table-21-decision-1-second-round) and makes
the retired "off-the-shelf from TFG-Flow" sentence unrecoverable, as intended.

Note the training-set count here (1,172,182) is the **fingerprint LMDB** count, consistent with
Table 19's "$\approx 1.17$M" and with [3.4](#34-geom-training-set-size); the MFM distillation subset
quoted in [1.4](#14-appendix-f8---mfm-distillation-data-decision-6d) is measured against the
processed dataset (1,141,716). Same corpus, two counts - worth one clause if the authors want the
appendix self-consistent.

**(b) The $\sigma_t^2/2$ note rewritten** (lines 2393-2402). What I wrote in 1.8(e) ended
*"$\mu$ therefore absorbs this time-dependent factor"*, which is **wrong**: a constant cannot
reproduce a time-dependent coefficient. Replaced with the measured statement - relative to $(1-t)/t$
the implemented coefficient is a factor $0.06$ too small at $t=0.1$ and $4.5$ too large at $t=0.9$ -
plus a marker recording that the reported GEOM-DRUGS runs therefore do not implement
\eqref{eq:ode} as stated, that whether the stated coefficient changes the result is being tested,
and that whether the QM9 path makes the same substitution is being checked. The equation and the
results text are left alone, as instructed. See [3.15](#315-the-guided-ode-as-implemented-is-not-the-guided-ode-as-written).

### 1.17 Figures 6, 7 and 8 wired to the regenerated files

Applied now so they resolve when the queued jobs land; the files are written by
`notebooks/09_dna_regenerated.ipynb` to `results/dna/regen/figures/` as both `.pdf` and `.png`.

| figure | line(s) | change |
|---|---|---|
| 6 | 1972 | four `\includegraphics` (`base.pdf`, `gen_vs_real.pdf`, `marginals.pdf`, `cycl_dist.pdf`) and the `\vspace` collapsed into one `figures/fig6_regenerated.pdf` at `width=\textwidth`. The caption keeps its original claims and gains a description of the three rows (position-wise frequencies; differences from real; global marginals and the cyclizability distributions - the difference maps are new and were not described before) and the fact that the generator is the parent-disjoint $L=50$ model |
| 7 | 1983, 1990, 1997, 2001 | the three `subfigure` blocks kept; `histogram4.png`, `histogram2.png`, `histogram.png` -> `fig7_regenerated_{a,b,c}.pdf` at the same `width=\linewidth`. Sub-captions keep "No / Moderate / Strong guidance" and gain the strength: panel (a) is $\gamma=0$ (known: the guided and unguided histograms coincide there), panels (b) and (c) carry `[TODO rerun: $\gamma$]` until job 49276291 lands (`results/dna/regen/fig7/index.csv`). The main caption now says the three strengths are chosen from a sweep because the originally published strengths were never recorded, and that each panel overlays the guided and unguided distributions |
| 8 | 2024 | `steering.png` -> one `figures/fig8_regenerated.pdf` at `width=\linewidth`; caption rewritten to describe the three panels (gradient norm on a log axis, value, guide-predicted endpoint score), to say the sample comes from the parent-disjoint $L=50$ model, and - as dna-regen requires - that it is **a representative example rather than the sample shown in the original submission, for which no producer survives**. Also fixed "The gradient clearly push" -> "The gradients push" |

**Build consequence the authors must act on.** The five new files are referenced as
`figures/fig{6,7,8}_regenerated*.pdf`, matching the `\graphicspath{{figures/}}` convention every
other figure uses. They are produced under `results/dna/regen/figures/`, so **they must be copied
into the paper's `figures/` directory** before the document can produce a final PDF; until then
pdftex renders them as draft boxes, exactly as it already does for the two missing Figure 5 files.
The composite `fig7_regenerated.pdf` also exists if the authors would rather have Figure 7 as a
single float - say so and I will collapse it like Figure 6.

### 1.18 Figure 4 caption replaced with the canonical text, and an invented rationale removed

`results/dna/fig4/paper_sentences.md` is now the canonical source for this caption; the version I
had reconstructed in [1.13](#113-figure-4-and-the-two-33-paragraphs-decision-3-text-errors-independent-of-the-rerun)
was confirmed factually correct with three corrections, all applied.

| # | line(s) | change |
|---|---|---|
| 1 | 659 | **An invented rationale removed.** I had written that the synthetic reward was used *"so that the reward is exactly computable on the conditioning states"*. **Nothing supports that**: both rewards are exactly computable ($f_{\mathrm{cyc}}$ is a small CNN in closed form) and the original script records no motivation. The paragraph is now the canonical replacement, which states the reward and the conditioning states and gives no reason |
| 2 | 631-653, 659 | "outer time" -> "conditioning time" throughout the caption and the paragraph, since the quantity is $t_{\mathrm{cond}}$, the conditioning time of $p_{1|t}(\cdot\mid x_t)$ |
| 3 | 640-641, 659 | "every curve is bounded below by the reference's own sampling error" -> "no curve can be resolved below its own sampling error". The old phrasing claimed a per-curve guarantee; the correct statement is about resolution |
| - | 631-653 | the whole caption replaced with section 3 of the canonical file, which also adds the panel legend ($|\hat V_t(x)-V_t(x)|$, NFE $=1$ for the dMFM, $|f_{\mathrm{cyc}}(x)-y^\star|$ per returned sample on the right), names the \textsc{sde mc} series, and states that each baseline was swept over its own budget and that the dMFM performs one backward pass per guided step |
| - | 642-647 | the two conditioning-state facts requested: states are drawn in order from one seeded stream, so **0--7 are exactly the original protocol** and the rest are extra independent draws for error bars; and since no log of the plotted run survives, the original's default of 8 is **an assumption about the submitted figure, not evidence** |
| - | 663 | the low-NFE marker now names what must settle it: `results/dna/fig4/summary.json` -> `panelB.matched_nfe_comparison` and `panelB.dmfm_wins_at_lowest_nfe` |
| - | 1981 | **new sentence in App. F.4**: the search baselines need a stochastic sampler in order to branch and use the SDE form of the same base generator, \eqref{eq:sde} with $V\equiv0$, which has the same marginals as the ODE sampler; best-of-$N$ and the dMFM use the deterministic sampler. Without it a reader cannot see why beam search, MCTS and Feynman--Kac do not collapse to best-of-$N$ on a deterministic flow. Evidence that the two samplers agree: `results/dna/fig4/calibration.json`. I put it in F.4 (the DNA steering appendix) rather than in 3.3, because the main text is already a page over - move it if you prefer it next to the claim |

Q1 answered in the caption: the `sde_mc` series is named. The legend string
`SDE MC (reference's own finite-$N$ error)` is dna-fig4's to set in the plot; the caption calls it
the \textsc{sde mc} series to match. Q2: 32 conditioning states, held as the `N_COND` placeholder
until `summary.json` exists (the canonical file marks all three placeholders "do not invent").

**Self-audit of my other round-3 additions**, prompted by correction 1. I went back through
everything I supplied rather than sourced and found one more of the same kind, now fixed:

* line 599 (GEOM): I had written that the head-predicted similarity falling means *"the shortfall is
  not an artifact of scoring with directly counted fingerprints instead of the guide"* - an inference
  I supplied. Replaced with the purely descriptive *"so both the directly counted metric and the
  quantity the reward optimises fall"*.
* Checked and kept, each traceable: "both are deterministic point estimates ... independent of $N$"
  (notebook 06 cells 7 and 20, and now the canonical caption); the matched-setup sentence at 663
  (given by the coordinator, and section 4 of the canonical file says to keep it); Figure 6's row
  descriptions and Figure 7's $\gamma=0$ coincidence (dna-regen, relayed); the $0.06$/$4.5$
  coefficient factors (relayed, and arithmetic on $(1-t)/t$); the $1.3\%$/$5.0\%$ B.1 deficits
  (arithmetic on Table 5); the FlashAttention clause (`scripts/dna/regen/README.md` verbatim); the
  guide-predictor paragraph (verbatim from the geom agent); Figure 8's "representative example rather
  than the sample shown in the original submission" (dna-regen). No other unsourced explanatory
  clause remains in anything I have written.

### 1.19 QM9 substituted: Tables 2, 21 and 22, the validity prose, and the steer-search description (decisions 4 and 5)

All numbers read from `python scripts/qm9/report_cells.py`, backed by
`results/qm9/rerun_3seed/rerun_3seed_summary.csv` (three seeds, n=1,000 per seed, target seeds held
out from all tuning). I re-derived every cell from that CSV rather than from the relayed message.

**Table 2** (lines 591-607). Submitted -> 3-seed mean $\pm$ SE:

| property | multiMFM | multiMFM-SS (now Algorithm 3) |
|---|---|---|
| $C_v$ | 0.42 -> $0.44\pm0.004$ | 0.40 -> $0.41\pm0.009$ |
| $\mu$ | 0.84 -> $0.82\pm0.033$ | 0.69 -> $0.53\pm0.016$ |
| $\alpha$ | 1.12 -> $1.17\pm0.017$ | 1.10 -> $1.01\pm0.024$ |
| $\Delta\varepsilon$ | 600 -> $592\pm7.2$ | 462 -> $413\pm3.7$ |
| $\varepsilon_{\rm HOMO}$ | 262 -> $270\pm4.3$ | 252 -> $243\pm5.1$ |
| $\varepsilon_{\rm LUMO}$ | 392 -> $395\pm9.2$ | 316 -> $265\pm6.9$ |

multiMFM is slightly worse on four properties and slightly better on two; **multiMFM-SS improves
markedly on four of six** and is now best in five of the six columns.

**Bold/underline recomputed, and the marks were already inconsistent.** The submitted table defines
no convention and does not follow one: in the $\mu$ column the second-best method was EEGSDE
(0.777) but the underline sat on UniGEM (0.805). I applied one rule - bold the best value in each
column among the steering methods, underline the second best, exclude the three reference rows - and
**stated it in the caption**, which previously explained neither mark. Consequences on rows I did not
otherwise touch: UniGEM loses its underlines on $C_v$, $\mu$, $\alpha$ and $\varepsilon_{\rm LUMO}$
(now third or fourth) and keeps its bold on HOMO; EEGSDE gains an underline on $\mu$ and keeps the
one on $\Delta\varepsilon$. Nothing else changed. If the authors prefer a different convention, say
so and I will reapply it.

**Table 2 caption** also gains the seed count and a `[TODO]` for the open question in item 7 below.

**Table 21** (lines 2480-2485, caption 2470). Unsteered $98.7 \to 98.4\pm0.23$ on all six;
multiMFM $90.3\pm0.20$, $90.6\pm0.13$, $90.5\pm0.15$, $90.8\pm0.49$, $91.5\pm0.84$, $90.9\pm0.70$;
multiMFM-SS $95.8$--$97.8 \to 100.0\pm0.00$ on all six. The $N$ column becomes $1{,}000\times3$ and
the caption says the entries are 3-seed means and that the unsteered arm is shared across properties
within a seed.

**Table 22** (lines 2517-2527, caption 2508). All cells become 3-seed means: unsteered $U$
$67.0 \to 70.8\pm0.51$, steered $U$ and $U_{\rm all}$ per property from the same CSV. **The "Steered
counts" column is gone**: a numerator/denominator pair is a per-run quantity with no mean over seeds,
which is the objection recorded in [4.3](#43-table-21-pb-validity-and-table-22-uniqueness). It is
replaced by **steer-search uniqueness** ($97.5$--$98.3\%$), which the report provides and which the
table previously lacked; the caption records the substitution.

**Prose** (lines 614, 711, 2492): $91.4\% \to 90.8\pm0.09\%$, $98.7\% \to 98.4\pm0.23\%$,
$97.1\% \to 100.0\pm0.00\%$, and F.9's ranges to $90.3$--$91.5\%$ and $100.0\%$. F.9 also now says
1,000 molecules per property **per seed, with three seeds**.

**The steer-search description** (line 614) rewritten around the measured facts:

* "stores on CPU the look-aheads that multiMFM computed" -> it **keeps only the best PB-valid
  look-ahead so far**, checking look-aheads in increasing guide-predicted error so that the first
  PB-valid improvement replaces the incumbent - the same returned molecule at about **0.1%** of the
  validity checks;
* look-aheads are **harvested at every guided step**, stated explicitly because the submitted runs
  had an undocumented time gate (`select_min_t = 0.3`) that the new path does not have - this closes
  ambiguity 1 of [2.](#2-algorithm-3--steer-search-consistency-review-no-edits-made);
* **"without additional generative-model evaluations" is now literally true and measured**: with
  forward hooks, multiMFM and multiMFM-SS both cost $14{,}848$ generative forward passes per
  molecule, a difference of exactly zero, against $14{,}912$--$14{,}937$ for the submitted
  harvesting (`nfe_generative_per_mol_mean` in the same CSV; unguided sampling is 128). The paper now
  quotes the figure.

**"Integrates the same guided ODE as multiMFM" deleted.** It was false - the two are tuned
separately, at guidance strengths (multiMFM / multiMFM-SS) of $2.8/5$, $0.78/3$, $3.4/11$,
$0.17/1$, $0.5/1.5$ and $0.44/3$ for $C_v$, the dipole moment, $\alpha$, $\Delta\varepsilon$, HOMO
and LUMO (verified in each run's `args.mu`). The text now says "the same guided ODE **form**" and
tabulates the twelve values inline. A `[TODO]` records the alternative the authors may prefer:
multiMFM-SS run at multiMFM's own strengths still beats the endpoint on all six properties but is
slightly worse than when tuned, and would let the paragraph claim a single ODE for both. This
resolves [3.5](#35-32-steer-search-integrates-the-same-guided-ode-as-multimfm) and, with the 3-seed
rerun, [3.6](#36-table-2--table-21-ss-columns-mix-replicates-and-mix-them-differently).

**Notation collision worth one authors' decision**: the guidance strength is called `mu` in the code
and I have written it as a bare number here to avoid clashing with $\mu$, the dipole moment in
Table 2. App. F.8 does call the GEOM coefficient $\mu$. One of the two needs renaming if both are to
be printed.

### 1.20 Two authors' decisions applied: the Table 2 MAE population, and the guidance strength renamed

**(a) Table 2 MAE population** (caption, line 435-439). The numbers stay as computed - **over all
1,000 generated molecules per seed** - and the caption is fixed instead. The `[TODO]` is replaced
with: *"Our MAEs are over all generated molecules, not over a validity-filtered subset; PoseBusters
validity is reported separately in Table~\ref{tab:pb-validity}."* No number changed.

The reasoning, recorded because it is the kind of choice a reviewer will probe: switching populations
mid-review would have improved six multiMFM cells ($C_v$ $0.444\to0.415$, $\mu$ $0.818\to0.740$,
$\alpha$ $1.166\to1.113$, $\Delta\varepsilon$ $592\to562$, HOMO $270\to259$, LUMO $395\to372$) while
leaving multiMFM-SS untouched, and would then be **averaging the two rows over different population
sizes** - about 91% of the generated molecules for multiMFM against 100% for steer-search. Reporting
both rows over everything generated keeps them comparable and keeps the validity cost visible in
Table 21 rather than hidden in the denominator. The PB-valid-only values remain available if a
reviewer asks: `results/qm9/rerun_3seed/rerun_3seed_summary.csv`, columns
`oracle_mae_pb_valid_only_{mean,se,s1,s2,s3}`.

**(b) Guidance strength renamed $\mu \to \gamma$.** $\mu$ stays the dipole moment, as in the QM9
literature and in every baseline table the paper reproduces. The guidance strength - the constant
multiplying the estimated value gradient - is now $\gamma$ everywhere it appears:

| line | before | after |
|---|---|---|
| 264 | "a single tuned constant ... crossing $1$ at $t^\ast=1/(1+\mu)$" | "a single tuned constant, the guidance strength $\gamma$ ... at $t^\ast=1/(1+\gamma)$" |
| 1282 | algorithms marker: "a tuned constant in front of the value gradient" | "the tuned guidance strength $\gamma$ ...", plus "Neither algorithm currently takes $\gamma$ as an input" |
| 610 | 3.2: bare numbers | "guidance strengths $\gamma$ ... where $\gamma$ is the constant multiplying the estimated value gradient" |
| 2416 | F.8: "the value gradient scaled by $\mu=0.25$" | "scaled by the guidance strength $\gamma=0.25$ (\texttt{mu} in the released code)" |
| 2420, 2426 | F.8 marker and $\sigma_t^2/2$ note: $\mu=0.5$, "the constant $\mu$" | $\gamma=0.5$, "the constant $\gamma$" |

$\gamma$ is defined at each of its three substantive uses rather than once, so that deleting the
equation marker later cannot orphan the definition. **Code mapping for anyone moving between paper
and code: the paper's $\gamma$ is the CLI/arg `mu`** (`results/qm9/rerun_3seed/runs/*/summary.json`
-> `args.mu`; `results/geom/table3_asdescribed_selection.json` -> `mu`). F.8 says so in one
parenthetical; the rest is here.

**Symbol audit after the rename** - no collisions introduced, two pre-existing ones worth knowing:

* $\gamma$ elsewhere is always decorated: $\gamma_{w,s}$ and $\gamma_{s,w}$ (ESD/LSD constants,
  App. C), $\gamma_\ell$ (adaLN gates), $\bar\gamma$ (the GLASS inner-path parameter, $=0$). The
  bare $\gamma$ is free, and Algorithm 3's commented-out line already called the gradient weighting
  $\gamma$, so the choice matches the authors' own earlier intent. Figure 7's sub-captions already
  used $\gamma$ for the DNA guidance strength, which is now consistent by construction.
* $\mu$ was triple-booked, none of it the guidance strength: the dipole moment (Tables 2, 21, 22),
  the **target property value** in Algorithm 3, $\mu(s)$ in the GLASS denoiser (line 898), and the
  Gaussian centres $\mu_R,\mu_B$ in the toy reward (App. F.7). Algorithm 3's use was the one most
  likely to confuse a reader, since it sat near a property table in which $\mu$ is a property; the
  authors have since had it renamed to $c^\star$, see
  [1.22](#122-algorithm-3s-target-property-value-renamed-mu-to-cstar-consistency-fix-in-two-steps).

**Also confirmed by the authors and left as they are**: the bold/underline rule for Table 2 and the
caption sentence stating it, with the inconsistency of the submitted marks recorded in
[1.19](#119-qm9-substituted-tables-2-21-and-22-the-validity-prose-and-the-steer-search-description-decisions-4-and-5)
(the $\mu$ column underlined UniGEM at 0.805 while EEGSDE at 0.777 was second); and replacing
Table 22's "Steered counts" column with steer-search uniqueness.

**Side effect on length**: the shorter Table 2 caption recovered a page. The main text is now **one**
page longer than the submission, not two - the Impact Statement is back on page 12. The renaming
changes nothing. No trim was made and the equation marker stays.

### 1.21 Table 12 completed, and Figure 1's identity settled

**(a) Table 12's third stage and its loss block** (lines 1705-1735). The three-stage split applied in
[1.3](#13-table-12---toy-training-stages-decision-6c) is confirmed against the checkpoints and
`results/toy/paper_values.json`, and is now complete:

| stage | batch | steps | peak lr | EMA | extra |
|---|---|---|---|---|---|
| base flow model | 4096 | 20,000 | 3e-4 | 0.999 | - |
| MFM, diagonal | 2048 | 10,000 | 3e-4 | 0.999 | - |
| MFM, ESD | 1024 | 10,000 | 1e-4 | 0.995 | inner-gap anneal 5,000 steps (added now) |

Warmup is 500 for all three and sits in the shared *Optimization* block.

**The `Total loss` row is replaced, which closes [3.9](#39-table-12-loss-block).** The printed
$\mathcal{L}=\mathcal{L}_y+\mathcal{L}_c+\mathcal{L}_{\mathrm{ESD}}$ is optimised by **no stage**, and
that is now established from the checkpoints' own `objective` fields rather than inferred:
`endpoint_mse_plus_ce` for the base stage, `diagonal_glass_velocity` (both modality weights 1.0) for
the diagonal stage, `esd_teacher_consistency_plus_diagonal_glass` (diagonal and consistency weights
both 1.0) for the ESD stage. The single row becomes three:

```
Base flow model                   & $\mathcal{L}=\mathcal{L}_y+\mathcal{L}_c$ \\
Diagonal stage                    & GLASS velocity match, both modalities weighted $1.0$ \\
ESD stage                         & $\mathcal{L}=\mathcal{L}_{\mathrm{diag}}+\mathcal{L}_{\mathrm{ESD}}$, both weighted $1.0$ \\
```

The three symbol-definition rows above them ($\mathcal{L}_y$, $\mathcal{L}_c$,
$\mathcal{L}_{\mathrm{ESD}}$) are unchanged. This was the one item I had left proposed-but-unapplied
because it sat outside decision 6; with the table already restructured into stages, leaving it would
have made the table contradict itself.

**Note for the record**: the reproducibility plan's 7.5 said Table 12 "describes the base/diagonal
stage" and gave only the ESD stage as missing. That was wrong in a way that mattered - the printed
block is the **base stage alone**, and the diagonal stage (batch 2048, 10,000 steps) was missing too.
Both are now in the table.

**(b) Figure 1 is plain GLASS, not a Meta Flow Map - resolved fact** (see
[3.17](#317-figure-1-as-submitted-is-glass-posterior-sampling-not-a-meta-flow-map)). The caption
claims a multimodal Meta Flow Map and the submitted panels do not implement one. The authors have
decided the figure **ships as multiMFM**, which will make the caption true, so **the caption is not
rewritten to say GLASS**; instead it carries a marker (line 92) recording that the shipped file does
not yet match the claim, that the toy agent is regenerating it, and listing what to add when the panel
is final: Norway steers colour only from $t=0.8$ at a fixed guidance ratio, both panels use
$N=10{,}000$, the number of flow-map steps, and the file name if it is no longer `norway.pdf`. I held
those three details out of the caption itself rather than writing them now, because they describe the
regenerated panel and the toy agent is still investigating why its multiMFM panel is weaker than the
published GLASS one. The marker is deliberately two lines rather than five: it sits in the page-1
caption, where anything longer costs a main-text page (I wrote it long first, measured that, and cut
it).

**(c) Figure files.** `figures/` now exists in this repo and holds `norway.pdf`, `fig1_glass.pdf`,
`fig1_mfm.pdf`, `multiMFM.png` and `multiMFM_SS.jpeg`, and `scripts/sync_paper_figures.sh` copies
regenerated figures in from `results/`. So Figure 1 does compile here; the file that is missing is
the *final* multiMFM panel, and it arrives the same way as the regenerated DNA figures. Nothing for me
to do. Note that my scratch build resolves `figures/` against the dMFM repo, which has its own
`norway.pdf`, so **the build I verify against is not proof that this repo's `figures/` is complete** -
`scripts/sync_paper_figures.sh` reports what is still missing and is the check to trust.

### 1.22 Algorithm 3's target property value renamed $\mu \to c^\star$ (consistency fix, in two steps)

Applied at five places, all inside Algorithm 3 and its introductory paragraph: the paragraph's
$||\mathcal{R}(y_1,a_1)-c^\star||$, the `REQUIRE` list's "target property value $c^\star$", `Set`
$r(y,a)=-\beta\|\mathcal R(y,a)-c^\star\|^2$, the per-look-ahead error, and the tilt. No occurrence of
either earlier symbol remains in the algorithm; Algorithm 2 never used one.

**Why the target had to stop being $\mu$**: $\mu$ is the dipole moment in Tables 2, 21 and 22, and
Algorithm 3 sat a page away using $\mu$ for something else entirely. With the guidance strength also
renamed ([1.20](#120-two-authors-decisions-applied-the-table-2-mae-population-and-the-guidance-strength-renamed)),
$\mu$ now means only the dipole moment in the experimental sections; the survivors are $\mu(s)$ in the
GLASS denoiser (App. C) and the Gaussian centres $\mu_R,\mu_B$ in the toy reward (App. F.7), both
decorated and in different sections.

**Why $c^\star$ and not $y^\star$**, which was the first instruction and is now reversed: inside
Algorithm 3 the continuous modality is $y$ and the discrete one $a$, so $y^\star$ read as a target
*for the continuous modality* when it is a target for the scalar that $\mathcal R(y,a)$ returns. I
applied $y^\star$ as instructed, flagged the collision, and it was reversed. $c^\star$ is the right
symbol because **the paper already uses $c$ for the target a property map is compared against**: in
3.2's GEOM task the target fingerprint is $c$ and the predictor is $\mathcal E(G)$, exactly parallel
to Algorithm 3's target for $\mathcal R(y,a)$. So $c^\star$ unifies the two conventions instead of
adding a third. $r^\star$ was the other candidate and reads badly, because $r$ is the reward function
in the same line.

**The DNA sections keep $y^\star$** (12 occurrences, 3.1 and App. F.1/F.4): there is no modality split
there, so it is unambiguous, and it is the symbol the text has used all along.

**Reversible.** If the authors prefer something else, the two alternatives I had listed are: accept
$y^\star$ on the grounds that $\mathcal{R}(y,a)-y^\star$ is clear from context, or keep $y^\star$ and
rename the algorithm's modalities - the latter a much larger edit touching Algorithms 2 and 3 and 2.4.
$c^\star$ is the minimal change that removes both collisions.

**Residual, not a local collision**: $c$ is the colour variable in the toy example (App. F.7) and
$\mathcal{L}_c$ is the discrete loss in Table 12. Neither is in Algorithm 3, and $c^\star$ is starred,
so nothing is ambiguous in place.

**Not touched**: the commented-out Algorithm 1 block still uses $\mu$ for the target. It is inert, so
it does not affect the document, but it would be inconsistent if ever uncommented.

### 1.23 Fig 5 / Tables 14-15: the conditioning states corrected, and two markers reframed

**(a) The conditioning states are generated, not held out.** App. F.2 and 3.3 both said "32
held-out conditioning states". They are not held out: `make_probe_states` in
`src/dmfm/experiments/ablate_glass_gradient_mc.py:434` draws Gaussian noise and calls
`flow_to_time(model, cfg, x0, t_end=t_eval, nfe=nfe)` with `t_eval = 0.5`, so the states are
produced by **integrating the base flow from noise to $t=0.5$**. I read the code rather than take
the correction on trust, because the run metadata for these runs records `t_eval` but not the state
construction. Applied in three places: 3.3, App. F.2 (which now says "rather than held out from the
data"), and the Figure 5 caption, which gains the construction.

**App. F.6 keeps "32 held-out states", and that is correct**: Table 18 comes from a different script,
`ablate_dmfm_one_step_gradient_mc.py`, whose `make_heldout_probe_states` really does noise held-out
test sequences (`probe_states: "x_t = beta(t) * one_hot(held_out_test_sequence) + (1 - beta(t)) *
Gaussian_noise"`, `probe_split: "test"`). Two studies, two constructions, one of which the submission
described wrongly - worth knowing before someone "fixes" F.6 to match F.2.

**(b) The F.6 marker reframed.** I had marked F.6's sentence - *"the scale-free analysis ... compares
GLASS Monte Carlo error across lengths, without measuring dMFM approximation error"* - as the one that
contradicts F.2 and the Figure 5 caption, implying it was the problem. **It is the opposite**: it is the
one sentence in the submission that states the limitation correctly, and it stays true after the dMFM
rerun, because the reference is itself a finite-sample estimate however the finite-$N$ side is
sampled. The marker now says so, and points at F.2 and the Figure 5 caption as the two that overclaim
and must be reconciled with it.

**(c) The $N=8$ marker.** The honest placeholder is that the dMFM estimates are *comparable to the
GLASS-only measurement at matched $N$*, which is what the marker now says, with no claim either way
until the full arrays land.

**Nothing to remove.** I checked both files for the earlier expectation that the dMFM curves would be
much worse or would flatten at a nonzero floor: it never entered the paper or this record. The
markers only ever said the cells were awaiting the rerun.

### 1.12 Placeholder inventory

Every marker now in `iclr2027_conferencemod.tex`, what fills it, and who owns it. **The greppable
marker text is the stable key, not the line number**: `grep -n "TODO" iclr2027_conferencemod.tex`
regenerates this list, and the numbers below drift by a few lines with every edit elsewhere in the
file. Verified against the file as it stands: 38 marker lines, 35 `[TODO rerun` and 3 `[TODO`.

**Waiting on reruns** (`[TODO rerun`):

| line(s) | marker text / what is missing | filled from | owner |
|---|---|---|---|
| 92 | Fig 1 caption: "the panels shown are plain GLASS ..." - plus the guidance schedule, $N$, the flow-map step count and the final file name | the regenerated multiMFM panel | toy |
| 333-337 | 3.1 prose: the four oracle-MAE figures and the "to within" figure | `results/dna/c0_guidance/table1_dmfm_n1000_summary.json` | dna-core |
| 365-377 | Table 1: ten cells plus the spanning row "awaiting the paired $n=1000$ dMFM ..." | same | dna-core |
| 612, 2463, 2494 | "interim, see Table 3" - on the 3.2 GEOM passage, the Table 21 GEOM row prose and F.9's sample-size sentence | `results/geom/table3_asdescribed.json` (Slurm 49274410) | geom |
| 616 | Table 3 caption: "our three rows and both validity figures are interim ..." | same | geom |
| 2423 | F.8: "these are the settings of the as-described run ..." | same | geom |
| 2433 | F.8: "the reported GEOM-DRUGS runs therefore do not implement (ODE) as stated" - the coefficient test and the QM9 answer | the coefficient sweep | geom |
| 655, 664, 665 | Fig 4 caption: `N_COND` (32, `summary.json -> panelA.args.n_conditions`), `N_OUT` (100, `panelB.n_outputs`), `N_REP` (3, `panelB_curves.csv -> n_repeats`); the canonical file marks all three "do not invent" | `results/dna/fig4/` | dna-fig4 |
| 676 | 3.3: "this claim is unsupported and must be rewritten to whatever the collected numbers show" | `results/dna/fig4/summary.json` (`panelB.matched_nfe_comparison`, `panelB.dmfm_wins_at_lowest_nfe`) | dna-fig4 |
| 692, 701 | Fig 5 caption and 3.3's four $N=8$ ranges plus the operating-point claim | the dMFM rerun of the scale-free study | scale-free |
| 1908, 1940 | Tables 14 and 15: "every entry in this table is awaiting the dMFM rerun" | same | scale-free |
| 2125 | F.6: the sentence that contradicts F.2 and the Fig 5 caption | same | scale-free |
| 755 | App. B.1 opening: Table 6 does not yet match the parent-disjoint description | the Evo 2 scoring job | dna-regen |
| 810 | Table 6 caption: its 56 cells and the three reference row labels | same | dna-regen |
| 830 | B.1 Evo 2 paragraph: six numbers | same | dna-regen |
| 2025, 2032 | Fig 7 sub-captions (b) and (c): the guidance strength $\gamma$ | `results/dna/regen/fig7/index.csv` (job 49276291) | dna-regen |

**Authors' decisions** (`[TODO`, not blocked on any run):

| line(s) | decision |
|---|---|
| 260, 1282 | whether \eqref{eq:multimodal steer} and Algorithms 2-3 are rewritten to match the constant-$\gamma$ sampler that was run, or the molecular runs are redone with the stated $\sigma_t^2/2$ - see [3.15](#315-the-guided-ode-as-implemented-is-not-the-guided-ode-as-written----the-largest-open-issue) |
| 610 | whether multiMFM-SS keeps its separately tuned guidance strengths or is rerun at multiMFM's |

**Closed since the first inventory**: Tables 2, 4, 5, 21 and 22, the QM9 validity prose, the
steer-search description, the B.1 diversity paragraph, 3.1's clustering sentence and the Table 2 MAE
population all carry final numbers with no markers left.

Numbers that are **present and interim** rather than placeheld, so the text reads as a coherent
negative result now and only the digits change later: the four Table 3 values, the six GEOM figures in
3.2, and the two Table 21 GEOM cells. Each sits under one of the markers above.

Not placeheld, because the current text is not wrong, only provisional: Figures 6, 7 and 8's image
files (wired to names that do not exist yet) and Table 16's four correlation coefficients. Templates
are in [4. Queued edits](#4-queued-edits).

---

## 2. Algorithm 3 / steer-search: consistency review (no edits made)

Read together: Algorithm 3 (`alg:mol mfm steer search`, lines 1242-1281), its introductory
paragraph (line 1241), the 3.2 steer-search passage (line 591) and the Discussion (line 664).
The qm9 agent is changing the code so the description becomes true (decision 5), so the question
here is only whether the three descriptions agree with each other.

**They do agree on the procedure**, and no edit was needed:

* look-aheads are the same `N` posterior samples the gradient estimator already draws
  (`\m X_{0,1}(\epsilon^y_i,\epsilon^a_i; t_k, y_{t_k}, a_{t_k})`, line 1254), so "no additional
  generative-model evaluations" (3.2, Discussion) is consistent with the pseudocode;
* selection is over look-aheads *and* the endpoint, restricted to PB-valid candidates, minimising
  the guide-predicted error, falling back to the endpoint when no candidate is PB-valid - the
  `IF S=\emptyset / ELSIF / ELSE` block (lines 1274-1279) implements exactly the 3.2 and
  Discussion sentences.

**Ambiguities an implementer must resolve, none of them resolvable from the text.** These are for
the qm9 agent (they are the points where the current code and Algorithm 3 could diverge again) and
for the authors (adding one clause each would make Algorithm 3 self-contained):

1. **At which outer steps are look-aheads harvested?** Algorithm 3 harvests inside the `FOR
   k=0..T-1` loop with no gate, i.e. at all T outer steps (T=128 for QM9), giving `T*N = 4096`
   candidates per molecule, each needing an RDKit conversion and a PoseBusters call. The recorded
   SS runs instead gate on outer time: `results/qm9/runs/multimfm_ss/k_*/summary.json` -> `args`
   has `select_min_t = 0.3`, `select_max_t = 1.01` (and `pb_candidates = 0`, `pb_workers = 16`),
   which appears nowhere in the paper. Either state the window in Algorithm 3 or in F.8.
2. **How many candidates are retained?** Algorithm 3 keeps a running arg-min (a single `S`, "the
   storage module is rewritten everytime ..."), whereas 3.2 says steer-search "stores on CPU the
   look-aheads that multiMFM computed" (plural, all of them). The returned molecule is the same
   either way, but the storage cost 3.2 claims is not. The running arg-min is the cheaper and
   better-specified reading; implement that.
3. **`\m X_{0,1}` is not one map evaluation in practice.** Algorithms 2 and 3 write a single
   solution-operator application; F.8 says the QM9 MFM was run with four diagonal integration steps
   (GEOM: 10). F.8 does reconcile this ("few-step posterior sampling rather than a single map
   evaluation"), but Algorithm 3 alone would mislead.
4. **Line 1252, "Evaluate the chosen guidance schedule with its endpoint regularization"**, is
   undefined: neither the guidance-strength schedule nor the "endpoint regularization" appears
   anywhere else in the paper. The recorded runs have `guide_every`, `guide_min_t = 0.05`,
   `guide_max_t = 0.95`, `soft_atom_temperature = 0.25`, `guidance_max_coord_rms`,
   `guidance_max_atom_rms`, `mu` and `reward_scale` - none of which are in the paper. This is the
   single largest reproducibility gap in 3.2 and is worth a short paragraph in F.8.
5. **`\beta` is never given a value** for the QM9 runs, although Algorithm 3 takes it as input and
   defines the reward through it (the only number in the file, `\beta=80`, is inside the
   commented-out ADiT paragraph at line 1763).
6. Cosmetic: `\text{err}(\cdot)` is used both for the stored error `err(S)` and for the endpoint's
   error `err(y_1,a_1)` without being defined; the `ELSIF` (line 1276) says "$\pi(y_1,a_1)$ is PB-valid" where
   line 1257 uses "$\pi(\cdot)=1$"; and the `REQUIRE` list asks for a "differentiable reward
   $r(y,a)$" that line 1 then overwrites.

One factual claim in 3.2 is **contradicted by the recorded SS runs** - see
[3.5](#35-32-steer-search-integrates-the-same-guided-ode-as-multimfm).

---

## 3. Unresolved discrepancies (audit; nothing edited)

Everything in this section is a place where the printed paper and the surviving records disagree, or
where the records cannot support what is printed. **No numbers were changed here.** Where a finding
has since been acted on, the entry says so and links to the edit. Line numbers quoted in this section
refer to the submitted text as it stood before the edits in section 1, so for the entries that were
acted on they will no longer match the current file - the edit's own entry has the current numbers.
Ordered roughly by severity.

### 3.1 Table 1 / 3.1: the DNA guidance runs look like GLASS, not dMFM

**Resolved by the authors (2026-09-29): Table 1 must report a dMFM run.** Table 1 and the 3.1
numbers are now placeholders awaiting dna-core's paired n=1000 dMFM run; see
[1.9](#19-table-1-and-31-reverted-to-placeholders-authors-decision-table-1-must-be-dmfm). The
finding itself is kept below because it is the reason.

`results/dna/PROVENANCE.md`: "The posterior samples behind the guidance gradient come from **GLASS**
on the base flow (`nfe_value` = 4 GLASS steps, mc = 8); `args.json` has no dMFM checkpoint and the
DNA repo's `docs/rebuttal_cyc.md` describes the run as 'GLASS posterior guidance (MC=8)'. Paper 3.1
presents Table 1 as dMFM guidance." The run directories are named `yeast_parent_glass_*`, and
`results/dna/c0_guidance/table1_n1000_summary.json` `_sampler` says the same. This affects the
headline DNA experiment (Table 1, 3.1, Figures 3, 7, 8 and App. B.1, all of which come from the
same sample sets). **Authors must decide**: re-label the experiment, or rerun it with the dMFM
posterior sampler. dna-core has been asked to confirm from the run arguments.

### 3.2 Tables 4-6 and App. B.1 come from the purged `split65k` models, not the parent-disjoint ones

**Resolved.** Tables 4 and 5 and the B.1 diversity paragraph now carry the parent-disjoint numbers
(see [1.14](#114-tables-4-and-5-and-the-app-b1-diversity-paragraph-substituted-final)); Table 6 and
the Evo 2 paragraph are still marked, waiting on the Evo 2 scoring job. The table below compares the
printed split65k values with the **final** parent-disjoint ones, superseding the round-1 recompute
column.

The printed Tables 4, 5 and 6 reproduce the June 2026 `split65k` records **exactly** (verified cell
by cell against `results/dna/june_split65k_recovered/`):

* Table 4: 0.5219/0.7377, 0.5219/0.7373, 0.5184/0.7372, 0.5201/0.7374 = the
  `train_sample`, `split_train_sample`, `split_val_sample`, `split_test_sample` rows of
  `yeast_split_dmfm_c0_diversity_summary_2026-06-09.selected_columns.txt` (all 8 cells match).
* Table 5: all 28 cells match `...head5.csv` (e.g. -1 guided guide score -0.989289, nearest
  split-train 0.39664, nearest test 0.43608, nearest gen 0.41748, pairwise 0.7010282; the two
  ratio columns reproduce as 0.7010282/0.737401 = 0.9507 and 0.41748/0.52010 = 0.8027).
* Table 6: all 56 cells match `yeast_split_dmfm_c0_evo2_summary_2026-06-10.csv` (NLL, both CI
  bounds, `evo2_perplexity_from_mean_nll`, delta-vs-held-out).

But App. B.1 (lines 712, 718) describes them as "Samples from the parent-disjoint L = 50 dMFM
trained on the 230 a-parents, novelty and nearest-neighbour distances ... measured against the 58
held-out test parents", and the row labels are `split_train`/`split_val`/`split_test`. The
`split65k` split is a **random-window** 65,000/8,000/9,368 split, not a parent split, and its
guide was `dinko/C0free_torch.pt`, not the parent-disjoint guide of Table 11. The parent-disjoint
recomputation at n=1000 (`results/dna/diversity/n1000/`, 2026-09-25) gives materially different
numbers, and weakens the paper's direction-specific claim:

| quantity | printed (split65k) | final parent-disjoint (n=1000) |
|---|---|---|
| -1 guided pairwise Hamming | 0.7010 (0.9507x held-out) | 0.7287 (0.9872x) |
| -1 guided nearest-generated | 0.4175 (0.8027x held-out) | 0.4960 (0.9504x) |
| +1 guided pairwise | 0.7372 (0.9997x) | 0.7348 (0.9956x) |
| +1 guided nearest-generated | 0.4913 (0.9447x) | 0.5091 (0.9755x) |
| -1 guided guide score | -0.9893 | -0.9932 |
| +1 guided guide score | 1.0010 | 0.9986 |
| held-out pairwise baseline | 0.7374 | 0.7381 |
| held-out nearest-generated baseline | 0.5201 | 0.5219 |

The qualitative effect of the switch: the negative target's clustering deficit shrinks from 4.9% to
1.3% (pairwise) and from 19.7% to 5.0% (nearest-generated), and the positive target's small deficit
grows slightly. The direction of the B.1 conclusion survives; its magnitude does not.

Only the paper's `train_sample` row coincides (= `full_sample`, same 82,368 windows, same seed-0
draw). **Authors must decide**: keep the split65k numbers and fix App. B.1's description of the
model and splits, or adopt the parent-disjoint n=1000 recomputation - in which case Tables 4-5, the
App. B.1 prose (lines 763, 787), the 3.1 sentence at line 401 ("more local clustering for the
negative target"), and Table 6 (Evo 2 must be rerun; the per-sequence scores behind the printed CIs
are lost) all change. Tracked in [4. Queued edits](#4-queued-edits) as a conditional item.

### 3.3 Table 3 / 3.2 GEOM: the headline numbers do not survive at the stated scale

**Acted on: the GEOM text now reports the negative result** with interim 4,096x3 numbers; see
[1.8](#18-geom-drugs-table-3-its-caption-app-f8-and-f9-table-21-decision-1-second-round).

* `0.227` (Table 3, line 605) is the best of a four-point (lambda, mu) grid at **n=64, one seed**:
  `results/geom/table3_n64_grid.csv`, `capgap_lam0.02_mu0.5` -> `guided_tanimoto = 0.2271617`
  (the other three points give 0.2175, 0.1985, 0.2090; two of the four have negative gain).
* `87.5%` (Table 3 caption line 597, Table 21 line 2402, F.9 line 2407) is that same n=64 run's
  `guided_pb_valid = 0.875` - which is also its **unguided** arm's PB validity
  (`unguided_pb_valid = 0.875`), i.e. the run shows no PB change at all.
* The 4,096 x 3 rerun of the same configuration (`results/geom/table3_n4096_replicates.csv`, job
  47833756) gives guided Tanimoto **0.2241** against unguided 0.2287, a **negative** paired gain of
  -0.00457 +/- 0.00081 (5.7 SEM below zero), and PB 88.0% -> 85.0%.
* `92.0%` base PB (Table 3 caption, Table 21) **appears in no record**:
  `results/geom/table21_geom_row.json` `mismatch` - the unsteered arms of the two runs give 87.5%
  (n=64) and 88.0% (n=4096x3) with the GEOM PB config; 92.0% coincides with the *training-time*
  validation PB on 100 molecules with the `no_strain` config at epochs 26/28/37, while the selected
  epoch-38 checkpoint logged 98.0% there. A fresh measurement of the released checkpoint with the
  GEOM config (job 49256841) gives 88.0%.
* F.9 line 2380 ("The GEOM-Drugs evaluation generated 4,096 molecules; its reported Tanimoto
  similarity and PB validity come from the same run") is therefore not true of the printed
  numbers, which come from n=64.

This is decision 1 (rerun as described and report what comes out); the geom agent owns it.

### 3.4 GEOM training-set size

Table 19 prints "$\approx 1.17$M conformers". The records give two different counts:
`results/geom/table19_geom_hparams.json` -> `training_set_molecules_processed_pt = 1141716`
(what the trainer and the distillation actually iterate) and
`training_set_molecules_fingerprint_lmdb = 1172182`. 1.17M matches the LMDB count, 1.14M the
processed dataset. Cosmetic, but the two numbers should not be used interchangeably; my F.8
sentence (1.4) deliberately quotes Table 19's own figure.

### 3.5 3.2: "Steer-search integrates the same guided ODE as multiMFM"

**Resolved: the sentence is deleted.** See
[1.19](#119-qm9-substituted-tables-2-21-and-22-the-validity-prose-and-the-steer-search-description-decisions-4-and-5).
The twelve guidance strengths are now printed in 3.2, and the alternative (running SS at multiMFM's
strengths) is marked as an authors' choice.

Not true of the recorded runs. Comparing `results/qm9/runs/multimfm/alpha/summary.json` `args` with
`results/qm9/runs/multimfm_ss/k_alpha_a/summary.json` `args`: the guidance coefficient differs
(`mu = 3.4` vs `mu = 11.0`), and the SS runs are separate jobs whose own endpoint is much worse than
the multiMFM row (`final_endpoint` oracle MAE 1.3226 / PB 0.705 for `k_alpha_a`, versus 1.1198 /
0.913 for the multiMFM alpha run). So the multiMFM-SS row of Table 2 is not "the multiMFM row plus a
selection step" but a differently tuned run. The three arms recorded per SS run are
`final_endpoint`, `best_lookahead_raw`, `best_lookahead_pbvalid`; the published number is
`best_lookahead_pbvalid`. Decision 5's rerun should either reuse the multiMFM settings (making the
sentence true) or the sentence should say that the SS runs were tuned separately.

### 3.6 Table 2 / Table 21 SS columns mix replicates, and mix them differently

**Resolved by the 3-seed rerun** (decision 4): every cell of Tables 2, 21 and 22 is now a mean $\pm$
SE over the same three seeds, so no aggregation rule is left to choose. Kept below as the record of
what the submitted tables did.

From `results/qm9/table2.csv` (`ss_published_mae_source`) and the per-run summaries:

| property | Table 2 SS value | source | Table 21 SS validity | source |
|---|---|---|---|---|
| $C_v$ | 0.40 (0.39780) | replicate a | 97.4 | a (0.974) |
| $\mu$ | 0.69 (0.68590/0.68624) | a or b (both round to 0.69) | 96.0 | b (0.960; a = 0.956) |
| $\alpha$ | 1.10 (1.10449) | mean(a, b) | 97.5 | mean(a, b) (0.970, 0.980) |
| $\Delta\varepsilon$ | 462 (462.28) | a (b = 480.12) | 95.8 | a (0.958; b = 0.968) |
| $\varepsilon_{\rm HOMO}$ | 252 (252.20) | mean(a, b) | 97.8 | b (0.978; a = 0.979) |
| $\varepsilon_{\rm LUMO}$ | 316 (315.73) | b only (`k_lumo_a` was killed) | 97.8 | b (0.978) |

So four different aggregation rules appear within one row, and Table 2 and Table 21 disagree about
which replicate a given cell comes from ($\mu$, HOMO). Decision 4 (3 seeds, mean +/- SE) resolves it.

### 3.7 Table 7: training time and GLASS NFE

* Line 1347 says "We trained the dMFM for 10 hours on one 80GB NVIDIA H100 GPU."
  `results/dna/training/wallclock_from_mtimes.json` gives 0.693 h for
  `yeast_parent_a_L50_dmfm_diagonal_h192_b4` (args.json -> last.ckpt), 0.63-0.69 h for the other
  lengths' diagonal runs, and 0.013 h for the L=400 diagonal-only fine-tune; the base generators
  took ~1.0 h each. Even summing the diagonal stage over the four lengths plus the four 4-step
  continuation runs does not obviously reach 10 h; mtimes are a lower bound, so this needs a
  decision rather than an edit (the L=400 4-step run reached step 129,000 of 150,000 and may have
  run much longer than its diagonal stage).
* Table 7's `GLASS NFE & $8$` has no counterpart in the arg dumps. The nearest recorded quantities
  are the *evaluation* settings `nfe_value = 4`, `nfe_traj = 64`, `mc = 8`
  (`results/dna/c0_guidance/*/args.json`) and `nfe_probe = 32`, `nfe_calibration = 32`
  (`results/dna/dmfm_vs_glass/*/run_metadata.json`) - none of them a training-time "GLASS NFE" of 8.
  Either the row means `mc = 8` (the number of posterior samples) and belongs in an inference table,
  or it describes a teacher setting that no surviving record contains.
* `Gap range $[0.75, 1.0]$` matches the diagonal runs (`mfm_consistency_min_gap 0.75`,
  `max_gap 1.0`) but **not** the 4-step continuation runs that produced the reported diagnostics
  (`min_gap 0.0`, `max_gap 0.25`). Like Table 12, Table 7 describes stage 1 only.

### 3.8 Table 9: the L=400 column mixes two runs, and the table describes stage 1

* `Selected step` 49,000 / 39,000 / 45,000 / 45,750 matches the **diagonal** students exactly
  (`results/dna/training/checkpoint_summary.json` -> `dmfm`).
* `Batch size` 64 / 64 / 32 / **16**: the diagonal runs used 64 / 64 / 32 / **8**
  (`dmfm_L400_args.json` -> `batch_size: 8`, `max_steps: 46000`; `results/dna/PROVENANCE.md`
  confirms the L=400 diagonal-only fine-tune ran `--batch_size 8` from step 45,000, so the selected
  step 45,750 was reached with its **last 750 steps at batch 8**). 16 is the 4-step continuation
  run's batch. So the L=400 column takes its step from one run and its batch size from another.
* `Learning rate 1e-4` matches the diagonal runs; the 4-step continuation runs used **2e-5**.
* The models behind Figure 5 and Tables 14, 15, 17 and 18 are the **4-step** students at steps
  15,000 / 149,000 / 34,000 / 129,000 (`checkpoint_summary.json` -> `dmfm_4step`; the checkpoint
  names appear in `results/dna/dmfm_vs_glass/*/L*/metadata.json`). Table 9 never mentions this
  second stage, so no reader can reconstruct the models that produced those results. A second
  "dMFM 4-step continuation" block in Table 9 would fix it; the numbers are all available.
* Base generator: Table 9's "Training steps 200,000" is the configured length
  (`base_L*_args.json` -> `steps: 200000`, `last.pt` at step 200,000), but the checkpoint used
  downstream is `best.pt` at step 50,000 (L=50), 55,000 (L=100), 45,000 (L=200), 45,000 (L=400)
  (`checkpoint_summary.json` -> `base.*.checkpoint_used_downstream`). Worth one clause.

### 3.9 Table 12 loss block -- CLOSED

**Applied**, with the checkpoint evidence, in
[1.21](#121-table-12-completed-and-figure-1s-identity-settled). Original entry:

`Total loss & $\mathcal{L}=\mathcal{L}_y+\mathcal{L}_c+\mathcal{L}_{\mathrm{ESD}}$` (line 1680) is
optimised by no stage: the base stage uses `L_y + L_c` (checkpoint `objective =
endpoint_mse_plus_ce`), the diagonal stage a GLASS velocity match (`objective =
diagonal_glass_velocity`, weights `y_weight = c_weight = 1.0`), the ESD stage
`diagonal_weight * L_diag + consistency_weight * L_ESD` (both 1.0, `objective =
esd_teacher_consistency_plus_diagonal_glass`). Source: the three toy checkpoints and
`src/toy/table12.py` `STAGE_LOSS`. Suggested replacement, if the authors want it (not applied,
because it is outside decision 6):

```
\multicolumn{2}{l}{\emph{Loss / curriculum}} \\
Base flow model                   & $\mathcal{L}=\mathcal{L}_y+\mathcal{L}_c$ \\
Meta flow map, diagonal stage     & GLASS velocity match, weights $1.0/1.0$ \\
Meta flow map, ESD stage          & $\mathcal{L}=\mathcal{L}_{\mathrm{diag}}+\mathcal{L}_{\mathrm{ESD}}$, weights $1.0/1.0$ \\
```

### 3.10 Table 20 QM9 objective row says "Diagonal + ESD" but the selected phase had no ESD term

`results/qm9/table20_mfm_training.json`: `esd_term_active: false`, `objective = "diagonal GLASS
distillation on teacher-generated conditioning states (+ direction and condition-sensitivity
terms); no ESD term"`, `stage: 1`. The same record shows QM9-only loss terms that Table 20 omits
entirely: `diag_direction_weight 5.0`, `diag_condition_jvp_weight 0.1`,
`diag_condition_jvp_direction_weight 0.25`, `rollout_*` weights. If `esd_term_active: false` is
right, the QM9 `Objective` cell and the ESD rows I added in 1.5 should say so instead. **Needs the
qm9 agent's confirmation** before any edit - I left the row as printed.

### 3.11 Minor / wording

* Table 2 caption says "Our rows report MAE over PB-valid molecules" without saying which
  regressor. The reported values are the **held-out oracle**'s MAE
  (`runs/multimfm/*/summary.json` -> `summary[guided].oracle_mae`, e.g. alpha 1.1198 -> 1.12), not
  the guide's (alpha guide MAE 0.7443). Line 1286 does say the oracle is used only for final
  evaluation, so this is consistent, just not stated where the numbers are.
  **The "PB-valid" half of that caption is simply wrong, now confirmed and still open**: the
  submitted numbers and the new 3-seed ones are both over all 1,000 generated molecules per seed, not
  over the PB-valid subset. Per-property PB-valid-only values exist in the same CSV
  (`oracle_mae_pb_valid_only_mean`) and differ materially for multiMFM - e.g. $C_v$ $0.444$ over all
  against $0.415$ PB-valid-only, $\Delta\varepsilon$ $592$ against $562$ - while for multiMFM-SS
  they coincide exactly, since its PB validity is $100\%$. The caption now carries a `[TODO]` naming
  both options; **the authors must pick one**, and picking "PB-valid only" would change six of the
  twelve cells in Table 2.
* Table 18's caption calls the entries "per-coordinate gradient MAE" and the F.6 text "mean
  per-coordinate absolute error"; they are the `mean_mae` column of
  `gradient_mae_summary_all_lengths.csv`, which matches, but the same runs' metadata defines a
  different normalisation (`E_RMS = ||.||_2 / sqrt(4L)`) for the "required MC" analysis. No change
  needed; noting it so nobody "fixes" the caption to RMS.
* Figure 5's two PDFs (`figures/dna_scaling_glass_relative.pdf`,
  `figures/dna_scaling_glass_cosine.pdf`) are **absent** from
  `/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dMFM/figures/`; the document compiles with them
  in draft (empty box) mode. They need to be regenerated or located before a camera-ready build.
* `iclr2027_conference.sty` does not exist in either repo (nor anywhere under the user's home,
  `netscratch` or the backup). See [Compilation](#compilation).
* Table 4's `Pairwise median` column (0.7400 for all four rows) is the only part of Tables 4-6 that
  could not be checked: the recovered `selected_columns.txt` does not include the median. It is
  consistent with `pairwise_gen_hamming_median = 37` of 50 in the same records.

### 3.12 Figure 4 right: the published dMFM curve does not match the only rerun of the same configuration

The one surviving rerun of the submitted dMFM configuration gives target errors 8-36% **higher** than
the curve that was plotted: MC=1 0.131 against 0.099 plotted, MC=2 0.105 against 0.078 (reported by
dna-fig4). Since the submitted run directory
(`workdir/yeast_split_dmfm_steering_benchmark_2026-05-13_12-51-45`) survives empty and the models it
used were purged, the discrepancy cannot be diagnosed - the parent-disjoint rerun will supersede both
curves. Recorded so that nobody reconciles the new figure against the submitted one.

### 3.13 NFE counts stand; only DNA sampling wall-clock was inflated

dna-core found and fixed a double network evaluation in the DNA base-flow step, present in the
original library as well. Its conclusion, accepted by the coordinator: **the published NFE counts are
correct and must not be doubled**, because they count one network evaluation per base-flow step,
which is the algorithmic cost; only wall-clock was affected, and only on base-trajectory steps.

No NFE number was changed, and **no footnote was added, because the paper reports no measured DNA
sampling time**. I grepped the whole file for wall-clock, runtime, throughput and time units: the
only timings anywhere are two *training* times - "We trained the dMFM for 10 hours on one 80GB NVIDIA
H100 GPU" (line 1354, itself an open audit item, [3.7](#37-table-7-training-time-and-glass-nfe)) and
"We trained the multiMFM for 24 hours..." (line 1753). The only wall-clock mention in 3.3 is the
existing caveat that "NFEs alone do not establish equal wall-clock cost", which carries no number.
**Authors: if you add any measured DNA sampling timing at camera-ready, measure it after dna-core's
fix**, or note that the pre-fix numbers overstate the base-sampling component (guided
value-gradient steps dominate, so the effect is small).

### 3.14 The GEOM guide's own head-predicted similarity falls too (retraction recorded)

In [1.8](#18-geom-drugs-table-3-its-caption-app-f8-and-f9-table-21-decision-1-second-round) I
declined the geom agent's proposed clause *"the guide's own (head-predicted) similarity does
improve"* because `results/geom/table3_n4096_shards.csv` showed it falling in all three replicates.
**The geom agent has now retracted that claim**, and the fact is machine-readable in
`results/geom/head_tanimoto_direction.json`: pooled head-predicted Tanimoto $0.23378 \to 0.23079$,
negative in 9 of 11 shards, and negative at **all five** held-out sweep points. It rises only at the
submitted $n=64$ point. So steering does not improve even the quantity it optimises at this scale.

The paper states the measured numbers (0.2338 to 0.2308, line 599) and **offers no mechanism**, which
is what the evidence supports. Do not add one until the coefficient test in
[3.15](#315-the-guided-ode-as-implemented-is-not-the-guided-ode-as-written) and the gradient-sign
diagnostic settle whether this is an implementation defect or a property of the method.

### 3.15 The guided ODE as implemented is not the guided ODE as written -- THE LARGEST OPEN ISSUE

**Confirmed on both datasets (2026-09-29). This is no longer a GEOM-only question.** The QM9 agent
reports that guidance enters its sampler in exactly one place and that neither $\sigma_t$ nor
$(1-t)/t$ appears there: the estimated value gradient is multiplied by the same kind of tuned
constant as GEOM. The ratio of that constant to the paper's coefficient runs $0.009$--$0.58$ at
$t=0.05$ and $3$--$209$ at $t=0.95$ across the six properties, crossing $1$ at $t^\ast=1/(1+\mu)$,
so it is not a reparameterisation of the stated sampler.

**Therefore \eqref{eq:multimodal steer} describes none of the molecular experiments**, and
Algorithms 2 and 3 inherit it through their "Advance ... with ODE" line. This is recorded in the
paper at three places, all marked and none of them a footnote: after
\eqref{eq:multimodal steer} (line 260, with the factor range, the crossing point, and the statement
that either the equation and both algorithms are rewritten to match what was run or the runs are
redone), in the Algorithms appendix (line 1286), and in App. F.8 for the GEOM coefficient
(lines 2411-2420). The numbers stand, because the constant was tuned; what does not stand is the
claim that the reported experiments implement the paper's sampler. A fix is being tested on
GEOM-DRUGS, where the paper's coefficient has been implemented and a held-out sweep is queued; QM9
was deliberately left as run. **Authors' decision, and it is a correctness question about the method
as reported, not a number.**

Original GEOM-only finding, kept for the record:

Equations \eqref{eq:ode} and \eqref{eq:multimodal steer} put $\sigma_t^2/2=(1-t)/t$ in front of the
estimated value gradient. The GEOM code multiplies it by a **constant** $\mu$ instead. The two are
not close: relative to $(1-t)/t$ the implemented coefficient is $0.06\times$ the stated one at
$t=0.1$ and $4.5\times$ at $t=0.9$. **The published GEOM-DRUGS runs therefore did not implement the
sampler the paper states.** This is recorded in App. F.8 (lines 2393-2402) and marked; the equation
and the results text are untouched pending two answers:

1. the geom agent is testing whether implementing the paper's coefficient turns the GEOM result
   positive - if it does, [3.3](#33-table-3--32-geom-the-headline-numbers-do-not-survive-at-the-stated-scale)
   and [3.14](#314-the-geom-guides-own-head-predicted-similarity-falls-too-retraction-recorded) are an
   implementation defect rather than a property of the method;
2. the QM9 agent is being asked whether its guidance path makes the same substitution. **If it does,
   then \eqref{eq:ode} describes none of the molecular experiments as run**, and the equation - not
   just the appendix - has to change. That would also touch Algorithm 2 and Algorithm 3, both of which
   say "Advance ... with ODE \eqref{eq:multimodal steer}", and the QM9 runs' recorded `mu` values
   (3.4 for $\alpha$, 11.0 for the SS runs; see
   [3.5](#35-32-steer-search-integrates-the-same-guided-ode-as-multimfm)) are exactly such constants.

Potentially the most consequential open item in this document: it is a correctness question about the
method as reported, not a number.

### 3.16 Evo 2 scoring path verified against the original (closed)

Table 6's pending numbers rest on the original scoring code. The regenerated Evo 2 module is derived
from the upstream original by a scripted list of anchored edits; under `ast.dump` with docstrings
stripped it is identical to the earlier independent reconstruction, and **13 of its 18 functions are
AST-identical to the original**. Those 13 are the whole scoring path: tokenisation, padding, the
optional BOS, windowing, strand handling, the reduction, what is averaged, the bootstrap, and every
summary column. The five that differ are argument parsing, dataset discovery, baseline-row selection
and CLI wiring, with defaults that reproduce the original behaviour; the 101-line textual difference
flagged earlier was docstrings and comments. No scorer rerun was needed - Table 6 is waiting only on
the Evo 2 scoring job for the parent-disjoint samples. Together with the no-flash-attention note now
in App. B.1 (see [1.14](#114-tables-4-and-5-and-the-app-b1-diversity-paragraph-substituted-final)),
the Evo 2 column is as well provenanced as it can be.

### 3.17 Figure 1 as submitted is GLASS posterior sampling, not a Meta Flow Map

**Established, not suspected.** The reproducibility plan only noted that Figure 1's PDF predated the
toy MFM and so "may" show plain GLASS guidance. The recovered notebook history settles it: it names
the sample files the figure was built from, whose names encode `glass8` (eight GLASS Euler steps), and
the earliest MFM-guidance notebook postdates the figure by five days. So the submitted Figure 1 -
the paper's opening illustration, captioned "posterior guidance using a multimodal Meta Flow Map" -
shows GLASS posterior sampling of the base model.

Resolved by decision: the figure is regenerated with multiMFM, so the caption becomes true rather
than being corrected downwards. Held as a marked placeholder meanwhile, see
[1.21](#121-table-12-completed-and-figure-1s-identity-settled). One open question that is the toy
agent's, not the paper's: its multiMFM panel is currently weaker than the published GLASS one,
whereas a four-step map is expected to match GLASS and does on the DNA side.

### 3.18 Figure 5 and Tables 14-15 mostly measure the reference's own Monte Carlo noise

**Derivable from the published data alone, before any rerun.** The reference is GLASS-2048, itself a
finite Monte Carlo estimate. Over a 256-fold increase in $N$ the published GLASS-versus-GLASS relative
$\ell_1$ falls only $3.66\times$, where independent sampling would give about $16\times$
($\sqrt{256}$), and it is still near $0.77$ at $N=256$. So **the apparent floor in the published
Figure 5 is mostly the reference's own Monte Carlo noise, not a property of the estimator being
measured.**

Three consequences, which change what the experiment can be said to measure. They are recorded here
now and go into App. F.2/F.3 once the full dMFM arrays confirm them:

1. the experiment **cannot separate** dMFM approximation error from the reference's Monte Carlo
   error;
2. what it **can** answer is whether substituting dMFM for GLASS at finite $N$ degrades the gradient
   estimate. Preliminary answer: no, not materially;
3. measuring the dMFM approximation error itself would need a reference converged in $N$ - a much
   larger reference - or common random numbers shared between the two samplers.

This **strengthens** App. F.6's sentence rather than contradicting it, which is why that marker was
reframed in [1.23](#123-fig-5--tables-14-15-the-conditioning-states-corrected-and-two-markers-reframed).
It is the one sentence in the submission that got this right.

**Why the earlier "nonzero floor" expectation was withdrawn.** The expectation was that the dMFM
curves must flatten, because dMFM carries an approximation error that does not shrink with $N$. A
controlled 4-state slice (`results/dna/scale_free_mc/dmfm/preliminary/`) overturns it: dMFM tracks
GLASS within a ratio of $0.85$--$1.15$ on pooled relative $\ell_1$ and within $0.06$ on cosine at
matched $N$, and it keeps decaying out to $N=256$ - by $4.32\times$ from $N=1$, against GLASS's
$3.66\times$. No floor is visible in the measured range. The slice is trustworthy despite its size
because its cell function **reproduces the published 32-state cells to 5e-7**, so the recomputed
GLASS column is apples-to-apples with the dMFM column; and the CPU-versus-GPU precision difference is
$0.006$ in relative $\ell_1$ against ratios that differ from 1 by 10-15%.

**Its numbers must not go in the paper**: 4 of 32 states, `motif` reward only, $L=50$ only, and it is
marked not-for-paper in its own README. Nothing from it is quoted in the tex - the $N=8$ marker says
only "comparable to the GLASS-only measurement at matched $N$".

---

## 4. Queued edits (waiting on reruns)

Each row is written so that only the numbers need substituting. `[X]` marks a value to fill.
Line numbers are current (post-1.x edits) and will shift by at most a line or two as these land.
**Any new table or figure must be added after the last existing one of its kind**, or every
later number shifts and the audit in section 5 stops matching.

### 4.1-4.3 QM9 (decisions 4 and 5; qm9 agent) - DONE

Table 2, Table 21's QM9 rows, Table 22, and the 3.2 / Discussion / F.9 validity prose are all
substituted with the final 3-seed numbers; see
[1.19](#119-qm9-substituted-tables-2-21-and-22-the-validity-prose-and-the-steer-search-description-decisions-4-and-5).
Nothing is queued for QM9 any more. Two things the authors still have to choose, both marked in the
tex:

* whether Table 2 reports MAE over all 1,000 molecules (as printed, and as submitted) or over the
  PB-valid subset (as the caption claims) - see
  [3.11](#311-minor--wording);
* whether multiMFM-SS keeps its separately tuned guidance strengths or is rerun at multiMFM's, which
  would let 3.2 claim a single guided ODE for both.

The consistency identity is preserved: $90.8$ is the mean of the new Table 21 multiMFM column
($90.3, 90.6, 90.5, 90.8, 91.5, 90.9$) and $100.0$ the mean of its SS column.

### 4.4 Table 3, its caption, and 3.2 (decision 1; geom agent)

**All applied with interim numbers** - see
[1.8](#18-geom-drugs-table-3-its-caption-app-f8-and-f9-table-21-decision-1-second-round). What is
left is a pure digit substitution once `results/geom/table3_asdescribed.json` exists (Slurm 49274410,
collected by `bash scripts/geom/collect_table3_asdescribed.sh`):

| line | now (interim) | replace with |
|---|---|---|
| 611 | `Our base generator (unguided) & $0.2287 \pm 0.0005$ \\` | `base_generator_unguided.tanimoto_mean ± tanimoto_se` |
| 612 | `multiMFM & $0.2241 \pm 0.0001$ \\` | `multimfm.tanimoto_mean ± tanimoto_se` (re-add `\textbf{}` only if it becomes the column best) |
| 603 | caption `$85.0\%$` / `$88.0\%$` | `multimfm.pb_valid_mean` / `base_generator_unguided.pb_valid_mean`; then delete the interim marker |
| 599 | `0.2287`, `-0.0046\pm0.0008`, `88.0\%`, `85.0\%`, `0.2338`, `0.2308` | the same objects plus `paired_gain.mean ± paired_gain.se` and the head-predicted pair; **re-decide the sign of the claim** (see the warning in 1.8) and delete the interim marker |
| 2422 | Table 21 `$4{,}096\times3$ & $88.0$ & $85.0$` | same two means; keep `\times3` only if the rerun is three seeds |
| 2427 | `$85.0\%$`, `$88.0\%$` | same; delete the interim marker |
| 2380-2392 | `\lambda=0.01`, `\mu=0.25`, "every second outer step", `t\in[0.1,0.9]`, "512 held-out" | already the as-described selection (`table3_asdescribed_selection.json`); only delete the interim marker |

Still outstanding for the geom agent: the structure head's **training details** (dataset split,
epochs, held-out fingerprint accuracy - `checkpoints/geom/structure_head/train_args.json` and the
val Tanimoto 0.8523 at epoch 34 in `docs/provenance/geom/checkpoints.md`) for a short F.8 paragraph;
and the head-predicted-similarity re-check flagged in 1.8.

### 4.5 Figure 4 (decision 3; dna-fig4 agent)

The caption and both 3.3 paragraphs are **already rewritten**, first in
[1.13](#113-figure-4-and-the-two-33-paragraphs-decision-3-text-errors-independent-of-the-rerun) and
then replaced with the canonical text in
[1.18](#118-figure-4-caption-replaced-with-the-canonical-text-and-an-invented-rationale-removed).
`results/dna/fig4/paper_sentences.md` is the canonical source; its section 5 maps every claim to the
file that settles it, and its section 6 carries notes for other agents (a non-purged clone of
`tullebulle/DNA-MFM@187fe7b` on holylabs; `scripts/score_yeast_c0_evo2.py` surviving as source there;
`docs/REPRODUCIBILITY_PLAN.md` 2 still calling the Fig 4 left producer "Lost"). What is left:

* Lines 622 `figures/MC_value.pdf` and 628 `figures/nfe_benchmark.pdf`: the PDFs are replaced in
  place, so no tex change is needed **unless the new figures are written under new names** - send me
  the names if so.
* Lines 642, 651, 652: substitute `N_COND`, `N_OUT`, `N_REP`. The pilot's choices
  (`results/dna/fig4/pilot_choice.json`) are worth a small appendix table - say so and I will add one.
* Line 663: the `dMFM achieves low target error at low NFEs` marker is deleted once the right panel
  lands, and the claim is rewritten to whatever `summary.json -> panelB.matched_nfe_comparison` and
  `panelB.dmfm_wins_at_lowest_nfe` show, not merely re-checked.
* If the baselines are cited as implementations of \citet{didi2026binder}, note that
  `didi2026binder` is one of the 14 missing bibliography keys (see [Compilation](#compilation)).

### 4.6 Figure 1 (decision 7; toy agent)

Line 91 `figures/norway.pdf`, caption lines 92-93. If the regenerated figure is multiMFM rather
than plain GLASS guidance the caption is already correct ("using a multimodal Meta Flow Map"); if
the panels change (a four-panel version `norway_base_smiley_four_panel_tweak.pdf` exists in the
figures directory), the caption needs the new panel list. Template:
`\caption{Examples of posterior guidance using a multimodal Meta Flow Map. $p_0(x)$ is the noise
distribution, and $p_1(x)$ is the base model distribution. [panel description]. The Norway example
steers only color coordinates, while the Smiley example steers both position and color.}`

### 4.7 Figures 6, 7, 8, Table 16, Table 6 (decision 9; dna-regen agent)

Tables 4 and 5 and the B.1 diversity prose are **done**
([1.14](#114-tables-4-and-5-and-the-app-b1-diversity-paragraph-substituted-final)). What remains,
with the caveats dna-regen has now settled folded in
([1.15](#115-figuretable-caveats-now-settled-recorded-not-yet-applied)):

| item | lines | what changes |
|---|---|---|
| Fig 6 | 1970-1982 (`base.pdf`, `gen_vs_real.pdf`, `marginals.pdf`, `cycl_dist.pdf`) | replaced by the single `results/dna/regen/figures/fig6_regenerated.pdf`, so the float collapses from four `\includegraphics` to one. Caption gains "parent-disjoint", since the printed figure's data are lost |
| Table 16 | 2036-2051 (data rows 2042-2045) | four Pearson $r$ / $p$ pairs `[r] & $[p]$ \\` (now 0.9991/9.5e-4, 0.1813/0.819, 0.9863/6.02e-157, 0.0264/0.711). **Caption must gain one clause**: no correlation code exists at either commit, so the values are recomputed from the caption's own description of the statistic and are an approximation of the printed ones |
| Fig 7 | 1985-2009 (`histogram4.png`, `histogram2.png`, `histogram.png`) | replaced by one `fig7_regenerated.pdf`, so the three `subfigure` blocks collapse into one image. The sub-captions "No / Moderate / Strong guidance" fold into the main caption and **must stay qualitative or say they are a reconstruction**: the published guidance strengths were never recorded, only the sequence of strengths is reconstructed |
| Fig 8 | 2028-2033 (`steering.png`) | replaced by `fig8_regenerated.pdf`. **The caption must not claim the published trajectory** - no producer for it exists at either commit, so this is a new example rather than the same sample. Current wording ("Demonstration of guidance using dMFM for one DNA generation sample") is already generic enough; only "We compare a guided and an unguided sample" needs checking against what the new figure shows |
| Table 6 | 791-798 (caption 785) | eight rows `& [n] & 49 & \texttt{forward\_rc\_mean} & [nll] [[lo], [hi]] & [ppl] & [delta] \\`, plus renaming the three reference rows to `a_train_sample` / `a_val_sample` / `test_sample`; waiting only on the Evo 2 scoring job |
| B.1 Evo 2 prose | 805 | six `[TODO rerun]` numbers, from the same job |

dna-regen: confirm that each regenerated PDF is a **single composite image** and I will collapse the
three floats accordingly; if they are still separate panels, send the filenames and I will swap them
one for one instead.

### 4.7b App. F.2/F.3: what the scale-free experiment measures (scale-free agent)

Once the full dMFM arrays land, App. F.2 and F.3 need the three consequences in
[3.18](#318-figure-5-and-tables-14-15-mostly-measure-the-references-own-monte-carlo-noise) stated
explicitly, since the tables' entries cannot be read as dMFM approximation error. Template for the
end of App. F.2, to be filled from the arrays:

> Because the reference is itself a finite-sample estimate, this comparison does not separate the
> posterior sampler's approximation error from the reference's own Monte Carlo error: over the
> $[X]$-fold range of $N$ the reference's error falls by only $[Y]$, against the $[Z]$ expected of
> independent sampling. What it measures is how far substituting the dMFM for GLASS at matched $N$
> moves the gradient estimate: the two differ by $[R]$ on pooled relative
> $\ell_1$ and $[C]$ on cosine. Isolating the dMFM approximation error would require a reference
> converged in $N$, or common random numbers shared between the two samplers.

The three numbers derivable from the published data today are $256$-fold, $3.66\times$ and about
$16\times$; the difference ranges must come from the 32-state arrays, not from the 4-state slice.

**Neutralised 2026-09-29 (coordinator).** This passage previously read "...degrades the gradient
estimate, *and it does not: the two agree within* $[R]$...", i.e. a fill-in-the-blank that asserted
its own conclusion before the arrays had landed. It now states the comparison without pre-judging
the direction, and whoever fills $[R]$ and $[C]$ must write the direction from the measured numbers.
This is not a stylistic preference: the analogous parity reading on the *steering* axis was
contradicted the same evening, when the n=800 Table 1 rerun put dMFM 0.040 +- 0.007 behind GLASS
after an n=64 slice had looked like a tie. See `paper_text_dmfm.md` edit E7 for the same fix applied
to the paper text, and its revision note for the steering result.

### 4.8 QM9 baseline sweep (decision 8)

No text exists for it yet, and **two constraints now apply if the lead includes it**:

* the sweep is fixed with every point measured, and **it reverses the direction of the old figure**,
  so **no "less compute" or "fewer NFEs" claim can be made from it**. I checked the current text and
  no such claim exists to remove: the only compute-flavoured sentence is 3.2's "training the
  multiMFM is only for increased sampling efficiency", which is about MFM versus GLASS posterior
  sampling, not about the baselines. So this is a constraint on new text, not a deletion;
* the qm9 agent recommends it go in the **appendix, not the main text** - which also suits the
  main-text page situation (see [Compilation](#compilation)).

Shape: a new appendix table would renumber Tables 3-22, so if it is a table it should go **last**, as
Table 23, after `tab:qm9-uniqueness`. Tell me the columns you want and I will add it there; the data
is `results/qm9/baselines/baseline_runs.csv`.

---

## 5. Audit: printed numbers that were verified against `results/`

Method: each table's cells were compared with the recorded run outputs (not with
`paper_values.json`, which is itself a transcription of the PDF; both agree everywhere I checked,
so the transcriptions are sound). "exact" = the recorded value rounds to the printed value at the
printed precision.

| table / claim | cells | verdict | source |
|---|---|---|---|
| Table 1 (as submitted, n=32) | 10/10 | exact - it is the n=32 pilot, confirming the plan's finding | `c0_guidance/pilot_n32/target_*/oracle/oracle_summary.csv` |
| Table 1 (as now edited, n=1000) | 10/10 | exact | `c0_guidance/table1_n1000_summary.json` |
| Fig 3 caption (58 parents, 8,294 windows, r 0.88/0.87, MAE 0.159/0.162) | 6/6 | exact | `c0_regressors/{guide,oracle}/metadata.json` |
| Table 2, our two rows (submitted) | 6/6 multiMFM | exact (oracle MAE); **both rows have since been replaced by the 3-seed rerun**, see 1.19 | `runs/multimfm/*/summary.json` |
| Table 2, multiMFM-SS row (submitted) | 6/6 | exact, but mixed replicates (3.6); **replaced**, see 1.19 | `runs/multimfm_ss/*/summary.json`, `table2.csv` |
| Table 2, 14 published baseline rows | not checked | reproduced from TFG-Flow Table 1 by the paper's own statement; outside `results/` | - |
| Table 3 | 1/1 (0.227) | exact, but n=64 best-of-grid; see 3.3 | `table3_n64_grid.csv` |
| Table 4 | 8/8 (+4 median cells unverifiable) | exact vs the June split65k records; see 3.2 | `june_split65k_recovered/...selected_columns.txt` |
| Table 5 | 28/28 | exact vs the same records (ratios recomputed) | `...head5.csv` |
| Table 6 | 56/56 | exact | `...evo2_summary_2026-06-10.csv` |
| App. B.1 prose (0.7372, 0.7374, 0.9997, 0.4913, 0.5201, 0.7010, 0.9507, 0.4175, 0.8027, 1.3529, 1.3454, 0.0075, 1.2350, -0.1104) | 14/14 | exact vs Tables 5-6 | same |
| Table 7 parameter counts (3,420,484 / 2,912,836 / 1,536 / 74,112 / 432,000) | 5/5 | exact - the four groups sum correctly from the per-module counts | `training/checkpoint_summary.json` |
| Table 7 architecture (L 50, K 4, d 192, 4 blocks, 6 heads, MLP 4.0, cond 192, softcap 50.0) | 8/8 | exact | `training/base_L50_args.json`, `dmfm_L50_args.json` |
| Table 7 `Gap range [0.75, 1.0]`, `GLASS NFE 8`, 10 h | - | see 3.7 | - |
| Table 9 base rows (200,000 steps; batch 512/256/128/64; lr 5e-5) | 9/9 | exact | `training/base_L*_args.json` |
| Table 9 dMFM rows | 4/4 selected steps exact; batch 3/4; lr stage-1 only | see 3.8 | `checkpoint_summary.json`, `dmfm_L*_args.json` |
| Table 9 windows per parent (143/135/121/93) | 4/4 | exact, and `floor((1044-L)/7)+1` checks out | `training/parent_split_seed0.json` |
| Table 10 (parent split) | 30/30 | exact | `training/parent_split_seed0.json` |
| Table 11 | 8/8 | exact | `c0_regressors/*/metadata.json` |
| Table 12 | see 1.3 | base-stage values exact; two stages were missing | toy checkpoints |
| Table 13 (exact-reward improvements, 24 numbers) | not re-derived | `exact_rewards/*/L*/reward_attainment_summary.csv` exist and carry the same protocol metadata (100 pairs, bootstrap 2000, seed 20260731); left to notebook 04's `compare()` | - |
| Table 14 (18 cells) | 18/18 | exact, they are the N in {8,128} rows of Table 15 | `scale_free_mc/*/gradient_scale_free_summary_all_lengths.csv` |
| Table 15 (108 cells) | not cell-by-cell | Table 14's subset checked; full check belongs to notebook 05 | - |
| 3.3 text (N=8 relative errors 1.40-2.55, cosines 0.18-0.32) | 4/4 | exact - min/max of the N=8 rows of Table 15 | Table 15 |
| Table 16 | 4/4 pairs | printed values match the transcription, but the underlying data are lost (decision 9) | - |
| Table 17 (32 cells) | not re-derived | `posterior_diversity/posterior_diversity_summary.csv` exists (32 sources x 16 futures, job 35378081); left to notebook 05 | - |
| **Table 18 (48 cells)** | **48/48** | **exact** - every entry equals `mean_mae` at 4 dp | `dmfm_vs_glass/*/gradient_mae_summary_all_lengths.csv` |
| Table 19 QM9 column (24 rows) | 24/24 | exact | `results/qm9/table19_base_training.json` |
| Table 19 GEOM column (24 rows) | 23/24 | exact except the training-set size; see 3.4 | `results/geom/table19_geom_hparams.json` |
| Table 20 QM9 column | 19/20 | exact except `Objective`; see 3.10 | `results/qm9/table20_mfm_training.json` |
| Table 20 GEOM column | 20/20 | exact (and four settings were missing; see 1.5) | `results/geom/table20_geom_hparams.json` |
| Table 21 QM9 (18 cells, submitted) | 18/18 | exact: unsteered 98.7 six times, multiMFM column = each run's `pb_pb_valid`, SS column = `best_lookahead_pbvalid`'s; **replaced**, see 1.19 | `runs/multimfm*/*/summary.json` |
| Table 21 GEOM row | 1/3 | 87.5 exact (n=64); 92.0 unsupported; N=4,096 wrong for the printed numbers; see 3.3 | `table21_geom_row.json` |
| Table 22 (24 cells, submitted) | 24/24 | exact, including the `945/989`-style counts and `U_all`; **replaced**, see 1.19 | `runs/multimfm/*/summary.json` |
| 3.2 / Discussion 91.4 / 98.7 / 97.1 | 3/3 | exact as means of Table 21's columns; **replaced by 90.8 / 98.4 / 100.0**, which are the means of the new columns | Table 21 |
| F.9 "QM9 used 1,000 molecules per property" | yes | `num_samples: 1000` in all six runs | `runs/multimfm/*/summary.json` |
| F.8 "four diagonal integration steps and 32 posterior samples within a 128-step outer ODE" | 3/3 | exact (`value_glass_steps 4`, `value_samples 32`, `sample_steps 128`) | `runs/multimfm/*/summary.json` |
| F.8 GEOM "10 flow-map integration steps and four posterior samples within a 100-step outer ODE" | 3/3 | exact (`mfm_steps 10`, `value_samples 4`, `sample_steps 100`) | `results/geom/table3_n64_grid.csv` |
| 3.1 "guidance improves around 95% of matched pairs" / F.1 "93-98%" | not re-derived | `exact_rewards/*/L*/reward_attainment_summary.csv` hold the per-setting fractions; notebook 04 | - |
| 3.1 "576 contiguous 1,044-bp parents, stride 7, overlap 43 bp" | yes | `parent_split_seed0.json` (`n_parents 576`, `parent_length 1044`, `stride 7`) | - |

---

## Compilation

`iclr2027_conference.sty` is **not present** in this repo, in
`/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dMFM/` (which has `neurips_2026.sty`,
`mfm3_preprint.cls` and `icml2026/icml2026.sty`), or anywhere under the user's home / netscratch /
the holylabs backup. As committed, `iclr2027_conferencemod.tex` therefore cannot be built by anyone
who does not already have that file; it should be added to the repo (or the ICLR 2027 kit
downloaded) before the next submission build.

For checking these edits the document was compiled in a scratch directory with
`iclr2026_conference.sty` (found at `~/grokking-workspace/paper_norm/`) renamed to
`iclr2027_conference.sty`, plus `fancyhdr.sty` from the dMFM repo's `icml2026/`, and with
`bibliography.bib`, `checklist.tex` and `figures/` symlinked from the dMFM repo:

```
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Result after every edit in section 1: **41 pages, no overfull boxes**, and seven "file not found"
errors, all of them figure files that pdftex renders as draft boxes:

* 2 pre-existing - `figures/dna_scaling_glass_{relative,cosine}.pdf` for Figure 5 (see 3.11);
* 5 introduced deliberately in [1.17](#117-figures-6-7-and-8-wired-to-the-regenerated-files) -
  `figures/fig6_regenerated.pdf`, `figures/fig7_regenerated_{a,b,c}.pdf`,
  `figures/fig8_regenerated.pdf`, which dna-regen's jobs write to
  `results/dna/regen/figures/` and which **must be copied into the paper's `figures/` directory**.

The page count is inflated by the draft boxes as well as by the markers: with all files present and
the markers deleted it will be well under 40. Table 21's `\tabcolsep` was reduced 4.5pt -> 3.0pt
because the wider `$4{,}096\times3$` cell made it 12.3pt overfull; that is the only formatting
change made.

**These three are pre-existing build blockers for the authors to fix, not artifacts of this pass**:
the missing `iclr2027_conference.sty`, the two missing Figure 5 PDFs, and the missing bibliography
keys below. None of them is in any agent's remit.

Two further pre-existing build problems, unrelated to these edits (both also present before them):

* 14 **undefined citations**, i.e. keys cited by `iclr2027_conferencemod.tex` that the dMFM repo's
  `bibliography.bib` (88 entries) does not contain: `axelrod2022geom`, `bansal2024universal`,
  `campbell2024generative`, `chung2024diffusionposteriorsamplinggeneral`, `didi2026binder`,
  `feng2025unigem`, `fromer2024pareto`, `he2024manifold`, `sarkar2026designing`, `song2023loss`,
  `vonessen2026tabasco`, `ye2024tfg`, `yu2023freedom`, `zeni2025generative`. The modified tex has
  moved ahead of the bibliography it is built against; the authors' current `.bib` needs to come
  with it. (No undefined `\ref`s.)

Pagination, with the caveat that **these counts are not reliable yet**: seven figures render as
fixed-size draft boxes rather than at their real aspect ratios, which moves floats and therefore page
breaks, and the style file is a 2026 substitute. The untouched document builds to 35 pages and the
current one to 41, of which about four pages are markers (all in the appendix) and the rest draft-box
slack.

**The main text has grown by one to two pages** - the Impact Statement has moved from page 11 to 12
or 13 depending on float placement - and that growth is substantive rather than scaffolding: the GEOM
negative-result passage, the Figure 4 caption, the two 3.3 sentences, the rewritten steer-search
paragraph with its twelve guidance strengths, and the marker under \eqref{eq:multimodal steer}. Two
measurements worth keeping: shortening the Table 2 caption recovered a page, and the two-line Figure 1
marker costs one, because both sit early and cascade through tight float placement.

The authors have been asked to decide on a trim and have not chosen one. Candidates, in the order I
would cut them: the twelve guidance strengths (they belong in F.8), the GEOM passage's detail (to
F.9), and the Figure 4 caption. The equation marker should be the last thing cut, since it is the
largest open correctness issue. The honest measurement can only be made once the figures are present
and the real `iclr2027_conference.sty` is in the repo.
**All 21 table labels (22 tables; Table 8 has no label), all 8 figure numbers, the 3 algorithms and
the appendix letters are unchanged**, verified after every round by diffing the `\newlabel` entries
in `paper.aux`.

---

## 6. APPENDED 2026-09-29 by the paper-text agent: proposed DNA prose for the dMFM reruns

**Nothing above this heading was altered, and nothing in `iclr2027_conferencemod.tex` was edited by
this round.** The authors are mid-submission and must control what lands in the manuscript.

The deliverable is **`docs/paper_text_dmfm.md`**: 16 proposed edits (E1-E16), each with the
section/table/figure, a quoted anchor of the current tex text, the proposed replacement and a
one-line justification citing the evidence file; then a placeholder table mapping every
`[[PLACEHOLDER]]` to the results file that fills it; then a sweep of every DNA location checked with
a verdict (12 need no change) and a list of ten claims that cannot be made even after the reruns.

Why a new file rather than more rows in §1 and §4 of this document: §1 records edits *applied* and
§4 records edits *queued with only numbers missing*. These are neither -- they are prose the authors
have not yet seen, produced for a decision (§7 item 8) that changed what the paper must say. Merging
them into §1 would misrepresent them as landed.

### What this round found that changes existing entries here

Three findings contradict entries above. They are recorded here so nobody applies the older entry;
the entries themselves are left as written, per the ownership rule.

1. **§1.14's Tables 4-5 and B.1 diversity paragraph are not final.** They were recomputed by
   `scripts/dna/tables4_5_c0_diversity.sh` on `results/dna/c0_guidance/n1000/`, whose
   `table1_n1000_summary.json` `_sampler` field reads "GLASS-posterior guidance on the
   parent-disjoint L=50 base flow". REPRODUCIBILITY_PLAN.md §7 item 8 requires the generated-sequence
   sets behind Tables 4-6 to be dMFM. So Table 5's four generated rows, the nine numbers in the B.1
   diversity paragraph and §3.1's clustering sentence must be re-opened (E2, E10, E11).
   **Table 4 is genuinely final**: `analyze_c0_diversity.summarize_reference_baseline` takes only
   real reference windows, so it is sampler-independent.
2. **The queued Figure 7 and Figure 8 regenerations are GLASS, not dMFM.**
   `scripts/dna/regen/fig7_guidance_sweep.sbatch` calls `dmfm.experiments.sample_c0_guidance`, whose
   posterior is `glass_integrate_diff` (`:246`), and `fig8_guidance_trace.sbatch` calls
   `dmfm.regen.guidance_trace`, which builds `api.glass_posterior_fn` (`:92`). Neither loads a
   student. §4.7's Figure 7 and Figure 8 rows therefore do not yet satisfy §7 item 8, and App. F.4's
   "tilting the dMFM in Figure 7" plus Figure 8's "guidance using dMFM" stay false (E14, E15).
3. **§3.15's coefficient finding is not molecular-only.** The DNA sampler's guidance coefficient is
   `min(8 sigma_t^2, 10)` against \eqref{eq:ode}'s `sigma_t^2/2`, i.e. 10x to 16x over the guidance
   window `[0.5, 0.95]` and exactly 16x wherever the cap does not bind (t >= 0.615); and with the
   scale-free rule the step's magnitude no longer depends on `||grad V||` at all. E12 states this in
   App. F.4 and offers one clause for the marker at line 260.

### Also worth recording

* **Fig 5 / Tables 14-15 use a different dMFM configuration from Table 1.** The rerun defaults to
  `--student dmfm --dmfm_steps 1` (`src/dmfm/scalefree/run_shard.py`), i.e. the *diagonal* student as
  a *single* map, while Table 1's rerun uses the 4-step ESD student at four jumps
  (`outputs/dna/table1_dmfm4_norm_n1000/*/shards/*/args.json`). Both satisfy §7 item 8's convention 1,
  which pairs the artifact to the jump count; but "the dMFM sampler" no longer identifies one
  configuration, so E3, E6, E7 and E8 name it in each place.
* **§1.23(b)'s reframing of the F.6 marker was right, and it is still not the whole fix.** After the
  rerun, F.6's sentence "the scale-free analysis ... compares GLASS Monte Carlo error across lengths"
  becomes wrong in the *other* direction, because the finite-N side is then dMFM. E9 rewrites it so
  the limitation is stated through the *reference* being a finite sample, which is the form that
  survives the rerun.
* **The Table 1 reruns run a matched GLASS arm.** Jobs 49352242/49357778 (`t1_dmfm4_norm`) and
  49352244/49357782 (`t1_glass_norm`) both carry `grad_normalize True`, `guidance_frac 8.0`,
  `coeff_cap 10.0`, `grad_clip 10.0`, `mc 8`, window `[0.50, 0.95]`. If a GLASS row is added to
  Table 1 from that arm, `docs/paper_text_dmfm.md`'s caveat 4 applies: parity within about one paired
  SEM is the supportable claim, not an improvement.
* **`GRAD_NORMALIZE` defaults to 0** in `scripts/dna/table1_c0_guidance_dmfm.sbatch`, so a resubmit
  that forgets `GRAD_NORMALIZE=1` would silently produce the clip-down configuration that §7 item 8
  forbids. The in-flight jobs do carry it (verified in their `args.json`).
* **Algorithm 3 does not touch the DNA experiments.** Algorithms 2 and 3 are multimodal/molecular
  throughout, and the paper contains no algorithm for dMFM *steering* -- only Algorithm 1, dMFM
  training. That is why the DNA guidance step has to be written out in App. F.4 (E12).
* **New discrepancy: the DNA reward as stated is not the reward that was run.** §3.1's
  \eqref{eqn:guidance reward dna} and App. F.4's restatement both give `r = -(f_cyc(x) - y*)^2`.
  `src/dmfm/api.py:315-330` (`c0_log_reward_fn`) exponentiates
  `reward_scale * (-0.5 ((f - y*)/reward_sigma)^2)`, and its own docstring says `= -11.1 (f - y*)^2`
  at `reward_sigma 0.15 / reward_scale 0.5` -- the values in every DNA C0 `args.json`: the published
  pilot, the GLASS n=1000 run, both in-flight Table 1 reruns and the Figure 7 sweep. A constant
  inside `log E[exp(r)]` is an inverse temperature on the look-ahead weights, so unlike a constant in
  front of `grad V` it cannot be absorbed into the tuned guidance strength; it changes the direction
  of the estimated gradient as well as its magnitude. **This applies to the submitted numbers as
  well as the reruns**, so no rerun removes it. Proposed one-clause disclosure in App. F.4 as E16 of
  `docs/paper_text_dmfm.md`. This is a disclosure, not a number: the reported results stand, because
  the constant was fixed throughout.
* **Figure 8's regeneration is not at the Table 1 operating point** either:
  `scripts/dna/regen/fig8_guidance_trace.sbatch` defaults to `--mc 128 --nfe_value 12
  --guidance_frac 1.0 --guide_t_start 0.0 --target_c0 0.30`, against Table 1's `mc 8`,
  `nfe_value 4`, `guidance_frac 8.0`, window `[0.50, 0.95]`, `y* in {-1, +1}`. E15 asks the caption
  to say so.

### Revision 2026-09-29 22:55: the dMFM-vs-GLASS comparison is settled against dMFM

**This supersedes the "parity/tie" reading in the first version of this section and in
REPRODUCIBILITY_PLAN.md §7 item 8's "indistinguishable at y*=+1" phrasing.** Source:
`results/dna/c0_guidance/table1_dmfm_PRELIMINARY_partial.json` (its own `_status` says not to cite it
as Table 1; the y*=+1 dMFM arm is nonetheless final at n=1000).

At y*=+1, both arms under the identical scale-free rule, sharing initial noise per `sample_idx`:

| arm | guided oracle MAE | paired improvement | pairs improved | n |
|---|---|---|---|---|
| dMFM, 4-jump ESD student at 4 jumps | **0.1933** [0.1836, 0.2036] | 0.980 | 97.5% | 1000 (final) |
| matched GLASS-4 | **0.1517** [0.1436, 0.1597] | 1.031 | 98.75% | 800 |
| per-sample paired difference, dMFM - GLASS | **+0.0403**, SEM 0.0072, CI [0.0260, 0.0549] (~5.6 SEM); dMFM better on 44.5% of pairs | | | 800 shared ids |
| withdrawn mis-wired configuration, for scale | 0.6103 | 0.563 | | 1000 |

y*=-1 is n=200 on the dMFM side (guided 0.1844) with **no matched GLASS arm yet**, so that target's
comparison is unmeasured. The n=32/64 parity runs are statistically compatible with this; they were
too noisy to resolve a 0.04 gap, and the tighter measurement governs.

**Consequences for the prose.**

* The supportable claims are: dMFM works as a demonstration (oracle MAE 1.174 -> 0.193, 97.5% of
  matched pairs improved, parent-disjoint oracle), and where dMFM and GLASS are compared **dMFM is
  modestly but clearly behind**. Never parity, never superiority.
* **Table 1's cells must not be filled from the published values or from
  `table1_n1000_summary.json`**, both of which are GLASS: the y*=+1 guided MAE moves 0.157 -> ~0.193
  and the paired improvement 1.013 -> 0.980.
* §3.1's "The guided errors of [TODO] and [TODO] are comparable to the oracle's own MAE of 0.16"
  survives as written, but changes character: under GLASS the guided error sat *below* 0.16, under
  dMFM it sits *above* it. The next sentence's argument (guidance reaches the target region rather
  than attaining it to within that figure) is thereby strengthened. No edit; flagged because the
  substitution is easy to get backwards.

**Which of the 16 proposed edits moved.** Two, both in `docs/paper_text_dmfm.md`:

* **E7** (App. F.2) asserted gradient-level parity before the arrays landed -- "and it does not: the
  two agree within [X] ..." -- and is now a neutral statement of what is measured, with the direction
  to be written from the numbers. E6's and E9's phrasing of the same question was already neutral and
  is unchanged.
* **Caveat 4** of the claims list is inverted, from "a tie is the supportable claim" to the statement
  above.

Scoped rather than changed: the E1 note and caveat 6 compare the four-jump student with the one-jump
student, an axis the n=800 run does not cover (only the four-jump configuration was rerun at scale),
so both now say so explicitly. No other edit asserted or implied parity: E2, E3, E5, E8, E10-E16 make
no dMFM-vs-GLASS comparative claim, and E4's replacement text deliberately separates gradient
accuracy from steering quality, which this result reinforces.

**One pre-existing template needs the same correction, and I have not touched it.** §4.7b's template
for the end of App. F.2 (above, unaltered) reads "... degrades the gradient estimate, and it does not:
the two agree within $[R]$ ...". That pre-judges the direction, on the gradient axis, in the same way
the withdrawn tie reading did on the steering axis. Whoever owns §4.7b should make it neutral;
`docs/paper_text_dmfm.md` E7 carries the neutral wording.

**Two findings from this round that other records contradict, restated because people are relying on
those records:** §1.14's "final" Tables 4-5 and B.1 diversity prose were recomputed on **GLASS**
sample sets and must be re-opened (Table 4 itself is genuinely final -- real windows only, so
sampler-independent); and Fig 5's dMFM configuration (diagonal student, **one** jump) differs from
Table 1's (4-step ESD student, **four** jumps), so "the dMFM sampler" does not identify one thing and
each location must name which.
