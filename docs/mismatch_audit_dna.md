# Per-row audit of every MISMATCH in the DNA notebooks (02, 03, 05)

Written 2026-09-29 by the mismatch-audit agent. Read-only pass over
`notebooks/0{2,3,5}_*.ipynb`, `results/dna/`, the read-only backup
`/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/`, the paper text
(`dMFM (40).pdf`, line-numbered extraction cited as `paper.txt:<line>`) and
`docs/REPRODUCIBILITY_PLAN.md` §7. Nothing outside this file was changed.

## Summary

**First correction: the counts in the task brief are string counts, not rows.** Grepping
`MISMATCH` in the notebook JSON gives 11 / 138 / 16 for notebooks 02 / 03 / 05, because every
`compare()` row is rendered twice (`text/plain` **and** `text/html`) and `show()` additionally
prints one `status counts:` line per table. The number of distinct `compare()` rows whose status is
`MISMATCH` is **4 in notebook 02, 66 in notebook 03, 5 in notebook 05 — 75 in total** (verified by
parsing the `status counts:` dicts in the executed outputs with `nbformat`; the per-cell breakdown is
02: cells 4/7/9 = 2/1/1; 03: cells 10/12/54/56/58 = 22/1/6/28/9; 05: cells 18/22 = 2/3, with cell 20
re-displaying cell 18's two rows). Notebooks 02 and 03 were last executed 2026-09-29 19:25 and 05 at
18:06, so the brief's 11/137/12 also predate the current executions.

Class totals over those 75 rows:

| class | rows | where |
|---|---|---|
| **rounding** | 2 | 05 (Table 18, `gc L100 MC=2`, `motif L50 MC=1`) |
| **by-design regeneration** | 65 | 03 (22 = Table 1 at n=1000, §7.2; 43 = Tables 4/5 + App. B.1 on parent-disjoint, §7.7) |
| **within-noise rerun difference** | 0 | — (noise intervals are quoted as supporting evidence inside the two classes above, but no row's *cause* is a nondeterministic rerun) |
| **paper-vs-records discrepancy** | 8 | 02 (4: Table 7 diagonal weight, Table 7 consistency weight, Table 7 "10 hours", Table 9 L=400 batch size); 03 (1: Table 1 printed at n=32 vs text's 1,000); 05 (3: App. F.6 t=0.3 vs t_eval=0.5 for all three rewards) |
| **suspected port/analysis bug** | 0 | — |
| **unexplained** | 0 | — |

In one paragraph: every one of the 75 mismatching rows is accounted for. Two are the paper
double-rounding a number that the original scoring job itself wrote out (verified digit by digit).
Eight are the paper disagreeing with its own run records independently of any rerun, all eight
already in §7.5/§7.9 and in `docs/paper_changes.md` §3.7/§3.8. The remaining 65 are the two
author-endorsed regenerations — Table 1 at n = 1,000 (§7.2) and Tables 4/5 + App. B.1 on the
parent-disjoint models (§7.7) — and in both cases the same notebook code reproduces the *submitted*
numbers exactly from the surviving records of the original runs (22/22 rows for Table 1's n = 32
pilot in cell 8; 20/20 for Table 4, 44/44 for Table 5 and 10/10 for App. B.1 against the recovered
June `split65k` records in cells 21/23/25), which is what rules out a port bug as the cause. No row
looks like our bug, and no row is unexplained.

---

## Notebook 02 — `02_dna_base_model.ipynb` (4 rows)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| diagonal loss weight (Table 7) | 0.35 | 1.0 | paper-vs-records discrepancy | paper: `paper.txt:1990` (PDF line 1336). Records: `results/dna/training/dmfm_L{50,100,200}_args.json`, `dmfm_L400_args.json`, `dmfm_L400_parent_esd_run_args.json`, `dmfm_4step_L{50,100,200,400}_args.json` — `mfm_diag_loss_weight = 1.0` in **all nine** | §7.5 (text fix applied to `iclr2027_conferencemod.tex`; `docs/paper_changes.md` line 48-49, 63: "No record anywhere contains 0.35 or 10.0 as a loss weight"). 0.35 is not a rounding of 1.0 and not reachable from any stored arg. |
| consistency (ESD) loss weight (Table 7) | 10.0 | 1.0 | paper-vs-records discrepancy | paper: `paper.txt:1991` (PDF 1337). Records: same nine files — `mfm_lsd_loss_weight = 1.0` in eight; the ninth (`dmfm_L400_args.json`, the diag-only fine-tune) has `0.0` with `mfm_consistency: false` | §7.5, same fix. |
| dMFM L50 training wall-clock, hours (Table 7 text) | 10 | 0.693 | paper-vs-records discrepancy (scope) | paper: `paper.txt:1957` (PDF 1311) "We trained the dMFM for 10 hours on one 80GB NVIDIA H100 GPU". Record: `results/dna/training/wallclock_from_mtimes.json` → `yeast_parent_a_L50_dmfm_diagonal_h192_b4` = 0.693 h (args.json mtime → `last.ckpt` mtime) | **New evidence, see the author-decision section.** The nine runs in that file sum to 6.758 h; the four dMFM diagonal students sum to 2.635 h. The four *4-step continuation* runs have no entry, and I derived them from backup mtimes: 2.146 / 2.107 / 2.122 / 2.135 h (L50/100/200/400, `args.json` → `log.log`, dirs `…/workdir/yeast_parent_a_L*_dmfm_4step_esd_gap_upto025_corrected_20260727`), i.e. 8.510 GPU-h. Diagonal + fine-tune + 4-step = **11.158 GPU-h** of dMFM training, so "10 hours" is consistent with the *total* over all eight dMFM runs on four GPUs and with no single run. |
| L400 dMFM batch size, last 750 steps (Table 9) | 16 | 8 | paper-vs-records discrepancy | paper: `paper.txt:2032` (PDF 1361, Table 9 `Batch size` row). Records: `results/dna/training/checkpoint_summary.json` → `dmfm.400` = `{run_dir: yeast_parent_a_L400_dmfm_diagonly_finetune_20260726, checkpoint: epoch=33-step=45750-v1.ckpt, global_step: 45750}`; `dmfm_L400_args.json` → `batch_size: 8, max_steps: 46000, mfm_consistency: false`; `dmfm_L400_finetune_log.log` line 10 → `MFM_DEBUG {'iter_step': 0, 'global_step': 45000, …}` i.e. it resumed at step 45,000 | §7.9 / `docs/paper_changes.md` §3.8: Table 9's L=400 column takes its selected step (45,750) from the diag-only fine-tune at batch 8 and its batch size (16) from the ESD run (`dmfm_L400_parent_esd_run_args.json`, `batch_size: 16`). The row is a *new* finding of this consolidation, not in the submitted paper's own errata. |

Context: the same two cells match 16/19 Table 7 rows and 32/33 Table 9 rows exactly (including all
five parameter counts to the unit, the four selected dMFM steps, and `windows_per_parent`), so the
reconstruction of the training records is not in question. One further Table 7 row, `GLASS NFE = 8`,
is `n/a` rather than MISMATCH because no stored argument corresponds to it at all — see the
author-decision section.

---

## Notebook 03 — `03_dna_cyclizability.ipynb` (66 rows)

### Cell 12 — one row (paper-vs-records)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| pairs per target (submitted text) vs submitted table | 1000 | 32 | paper-vs-records discrepancy | paper text: `paper.txt:453` (PDF 307) "For each target, we generate 1,000 [guided–unguided pairs]". Paper table: `paper.txt:420-421` (PDF 277-278) = the n = 32 pilot. Records: `results/dna/c0_guidance/pilot_n32/target_{1p0,m1p0}/args.json` → `n_samples: 32` (jobs 35242279 / 35242278) | Brief item 1 / §7.2. The printed table is the 32-pair pilot; the text claims 1,000. Resolved by the §7.2 decision to print n = 1000. |

### Cell 10 — 22 rows, submitted Table 1 (n = 32) vs the n = 1,000 rerun

All 22 are **by-design regeneration** (§7.2: "Table 1 matched the n=32 pilot while the text said
1,000 pairs. Decided: switch to n=1000"). Evidence that the cause is the sample size and not our
code: cell 8 of the same notebook reproduces the **submitted** table from the pilot records 22/22
rows, cell by cell including the bootstrap CIs, with the same `build_table1_summary` code path
(`notebooks/_dna_helpers.py`), and `results/dna/PROVENANCE.md` records that the recomputation equals
the `oracle_summary.csv` written by the original scoring jobs to ≤ 2.2e-16. Run dirs:
submitted = `workdir/yeast_parent_glass_pilot_target{1p0,m1p0}_g8.0_seed0/` (jobs 35242279 /
35242278, 2026-07-26), ours = `workdir/yeast_parent_glass_n1000_target{1p0,m1p0}_g8.0_seed0/`
(jobs 48651189 / 48651190, 2026-09-25), copied to `results/dna/c0_guidance/{pilot_n32,n1000}/`.

Sanity check on the classification (cell 6's own output): **all six** point estimates that have a
printed CI fall inside the submitted table's 95% CI, and the CI narrows by the expected factor
(√(1000/32) = 5.59; observed pilot/n1000 width ratios 4.87–6.19).

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| y*=+1 oracle_mean_unguided | -0.169 | -0.171460 | by-design regeneration | `c0_guidance/n1000/target_1p0/oracle/oracle_per_sample.csv` | no printed CI; unguided branch is deterministic, the pilot's 32 sequences are the n=1000 run's first 32 (verified 32/32 below) |
| y*=+1 oracle_mean_guided | 0.912 | 0.917096 | by-design regeneration | same | no printed CI |
| y*=+1 unguided_mae | 1.170 | 1.173698 | by-design regeneration | same | inside the submitted CI [1.038, 1.291] |
| y*=+1 unguided_mae CI low | 1.038 | 1.148523 | by-design regeneration | same | CI narrows 4.87x at n=1000 |
| y*=+1 unguided_mae CI high | 1.291 | 1.200498 | by-design regeneration | same | nested in [1.038, 1.291] |
| y*=+1 guided_mae | 0.157 | 0.148292 | by-design regeneration | same | inside [0.119, 0.196] |
| y*=+1 guided_mae CI low | 0.119 | 0.141194 | by-design regeneration | same | CI narrows 5.32x |
| y*=+1 guided_mae CI high | 0.196 | 0.155643 | by-design regeneration | same | nested in [0.119, 0.196] |
| y*=+1 paired_improvement | 1.013 | 1.025406 | by-design regeneration | same | inside [0.881, 1.142] |
| y*=+1 paired_improvement CI low | 0.881 | 0.999189 | by-design regeneration | same | CI narrows 4.91x |
| y*=+1 paired_improvement CI high | 1.142 | 1.052215 | by-design regeneration | same | nested in [0.881, 1.142] |
| y*=-1 oracle_mean_unguided | -0.169 | -0.171460 | by-design regeneration | `c0_guidance/n1000/target_m1p0/oracle/oracle_per_sample.csv` | identical to the +1 row: both targets use seed 0, so the unguided pool is shared |
| y*=-1 oracle_mean_guided | -0.916 | -0.948533 | by-design regeneration | same | no printed CI |
| y*=-1 unguided_mae | 0.831 | 0.832561 | by-design regeneration | same | inside [0.709, 0.964] |
| y*=-1 unguided_mae CI low | 0.709 | 0.806120 | by-design regeneration | same | CI narrows 4.97x |
| y*=-1 unguided_mae CI high | 0.964 | 0.857484 | by-design regeneration | same | nested in [0.709, 0.964] |
| y*=-1 guided_mae | 0.153 | 0.119226 | by-design regeneration | same | inside [0.118, 0.190]; largest shift of the six (0.033) |
| y*=-1 guided_mae CI low | 0.118 | 0.113609 | by-design regeneration | same | **the one CI that is not nested** in the pilot's (0.1136 < 0.118); point estimate still inside |
| y*=-1 guided_mae CI high | 0.190 | 0.125235 | by-design regeneration | same | CI narrows 6.19x |
| y*=-1 paired_improvement | 0.678 | 0.713335 | by-design regeneration | same | inside [0.554, 0.816] |
| y*=-1 paired_improvement CI low | 0.554 | 0.686624 | by-design regeneration | same | CI narrows 4.95x |
| y*=-1 paired_improvement CI high | 0.816 | 0.739622 | by-design regeneration | same | nested in [0.554, 0.816] |

**Extra check I ran, because the n = 1,000 run used the reconstructed sampler.** `PROVENANCE.md`
(`original_scripts`, item 2) records that the reconstruction differs from DNA-MFM@187fe7b's
`sample_yeast_c0_guidance_hist.py` in 17 lines, all in the guide-model wrapper, and the guided MAE
moves the *same direction* for both targets (0.157 → 0.148 and 0.153 → 0.119), which is what a
systematic implementation difference would look like. Restricting the n = 1,000 run to the pilot's
own 32 `sample_idx` settles it:

| target | unguided MAE pilot / n1000-first-32 / n1000-all | guided MAE pilot / n1000-first-32 / n1000-all | sequences identical on those 32 ids |
|---|---|---|---|
| +1 | 1.169813 / 1.169812 / 1.173698 | 0.157120 / **0.186205** / 0.148292 | unguided 32/32, guided 3/32 |
| -1 | 0.831136 / 0.831137 / 0.832561 | 0.152637 / **0.161241** / 0.119226 | unguided 32/32, guided 8/32 |

On matched ids the reconstruction is slightly *worse*, not better, so the n = 1,000 improvement comes
from the other 968 pairs, not from the script difference. The unguided branch is bit-identical
(32/32 sequences, MAE to 1e-6), and the guided branch differs on 24–29 of 32 ids, consistent with the
recorded nondeterministic CUDA backward. Reproduce with
`results/dna/c0_guidance/{pilot_n32,n1000}/target_*/oracle/oracle_per_sample.csv`.

### Cells 54, 56, 58 — 43 rows, submitted Tables 4/5 + App. B.1 (`split65k`) vs the parent-disjoint regeneration

All 43 are **by-design regeneration** (§7.7: "Tables 4-6 and App B.1 were computed on the purged
`split65k` models, not the parent-disjoint models B.1 describes. Decided: regenerate on
parent-disjoint and report the weaker result"). Cells 58's nine rows are the App. B.1 prose
restatements of nine cells already listed in 54/56, so the 43 rows cover 34 distinct quantities.

Evidence that the cause is the model/split and not our analyser, in order of strength:

1. `results/dna/diversity/regen/table4_real_baselines.csv` and `table5_generated.csv` (the ported
   `dmfm.experiments.analyze_c0_diversity`) are **byte-identical** to the round-1 collection
   `results/dna/diversity/n1000/table6_baselines.csv` and `table7_generated.csv`
   (`workdir/rebuttal_parent_c0_diversity_n1000_20260925/`, CPU, 2026-09-25) — verified with `diff`.
2. The one row where the two datasets coincide reproduces the paper: the paper's `train_sample` row
   (`0.5219 / 0.7377 / 0.7400`) equals our `full_sample` row (`0.521880 / 0.737699 / 0.740000`) to
   2e-5 and 1e-6 — the same 82,368 L=50 windows and the same seed-0 draw of 1,000
   (`PROVENANCE.md`, Tables 4-5 section; notebook cell 29 shows 4/4 match). This validates the
   ported metric functions against the June analyser end to end.
3. Cells 21/23/25 reproduce the **printed** Tables 4 and 5 and the App. B.1 prose from the recovered
   June records 20/20, 44/44 and 10/10 (`results/dna/june_split65k_recovered/`, Codex transcript
   lines 1003/1055/2569, hashes in `_recovery_manifest.json`).

Paper side: `paper_values.json` `table4_real_baselines` / `table5_generated_diversity` / `appB1_text`,
spot-checked against `paper.txt:1325-1326` (Table 4), `paper.txt:1335` (Table 5, the -1 guided row)
and `paper.txt:1349` (App. B.1 prose). Ours: `results/dna/diversity/regen/*.csv`.

Noise reference used in the notes below — the analyser's own size-matched held-out null at n = 1,000
(1,000 draws), `results/dna/diversity/regen/size_matched_test_null_summary.csv`:
pairwise `[0.737390, 0.739748]` (width 0.00236), nearest-generated `[0.519880, 0.524961]`
(width 0.00508), nearest-a_train `[0.453999, 0.458480]` (width 0.00448).

#### Cell 54 — Table 4, real-data baselines (6 rows)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| split_train_sample · nearest_generated | 0.5219 | 0.520000 | by-design regeneration | `diversity/regen/table4_real_baselines.csv` row `a_train_sample` | diff -0.00190; both values inside the nearest-gen null band |
| split_train_sample · pairwise_generated | 0.7373 | 0.736074 | by-design regeneration | same | diff -0.00123 (≈ half the band width); neither value is inside the band, which is computed on the `test` split |
| split_val_sample · nearest_generated | 0.5184 | 0.519360 | by-design regeneration | row `a_val_sample` | diff +0.00096; both just below the band |
| split_val_sample · pairwise_generated | 0.7372 | 0.738451 | by-design regeneration | same | diff +0.00125; ours inside the band, paper below it |
| split_test_sample · nearest_generated | 0.5201 | 0.521920 | by-design regeneration | row `test_sample` | diff +0.00182; both inside the band |
| split_test_sample · pairwise_generated | 0.7374 | 0.738115 | by-design regeneration | same | diff +0.00072; both inside the band |

The paper's `split_*` rows are 1,000-window draws from the `split65k` random-window split of
`data/yeast_mid50.pt`; ours are from the parent-disjoint `a_train` / `a_val` / `test` splits of
`data/yeast_parent_disjoint/yeast_parent_L50.pt`. Different subsets of different files, so a
4th-decimal difference is expected; all six differences are at or below the scale of the analyser's
own subsample noise. `exact_unique` (1.0000), `n` (1000) and `pairwise_median` (0.7400) match in
every row, and the `train_sample` row matches in all four columns.

#### Cell 56 — Table 5, generated sets (28 rows)

`n`, `exact_unique`, `novel_train` and `novel_test` match in all four rows (16 matches). All 28
mismatches are **by-design regeneration**. Paper rows come from
`workdir/yeast_split_dmfm_c0_diversity_2026-06-09/` on the purged `split65k` teacher + dMFM with
guide `dinko/C0free_torch.pt`, `mc 16`, `reward_scale 1.0`, guidance t ∈ [0.01, 0.95], seeds 0 (+1)
and 1 (−1); ours from `results/dna/c0_guidance/n1000/` on the parent-disjoint L=50 base DFM with the
`park_cnn` guide, `mc 8`, `reward_scale 0.5`, guidance t ∈ [0.5, 0.95], seed 0 for both targets
(`PROVENANCE.md`, Table 1 and Tables 4-5 sections).

| item | paper | ours | class | note (all evidence = `diversity/regen/table5_generated.csv`) |
|---|---|---|---|---|
| -1_guided · guide_score_mean | -0.9893 | -0.993239 | by-design regeneration | diff -0.00394 |
| -1_guided · nearest_train | 0.3966 | 0.445820 | by-design regeneration | diff +0.04922 = 11x the nearest-a_train band width |
| -1_guided · nearest_test | 0.4361 | 0.473680 | by-design regeneration | diff +0.03758 |
| -1_guided · nearest_gen | 0.4175 | 0.496040 | by-design regeneration | diff +0.07854 = 15x the nearest-gen band width — the substantive §7.7 finding |
| -1_guided · pairwise_gen | 0.7010 | 0.728695 | by-design regeneration | diff +0.02770 = 12x the pairwise band width |
| -1_guided · pairwise_over_heldout | 0.9507 | 0.987237 | by-design regeneration | the App. B.1 clustering statistic; §7.7 records exactly this move (0.9507x → 0.9872x) |
| -1_guided · nearest_gen_over_heldout | 0.8027 | 0.950414 | by-design regeneration | diff +0.14771, the largest single change in the DNA section |
| -1_unguided · guide_score_mean | -0.1692 | -0.176716 | by-design regeneration | diff -0.00752 |
| -1_unguided · nearest_train | 0.4449 | 0.458260 | by-design regeneration | diff +0.01336; ours inside the nearest-a_train band, paper below it |
| -1_unguided · nearest_test | 0.4779 | 0.481700 | by-design regeneration | diff +0.00380 |
| -1_unguided · nearest_gen | 0.5210 | 0.522060 | by-design regeneration | diff +0.00106, 0.21x the band width |
| -1_unguided · pairwise_gen | 0.7364 | 0.737431 | by-design regeneration | diff +0.00103, 0.44x the band width; ours inside the band |
| -1_unguided · pairwise_over_heldout | 0.9987 | 0.999074 | by-design regeneration | diff +0.00037 |
| -1_unguided · nearest_gen_over_heldout | 1.0017 | 1.000268 | by-design regeneration | diff -0.00143 |
| +1_guided · guide_score_mean | 1.0010 | 0.998595 | by-design regeneration | diff -0.00241 |
| +1_guided · nearest_train | 0.4446 | 0.455940 | by-design regeneration | diff +0.01134 |
| +1_guided · nearest_test | 0.4769 | 0.480400 | by-design regeneration | diff +0.00350 |
| +1_guided · nearest_gen | 0.4913 | 0.509140 | by-design regeneration | diff +0.01784 |
| +1_guided · pairwise_gen | 0.7372 | 0.734848 | by-design regeneration | diff -0.00235 |
| +1_guided · pairwise_over_heldout | 0.9997 | 0.995574 | by-design regeneration | diff -0.00413 |
| +1_guided · nearest_gen_over_heldout | 0.9447 | 0.975513 | by-design regeneration | diff +0.03081 |
| +1_unguided · guide_score_mean | -0.1766 | -0.176716 | by-design regeneration | diff -0.000116, i.e. 2.3x the printed-precision tolerance — the most marginal MISMATCH in the three notebooks |
| +1_unguided · nearest_train | 0.4466 | 0.458260 | by-design regeneration | diff +0.01166 |
| +1_unguided · nearest_test | 0.4787 | 0.481700 | by-design regeneration | diff +0.00300 |
| +1_unguided · nearest_gen | 0.5215 | 0.522060 | by-design regeneration | diff +0.00056, 0.11x the band width |
| +1_unguided · pairwise_gen | 0.7370 | 0.737431 | by-design regeneration | diff +0.00043, 0.18x the band width |
| +1_unguided · pairwise_over_heldout | 0.9995 | 0.999074 | by-design regeneration | diff -0.00043 |
| +1_unguided · nearest_gen_over_heldout | 1.0027 | 1.000268 | by-design regeneration | diff -0.00243 |

Note on the two unguided rows: in the paper they differ from each other (`-0.1692` vs `-0.1766`,
`0.4449` vs `0.4466`) because the June runs used seeds 0 and 1; our two unguided sets are identical
because both n = 1,000 runs used seed 0. That is recorded in the notebook's own markdown (cell 32) and
in `diversity/n1000/unguided_identity.json`, and it is why all seven of our `-1_unguided` and
`+1_unguided` values are pairwise equal. Not a bug: the unguided branch does not depend on the target.

#### Cell 58 — App. B.1 prose (9 rows, all duplicates of cells 54/56)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| held-out pairwise | 0.7374 | 0.738115 | by-design regeneration | `table4_real_baselines.csv` `test_sample` | = cell 54 `split_test_sample · pairwise_generated` |
| held-out nearest-gen | 0.5201 | 0.521920 | by-design regeneration | same | = cell 54 `split_test_sample · nearest_generated` |
| +1 guided pairwise | 0.7372 | 0.734848 | by-design regeneration | `table5_generated.csv` | = cell 56 `+1_guided · pairwise_gen` |
| +1 guided pairwise / held-out | 0.9997 | 0.995574 | by-design regeneration | same | = cell 56 `+1_guided · pairwise_over_heldout` |
| +1 guided nearest-gen | 0.4913 | 0.509140 | by-design regeneration | same | = cell 56 `+1_guided · nearest_gen` |
| -1 guided pairwise | 0.7010 | 0.728695 | by-design regeneration | same | = cell 56 `-1_guided · pairwise_gen` |
| -1 guided pairwise / held-out | 0.9507 | 0.987237 | by-design regeneration | same | the sentence App. B.1 builds its clustering claim on |
| -1 guided nearest-gen | 0.4175 | 0.496040 | by-design regeneration | same | = cell 56 `-1_guided · nearest_gen` |
| -1 guided nearest-gen / held-out | 0.8027 | 0.950414 | by-design regeneration | same | the second clustering number; 0.8027 → 0.9504 |

---

## Notebook 05 — `05_dna_mc_scaling.ipynb` (5 rows)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| gc L100 MC=2 (Table 18) | 0.1234 | 0.123347 | **rounding** (double-rounded) | ours = `dmfm_vs_glass/gc/L100/gradient_mae_per_condition.csv` grouped mean = 0.12334650498240758, equal to the original job's own `dmfm_vs_glass/gc/gradient_mae_summary_all_lengths.csv` row `length=100, mc=2, mean_mae=0.1233465049824076` to 1.4e-17. Paper: `paper.txt:2630` (PDF 1734) | `round(x, 4) = 0.1233`; `round(round(x, 5), 4)` half-up `= round(0.12335, 4) = 0.1234`. The paper printed a value that had already been rounded to 5 decimals. Verified digit by digit; notebook cell 20 shows the same diagnostic. |
| motif L50 MC=1 (Table 18) | 0.2188 | 0.218746 | **rounding** (double-rounded) | ours = `dmfm_vs_glass/motif/L50/gradient_mae_per_condition.csv` grouped mean = 0.2187463587191871, exactly equal to the stored summary's `length=50, mc=1, mean_mae`. Paper: `paper.txt:2634` (PDF, Motif L=50 row) | `round(x, 4) = 0.2187`; `round(0.21875, 4)` half-up `= 0.2188`. Same mechanism. |
| gc outer time t (App. F.6) | 0.3 | 0.5 | paper-vs-records discrepancy | paper: `paper.txt:2609` (PDF 1722) "32 held-out states at outer time t = 0.3". Record: `results/dna/dmfm_vs_glass/gc/run_metadata.json` → `args.t_eval = 0.5`; run `workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/gc/`, job array 35642548 | §7.5 (text fix t=0.3 → t=0.5 already applied to `iclr2027_conferencemod.tex`); brief item 3. |
| motif outer time t (App. F.6) | 0.3 | 0.5 | paper-vs-records discrepancy | `dmfm_vs_glass/motif/run_metadata.json` → `t_eval = 0.5` | same |
| conjunction outer time t (App. F.6) | 0.3 | 0.5 | paper-vs-records discrepancy | `dmfm_vs_glass/conjunction/run_metadata.json` → `t_eval = 0.5` | same |

The other 46 of the 48 Table 18 cells match to the paper's printed 4 decimals, and all twelve
protocol rows other than `t` (32 states, 8 pools, GLASS-128 reference, RK4-50) match exactly, so the
two rounding rows are isolated and the recomputation path is sound.

---

## Rows that need an author decision

1. **Table 7 "10 hours" (notebook 02).** `docs/paper_changes.md` §3.7 left this open because summing
   the recorded runs did "not obviously reach 10 h". It now does, if the sentence is read as a total:
   the four diagonal students are 2.635 h, the L=400 diag-only fine-tune 0.013 h, and the four 4-step
   continuation runs 8.510 h by backup mtimes (2.146 / 2.107 / 2.122 / 2.135 h for L=50/100/200/400,
   `args.json` → `log.log` in
   `/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir/yeast_parent_a_L*_dmfm_4step_esd_gap_upto025_corrected_20260727/`),
   i.e. **11.158 GPU-hours** in total, run on four GPUs over ~3.9 h of elapsed time. The authors need
   to decide whether the sentence means one run (then it should say ~0.7 h + ~2.1 h in two stages) or
   the aggregate (then it should say "about 11 GPU-hours across eight runs"). These four wall-clocks
   are not yet in `results/dna/training/wallclock_from_mtimes.json` — adding them is a change to a
   file another agent owns, so I am reporting it rather than making it.
2. **Table 9's L=400 batch size (notebook 02).** Whether to add the "dMFM 4-step continuation" block
   that `docs/paper_changes.md` §3.8 proposes, or to annotate the L=400 column as two runs. Either
   way the printed 16 does not describe the 750 steps that produced the selected checkpoint.
3. **Table 7 `GLASS NFE = 8`** — reported as `n/a`, not MISMATCH, because no stored argument
   corresponds to it (`docs/paper_changes.md` §3.7). Someone has to say what the row meant.
4. **Table 1's identity (notebook 03).** §7.2 switched it to n = 1000, but §7.8 then decided it must
   become a **dMFM** run; `results/dna/c0_guidance/table1_dmfm_n1000_summary.json` did not exist when
   notebook 03 was last executed (cell 14 prints "PENDING"), so the 22 rows in cell 10 are still
   GLASS-vs-GLASS. When the dMFM summary lands, those 22 rows change again and this audit's cell-10
   block must be redone. The dMFM arrays (49277977 / 49308261 / 49293510) are owned by the dna-core
   agent.
5. **Whether cell 10 should stay in the notebook at all.** Its 22 rows are the deliberate
   submitted-vs-revised comparison; cells 3 and 8 already show 22/22 match against the revised and
   the submitted table respectively. It is honest but it is also the single largest contributor to the
   MISMATCH count, and it is what made the count look alarming.

## Rows that look like our bug, ranked by how much they'd change a published number

**None.** Every one of the 75 rows has a cause outside our code, and in each case the same code path
reproduces the corresponding original record exactly. For completeness, the three hypotheses I tested
and cleared, in descending order of what they would have cost:

1. *The reconstructed Table 1 sampler improves guided MAE (would change Table 1 and §3.1's headline
   numbers).* Cleared: restricted to the pilot's own 32 `sample_idx`, the n = 1,000 run's guided MAE
   is **worse** (0.1862 vs 0.1571 for y\*=+1; 0.1612 vs 0.1526 for y\*=−1), so the full-sample
   improvement is not an artefact of the 17-line guide-wrapper difference. Unguided is bit-identical
   (32/32 sequences, MAE to 1e-6). Command:
   `results/dna/c0_guidance/{pilot_n32,n1000}/target_*/oracle/oracle_per_sample.csv`.
2. *The ported diversity analyser changed Tables 4/5 (would change App. B.1's clustering claim, the
   largest single move in the DNA section: 0.8027x → 0.9504x).* Cleared two ways:
   `diversity/regen/*.csv` is byte-identical to the independently produced round-1 collection
   `diversity/n1000/*.csv` (`diff`), and the one row whose input data is unchanged — the paper's
   `train_sample` — is reproduced to 2e-5 (nearest-gen) and 1e-6 (pairwise).
3. *Table 18's two odd cells are an aggregation error in the notebook (would change two Table 18
   cells in the 4th decimal).* Cleared: the notebook's grouped mean equals the value the original
   scoring job wrote into `gradient_mae_summary_all_lengths.csv` to 1.4e-17 for both cells, so the
   discrepancy is entirely in the paper's printing.

If any of these needs to be re-opened later, the checks to run are: (1) rerun
`dmfm.experiments.sample_c0_guidance` at n = 32, seed 0, batch 8 and compare guided MAE on ids 0–31
against 0.1862 / 0.1612 rather than against the pilot; (2) re-run
`dmfm.experiments.analyze_c0_diversity` with `--train_pt` pointed at a `split65k`-style random-window
split, if one is ever reconstructed, and check that the `split_*` rows move back toward
0.5219 / 0.7373; (3) recompute Table 18 from `L*/gradient_pairs.pt` in the backup rather than from
the per-condition CSVs.
