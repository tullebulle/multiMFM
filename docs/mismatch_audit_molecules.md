# Per-row audit of the MISMATCH rows in the molecular notebooks

Notebooks `notebooks/07_qm9_property_guidance.ipynb` and `notebooks/08_geom_fingerprint.ipynb`.
Audited 2026-09-29 from the executed notebooks' rendered outputs, the `results/` files behind each
row, `results/{qm9,geom}/PROVENANCE.md`, the original run records, and
`docs/REPRODUCIBILITY_PLAN.md` §7. Read-only: nothing outside this file was changed.

## First correction: the counts are doubled

The reported counts of **42** (nb 07) and **16** (nb 08) are occurrences of the string `MISMATCH`
in the notebook JSON, and every `compare()` DataFrame is stored twice — once as `text/plain` and
once as `text/html`. Per-cell counts are identical in the two representations:

| notebook | cell | `text/plain` | `text/html` | distinct rows |
|---|---|---|---|---|
| 07 | 9 (Table 2 SS replicates) | 5 | 5 | 5 |
| 07 | 13 (Table 2, 3-seed rerun) | 11 | 11 | 11 |
| 07 | 26 (Table 21 as published vs mean) | 1 | 1 | 1 |
| 07 | 30 (Table 21 / text claims, 3-seed) | 3 | 3 | 3 |
| 07 | 39 (Table 20 objective) | 1 | 1 | 1 (not a `compare()` row — a hand-written note in a plain DataFrame) |
| 08 | 10 (Table 3 vs paper) | 3 | 3 | 3 |
| 08 | 29 (Table 21 GEOM row) | 5 | 5 | 5 |

So there are **20 `compare()` MISMATCH rows + 1 hand-annotated mismatch = 21 flagged items in
notebook 07**, and **8 in notebook 08**: **29 items**, not 58. Everything below covers all 29.

## Summary

Of the 29 items, **none is a rounding artifact** and **none is a port or analysis bug in the
recomputation**. Eleven are *within-noise rerun differences*: the paper value lies inside the
interval of the recomputed value once the submitted run's own sampling error is counted — this
covers all six Table 2 multiMFM rows (0.2–1.7 sigma), the α and HOMO steer-search rows (1.7 sigma
each, driven by the enormous a/b replicate spread), both PB-validity text claims for multiMFM and
the base model (1.7 and 0.8 sigma), and the GEOM guided PB validity at 4,096 × 3 (87.5 % sits
inside the 3-seed 95 % t-CI [82.32, 87.67]). Six are *replicate-selection artifacts* confined to
Table 2 / Table 21's SS row, which mixed `k_*_a` / `k_*_b` under four different rules. Five are
*by-design regenerations* the authors endorsed (§7 items 1, 4, 5): the three steer-search
properties whose Algorithm-3 result genuinely moved (μ 10.2 sigma, Δε 3.7 sigma, LUMO 3.6 sigma),
the SS PB validity that Algorithm 3 drives to 100 %, and the GEOM Table 3 Tanimoto at 4,096 × 3.
Seven are *paper-vs-records discrepancies* independent of any rerun: Table 20's QM9 objective, and
six rows around GEOM's N = 4,096 claim and its 92.0 % base PB validity, which appears in no
4,096-molecule record. Nothing is *unexplained*.

| class | count | items |
|---|---|---|
| rounding | **0** | — |
| by-design regeneration | **5** | nb07: T2 SS μ, T2 SS Δε, T2 SS LUMO, mean PB SS; nb08: T3 Tanimoto at 4,096 × 3 |
| replicate-selection artifact | **6** | nb07: SS mean(a,b) Δε (MAE), SS rep a α, SS rep a HOMO, SS rep b Δε, SS rep b HOMO, T21 SS mean(a,b) Δε |
| within-noise rerun difference | **11** | nb07: all 6 T2 multiMFM rows, T2 SS α, T2 SS HOMO, mean PB multiMFM, base PB; nb08: multiMFM PB at 4,096 × 3 |
| paper-vs-records discrepancy | **7** | nb07: T20 objective; nb08: base PB 92.0 % (×3 rows), N = 4,096 (×2 rows), unsteered PB at 4,096 × 3 |
| suspected port/analysis bug | **0** | see "rows that look like our bug" — the two candidates were checked and cleared |
| unexplained | **0** | — |

Two caveats that follow the brief's instruction to keep the GEOM runs separate. (1) The n = 64
single-seed grid (job 35670937) and the 4,096 × 3 rerun of that same configuration (job 47833756)
are distinct evidence and are labelled as such in every row below. (2) §7 item 1's `grad-sign`
diagnostic is **still outstanding** — `results/geom/grad_sign.json` and `grad_sign_cpu.json` do not
exist, job 49293660 is PENDING — so the negative-gain result is *not* treated as settled anywhere
in this audit. The as-described 4,096 × 3 evaluation (job 49274410, still RUNNING) now argues
against it; see "In flight" below.

---

## Notebook 07 — QM9 property guidance

`paper` values from `results/qm9/paper_values.json`; `ours` as rendered in the executed notebook.

### Cell 9 — Table 2 steer-search replicates (5 rows)

The SS row was run twice per property with **identical** arguments (`k_<prop>_a`, `k_<prop>_b`,
seed 8, target seed 2026); they differ only through nondeterministic CUDA kernels. `k_lumo_a` was
cancelled by Slurm at 17:16 on 2026-07-25 and left no summary. The printed row does not follow one
rule: Cv = a, μ = a or b (both round to 0.69), α = mean(a, b), Δε = a, HOMO = mean(a, b),
LUMO = b. These five rows are the diagnostic that shows it; each MISMATCH is a replicate the
printed number did *not* come from, or the consistent-mean alternative.

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| SS mean(a,b) Δε [meV] | 462.00 | 471.202668 | replicate-selection artifact | `results/qm9/table2.csv` (`ss_oracle_mae_mean_ab`); runs `runs/multimfm_ss/k_gap_{a,b}/summary.json` | The paper's 462 is **replicate a alone** (462.283956 → "462"). b is 480.121380. One consistent rule — mean of available replicates — gives **471.20**, i.e. Δε would be printed as 471, not 462. |
| SS replicate a α [Bohr³] | 1.10 | 1.152381 | replicate-selection artifact | `table2.csv` `ss_oracle_mae_a`; `runs/multimfm_ss/k_alpha_a` | The paper's 1.10 is **mean(a, b)** = 1.104487. a = 1.152381, b = 1.056593. The a/b spread, 0.0958 Bohr³, is the largest of the six properties and is pure run-to-run noise. |
| SS replicate a HOMO [meV] | 252.00 | 253.699508 | replicate-selection artifact | `table2.csv` `ss_oracle_mae_a`; `runs/multimfm_ss/k_homo_a` | The paper's 252 is **mean(a, b)** = 252.204866. a = 253.699508, b = 250.710224. Both round to 254 / 251, so the mean is what produced "252". |
| SS replicate b Δε [meV] | 462.00 | 480.121380 | replicate-selection artifact | `table2.csv` `ss_oracle_mae_b`; `runs/multimfm_ss/k_gap_b` | Same as the Δε row above, from the other side: b is 18 meV worse than a on identical arguments. |
| SS replicate b HOMO [meV] | 252.00 | 250.710224 | replicate-selection artifact | `table2.csv` `ss_oracle_mae_b`; `runs/multimfm_ss/k_homo_b` | b alone rounds to 251. Note the paper used **b for HOMO's PB validity** (97.8) but the **mean for HOMO's MAE** — two different rules within one table cell pair. |

Not rounding: verified digit by digit. 471.202668 → "471"; 480.121380 → "480"; 253.699508 → "254";
250.710224 → "251"; 1.152381 → "1.15". None of these double-round to the printed value.

### Cell 13 — Table 2 on 3 evaluation seeds (11 rows)

Both rows were rerun at n = 1000 on seeds 1/2/3 with target seeds 2027–2029 (held out from all
tuning; every submitted configuration was selected on seed 8 / target seed 2026). §7 item 4
endorsed the 3-seed rerun; §7 item 5 endorsed changing the steer-search look-ahead source to match
Algorithm 3. `ours` = mean over 3 seeds, `SE` = sd/√3, from
`results/qm9/rerun_3seed/rerun_3seed_summary.csv` (36/36 runs COMPLETED, Slurm 49276185 / 49279470
/ 49279656).

**Interval test used.** The 3-seed SE alone is the wrong yardstick: the submitted number is a
single n = 1000 draw with its own sampling error, printed in notebook cell 7 (Cv 0.0135, μ 0.039,
α 0.0346, Δε 18.8, HOMO 7.56, LUMO 12.4 — the per-molecule SEM of each submitted run). The test
below is `|diff| / sqrt(submitted_SEM² + rerun_SE²)`, verified numerically.

| item | paper | ours (3-seed) | class | evidence | note |
|---|---|---|---|---|---|
| T2 multiMFM Cv [cal/(mol K)] | 0.42 | 0.444352 ± 0.00364 | within-noise rerun difference | `rerun_3seed_summary.csv`; submitted `runs/multimfm/cv/summary.json` | diff +0.0244; combined sigma 0.0140 → **1.74 sigma**. The 3-seed t-CI alone ([0.4287, 0.4600]) excludes 0.42, but the submitted run's own SEM (0.0135) is larger than the seed-to-seed sd (0.0063), so the honest combined test is 1.7 sigma. Regression is expected: the (μ, r_s) grid was tuned on the reported seed (PROVENANCE discrepancy 4). |
| T2 multiMFM μ [D] | 0.84 | 0.818393 ± 0.0329 | within-noise rerun difference | same | diff −0.0216 → **0.42 sigma**; 3-seed t-CI [0.677, 0.960] contains 0.84. |
| T2 multiMFM α [Bohr³] | 1.12 | 1.166201 ± 0.0174 | within-noise rerun difference | same | diff +0.0462 → **1.19 sigma**; t-CI [1.091, 1.241] contains 1.12. |
| T2 multiMFM Δε [meV] | 600 | 592.30 ± 7.15 | within-noise rerun difference | same | diff −7.70 → **0.38 sigma**; t-CI [561.5, 623.1] contains 600. |
| T2 multiMFM HOMO [meV] | 262 | 270.15 ± 4.30 | within-noise rerun difference | same | diff +8.15 → **0.94 sigma**; t-CI [251.6, 288.7] contains 262. |
| T2 multiMFM LUMO [meV] | 392 | 395.25 ± 9.18 | within-noise rerun difference | same | diff +3.25 → **0.21 sigma**; t-CI [355.7, 434.8] contains 392. |
| T2 multiMFM-SS μ [D] | 0.69 | 0.527832 ± 0.0158 | by-design regeneration (§7 item 5) | `rerun_3seed_summary.csv` variants `ss` and `ss_base_endpoint` | diff −0.162 → **10.2 sigma** (the a/b replicate spread here is only 0.0003, so the submitted number is very tight). Genuinely changed, and the cause is isolated: the *paired* submitted-harvesting selection on the same trajectories gives 0.661 ± 0.018, close to the submitted 0.69. The move is the Algorithm-3 look-ahead source, not the seeds. |
| T2 multiMFM-SS α [Bohr³] | 1.10 | 1.009483 ± 0.0239 | within-noise rerun difference | same | diff −0.0905; the submitted value is mean(a, b) whose sd from the a/b spread is 0.0479 → combined sigma 0.0536 → **1.69 sigma**; 3-seed t-CI [0.907, 1.113] contains 1.10. α's replicate noise is large enough that this SS "improvement" is not established. |
| T2 multiMFM-SS Δε [meV] | 462 | 413.26 ± 3.66 | by-design regeneration (§7 item 5) | same | diff −48.7; submitted sd from the a/b spread 12.6 → combined 13.1 → **3.71 sigma**. Paired submitted-harvesting on the same trajectories: 473.3 ± 5.4, i.e. back at the submitted level. Attributable to Algorithm 3. |
| T2 multiMFM-SS HOMO [meV] | 252 | 243.08 ± 5.10 | within-noise rerun difference | same | diff −8.92; submitted (mean of a, b) sd 1.50 → combined 5.31 → **1.68 sigma**; t-CI [221.1, 265.0] contains 252. |
| T2 multiMFM-SS LUMO [meV] | 316 | 264.69 ± 6.86 | by-design regeneration (§7 items 4 + 5) | same, plus `results/qm9/ss_mfm_reselect/ss_mu_decision.csv` | diff −51.3 → **3.62 sigma** (no replicate a exists for LUMO, so the non-SS per-sample SEM 12.4 is used as the submitted sd — stated as a proxy). Two endorsed changes apply here: Algorithm 3 *and* the only μ the pre-registered reselection rule changed, 1.5 → 3.0, decided on the seed-8 tuning set (297.5 ± 18.6 → 245.2 ± 13.8 meV). |

Note that `T2 multiMFM-SS Cv` reads **match** (0.40 vs 0.413920) only because of a tolerance defect
in `compare()` — see "rows that look like our bug", item 3. At the paper's printed precision it is
a MISMATCH, and its class would be within-noise (1.35 sigma).

### Cell 26 — Table 21 as published vs the consistent mean (1 row)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| T21 SS mean(a,b) Δε PB (%) | 95.8 | 96.30 | replicate-selection artifact | `results/qm9/table21.csv` (`ss_pb_valid_{a,b,mean_ab}`) | The paper's 95.8 is **replicate a** (a = 95.8, b = 96.8). Mean = 96.3. The other five properties' means happen to round to the printed value, so Δε is the only PB cell the mixing changes. All 18 "as published" / unsteered / multiMFM rows in this cell match exactly. |

### Cell 30 — Table 21 and the §3.2 / §4 text claims on the 3-seed reruns (3 rows)

| item | paper | ours (3-seed) | class | evidence | note |
|---|---|---|---|---|---|
| mean PB multiMFM (%) | 91.4 | 90.7722 ± 0.09 | within-noise rerun difference | `rerun_3seed_summary.csv` `pb_valid_*` | The paper's 91.4 is internally consistent — the mean of its own six printed values is 91.433, and all six match the records exactly (cell 26). Against the rerun: diff −0.628 pp; the submitted mean's binomial SE is 0.89/√6 = 0.362 pp → combined 0.375 → **1.68 sigma**. Half the shift is the base model itself: its PB on the new target draws is 0.333 pp lower (row below). Per property the picture is mixed (Δε 0.2 sigma, HOMO 1.0, LUMO 1.9, α 5.4, μ 7.5), which is a target-draw signature rather than a systematic defect. |
| base PB (%) | 98.7 | 98.3667 ± 0.23 | within-noise rerun difference | same, variant `unsteered` | diff −0.333 pp; submitted binomial SE 0.358 pp → combined 0.43 → **0.78 sigma**; 3-seed t-CI [97.38, 99.36] contains 98.7. |
| mean PB SS (%) | 97.1 | 100.000 ± 0.00 | by-design regeneration (§7 item 5) | same, variant `ss`; `results/qm9/rerun_3seed/rerun_3seed_long.csv` | 100.0 % on all 6 properties × 3 seeds = 18,000 molecules, **not a bug**: Algorithm 3 (paper Algorithm 3, lines 9–12 and 18–24) makes PB validity π part of the selection criterion, and the Algorithm-3 rerun offers 3,680 look-aheads per molecule (32 posterior samples × 115 guided steps) against ≈89 for the submitted base-endpoint harvest. `frac_with_valid_candidate` = 1.000 in all 18 runs, and the flag is independently recomputed by PoseBusters on the *returned* molecule (`steer_search.evaluate_state` → `molecule_records`), not carried over from the selection. The paired submitted-harvesting variant on the same trajectories gives 96.84 % mean, i.e. the submitted 97.1 reproduces (diff 0.26 pp). Against the submitted number the gap is +2.9 pp = **13.3 sigma**. |

### Cell 39 — Table 20, QM9 MFM distillation objective (1 hand-annotated item)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| T20 objective | "Diagonal + ESD" | diagonal GLASS distillation on teacher-generated conditioning states (+ direction 5.0, condition-sensitivity 0.1/0.25) | paper-vs-records discrepancy | `results/qm9/table20_mfm_training.json` (from `args`/`student_cfg` inside `checkpoints/mfm_student/student_step_1000.pt`); `results/qm9/PROVENANCE.md` discrepancy 3 | The stored final phase is `--stage 1 --rollout-condition-source generated`; **no ESD term is active** (the ESD weight in the argument dict is unused in stage 1). The authors' own `training_tables_molecular.tex` / `tables_to_fix.tex` say the same. Independent of any rerun. Not a `compare()` row — it is a note the notebook author wrote into a plain DataFrame, which is why it appears in the count. The 14 numeric Table 20 rows in the same cell all match. |

---

## Notebook 08 — GEOM-Drugs structural fingerprint guidance

### Cell 10 — Table 3 against the paper (3 rows)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| multiMFM Tanimoto (4,096 × 3 rerun) | 0.227 | 0.224070 | by-design regeneration (§7 item 1) | `results/geom/table3_n4096_replicates.csv` (job **47833756**, 2026-09-22, 3 replicates × 4 shards of 1,024, `SRC/outputs/capgap_4096_3seed/`); `table3_per_molecule.csv` | The submitted 0.227162 is the λ = 0.02, μ = 0.5 point of the **n = 64, seed 0, four-point grid** (job **35670937**, `table3_n64_grid.csv`) — the best of four gains (+0.0199, +0.0103, +0.0018, −0.0087). The 4,096 × 3 rerun of the *same* configuration: per-replicate guided 0.224149 / 0.224159 / 0.223845 → 0.224051 ± 0.000103, 95 % t-CI [0.22361, 0.22449], which does **not** contain 0.227. Two things to separate: the *level* 0.227 is inside the n = 64 run's own SEM (guided mean 0.227162 ± 0.011883, n = 64), so the printed digit is not contradicted by its own run; the *sign of the gain* is what moves, +0.019945 ± 0.010391 (n = 64) vs −0.004570 ± 0.000805 (pooled 11,260) = **2.35 sigma**. `table3_n64_fragility.json` (200,000 bootstrap draws of 64 from the 11,260 population) puts the one-draw probability of reaching +0.0199 at **1.06 %**, 4.16 % as best of four — matching the 2.35-sigma figure. Caveat recorded there: the n = 64 targets were an easier draw (unguided 0.2072 vs 0.2287), so subsampling plus best-of-four explains part but not all. **Not settled**: see "In flight". |
| base PB validity (%) | 92.0 | 87.5 | paper-vs-records discrepancy | `table3_n64_grid.csv` `unguided_pb_valid`; `results/geom/table21_geom_row.json` | The unguided arm of the very run that produced the printed 87.5 % guided value is *also* 87.5 % — so the paper's Table 3 caption contrast ("87.5 % versus 92.0 % for its base generator") has no support in that run. 92.0 % appears in **no** 4,096-molecule record; see the Table 21 rows below for where it does appear. Independent of any rerun. |
| molecules evaluated (N) | 4096 | 64 | paper-vs-records discrepancy | `table3_n64_grid.csv` `num_mols` = 64; App. F.9 text in `results/geom/paper_values.json` `text_claims.num_molecules_evaluated` | App. F.9 states the GEOM evaluation generated 4,096 molecules and that the Tanimoto and PB numbers come from the same run. Both printed numbers come from a 64-molecule run. This is the root discrepancy §7 item 1 is about. |

### Cell 29 — Table 21, GEOM-Drugs row (5 rows)

| item | paper | ours | class | evidence | note |
|---|---|---|---|---|---|
| multiMFM PB validity (%), 4,096 × 3 | 87.5 | 84.9876 | within-noise rerun difference | `table3_n4096_replicates.csv` `guided_pb_valid` (job 47833756) | Per-replicate 86.035 / 83.887 / 85.059 → 84.9935 ± 0.6211, **95 % t-CI [82.32, 87.67] contains 87.5**. The submitted value's own binomial SE at n = 64 is 4.13 pp, so the two measurements are statistically compatible; the drop is not established. (§7 item 1 quotes the *within-rerun* fall 88.0 → 85.0, which is a different, paired comparison and is significant.) |
| unsteered PB validity (%) | 92.0 | 87.5 | paper-vs-records discrepancy | `table3_n64_grid.csv` `unguided_pb_valid`; `table21_geom_row.json` `records.selected_run_n64` | Same measurement as the Table 3 row above, under `configs/geom/posebusters_geom.yaml`. |
| unsteered PB validity (%), 4,096 × 3 | 92.0 | 88.0238 | paper-vs-records discrepancy | `table3_n4096_replicates.csv` `unguided_pb_valid` | Per-replicate 88.525 / 87.354 / 88.249 → 88.0425 ± 0.3536, 95 % t-CI **[86.52, 89.56] excludes 92.0**. So 92.0 % is not reachable even at 4,096 molecules on the released checkpoint — this is not a small-n effect. |
| unsteered PB validity (%), selected checkpoint at training validation | 92.0 | 98.0 | paper-vs-records discrepancy | `results/geom/teacher_training_pb_validity.csv` (wandb `val/pb_intersection` of the teacher run, N = 100, 100 steps, `posebusters_no_strain.yaml`); `table21_geom_row.json` `records.base_model_training_validation` | The *only* place 92.0 % occurs anywhere in the GEOM records is this training-time validation — at epochs **26, 28 and 37**, under a different N (100) and a different PB config (no-strain). The released epoch-38 checkpoint logged **98.0 %** there. The standalone `check_pb_validity.py` curve (N = 200) never covered step 89,310 and peaks at 90.5 %; the first GEOM-config measurement on the released checkpoint (job 49256841) gives 88.0 %. So 92.0 % corresponds to no measurement of the released model. |
| molecules evaluated (N) | 4096 | 64 | paper-vs-records discrepancy | `table21_geom_row.json` `records.selected_run_n64.N` | Same as the Table 3 N row; Table 21's GEOM row repeats the 4,096 claim for both percentages. |

---

## Rows that need an author decision

1. **Table 2 / Table 21 SS replicate mixing (6 rows).** §7 item 4 already decides the *fix* — rerun
   with 3 seeds and report mean ± SE — but the submitted table still has to be described honestly
   in the revision. Under one consistent rule (mean of available replicates) the submitted SS row
   reads **0.40 / 0.69 / 1.10 / 471 / 252 / 316** and PB **97.35 / 95.8 / 97.5 / 96.3 / 97.85 /
   97.8**; only Δε changes (462 → 471 MAE, 95.8 → 96.3 PB). Decide whether the paper reports the
   consistent mean of the submitted runs, the 3-seed rerun, or both.
2. **Which steer-search improvements survive (5 rows).** After the Algorithm-3 rerun on held-out
   seeds, only **μ, Δε and LUMO** improve beyond noise (10.2 / 3.7 / 3.6 sigma). **Cv, α and HOMO**
   do not (1.4 / 1.7 / 1.7 sigma). The revision needs to say which claim it is making. Related:
   §7 item 9 already flags that §3.2's "steer-search integrates the same guided ODE as multiMFM"
   is false (SS used μ = 11.0 against 3.4 for α), and the LUMO rerun adds a second μ change
   (1.5 → 3.0).
3. **Steer-search PB validity of 100.0 %.** Algorithm 3 as written makes PB validity a selection
   criterion over 3,680 candidates, so the honest number is 100.0 % on 18,000 molecules, not
   97.1 %. Decide how to present a validity metric that the method optimises by construction —
   and note that at the same time the guide MAE of the returned molecule collapses (Δε: guide MAE
   21–35 meV against oracle MAE 406–419 meV; `rerun_3seed_long.csv`), i.e. the selection is
   largely gaming the guide. `frac_selected_lookahead` ≈ 1.000 and `selected_t_mean` ≈ 0.60–0.74:
   the returned molecule is almost never the trajectory endpoint but an MFM one-shot posterior
   sample taken from t ≈ 0.7. That is exactly Algorithm 3 line 7, so it is not a bug, but it means
   the reported molecules are not draws from the t = 1 generative distribution.
4. **Table 3 / Table 21 GEOM N = 4,096 (2 rows) and the 92.0 % base PB (3 rows).** Both printed
   percentages and the Tanimoto come from 64 molecules; 92.0 % corresponds to no measurement of
   the released checkpoint. §7 item 1 covers the rerun; the caption sentence and App. F.9 still
   need rewriting whichever way the rerun lands.
5. **Table 20's QM9 objective** ("Diagonal + ESD" vs a stage-1 diagonal objective with no ESD
   term). §7 item 5 fixed the GEOM half of Table 20; the QM9 objective cell is not in that list
   and is still a live text fix.

## Rows that look like our bug, ranked by how much they would change a published number

**No row in either notebook is classified as a suspected port or analysis bug.** The two
candidates were checked directly and cleared, and one real defect exists in the shared helper. In
descending order of how much a published number would move:

1. **The GEOM negative-gain conclusion is not established, and the doubt is ours, not the
   paper's — would change Table 3's headline number and the abstract.** Three open threads, none
   resolved as of 2026-09-29 19:30:
   - The `grad-sign` diagnostic (§7 item 1) has produced nothing: neither
     `results/geom/grad_sign.json` nor `grad_sign_cpu.json` exists; job **49293660**
     (`serial_requeue`, the only route) is PENDING. Until it lands, a mis-signed or mis-scaled
     guidance term is not excluded.
   - The symptom that motivates the doubt is real: the *guide's own* head-predicted Tanimoto also
     falls at scale (`results/geom/head_tanimoto_direction.json`: pooled 0.23378 → 0.23079, gain
     −0.00299, 9 of 11 shards negative; all 5 held-out sweep points negative, head gain in
     [−0.0119, −0.0053]) while rising at the single submitted n = 64 point (+0.01261). A method
     that lowers the objective it is ascending is what a sign or scale defect looks like.
   - The published GEOM runs applied a **constant** μ, not the paper's `sigma_t^2/2 = (1-t)/t`
     (`results/geom/PROVENANCE.md`, "The guidance coefficient"). Inside the recorded window
     t ∈ [0.1, 0.9] that is 0.06× the stated coefficient at t = 0.1 and 4.5× at t = 0.9. The
     constant-vs-paper comparison (CPU jobs **49293829** / **49293743** / **49293831**) is PENDING.
   - **What I would check, in order:** (a) the `grad-sign` verdict; (b) the CPU constant-vs-paper
     coefficient comparison at λ = 0.01, which shares a device and so carries no TF32 confound;
     (c) λ = 1.0, the literal reading of the paper's reward `exp(-||E(G)-c||^2)`, which is
     *unmeasured* on the held-out pool (it was in the 10 cancelled GPU grid points).
2. **The held-out tuning grid is too noisy to select on — would change which (λ, μ) Table 3
   reports.** `results/geom/table3_asdescribed_tuning.csv` selects on 5 points of n = 256 whose
   per-point `paired_gain_sem` is 0.0043–0.0053, while the effect being selected is of order
   0.001–0.005. The selected point (λ = 0.01, μ = 0.25) measured −0.0070 ± 0.0050 at n = 256 and
   is currently measuring **+0.00066 ± 0.00071** at n = 8,704 (see "In flight") — a 1.5-sigma
   reversal that the selection could not have resolved. What I would check: re-measure the grid at
   n ≥ 2,048 per point, or select on the paired gain's t-statistic rather than its point estimate.
3. **`compare()`'s tolerance is 10× too loose for paper values with a trailing zero — changes no
   published number, but it makes these notebooks *under*-report mismatches.**
   `notebooks/_common.py` infers the paper's printed precision from `repr(float(paper))`, so
   `0.40` → `'0.4'` → 1 decimal → tolerance 0.05 instead of 0.005, and `92.0` → 0 decimals →
   tolerance 0.5 instead of 0.05. Verified for every paper value used in these two notebooks.
   Three rows in notebook 07 currently read **match** but are MISMATCH at the paper's printed
   precision: `SS replicate b Cv` (0.40 vs 0.405983), `SS replicate b α` (1.10 vs 1.056593), and
   `Table 2 multiMFM-SS Cv` (0.40 vs 0.413920). It cannot turn a MISMATCH into a match, so no row
   above is affected in the other direction, and all three would fall into classes already present
   (replicate-selection artifact, replicate-selection artifact, within-noise). I do not own
   `_common.py`; the change needed is to take the decimal count from the transcribed string rather
   than the float, or to store printed precision alongside each value in `paper_values.json`.
4. **Checked and cleared — the two things that would have invalidated the steer-search rerun.**
   Recorded here so they are not re-litigated.
   - *Did the Algorithm-3 selection leak the held-out oracle?* No. Only `guide_reg` reaches the
     scorer (`src/multimfm/steer_search.py:373`, inside `guided_sample_collect`); `oracle_reg` is
     constructed at `:463` and used only in `evaluate_state` (`:404`). A leak would have explained
     the 10.2-sigma μ improvement; it does not exist.
   - *Is the 100 % PB validity a carried-over flag?* No. `lazy_select` (`:242`) checks candidates
     with `pool.valid(decode(...))` in error order (`:279`), and the returned state is then
     re-evaluated from scratch by `evaluate_state` → `molecule_records` (`:404`), which recomputes
     PoseBusters. The 100.0 % is 18,000 independent PoseBusters passes.
   - The submitted multiMFM row itself reproduces exactly from the stored run files (notebook cell
     7: all six properties **match**), which is the strongest available evidence that the QM9 port
     is faithful.

## In flight — read this before quoting the GEOM rows

The as-described 4,096 × 3 evaluation (§7 item 1, Slurm array **49274410**, `kempner_requeue`) was
**still RUNNING** during this audit: 7 of 24 tasks running, 17 shard JSONs on disk in
`results/geom/runs/asdescribed_eval/` (directory mtime 2026-09-29 19:28:27).
`results/geom/table3_asdescribed.json` does not exist yet, so notebook 08's cells 21 and 23 render
"PENDING" and contribute no MISMATCH rows. A read-only pool of the 17 landed shards (λ = 0.01,
μ = 0.25, guide every 2nd outer step in t ∈ [0.1, 0.9], 8,704 paired molecules) gives:

| quantity | value |
|---|---|
| unguided FP2 Tanimoto | 0.226735 |
| guided FP2 Tanimoto | 0.227392 |
| molecule-weighted paired gain | **+0.000658** (sd over shards 0.002933, SEM 0.000711) |
| negative-gain shards | 8 of 17 |
| head-predicted Tanimoto | 0.231924 → 0.232508, gain **+0.000584** (7 of 17 shards negative) |
| PB validity | 87.707 % → 86.960 % |

So at the held-out-selected setting the gain is currently **indistinguishable from zero and
slightly positive**, in both the counted and the head-predicted metric — the opposite sign from the
λ = 0.02, μ = 0.5 rerun (−0.004570 ± 0.000805) that §7 item 1 is written around. Caveats: replicate
2 contributes only 1 of its 8 shards and is the most positive (+0.00524); dropping replicate 2
leaves +0.00037. This does not settle anything either, but it means **§7 item 1's "guidance is
negative at every setting tried" should not be quoted as final**, independently of the `grad-sign`
diagnostic. Re-pool from `table3_asdescribed.json` once the array finishes.

## Files read, with mtimes (audit is reproducible against these)

| file | mtime (2026-09-29) |
|---|---|
| `notebooks/07_qm9_property_guidance.ipynb` | 15:52:23 |
| `notebooks/08_geom_fingerprint.ipynb` | 15:55:35 |
| `notebooks/_common.py` | 16:35:29 |
| `docs/REPRODUCIBILITY_PLAN.md` | 19:23:16 |
| `src/multimfm/steer_search.py` | 14:54:04 |
| `results/qm9/PROVENANCE.md` | 15:52:58 |
| `results/qm9/paper_values.json` | 13:18:46 |
| `results/qm9/table2.csv` | 13:17:54 |
| `results/qm9/table21.csv` | 13:17:54 |
| `results/qm9/table20_mfm_training.json` | 13:17:56 |
| `results/qm9/rerun_3seed/rerun_3seed_summary.csv` | 15:51:32 |
| `results/qm9/rerun_3seed/rerun_3seed_long.csv` | 15:51:32 |
| `results/geom/PROVENANCE.md` | 18:03:44 |
| `results/geom/paper_values.json` | 13:18:43 |
| `results/geom/table3_n64_grid.csv` | 13:14:32 |
| `results/geom/table3_n4096_replicates.csv` | 13:14:32 |
| `results/geom/table3_per_molecule.csv` | 13:28:24 |
| `results/geom/table3_n64_fragility.json` | 14:47:45 |
| `results/geom/table21_geom_row.json` | 13:18:26 |
| `results/geom/head_tanimoto_direction.json` | 15:20:00 |
| `results/geom/table3_asdescribed_tuning.csv` | 18:01:13 |
| `results/geom/table3_asdescribed_selection.json` | 18:01:13 |
| `results/geom/runs/asdescribed_eval/` (17 shards) | 19:28:27 |

One staleness note: `results/geom/table3_asdescribed_{tuning.csv,selection.json}` were rewritten at
18:01, after notebook 08 was executed at 15:55. The five rows' *values* are unchanged (verified
against the notebook's rendered cell 19 to 4 decimals); the rewrite added the `mu_schedule`,
`device` and `slurm_*` columns. No MISMATCH row depends on them.
