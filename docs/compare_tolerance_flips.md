# `compare()` tolerance fix: which rows flipped

Written 2026-09-29 after `notebooks/_common.py::compare()` stopped inferring the paper's printed
precision from `repr(float(paper))`. Scope: all ten `notebooks/*.ipynb`, 612 `compare()` rows.

**Headline: 4 rows flipped `match` -> MISMATCH. Three were predicted by
`docs/mismatch_audit_molecules.md`; the fourth was not, and it is a real disagreement, not a
tolerance artifact. No row flipped MISMATCH -> `match`. No `ours` value was changed.**

| | count |
|---|---|
| `compare()` rows, all notebooks | 612 |
| flipped `match` -> MISMATCH | **4** |
| flipped MISMATCH -> `match` | 0 |
| rows still `tol_src == "inferred"` | **0** |
| final status | 498 match / 107 MISMATCH / 7 n/a |

---

## 1. The flips

Pre-fix tolerances are reconstructed from the behaviour the audit documents (decimals taken from the
float with trailing zeros already lost, so `0.40` -> `0.4` -> 1 decimal -> 0.05, and `96.0` -> `96`
-> 0 decimals -> 0.5). All four are in `07_qm9_property_guidance.ipynb`.

| item | paper | ours | old | new | tol before | tol after | anticipated? |
|---|---|---|---|---|---|---|---|
| `SS replicate b cv [cal/(mol K)]` (cell 9) | `0.40` | 0.405983 | match | **MISMATCH** | 0.05 | 0.005 | yes — audit item 3 |
| `SS replicate b alpha [Bohr^3]` (cell 9) | `1.10` | 1.056593 | match | **MISMATCH** | 0.05 | 0.005 | yes — audit item 3 |
| `Table 2 multiMFM-SS cv [cal/(mol K)]` (cell 13, 3-seed rerun) | `0.40` | 0.413920 | match | **MISMATCH** | 0.05 | 0.005 | yes — audit item 3 |
| `T21 SS mean(a,b) mu` (cell 26) | `96.0` | 95.80 | match | **MISMATCH** | 0.5 | 0.05 | **no** |

All four now report `tol_src == "printed"`: the tolerance is half a unit in the last place the paper
actually prints, taken from the transcribed string.

### Verdict on each

* The three audit-predicted rows are the trailing-zero case the fix was written for. Table 2 prints
  multiMFM-SS as `0.40` and `1.10`; at 2 decimals the disagreements (0.005983, 0.043407, 0.013920)
  exceed half a unit in the last place. The audit already classified them — two
  replicate-selection artifacts and one within-noise rerun difference — so they add no new class of
  problem, only honest bookkeeping. This is the intended effect of the fix.

* **`T21 SS mean(a,b) mu` is the unanticipated flip, and it is genuine.** From
  `results/qm9/table21.csv`, the mu row's SS PB validity is 95.6 % in replicate *a* and 96.0 % in
  replicate *b*; `ss_published_pb_source` for mu is `b`. So the paper's printed `96.0` is replicate
  *b* alone, while the mean of the two available replicates is 95.80. Table 21 prints one decimal
  ("All percentages are rounded to one decimal place", paper.txt 2915-2916), so at the paper's own
  precision 95.80 != 96.0. This is not a new defect: it is the already-recorded "Table 2/21 SS row
  mixes replicates `k_*_a` / `k_*_b` inconsistently" discrepancy, which was previously visible only
  on the `gap` row (published `95.8` = replicate *a*, mean(a,b) = 96.3, MISMATCH before and after
  the fix) and is now visible on `mu` as well. The pre-fix tolerance of 0.5 — a whole half-percent
  on a quantity printed to a tenth — was hiding it. Nothing in `ours` changed; only the tolerance
  did.

No flip points at a computation that is wrong. Three are precision bookkeeping on known
discrepancies; the fourth exposes one more cell of a replicate-mixing problem already on the record.

---

## 2. Rows still `tol_src == "inferred"`: zero

The first re-execution with the fixed `compare()` left 162 rows `inferred`. Every one has been
resolved by transcribing the paper's printed literal, so the final tolerance breakdown is:

| `tol_src` | rows | where |
|---|---|---|
| `atol` | 443 | DNA notebooks (02-06, 09) via `_dna_helpers.cmp()`, which already passed an explicit `dec`/`atol` per row |
| `printed` | 158 | paper literal now stored as a string in `results/*/paper_values.json` |
| `decimals` | 4 | values the paper prints in scientific notation (below) |
| `inferred` | **0** | — |
| (blank) | 7 | `status == "n/a"`: no paper value, or `ours` not yet computed |

Four rows cannot use a printed string, because a decimal count is meaningless for a number printed
in scientific notation. They pass `decimals=` instead, which is explicit rather than inferred:

| row | paper prints | `decimals` | tol |
|---|---|---|---|
| `01` Peak learning rate (Table 12) | `3 x 10^-4` | 4 | 5e-5 |
| `01` ESD Peak learning rate (command line) | `1e-4` | 4 | 5e-5 |
| `07` learning rate (Table 19, QM9) | `2 x 10^-3` | 3 | 5e-4 |
| `07` peak LR (Table 20, QM9) | `10^-6` | 6 | 5e-7 |

The last one is worth noting: inferring from the float gave `repr(1e-06) == "1e-06"`, no decimal
point, 0 decimals, **tolerance 0.5 on a value of 1e-6**. The row matched only because the stored
value is exactly equal. `tol_src == "inferred"` is what made that visible.

---

## 3. A residual defect in the fixed `compare()` (integers become 10x too strict)

`compare()` now does `paper = float(paper_raw)` before `repr()`, so a paper value transcribed as a
JSON **integer** gains a spurious `.0`: `repr(float(600)) == "600.0"` -> 1 decimal -> tolerance
0.05, where the paper prints `600` and the correct half-unit is 0.5. The pre-fix code gave 0
decimals here. So for integers the fix is over-strict, not too loose.

This showed up as 10 rows that went MISMATCH in the intermediate run and back to `match` once the
value was transcribed as the string `"600"` etc. They were `match` before the fix and are `match`
now, so they are **not** flips — but they must not be mistaken for the tolerance being widened to
force a pass. 0.5 is half a unit in the last place the paper prints; 0.05 was never the paper's
precision.

| item (nb 07) | paper | ours | tol intermediate | tol final |
|---|---|---|---|---|
| `Table 2 multiMFM gap [meV]` (c7) | `600` | 599.566363 | 0.05 `inferred` | 0.5 `printed` |
| `Table 2 multiMFM homo [meV]` (c7) | `262` | 262.328058 | 0.05 | 0.5 |
| `Table 2 multiMFM lumo [meV]` (c7) | `392` | 391.916283 | 0.05 | 0.5 |
| `SS as published gap [meV]` (c9) | `462` | 462.283956 | 0.05 | 0.5 |
| `SS as published homo [meV]` (c9) | `252` | 252.204866 | 0.05 | 0.5 |
| `SS as published lumo [meV]` (c9) | `316` | 315.729827 | 0.05 | 0.5 |
| `SS mean(a,b) homo [meV]` (c9) | `252` | 252.204866 | 0.05 | 0.5 |
| `SS mean(a,b) lumo [meV]` (c9) | `316` | 315.729827 | 0.05 | 0.5 |
| `SS replicate a gap [meV]` (c9) | `462` | 462.283956 | 0.05 | 0.5 |
| `SS replicate b lumo [meV]` (c9) | `316` | 315.729827 | 0.05 | 0.5 |

Every notebook row is now immune (all integer paper values are strings), but the defect remains for
any future bare-int `paper`. Suggested change in `notebooks/_common.py` (not owned by this work):
in the inference branch, take `repr(paper_raw)` rather than `repr(float(paper_raw))`, so an `int`
yields 0 decimals. Reproduce with:

```python
compare([{"item": "x", "paper": 600,   "ours": 599.566363}])  # MISMATCH, tol 0.05, inferred
compare([{"item": "x", "paper": "600", "ours": 599.566363}])  # match,    tol 0.5,  printed
```

---

## 4. What changed on disk

Only the printed form of already-transcribed paper numbers, plus the notebook plumbing that carries
it. **No `ours` value, no results data, and no tolerance beyond the paper's own printed precision.**

Paper numbers that the paper prints with a fixed number of decimals are now stored as **strings**
holding the printed literal (thousands separators dropped so `float()` parses them). A JSON number
cannot carry a trailing zero, which is the whole cause of the original defect. Each file carries a
`_precision` note explaining this.

* `results/qm9/paper_values.json` — Table 2 (all 16 rows), Table 21, Table 22, App. F.9 text
  claims, Table 19 and Table 20 QM9 columns. Six values had genuinely lost their printed trailing
  zero in the float transcription and are now correct: `Lower bound cv` 0.04 -> `"0.040"`,
  `EEGSDE alpha` 2.5 -> `"2.50"`, `Cond-Flow alpha` 3.1 -> `"3.10"`, `TFG alpha` 3.9 -> `"3.90"`,
  `multiMFM-SS cv` 0.4 -> `"0.40"`, `multiMFM-SS alpha` 1.1 -> `"1.10"`. The last two are what the
  three predicted flips turn on.
* `results/toy/paper_values.json` — `table12_printed`, `table12_missing_esd_stage`.
* `results/geom/paper_values.json` — `table3` rows and PB percentages, `text_claims.num_molecules_evaluated`.
* `results/geom/table21_geom_row.json` — `paper.N`, `paper.unsteered_pct`, `paper.multimfm_pct`
  (the `records.*` recomputed values stay numeric).
* `notebooks/01_toy_multimodal.ipynb` — `decimals=4` for the two scientific-notation learning rates.
* `notebooks/07_qm9_property_guidance.ipynb` — Table 22 counts kept as printed strings instead of
  `int()`; `"paper": "50000"` for the training-set size; `decimals=3` / `decimals=6` for the two
  scientific-notation learning rates.
* `notebooks/08_geom_fingerprint.ipynb` — dropped two `float()` casts that would have discarded the
  printed literal.

Sources for every literal: paper.txt 350-364 (Table 2), 385-391 (Table 3), 1460-1489 (Table 12),
1840-1864 (Table 19), 1893-1912 (Table 20), 1916-1923 (Table 21), 2918-2925 (Table 22), 1874-1876
(App. F.9); ESD-stage values from the verbatim command in
`docs/provenance/toy/recovered/cursor_terminal_1.txt` lines 30-40, which the paper omits.

---

## 5. Execution log

Jobs were landing throughout, so each notebook's execution time is recorded. `RERUN = False`, login
node, `multimfm` kernel, `--ExecutePreprocessor.timeout=1800`. Final pass, 2026-09-29:

| notebook | start (local) | elapsed | error cells |
|---|---|---|---|
| `00_setup` | 20:52:01 | 23 s | 0 |
| `01_toy_multimodal` | 20:52:24 | 142 s | 0 |
| `02_dna_base_model` | 20:54:46 | 7 s | 0 |
| `03_dna_cyclizability` | 20:54:53 | 6 s | 0 |
| `04_dna_exact_rewards` | 20:54:59 | 5 s | 0 |
| `05_dna_mc_scaling` | 20:55:04 | 6 s | 0 |
| `06_dna_benchmarks` | 20:55:10 | 5 s | 0 |
| `07_qm9_property_guidance` | 20:55:15 | 8 s | 0 |
| `08_geom_fingerprint` | 20:55:23 | 4 s | 0 |
| `09_evo2_plausibility` | 20:55:27 | 2 s | 0 |

**Zero error cells in all ten notebooks.** Cells whose inputs have not landed still print `pending`
as intended (02 cells 22/23/25/27; 03 cells 14/42/43/47/48; 04 cell 9; 05 cell 13; 08 cells 15/21;
09 cells 0/15/16/19). `06_dna_benchmarks` contributes no `compare()` rows at all: both its `cmp()`
calls sit behind `if pa is not None` / `if pb is not None` guards and the Fig. 4 rerun (Slurm
`49298117`) was still PENDING. Those two tables will need re-checking when it lands.

### One `ours` change, and it is not mine

Two rows in `01_toy_multimodal.ipynb` cell 13 moved because `results/toy/steering_metrics.json` was
regenerated at 18:19 — after that notebook's previously stored outputs (16:44) and before any of my
runs. Recorded so it is not read as a tolerance effect:

| item | ours in the stored 16:44 outputs | ours on re-execution | status |
|---|---|---|---|
| `Smiley GLASS, mean reward` | 0.586990 | 0.561365 | match, both |
| `Smiley GLASS, colour match` | 0.988700 | 0.956200 | match, both |

Both rows use `compare(..., atol=0.03, rtol=0.06)`, so `tol_src == "atol"` and the tolerance fix
does not touch them.

---

## 6. Cross-check against the two audits

* `docs/mismatch_audit_molecules.md` item 3 predicted exactly three nb 07 rows would turn out to be
  MISMATCH at printed precision: `SS replicate b Cv` (0.40 vs 0.405983), `SS replicate b alpha`
  (1.10 vs 1.056593), `Table 2 multiMFM-SS Cv` (0.40 vs 0.413920). **All three flipped, with those
  exact values.** The audit's claim that the defect "cannot turn a MISMATCH into a match" also
  holds: zero rows moved in that direction.
* The audit's worked example `92.0 -> 0 decimals -> tolerance 0.5` is confirmed as the pre-fix
  behaviour and is the mechanism behind the fourth, unanticipated flip (`T21 SS mean(a,b) mu`,
  96.0 vs 95.80). The audit checked "every paper value used in these two notebooks" for the
  *trailing-zero* direction but did not enumerate the integral-float case, which is why this one
  was missed.
* `docs/mismatch_audit_dna.md` predicted no tolerance flips, and there were none: the DNA notebooks
  go through `_dna_helpers.cmp()`, which already passed an explicit `dec` per row precisely because
  `compare()` "infers decimals from the float, which drops trailing zeros". All 443 DNA rows report
  `tol_src == "atol"` and not one changed status. The audit's most marginal DNA row
  (`+1_unguided guide_score_mean`, -0.1766 vs -0.176716, "2.3x the printed-precision tolerance")
  was MISMATCH before and remains MISMATCH.

## 7. Verdict

No flip is a genuine computational problem. Three are the intended effect of the fix on
disagreements the molecules audit had already found and classified. The fourth, `T21 SS mean(a,b)
mu`, is a real disagreement at the paper's printed precision, but it belongs to the known Table 2/21
replicate-mixing discrepancy rather than to a new fault — the paper's `96.0` is replicate *b*, and
the mean of the two replicates is 95.80. What the fix mainly bought is the `tol_src` column: it
turned 162 silently-inferred tolerances into 158 tolerances read off the paper plus 4 stated
explicitly, and it caught one tolerance of 0.5 sitting on a value of 1e-6.
