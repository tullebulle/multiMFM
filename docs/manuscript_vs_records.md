# The working manuscript's own-results do not match the records, in two tables

Found 2026-10-01 while reviewing wording. This is a provenance question, not a wording one,
and it touches the headline results table.

## What agrees with what

For QM9 Tables 2 and 21 there are three sources:

| source | what it is |
|---|---|
| `dMFM (40).pdf`, transcribed into `results/qm9/paper_values.json` | the numbers the published PDF prints |
| `results/qm9/table2.csv`, `table21.csv` | the 2026-09 reruns, built by `qm9_results tables` from `runs/` |
| `iclr2027_conferencemod.tex` | the working manuscript in this repo |

**The published PDF and the reruns agree with each other**, to the digit, on every value
checked -- 12/12 in Table 21 and 12/12 in Table 2. That is the strongest provenance available
and it means the reruns reproduced the publication.

**The working `.tex` agrees with neither.**

### Table 21 (PoseBusters validity)

| | published = rerun | working `.tex` |
|---|---|---|
| multiMFM mu / alpha / HOMO / LUMO | 91.6 / 91.3 / 92.3 / 92.2 | 90.6 / 90.5 / 91.5 / 90.9 |
| multiMFM-SS, all six | 97.4 / 96.0 / 97.5 / 95.8 / 97.8 / 97.8 | **100.0 +- 0.00** |

Only Cv's multiMFM cell matches. The prose at line 2599 repeats the 100.0 claim.

### Table 2 (oracle MAE) -- the same pattern, and it matters more

| property | working `.tex` SS | published = rerun SS |
|---|---|---|
| Cv | 0.41 | 0.40 |
| mu | **0.53** | 0.69 |
| alpha | **1.01** | 1.10 |
| gap | **413** | 462 / 471 |
| HOMO | **243** | 252 |
| LUMO | **265** | 316 |

The working `.tex` is better than the verified rerun on **5 of 6** SS properties, by wide
margins. For plain multiMFM it is better on 2 of 6 and scattered both ways, which looks like
ordinary run-to-run variation. The SS row is the systematic one.

## What is known about where the `.tex` numbers came from

* `git log -S` places both disputed strings (`100.0 \pm 0.00`, `90.6 \pm 0.13`) in **d2d6c2a**,
  "Add the reproducibility plan, audits and working manuscript copy" -- the commit that first
  added the `.tex` to this repo. They were present on arrival.
* Nothing in `results/` produces them. No run on record reports SS gap 413, LUMO 265 or mu 0.53.
* The working `.tex` carries `+-` error bars that the published values lack, so it is plausibly
  a **later** revision built from multi-seed runs that were never committed here.

## What would settle it

1. The Overleaf history for `iclr2027_conferencemod.tex`: if a later revision replaced Tables 2
   and 21 together, its date identifies the runs.
2. Or the runs themselves -- anything producing SS gap 413 / LUMO 265 / mu 0.53.

Until one of those exists, **neither table should be edited**: replacing the `.tex` values with
the rerun values would be correct if the `.tex` is stale, and would be deleting a newer result
if it is not. The two cases are indistinguishable from inside this repository.

## One thing that is independently true

In every SS run on record, `pb_pb_valid` equals `frac_with_valid_candidate` exactly (0.95-0.97).
So the SS validity figure is "the fraction of molecules for which at least one look-ahead passed
PoseBusters" -- a selection-conditioned quantity. When no candidate passes, SS returns the
endpoint and the molecule is invalid. Whatever number the table ends up carrying, the text should
say that, because 100% validity would otherwise read as a property of the method rather than of
the selection rule. See `docs/ss_guide_oracle_gap.md`, which raised the same point.
