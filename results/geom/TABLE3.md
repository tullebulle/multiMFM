# Table 3 (GEOM-DRUGS fingerprint guidance): where every number comes from

All rows are the directly counted FP2 Tanimoto between a generated molecule and its target
fingerprint. No learned evaluator is involved, so there is no guide/oracle gap to argue about.

| table row | value | source |
|---|---|---|
| floor, <= 71 atoms | 0.21951 +- 0.00019 | `results/geom/rescue/fp2_population.json`, 200k random real pairs, 20-30 atoms x targets <= 71 (the Table 3 pairing) |
| base generator, <= 71 | 0.2347 +- 0.0027, PB 0.8646 | unguided arm of `runs/fp_confirm2/*ge1_mc8*`, 3 seeds x 256 |
| **multiMFM, <= 71** | **0.2737 +- 0.0050, PB 0.8893** | guided arm of the same runs |
| floor, <= 30 atoms | 0.20991 +- 0.00019 | same population file, <= 30 x <= 30 (TFG-Flow's cap) |
| TFG-Flow uncond. / guided | 0.170 +- 0.001 / 0.208 +- 0.002 | reported in their paper; cap verified as `GEOMDrugDataset.max_length = 30` in their checkout |
| base generator, <= 30 | 0.2270 +- 0.0046 | unguided arm of `runs/fp_matched30/*ours_max30*`, 3 seeds x 256 |
| **multiMFM, <= 30** | **0.2516 +- 0.0073, PB 0.8880** | guided arm of the same runs |

Margins printed in the table are value minus the floor of that block.

## Settings

| | <= 71 block | <= 30 block |
|---|---|---|
| reward | soft Tanimoto (`fingerprint_tanimoto`) | same |
| lambda | 20 | 20 |
| gamma (`mu`) | 2.0 | 2.0 |
| flow-map steps | 2 | 2 |
| posterior samples | 8 | 4 |
| guided every | 1 step | 2 steps |
| window | t in [0.1, 0.9] | same |
| outer trajectory | 100 steps | same |
| weights | EMA | same |

The matched-cap block runs a *cheaper* configuration than the full-scale one, so the comparison
to TFG-Flow is made at less compute than our best setting, not more.

## Three things the table is designed to prevent

1. **Reading the raw Tanimoto as if 0 were the baseline.** It is not: two unrelated real
   molecules of this size already score ~0.22, because a path-based fingerprint pays for
   genericness. Hence the floor row and the margin column.
2. **Comparing across atom caps.** TFG-Flow caps at 30 heavy atoms and we generate up to 71.
   The lower block repeats our experiment under their cap so the comparison is like for like.
3. **Comparing across averaging conventions.** They score every generated sample and count an
   unconvertible one as 0; we report over convertible molecules. Our conversion rate in these
   runs is 1.0000, so the two conventions give us the same number -- which is why the lower
   block can honour theirs at no cost. Stated in `MATCHED30.md` as well.

## What is not claimed

A win on relative improvement. Ours is +16.58% +- 0.95 (<= 71) and +10.76% +- 0.95 (<= 30) over
the matching unguided arm; TFG-Flow reports +22.4%. Relative gain rewards a weak baseline, and
their unconditional 0.170 is below the floor of its own size regime -- which their
count-invalid-as-zero convention is enough to explain on its own.

## Reproduce

```bash
sbatch scripts/geom/fp_confirm2.sbatch         # the <= 71 block
sbatch scripts/geom/fp_matched_protocol.sbatch # the <= 30 block
```
