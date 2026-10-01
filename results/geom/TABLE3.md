# Table 3 (GEOM-DRUGS fingerprint guidance): where every number comes from

All rows are the directly counted FP2 Tanimoto between a generated molecule and its target
fingerprint. No learned evaluator is involved, so there is no guide/oracle gap to argue about.

| table row | value | source |
|---|---|---|
| floor, <= 71 atoms | 0.21951 +- 0.00019 | `results/geom/rescue/fp2_population.json`, 200k random real pairs, 20-30 atoms x targets <= 71 |
| base generator, <= 71 | 0.2288 +- 0.0006, PB 87.5 | unguided arms, `runs/fp_table3_n4096/ours_n4096_*` |
| **multiMFM, <= 71** | **0.2760 +- 0.0016, PB 89.5** | guided arms of the same runs |
| floor, <= 30 atoms | 0.20991 +- 0.00019 | same population file, <= 30 x <= 30 (TFG-Flow's cap) |
| TFG-Flow uncond. / guided | 0.170 +- 0.001 / 0.208 +- 0.002 | their paper; cap verified as `max_length = 30` in their checkout |
| base generator, <= 30 | 0.2259 +- 0.0004, PB 87.5 | unguided arms, `runs/fp_table3_n4096/ours_max30_n4096_*` |
| **multiMFM, <= 30** | **0.2488 +- 0.0005, PB 89.0** | guided arms of the same runs |

Each row is **three replicates of 4,096 molecules** (12 shards of 1,024, pooled per molecule), on
twelve seeds never used for tuning. Full breakdown in `results/geom/table3_n4096_confirmed.json`.
Relative gains are +20.65% +- 0.36 (<= 71) and +10.12% +- 0.15 (<= 30); every one of the six
replicates is individually positive.

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

## A negative that is not in the paper (2026-10-01)

At matched inference cost on this task, **best-of-N beats both guided methods**:

| method | Tanimoto | NFE | PB |
|---|---|---|---|
| multiMFM ge2/s2/mc1 | 0.2396 | 238 | 0.898 |
| DPS, tuned (gamma = 1) | 0.2400 | 238 | 0.818 |
| best-of-2 | **0.2586** | 200 | 0.979 |
| multiMFM ge2/s2/mc4 | 0.2523 | 652 | 0.872 |
| best-of-4 | **0.2914** | 400 | 1.000 |
| best-of-10 | **0.3281** | 1,000 | 1.000 |

DPS was tuned on its own grid and peaks at gamma = 1 (+5.53%), collapsing to -12.75% at gamma = 8,
so this is not an under-tuned baseline. Sources: `runs/fp_nfe_sweep/`, `runs/fp_dps_tune/`.

**The comparison is sound, not broken.** best-of-1 performs no selection and must therefore
reproduce the unguided arm; it returns 0.2300 +- 0.0027 against the unguided 0.2275, a difference
of 0.0025 against a threshold of 0.006 fixed before the control was run
(`runs/fp_bon_control/`, `scripts/geom/fp_bon_control.sbatch`).

So the headline above is true as stated -- guidance beats its own unguided arm, and beats
TFG-Flow -- but on this task a trivial search baseline reaches higher absolute similarity for less
compute, and at higher validity. The paper currently makes no claim about GEOM baselines. Deferred
by the authors on 2026-10-01 for later discussion; it is deferred, not refuted.
