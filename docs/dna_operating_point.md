# dMFM's own operating point for every DNA result: the (steps, MC) split at matched NFE

Status 2026-09-30, complete for L = 50. Fig 7 and Table 13 ran on an H200 at full n; Table 18, Fig 4 and Fig 8 are CPU runs at reduced n. Owner: the operating-point agent of the 2026-09-30 round.
Companions: `docs/dmfm_deficit_rescue.md` (where the split was found and the operating point was
chosen), `docs/dmfm_glass_parity.md` (the two wiring defects), `docs/REPRODUCIBILITY_PLAN.md`
section 7 item 8 (the decision record and its three conventions).

## What this document is for

The dMFM value gradient costs **`steps x MC`** network calls: `steps` composed flow-map jumps of
the posterior, `MC` draws from it. One NFE budget can therefore be spent many ways, and the
deficit-rescue round found that Table 1's dMFM row had been spending it *GLASS's* way -- 4
posterior steps with MC 8, which is the GLASS Table 1 estimator -- when a one-step flow map
affords MC 32 at the same 32 calls. Moving Table 1's dMFM row from 4 x 8 to 1 x 32 takes the
guided oracle MAE from 0.193 / 0.172 to about 0.146 / 0.120, i.e. from a 6-9 SEM deficit against
matched GLASS to a tie.

This document asks the same question of **every other DNA result**, and answers three things per
result: what split it currently uses, what the NFE-matched grid over `(steps, MC)` gives, and
where the optimum is.

## The honesty rules this follows

The split is a property of the method, like a step size: choosing it well is correct practice,
choosing it *per result after seeing that result's outcome* is not. So:

1. **The operating point was chosen once, before any of the runs below.** It was chosen on the
   Table 1 C0 task -- `results/dna/deficit_rescue/nfe_L50_n200/`, the sweep of
   `docs/dmfm_deficit_rescue.md` section 3, at `y* = +1` with `y* = -1` agreeing -- and it is
   **dMFM = 1 composed step x MC 32** and **GLASS = 2 Euler steps x MC 16**, both at NFE 32. It
   is not re-chosen for any table here. Where a grid below prefers something else, that is
   reported as a finding, not adopted.
2. **GLASS gets the same courtesy wherever a GLASS comparison appears**: its own best split at
   matched NFE, which is 2 x 16 and not the 4 x 8 it was given. Both arms also get the same
   scale-free gradient rule (convention 2), or neither does. The items where a GLASS arm exists are
   Fig 7, Fig 8 and Table 18, and all three grids below run the full four-point split ladder on the
   GLASS side too. Table 13's producer has no GLASS path, and Fig 4 (right) compares against
   best-of-N / Feynman-Kac / beam / MCTS, each of which has its own budget knob and its own
   pre-registered pilot (`dmfm.benchmarks.plan`), so there is no GLASS arm to re-split there.
3. **Full grids and paired statistics with SEMs on shared ids** are reported, not just the best
   point. Every CI is a 2,000-resample percentile bootstrap with `default_rng(0)`, the method
   behind Table 1 and behind the deficit-rescue tables.

The chosen grid at a budget of NFE `B` is `(steps, MC) = (1, B), (2, B/2), (4, B/4), (8, B/8)`.
Convention 1 fixes the artifact per step count: `paths.dmfm_ckpt_for_steps(L, steps)`, i.e. the
**diagonal** student at one composed step and the 4-step ESD student above that.

## The anchor: the choice, and the statistics it was made on

Re-analysed from `results/dna/deficit_rescue/nfe_L50_n200/` into
`results/dna/operating_point/table1_anchor/` (no new compute). n = 200 paired per target, one
H200, every configuration under the scale-free rule at `guidance_frac 8`, guided oracle MAE:

| NFE | split | dMFM y* = +1 | dMFM y* = -1 | GLASS y* = +1 | GLASS y* = -1 |
|---|---|---|---|---|---|
| 8 | 1 x 8 | 0.1830 | 0.1454 | -- | -- |
| 16 | 2 x 8 | 0.1965 | 0.1468 | 0.1403 | 0.1381 |
| **32** | **1 x 32** | **0.1516** | **0.1103** | *not measured* | *not measured* |
| 32 | 2 x 16 | 0.1694 | 0.1221 | **0.1411** | **0.1178** |
| 32 | 4 x 8 | 0.1955 | 0.1717 | 0.1456 | 0.1303 |
| 32 | 8 x 4 | *not measured* | *not measured* | *not measured* | *not measured* |
| 64 | 8 x 8 | 0.1932 | 0.1809 | 0.1878 | 0.1601 |

Paired on shared ids, dMFM 1 x 32 against dMFM 4 x 8: **-0.0439 +- 0.0150** at y* = +1 (CI
[-0.0750, -0.0149]) and **-0.0614 +- 0.0118** at y* = -1 (CI [-0.0837, -0.0383]), i.e. 2.9 and
5.2 SEM. GLASS 2 x 16 against GLASS 4 x 8: -0.0046 +- 0.0114 and -0.0126 +- 0.0107, i.e. GLASS
barely cares. So **dMFM = 1 x 32, GLASS = 2 x 16** is the operating point, chosen here and reused
unchanged below. Two corners of that grid were never run even in this sweep -- 8 x 4 on either
arm, and 1 x 32 on the GLASS arm -- and the grids below supply them.

## Verdict table

| paper item | current split | best split at matched NFE | number now | number at the optimum | move significant? |
|---|---|---|---|---|---|
| **Table 1 / Fig 3** (settled before this round) | 4 x 8 | **dMFM 1 x 32**, GLASS 2 x 16 | oracle MAE 0.193 / 0.172 (n=1000) | **0.146 / 0.120** (n=600) | **yes** -- paired -0.044 +- 0.015 and -0.061 +- 0.012 at n=200 |
| **Table 13** (exact motif / conjunction) | **1 x 8** (NFE 8), clip-down rule | **1 x 8** -- already it, at NFE 8 and NFE 32, under either rule and with either student | paired reduction **+0.1262** (n=100, GPU; published +0.1263) | unchanged | **no** -- 1 x 8 beats 2 x 4 / 4 x 2 / 8 x 1 by **3.9 / 9.9 / 13.7 SEM**. *But the scale-free rule at the same split gives +0.0130 +- 0.0030 (4.4 SEM) in dMFM's favour* |
| **Table 17** (posterior diversity) | reward-free, 1 posterior step, **no MC** | **not applicable** | 0.563 / 0.555 (dMFM / GLASS, t = 0.5, L = 50) | unchanged | **no** -- there is no value gradient to split |
| **Table 18** (dMFM vs GLASS-128 gradient MAE) | 4 x 8 (NFE 32) | **8 x 4 by the printed metric -- an artifact**; **1 x 32 by direction, on both rewards** | gc 0.0892 MAE / cos 0.212; motif 0.1636 / cos 0.139 (published 0.1049 / 0.1196) | gc 0.0726 / **cos 0.327**; motif **0.0479 / cos 0.318** at 1 x 32 | **the metric is the problem, not the split**: reference norm 0.006-0.090 vs estimates 0.18-10, corr(MAE, est norm) = 0.95 (gc) / 0.995 (motif). On gc 1 x 32 vs 4 x 8 is 0.5 SEM; on motif 4 x 8 is the worst split on both columns |
| **Fig 5 / Tables 14-15** (scale-free MC) | published: GLASS **8 x N**; dMFM rerun: **1 x N** | **not applicable** -- MC is the x-axis | -- | unchanged | **no**; but the rerun's `N` costs `N` calls and the published one's costs `8N`, which the paper should say |
| **Fig 4 right** (NFE benchmark) | **1 x MC**, MC in 1..16 | **1 x MC** -- already it | 0.0613 guide MAE at MC 16 (n=100x3) | unchanged | **no** -- at NFE 16 (n=32), 1 x 16 = 0.0424 against 2 x 8 = 0.0968 (3.4 SEM), 4 x 4 = 0.248 (4.5), 8 x 2 = 0.513 (8.2) |
| **Fig 7** (tilt ladder, `gf = 8`) | 4 x 8 (NFE 32) | **dMFM 1 x 32**, GLASS 2 x 16 | guide MAE 0.120, within-0.10 0.572 (n=1000); 0.158 / 0.53 in this run | **guide MAE 0.047, within-0.10 0.86** (n=64, H200) | **yes on the plotted quantity** -- paired **-0.111 +- 0.023** on the guide (4.9 SEM); 1.3 SEM on the oracle |
| **Fig 8** (one trajectory) | 4 x 8 (NFE 32) | dMFM 2 x 16 on the 4 traced ids; 1 x 32 is the applied point | mean \|C0-0.30\| 0.172 over 12 ids (0.126 over the recorded 4) | **0.076 at 1 x 32** over 12 ids | **marginal**: paired -0.095 +- 0.042 (2.3 SEM); no published number depends on it |

Fig 7's and Table 13's rows are **GPU** runs (H200, TF32) at n = 64 and at the published n = 100 --
those two jobs eventually got capacity. Table 18 (8 probes x 2 repeats), Fig 4 (32 outputs) and Fig 8
(12 ids) are CPU runs at reduced n because the account's GPUs were otherwise full all session; see
**Compute** below. Each per-item section gives the paired statistics and a
reproduction check of the *published* configuration in the same run, so the comparisons are
within-run and the reproduction check says how far the run sits from the published value.

## Where the split does and does not apply

Not every DNA result has a split to choose, and saying so is part of the answer.

### Table 17 -- no value gradient, so no split

`dmfm.experiments.eval_posterior_diversity` is **reward-free**: it draws `n_futures = 16`
posterior futures for each of 32 held-out conditioning states and compares their diversity with
matched GLASS futures. There is no reward, no value estimate, no Monte Carlo average of a reward
and therefore no `steps x MC` product to divide up. `--mc` does not exist in that module. Its
only posterior knob is `--dmfm_steps` (1 in the published run,
`results/dna/posterior_diversity/run_config.csv`) against `--glass_steps` (100), and the *point*
of the table is that one flow-map jump reproduces the diversity of a 100-step GLASS integration
-- a 100-fold NFE ratio that the table exists to display. Matching NFE here would mean either
giving dMFM 100 jumps or cutting GLASS to one, both of which delete the claim. **Nothing to
change, and nothing to re-measure.**

### Fig 5 / Tables 14-15 -- MC *is* the x-axis, so the split is not a free knob

These measure the accuracy of the finite-`N` value **gradient** itself against a high-`N`
reference, as a function of `N`. `N` is the MC count. Re-splitting a fixed NFE budget would
replace the figure's independent variable, turning "error against the number of posterior draws"
into "error against how a budget was divided", which is a different (and, per Table 18 below,
useful) figure but not this one. Two things do follow from the split question, and both are
recorded rather than acted on:

* **The level of the curve depends on the step count**, because a `k`-step posterior is a
  different estimator from a 1-step one. The published run uses **GLASS with `nfe_value = 8`** on
  *both* sides (`results/dna/scale_free_mc/*/run_metadata.json`: `nfe_value 8`,
  `reference_mc 2048`), so each plotted point at MC `N` in fact costs `8N` network calls. The
  dMFM rerun (`dmfm.scalefree.run_shard`, `--dmfm_steps 1` by default, student tag `diag1step`)
  uses one composed flow-map step, so its points cost `N`. That is already the chosen operating
  point, and it is also the cheaper one, so **the rerun needs no change** -- but **the two runs'
  x-axes are not the same cost axis**, and the paper should say `N` means draws, not calls. (That
  rerun is itself still incomplete: `results/dna/scale_free_mc/dmfm/` holds only a 4-state
  preliminary and `protocol_check.json`, not `summary.json`. Its split is nonetheless already
  settled by its default, so nothing there is waiting on this document.)
* **Which step count the finite-`N` estimator should use** is exactly what the Table 18 grid
  below measures, on the same quantity (gradient error against a GLASS reference). So Fig 5's
  level question is answered there instead of by re-plotting Fig 5.

### Table 13 -- the code can only spend its NFE one way

`dmfm.experiments.sample_reward_attainment` calls `one_step_dmfm`, i.e. a single direct map
`F(0, 1, eps | t, x_t)`, and exposes **no posterior-step argument at all**; `--mc 8` is its only
value-gradient knob. So the published Table 13 configuration is **1 x 8, NFE 8**, and the split
is at dMFM's own end of the trade already -- the published run could not have been at the wrong
split, because the module has only one. What Table 13 *does* have is (i) a four-times-smaller
NFE budget than the campaign's operating point, and (ii) the **stock clip-down** gradient rule
rather than the scale-free rule of convention 2. Measured here: the dMFM value-gradient norm on
this reward is about 1.4, far below `grad_clip = 10`, so **clip-down is a no-op** and the guided
step size is set by the raw gradient magnitude -- which means changing MC under that rule moves
the step size as well as the noise. Both rules are therefore run.

## Table 13 -- the published split is already the best one, but the published *rule* is not

**Primary run: GPU, n = 100, the published n.** `results/dna/operating_point/table13_grid/motif_nfe8/`
(Slurm **49465154_0**, `kempner_h200`, NVIDIA H200, TF32 on -- the GPU job did start, late in the
session). `dmfm.experiments.operating_point_reward_grid`, L = 50, exact motif reward. Everything else
is `scripts/dna/table13_reward_attainment.sbatch`: `seed 20260731`, `nfe_traj 64`, `t_end 0.999`,
window [0.5, 0.95], `guidance_frac 8`, `coeff_cap 10`, `grad_clip 10`, 1,024-sample target
calibration at the 90th percentile, 2,000-resample bootstrap. A CPU run at n = 32
(`table13_grid_cpu/motif_nfe8/`, Slurm 49467013) agrees throughout and is kept as corroboration.

**Reproduction of the published cell first.** The published configuration is 1 x 8 under clip-down:

| quantity | published (n = 100) | here (n = 100, H200) |
|---|---|---|
| unguided \|score - target\| | 0.15229 | **0.15229** |
| guided \|score - target\| | 0.0260 [0.0215, 0.0306] | **0.0261** [0.0218, 0.0307] |
| paired reduction (the Table 13 number) | +0.1263 [0.1092, 0.1442] | **+0.1262** [0.1088, 0.1447] |
| fraction of pairs improved | 0.98 | 0.95 |

That is an essentially exact reproduction, so everything below is measured on Table 13's own quantity.

**Control:** the unguided |score - target| is 0.15229 for every configuration in the run, i.e. the
unguided arm is shared and only the guidance term differs.

### The grid at Table 13's own budget, NFE 8 (n = 100, GPU)

| split | rule | guided \|err\| [CI] | paired reduction [CI] | frac improved | mean raw \|grad V\| | mean post-rule \|grad V\| |
|---|---|---|---|---|---|---|
| **1 x 8** (published) | clip-down | **0.0261** [0.0218, 0.0307] | **+0.1262** [0.1088, 0.1447] | 0.95 | 1.40 | 1.27 |
| 2 x 4 | clip-down | 0.0372 [0.0320, 0.0425] | +0.1150 [0.0983, 0.1328] | 0.92 | 2.17 | 1.41 |
| 4 x 2 | clip-down | 0.0953 [0.0806, 0.1095] | +0.0570 [0.0453, 0.0700] | 0.88 | 6.34 | 0.87 |
| 8 x 1 | clip-down | 0.1496 [0.1313, 0.1675] | +0.0027 [0.0009, 0.0051] | **0.13** | 9.00 | **0.053** |
| **1 x 8** | scale-free | **0.0130** [0.0095, 0.0169] | **+0.1393** [0.1204, 0.1597] | 0.98 | 0.81 | 9.92 |
| 2 x 4 | scale-free | 0.0191 [0.0147, 0.0238] | +0.1332 [0.1142, 0.1532] | 0.95 | 1.28 | 9.95 |
| 4 x 2 | scale-free | 0.0253 [0.0204, 0.0303] | +0.1270 [0.1085, 0.1475] | 0.92 | 1.50 | 9.86 |
| 8 x 1 | scale-free | 0.0298 [0.0240, 0.0362] | +0.1224 [0.1018, 0.1434] | 0.88 | 4.36 | 8.01 |

Paired on shared `sample_id`, on guided |score - target| (negative favours the first):

| comparison | mean diff +- SEM | CI | SEM units |
|---|---|---|---|
| 1 x 8 vs 2 x 4, clip-down | **-0.0112 +- 0.0029** | [-0.0170, -0.0057] | 3.9 |
| 1 x 8 vs 4 x 2, clip-down | **-0.0692 +- 0.0070** | [-0.0831, -0.0558] | 9.9 |
| 1 x 8 vs 8 x 1, clip-down | **-0.1235 +- 0.0090** | [-0.1408, -0.1054] | 13.7 |
| 1 x 8 vs 2 x 4, scale-free | **-0.0060 +- 0.0030** | [-0.0124, -0.0004] | 2.0 |
| 1 x 8 vs 4 x 2, scale-free | **-0.0122 +- 0.0032** | [-0.0185, -0.0062] | 3.8 |
| 1 x 8 vs 8 x 1, scale-free | **-0.0168 +- 0.0037** | [-0.0243, -0.0096] | 4.5 |
| **1 x 8 clip-down vs 1 x 8 scale-free** | **+0.0130 +- 0.0030** | [+0.0073, +0.0187] | **4.4** |

(The last row is the clip-down arm *minus* the scale-free arm, so positive means clip-down is worse.)

The CPU n = 32 run gives the same ordering and the same signs throughout (1 x 8 clip-down 0.0250
[0.0178, 0.0327], paired +0.1086; 1 x 8 scale-free 0.0094, paired +0.1243), with wider intervals.

### The same grid at NFE 32, and with the diagonal student at every step count

`motif_nfe32/` (the campaign's budget) and `motif_nfe8_diagstudent/` (the diagonal student forced at
every step count, so the split is separated from the student swap). Guided |score - target|:

| split | rule | NFE 32, convention-1 student | NFE 8, diagonal student at every step count |
|---|---|---|---|
| 1 x MC | clip-down | 0.0207 | 0.0250 |
| 2 x MC/2 | clip-down | 0.0216 | 0.0377 |
| 4 x MC/4 | clip-down | 0.0588 | 0.0701 |
| 8 x MC/8 | clip-down | 0.1295 (0.28 of pairs improved) | 0.1336 (0.25 of pairs improved) |
| **1 x MC** | scale-free | **0.0064** (1.00 of pairs improved) | **0.0094** |
| 2 x MC/2 | scale-free | 0.0110 | 0.0156 |
| 4 x MC/4 | scale-free | 0.0278 | 0.0183 |
| 8 x MC/8 | scale-free | 0.0162 | 0.0206 |

Both extra grids reproduce the NFE-8 ordering: **one composed step is best at every budget and under
either rule and with either student**, and under clip-down the 8-step posterior is nearly inert
(0.25-0.28 of pairs improved). So the step-count effect is the split's, not the artifact's.

**Reading, three findings.**

1. **Table 13's split is already the optimum at its own budget**, under either rule, and the shipped
   code could not have produced any other. So there is **no operating-point correction to make to
   Table 13**: unlike Table 1, this result was never at GLASS's estimator split. Its number does not
   move.
2. **Its gradient *rule* does move it, significantly.** At the same 1 x 8 split, replacing the stock
   clip-down rule with the scale-free rule of convention 2 takes the guided error from 0.0250 to
   0.0094, a paired -0.0156 +- 0.0043 (3.7 SEM), and the printed paired reduction from +0.109 to
   +0.124. This is a **convention-2 gap, not a split gap**, and it is the same gap
   `docs/dna_findings_audit.md` identified in general: the dMFM value-gradient norm on this reward is
   0.7-1.2, far **below** `grad_clip = 10`, so clip-down is a no-op and the guided step size is set
   by whatever magnitude the gradient happens to have.
3. **Under clip-down, a deeply composed posterior does not steer at all.** At 8 x 1 the raw gradient
   norm is 0.079 and the paired reduction is +0.0030 with only **2 of 32 pairs improved** -- the
   guidance term is numerically absent, because clip-down can only scale *down*. At 8 x 4 (NFE 32)
   the same thing happens for a subtler reason: the *mean* raw norm is 15.4 but only 0.9 % of guided
   steps exceed `grad_clip = 10`, so the distribution is a near-zero body with rare huge spikes, and
   the post-rule step averages **0.154** -- 9 of 32 pairs improved, guided error 0.1295 against an
   unguided 0.1336, i.e. no steering. The scale-free rule restores both (+0.1012 and +0.1175, 29 and
   31 of 32 pairs). That is the same mechanism the findings audit found for `glass_euler32`, now seen
   on the dMFM side, on an exact reward, and resolved by step count: **the heavy tail is a property
   of composing many jumps.**

## Fig 7 -- the tilt ladder at `guidance_frac 8`, every split

**Primary run: GPU, n = 64.** `results/dna/operating_point/fig7_grid_nfe32_n64/` (Slurm **49465152**,
`kempner_h200`, NVIDIA H200 `holygpu8a06301`, TF32 on -- the GPU job did start, late in the session).
`dmfm.experiments.operating_point_c0` at `--targets 0.30`, all eight NFE-32 splits with the
scale-free rule on both arms and `guidance_frac 8` -- Fig 7's own moderate panel and the published
Table 1 strength. Fig 7's published dMFM arm is `dmfm4_fm4@norm@mc=8`, which is one of the eight. A
CPU run at n = 32 (`fig7_grid_cpu_nfe32_n32/`, Slurm 49466900) agrees throughout.

**Control:** the unguided oracle and guide scores are bit-identical across all eight splits at every
id (max spread 0.0, unguided sequences character-identical), so the whole difference is the guidance
term. Unguided oracle |C0 - 0.30| = 0.532.

Guide MAE is the quantity Fig 7's histograms show (how far the tilted distribution sits from the
target); the oracle column is the held-out check.

| arm | split | guide MAE [CI] | guide within 0.10 | oracle MAE [CI] | mean raw \|grad V\| |
|---|---|---|---|---|---|
| GLASS | 2 x 16 (GLASS's chosen point) | **0.0245** [0.0161, 0.0363] | 0.95 | 0.1353 [0.1133, 0.1581] | 1.10 |
| GLASS | 1 x 32 | 0.0367 [0.0262, 0.0483] | 0.94 | **0.1192** [0.0918, 0.1479] | 1.67 |
| **dMFM** | **1 x 32 (the chosen operating point)** | **0.0470** [0.0354, 0.0592] | 0.86 | 0.1365 [0.1150, 0.1590] | 2.74 |
| GLASS | 4 x 8 | 0.0539 [0.0408, 0.0686] | 0.83 | 0.1408 [0.1148, 0.1680] | 1.61 |
| dMFM | 2 x 16 | 0.0645 [0.0512, 0.0799] | 0.77 | 0.1605 [0.1345, 0.1898] | 4.93 |
| GLASS | 8 x 4 | 0.0969 [0.0746, 0.1242] | 0.58 | 0.1832 [0.1513, 0.2170] | 6.95 |
| **dMFM** | **4 x 8 (the current Fig 7 arm)** | **0.1577** [0.1175, 0.2002] | 0.53 | 0.1703 [0.1279, 0.2183] | 9.75 |
| dMFM | 8 x 4 | 0.1601 [0.1239, 0.2008] | 0.47 | 0.1836 [0.1465, 0.2225] | 10.39 |

For scale, the recorded Fig 7 ladder at `gf = 8` and n = 1000
(`results/dna/dmfm_demos/fig7/ladder_summary.csv`) gives dMFM 4 x 8 guide MAE **0.1204**, within-0.10
**0.572**, oracle **0.1768**, and GLASS 4 x 8 guide MAE **0.0477**, within-0.10 **0.883**, oracle
**0.1261**. This n = 64 run puts the same two arms at 0.1577 / 0.53 / 0.1703 and 0.0539 / 0.83 /
0.1408, so it is measuring the ladder's quantity.

Paired on shared ids (negative favours the first):

| comparison | guide MAE | oracle MAE |
|---|---|---|
| dMFM 1 x 32 vs dMFM 4 x 8 | **-0.1107 +- 0.0226** (4.9 SEM) | -0.0337 +- 0.0257 (1.3 SEM) |
| dMFM 2 x 16 vs dMFM 4 x 8 | **-0.0932 +- 0.0227** (4.1 SEM) | -0.0098 +- 0.0261 (0.4 SEM) |
| dMFM 8 x 4 vs dMFM 4 x 8 | +0.0024 +- 0.0316 (0.1 SEM) | +0.0134 +- 0.0311 |
| GLASS 2 x 16 vs GLASS 4 x 8 | **-0.0294 +- 0.0082** (3.6 SEM) | -0.0054 +- 0.0160 |
| dMFM 1 x 32 vs GLASS 2 x 16 | **+0.0225 +- 0.0087** (2.6 SEM) | +0.0012 +- 0.0187 (0.1 SEM) |

**Reading.**

* **Fig 7's dMFM arm moves, and the move is large and significant on the figure's own quantity.**
  Guide MAE **0.158 -> 0.047** and within-0.10 **0.53 -> 0.86**, paired **-0.111 +- 0.023 (4.9 SEM)**.
  Scaled to the published n = 1000 ladder point the implied change is guide MAE 0.120 -> about 0.04
  and within-0.10 0.57 -> about 0.86: the dMFM histogram goes from visibly broader than GLASS's to
  close to it.
* **On the oracle column the same move is only 1.3 SEM.** The guide is what the sampler optimises and
  what Fig 7 plots; the oracle is a held-out proxy with its own 0.16 MAE floor and it compresses the
  difference, exactly as `docs/dmfm_deficit_rescue.md` section 1 found. Significant on the plotted
  quantity, not resolved on the oracle at this n.
* **GLASS also prefers 2 x 16 here** (-0.029 +- 0.008 against its own 4 x 8, 3.6 SEM), so giving GLASS
  its own best split is not a formality.
* **GLASS is still ahead on the guide, but the two arms are a dead tie on the oracle.** dMFM 1 x 32
  against GLASS 2 x 16 is +0.023 +- 0.009 on the guide (2.6 SEM, dMFM behind) and
  **+0.001 +- 0.019 on the oracle (0.1 SEM)**. **No dMFM win is claimed** -- the correction removes an
  artificial 3.2x gap on the guide and leaves a real 1.9x one; on the held-out oracle it leaves
  nothing.
* Because the move is real and large, **the `gf = 8` point of the ladder is worth regenerating at the
  published n = 1000** at each arm's own operating point.
  `scripts/dna/operating_point/fig7_at_operating_point.sbatch` does exactly that (dMFM 1 x 32 and
  GLASS 2 x 16); it was queued from elsewhere in this session as Slurm **49476650**
  (`kempner_h200`, array 0-1), still pending at the time of writing.

## Table 18 -- the gradient-accuracy grid, and a problem with its reference

`results/dna/operating_point/table18_grid_cpu/gc/` (Slurm 49466901),
`dmfm.experiments.operating_point_gradient_grid`, L = 50, `gc` reward, 8 probes x 2 repeats, CPU.
Probe states, the GLASS-128 RK4-50 reference noise pool and the nested candidate MC pools are drawn
from CPU generators seeded exactly as the published Table 18 run seeds them, and a smaller draw is a
row-major prefix of a larger one, so this run's probes **are** the published run's probes 0-7 and its
candidate pools **are** their pools for repeats 0-1 and MC <= 8.

**Reproduction check, cell by cell.** On the 64 `(probe, repeat, mc)` cells this run shares with the
published Table 18 (`results/dna/dmfm_vs_glass/gc/L50/gradient_mae_per_condition.csv`), mean MAE
**0.1504 here against 0.1404 published** (7 % on the mean), median per-cell relative difference 8 %,
max 111 % on one cell (`validation_vs_published_table18.csv`). These per-cell numbers are extremely
heavy-tailed and the two runs differ in arithmetic (CPU fp32 against GPU TF32, with a
nondeterministic backward), so that is a reproduction, not an identity -- close enough that this grid
is measuring Table 18's quantity, not close enough to quote a cell of it as the published value.

### The finding that matters here is about Table 18's reference, not about the split

Across the 8 probes the **reference** gradient -- GLASS with 128 terminal-noise samples, RK4, 50
steps, to `end_time 0.999` -- has L2 norm between **0.0057 and 0.090, median 0.018**, while the
finite-MC estimates scored against it have norms **0.18 to 7.2**, i.e. 10x to 400x larger. Table 18's
metric is the coordinate-wise MAE between the two, so with a reference that is numerically ~0 **the
metric is very nearly a measure of the estimate's own gradient magnitude**: over every cell in the
grid, `corr(MAE, estimate |g|) = 0.947`.

The consequence is that Table 18's *own* optimum over the split grid is an artifact. Ranked by the
printed MAE at NFE 32 the best dMFM split is **8 x 4**, whose gradient norm is 0.18 -- and Table 13's
grid shows independently that the 8-step composed posterior's gradient is a near-zero body with rare
spikes (post-clip step 0.154 at NFE 32, 0.047 at NFE 8) and that under the published clip-down rule
it steers essentially not at all (9 of 32 and 2 of 32 pairs improved). So Table 18's metric rewards
the configuration that cannot guide.

The direction-sensitive column tells the other story, and it agrees with the chosen operating point.

### The grid, in full

All nine configurations at NFE 32, 8 probes x 2 repeats. `mae` is Table 18's printed metric; `est
|g|` is the estimate's own L2 norm, for comparison with the reference's 0.006-0.090; `cos` is the
direction agreement with the same reference.

| arm | split | Table 18 metric (MAE) | est \|g\| | cos with reference |
|---|---|---|---|---|
| dMFM (4-step student) | 8 x 4 | **0.0036** | **0.18** | 0.235 |
| dMFM (4-step student) | **1 x 32 (the chosen operating point)** | 0.0726 | 1.95 | **0.327** |
| dMFM (diagonal student) | 1 x 32 | 0.0859 | 2.44 | 0.284 |
| dMFM (4-step student) | 2 x 16 | 0.0861 | 2.97 | 0.288 |
| dMFM (4-step student) | **4 x 8 (current Table 18 row)** | 0.0892 | 4.99 | 0.212 |
| GLASS | 4 x 8 | 0.1035 | 2.24 | 0.440 |
| GLASS | 2 x 16 | 0.1269 | 2.29 | 0.541 |
| GLASS | 8 x 4 | 0.1415 | 7.22 | 0.136 |
| GLASS | 1 x 32 | 0.1515 | 2.67 | **0.545** |

Paired on shared `(probe, repeat)` cells against the current 4 x 8 row:

| comparison | mean diff +- SEM | SEM units |
|---|---|---|
| 1 x 32 vs 4 x 8 | -0.0166 +- 0.0362 | 0.5 |
| 2 x 16 vs 4 x 8 | -0.0031 +- 0.0309 | 0.1 |
| diagonal 1 x 32 vs 4 x 8 | -0.0034 +- 0.0366 | 0.1 |
| 8 x 4 vs 4 x 8 | **-0.0856 +- 0.0347** | 2.5 |

**Reading.**

* **By Table 18's printed metric the split does not matter** -- 1 x 32, 2 x 16 and 4 x 8 are within
  0.5 SEM of each other -- **except for the one move that is significant, to 8 x 4, and that one
  should not be taken.** It wins by producing a gradient of norm 0.18 against a reference of norm
  0.018, and the same posterior is the one that does not steer.
* **On the direction-sensitive column the chosen operating point is the best dMFM split**: cos 0.327
  at 1 x 32 against 0.288 at 2 x 16, 0.212 at 4 x 8 and 0.235 at 8 x 4. Since convention 2 normalises
  the gradient's magnitude away before it reaches the sampler, direction is the part that matters --
  so 1 x 32 is the right choice here too, and it happens to move Table 18's printed number slightly
  the wrong way (0.089 -> 0.073 is an improvement, but not a significant one) while making the
  estimator better on the axis that acts.
* **GLASS's gradients agree with the reference in direction about 1.6-2.6x better than dMFM's**
  (cos 0.44-0.55 against 0.21-0.33). That is unsurprising -- the reference *is* GLASS on the base
  model -- and it is the honest reading of Table 18: it measures how close the distilled posterior is
  to the teacher's, and dMFM is not closer.
* **Table 18 as printed needs a metric decision from the authors, not a split decision.** Whatever
  split is used, the MAE against this reference is not a measure of gradient accuracy.

### The same grid on the `motif` reward -- not a `gc` artifact, and there 4 x 8 is the worst split

`table18_grid_cpu/motif/` (Slurm 49471660, complete: 8 probes x 2 repeats) reproduces every feature
of the `gc` grid and sharpens one. Reference norm **0.0055-0.0352, median 0.0097** against estimate
norms 0.26-10.0; **`corr(MAE, estimate |g|) = 0.995`**. At NFE 32:

| arm | split | Table 18 metric (MAE) | est \|g\| | cos with reference |
|---|---|---|---|---|
| dMFM (4-step student) | 8 x 4 | **0.0048** | **0.26** | 0.218 |
| dMFM (4-step student) | **1 x 32 (the chosen operating point)** | 0.0479 | 1.68 | **0.318** |
| dMFM (4-step student) | 2 x 16 | 0.0499 | 2.35 | 0.231 |
| dMFM (diagonal student) | 1 x 32 | 0.0629 | 2.77 | 0.296 |
| GLASS | 8 x 4 | 0.0961 | 5.08 | 0.155 |
| GLASS | 4 x 8 | 0.0998 | 3.12 | 0.440 |
| GLASS | 2 x 16 | 0.1583 | 3.89 | 0.628 |
| **dMFM (4-step student)** | **4 x 8 (current Table 18 row)** | **0.1636** | 9.98 | **0.139** |
| GLASS | 1 x 32 | 0.1990 | 4.82 | **0.639** |

On this reward the published 4 x 8 row is the **worst** dMFM split on *both* columns -- MAE 0.164
against 0.048 at 1 x 32, and cosine 0.139 against 0.318 -- so here the operating-point correction
improves Table 18's printed number as well as its direction, and by 3.4x. Reproduction on the 64
shared `(probe, repeat, mc)` cells: mean MAE 0.597 here against 0.520 published, median 7 % per cell.
(This reward's per-cell MAEs are far more heavy-tailed than `gc`'s, which is why that subset mean sits
well above the table's 32-probe, 8-repeat value of 0.1825 -- the comparison that matters is mine
against published *on the same cells*.)

## Fig 4 (right) -- the plotted curve is already at dMFM's own operating point

`results/dna/operating_point/fig4_grid_cpu/` (convention-1 student per step count) and
`fig4_grid_cpu_diag/` (the diagonal student at every step count, so the split is separated from the
student swap). `dmfm.benchmarks.run_steering.run_job` through
`scripts/dna/operating_point/fig4_grid.py`, so the shard format, the NFE accounting and the metric
are the figure's own; Fig 4's own **clip-down** rule is kept, not the scale-free one, because the
question is about Fig 4 as it stands. n_outputs = 8, one repeat, CPU. Metric: guide MAE against
target C0 0.30, exactly the plotted quantity. `nfe_gen_per_output` is identical inside each group
(824 at NFE 8, 1552 at NFE 16), i.e. the budget really is matched.

`plan._dmfm(mc)` sets `nfe_value = 1`, so the published curve is **1 x MC** for MC in 1..16 and its
top point costs 1552 NFE per output. Splitting that budget the other ways (guide MAE; n = 32 in the
main column, n = 8 for the two control columns):

| NFE per value gradient | split | n = 32, convention-1 student | n = 8, convention-1 | n = 8, diagonal student |
|---|---|---|---|---|
| 8 | **1 x 8 (on the published curve)** | **0.0496** | 0.048 | 0.048 (same artifact) |
| 8 | 2 x 4 | 0.1366 | 0.114 | 0.095 |
| 8 | 4 x 2 | 0.2888 | 0.164 | 0.252 |
| 8 | 8 x 1 | 0.5746 | 0.496 | 0.622 |
| 16 | **1 x 16 (the curve's top point)** | **0.0424** | 0.074 | 0.074 (same artifact) |
| 16 | 2 x 8 | 0.0968 | 0.078 | 0.059 |
| 16 | 4 x 4 | 0.2484 | 0.256 | 0.194 |
| 16 | 8 x 2 | 0.5132 | 0.447 | 0.460 |

Paired on shared outputs at n = 32 (negative favours 1 x MC):

| comparison | mean diff +- SEM | SEM units |
|---|---|---|
| NFE 8: 1 x 8 vs 2 x 4 | **-0.0870 +- 0.0197** | 4.4 |
| NFE 8: 1 x 8 vs 4 x 2 | **-0.2392 +- 0.0486** | 4.9 |
| NFE 8: 1 x 8 vs 8 x 1 | **-0.5249 +- 0.0644** | 8.2 |
| NFE 16: 1 x 16 vs 2 x 8 | **-0.0544 +- 0.0160** | 3.4 |
| NFE 16: 1 x 16 vs 4 x 4 | **-0.2060 +- 0.0453** | 4.5 |
| NFE 16: 1 x 16 vs 8 x 2 | **-0.4708 +- 0.0573** | 8.2 |

Published curve for scale, at n = 100 x 3 repeats: 0.0736 at MC 8 and 0.0613 at MC 16
(`results/dna/fig4/panelB_curves.csv`), so the n = 8 reproduction of the 1 x MC points is in the
right place.

**Reading.** One posterior step is the best or joint-best split at both budgets, and 4 or 8 steps
is catastrophic (3-9x worse) under either student. So **Fig 4's dMFM curve does not move**: it was
already parameterised the way the deficit-rescue analysis says it should be, and this is the one
DNA result whose dMFM estimator was set up correctly from the start. Fig 4 (left) is the same case
-- `value_error.py --dmfm-nfe 1`, x-axis N.

**The n = 8 ambiguity is settled at n = 32.** At 8 outputs, 1 x 16 (0.074) against 2 x 8 with the
diagonal student (0.059) was not resolvable; at 32 outputs 1 x 16 (0.0424) beats 2 x 8 (0.0968) by
3.4 SEM. One posterior step wins at both budgets, and by a wide margin over 4 and 8 steps, so the
plotted curve stays exactly where it is.

**One trap recorded.** `run_steering.run_job` writes `meta["checkpoints"]["dmfm"] =
paths.dmfm_ckpt(length)` unconditionally, whatever student was loaded, so a shard's JSON names the
diagonal student even for a 4-step-ESD-student run. `student_mode.json` in each grid directory is
authoritative instead. Anyone reading Fig 4 shard provenance should know this.

## Fig 8 -- one trajectory at every split (complete)

`results/dna/operating_point/fig8_grid_nfe32/`, `dmfm.regen.guidance_trace` on CPU (fp32,
deterministic), `logs/op_fig8_grid.log`. Everything as
`scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch` at `guidance_frac 8` except the split: target C0
0.30, `sample_ids 0 1 2 3`, `nfe_traj 64`, window [0.5, 0.95], scale-free rule on both arms.

**Control:** the unguided |C0 - 0.30| is 0.5010 for every split on both arms, and the unguided
endpoints are identical to the recorded Fig 8 dMFM run
(`results/dna/dmfm_demos/fig8/dmfm/trace.csv`) to every printed digit. So all splits share one
trajectory and differ only in the guidance term.

| arm | split | mean \|C0 - 0.30\| over the 4 ids | id 0 (the id Fig 8 displays) | mean raw \|grad V\| |
|---|---|---|---|---|
| dMFM | 1 x 32 (the chosen operating point) | **0.0958** | 0.196 | 2.08 |
| dMFM | 2 x 16 | **0.0453** | 0.101 | 4.25 |
| dMFM | 4 x 8 (**current**) | 0.1260 | **0.037** | 6.10 |
| dMFM | 8 x 4 | 0.1618 | 0.097 | 32.86 |
| GLASS | 1 x 32 | 0.0598 | 0.018 | 1.58 |
| GLASS | 2 x 16 (GLASS's chosen point) | **0.0102** | 0.007 | 1.06 |
| GLASS | 4 x 8 | 0.0135 | 0.015 | 1.77 |
| GLASS | 8 x 4 | 0.1354 | 0.191 | 5.50 |

**n = 4. This cannot choose anything and is not used to.** What it does show, and what generalises:
the dMFM raw value-gradient norm grows by 16x from 1 to 8 composed steps (2.08 -> 32.86) while
GLASS's grows only from 1.58 to 5.50 (and is 1.0-1.8 at 2 and 4 steps). That is the heavy-tail mechanism `docs/dmfm_glass_parity.md`
identified, now resolved by step count: **the tail is a property of composing many jumps, not of
the dMFM posterior as such.** It is also why the scale-free rule matters more for dMFM at 4 and 8
steps than at 1.

**Extended to 12 ids for the two dMFM splits and 8 for the GLASS control**
(`fig8_grid_nfe32_ids4to7/` and `fig8_grid_nfe32_ids8to11/`, added on the login node with the same
settings). |C0 - 0.30| per id and the mean:

| id | dMFM 1 x 32 | dMFM 4 x 8 (current) | GLASS 2 x 16 |
|---|---|---|---|
| 0 (the id Fig 8 displays) | 0.196 | **0.037** | 0.007 |
| 1 | 0.034 | 0.218 | 0.014 |
| 2 | 0.092 | 0.053 | 0.004 |
| 3 | 0.062 | 0.196 | 0.017 |
| 4 | 0.002 | 0.217 | 0.012 |
| 5 | 0.059 | 0.037 | 0.026 |
| 6 | 0.145 | 0.345 | 0.001 |
| 7 | 0.060 | 0.379 | 0.047 |
| 8 | 0.095 | 0.160 | -- |
| 9 | 0.004 | 0.230 | -- |
| 10 | 0.085 | 0.006 | -- |
| 11 | 0.085 | 0.182 | -- |
| **mean** | **0.076** (12 ids) | **0.172** (12 ids) | **0.016** (8 ids) |

Paired over the 12 shared ids, dMFM 1 x 32 against dMFM 4 x 8: **-0.0952 +- 0.0417 (2.3 SEM)** -- the
same size of move as Fig 7's and now just past 2 SEM. Ids 0 and 10 are the two where 4 x 8 looks
better, which is exactly why a figure should not be read off one trajectory.

At the chosen operating point the mean over the four recorded ids improves from 0.126 to 0.096 (and
over 12 ids from 0.172 to 0.076), but the *displayed* id 0 gets worse (0.037 -> 0.196). Fig 8 illustrates a mechanism and prints no number, so the honest
statement is that the figure survives at either split and that whichever is printed, the displayed
id should be fixed in advance (it was: id 0, `docs/dmfm_demo_reruns.md`) rather than chosen after
seeing the trace.

## Consequences for the paper, for the authors to decide

1. **Only Table 1 / Fig 3 was at the wrong split.** Of the seven other DNA items, three had no split
   to get wrong (Table 17 is reward-free; Fig 5 / Tables 14-15 and Fig 4 left have MC as their
   x-axis), two were already at one composed posterior step (Fig 4 right, and the dMFM rerun of
   Fig 5), one could not have been anywhere else (Table 13, whose producer hard-wires a single map),
   and two were at 4 x 8 because they were built to match Table 1's operating point (Fig 7, Fig 8).
   So the correction propagates to **Fig 7 and Fig 8 only**, and the paper's story gets *more*
   coherent: every dMFM DNA result then uses one composed flow-map step, which is also the artifact
   (`paths.dmfm_ckpt_for_steps(L, 1)`, the diagonal student) that Tables 13, 17 and Fig 4 already use.
2. **State the split, not just the NFE**, wherever a dMFM value gradient appears -- already agreed as
   convention 3. Add that Fig 5 / Tables 14-15's `N` is a number of *draws*, not of network calls:
   the published GLASS estimator there spends `8N` calls per point, the dMFM rerun spends `N`.
3. **Convention 2 (the scale-free rule) is not yet applied to Table 13 or Fig 4**, and on Table 13 it
   is worth a significant improvement (paired -0.0156 +- 0.0043, printed reduction +0.109 -> +0.124).
   Those two results still use the stock clip-down rule. Whether to restate them under convention 2
   is an authors' decision; this document only measures it.
4. **Table 18's metric, not its split, is what needs a decision (new).** Its GLASS-128 RK4-50
   reference to `end_time 0.999` has gradient norm 0.006-0.090 while the estimates it scores have
   0.18-7.2, and `corr(MAE, estimate norm) = 0.95` over the grid, so the printed MAE is close to a
   measure of the estimate's own magnitude. By that metric the "best" split is the one whose gradient
   has collapsed and which Table 13's grid shows does not steer. A magnitude-invariant metric
   (cosine, or the relative L1 that Fig 5 / Tables 14-15 already use) would say something about
   gradient accuracy; this one does not. Reported, not fixed.
5. **Do not claim a dMFM win anywhere on the strength of the split.** GLASS improves on its own grid
   too, and at every budget measured here GLASS's best split is at least as good as dMFM's best --
   at Fig 7's target, dMFM 1 x 32 is still +0.018 +- 0.010 behind GLASS 2 x 16 on the guide.

## Compute, and why most of this ran on CPU

Measured 2026-09-30 15:30-15:40 EDT. The account's shared 16-GPU QOS cap was **fully consumed by
the same user's other projects** (`squeue -u ... -t R` summed to exactly `gres/gpu:16`), so a
1-GPU job of mine could not start whatever its priority; on top of that there were 54-98 pending
jobs ahead of mine on `kempner_h200` / `kempner_h100`, and 1,537 ahead on `kempner_requeue`. The
four GPU jobs (`dna_op_c0` 49465152, `dna_op_t18` 49464001, `dna_op_t13` 49465154,
`dna_op_fig4` 49464821) were submitted and left queued. **Two of them eventually started**, about 90
minutes later, once the other projects released GPUs: 49465152 (Fig 7's grid at n = 64, 2 min 49 s on
an H200) and 49465154_0 (Table 13's NFE-8 grid at the published n = 100, 3 min 19 s), and those are
now the primary runs for those two items. 49465154_1 (Table 13 at NFE 32), 49464001 (Table 18) and
49464821 (Fig 4) were still pending.

CPU was the route that actually ran. `serial_requeue` had 8,260 jobs ahead of ours; `kozinsky` and
`kozinsky_requeue` (the user's own lab association) had an **empty** queue, and every CPU grid here
ran there. `kozinsky_requeue` preempted two jobs within six minutes
(`CANCELLED ... DUE TO PREEMPTION`), so the non-resumable harnesses were moved to the
non-preemptible `kozinsky`. The login node is not an alternative: `nproc` reports **1** core there
with a load average above 20.

Consequence, stated plainly: **the CPU grids run at smaller n than the GPU jobs would have** (Fig 7
at n = 32 rather than 64, Table 13 at n = 32 rather than the published 100, Table 18 at 8 probes x 2
repeats rather than 32 x 8, Fig 4 at 8 outputs rather than 32). Their CIs are correspondingly wide,
and CPU fp32 is not comparable with GPU TF32 in absolute terms. Every conclusion below is therefore
drawn from *within-run* paired comparisons on shared ids, and the effect sizes that matter here
(2-6x differences in gradient MAE, 0.05-0.4 in guide MAE) are far larger than those CIs. Where a
grid shows only a marginal move, that is said rather than resolved.

## Method notes

* `src/dmfm/experiments/operating_point_c0.py` -- the C0 grid. A thin wrapper over
  `rescue_dmfm_glass_c0` (so the harness, the shared noise and MC pools, the scale rules and the
  output format are that module's, unchanged) that adds the `glass_euler1` configuration the
  matched-NFE grid needs at the 1 x MC corner and which no earlier sweep defined.
* `src/dmfm/experiments/operating_point_gradient_grid.py` -- the Table 18 grid. Reuses
  `ablate_dmfm_one_step_gradient_mc`'s probe construction, reward calibration, dMFM estimator and
  GLASS reference estimator **and its seeds**, so at the same `--seed`, `--n_probes` and
  `--t_eval` the conditioning states and reference gradients are the published Table 18 run's and
  the `steps = 4` rows are directly comparable with it. The reference does not depend on the dMFM
  sampler, so it is computed once per probe and shared by the whole grid.
* `src/dmfm/experiments/operating_point_reward_grid.py` -- the Table 13 grid. Adds the missing
  posterior-step knob (`flow_map_dmfm`; `steps = 1, end_time = 1.0` *is* `one_step_dmfm`, so the
  1 x 8 row reproduces the published configuration) and the scale-free rule, reusing
  `sample_reward_attainment`'s reward, calibration, trajectory, seeds and statistics.
* `scripts/dna/operating_point/fig4_grid.py` -- Fig 4 (right). Constructs `dmfm` jobs with
  `nfe_value != 1` and runs them through `dmfm.benchmarks.run_steering.run_job`, so the shard
  format, the NFE accounting and the metrics are the figure's own. `plan.py` is not touched.
* `scripts/dna/operating_point/fig8_grid_cpu.sh` -- Fig 8. `dmfm.regen.guidance_trace` at each
  split, on CPU, which is deterministic and is the route the recorded Fig 8 run used.
* `scripts/dna/operating_point/collect.py` -- per-split summaries and paired differences on the
  ids (or `(probe, repeat)` cells) two splits actually share.

Nothing outside `docs/dna_operating_point.md`, `results/dna/operating_point/`,
`scripts/dna/operating_point/` and the three new `src/dmfm/experiments/operating_point_*.py`
modules was created or modified. In particular `parity_dmfm_glass_c0.py`,
`rescue_dmfm_glass_c0.py`, `sample_reward_attainment.py`,
`ablate_dmfm_one_step_gradient_mc.py` and `dmfm/benchmarks/plan.py` are imported, not edited, so
no stored result can change.

## Files and runs

New, and nothing else touched:

* `docs/dna_operating_point.md` (this file)
* `src/dmfm/experiments/operating_point_c0.py` -- C0 grids; wraps `rescue_dmfm_glass_c0`, adding
  `glass_euler1` (and `diag_fm2` / `diag_fm8` as controls) to its config table **in process only**
* `src/dmfm/experiments/operating_point_gradient_grid.py` -- Table 18's grid with one shared
  reference per probe
* `src/dmfm/experiments/operating_point_reward_grid.py` -- Table 13's grid, adding the posterior-step
  knob its producer lacks and the scale-free rule
* `scripts/dna/operating_point/{c0_grid,table18_grid,table13_grid,fig4_grid,cpu_hedge,fig7_at_operating_point}.sbatch`,
  `scripts/dna/operating_point/{fig4_grid.py,fig8_grid_cpu.sh,collect.py}`
* `results/dna/operating_point/` with `PROVENANCE.md` (job IDs, commands, the cross-device warning
  and the per-run controls) and `table1_anchor/README.md`

Imported and **not** edited, so no stored result can change: `parity_dmfm_glass_c0.py`,
`rescue_dmfm_glass_c0.py`, `sample_reward_attainment.py`, `ablate_dmfm_one_step_gradient_mc.py`,
`ablate_glass_gradient_mc.py`, `dmfm/scalefree/`, `dmfm/benchmarks/{plan,run_steering,core}.py`,
`dmfm/regen/guidance_trace.py`.

## What remains open

1. **Two of the four GPU grids are still queued.** 49465152 (Fig 7, n = 64) and 49465154_0
   (Table 13, n = 100) ran and are the primary runs above. Outstanding: **49464001** (Table 18 at
   32 probes x 8 repeats, all three rewards) and **49464821** (Fig 4 at 32 outputs x 3 repeats), plus
   **49465154_1** (Table 13 at NFE 32, n = 100). They write to separate directories
   (`table18_grid/`, `fig4_grid/`, `table13_grid/motif_nfe32/`), so nothing is overwritten. No
   conclusion here is sample-size limited -- every effect called significant is 2.0-13.7 SEM -- but
   the Table 18 and Fig 4 CIs would tighten.
2. **Fig 7 at the published n = 1000** at each arm's operating point
   (`scripts/dna/operating_point/fig7_at_operating_point.sbatch`). This is the one item where a
   printed number really moves, so the paper's version should come from a full-n run on the production
   path. I left it unsubmitted; it was submitted from elsewhere in the same session as Slurm
   **49476650** (`kempner_h200`, array 0-1) and was still pending. It writes to
   `results/dna/operating_point/fig7_operating_point_n1000/{dmfm_s1mc32,glass_s2mc16}/`.
3. **Table 18's reference metric.** A decision for the authors (section above). The clean experiment,
   if they want one, is to rerun the Table 18 reference at a less saturated `glass_end_time` and see
   whether the reference norm becomes comparable to the estimates'.
4. **Table 13 and Fig 4 under convention 2.** Both still use the stock clip-down rule. On Table 13
   the scale-free rule is worth a significant improvement; on Fig 4 it was not measured here, because
   changing it would change what the figure is.
5. **Other lengths and rewards.** Everything here is L = 50, plus `motif` for Table 13 and `gc` for
   Table 18. A second Table 18 reward (`motif`, Slurm 49471660) was running when this was written and
   landed in `results/dna/operating_point/table18_grid_cpu/motif/`; it is the check that the
   reference-norm collapse is not `gc`-specific; it **completed** and confirmed it. The `conjunction`
   reward and L = 100/200/400 were not run.
