# dMFM vs GLASS parity on the DNA C0-guidance experiment (Table 1 / §7 item 8)

Status 2026-09-29. Owner: the parity agent of the 2026-09-29 reproducibility round.
Everything here is L=50, the parent-disjoint base DFM, the parent-disjoint C0 guide and the
parent-disjoint C0 oracle, i.e. the Table 1 setting.

## Verdict

**Superseded in part, 2026-09-29 late.** The n=800 Table 1 dMFM rerun on shared ids, under the
identical scale-free rule, measures dMFM **0.040 +- 0.007 (about 5.6 SEM) behind GLASS** at
y* = +1 -- dMFM guided 0.193 [0.184, 0.204] against GLASS 0.152 [0.144, 0.160]. That is the
governing number. My n=64 paired estimate below (+0.017 +- 0.018) is *statistically compatible*
with it but far less precise, so **do not quote this document for parity.** The correct framing is:
the corrected dMFM steers well as a demonstration (unguided 1.17 -> guided 0.19, 97.5% of pairs
improved) but is **modestly and significantly behind GLASS wherever the two are compared**.

**What stands, and is the point of this document: two wiring defects, found and fixed.** Together
they are what took the DNA dMFM guided oracle MAE from 0.610 to 0.193 at full n; both were then
confirmed independently at n=800. Neither is a property of the method.

**What does not stand:** the "matched-setting parity" reading in section 2 below and its
best-of-grid follow-up in section 6. Both are n=32-64 measurements whose confidence intervals
straddle the n=800 answer. They are kept because the *knob-by-knob* effects they isolate are large
compared with their error bars and are not reproduced anywhere else -- the ranking of posterior
artifacts, step counts and guidance rules is the useful content -- but every statement in them of
the form "dMFM equals GLASS" should be read as "dMFM is within this run's noise of GLASS, and a
tighter run put it 0.040 +- 0.007 behind".

**Defect 1 (checkpoint/step mismatch).** The n=1000 "Table 1 with dMFM" run (Slurm 49277977,
2026-09-29) composes **four** posterior flow-map steps of the **wrong dMFM artifact**. It loads the
*diagonal* student `checkpoints/dna/dmfm/L50/epoch=95-step=49000.ckpt`, which every other dMFM
result in the paper uses as a **one-step** map, instead of the *4-step ESD* student
`checkpoints/dna/dmfm_4step/L50/epoch=29-step=15000.ckpt`, which is distilled and
checkpoint-selected for exactly that four-step composition and is the artifact Table 18 measures.
The cost is measurable on the value gradient: against a Table-18-style reference the 4-step student
at 4 steps has the *same* direction accuracy as the Table 1 GLASS estimator (cosine 0.430 / 0.300
at t = 0.5 / 0.7 vs GLASS's 0.430 / 0.358), while the diagonal student at 4 steps is 2.5-3x worse
(0.171 / 0.117) -- the worst of the nine samplers tried.

**Defect 2 (guidance scale, the dominant one).** `guided_sample` applies
`dt * min(guidance_frac * sigma_t^2, coeff_cap) * clip(grad V, grad_clip)`, and the clip **only
clips down**. Table 1's constants (`guidance_frac 8`, `coeff_cap 10`, `grad_clip 10`) were tuned to
the GLASS-4 gradient magnitude, which along the guided trajectory has mean norm 1.1-3.1 and exceeds
the clip at 1-5% of guided steps. Every dMFM posterior's gradient there has mean norm 4.6-125 and
exceeds the clip at 6-52% of steps; the configuration actually shipped sits at 39.2 / 52%.
So under the stock rule the dMFM arm's trajectory is dominated by a minority of saturated steps -- the gradient is heavy-tailed where GLASS's is not -- and it is driven
off the base-flow manifold. That, not the student, is what produces the 3x-worse MAE: at the stock
constants the corrected 4-step student is *worse* than the mis-paired one (0.729 vs 0.632 at
y* = +1). Note that this is **not** fixed by simply turning the dMFM arm down: under clip-down,
`guidance_frac` 8 -> 4 -> 2 -> 1 makes it monotonically worse (0.729 -> 0.792 -> 0.898 -> 1.109) and
so does shrinking `grad_clip`. What fixes it is making the step *scale-free*, so every guided step
contributes an equal-magnitude nudge instead of a few saturated ones (section 6).

**Matched-setting measurement (n = 64, superseded by n=800 -- see the note above).** Applying the
*same* scale-free rule to both arms -- rescale the gradient to exactly `grad_clip` instead of only
clipping it down, `guidance_frac 8` unchanged, nothing else touched -- dMFM came within one paired
SEM of GLASS at this sample size. The n=800 rerun later resolved that "within noise" into a real
0.040 +- 0.007 deficit, so read the table as *how far the fixes get you*, not as parity:

| y* | GLASS, scale-free | best dMFM, scale-free | paired difference |
|---|---|---|---|
| +1 | 0.171 | **0.189** (diagonal student, 4 steps) | +0.017 +- 0.018 |
| -1 | 0.128 | **0.153** (4-step student, 4 steps) | +0.025 +- 0.019 |

For reference, GLASS at the stock Table 1 rule is 0.160 (+1) and 0.157 (-1), and the shipped dMFM
run is 0.632 (+1) and 0.380 (-1). So the fixes move dMFM from ~4x GLASS's error to **within about
25% of it** -- which the n=800 rerun pins at 0.193 vs 0.152, a 0.040 +- 0.007 deficit. The headline
is the size of the repair, not a tie.

Letting each arm pick its own `guidance_frac` from the *same* grid under the *same* scale-free rule
(section 6, n=32) sharpens this further. dMFM's optimum is `guidance_frac 16` and GLASS's is 4-8:

| y* | GLASS best-of-grid | dMFM best-of-grid (`gf=16`) |
|---|---|---|
| +1 | 0.150 +- 0.023 (`gf=4`, stock rule) | **0.144 +- 0.020** |
| -1 | 0.104 +- 0.014 (`gf=8`, scale-free) | 0.165 +- 0.025 |

i.e. at this sample size dMFM and GLASS are statistically indistinguishable at y* = +1 (a 0.006
difference against SEMs of ~0.02, a quarter of one SEM -- never a win) and dMFM is about 0.06
(~2 SEM) behind at y* = -1. The n=800 rerun supersedes the y* = +1 reading with a measured
0.040 +- 0.007 deficit. The reason dMFM's optimum sits higher is mechanical: normalising to
`grad_clip` *lowers* the step wherever the raw gradient exceeded 10, which is 30-66% of guided steps
for dMFM and 2-17% for GLASS, so a larger `guidance_frac` restores the same total drive. GLASS gains
nothing from a larger `guidance_frac`, as expected when its clip almost never binds.

**A third finding, which the authors need to see.** A *more accurate* GLASS posterior guides
*worse*, not better: `glass_euler32` (32 Euler steps instead of 4) leaves the guided oracle MAE
exactly at the unguided value (1.1905 vs 1.1907 at y* = +1; 0.8098 vs 0.8098 at y* = -1) on an
effectively zero gradient (mean norm 0.002 at y* = -1; the mean of 28.8 at y* = +1 is a rare-spike
artifact -- only 0.4% of guided steps exceed the clip at all). The reference posterior (GLASS-25 RK4 to 0.999) has value
gradient norm ~0.15, i.e. **100-700x smaller than any of the finite-step estimators used for
guidance**. The C0-guidance signal of Table 1 is therefore carried by the discretisation error of a
deliberately coarse 4-step posterior, and `guidance_frac`/`grad_clip` are calibrated against that
error. This is a property of the experiment as published, not of dMFM, but it means "use a better
posterior sampler" is not, in this setup, a route to better steering, and it is why the guidance
constants cannot be transported between posterior samplers without renormalisation.

The two students are not interchangeable at a given step count:

| | `checkpoints/dna/dmfm/L50` (diagonal) | `checkpoints/dna/dmfm_4step/L50` |
|---|---|---|
| run dir | `yeast_parent_a_L50_dmfm_diagonal_h192_b4` | `yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727` |
| `mfm_consistency_gap_mode` | `random` | `long` |
| ESD gap range | random ordered-uniform pair, mean 1/3 | `[0.0, 0.25]` |
| `mfm_consistency_anneal_end_step` | 1 (<= warmup 1000, so full-random after step 1001) | 25000 |
| checkpoint metric | `val_loss` | `val_mfm_probe_esd_gap_0.25` |
| `init_ckpt` | -- (from the base DFM) | the diagonal student |
| used at | **1** step: Table 13 (`one_step_dmfm`), Table 17 (`dmfm_steps 1`), Fig 4 / `dmfm.benchmarks` (`nfe_value 1`) | **4** steps: Table 18 (`dmfm_sampler flow_map`, `dmfm_steps 4`) |

So every dMFM result the paper reports pairs a student with the jump it is used at -- the
random-gap student as a single map, the gap-<=0.25 student in four steps -- except the C0-guidance
run, which composes the random-gap student four times. (Note that the diagonal student's ESD loss
*did* see gaps near 0.25: with `gap_mode=random` and `anneal_end_step 1 <= warmup 1000`,
`_sample_s_u` returns a full ordered-uniform pair from step 1001 on. It is not that 0.25 is
out-of-distribution for it, but that it was never annealed to, or selected on, small gaps, whereas
the 4-step student was both -- `val_mfm_probe_esd_gap_0.25` is literally its checkpoint metric.)
That asymmetry is the shape of the paradox in the brief: the *gradient* tables (13, 17, 18) agree
with GLASS because they use the matched pairing, and only the end-to-end guided sampler was
mis-wired.

Ruled out along the way:

* **EMA.** Neither checkpoint contains EMA weights at all (`state_dict` has 78 `model.*`
  tensors, no `ema`/`*_ema` key, no EMA callback in `callbacks`). Nothing to load, so raw-vs-EMA
  cannot be the defect. Both are Lightning `ModelCheckpoint` files loaded strictly (`load_student`
  raises on any missing/unexpected key).
* **Call signature / time convention.** The guided sampler reaches the student through
  `api.dmfm_posterior(..., sampler="flow_map")` -> `ablate_dmfm_one_step_gradient_mc.sample_dmfm`,
  which is the *same function* Table 18 calls, and `DNAMFMStudent.forward(s, t, x, t_cond, x_cond)`
  returns `X_{s,t}(x) = x + (t-s)/(1-s) (Psi_{s,t}(x) - x)`, a clean-sample estimate, as GLASS's
  posterior does. No sign or convention flip: the only difference from Table 18 is the checkpoint
  and `end_time`.
* **MC budget and guidance strength.** Both arms use MC=8, the same frozen per-sample noise pool
  (`seed + 1e6 + i`), `guidance_frac 8.0`, `coeff_cap 10.0`, `grad_clip 10.0`, `nfe_traj 64`,
  `t_max 0.95`, window `[0.5, 0.95]`, `reward_sigma 0.15`, `reward_scale 0.5`, batches of 8,
  seed 0. Verified from the stored `args.json` of both n=1000 runs. Nothing was tuned per arm.
* **Step count silently defaulting.** `--nfe_value 4` is passed explicitly by
  `scripts/dna/table1_c0_guidance_dmfm.sbatch`; the module default of 1 is never in play. The
  jump schedule is `torch.linspace(0, 1, 5)`, i.e. four equal jumps of 0.25, with no annealing
  knob involved.

## What was run

New module `dmfm.experiments.parity_dmfm_glass_c0` (two modes) and wrapper
`scripts/dna/parity/dmfm_glass_parity.sbatch`. Every configuration goes through the *same*
sampler with the *same* initial noise (`seed + i`), the *same* frozen MC pool
(`seed + 1e6 + i`), the same guide and the same Table 1 guidance settings; only the posterior
sampler changes, and where a label says so, exactly one named knob.

Configuration labels used below:

| label | posterior |
|---|---|
| `glass_euler4` | GLASS on the base DFM, 4 Euler steps to 1.0 -- the Table 1 estimator |
| `glass_euler32` | GLASS, 32 Euler steps -- intended as a posterior-accuracy ceiling; it turns out to guide not at all (section 2) |
| `diag_fm4` | diagonal student, 4 composed flow-map steps -- **what job 49277977 ran** |
| `diag_fm1` | diagonal student, 1 map (its Table 13/17 pairing) |
| `dmfm4_fm4` | 4-step ESD student, 4 composed steps -- **the author's "4-step dMFM"** |
| `dmfm4_fm1`, `dmfm4_fm2`, `dmfm4_fm8` | the 4-step student forced to 1 / 2 / 8 steps |
| `*_e999` | same but integrating to `end_time = 0.999` (Table 18's integrator) |
| `*@norm` | the value gradient rescaled to exactly `grad_clip = 10` instead of only clipped down |
| `*@gf=<x>` | `guidance_frac` changed to `<x>` for that configuration only |

## 1. Value-gradient agreement (no sampling)

`--mode grad`. Two references were run, because the choice matters:

* **Slurm 49350990**, `results/dna/parity/grad_L50_rk4ref/` -- reference = GLASS, **25 RK4 steps to
  end_time 0.999**, i.e. Table 18's reference integrator. 4 probe states from the base flow, both
  targets, MC=8, the *identical* MC pool for every configuration. **This is the one to quote.**
* **Slurm 49345753**, `results/dna/parity/grad_L50/` -- reference = GLASS, 100 Euler steps to
  end_time **1.0**, 8 probes. Kept for the record only: a fine Euler grid run all the way to t=1
  crosses the singular linear-DFM data-time velocity, driving the endpoint onto a hard corner, so
  that reference is numerically degenerate and its cosines are compressed towards zero.

Cosine similarity of the finite-MC value gradient with the reference gradient, by outer time,
against the RK4-to-0.999 reference:

| config | cosine t=0.5 | cosine t=0.7 | mean \|g\| t=0.5 | mean \|g\| t=0.7 | endpoint L1 to ref, t=0.7 |
|---|---|---|---|---|---|
| `dmfm4_fm2` | **0.591** | **0.441** | 52.0 | 40.7 | 0.041 |
| `dmfm4_fm1` | 0.507 | 0.369 | 35.7 | 108 | 0.053 |
| `dmfm4_fm4_e999` | 0.439 | 0.340 | 29.8 | 8.68 | 0.010 |
| `glass_euler4` (Table 1) | 0.430 | 0.358 | 17.5 | 38.8 | 0.048 |
| `dmfm4_fm4` (corrected) | 0.430 | 0.300 | 27.6 | 8.65 | 0.014 |
| `diag_fm1` | 0.261 | 0.296 | 21.4 | 40.2 | 0.078 |
| `dmfm4_fm8` | 0.251 | 0.284 | 93.8 | 0.055 | 0.004 |
| `diag_fm4_e999` | 0.195 | 0.141 | 22.0 | 27.4 | 0.030 |
| `diag_fm4` (job 49277977) | **0.171** | **0.117** | 21.8 | 30.6 | 0.033 |
| *reference* `glass rk4-25 @0.999` | 1 | 1 | **0.151** | **0.166** | 0 |

Reading:

* **The corrected pairing matches GLASS on gradient direction** (0.430 / 0.300 vs 0.430 / 0.358),
  and is markedly closer to the reference *endpoint* (L1 0.014 vs 0.048 at t=0.7).
* **The shipped pairing is the worst of the nine** (0.171 / 0.117), 2.5-3x below both. So the
  checkpoint mismatch is not cosmetic -- it measurably degrades the guidance direction.
* Forcing either student to a step count it was not distilled for costs accuracy in both
  directions: `dmfm4_fm8` (gap 0.125) drops to 0.251 and `diag_fm4`/`diag_fm1` to 0.171/0.261.
  `end_time 0.999` is a small, consistent improvement (+0.02 to +0.04 cosine) for both students.
* The reference gradient norm is **0.15**, while every estimator actually used for guidance has a
  norm of 8-125. Finite-step posterior value gradients are two to three orders of magnitude larger
  than the well-integrated one. This is the root of defect 2, and see section 2's `glass_euler32`
  row for its consequence.
* Near t=0.9 the reference norm falls to 5e-5 and all cosines go negative; the guidance term there
  is numerically irrelevant. Per-sample relative L2 is in the CSV but is dominated by division by
  those near-zero norms -- use the cosine and `rel_l2_pooled`.

## 2. End-to-end guided sampling, matched noise

`--mode sample`, Slurm 49349568 (gpu_test, A100-SXM4-40GB), `results/dna/parity/sample_L50_n64/`.
64 paired unguided/guided samples per target per configuration. Identical initial noise, identical
frozen MC=8 pool, identical guide, identical Table 1 trajectory settings; only the posterior
sampler changes, plus one named knob where the label says so. Scores are from the independent
park_cnn oracle, reverse-complement averaged, exactly as Table 1. `vs GLASS` and `vs GLASS-norm`
are the **per-sample paired** difference in guided oracle error against `glass_euler4` and
`glass_euler4@norm`, with its SEM; positive means worse than that baseline.

**y\* = +1** (unguided oracle MAE 1.191, n = 64):

| config | guided oracle MAE | guided oracle mean | mean \|g\| | frac steps \|g\| > clip | vs GLASS | vs GLASS-norm |
|---|---|---|---|---|---|---|
| `glass_euler4` (Table 1) | **0.160** | 0.909 | 3.11 | 0.054 | 0 | -0.011 +- 0.021 |
| `glass_euler4@norm` | 0.171 | 0.908 | 2.71 | 0.063 | +0.011 +- 0.021 | 0 |
| `diag_fm4@norm` | **0.189** | 0.900 | 15.5 | 0.379 | +0.029 +- 0.025 | **+0.017 +- 0.018** |
| `diag_fm1` | 0.200 | 0.854 | 11.6 | 0.371 | +0.040 +- 0.029 | +0.029 +- 0.024 |
| `dmfm4_fm1` | 0.211 | 0.862 | 10.2 | 0.363 | +0.051 +- 0.025 | +0.039 +- 0.021 |
| `dmfm4_fm4@norm` | 0.228 | 0.900 | 15.5 | 0.304 | +0.068 +- 0.029 | +0.057 +- 0.026 |
| `diag_fm4` (**job 49277977**) | 0.632 | 0.388 | 39.2 | 0.521 | +0.473 +- 0.053 | +0.461 +- 0.055 |
| `dmfm4_fm4_e999` | 0.652 | 0.356 | 36.7 | 0.429 | +0.492 +- 0.042 | +0.481 +- 0.047 |
| `dmfm4_fm4@gf=32` | 0.670 | 0.344 | 38.3 | 0.433 | +0.510 +- 0.051 | +0.499 +- 0.052 |
| `dmfm4_fm4` (checkpoint fix only) | 0.729 | 0.271 | 37.3 | 0.421 | +0.569 +- 0.044 | +0.558 +- 0.049 |
| `dmfm4_fm8` | 1.180 | -0.180 | 125 | 0.113 | +1.020 +- 0.045 | +1.009 +- 0.047 |
| `glass_euler32` | 1.191 | -0.191 | 28.8 | 0.004 | +1.031 +- 0.043 | +1.019 +- 0.044 |

**y\* = -1** (unguided oracle MAE 0.810, n = 64):

| config | guided oracle MAE | guided oracle mean | mean \|g\| | frac steps \|g\| > clip | vs GLASS | vs GLASS-norm |
|---|---|---|---|---|---|---|
| `glass_euler4@norm` | **0.128** | -0.951 | 1.14 | 0.017 | -0.029 +- 0.019 | 0 |
| `dmfm4_fm4@norm` | **0.153** | -0.904 | 7.55 | 0.229 | -0.004 +- 0.019 | **+0.025 +- 0.019** |
| `glass_euler4` (Table 1) | 0.157 | -0.932 | 1.41 | 0.008 | 0 | +0.029 +- 0.019 |
| `dmfm4_fm1` | 0.171 | -0.906 | 4.59 | 0.083 | +0.014 +- 0.021 | +0.043 +- 0.021 |
| `diag_fm1` | 0.180 | -0.855 | 4.55 | 0.113 | +0.023 +- 0.022 | +0.052 +- 0.019 |
| `diag_fm4@norm` | 0.199 | -0.853 | 8.77 | 0.300 | +0.042 +- 0.024 | +0.071 +- 0.022 |
| `diag_fm4` (**job 49277977**) | 0.380 | -0.632 | 13.4 | 0.396 | +0.223 +- 0.039 | +0.252 +- 0.037 |
| `dmfm4_fm4@gf=32` | 0.397 | -0.629 | 17.7 | 0.438 | +0.240 +- 0.036 | +0.269 +- 0.036 |
| `dmfm4_fm4` (checkpoint fix only) | 0.407 | -0.613 | 17.5 | 0.404 | +0.250 +- 0.031 | +0.279 +- 0.031 |
| `dmfm4_fm4_e999` | 0.451 | -0.562 | 18.0 | 0.425 | +0.294 +- 0.034 | +0.323 +- 0.036 |
| `dmfm4_fm8` | 0.735 | -0.265 | 13.3 | 0.063 | +0.577 +- 0.046 | +0.607 +- 0.048 |
| `glass_euler32` | 0.810 | -0.190 | 0.002 | 0 | +0.653 +- 0.045 | +0.682 +- 0.047 |

Sanity checks built into the design: every configuration's *unguided* column is identical
(1.1907 / 0.8098) because the unguided half is driven by the same `x0` with no gradient, and
`glass_euler4` reproduces the Table 1 n=1000 oracle MAE (0.160 / 0.157 here against 0.148 / 0.119
at n=1000) within the n=64 spread.

### Every knob varied, and what it did

| knob | change | effect on guided oracle MAE (y\* = +1 / -1) |
|---|---|---|
| **posterior artifact** at 4 steps | diagonal -> 4-step ESD student | 0.632 -> 0.729 / 0.380 -> 0.407 -- **worse** at the stock guidance rule, because the corrected student is the more over-driven of the two; **better** once the scale is fixed (see below) |
| **posterior step count** | 4 -> 1 (4-step student) | 0.729 -> 0.211 / 0.407 -> 0.171 -- the biggest single stock-rule improvement, and it works by *lowering the gradient norm* from 37 to 10, not by improving the direction (section 1 shows a 1-step map is no more accurate) |
| | 4 -> 8 (4-step student) | 0.729 -> 1.180 / 0.407 -> 0.735 -- guidance destroyed, norm 125 |
| | 4 -> 1 (diagonal student) | 0.632 -> 0.200 / 0.380 -> 0.180 -- same mechanism |
| **inner end_time** | 1.0 -> 0.999 (4-step student, 4 steps) | 0.729 -> 0.652 / 0.407 -> 0.451 -- small, inconsistent in sign; improves the gradient *direction* (section 1) but not the outcome |
| **guidance scale rule** | clip-down -> rescale-to-clip (`@norm`), applied to **both** arms | GLASS 0.160 -> 0.171 / 0.157 -> 0.128; dMFM 4-step 0.729 -> **0.228** / 0.407 -> **0.153**; dMFM diagonal 0.632 -> **0.189** / 0.380 -> **0.199**. **This is the knob that closes the gap**, and it is a change to the *comparison*, not to dMFM: the same rule is applied to GLASS, where it costs 0.011 at +1 and gains 0.029 at -1 |
| **guidance_frac** | 8 -> 32 (4-step student, stock rule) | 0.729 -> 0.670 / 0.407 -> 0.397 -- barely moves, because the clip already saturates 42% of steps; confirms `guidance_frac` is not the operative knob once the clip binds |
| **MC count** | not varied: both arms use N = 8 throughout, the paper's DNA operating point, with the identical frozen per-sample pool | -- |
| **EMA** | not applicable: neither checkpoint contains EMA weights | -- |
| **posterior accuracy** | GLASS 4 -> 32 Euler steps | 0.160 -> 1.191 / 0.157 -> 0.810, i.e. **guidance vanishes entirely**; gradient norm 3.11 -> 28.8 (heavy-tailed: only 0.4% of steps above the clip) at +1 and 1.41 -> 0.002 at -1 |

Two things follow. First, **the honest matched-setting comparison is the `@norm` block**: run both
arms under the same scale-free rule and dMFM equals GLASS to within a paired SEM. Second, the
apparent 3x gap in the shipped run is a guidance-strength artifact, and it is *not* fixed by turning
a knob in dMFM's favour -- `guidance_frac 8` is unchanged, and the one rule that changes is applied
identically to GLASS.

## 3. The n=1000 reference numbers already on disk

`results/dna/c0_guidance/table1_dmfm_n1000_summary.json` does **not** exist yet, but the raw
dMFM n=1000 arrays **have landed**: Slurm 49277977 finished all five shards for both targets in
`outputs/dna/table1_dmfm_n1000/target_{1p0,m1p0}/shards/*/sample_scores.csv` (1,000 pairs each,
H100). They are the *defective* `diag_fm4` configuration -- every shard's `args.json` records
`student_ckpt = checkpoints/dna/dmfm/L50/epoch=95-step=49000.ckpt`, `nfe_value = 4`. Guide-scored
(not oracle-scored) guided MAE from those arrays:

| posterior, n=1000 | guided MAE (guide), y*=+1 | y*=-1 |
|---|---|---|
| GLASS (`results/dna/c0_guidance/n1000`) | 0.0579 | 0.0441 |
| dMFM `diag_fm4` (49277977) | 0.5757 | 0.3652 |

The oracle-scored GLASS n=1000 values are 0.148 [0.141, 0.156] and 0.119 [0.114, 0.125]
(`results/dna/c0_guidance/table1_n1000_summary.json`). So the ~3x gap in the brief is confirmed at
n=1000, not just n=8 -- and it is measured on the mis-paired configuration.

## 3b. What this does to Table 1 and to REPRODUCIBILITY_PLAN.md §7 item 8

§7 item 8 decided to "rerun Table 1 with dMFM, n=1000, paired", on the evidence that "the one
surviving matched dMFM run (n=32, unpaired) suggests markedly worse guided MAE". That evidence, and
the n=1000 rerun built on it, are both from the mis-wired configuration. Concretely:

1. **Do not print the 49277977 numbers as Table 1.** They are the diagonal student composed in four
   steps under guidance constants calibrated for GLASS. Whatever the authors decide to publish, that
   configuration is not "the dMFM".
2. **The n=1000 rerun has to be redone**, with the step-matched student. The corrected command is
   now the default of `scripts/dna/table1_c0_guidance_dmfm.sbatch` (which writes to
   `outputs/dna/table1_dmfm4_n1000`, so the legacy shards stay intact) and costs ~2 min per
   200-pair shard on an H100 -- the same as the run already done.
3. **A decision is needed on the guidance rule, and it is the substantive one.** With the stock
   Table 1 rule the corrected 4-step dMFM is still ~4x worse than GLASS (0.729 / 0.407), because the
   rule is calibrated to the GLASS-4 gradient magnitude, and turning the dMFM arm *down* makes it
   worse rather than better (section 6). With the gradient normalised to `grad_clip` -- applied to
   *both* arms, so the comparison stays honest -- dMFM closes most of the gap at matched
   `guidance_frac 8` (0.189-0.228 / 0.153-0.199 against 0.171 / 0.128) and the n=800 rerun pins the
   residual at 0.040 +- 0.007 behind GLASS at y* = +1. The paper cannot claim dMFM parity on
   DNA while using a guidance rule whose effective strength depends on which posterior sampler is
   plugged in. Either report the scale-free rule for both arms, or state the per-arm
   `guidance_frac`/`grad_clip` explicitly and say they were calibrated separately -- and if the
   latter, say so for GLASS too, whose published `guidance_frac 8` is not its optimum either
   (gf = 4 is better at y* = +1).
   `--grad_normalize` (off by default) implements the scale-free rule in the production sampler.
4. **App. F.4 / §3.1 wording.** "We evaluate dMFM for DNA design" is reachable after the rerun, but
   the appendix must say which student (the 4-step ESD one), how many composed steps (4), and which
   gradient-scale rule. None of the three is currently stated anywhere in the paper.
5. **New discrepancy for the §7.9 list.** The DNA C0-guidance signal comes from the discretisation
   error of the 4-step posterior, not from the posterior itself: the well-integrated value gradient
   has norm ~0.15 where the 4-step estimators have 3-125, and a 32-step GLASS posterior guides not
   at all. Any statement of the form "more posterior accuracy gives better steering" is false for
   this experiment, and the Fig 7 guidance-strength sweep is really a sweep of that error times
   `guidance_frac`.

Not affected: Tables 13, 17, 18 and the Fig 4 / `dmfm.benchmarks` value-error work all pair a
student with the jump it was distilled for (`one_step_dmfm`, `dmfm_steps 1`, `nfe_value 1`, and
Table 18's `dmfm_steps 4` on the 4-step student). I checked every `paths.dmfm_ckpt` consumer in
`src/dmfm`; the C0-guidance path was the only mismatch.

## 4. Code changes

All changes keep the old behaviour reachable behind a flag.

* `src/dmfm/paths.py`: added `dmfm_student_kind(n_steps)`, `dmfm_ckpt_for_steps(length, n_steps)`
  and `dmfm_args_for_steps` -- one place recording which artifact each posterior step count was
  distilled for (1 step -> `dmfm`, any few-step composition -> `dmfm_4step`).
* `src/dmfm/experiments/sample_c0_guidance_dmfm.py`: `--student_ckpt` now defaults to `None` and
  resolves to the artifact matching `--nfe_value`; added `--student_kind {auto,dmfm,dmfm4}` and
  `--allow_gap_mismatch`. Without the latter, a `--nfe_value` whose composed jump `1/nfe_value`
  does not match the loaded student's distillation jump is a hard error naming the right artifact.
  The resolved checkpoint is written into the run's `args.json` before sampling starts.
  **Job 49277977 is reproduced exactly with**
  `--student_ckpt checkpoints/dna/dmfm/L50/epoch=95-step=49000.ckpt --allow_gap_mismatch`.
* `src/dmfm/api.py`: `guided_sample` gained `grad_normalize: bool = False`. With the default the
  loop is byte-identical to before (`clamp(max=1.0)` on the clip factor), so no stored result
  changes; `True` rescales the gradient to exactly `grad_clip`.
* `src/dmfm/experiments/sample_c0_guidance_dmfm.py`: also gained `--grad_normalize` (off by
  default), forwarded to `api.guided_sample` in the `--paired` path.
* `src/dmfm/experiments/parity_dmfm_glass_c0.py`: new; the harness described above. Nothing else
  imports it, so no stored result can change because of it.
* `scripts/dna/table1_c0_guidance_dmfm.sbatch`: `DMFM_CKPT` now defaults to
  `paths.dmfm_ckpt_for_steps(50, NFE_VALUE)` and the default `OUT_ROOT` becomes
  `outputs/dna/table1_dmfm4_n${N_SAMPLES}`, so a corrected run cannot be merged with the legacy
  shards in `outputs/dna/table1_dmfm_n1000`. `LEGACY_DIAG_STUDENT=1` restores the old checkpoint,
  the old `OUT_ROOT` and `--allow_gap_mismatch`; `NFE_VALUE` is now a variable.
* `tests/dna/test_script_args.py::test_table1_dmfm_matches_table1_settings`: the assertion
  `"/dmfm/L50/" in student_ckpt` became "the student is `paths.dmfm_ckpt_for_steps(50, nfe_value)`
  and `--allow_gap_mismatch` is off".

## 5. Files

* `docs/dmfm_glass_parity.md` (this file)
* `src/dmfm/experiments/parity_dmfm_glass_c0.py`
* `scripts/dna/parity/dmfm_glass_parity.sbatch`
* `results/dna/parity/PROVENANCE.md`
* `results/dna/parity/grad_L50_rk4ref/` -- the quotable gradient diagnostic (Slurm 49350990;
  reference GLASS-25 RK4 to 0.999)
* `results/dna/parity/grad_L50/` -- the same with the degenerate GLASS-100-Euler-to-1.0 reference,
  kept only so the reference choice is on the record (Slurm 49345753)
* `results/dna/parity/sample_L50_n64/` -- `sample_scores.csv`, `sample_summary.csv`,
  `run_metadata.json` (Slurm 49349568)
* `results/dna/parity/sample_L50_n32_knobs/` -- the guidance-strength sweep (Slurm 49351325),
  section 6
* `results/dna/parity/sample_L50_n32_knobs_hi/` -- the symmetric top-of-range extension of that
  grid (Slurm 49353655), section 6

## 6. Guidance-strength sweep: best-of-grid vs best-of-grid

Slurm **49351325** (gpu_test A100, 5 min), n=32 pairs per target,
`results/dna/parity/sample_L50_n32_knobs/`. Same noise and MC pool as section 2. This exists so the
parity claim is not an artifact of comparing each arm at a single strength: both arms are swept over
the same `guidance_frac` grid, with and without the scale-free rule, and `grad_clip` is swept for
dMFM. Guided oracle MAE +- SEM:

**y\* = +1** (unguided 1.170):

| config | gf | clip | scale-free | guided oracle MAE | mean \|g\| | frac \|g\| > clip |
|---|---|---|---|---|---|---|
| `dmfm4_fm4@norm@gf=16` | 16 | 10 | yes | **0.144 +- 0.020** | 17.8 | 0.300 |
| `glass_euler4@gf=4` | 4 | 10 | no | **0.150 +- 0.023** | 3.56 | 0.042 |
| `glass_euler4` (Table 1) | 8 | 10 | no | 0.160 +- 0.023 | 2.89 | 0.042 |
| `glass_euler4@norm@gf=8` | 8 | 10 | yes | 0.166 +- 0.027 | 2.79 | 0.058 |
| `glass_euler4@norm@gf=4` | 4 | 10 | yes | 0.171 +- 0.018 | 2.92 | 0.050 |
| `glass_euler4@norm@gf=2` | 2 | 10 | yes | 0.185 +- 0.028 | 4.25 | 0.058 |
| `diag_fm4@norm@gf=4` | 4 | 10 | yes | 0.210 +- 0.028 | 19.6 | 0.442 |
| `dmfm4_fm4@norm@gf=8` | 8 | 10 | yes | 0.219 +- 0.037 | 17.2 | 0.325 |
| `glass_euler4@gf=2` | 2 | 10 | no | 0.230 +- 0.045 | 6.95 | 0.167 |
| `dmfm4_fm4@norm@gf=4` | 4 | 10 | yes | 0.256 +- 0.031 | 12.4 | 0.308 |
| `dmfm4_fm4@norm@gf=2` | 2 | 10 | yes | 0.385 +- 0.052 | 16.9 | 0.333 |
| `dmfm4_fm4@clip=4` | 8 | 4 | no | 0.753 +- 0.080 | 46.2 | 0.583 |
| `dmfm4_fm4@gf=4` | 4 | 10 | no | 0.792 +- 0.072 | 40.2 | 0.517 |
| `dmfm4_fm4@clip=2` | 8 | 2 | no | 0.841 +- 0.069 | 46.3 | 0.658 |
| `dmfm4_fm4@gf=2` | 2 | 10 | no | 0.898 +- 0.072 | 48.9 | 0.483 |
| `dmfm4_fm4@gf=1` | 1 | 10 | no | 1.109 +- 0.059 | 65.1 | 0.583 |

**y\* = -1** (unguided 0.831):

| config | gf | clip | scale-free | guided oracle MAE | mean \|g\| | frac \|g\| > clip |
|---|---|---|---|---|---|---|
| `glass_euler4@norm@gf=8` | 8 | 10 | yes | **0.104 +- 0.014** | 1.15 | 0.017 |
| `glass_euler4@norm@gf=2` | 2 | 10 | yes | 0.107 +- 0.016 | 1.57 | 0.017 |
| `glass_euler4@norm@gf=4` | 4 | 10 | yes | 0.127 +- 0.013 | 1.23 | 0.017 |
| `dmfm4_fm4@norm@gf=16` | 16 | 10 | yes | **0.165 +- 0.025** | 10.4 | 0.317 |
| `glass_euler4@gf=4` | 4 | 10 | no | 0.178 +- 0.024 | 1.86 | 0.017 |
| `glass_euler4` (Table 1) | 8 | 10 | no | 0.183 +- 0.024 | 1.55 | 0.017 |
| `dmfm4_fm4@norm@gf=8` | 8 | 10 | yes | 0.194 +- 0.025 | 8.68 | 0.308 |
| `glass_euler4@gf=2` | 2 | 10 | no | 0.200 +- 0.024 | 2.38 | 0.017 |
| `diag_fm4@norm@gf=4` | 4 | 10 | yes | 0.218 +- 0.031 | 10.8 | 0.383 |
| `dmfm4_fm4@norm@gf=4` | 4 | 10 | yes | 0.251 +- 0.029 | 15.1 | 0.358 |
| `dmfm4_fm4@norm@gf=2` | 2 | 10 | yes | 0.319 +- 0.036 | 18.0 | 0.342 |
| `dmfm4_fm4@gf=4` | 4 | 10 | no | 0.480 +- 0.050 | 30.8 | 0.467 |
| `dmfm4_fm4@clip=4` | 8 | 4 | no | 0.508 +- 0.056 | 20.1 | 0.567 |
| `dmfm4_fm4@clip=2` | 8 | 2 | no | 0.557 +- 0.055 | 28.3 | 0.633 |
| `dmfm4_fm4@gf=2` | 2 | 10 | no | 0.560 +- 0.064 | 30.1 | 0.525 |
| `dmfm4_fm4@gf=1` | 1 | 10 | no | 0.679 +- 0.060 | 34.1 | 0.425 |

This sharpens defect 2 and partly corrects the simple "over-driven" reading:

* **Turning the dMFM arm *down* under the stock rule makes it monotonically worse**, not better:
  `guidance_frac` 8 -> 4 -> 2 -> 1 gives 0.729 -> 0.792 -> 0.898 -> 1.109 at y* = +1, and
  `grad_clip` 10 -> 4 -> 2 gives 0.729 -> 0.753 -> 0.841. So the problem is not simply that the
  total drive is too large. The precise statement is that the dMFM posterior's value gradient is
  **heavy-tailed along the trajectory** (mean norm 8-65 with the clip binding at 30-66% of steps,
  against GLASS's 1.1-7 binding at 2-17%), so under clip-down the trajectory is dominated by a
  minority of saturated steps. Rescaling to a fixed norm removes the tail and makes every guided
  step contribute equally; *then* more strength helps, monotonically, up to gf = 16.
* **Best-of-grid vs best-of-grid**: at y* = +1 the corrected 4-step dMFM at gf = 16 with the
  scale-free rule is **0.144 +- 0.020** against GLASS's best **0.150 +- 0.023** -- dMFM is at least
  as good. At y* = -1 it is **0.165 +- 0.025** against GLASS's best **0.104 +- 0.014**, so dMFM is
  behind by ~0.06 (about 2 SEM) on that target while still beating stock GLASS (0.183).
* **So the one knob that has to differ is `guidance_frac`: 16 for dMFM against 4-8 for GLASS**, and
  the reason is mechanical rather than cosmetic -- normalising the gradient to `grad_clip` *lowers*
  the dMFM step wherever the raw gradient exceeded 10 (30-66% of steps), so a larger
  `guidance_frac` is needed to restore the same total drive. GLASS does not benefit from a larger
  `guidance_frac` (its optimum on this grid is 4-8 and gf = 2 is already near-optimal at y* = -1),
  which is exactly what one expects when the clip almost never binds. The matched-`guidance_frac`
  comparison is still reported in section 2 and in the row `dmfm4_fm4@norm@gf=8` above, so nothing
  rests on the per-arm choice.

A follow-up run (**Slurm 49355914**, `kempner_h200`, authoritative; writes
`results/dna/parity/sample_L50_n32_knobs_hi/`) extends the scale-free `guidance_frac` grid to 16 and
32 for GLASS as well, so the best-of-grid comparison is symmetric to the top of the range. It is
**no longer load-bearing**: since the n=800 rerun measures the deficit directly, this run can only
show whether GLASS improves further at the top of its grid, i.e. whether the measured deficit is
*wider* than 0.040 -- it cannot restore a tie. If the directory is absent the job never got
capacity (the cluster went to zero idle nodes on every reachable partition around 23:00 on
2026-09-29); the conclusions above already give GLASS its best of gf in {2, 4, 8} with and without
the rule, and the item can be closed on the n=800 result alone.

Earlier job ids for the same run, for the record: 49353655 (gpu_test, cancelled -- its nodes sit
inside the root `powerwork2` maintenance reservation), 49355650 and 49361841 (kempner_requeue,
cancelled -- I rank ~2378 of 2438 there), 49355964 (gpu_requeue, cancelled -- `albergo_lab` scores
priority 4), 49356326 (gpu_test, cancelled). Only 49355914 is live, so there is no duplicate GPU
burn if capacity frees.

