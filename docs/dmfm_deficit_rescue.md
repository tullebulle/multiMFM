# Is the dMFM-vs-GLASS deficit on the DNA C0 experiment our error?

Status 2026-09-30. Owner: the deficit-rescue agent of the 2026-09-30 round.
All three sweeps are complete on both targets (n = 200, 200 and 600). Outstanding: the
checkpoint-criterion retrain, which is written but never obtained GPU capacity (section 4b).
Everything here is L=50, the parent-disjoint base DFM, the parent-disjoint C0 guide and the
parent-disjoint C0 oracle, i.e. the Table 1 setting. Companion documents:
`docs/dmfm_glass_parity.md` (the two wiring defects, found and fixed) and
`docs/REPRODUCIBILITY_PLAN.md` section 7 item 8 (the decision record).

## The number under examination

At n = 1000 per target, on shared sample ids, with both arms under the identical scale-free
guidance rule at `guidance_frac 8`:

| y* | dMFM guided oracle MAE | matched GLASS-4 | paired diff (dMFM - GLASS) |
|---|---|---|---|
| +1 | 0.193 [0.184, 0.204] | 0.153 [0.145, 0.160] | +0.041 +- 0.006 |
| -1 | 0.172 [0.164, 0.180] | 0.124 [0.117, 0.130] | +0.049 +- 0.006 |

An author expects dMFM to match GLASS, and two earlier "dMFM is worse" findings were both our
bugs, so a third defect was entirely plausible. This document records what was checked.

## Verdict

**Yes -- a defect, and it is in the comparison, not in dMFM.** The n=1000 comparison pins *both*
arms to a value gradient of **4 posterior steps x MC 8**. Four Euler steps is the Table 1 GLASS
estimator and MC 8 is the paper's DNA budget; neither is dMFM's operating point. Because the value
gradient costs `n_steps x mc` network calls, a one-step flow map can afford **MC 32 at the same
NFE**, and a flow map is exactly the artifact that can do that. Measured at n=200 paired per target,
on one H200, under the identical scale-free rule at `guidance_frac 8`
(`results/dna/deficit_rescue/nfe_L50_n200/`, and confirmed at n = 600 on the headline pair in
`results/dna/deficit_rescue/matched_L50_n600/`):

| comparison (paired, same ids) | y* = +1 | y* = -1 |
|---|---|---|
| **n = 1000**, the published operating point: 4 steps x MC 8, both arms | **+0.041 +- 0.006** | **+0.049 +- 0.006** |
| n = 200, the same operating point, reproduced on one H200 | +0.050 +- 0.013 | +0.041 +- 0.011 |
| **n = 600**, dMFM 1 step x MC 32 vs GLASS 4 steps x MC 8, both NFE 32 | **-0.002 +- 0.006** | **-0.010 +- 0.006** |
| n = 600, best dMFM vs best GLASS, both NFE 32 | +0.004 +- 0.006 | +0.003 +- 0.006 |

Positive means dMFM worse. Rows 1-2 are the same effect at two sample sizes, so the n=200 sweep is
measuring what the n=1000 run measured. Rows 3-4 are the **same total NFE** with each arm spending
it the way it prefers, at n = 600 with SEMs as tight as the n=1000 numbers -- and the deficit is
gone at both targets, in both directions of the comparison, with every knob offered to both arms.

Guided oracle MAE at the corrected operating point, n = 600 paired per target, both arms under the
same scale-free rule at `guidance_frac 8`, one H200 per target:

| config | steps x MC (NFE) | y* = +1 | y* = -1 | guide MAE (+1 / -1) |
|---|---|---|---|---|
| `glass_euler2@norm@mc=16` (best GLASS) | 2 x 16 (32) | 0.143 [0.134, 0.152] | 0.117 [0.109, 0.126] | 0.037 / 0.021 |
| `dmfm4_fm1@norm@mc=32` | 1 x 32 (32) | **0.146** [0.139, 0.156] | **0.120** [0.113, 0.128] | 0.054 / 0.036 |
| `glass_euler4@norm` (the n=1000 GLASS arm) | 4 x 8 (32) | 0.149 [0.140, 0.158] | 0.129 [0.121, 0.138] | 0.052 / 0.036 |
| `diag_fm1@norm@mc=32` (the convention's 1-step artifact) | 1 x 32 (32) | 0.154 [0.144, 0.164] | **0.115** [0.108, 0.123] | 0.057 / 0.032 |
| *published dMFM row, for contrast* | 4 x 8 (32) | *0.193 [0.184, 0.204]* | *0.172 [0.164, 0.180]* | *0.125 / 0.105* |

All four NFE-32 configurations lie inside each other's intervals; the published dMFM row lies
outside all of them. On the *guide* -- the function the sampler actually optimises -- dMFM goes from
2.4-2.8x GLASS's error at the published operating point to 0.054 / 0.036 against GLASS-4's
0.052 / 0.036, i.e. parity.

Three other hypotheses were tested and found clean or not decisive:

* **Oracle scoring, output format, ids, bootstrap:** identical across arms; the deficit reproduces
  to the fourth decimal on an independent re-score (section 1). No defect.
* **The shipped 4-step student's checkpoint selection:** genuinely questionable -- it is step 15,000
  of a 150,000-step run, chosen by the only validation metric in that run that gets *worse* with
  training, and it ranks 127th of 150 on val_loss (section 2). No later checkpoint survives anywhere,
  so testing it needs a retrain; that retrain is written
  (`scripts/dna/rescue/train_dmfm4_longer.sbatch`, ~35 min on one H100) but was not run. It is also
  no longer load-bearing: the same artifact reaches parity once it is used at one step with MC 32,
  so a better checkpoint cannot change the conclusion -- it could only make the corrected number
  better still.
* **The guidance rule and its strength: a second-order knob, not the fix.** Sweeping
  `guidance_frac` over {8, 16, 32} under the scale-free rule for *both* arms, plus a
  magnitude-preserving rule (`@relnorm`) for *both* arms, at the published 4 steps x MC 8: GLASS's
  best on the whole grid is the setting already published, and dMFM's best is +0.058 +- 0.014 and
  +0.018 +- 0.012 behind it. The knobs do help dMFM at y* = -1 (0.182 -> 0.152, -0.030 +- 0.013)
  and not at all at y* = +1; the matched-`gf` rows that look like parity are GLASS being dragged
  down, not dMFM coming up (section 3b). That section also corrects `docs/dmfm_glass_parity.md`
  section 6: its n=32 "best-of-grid tie at `gf = 16`" does not survive at n = 200
  (0.213 [0.189, 0.239] against a claimed 0.144 +- 0.020).

## 1. Oracle scoring is identical across the arms -- CLOSED, no defect

Re-scored both stored n=1000 arms from their saved sequences in a single process, one oracle
object, one code path (`api.load_c0("oracle")` + `api.oracle_score(..., rc_average=True)`), and
recomputed the paired statistic with the same 2,000-resample `default_rng(0)` bootstrap:

| arm | y* | recomputed guided oracle MAE | stored |
|---|---|---|---|
| dMFM | +1 | 0.1933 | 0.1933 |
| GLASS | +1 | 0.1527 | 0.1527 |
| dMFM | -1 | 0.1723 | 0.1723 |
| GLASS | -1 | 0.1235 | 0.1235 |

Paired dMFM - GLASS: **+0.0406 +- 0.0063**, bootstrap CI [+0.0285, +0.0532] at y* = +1, and
**+0.0488 +- 0.0055**, CI [+0.0383, +0.0595] at y* = -1. Checked and clean: 1000 unique
`sample_idx` per arm and target covering exactly 0..999; every sequence 50 nt (no padding); the
1000 *unguided* sequences are character-identical across the two arms and their oracle scores agree
to 0.0e+00; the two runs' `args.json` differ in exactly two keys, `posterior` and `out_dir`, with
`guidance_frac 8.0`, `coeff_cap 10.0`, `grad_clip 10.0`, `grad_normalize true`, `mc 8`,
`nfe_value 4`, `nfe_sample 64`, `t_max 0.95`, window [0.5, 0.95], `reward_sigma 0.15`,
`reward_scale 0.5`, `seed 0`, `batch_size 8` on both sides. Every shard's `args.json` records
`student_ckpt = checkpoints/dna/dmfm_4step/L50/epoch=29-step=15000.ckpt`, i.e. the step-matched
artifact, not the withdrawn diagonal one.

So nothing in the dMFM arm's output format or in the scoring perturbs the comparison.

### The deficit is larger on the guide than on the oracle

The oracle is a held-out proxy with its own MAE of 0.16 against measured C0, so it compresses
differences near that floor. On the *guide*, which is the function the sampler actually optimises:

| y* | dMFM guide MAE | GLASS guide MAE | dMFM within 0.10 | GLASS within 0.10 |
|---|---|---|---|---|
| +1 | 0.125 | 0.052 | 56.1% | 86.3% |
| -1 | 0.105 | 0.037 | 60.8% | 92.6% |

i.e. a factor 2.4-2.8, not 1.3-1.4. The deficit is therefore a genuine difference in how well the
guided trajectory optimises the objective, not an oracle artifact, and it is *understated* by the
oracle column that Table 1 prints.

### It is not mode collapse or over-driving either

Guided sets at n=1000, against the shared unguided set:

| set | y* | unique seqs | positional entropy (bits/2) | 3-mer entropy |
|---|---|---|---|---|
| unguided | -- | 1000/1000 | 0.9807 | 0.9780 |
| GLASS | +1 | 1000/1000 | 0.9811 | 0.9724 |
| dMFM | +1 | 1000/1000 | 0.9778 | 0.9696 |
| GLASS | -1 | 1000/1000 | 0.9723 | 0.9417 |
| dMFM | -1 | 1000/1000 | 0.9737 | 0.9492 |

No collapse in either arm, and at y* = -1 the dMFM set is very slightly *less* concentrated than
the GLASS one -- consistent with weaker steering, not with an over-driven trajectory.

## 2. The shipped 4-step student is the only one that exists -- and its selection criterion is
   anti-correlated with every other measure of the student

`checkpoints/dna/dmfm_4step/L50/epoch=29-step=15000.ckpt` is step 15,000 of a **150,000**-step run
(`workdir/yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727`), kept as the argmin of
`val_mfm_probe_esd_gap_0.25`. No other checkpoint of that run survives anywhere: the backup's run
directory holds that one file plus a **dangling** `last.ckpt -> epoch=138-step=71000.ckpt`, and the
netscratch copy is the same. The other lengths kept a late checkpoint each (L100 step 149,000, L200
step 34,000, L400 step 129,000), so L50 is the odd one out. `checkpoints/dna/dmfm_lineage` holds
only an L400 diagonal artifact.

What the run's 150 surviving validation records (`val_*.csv`, one per 1,000 steps) show, by
15,000-step block median:

| steps | val_loss | diag loss | esd loss | probe diag MSE | probe ESD gap 0.0625 | gap 0.125 | **gap 0.25 (the selection metric)** |
|---|---|---|---|---|---|---|---|
| 0-15k | 0.3668 | 0.0488 | 0.3212 | 8.17e-5 | 2.05e-3 | 8.18e-3 | **2.21e-3** |
| 15-30k | 0.3319 | 0.0396 | 0.2965 | 5.55e-5 | 7.77e-4 | 2.53e-3 | 3.73e-3 |
| 45-60k | 0.2927 | 0.0345 | 0.2574 | 4.23e-5 | 2.66e-4 | 9.35e-4 | 6.02e-3 |
| 90-105k | 0.2843 | 0.0308 | 0.2543 | 2.94e-5 | 3.87e-4 | 7.55e-4 | 2.79e-2 |
| 135-150k | **0.2644** | **0.0245** | **0.2378** | **2.42e-5** | **2.03e-4** | **6.77e-4** | **4.95e-2** |

Every quantity improves with training -- val_loss by 28%, the diagonal probe by 3.4x, the ESD probes
at gaps 0.0625 and 0.125 by 10-12x -- **except the one used to choose the checkpoint, which gets 22x
worse.** Its argmin, step 15,000, ranks **127th of 150** on val_loss (0.3466 against a best of
0.1658). Its own series is also extremely heavy-tailed (min 7.9e-4 at step 15,000, median 8.8e-3,
p90 6.6e-2, max 2.9e-1 on a 64-sample fixed probe), so the argmin is partly a draw from that tail.

Two further reasons the metric is a poor proxy for the guidance use case, from
`mfm_diag_distill.fixed_probe_metrics`: it is a *differential* ESD residual computed with
`torch.func.jvp`, not an error of the composed four-step map; and it is evaluated entirely at
**`t_cond = 0`**, while the guided sampler only ever calls the student at `t_cond` in [0.5, 0.95].
A plausible mechanism for the anti-correlation: early in training the student is still close to its
diagonal initialisation, its velocity barely varies in `s`, the JVP term is small and the residual
at the widest gap is therefore small; as real two-time structure is learned the residual grows.

Not a defect after all, for the record: `mfm_consistency_anneal_end_step 25000` is **inert** for
this run. `MFMDiagDistill._sample_s_u` applies the anneal only when `gap_mode == "random"`; with
`gap_mode == "long"` it returns `delta ~ U(min_gap, max_gap)` immediately. So the student was
trained on gaps up to 0.25 from step 1 and step 15,000 is not "before the anneal finished".
(`docs/dmfm_glass_parity.md`'s student-comparison table lists the anneal step for both students as
if it applied to both; it applies only to the diagonal one.)

Since no later checkpoint survives, the hypothesis was tested by **re-running the distillation
unchanged except for the checkpoint criterion** (`scripts/dna/rescue/train_dmfm4_longer.sbatch`:
same `init_ckpt`, teacher, data, split, objective, gap range, lr, batch, grad clip and validation
cadence; only `--mfm_checkpoint_metric val_loss` and the output directory differ, so no shipped
artifact and no stored result is touched).

## 3. Defect found: the comparison was pinned at GLASS's operating point, not dMFM's

`results/dna/deficit_rescue/nfe_L50_n200/`, Slurm 49431420 (array, `kempner_h200`). **Both targets
ran on the same GPU and the same node** (NVIDIA H200, `holygpu8a03504`, TF32 on), so nothing here
crosses card generations; within each target every configuration also shares the initial noise, so
the *unguided* oracle score is bit-identical across all twelve configurations (verified: maximum
spread 0.0 per id, all unguided sequences identical), and the entire difference is the guidance
term. All rows use the scale-free rule (`@norm`) at `guidance_frac 8`; `vs GLASS-4` is the
per-sample paired difference in guided oracle error against `glass_euler4@norm`, the GLASS arm of
the n=1000 run.

**y\* = +1** (unguided oracle MAE 1.157, n = 200):

| config | steps | MC | NFE | guided oracle MAE [95% CI] | guide MAE | vs GLASS-4 |
|---|---|---|---|---|---|---|
| `glass_euler2@norm` | 2 | 8 | **16** | **0.140** [0.127, 0.156] | 0.035 | -0.005 +- 0.010 |
| `glass_euler2@norm@mc=16` | 2 | 16 | 32 | 0.141 [0.126, 0.157] | 0.034 | -0.005 +- 0.011 |
| `glass_euler4@norm` (n=1000 GLASS arm) | 4 | 8 | 32 | 0.146 [0.131, 0.160] | 0.056 | 0 |
| `dmfm4_fm1@norm@mc=32` | 1 | 32 | 32 | **0.152** [0.136, 0.168] | 0.060 | **+0.006 +- 0.011** |
| `dmfm4_fm2@norm@mc=16` | 2 | 16 | 32 | 0.169 [0.150, 0.191] | 0.080 | +0.024 +- 0.012 |
| `diag_fm1@norm` | 1 | 8 | **8** | 0.183 [0.163, 0.202] | 0.090 | +0.037 +- 0.013 |
| `dmfm4_fm1@norm` | 1 | 8 | **8** | 0.183 [0.163, 0.205] | 0.096 | +0.037 +- 0.013 |
| `glass_euler8@norm` | 8 | 8 | 64 | 0.188 [0.168, 0.210] | 0.104 | +0.042 +- 0.013 |
| `diag_fm4@norm` | 4 | 8 | 32 | 0.188 [0.167, 0.210] | 0.116 | +0.043 +- 0.014 |
| `dmfm4_fm8@norm` | 8 | 8 | 64 | 0.193 [0.171, 0.217] | 0.135 | +0.048 +- 0.014 |
| `dmfm4_fm4@norm` (n=1000 dMFM arm) | 4 | 8 | 32 | 0.196 [0.173, 0.220] | 0.133 | **+0.050 +- 0.013** |
| `dmfm4_fm2@norm` | 2 | 8 | 16 | 0.197 [0.174, 0.219] | 0.110 | +0.051 +- 0.014 |

**y\* = -1** (unguided oracle MAE 0.824, n = 200):

| config | steps | MC | NFE | guided oracle MAE [95% CI] | guide MAE | vs GLASS-4 |
|---|---|---|---|---|---|---|
| `dmfm4_fm1@norm@mc=32` | 1 | 32 | 32 | **0.110** [0.098, 0.123] | 0.034 | **-0.020 +- 0.010** |
| `glass_euler2@norm@mc=16` | 2 | 16 | 32 | 0.118 [0.104, 0.133] | 0.023 | -0.013 +- 0.009 |
| `dmfm4_fm2@norm@mc=16` | 2 | 16 | 32 | 0.122 [0.108, 0.136] | 0.053 | -0.008 +- 0.010 |
| `glass_euler4@norm` (n=1000 GLASS arm) | 4 | 8 | 32 | 0.130 [0.117, 0.143] | 0.033 | 0 |
| `glass_euler2@norm` | 2 | 8 | **16** | 0.138 [0.123, 0.154] | 0.022 | +0.008 +- 0.010 |
| `diag_fm1@norm` | 1 | 8 | **8** | 0.145 [0.128, 0.162] | 0.063 | +0.014 +- 0.012 |
| `dmfm4_fm1@norm` | 1 | 8 | **8** | 0.145 [0.128, 0.163] | 0.067 | +0.015 +- 0.012 |
| `dmfm4_fm2@norm` | 2 | 8 | 16 | 0.147 [0.131, 0.164] | 0.065 | +0.017 +- 0.012 |
| `glass_euler8@norm` | 8 | 8 | 64 | 0.160 [0.144, 0.177] | 0.094 | +0.030 +- 0.012 |
| `dmfm4_fm4@norm` (n=1000 dMFM arm) | 4 | 8 | 32 | 0.172 [0.152, 0.191] | 0.105 | **+0.041 +- 0.011** |
| `diag_fm4@norm` | 4 | 8 | 32 | 0.175 [0.157, 0.194] | 0.110 | +0.045 +- 0.012 |
| `dmfm4_fm8@norm` | 8 | 8 | 64 | 0.181 [0.161, 0.201] | 0.102 | +0.055 +- 0.011 |

Reading:

* **The n=1000 deficit reproduces at the published operating point on both targets**:
  +0.050 +- 0.013 and +0.041 +- 0.011 here, against +0.041 +- 0.006 and +0.049 +- 0.006 at n=1000.
  So this sweep is measuring the same effect.
* **It disappears at matched NFE once dMFM chooses its own split of that NFE.** One composed
  flow-map step with MC 32 is the same 32 network calls per value gradient as GLASS's four Euler
  steps with MC 8, and gives +0.006 +- 0.011 (CI [-0.016, +0.028], 51.0% of pairs better) at
  y* = +1 and **-0.020 +- 0.010** (CI [-0.038, -0.002], 56.0% better) at y* = -1.
* **Best-of-grid against best-of-grid, both at NFE 32**, is +0.011 +- 0.011 and -0.007 +- 0.009:
  a tie at both targets. Against GLASS's best anywhere on its grid -- two Euler steps at *half* the
  NFE -- it is +0.011 +- 0.011 and -0.028 +- 0.009, i.e. GLASS is the cheaper arm at y* = +1 and
  dMFM the better one at y* = -1. **Neither arm dominates. Do not claim a dMFM win.**
* **The two arms want the NFE spent differently, and the difference is mechanical.** Fewer posterior
  steps help both (GLASS 2 < 4 < 8 steps at both targets; dMFM 1 < 2 <= 4, 8) -- but see section 4:
  every configuration integrates to `end_time = 1.0`, so that ordering is confounded with the
  singular data time and must not be read as "accuracy hurts". But *more MC draws* help dMFM a great deal and GLASS almost not at all: dMFM one-step
  goes 0.183 -> 0.152 and 0.145 -> 0.110 from MC 8 to MC 32, while GLASS-2 moves 0.140 -> 0.141 and
  0.138 -> 0.118 from MC 8 to MC 16. The dMFM posterior's value gradient is the noisier of the two
  under a finite MC average -- which is exactly what Fig 5 / Tables 14-15 are about -- so a cheap
  posterior plus more draws is the right trade for it, and 4 steps x MC 8 spends its budget the way
  GLASS wants it spent.
* **The guide column is the sharper evidence.** At the published operating point dMFM's guide MAE is
  2.4-2.8x GLASS's (0.133 / 0.105 against 0.056 / 0.033). At one step with MC 32 it is
  0.060 / 0.034 against 0.056 / 0.033 -- parity on the objective the sampler optimises. Whatever
  the oracle column does, the guidance term itself is no longer weaker.
* One cell is missing: the *convention-compliant* one-step dMFM artifact is the diagonal student
  (`paths.dmfm_ckpt_for_steps(50, 1)`), and `diag_fm1@norm@mc=32` was not in this sweep. At MC 8
  the two one-step artifacts are indistinguishable (0.1825 / 0.1445 against 0.1830 / 0.1454), so the
  matched-NFE row should be similar, but that is an expectation and not a measurement. It is
  included in `scripts/dna/rescue/eval_retrained_student.sbatch`.

### 3a. The headline pair at n = 600

`results/dna/deficit_rescue/matched_L50_n600/`, Slurm 49438707 (`kempner_h200`; y* = +1 on
`holygpu8a03504`, y* = -1 on `holygpu8a03303`, both H200, TF32 on). Four configurations, all at
the same value-gradient NFE of 32, 600 paired ids per target. Control: the unguided oracle score is
bit-identical across all four configurations at every id (maximum spread 0.0), so the whole
difference is the guidance term.

| config | steps x MC | y* = +1 MAE [CI] | y* = -1 MAE [CI] | guide MAE +1 / -1 | guide within 0.10 |
|---|---|---|---|---|---|
| `glass_euler2@norm@mc=16` | 2 x 16 | **0.1429** [0.1340, 0.1521] | 0.1170 [0.1089, 0.1260] | 0.037 / 0.021 | 0.920 / 0.977 |
| `dmfm4_fm1@norm@mc=32` | 1 x 32 | 0.1464 [0.1386, 0.1555] | 0.1199 [0.1125, 0.1276] | 0.054 / 0.036 | 0.855 / 0.923 |
| `glass_euler4@norm` (n=1000 GLASS arm) | 4 x 8 | 0.1488 [0.1397, 0.1581] | 0.1294 [0.1209, 0.1382] | 0.052 / 0.036 | 0.855 / 0.920 |
| `diag_fm1@norm@mc=32` | 1 x 32 | 0.1536 [0.1437, 0.1636] | **0.1150** [0.1082, 0.1227] | 0.057 / 0.032 | 0.845 / 0.955 |

Paired differences (positive = the first named is worse):

| pair | y* = +1 | y* = -1 |
|---|---|---|
| `dmfm4_fm1@mc=32` - `glass_euler4` | **-0.0024 +- 0.0063** [-0.0145, +0.0097] | **-0.0096 +- 0.0058** [-0.0208, +0.0017] |
| `dmfm4_fm1@mc=32` - `glass_euler2@mc=16` | +0.0035 +- 0.0064 [-0.0086, +0.0162] | +0.0029 +- 0.0055 [-0.0083, +0.0130] |
| `diag_fm1@mc=32` - `glass_euler4` | +0.0047 +- 0.0066 [-0.0087, +0.0171] | -0.0144 +- 0.0057 [-0.0255, -0.0036] |
| `diag_fm1@mc=32` - `glass_euler2@mc=16` | +0.0107 +- 0.0070 [-0.0025, +0.0245] | -0.0019 +- 0.0053 [-0.0125, +0.0082] |

Every one of the eight differences is within about 2 SEM of zero, and the two that reach it favour
dMFM. The published operating point's +0.041 / +0.049 has no counterpart here. The
convention-compliant one-step artifact (the *diagonal* student, `paths.dmfm_ckpt_for_steps(50, 1)`)
behaves like the ESD student at this operating point -- slightly worse at y* = +1, slightly better
at y* = -1 -- which fills the cell section 3 had to leave as an expectation.

### 3b. Guidance rule and guidance strength, swept symmetrically -- a second-order knob, not the fix

`results/dna/deficit_rescue/rule_L50_n200/`, Slurm 49431419 (`kempner_h100`; y\* = +1 on
`holygpu8a15102`, y\* = -1 on `holygpu8a17104`, both H100, TF32 on). **Complete, both targets, n =
200.** This sweep is H100 while sections 3 and 3a are H200, so its *absolute* numbers are not
comparable with theirs; every comparison drawn from it is between configurations inside this run.

Every configuration here is **4 posterior steps x MC 8** -- the published operating point -- so this
isolates the rule and the strength from the operating point of section 3. All twelve are offered to
both arms:

* `@norm` -- the scale-free rule of the n=1000 runs: rescale the gradient to exactly `grad_clip` at
  every guided step.
* `@relnorm` -- the **magnitude-preserving** rule: divide by the batch median of the gradient's own
  norm and multiply by `grad_clip`, so the *median* sample's step is exactly `grad_clip` (the same
  calibration as `@norm`, hence `guidance_frac` means the same thing), then clamp any sample above
  four times that. This is the answer to "rescaling destroys magnitude information dMFM may carry
  usefully": the skew survives (mean post-rule norm 19.4 for dMFM against 17.4-17.7 for GLASS, with
  34-35% and 25-26% of samples at the tail ceiling), and it is one recipe applied identically to both.

Guided oracle MAE with bootstrap CI (`|g| raw` / `|g| post` = mean gradient norm before / after the
rule):

| config | rule | gf | y\* = +1 [CI] | y\* = -1 [CI] |
|---|---|---|---|---|
| `glass_euler4@norm@gf=8` (n=1000 GLASS arm) | scale-free | 8 | **0.139** [0.126, 0.154] | **0.134** [0.119, 0.150] |
| `glass_euler4` (published clip-down) | clip-down | 8 | 0.144 [0.127, 0.161] | 0.138 [0.123, 0.153] |
| `glass_euler4@relnorm@gf=8` | relative | 8 | 0.160 [0.141, 0.181] | 0.165 [0.146, 0.185] |
| `glass_euler4@norm@gf=16` | scale-free | 16 | 0.177 [0.158, 0.197] | 0.159 [0.142, 0.178] |
| `glass_euler4@relnorm@gf=16` | relative | 16 | 0.187 [0.167, 0.207] | 0.156 [0.138, 0.174] |
| `glass_euler4@norm@gf=32` | scale-free | 32 | 0.191 [0.170, 0.214] | 0.161 [0.144, 0.180] |
| `diag_fm4@norm@gf=8` | scale-free | 8 | 0.180 [0.159, 0.202] | 0.177 [0.159, 0.195] |
| `dmfm4_fm4@norm@gf=8` (n=1000 dMFM arm) | scale-free | 8 | 0.205 [0.182, 0.229] | 0.182 [0.162, 0.201] |
| `dmfm4_fm4@norm@gf=16` | scale-free | 16 | 0.213 [0.189, 0.239] | **0.154** [0.136, 0.171] |
| `dmfm4_fm4@norm@gf=32` | scale-free | 32 | **0.197** [0.174, 0.221] | 0.154 [0.137, 0.171] |
| `dmfm4_fm4@relnorm@gf=8` | relative | 8 | 0.201 [0.179, 0.224] | **0.152** [0.134, 0.171] |
| `dmfm4_fm4@relnorm@gf=16` | relative | 16 | 0.206 [0.183, 0.230] | 0.169 [0.150, 0.189] |

Paired differences, matched rule and matched `guidance_frac`:

| rule | gf | y\* = +1 | y\* = -1 |
|---|---|---|---|
| scale-free | 8 | +0.065 +- 0.014 | +0.048 +- 0.012 |
| scale-free | 16 | +0.036 +- 0.015 | -0.006 +- 0.013 |
| scale-free | 32 | +0.006 +- 0.016 | -0.007 +- 0.013 |
| relative | 8 | +0.040 +- 0.015 | -0.014 +- 0.014 |
| relative | 16 | +0.019 +- 0.016 | +0.013 +- 0.013 |

Reading, and it is mixed but does not change the verdict:

* **GLASS's best on the whole grid is the setting already published**, `@norm@gf=8`, at both targets
  (0.139 / 0.134), and the stock clip-down rule is statistically the same for it (0.144 / 0.138).
  Every other rule or strength makes GLASS worse. So nothing here was taken away from GLASS.
* **The knobs help dMFM at y\* = -1 and not at y\* = +1.** At y\* = -1, dMFM improves from its
  published 0.182 to 0.152-0.154 under `gf` 16/32 or `@relnorm@gf=8` -- paired against its own
  published setting, **-0.030 +- 0.013**, a real effect. At y\* = +1 its whole grid is flat within
  noise (0.197-0.213 against 0.205; best-vs-published -0.008 +- 0.016). The heavy-tail hypothesis is
  therefore **half-supported at best**: `@relnorm` is dMFM's best rule at y\* = -1 (0.152) and does
  nothing at y\* = +1 (0.201 against `@norm`'s 0.205).
* **But no rule or strength reaches GLASS at this operating point.** Best-of-grid against
  best-of-grid, both arms at 4 steps x MC 8: **+0.058 +- 0.014** at y\* = +1 and
  **+0.018 +- 0.012** at y\* = -1. Compare the operating-point change of sections 3 and 3a, which
  gives -0.002 +- 0.006 and -0.010 +- 0.006. **The rule and the strength are a second-order knob;
  the NFE split is the fix.**
* **The matched-`gf` rows that look like parity are two degraded arms meeting.** At y\* = -1,
  `gf = 32` gives -0.007 +- 0.013 -- but that is dMFM at 0.154 against GLASS dragged from 0.134 to
  0.161. Quoting it as parity would be exactly the error the brief warns about; the honest row is
  best-of-grid against best-of-grid, above.
* **This corrects `docs/dmfm_glass_parity.md` section 6.** That document's best-of-grid reading gave
  `dmfm4_fm4@norm@gf=16` as 0.144 +- 0.020 at y\* = +1 (n=32, A100, job 49351325) against GLASS's
  best 0.150, i.e. a tie. At n = 200 on one card the same configuration measures
  **0.213 [0.189, 0.239]**, and the H200 rerun of that n=32 grid had already put it at 0.185; at
  y\* = -1 it measures 0.154 against that document's 0.165. The n=32 "tie at gf = 16" was
  small-sample noise and should not be quoted. What does survive from it is that dMFM's optimum
  `guidance_frac` at 4 steps is above GLASS's -- but only at y\* = -1, and only to 0.154.

## 4. Why fewer posterior steps steer better here -- an observation, with a large caveat

**Read `docs/dna_findings_audit.md` (c) first.** While this document was being written, the
findings-audit agent of the same round **refuted** the stronger version of this reading. The claim
"a well-integrated posterior does not guide at all", which rested on `glass_euler32`, was our error
twice over: that arm integrates 32 Euler steps to the *singular* data time `t = 1`, where the
endpoint is a simplex vertex to float precision and its "gradient" is float noise (norm 1.2e-5), and
it was the one arm never given the scale-free rule. Fixed (`end_time 0.999` plus the rule), an
accurate posterior *does* guide: guided oracle MAE 0.1939 with 8/8 pairs improved against 0.1548 for
the Table 1 four-step estimator, a paired +0.039 +- 0.064 at n=8, i.e. 0.6 SEM. So
**"more posterior accuracy gives worse steering" is withdrawn**, and the `_caveat_posterior_accuracy`
field of `results/dna/c0_guidance/table1_dmfm_n1000_summary.json` and the corresponding plan item are
withdrawn with it.

What my own n=200 data does show, and all it shows, is a **step-count ordering at fixed
`end_time = 1.0` and fixed rule**, on both arms and at both targets:

| arm | 1 step | 2 steps | 4 steps | 8 steps |
|---|---|---|---|---|
| GLASS Euler, y* = +1 | -- | 0.140 | 0.146 | 0.188 |
| GLASS Euler, y* = -1 | -- | 0.138 | 0.130 | 0.160 |
| dMFM flow map, y* = +1 | 0.183 | 0.197 | 0.196 | 0.193 |
| dMFM flow map, y* = -1 | 0.145 | 0.147 | 0.172 | 0.181 |

and a negative rank correlation, in the older `sample_L50_n64` / `grad_L50_rk4ref` pair, between a
configuration's endpoint L1 distance from the reference posterior and its guided oracle MAE
(Spearman -0.96 under the stock rule; -1.0 at y* = +1 on the three configurations measured under the
scale-free rule).

**The caveat is that both of these are confounded with the same `end_time = 1.0` singularity the
findings audit identified.** Every configuration in my sweeps integrates to `t = 1`, and saturation
of the endpoint grows with the number of steps, so "8 steps is worse than 2" may be measuring
approach to the singular data time rather than posterior accuracy. The clean experiment is the step
sweep repeated at `end_time = 0.999` for both arms; `CONFIGS` already has `*_e999` entries for the
dMFM side and the findings audit has added GLASS ones, and it was not run here.

**Nothing in the verdict depends on this section.** The headline is a direct paired measurement at
matched NFE (sections 3 and 3a); it does not rest on any account of *why* a one-step map plus MC 32
beats four steps plus MC 8. What the sections above do establish mechanically, and independently of
the accuracy question, is that the two arms' value gradients differ in **MC noise**: going from
MC 8 to MC 32 moves the dMFM one-step arm by 0.031 / 0.035, and going from MC 8 to MC 16 moves
GLASS-2 by -0.001 / 0.020. That, not posterior accuracy, is what makes the NFE split the operative
knob.

## 4b. What remains open

1. **The step-count sweep at `end_time = 0.999`** (section 4). Every configuration in every sweep
   here integrates to `t = 1`, the singular data time, so the step-count ordering is confounded. The
   clean version reruns `EXP=nfe` with the `*_e999` configurations on both arms. Nothing in the
   verdict depends on it.
2. **The matched-NFE headline at n = 1000.** Section 3a took it to n = 600 with SEMs of 0.006, the
   same precision as the n=1000 reference, so the conclusion is not sample-size limited. But Table 1
   itself is n = 1000, so the row that goes into the paper should be produced by
   `scripts/dna/table1_c0_guidance_dmfm.sbatch` at the corrected operating point
   (`NFE_VALUE=1` plus `--mc 32`, which that wrapper does not yet expose as a variable) and
   collected by `scripts/dna/collect_table1_dmfm.sh`, so that the printed number comes from the
   production path rather than from this harness.
3. **The checkpoint-criterion retrain (section 2)** --
   `scripts/dna/rescue/train_dmfm4_longer.sbatch`, ~35 min on one H100 for 40,000 steps. Submitted
   as 49433477 and **cancelled by me** once it stopped being load-bearing (the shipped student
   reaches parity at one step with MC 32, so "is there a better checkpoint" cannot change the
   conclusion) and because it was competing with the two sweeps above for my own share of a
   saturated quota. The selection-metric finding in section 2 stands on the training run's own
   validation records and needs no GPU; what is untested is whether a later checkpoint *also*
   guides better.
4. **Hardware.** Everything in section 3 is one H200 (`holygpu8a03504`), both targets, TF32 on.
   The n=1000 reference numbers are H100 (jobs 49352242 / 49352244), also TF32 on. The n=200
   reproduction of the published operating point (+0.050 / +0.041 against n=1000's +0.041 / +0.049)
   is the cross-generation consistency check, and it passes; but a 0.02 effect is within the range
   hardware differences have been seen to fake in this repo, so the matched-NFE result at y* = -1
   (-0.020 +- 0.010) should be reconfirmed on a second card before it is quoted as a dMFM advantage.
   The tie at y* = +1 does not depend on that.

## 4c. Consequences for the paper, for the authors to decide

* **Table 1's dMFM row should be regenerated at the matched-NFE operating point**, i.e. the dMFM
  posterior as *one* composed flow-map step with MC 32, not four steps with MC 8. That is the same
  value-gradient NFE as the GLASS estimator Table 1 was calibrated on, it is the operating point a
  flow map exists to provide, and it takes the dMFM guided oracle MAE from 0.193 / 0.172 to about
  0.152 / 0.110 at n = 200 -- i.e. onto GLASS's level rather than 0.04-0.05 behind it. The
  in-text sentence would become "reduces oracle MAE from 1.17 to about 0.15 for y* = +1 and from
  0.83 to about 0.11 for y* = -1", pending an n = 1000 rerun.
* **Whichever operating point is printed, the paper must state the split, not just the NFE**: the
  number of composed posterior steps *and* the MC count, because they trade off at fixed cost and
  the two posteriors prefer opposite ends of that trade. `docs/REPRODUCIBILITY_PLAN.md` section 7
  item 8's two mandated conventions (step-matched student, scale-free rule) are necessary but not
  sufficient; the (steps, MC) split is a third convention that has to be recorded.
* **The convention "the student distilled for the number of composed jumps used" now points at the
  diagonal student** for the C0 demo, because the best operating point is one step, and
  `paths.dmfm_ckpt_for_steps(L, 1)` is the diagonal artifact. That happens to be the artifact every
  other dMFM result in the paper uses (Tables 13, 17, Fig 4), which makes the DNA story more
  coherent, not less -- but it means the "4-step ESD student at 4 steps" pairing chosen on
  2026-09-29 is the right fix for the *wrong* operating point. At MC 8 the two one-step artifacts
  are indistinguishable (0.183 / 0.145 vs 0.183 / 0.145); at MC 32 only the ESD student was measured.
* **Do not claim a dMFM win.** GLASS improves on its own grid too (two Euler steps at half the NFE),
  and best-against-best is a tie at y* = +1.

## 5. Files and runs

* `docs/dmfm_deficit_rescue.md` (this file)
* `src/dmfm/experiments/rescue_dmfm_glass_c0.py` -- symmetric sweep harness; adds to
  `parity_dmfm_glass_c0` the GLASS step counts `glass_euler2`/`glass_euler8`, a per-configuration
  MC budget (`@mc=`), and the relative-magnitude guidance rule (`@relnorm`). Imports the parity
  module rather than editing it, so no stored parity result can change.
* `scripts/dna/rescue/deficit_rescue.sbatch` -- the three sweeps (`EXP=nfe`, `EXP=matched`,
  `EXP=rule`); one array task per target, and a target whose `sample_scores.csv` exists is skipped
* `scripts/dna/rescue/collect_deficit_rescue.py` -- per-config summary + paired bootstrap
* `scripts/dna/rescue/train_dmfm4_longer.sbatch` -- the checkpoint-criterion retrain (not run)
* `scripts/dna/rescue/eval_retrained_student.sbatch` -- scores any 4-step student against the same
  GLASS arm on the same ids (not run)
* `results/dna/deficit_rescue/{nfe_L50_n200,matched_L50_n600,rule_L50_n200}/` --
  `sample_scores.csv` + `run_metadata.json` per target, `summary.csv` and `paired.csv` per sweep
* `results/dna/deficit_rescue/PROVENANCE.md`

Slurm: 49431420 (`nfe`, H200, both targets), 49438707 (`matched`, H200, both targets), 49431419
(`rule`, H100, y* = +1 only). Nothing outside `docs/dmfm_deficit_rescue.md`,
`src/dmfm/experiments/rescue_dmfm_glass_c0.py`, `scripts/dna/rescue/` and
`results/dna/deficit_rescue/` was created or modified; in particular
`src/dmfm/experiments/parity_dmfm_glass_c0.py` is imported, not edited, so no stored parity result
can change.
