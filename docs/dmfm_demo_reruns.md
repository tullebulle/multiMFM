# The DNA demos rerun with dMFM: Fig 7, Fig 8, Tables 4-6

Status 2026-09-30. Owner: the "dMFM demos" agent of the 2026-09-29/30 reproducibility round.
Decision being implemented: `docs/REPRODUCIBILITY_PLAN.md` S7 item 8, **"Resolution chosen by
the authors (2026-09-29): do not relabel the paper to say GLASS -- rerun every DNA demo with
dMFM, so the paper's dMFM claims are backed by dMFM runs."** Parity, which is what makes this
viable, is established in `docs/dmfm_glass_parity.md`.

Everything is L = 50, the parent-disjoint base DFM, the parent-disjoint C0 guide (guidance) and
the parent-disjoint C0 oracle (scoring) -- the Table 1 setting.

## Summary: published vs dMFM, per demo

| demo | published claim | published number | dMFM rerun | verdict |
| --- | --- | --- | --- | --- |
| **Fig 7** | the C0 distribution tilts further toward the target as strength grows; at strength 0 the histograms coincide | none printed | at `gf` 0/1/2/4/8: guide mean -0.177 / 0.095 / 0.206 / 0.280 / 0.294 (target 0.30), mean \|C0-0.30\| 0.560 / 0.280 / 0.163 / 0.118 / 0.120, oracle MAE 0.550 / 0.300 / 0.202 / 0.173 / 0.177; `gf` 0 gives 1000/1000 identical sequences | **SURVIVES** -- with the addition that the ladder is not monotone to the top (flat at `gf` 4-8, regressed to 0.144 by `gf` 32) and that matched GLASS tilts better at every strength (guide MAE 0.033 at `gf` 4, 0.048 at `gf` 8). 12 of 14 ladder points in; see "Still outstanding". |
| **Fig 8** | "the gradient clearly push the discrete variables in ambient space towards a sequence of higher reward" | none printed | `sample_idx 0`: guided C0 +0.337 vs unguided -0.361 (target 0.30), \|C0-0.30\| 0.661 -> 0.037, value -4.85 -> -0.015 | **SURVIVES** -- but dMFM is the looser arm on all four recorded samples (\|C0-0.30\| 0.037/0.218/0.053/0.196 vs GLASS 0.015/0.009/0.017/0.012) |
| **Table 4** | real-data diversity baselines | 1.0000 / 0.5219 / 0.7377 / 0.7400 etc. | bit-identical to the GLASS recompute (posterior-independent) | **SURVIVES trivially**, no change needed |
| **Table 5**, +1 set | "+1 guided preserves held-out-level global diversity"; pairwise 0.7372 = 0.9997x, nearest-gen 0.4913 = 0.9447x | 0.7372 / 0.9997x / 0.4913 / 0.9447x | 0.7355 / **0.9964x** / 0.5067 / **0.9708x** | **SURVIVES** (clustering slightly milder than published) |
| **Table 5**, -1 set | "-1 guided is measurably more clustered", pairwise 0.7010 = 0.9507x, nearest-gen 0.4175 = 0.8027x | 0.7010 / 0.9507x / 0.4175 / 0.8027x | 0.7327 / **0.9927x** / 0.5021 / **0.9621x** | **DOES NOT SURVIVE** -- and did not survive the parent-disjoint move either (GLASS control 0.9912x / 0.9367x), so this is S7 item 7, not a dMFM effect |
| **Table 6**, ordering | guided sets stay Evo 2-plausible; -1 < held-out < +1 ~ unguided | -- | -1 guided 1.3344, held-out 1.3412, +1 guided 1.3581, unguided 1.3653 | **SURVIVES** |
| **Table 6**, +1 magnitude | "+1 guided NLL 1.3529 close to held-out 1.3454, delta 0.0075, bootstrap intervals overlap" | delta +0.0075, CIs overlap | delta **+0.0169**, CIs [1.3550, 1.3613] vs [1.3362, 1.3459] -- **do not overlap** | **DOES NOT SURVIVE** as stated (GLASS control also fails the overlap test: delta +0.0197) |
| **Table 6**, -1 magnitude | "-1 guided: substantially lower NLL than held-out, 1.2350 vs 1.3454, delta -0.1104" | delta -0.1104 | delta **-0.0068** (16x smaller) | **DOES NOT SURVIVE** (GLASS control -0.0180, so again S7 item 7) |
| **any dMFM/GLASS parity claim on DNA** | -- | -- | dMFM - matched GLASS oracle MAE +0.041 +- 0.006 (y\*=+1) and +0.049 +- 0.006 (y\*=-1) at n = 1000 (`dna-core`) | **DOES NOT SURVIVE**; the n = 64 "tie" was a small-sample artifact |

## The two conventions every run here uses

Both are mandated by S7 item 8 and both must be stated in the paper:

1. **Student = the artifact distilled for the number of composed jumps used.** Four composed
   flow-map steps means `paths.dmfm_ckpt_for_steps(50, 4)` =
   `checkpoints/dna/dmfm_4step/L50/epoch=29-step=15000.ckpt`, the 4-step ESD student. Never the
   diagonal one-step student composed 4x.
2. **Scale-free gradient rule** (`--grad_normalize`): the value gradient is rescaled to exactly
   `grad_clip = 10` rather than only clipped down, applied to **both** arms wherever a GLASS
   comparison is shown.

Carry-over caveat no rerun fixes (S7 item 9): this is an operating point -- N = 8 Monte-Carlo
draws at 4 posterior steps -- not an asymptotically accurate value gradient. The C0 guidance
signal rides on the 4-step posterior's discretisation error; a 32-step GLASS posterior does not
guide at all. Report it as "N = 8 at 4 steps", never as an accurate value-gradient estimate.

## Scope and what was NOT in it

In scope here: Fig 7, Fig 8, and the generated-sequence sets behind Tables 4, 5 and 6.
Not here: Table 1 / Fig 3 (the `dna-core` agent's dMFM n = 1000 run, Slurm 49352242, whose
sample sets Tables 4-6 below consume), Fig 5 / Tables 14-15, Fig 4, Fig 6 / Table 16
(base-model marginals, unaffected by the posterior choice), GEOM and QM9.

## Jobs cancelled, and why

| job | what it was | action |
| --- | --- | --- |
| **49298146** `dna_regen_fig7_gr` (array 0-5%2, 8 h, worst case ~48 GPU-h) | Fig 7 guidance sweep through `dmfm.experiments.sample_c0_guidance`, which loads only the base DFM and the C0 guide and integrates the **GLASS** posterior velocity. No `--posterior`/`POSTERIOR`-style flag exists anywhere in that module: it is GLASS-only by construction. | **Cancelled** 2026-09-29, never started, nothing lost. Replaced by Slurm **49425774**. |
| **49298115** `dna_regen_fig8_gr` (2 h) | Fig 8 trajectory trace through `dmfm.regen.guidance_trace`, which called `api.glass_posterior_fn` unconditionally -- also GLASS-only. | **Cancelled** 2026-09-29, never started. Replaced by a CPU run of the same module with the new `--posterior dmfm` path. |

Both were also on the starved submission route (`albergo_lab` / `gpu_requeue`, 8 h walltime,
priority ~75k) rather than `kempner_albergo_lab` / `kempner_requeue` with a short walltime
(priority ~2.77M), so neither was going to start in any case. The GLASS scripts are left in
place and still runnable; they are the record of what the published figures ran.

## Fig 7's recomputed cost

The ~46 GPU-h estimate for Fig 7 came from the published launcher's "faithful estimator":
MC = 128 x `nfe_value` = 12 = **1536** posterior model evaluations per guided step per sample,
for 6 strengths at n = 1000. The dMFM conventions fix the posterior at 4 composed steps, and
MC = 8 is the paper's DNA operating point, so the value gradient costs
8 x 4 = **32** evaluations per guided step -- **48x cheaper**. Predicted ~1 GPU-h per 7-point
arm.

**Measured:** 4.9-5.6 min per (arm, strength) at n = 1000 on one H100/A100 (`kempner_requeue`),
so **0.6 GPU-h for a 7-point arm and ~1.2 GPU-h for both arms** -- against ~46 GPU-h for a
single GLASS arm at the launcher estimator. The ~46 GPU-h figure no longer applies.

Queue route used, after the round's measurement that `albergo_lab` on `gpu_requeue` /
`serial_requeue` is effectively unschedulable: `--partition=kempner_requeue
--account=kempner_albergo_lab` with a 40 min walltime for Fig 7 and 1 h for Evo 2 (priority
2.77M, tasks started within a minute); Fig 8 and the diversity analysis are CPU-only and were
run directly rather than queued.

## Fig 7 -- tilt of the cyclizability distribution vs guidance strength

### What the published version actually ran

`dirichlet-flow-matching/scripts/sample_yeast_c0_guidance_hist.py`, ported verbatim to
`dmfm.experiments.sample_c0_guidance`. That module loads **only** the base DFM and the C0 guide;
its value gradient is `nfe_value` Euler steps of the **GLASS** posterior velocity
(`glass_integrate_diff`). There is no dMFM code path and no flag that could select one, so
S 3.1's "We evaluate dMFM for DNA design" and App. F.4's "the effect of tilting the dMFM in
Figure 7" do not describe it. The published panels used an early-May-2026 base flow and the
purged Keras guide, and **their guidance strengths were never recorded** (the surviving records
are target-0.3 runs from 2026-05-02). The surviving launcher's estimator settings are MC = 128,
`nfe_traj` 128, `nfe_value` 12, batch 16, window [0.5, 0.95], `coeff_cap` 10, `grad_clip` 10,
n = 1000, target C0 = 0.30, and the guidance rule was the stock clip-down one.

### What the dMFM rerun ran

Slurm **49425774**, `scripts/dna/dmfm_demos/fig7_tilt_ladder.sbatch`,
`dmfm.experiments.sample_c0_guidance_dmfm --paired`, whose GLASS mode is numerically identical
to the published sampler (`tests/dna/test_api.py::test_paired_glass_equals_table1_sampler`).

| | value |
| --- | --- |
| student | `checkpoints/dna/dmfm_4step/L50/epoch=29-step=15000.ckpt`, resolved by `paths.dmfm_ckpt_for_steps(50, 4)` |
| posterior steps | 4 composed flow-map steps to `end_time` 1.0 |
| N (MC) | 8, frozen per-sample pool `seed + 1e6 + i` |
| gradient-scale rule | scale-free (`--grad_normalize`), rescale to exactly `grad_clip = 10`, on **both** arms |
| strength ladder | `guidance_frac` in {0, 1, 2, 4, 8, 16, 32} |
| trajectory | `nfe_sample` 64, `t_max` 0.95, guidance window [0.50, 0.95] |
| other | `coeff_cap` 10, `reward_sigma` 0.15, `reward_scale` 0.5, target C0 = 0.30 |
| seeds / n | seed 0, n = 1000 paired samples per ladder point (initial noise `seed + i`, shared by each pair) |
| control arm | the same ladder with `--posterior glass` (4 Euler steps) at the same N, the same rule and the same noise |

**Two protocol choices, and why.**

* *Estimator.* The published launcher's MC = 128 / `nfe_value` = 12 estimator is **not** used.
  Convention 1 fixes the dMFM posterior at the 4 composed steps its student was distilled for,
  and MC = 8 is the paper's DNA operating point (Table 1, Fig 3, and now Fig 5). Keeping the
  launcher's budget would have meant a student used at a jump it was not distilled for, plus a
  48x cost. The consequence is that the GLASS column here is the Table 1 GLASS estimator, not
  the published Fig 7 one.
* *Trajectory resolution.* `nfe_sample` 64 rather than the launcher's 128, so that Fig 7 shares
  one protocol with Table 1 / Fig 3 / the parity study. Since the published strengths are
  unrecorded, exact reproduction of the published panels is impossible either way; protocol
  consistency across the dMFM campaign is the more useful property.

**Ladder choice, fixed before any output was inspected.** `guidance_frac` 0 is Fig 7(a): the
coefficient is exactly zero, so the guided branch *is* the unguided branch (verified -- all 1000
guided sequences are character-identical to their unguided partners and max |delta| = 0). The
rest is a geometric ladder in powers of two chosen to bracket both arms' optima as measured
under the scale-free rule in `docs/dmfm_glass_parity.md` section 6 (GLASS 4-8, dMFM 16 at
y* = +-1), with one point above dMFM's. The **whole ladder is reported below**; the three figure
panels are, fixed in advance, `guidance_frac` 0 / 8 / 32 for (a) none / (b) moderate /
(c) strong.

### Result: the whole ladder

Target C0 = 0.30, n = 1000 paired samples per row. `guide` = the A_train guide regressor,
i.e. the quantity the published histograms plot and the quantity guidance optimises;
`oracle` = the independent B_train regressor, reverse-complement averaged (the Table 1
protocol). `oracle paired improvement` is the per-sample
`|oracle_unguided - 0.30| - |oracle_guided - 0.30|` with a 2,000-resample bootstrap CI.

Unguided reference (identical in every row, same x0): guide mean -0.1767, guide MAE 0.5599, oracle MAE 0.5502

| arm | gf | n | guide mean | guide MAE | guide frac within 0.10 | oracle mean | oracle MAE | oracle paired improvement | CI low | CI high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dmfm | 0 | 1000 | -0.1767 | 0.5599 | 0.0800 | -0.1715 | 0.5502 | 0.0000 | 0.0000 | 0.0000 |
| dmfm | 1 | 1000 | 0.0955 | 0.2799 | 0.2730 | 0.0802 | 0.2997 | 0.2505 | 0.2319 | 0.2691 |
| dmfm | 2 | 1000 | 0.2056 | 0.1630 | 0.4570 | 0.1816 | 0.2018 | 0.3485 | 0.3279 | 0.3690 |
| dmfm | 4 | 1000 | 0.2800 | 0.1183 | 0.5720 | 0.2467 | 0.1734 | 0.3769 | 0.3563 | 0.3985 |
| dmfm | 8 | 1000 | 0.2944 | 0.1204 | 0.5720 | 0.2642 | 0.1768 | 0.3735 | 0.3523 | 0.3953 |
| dmfm | 32 | 1000 | 0.2809 | 0.1436 | 0.4640 | 0.2572 | 0.1799 | 0.3703 | 0.3491 | 0.3919 |
| glass | 0 | 1000 | -0.1767 | 0.5599 | 0.0800 | -0.1715 | 0.5502 | 0.0000 | 0.0000 | 0.0000 |
| glass | 1 | 1000 | 0.2778 | 0.0783 | 0.7580 | 0.2555 | 0.1448 | 0.4054 | 0.3853 | 0.4261 |
| glass | 4 | 1000 | 0.3003 | 0.0326 | 0.9560 | 0.2857 | 0.1287 | 0.4215 | 0.4010 | 0.4425 |
| glass | 8 | 1000 | 0.2964 | 0.0477 | 0.8830 | 0.2800 | 0.1261 | 0.4242 | 0.4043 | 0.4444 |
| glass | 16 | 1000 | 0.2895 | 0.0927 | 0.6640 | 0.2717 | 0.1484 | 0.4019 | 0.3812 | 0.4236 |
| glass | 32 | 1000 | 0.2869 | 0.1151 | 0.5690 | 0.2682 | 0.1547 | 0.3955 | 0.3737 | 0.4171 |

**Incomplete as of this writing:** dmfm gf=16, glass gf=2.
`glass gf=2` (array task 9) **failed** on its first attempt: it landed on an
`NVIDIA RTX PRO 6000 Blackwell Server Edition` node of the `kempner_requeue` pool, whose
`sm_120` compute capability the pinned torch 2.5.1+cu124 does not support
(`RuntimeError: CUDA error: no kernel image is available for execution on the device`,
40 s in). That is a heterogeneous-pool hazard for the whole round, not a bug in this code:
any GPU job on `kempner_requeue` needs `--constraint="a100|h100|h200"`. It was resubmitted
with that constraint as Slurm **49436287**. `dmfm gf=16` is array task 5, still
`PENDING (Priority)`. Neither is load-bearing for the verdicts below; rebuild the table with
`python scripts/dna/dmfm_demos/collect.py --what fig7` then
`python scripts/dna/dmfm_demos/md_tables.py fig7` once they land.

**Verdict: the Fig 7 claim SURVIVES with dMFM.** The figure's claim is that the cyclizability
distribution tilts further toward the target as the strength of tilt grows, and that at strength
zero the guided and unguided histograms coincide. Both hold:

* At `guidance_frac = 0` the two histograms are not merely similar but identical -- all 1000
  guided sequences are character-identical to their noise-matched unguided partners, so Fig 7(a)
  is exact rather than approximate.
* From 0 to 8 the dMFM-guided distribution moves onto the target monotonically in the mean:
  guide mean -0.177 -> 0.095 -> 0.206 -> 0.280 -> 0.294 against the target 0.30; mean
  |C0 - 0.30| 0.560 -> 0.280 -> 0.163 -> 0.118 -> 0.120; the fraction within 0.10 of the target
  8.0% -> 27.3% -> 45.7% -> 57.2% -> 57.2%. The independent oracle agrees (MAE 0.550 -> 0.300 ->
  0.202 -> 0.173 -> 0.177), so this is not the guide scoring its own output. The pre-declared
  panel (b) strength, `gf = 8`, therefore sits at the plateau of the ladder rather than past it.
* The unguided column is byte-identical in every row (same `x0`, no gradient), so the whole
  movement is the guidance term. As a cross-arm check, the `gf = 0` rows of the dMFM and GLASS
  arms agree to every printed digit (guide MAE 0.559881, oracle MAE 0.550239), which they must:
  with a zero coefficient the posterior sampler cannot enter the trajectory at all.

**Honest additions the caption does not currently make.**

1. *The ladder has an optimum, it is not monotone to the top.* dMFM is flat between
   `guidance_frac` 4 and 8 (guide MAE 0.118 and 0.120, both with 57.2% within 0.10 of the target)
   and by 32 it has regressed to 0.144; `gf = 16` is the one dMFM point still pending, so the
   optimum is "4-8, and 32 is past it". GLASS's optimum is also 4-8 (0.033 / 0.048), falling to
   0.115 by 32. So "strong guidance" past the optimum tilts the distribution *past* the target and
   re-widens it. The pre-declared panels 0 / 8 / 32 turn out to be a fair illustration --
   (b) `gf = 8` is at the plateau and (a) `gf = 0` is exact -- provided the caption does not
   present (c) `gf = 32` as simply "more of the same": it is over-driven, and it is the only panel
   where the guided distribution is worse than at (b). **I am reporting the whole ladder rather
   than choosing panels to flatter the claim**, and the panel rule was fixed before the numbers
   were seen; which three the paper prints is an authors' decision.
2. *At this target GLASS tilts considerably better than dMFM, at every matched strength.* Guide
   MAE, GLASS against dMFM: `gf` 1 0.078 vs 0.280, `gf` 4 0.033 vs 0.118, `gf` 8 0.048 vs 0.120,
   `gf` 32 0.115 vs 0.144; oracle MAE at `gf` 8, 0.126 vs 0.177. GLASS also reaches 75.8% of
   samples within 0.10 of the target at `gf` 1, which dMFM does not reach anywhere on the ladder
   (best 57.2%). The gap is the same sign and roughly the same size as `dna-core`'s
   n = 1000 Table 1 result (+0.041 / +0.049 oracle MAE, 6-9 SEM). So Fig 7 supports "dMFM steering
   tilts the distribution, controllably, with strength" and **not** "dMFM tilts as well as GLASS".
3. *dMFM's optimum is lower here than at y\* = +-1.* `docs/dmfm_glass_parity.md` section 6 found
   dMFM best at `guidance_frac = 16` for y\* = +-1; at target 0.30 it is 4-8. That is expected --
   0.30 is close to the base model's own distribution (unguided mean -0.177), so less drive is
   needed -- but it means `guidance_frac` is not transferable across targets and has to be stated
   per figure.

## Fig 8 -- one guided vs one unguided sample

### What the published version actually ran

The producer of `figures/steering.png` is **lost** (it is in none of the source repos, the
2026-09-25 reconstruction, the holylabs backup or the upstream `tullebulle/DNA-MFM` clone), so
what exists is a reimplementation of its three panels: `dmfm.regen.guidance_trace`. That module
called `api.glass_posterior_fn(base, n_steps=nfe_value)` unconditionally -- GLASS on the base
DFM -- at the recovered launcher settings MC = 128, `nfe_value` 12, `nfe_traj` 128, guidance
window [0.0, 0.95], `guidance_frac` 1.0, stock clip-down rule. Its queued run (49298115) was
cancelled.

### What the dMFM rerun ran

`scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch`, i.e. the same module through the new
`--posterior dmfm --grad_normalize` path, at the campaign operating point:

| | value |
| --- | --- |
| student / posterior steps | 4-step ESD student, 4 composed flow-map steps (`paths.dmfm_ckpt_for_steps(50, 4)`) |
| N (MC) | 8 |
| gradient-scale rule | scale-free (`--grad_normalize`), both arms |
| strength | `guidance_frac` 8 (the Table 1 / Fig 3 value), `coeff_cap` 10, `grad_clip` 10 |
| trajectory | `nfe_traj` 64, `t_max` 0.95, guidance window [0.50, 0.95] |
| other | target C0 = 0.30, `reward_sigma` 0.15, `reward_scale` 0.5 |
| seeds / n | seed 0, `sample_ids 0 1 2 3` |
| control arm | the same settings with `--posterior glass` (4 Euler steps) |
| device | **CPU**, deterministic; the noise and the MC pool are drawn by seeded CPU generators, so CPU and GPU runs share identical randomness, and no published number depends on this trajectory. ~3 min per arm. |

**Sample-selection rule, fixed before any output was looked at:** all four of
`sample_ids 0 1 2 3` are recorded in `trace.csv`, and the figure shows **`sample_idx 0`** -- the
lowest recorded id at the campaign seed. Nothing is picked on appearance.

## Tables 4, 5 and 6 -- the generated-sequence sets

### What the published version actually ran

These three tables score *generated sequences*, so they inherit whatever produced those
sequences. Two separate problems, one already recorded and one from S7 item 8:

* **S7 item 7:** the published Tables 4-6 and App. B.1 were computed on the purged `split65k`
  models, not the parent-disjoint models App. B.1 describes.
* **S7 item 8:** the sets themselves were produced by **GLASS** guidance on the base DFM. That is
  true both of the published `split65k` sets and of the parent-disjoint recompute in
  `results/dna/diversity/regen/`, which was computed on `results/dna/c0_guidance/n1000/` --
  the GLASS n = 1000 sets. So as they stand, "Diversity statistics for generated C0-steered
  yeast **dMFM** samples" (Table 5's caption) describes GLASS output.

The queued GLASS Evo 2 job for the parent-disjoint recompute (**49298971**, `MODE=c0`, scoring
`results/dna/c0_guidance/n1000/`) was on the starved route and has not run; it is left queued and
is not needed, because the matched-GLASS control below is a better comparison (same guidance rule
as the dMFM arm) and was run on the fast route.

### What the dMFM rerun ran

The generated sets are **not** regenerated here: they are the Table 1 / Fig 3 dMFM n = 1000 sets
produced by the `dna-core` agent (Slurm **49352242**,
`outputs/dna/table1_dmfm4_norm_n1000/target_{1p0,m1p0}/merged/`), so Tables 4-6 and Table 1 agree
by construction. Those use exactly the two conventions: 4-step ESD student at `nfe_value 4`,
`--grad_normalize`, MC = 8, `nfe_sample 64`, window [0.50, 0.95], `guidance_frac 8`,
`coeff_cap 10`, `grad_clip 10`, `reward_sigma 0.15`, `reward_scale 0.5`, seed 0, n = 1000 paired
samples per target, initial noise `seed + i` shared by each pair. The matched GLASS control is
Slurm **49352244** (`outputs/dna/table1_glass_norm_n1000/...`), identical except `--posterior
glass`. Both arms' unguided halves are bit-identical to the stored GLASS n = 1000 run, so every
difference below is attributable to the guidance term alone.

* **Tables 4-5:** `dmfm.experiments.analyze_c0_diversity` via
  `scripts/dna/dmfm_demos/tables4_5_diversity_dmfm.sh`, CPU, all published analyser defaults
  (`--baseline_n 1000 --seed 0 --max_pairwise_n 3000 --k_nearest_train 5 --pair_chunk 128
  --size_matched_draws 1000`), splits `a_train` / `a_val` / `test`.
* **Table 6:** `dmfm.experiments.score_yeast_c0_evo2` via
  `scripts/dna/dmfm_demos/table6_evo2_dmfm.sbatch` in the separate `evo2` env, Evo 2 7B base,
  `--targets -1 1 --baseline_n 1000 --batch_size 1 --bootstrap 2000`, RC-averaged, seed 0 --
  identical to `scripts/dna/regen/score_evo2.sbatch MODE=c0` in every argument except
  `--run_glob` and `--out_dir`.

**Table 4 is posterior-independent** (real-data baselines only). Verified: every cell of the dMFM
run's `table6_baselines.csv` is bit-identical to
`results/dna/diversity/regen/table4_real_baselines.csv`. It is recomputed only so Table 5's
ratios divide by the same denominators. **Table 4 therefore survives trivially and needs no
change.**

### Result, and the verdict

Target C0 = 0.30. `|C0-0.30|` is the distance of the final hard sequence's guide C0 from the
target; `V` is the finite-MC value (the log-mean-exp reward, maximum 0).

| arm | sample_idx | final C0 guided | final C0 unguided | \|C0-0.30\| guided | \|C0-0.30\| unguided | mean \|g\| in window | frac \|g\| > clip | V final guided | V final unguided |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dmfm | 0 | +0.3371 | -0.3607 | 0.0371 | 0.6607 | 3.32 | 0.133 | -0.015 | -4.850 |
| dmfm | 1 | +0.5183 | -0.3488 | 0.2183 | 0.6488 | 4.11 | 0.200 | -0.530 | -4.677 |
| dmfm | 2 | +0.3529 | +0.5283 | 0.0529 | 0.2283 | 6.08 | 0.133 | -0.031 | -0.579 |
| dmfm | 3 | +0.1042 | -0.1662 | 0.1958 | 0.4662 | 10.89 | 0.100 | -0.426 | -2.415 |
| glass | 0 | +0.3151 | -0.3607 | 0.0151 | 0.6607 | 3.71 | 0.100 | -0.544 | -4.850 |
| glass | 1 | +0.2908 | -0.3488 | 0.0092 | 0.6488 | 0.76 | 0.000 | -0.001 | -4.677 |
| glass | 2 | +0.3174 | +0.5283 | 0.0174 | 0.2283 | 1.01 | 0.000 | -0.003 | -0.579 |
| glass | 3 | +0.3122 | -0.1662 | 0.0122 | 0.4662 | 1.62 | 0.000 | -0.000 | -2.415 |

**Verdict: the Fig 8 claim SURVIVES with dMFM.** The caption's claim is that "the gradient
clearly push the discrete variables in ambient space towards a sequence of higher reward". On the
figure's sample (`sample_idx 0`, the pre-declared choice) the dMFM-guided trajectory lands at
C0 = +0.337 against the target 0.30, while its noise-matched unguided partner lands at -0.361:
`|C0 - 0.30|` falls from 0.661 to 0.037, and the value rises from -4.85 to -0.015. All three
published panel behaviours reproduce: the gradient norm is O(1-10) through the guided window, the
guided value climbs toward its maximum of 0 while the unguided one falls away, and the guided
predicted-endpoint score converges onto the target while the unguided one drifts to the base
model's typical score.

**Honest caveats.** (i) dMFM is the looser of the two arms across all four recorded samples:
`|C0 - 0.30|` is 0.037 / 0.218 / 0.053 / 0.196 for dMFM against 0.015 / 0.009 / 0.017 / 0.012 for
matched GLASS, so on this figure's evidence GLASS is more reliably on-target -- consistent with
Table 1's n = 1000 paired gap (below). A single-sample demonstration is not evidence about which
posterior steers better, and Fig 8 should not be read as such. (ii) The published trajectory
cannot be reproduced at all: its producer is lost, and the base model and guide it used are
purged. This is a reimplementation of the panels, with dMFM, not a reproduction of the figure.

### Table 5 -- result, and the verdict

Normalised Hamming fraction, L = 50. "held-out" is the 1,000-window `test_sample` baseline of
Table 4 (pairwise 0.7381, nearest-generated 0.5219 on the parent-disjoint splits; the paper's
split65k denominators were 0.7374 / 0.5201). Oracle MAE is `|oracle C0 - y*|`, mean over 1,000.

| source | set | guide score | nearest gen | pairwise gen | pairwise / held-out | nearest gen / held-out | oracle MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| published (split65k, GLASS) | -1 guided | -0.9893 | 0.4175 | 0.7010 | 0.9507 | 0.8027 |  |
| published (split65k, GLASS) | +1 guided | 1.0010 | 0.4913 | 0.7372 | 0.9997 | 0.9447 |  |
| published (split65k, GLASS) | unguided | -0.1692 | 0.5210 | 0.7364 | 0.9987 | 1.0017 |  |
| GLASS parent-disjoint, stock rule | -1 guided | -0.9932 | 0.4960 | 0.7287 | 0.9872 | 0.9504 | 0.1192 |
| GLASS parent-disjoint, stock rule | +1 guided | 0.9986 | 0.5091 | 0.7348 | 0.9956 | 0.9755 | 0.1483 |
| GLASS parent-disjoint, stock rule | unguided | -0.1767 | 0.5221 | 0.7374 | 0.9991 | 1.0003 | 0.8326 |
| GLASS parent-disjoint, scale-free | -1 guided | -0.9928 | 0.4889 | 0.7316 | 0.9912 | 0.9367 | 0.1235 |
| GLASS parent-disjoint, scale-free | +1 guided | 0.9990 | 0.5087 | 0.7377 | 0.9994 | 0.9747 | 0.1527 |
| GLASS parent-disjoint, scale-free | unguided | -0.1767 | 0.5221 | 0.7374 | 0.9991 | 1.0003 | 0.8326 |
| dMFM parent-disjoint, scale-free | -1 guided | -0.9390 | 0.5021 | 0.7327 | 0.9927 | 0.9621 | 0.1723 |
| dMFM parent-disjoint, scale-free | +1 guided | 0.9817 | 0.5067 | 0.7355 | 0.9964 | 0.9708 | 0.1933 |
| dMFM parent-disjoint, scale-free | unguided | -0.1767 | 0.5221 | 0.7374 | 0.9991 | 1.0003 | 0.8326 |

**Verdict: Table 5's positive claim SURVIVES with dMFM; its negative claim does not, and did not
already survive the parent-disjoint move.**

* *Survives.* Every dMFM generated set is exact-unique and exact-novel against both `a_train` and
  the held-out test parents (1.0000 in all four columns, as published). The `y* = +1` guided set
  preserves held-out-level global diversity: pairwise 0.7355 = **0.9964x** held-out against the
  published 0.7372 = 0.9997x, with nearest-generated 0.5067 = 0.9708x versus the published
  0.4913 = 0.9447x -- the same reading, "global pairwise diversity remains effectively
  held-out-level, with mild local clustering", with the clustering slightly *milder*.
* *Does not survive.* The published claim that the `y* = -1` guided set is "measurably more
  clustered, with pairwise diversity 0.7010 or 0.9507x the held-out baseline, and
  nearest-generated distance 0.4175 or 0.8027x" is **not reproduced**: dMFM gives 0.7327 =
  0.9927x and 0.5021 = 0.9621x, i.e. essentially held-out-level. This is **not** a dMFM effect --
  the parent-disjoint GLASS recompute already gave 0.9872x / 0.9504x (S7 item 7) and the matched
  scale-free GLASS control gives 0.9912x / 0.9367x. The `split65k` -1 clustering is an artifact of
  the purged models, and the App. B.1 prose built on it (0.9507x, 0.8027x) has to be rewritten
  whichever posterior the paper reports.
* *Cost of switching posterior.* dMFM attains the targets slightly less well than matched GLASS:
  guide score mean -0.9390 / +0.9817 against GLASS's -0.9928 / +0.9990, and oracle MAE
  0.1723 / 0.1933 against 0.1235 / 0.1527. This is the same ~0.04-0.05 gap `dna-core` measures on
  Table 1 at n = 1000 (paired dMFM - GLASS +0.041 +- 0.006 at +1, +0.049 +- 0.006 at -1, i.e.
  6-9 SEM). **dMFM is not at parity with matched GLASS at n = 1000**; the earlier n = 64 "tie" in
  `docs/dmfm_glass_parity.md` was a small-sample version of the same effect. What survives is that
  both arms move oracle MAE from 1.17 / 0.83 (unguided) to 0.12-0.19.

### Table 6 -- result, and the verdict

Evo 2 7B base, forward + reverse-complement mean log-probability, NLL with a 2,000-resample
bootstrap CI over sequence-level NLL. `delta held-out` is against each run's own
`split_test_sample` reference row.

| source | set | NLL | CI | delta held-out |
| --- | --- | --- | --- | --- |
| published (split65k, GLASS) | -1 guided | 1.2350 | [1.2252, 1.2445] | -0.1104 |
| published (split65k, GLASS) | -1 unguided | 1.3613 | [1.3584, 1.3641] | +0.0159 |
| published (split65k, GLASS) | +1 guided | 1.3529 | [1.3497, 1.3561] | +0.0075 |
| published (split65k, GLASS) | +1 unguided | 1.3619 | [1.3591, 1.3646] | +0.0165 |
| published (split65k, GLASS) | held-out test | 1.3454 | [1.3406, 1.3501] | +0.0000 |
| GLASS parent-disjoint, scale-free | -1 guided | 1.3225 | [1.3174, 1.3275] | -0.0180 |
| GLASS parent-disjoint, scale-free | -1 unguided | 1.3645 | [1.3614, 1.3675] | +0.0240 |
| GLASS parent-disjoint, scale-free | +1 guided | 1.3602 | [1.3571, 1.3634] | +0.0197 |
| GLASS parent-disjoint, scale-free | +1 unguided | 1.3645 | [1.3614, 1.3675] | +0.0240 |
| GLASS parent-disjoint, scale-free | held-out test | 1.3405 | [1.3355, 1.3452] | +0.0000 |
| dMFM parent-disjoint, scale-free | -1 guided | 1.3344 | [1.3303, 1.3381] | -0.0068 |
| dMFM parent-disjoint, scale-free | -1 unguided | 1.3653 | [1.3622, 1.3683] | +0.0241 |
| dMFM parent-disjoint, scale-free | +1 guided | 1.3581 | [1.3550, 1.3613] | +0.0169 |
| dMFM parent-disjoint, scale-free | +1 unguided | 1.3653 | [1.3622, 1.3683] | +0.0241 |
| dMFM parent-disjoint, scale-free | held-out test | 1.3412 | [1.3362, 1.3459] | +0.0000 |

**Verdict: Table 6's *direction* survives with dMFM; both of its quantitative claims do not, and
neither survived the parent-disjoint move either.**

* *Survives.* The ordering the table is about is reproduced: the `y* = -1` guided set is the most
  Evo 2-plausible of the generated sets (NLL 1.3344), the `y* = +1` guided set sits just above
  held-out (1.3581 vs 1.3412), and both guided sets are more plausible than their own unguided
  partners (1.3653). Guidance does not push sequences off the sequence model's manifold.
* *Does not survive (1).* "For `target_c0 = -1`, Evo2 assigns substantially lower NLL than
  held-out real data, 1.2350 versus 1.3454 with delta = -0.1104". With dMFM the gap is
  **delta = -0.0068**, 16x smaller, and the CI only just clears held-out's. The matched GLASS
  control gives -0.0180, so this is again the `split65k`-vs-parent-disjoint effect (S7 item 7),
  not a posterior effect. "Substantially lower" cannot be said of either arm.
* *Does not survive (2).* "the guided samples have mean NLL 1.3529, close to the held-out
  `split_test_sample` value of 1.3454 with delta = 0.0075, and the bootstrap intervals overlap."
  With dMFM: 1.3581 against 1.3412, delta = +0.0169, and the intervals [1.3550, 1.3613] and
  [1.3362, 1.3459] **do not overlap**. The matched GLASS control also fails the overlap test
  (delta = +0.0197). The honest statement is "within 0.02 nats, about 1.3% of the NLL, but
  statistically resolvable at n = 1000", not "the intervals overlap".
* *Precision note.* The two runs' own held-out reference rows differ by 0.0007 nats (1.3405 vs
  1.3412) although both score the same 1,000 seed-0 windows, so differences below ~0.001 nats
  between the arms are not meaningful.

## What does NOT survive

1. **Table 5 / App. B.1: the negative-target clustering claim.** "the `target_c0 = -1` guided set
   is exact-novel but measurably more clustered, with pairwise diversity 0.7010 or 0.9507x the
   held-out baseline, and nearest-generated distance 0.4175 or 0.8027x the held-out baseline."
   With dMFM it is 0.7327 = 0.9927x and 0.5021 = 0.9621x -- effectively held-out-level. **Not a
   dMFM effect:** the parent-disjoint GLASS recompute already gave 0.9872x / 0.9504x and the
   matched scale-free GLASS control gives 0.9912x / 0.9367x, so this is the S7-item-7 finding
   (purged `split65k` models), confirmed again. The App. B.1 sentence must be rewritten whichever
   posterior the paper reports.
2. **Any claim of dMFM/GLASS parity on DNA.** At n = 1000 paired, `dna-core` measures dMFM minus
   matched GLASS = +0.041 +- 0.006 oracle MAE at y* = +1 and +0.049 +- 0.006 at y* = -1 (6-9 SEM).
   The n = 64 "tie" in `docs/dmfm_glass_parity.md` section 2 was a small-sample version of the
   same effect. Everything in this document is consistent with that: on Tables 5/6's sets dMFM's
   oracle MAE is 0.172 / 0.193 against GLASS's 0.124 / 0.153, and on Fig 8's four samples dMFM is
   the looser arm. **What survives is "dMFM steering moves the distribution onto the target", not
   "dMFM steers as well as GLASS".**
3. **Fig 7's published panels cannot be reproduced, only re-derived.** Their guidance strengths
   were never recorded, their base flow was an early-May-2026 model and their guide is purged.
   The ladder here is a defensible reconstruction of the *sequence* "no / moderate / strong", not
   of the published settings. It also uses a different estimator (MC = 8 x 4 steps, the paper's
   DNA operating point, rather than the launcher's MC = 128 x 12 steps) and a different
   trajectory resolution (`nfe_sample` 64 rather than 128), both forced by convention 1 and by
   keeping one protocol across the campaign.
4. **The operating-point caveat, unchanged by any rerun** (S7 item 9). N = 8 at 4 posterior steps
   is not an asymptotically accurate value gradient: the well-integrated value gradient has norm
   ~0.15 where these estimators have 1-40, and a 32-step GLASS posterior does not guide at all.
   Fig 7's strength sweep is therefore a sweep of (discretisation error x `guidance_frac`), and
   must be reported as such rather than as a sweep of a converged guidance term.
5. **The implemented reward is not eq (17).** It is `-11.111 (f - y*)^2` (`reward_scale 0.5`,
   `reward_sigma 0.15`), and inside `log mean_j exp(r)` that constant is an inverse temperature,
   so it is part of the configuration and cannot be absorbed into `guidance_frac`. It applies to
   every run here, including Fig 7's ladder. (Recorded by `dna-core` as `_reward`.)

## Still outstanding

* **2 of the 14 Fig 7 ladder points**: `dmfm gf = 16` (array task 5 of Slurm 49425774, still
  `PENDING (Priority)`) and `glass gf = 2` (task 9, resubmitted as Slurm **49436287**). Neither is
  load-bearing: dMFM's optimum is already pinned to 4-8 by the landed points and GLASS's ladder
  already spans 0, 1, 4, 8, 16, 32. Rebuild with
  `python scripts/dna/dmfm_demos/collect.py --what fig7` then
  `python scripts/dna/dmfm_demos/md_tables.py fig7`; each task skips itself if its `summary.json`
  exists, so no further resubmission is needed.
* **Heterogeneous-GPU hazard for the whole round.** `glass gf = 2` failed 40 s in with
  `RuntimeError: CUDA error: no kernel image is available for execution on the device` because it
  landed on an `NVIDIA RTX PRO 6000 Blackwell Server Edition` node of `kempner_requeue`: compute
  capability `sm_120`, which the pinned torch 2.5.1+cu124 does not build for (it supports
  sm_50-sm_90). This will hit **any** GPU job submitted to `kempner_requeue` without a card
  constraint, silently costing a task each time it lands there. Every GPU submission in this round
  should carry `--constraint="a100|h100|h200"`; the resubmission does. Worth adding to
  `scripts/dna/_header.sh`'s sibling sbatch templates, which I do not own.
* The `ArrayTaskThrottle` of 49425774 was raised from 4 to 6 mid-run to clear a stuck
  `JobArrayTaskLimit` (5 tasks pending while 0 were running); after that they became schedulable.
  A hedge submission of the stragglers to `gpu_test` was refused with
  `sbatch: error: QOSMaxSubmitJobPerUserLimit` -- that is the **`gpu_test` QOS's own**
  `MaxSubmitJobsPerUser=2` / `MaxJobsPerUser=2`, not an account cap (`sacctmgr` shows the `normal`
  QOS unlimited and the `albergo_lab` / `kempner_albergo_lab` / `kozinsky_lab` association limits
  at 10100). Submission to the kempner partitions works normally.
* **Cancel-and-resubmit of three other agents' starved jobs, which I was asked to take over but
  could not do**: `49298891` / `49297229` (`dna_sf_dmfm_gr` / `dna_sf_dmfm_cpu`, Fig 5 /
  Tables 14-15) and `49294651` (`dna_regen_fig6_cpu`). Both halves were refused by this session's
  permission layer -- the `scancel`, and then the fast-route `sbatch` resubmission as well. They
  are left for whoever has the permission to run them. Their *configuration* is fine -- I checked: `dmfm.scalefree.run_shard`
  defaults to `--student dmfm --dmfm_steps 1`, which is `paths.dmfm_ckpt_for_steps(L, 1)`, so
  convention 1 holds, and convention 2 (the guidance-step scale rule) does not apply because
  Fig 5 / Tables 14-15 measure value-gradient accuracy without any guided sampling step. What they
  need is only resubmission on the fast route: `sbatch --time=40:00
  scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch` for the non-L400 tasks (1,2,3,5,6,7,9,10,11)
  and `--time=1:30:00 --array=0,4,8` for L = 400, with the script's own
  `kempner_requeue`/`kempner_albergo_lab` defaults rather than the `gpu_requeue`/`albergo_lab`
  overrides they were submitted with; the CPU array is a fallback for a saturated
  `kempner_requeue` and is redundant if the GPU route runs. `49294651` has the GPU twin
  `scripts/dna/regen/fig6_table16_marginals.sbatch` (2 tasks, already on the fast route's
  defaults), and `outputs/dna/regen/fig6*` is empty, so nothing is lost by moving it.
  One caveat for whoever resubmits the Fig 5 GPU array: the starved `49298891` writes to the same
  `outputs/dna/scale_free_mc_dmfm` tree, and `--skip_existing` only guards at shard granularity,
  so either cancel it first or give the new array its own `OUTPUT_ROOT`.
* **`49298971`** (`dna_regen_evo2_gr`, GLASS Table 6 on the stock-rule parent-disjoint sets) is
  left queued. It is superseded by `table6/glass_norm/`, which is a better control.

## Notebook cells that will need updating

Not touched by me, per the ownership split. Both notebooks currently read the GLASS regeneration
tree `G = results/dna/regen`; the dMFM tree is `results/dna/dmfm_demos` and deliberately carries
the *same* file and column names where the old cells read them, so most of these are path
changes. `dmfm.regen.plots.fig7`, `fig7_panels` and `fig8` all work unchanged on the new files
(verified).

**`notebooks/03_dna_cyclizability.ipynb`**

| cell | change |
| --- | --- |
| 17 (md) | "Figs 7-8 -- moved to `notebooks/09_dna_regenerated.ipynb`" -- that notebook does not exist in the tree (the 09 slot is `09_evo2_plausibility.ipynb`). Retarget the pointer, and say the figures are now dMFM. |
| 41 (md) | Rewrite: the published figure is a GLASS experiment; this is the dMFM rerun. State student (4-step ESD), composed steps (4), N (8), the scale-free rule, and that the estimator differs from the published launcher's. |
| 42 (code) | `fig7_index = G / "fig7" / "index.csv"` -> `RESULTS / "dna" / "dmfm_demos" / "fig7" / "index.csv"`. Same columns; add `arm` to the displayed columns and filter to `arm == "dmfm"` (the file holds both arms). |
| 43 (code) | Panel choice currently "smallest / middle / largest of the sweep". Replace with the pre-declared panels `guidance_frac` 0 / 8 / 32, and resolve `r.scores_csv` against `RESULTS / "dna" / "dmfm_demos"` (not `.../regen`). |
| 44 (md) | Keep the "no numbers to compare" point; replace the strength-choice sentence with the declared ladder and panel rule; add the oracle column and the note that gf > 8 does not help at target 0.30. |
| 45 (code) | `RERUN` branch: add `--posterior dmfm` equivalent, i.e. use `sample_c0_guidance_dmfm --paired --posterior dmfm --grad_normalize --nfe_value 4` rather than `api.c0_guided_pairs` (which is GLASS). Otherwise the RERUN path contradicts the stored results. |
| 46 (md) | Note the module is still a new implementation, and that it now has a `--posterior dmfm` path. |
| 47, 48 (code) | `fig8_trace = G / "fig8" / "trace.csv"` -> `RESULTS / "dna" / "dmfm_demos" / "fig8" / "dmfm" / "trace.csv"`; `sample_idx=0` explicitly (the declared rule) instead of `tr.sample_idx.iloc[0]`. Optionally show the `glass` arm beside it. |
| 49 (md) | Keep; add the four-sample `\|C0-0.30\|` comparison and the caveat that Fig 8 is not evidence about which posterior steers better. |
| 51-59 (md+code) | Tables 4-5: `DIV = RESULTS / "dna" / "diversity" / "regen"` currently gives the GLASS recompute. Add a dMFM column from `RESULTS / "dna" / "dmfm_demos" / "diversity" / "dmfm4_norm"` (files `table6_baselines.csv` = Table 4, `table7_generated.csv` = Table 5, `table7_additions.csv`) and the matched control `.../glass_norm`. Cell 59's "reading the comparison" markdown needs the negative-target clustering conclusion updated. |
| 38 (code) | The "Tables 4-5 RERUN" cell should call `scripts/dna/dmfm_demos/tables4_5_diversity_dmfm.sh` (dMFM sets) rather than the GLASS sets. |

**`notebooks/09_evo2_plausibility.ipynb`**

| cell | change |
| --- | --- |
| 14 (md) | Add: the scorer is unchanged, but the *scored sets* are now dMFM-guided; the GLASS-scored version stays available. |
| 15, 16 (code) | `t6_new = G / "table6" / "evo2_summary.csv"` -> `RESULTS / "dna" / "dmfm_demos" / "table6" / "dmfm4_norm" / "evo2_summary.csv"`, and add `.../glass_norm/evo2_summary.csv` as the matched control column. The `target_c0` / `source` / `set` keys the cell matches on are unchanged. |
| 19 (md/code) | "How to reproduce the published Table 6 exactly" -- add that the published sets were GLASS-guided and are in `results/dna/c0_guidance/n1000/`, while Table 6 as reported is now scored on `results/dna/c0_guidance/dmfm_n1000/`. |

**`notebooks/02_dna_base_model.ipynb`** -- no change needed (Fig 6 / Table 16 are base-model
marginals, unaffected by the posterior choice).

## Files and code

Created (all mine):

* `docs/dmfm_demo_reruns.md` (this file)
* `results/dna/dmfm_demos/` -- `PROVENANCE.md`, `fig7/`, `fig8/`, `diversity/`, `table6/`
* `scripts/dna/dmfm_demos/` -- `fig7_tilt_ladder.sbatch`, `fig8_trace_dmfm.sbatch`,
  `tables4_5_diversity_dmfm.sh`, `table6_evo2_dmfm.sbatch`, `collect.py`, `md_tables.py`
* `outputs/dna/dmfm_demos/` (gitignored run tree) -- `fig7/`, `diversity/`, `table6/`, `jobs.tsv`

Changed, additively, with the published behaviour as the default:

* `src/dmfm/regen/guidance_trace.py` -- new `--posterior {glass,dmfm}`, `--student_ckpt`,
  `--allow_gap_mismatch`, `--grad_normalize`; `trace_guided_vs_unguided` gained matching
  `posterior` / `student` / `grad_normalize` keyword arguments; `trace.csv` gained a
  `grad_norm_applied` column; `metadata.json` gained `student_ckpt` and records `posterior` /
  `grad_normalize` in `settings`. With no new flag the module runs exactly as before
  (`posterior="glass"`, clip-down rule), so no stored result changes. The student is resolved by
  `paths.dmfm_ckpt_for_steps(length, nfe_value)` and a mismatch is a hard error naming the right
  artifact, as in `sample_c0_guidance_dmfm`.

Fig 7 and Tables 4-6 needed **no** code change: `sample_c0_guidance_dmfm --paired --posterior
dmfm --grad_normalize`, `analyze_c0_diversity` and `score_yeast_c0_evo2` already accept
everything required.

No shared-file changes are needed: no new Python dependency (`md_tables.py` formats markdown
itself rather than pulling in `tabulate`), no `pyproject.toml`, `environment.yml`, `README.md` or
`.gitignore` edit.
