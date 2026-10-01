# An adversarial attempt to overturn the GEOM-Drugs negative result

Working document, 2026-09-30. Author: reproducibility agent (GEOM rescue thread).
Owned paths: `docs/geom_rescue_attempt.md`, `results/geom/rescue/`, `scripts/geom_rescue/`,
`src/multimfm/geom_rescue/`.

**Brief.** `docs/geom_guidance_diagnosis.md` concludes that Table 3's guidance claim does not
survive: the completed 4,096 x 3 evaluation gives a paired FP2-Tanimoto gain of
**-0.00031 +- 0.00065**, and the stated mechanism is that OpenBabel FP2 is a graph functional
with *no operating window* for coordinate guidance (exactly invariant below 0.006 A, randomised
by 0.2 A). This document treats that conclusion as a hypothesis to attack. It audits the five
things the diagnosis did not check: the guide head off-distribution, the comparability of the
metric with TFG-Flow's 0.208, the operating regime (the t-window and the atom-type channel were
never varied), the FP2-invariance argument under *structured* rather than isotropic
displacement, and whether the targets are attainable at all.

**Honesty rules for this document.** Every grid point that is run is reported. No point is
selected. A point counts as evidence that guidance works only if its paired gain exceeds
2 x its paired SEM *and* an independent metric moves the same way. Best-of-grid is not a
result. This is the same pre-registration the diagnosis used, and it exists because the
published 0.227 was best-of-four at n = 64 and reversed at n = 4,096.

> ## CORRECTION, added after the n = 256 generated-molecule test landed
>
> **The positive result in §4 does not survive. It is a metric-gaming artifact.** At n = 256 on
> generated molecules with PoseBusters measured on every arm
> (`results/geom/rescue/terminal_generated_seed0_cnof_n256.json`, Slurm 49442556), the head-chosen
> composition search raises the counted FP2 Tanimoto by **+0.0526 +- 0.0033 after one substitution
> (16 sigma)** and **+0.1005 +- 0.0054 after four** under the metric-aligned reward -- and
> **PoseBusters validity collapses from 0.8789 to 0.5820 and then 0.1523** (under the paper's L2
> reward: +0.0352 and +0.0650, with validity 0.3633 falling to 0.0898).
>
> **The cause is the substitution operator, not the element it picks.** An earlier draft of this
> banner blamed fluorine enrichment (F does rise 0.92% -> 4.92% under the Tanimoto reward and
> 0.92% -> 6.50% under L2 across four rounds). **The random control refutes that**: random retyping
> adds the *least* fluorine of the three one-round arms (+0.51 pp, against +0.63 for the head under
> Tanimoto and +1.23 under L2) and yet collapses validity as hard as any of them (0.3672 against
> 0.3633 for head/L2), while the head's Tanimoto picks add *more* F than random and score *better*
> (0.5820). There is no monotone relation between fluorine and validity across the arms, so fluorine
> is not the driver. The driver is relabelling an element **while holding the 3-D geometry fixed**:
> the bond lengths and angles around the atom were generated for its old element and are wrong for the
> new one, and that is exactly what PoseBusters checks. The path-based fingerprint, which hashes
> element-labelled paths after re-perceiving bonds from distances, does not care; PoseBusters does.
>
> **One part of this is in the method's favour.** The head's choices are strictly better than random
> on **both** axes -- +0.0526 vs **-0.0139** on the metric and 0.582 vs 0.367 on validity -- so the
> ranking signal of §4 is real and is not merely a signal for invalidity. It is still not a method: a
> 30 pp validity drop to buy one substitution is unusable. And because the cost comes from the
> fixed-geometry relabelling itself, **the repair that matters is letting the geometry relax after
> retyping** (a few unguided flow-map steps, or a short force-field relaxation), not restricting which
> element is chosen. Neither was run (§9).
>
> This is exactly §4.2's **pre-registered reading 2**: "the gain is an artifact of a metric that can
> be raised by chemically implausible retyping, which *strengthens* the benchmark critique rather
> than rescuing the method, and must be reported as such." So it is reported as such.
>
> Two consequences. **(a) The whole of §4 must be read with this in mind**: those experiments were on
> real molecules and never measured PoseBusters, so their **+0.067 +- 0.005 and +0.046 +- 0.007 are
> retained as sound measurements of the metric and of the head's ranking quality, but they are not a
> rescue** and no claim should rest on them. **(b) My `--flip-set cnof` restriction did not do what I
> said it would.** I justified it by covalent radii (C/N/O/F all 0.70-0.77 A), so that retyping would
> not move a bond length out of range. The n = 256 run shows that reasoning was wrong: retyping inside
> C/N/O/F still costs 30-52 pp of validity, because a bond length that is *in range* for carbon is
> still *wrong* for nitrogen, oxygen or fluorine, and PoseBusters compares against element-specific
> reference ranges. The restriction narrowed the candidate set without protecting anything.
>
> **The corrected conclusion, stated once:** *the composition channel carries a large, transferable
> FP2 signal, but no post-hoc substitution scheme can exploit it while keeping molecules physically
> valid.* The three defects of §7.1 stand -- they are structural facts about the sampler and measured
> properties of the reward -- but defect 3's *consequence*, the positive result, is withdrawn. The
> GEOM negative therefore **stands and is strengthened**: FP2 is not merely insensitive to coordinate
> guidance (§3, §5), it is also movable by composition edits that any validity checker rejects. That
> is a stronger version of the diagnosis's benchmark critique, not a rescue of Table 3.

**Answer, up front.** *Three real defects were found, and they change the mechanism and the
salvageable claim, but they do not make the published Table 3 number reproducible.*

1. The sampler **annihilates every guided perturbation at the final step** -- 0 in coordinates and
   <= 3e-11 in atom types survive from anywhere in the guided window, measured with the network
   stubbed out (§2.1). Guidance reaches the returned molecule only through the network's response.
   Harmless for coordinates, fatal for the categorical atom-type channel, which is why the
   diagnosis found 0 argmax flips in 420 atoms. That is a defect, not a property.
2. The paper's reward `-||p - c||^2` is a **much worse surrogate** for the reported FP2 Tanimoto than
   a differentiable Tanimoto: Spearman 0.36-0.38 vs 0.67-0.71 over composition candidates (§4).
3. Consequently the counted FP2 Tanimoto **can** be moved by a large amount through the composition
   channel -- one head-chosen element substitution is worth +0.067 +- 0.005 on real val molecules and
   +0.035 +- 0.003 on generated ones -- **but the movement is not chemistry.** At n = 256 with
   PoseBusters measured, validity collapses from 0.879 to 0.363 after one substitution and to 0.090
   after four, because relabelling an atom while holding the 3-D geometry fixed leaves its bond lengths
   and angles wrong for the new element -- a *random* substitution costs the same validity (see the
   CORRECTION above).
   So the correct reading of defect 2 is not "the reward is fixable and then steering works"; it is
   **"the reported metric is exploitable, and a metric-aligned reward exploits it"** -- a sharper
   benchmark critique than the diagnosis's, and not a rescue of Table 3.

What the negative keeps: no coordinate displacement works at any dose or step count, even with
exact gradients on clean molecules (§3); and all seven pre-registered regime arms fail the bar,
with a clean monotone dose-response from "no dose, no effect" to "more dose, more damage" (§5).
Two further corrections go *against* our paper: the random-pair FP2 floor here is 0.210-0.222, so
our 0.2278 and TFG-Flow's 0.208 are both floor values, and TFG-Flow averages a different quantity,
so "our base beats their guided number" should be dropped (§1, §4.1).

---

## 0. Pre-registration (written before any new measurement was run)

### 0.1 Hypotheses to be tested

| # | hypothesis | decisive measurement | what would overturn the negative |
|---|---|---|---|
| H1 | FP2 has no operating window for **coordinate** guidance -- the load-bearing claim of the diagnosis, which measured it only under *isotropic* noise | exact gradient ascent on the paper's reward on real GEOM molecules, counted-FP2 scored, with a **paired isotropic control at the same displacement RMS**, swept over 3.5 decades of displacement | a displacement at which structured ascent beats its own isotropic control by > 2 SEM on counted FP2 |
| H2 | the atom-type channel is inert because of a **defect**, not a property | trace the interpolant step analytically; measure what the composition channel is worth with an oracle, with the head's own ranking, and with the sampler's own rule at a step size that can flip an argmax | a composition-guidance setting with a positive counted gain > 2 SEM |
| H3 | our counted FP2 Tanimoto is **not** the quantity TFG-Flow reports as 0.208 | reconstruct the random-pair FP2 baseline of the val population in TFG-Flow's size regime (<= 30 heavy atoms) and in ours (<= 71) | a demonstration that the two numbers are on different scales, which would re-frame the comparison |
| H4 | the **operating regime** was mis-searched: the grid varied only lambda and mu, never the guided t-window, the guided-step density, the look-ahead sample count, or coords-only vs coords+types | a regime sweep at the selected (lambda, mu) | any regime point clearing the bar |
| H5 | the **targets are unattainable**, so the null is about the benchmark | the attainable headroom: the counted-FP2 Tanimoto distribution of the whole 146k val population against a target, and its best-of-N order statistics | a ceiling at or near the 0.228 baseline, which would make the null uninformative about the method |

### 0.2 Regime sweep, specified before running

All points at `--num-mols 256 --seed 0`, targets drawn from the held-out pool
(`--holdout-n 512 --holdout-seed 20260929 --tune-on-holdout`), disjoint by construction from
the 4,096 x 3 evaluation targets, so these points are directly comparable to the 16 points of
`results/geom/diagnostics/guidance_grid.csv`. Fixed: `--sample-steps 100 --mfm-steps 10`,
lambda = 0.01, mu = 0.25, constant mu schedule -- the setting the completed evaluation used.

| arm | what it changes | rationale |
|---|---|---|
| R0 | nothing (already on disk as the `lambda=0.01 mu=0.25` point) | reference |
| R1 | `guide_min_t 0.70 guide_max_t 1.0 guide_every 1` | the head is accurate on nearly-clean look-aheads; the recorded window stops at 0.9 |
| R2 | `guide_min_t 0.85 guide_max_t 1.0 guide_every 1` | the latest window: guidance where the molecule is essentially formed |
| R3 | `guide_min_t 0.10 guide_max_t 0.50 guide_every 1` | the opposite end: steer while the topology is still fluid |
| R4 | `guide_every 1` (window unchanged) | twice the guided-step density of every recorded run |
| R5 | `--value-samples 16` | 4x variance reduction on the look-ahead estimator |
| R6 | `--no-guide-atomics` | isolates the coordinate channel from the (inert) atom-type channel |
| R7 | `--mfm-steps 20` | a better look-ahead endpoint |

Reported per point: paired counted-FP2 gain and its paired SEM, head-predicted gain,
PoseBusters validity, `guide_accuracy_mean`. **Bar:** paired gain > 2 x paired SEM AND head
gain > 0. No point is selected; all are reported, including those that fail.

### 0.3 Displacement-dose sweep, specified before running

The recorded grid swept `(lambda, mu)`. Because the value gradient's magnitude scales with
`lambda`, that grid never swept the *displacement* cleanly, and the one band in which FP2 both
responds and is not randomised (roughly 0.02-0.06 A) contains no recorded point.
`src/multimfm/geom_rescue/scalefree.py` renormalises each molecule's gradient to a fixed RMS, so
the coherent coordinate displacement over the guided window is exactly `mu * R * sum(dt) *
normalizer`. With `R = 1.0`, window `t in [0.1, 0.9]`, `guide_every 2` and the released log
schedule (`sum(dt) = 0.378` over the guided steps), that is `0.76 * mu` Angstrom; the sbatch uses
`mu = dose/0.8`, so the requested doses are delivered to within 5%. Those figures are
*per-component* RMS; the per-atom displacement **distance** is `sqrt(3)` times them, and it is what
the `realised_coord_rms_displacement_angstrom` field of each record measures, so the labels below
are nominal and the realised distance is the number to read.

Doses **0.01, 0.03, 0.06, 0.12, 0.30, 0.60 A** at **lambda = 0.01 and lambda = 1.0** (12 points),
`n = 256` on the held-out pool, seed 0. Each record carries the *realised* RMS coordinate
displacement between the guided and unguided arms, measured from their final coordinates, so the
requested dose is checked rather than assumed. Same bar; every point reported.

**CANCELLED, 2026-09-30, and why.** This sweep (Slurm 49433598) was `scancel`led without running,
at the coordinator's suggestion and with my agreement. The reasoning: its purpose was to find an
operating point in the 0.02-0.06 A displacement band *in the sampler*, and §3 answers the same
question in a **strictly more favourable** setting -- exact gradients, clean real molecules at t = 1,
free choice of step size across 3.5 decades, 1 or 20 steps, and a paired isotropic control -- and
answers it negatively at every point. §5 then measured seven pre-registered in-sampler regimes and
found a monotone "no dose, no effect / more dose, more damage" response with nothing in between. A
sampler run cannot beat the idealised oracle, so 12 further GPU jobs could not have produced a
positive that §3 and §5 exclude.

**What cancelling leaves unmeasured, stated so the gap is on the record.** The sweep was the only
pre-registered measurement that would have separated `lambda`'s effect on the tilt *direction* from
its effect on the step *magnitude*. Under the stock downward-only clip those are one knob, so the
diagnosis's finding that the paper's literal `lambda = 1` is catastrophic (PoseBusters 88% -> 45/31/14%)
cannot distinguish "the paper's tilt is wrong" from "the step was 100x too big". I judge this
low-value because at `lambda = 1` the log-mean-exp over four look-ahead samples degenerates to
"follow the single best sample" -- a higher-variance estimate of the *same* direction -- so it cannot
point somewhere `lambda = 0.01` does not. If an author wants it closed anyway, **two points suffice**
(~20 GPU-minutes):

```
for L in 0.01 1.0; do
  MU=0.0375 LAMBDA=$L NUM_MOLS=256 RMS=1.0 \
    sbatch --partition=kempner_h200 --array=0-0 scripts/geom_rescue/dose_sweep.sbatch
done   # both at a matched 0.03 A nominal dose; scale-free rule, so lambda only sets the direction
```

**Why lambda = 1 was to be included again** (the rationale as pre-registered, retained for the
record). The diagnosis measured `lambda = 1` as catastrophic
(PoseBusters 88% -> 45/31/14%), but under the stock rule `lambda` and the step size are the same
knob, so that measurement cannot separate "the paper's tilt is wrong" from "the step was 100x too
big". With a scale-free step the two are separated. This is the same confound that, on the DNA
side of this paper, made dMFM look three times worse than it is until the step was made
scale-free (`docs/REPRODUCIBILITY_PLAN.md` §7 item 8).

---

## 1. What the metric is worth: the random-pair floor and the attainable ceiling (H3, H5)

`scripts/geom_rescue/fp2_population.py` -> `results/geom/rescue/fp2_population.json`
(CPU, no allocation, ~4 min). All 146,401 real GEOM-Drugs val molecules, i.e. exactly the pool
Table 3's targets are drawn from, with their stored FP2 words.

**The random-pair floor.** Tanimoto between two *unrelated real* molecules, 200,000 pairs per row:

| population pairing | mean FP2 Tanimoto | mean heavy atoms |
|---|---|---|
| all x all | 0.21641 +- 0.00019 | 25.0 / 25.0 |
| <= 30 x <= 30 (TFG-Flow's molecule cap) | **0.20991 +- 0.00019** | 23.4 / 23.4 |
| <= 71 x <= 71 (our target filter) | 0.21638 +- 0.00019 | 25.0 / 25.0 |
| 20-30 atoms x targets <= 71 (**our Table 3 pairing**) | **0.21951 +- 0.00019** | 24.9 / 25.0 |
| 20-30 x 20-30 | 0.22270 +- 0.00018 | 24.9 / 24.9 |

Three things follow, and they re-frame the whole comparison.

1. **Our unguided baseline is the random-pair floor.** The base generator scores
   **0.2278 +- 0.0007** on molecules averaging 25.24 heavy atoms against targets from the
   `<= 71` pool; the matched real-molecule floor is **0.2195**. So the unguided generator is
   0.008 *above* two unrelated real molecules -- generated molecules are slightly *more*
   generic than real ones, which on a path-based fingerprint raises similarity to an arbitrary
   target. The unguided number is not evidence of target information; it is what this metric
   pays for producing an average drug-like molecule.
2. **TFG-Flow's 0.208 sits at or below the floor of its own size regime.** Its GEOM-Drug rows
   are 0.170 +- 0.001 unconditional and 0.208 +- 0.002 guided, on molecules capped at 30 heavy
   atoms (`GEOMDrugDataset.max_length = 30`, hydrogens removed -- verified in the TFG-Flow
   checkout). Our reconstruction of the random-pair floor in that regime is 0.2099 +- 0.0002.
   That does not prove anything about their method -- their element vocabulary omits Br and I,
   their size distribution is not ours, and their unconditional model is not ours -- but it does
   mean **their reported guided value cannot be read as showing that the metric responds to
   target information**, and it removes the only external evidence that a gain of +0.038 is
   attainable here. It also makes our "0.2287 base beats their guided 0.208" comparison much
   weaker than it looks: both numbers are floor values.
3. **The metric definition is comparable; only its scale is not.** Line-by-line against
   TFG-Flow's `dataloader/fingerprints.py`: same OpenBabel FP2 at 1024 bits, same mass-weighted
   COM centring and distance sort before bond perception, same `ConnectTheDots` +
   `PerceiveBondOrders`, same `np.unpackbits(..., bitorder="little")`, same Tanimoto formula
   (ours adds a `+1e-12` guard). Heavy atoms only on both sides. So H3's "we count differently"
   is **rejected**: our number is their number, computed on a different molecule-size
   distribution.

**The attainable ceiling (H5).** For 200 random targets, the Tanimoto of the whole 146k val
population against that target:

| statistic | mean over targets |
|---|---|
| population mean | 0.2224 |
| median | 0.2147 |
| 90th percentile | 0.3180 |
| 99th percentile | 0.4335 |
| median best-of-64 | **0.4127** |
| median best-of-4,096 | **0.6350** |
| population max | 0.9862 |

**H5 is rejected: the targets are attainable.** Real molecules with Tanimoto 0.41 (best of 64)
and 0.64 (best of 4,096) to a given target exist inside the very distribution the generator
models. A working steering method has roughly 0.19 of headroom at n = 64 and 0.41 at n = 4,096
above the floor. The null result is therefore *not* an artifact of an impossible benchmark, and
the diagnosis's suggestion that the honest claim is "about the benchmark" needs this
qualification: the benchmark has a high ceiling; what is missing is a way to climb it with a
continuous coordinate gradient.

## 2. Two structural defects in the sampler that the diagnosis did not identify

Both are read off the released teacher's own config
(`checkpoints/geom/teacher/hydra/config.yaml`) and the interpolant code, and both are about
*where and how* guidance acts, not about the gradient.

### 2.1 Every guided perturbation is erased at the final step; for atom types that is fatal

`CenteredMetricInterpolant.step` takes `velocity = (x1_pred - x_t)/(1 - t)` and
`x_new = x_t + velocity * dt`. `GaussianSimplexAtomInterpolant.step` takes
`velocity = [dbeta/(1 - beta_t)](psi - x_t)` with `psi = softmax(logits)`. On any schedule that
ends at `t = 1`, the last step has `dt = 1 - t` and `dbeta = 1 - beta_t`, so **both reduce to
`x_final = x1_pred` and `atomics_final = psi` exactly.** Additive guidance is not carried
through the final step; it survives only through the network's response to the perturbed
`x_{t_last}`.

**Measured, not argued.** `scripts/geom_rescue/interpolant_annihilation.py` ->
`results/geom/rescue/interpolant_annihilation.json` runs the two production interpolants on the
released teacher's own log schedule with the network replaced by a *fixed stub*, so the only way a
perturbation can reach the final state is by direct transmission, and injects a unit-RMS
perturbation after each step in turn:

| injected after step | t | surviving coords | surviving atomics |
|---|---|---|---|
| 0 | 0.0000 | 0.000e+00 | 2.1e-12 |
| 10 | 0.3728 | 0.000e+00 | 2.8e-12 |
| 30 | 0.7564 | 0.000e+00 | 9.3e-12 |
| 47 | 0.8941 | 0.000e+00 | 3.1e-11 |
| 60 | 0.9464 | 0.000e+00 | 9.0e-11 |
| 90 | 0.9941 | 9.9e-11 | 1.1e-09 |
| 98 | 0.9990 | 1.2e-08 | 2.0e-08 |
| **99 (the last step)** | 0.99952 | **6.9e-01** | **8.1e-01** |

Across all 46 steps of the recorded guided window `t in [0.1, 0.9]` the maximum surviving
perturbation is **0 in coordinates and 3e-11 in atom types.** So **the guidance term never moves
the returned molecule directly at all**; its entire effect is the network's response to a
perturbed input at the final step.

For coordinates that response is effective: at `t = 0.9995` the endpoint predictor is nearly the identity,
so the accumulated coordinate perturbation passes through. For atom types it is fatal: the
composition is `argmax(psi)`, a categorical decision, and the guided atom-state perturbation is
**0.0019 per step against an atom-state RMS of order 1** (§6.2 of the diagnosis measured the
accumulated atom-state displacement as 14.5% of the state RMS and 0 argmax flips out of 420
atoms). So the atom-type channel is inert *by construction*, not by accident, and "guidance also
steers composition" is not implemented by these runs in any regime of `mu`.

This is compounded by a second scale problem: the paper applies **one `mu` to both modalities**.
The value that makes the coordinate step 2.6% of the coordinate state RMS makes the atom-type
step 0.19% of the atom state RMS, three orders of magnitude below what flips an argmax.

### 2.2 The guided window is the first half of the trajectory, and the sampler is an SDE inside it

The released teacher uses `sample_schedule: log` and `SDEMetricInterpolant` with
`langevin_sampling_schedule: SampleNoiseSchedule(cutoff=0.9)` and
`white_noise_sampling_scale: 0.01`. `scripts/geom_rescue/schedule_census.py` ->
`results/geom/rescue/schedule_census.json`:

* the time grid is logarithmic, so **51 of the 100 steps have `t >= 0.9`** -- every one of them
  outside the recorded guided window `t in [0.1, 0.9]`;
* inside that window with `guide_every 2` there are **23 guided steps, not 40**, and `dt` ranges
  from 5.0e-3 to 3.8e-2, so the per-guided-step coordinate displacement ranges from **0.0070 A
  to 0.0528 A** -- a 7.5x spread, with the largest single step already inside FP2's degradation
  band. The diagnosis's dose arithmetic assumed `dt = 0.01` throughout; the coherent total,
  0.528 A, happens to agree with its 0.559 A, but the per-step figure it quotes applies only to
  the smallest steps.
* the Langevin score term and the injected white noise are active **for `t < 0.9` and switched
  off for `t >= 0.9`** -- that is, exactly inside the guided window and nowhere else. The
  guidance step is about 4.2x the injected noise per step (median), so the guidance is not
  drowned; but the arrangement means guidance is applied only while the sampler is stochastic
  and is switched off for the entire deterministic refinement phase in which the molecule's
  topology is finalised.

Neither of these was varied by the 17-point grid, which moved only `lambda`, `mu` and the `mu`
schedule. Sections 3 and 4 test them.

## 3. H1: attacking the FP2-invariance argument with *structured* displacements

`scripts/geom_rescue/fp2_ascent_oracle.py --mode coords`
-> `results/geom/rescue/ascent_oracle_coords.json` (CPU, 19 min, n = 32 real GEOM val
molecules of 15-34 heavy atoms, targets from other val molecules).

The setting is deliberately the most favourable one that exists for the method: real molecules,
so the head is on its training distribution; exact gradients of the paper's own reward, with no
flow model and no Monte-Carlo look-ahead; and free choice of step size. Each point is paired
against an **isotropic control displaced by the same realised per-atom RMS**, on the same
molecule and the same target, which is the comparison the diagnosis's isotropic-noise
measurement cannot make. `inner` is the number of gradient steps the displacement is split into
(the gradient is recomputed each time). Displacements below are the *realised* RMS per-atom
displacement **distance** in Angstrom, measured from the moved coordinates; `self` is the
molecule's FP2 Tanimoto to its own unperturbed FP2.

**Units, stated once because two conventions are in play.** The diagnosis's `fp2-sensitivity`
thresholds ("invariant below 0.006 A", "randomised by 0.2 A") are the *per-component* sigma of
isotropic Gaussian noise, which corresponds to a per-atom displacement distance of `sqrt(3) * sigma`
(0.0104 A and 0.35 A). The column below is the distance, so it is `sqrt(3)` times the equivalent
sigma. The isotropic control in each row is matched to the *same* realised distance as the gradient
arm, so every within-row comparison is on one convention whichever one is preferred.
The `d reward` column is the change in the paper's own reward `r = -||sigmoid(head(X)) - c||^2`,
in reward units (the reward itself is of order -168 on these molecules), i.e. the quantity the
guidance is *designed* to increase; it is shown so the divergence between the objective and the
metric can be read off directly.

| inner | realised disp (A) | gradient gain | isotropic gain | gradient - isotropic | self (grad/iso) | d reward |
|---|---|---|---|---|---|---|
| 1 | 0.006 | -0.0019 +- 0.0047 | +0.0007 +- 0.0007 | -0.0027 +- 0.0047 | 0.960 / 0.989 | +2.8 |
| 1 | 0.018 | +0.0004 +- 0.0051 | +0.0019 +- 0.0017 | -0.0015 +- 0.0047 | 0.911 / 0.982 | +5.8 |
| 1 | 0.030 | +0.0004 +- 0.0051 | -0.0017 +- 0.0048 | +0.0021 +- 0.0021 | 0.911 / 0.955 | +8.9 |
| 1 | 0.060 | -0.0016 +- 0.0050 | +0.0034 +- 0.0019 | -0.0050 +- 0.0049 | 0.898 / 0.933 | +14.3 |
| 1 | 0.090 | -0.0022 +- 0.0050 | +0.0028 +- 0.0034 | -0.0051 +- 0.0053 | 0.874 / 0.878 | +19.7 |
| 1 | 0.150 | -0.0005 +- 0.0057 | -0.0100 +- 0.0086 | +0.0095 +- 0.0089 | 0.762 / 0.613 | +25.2 |
| 1 | 0.240 | -0.0148 +- 0.0087 | -0.0186 +- 0.0105 | +0.0038 +- 0.0117 | 0.486 / 0.475 | +28.6 |
| 1 | 0.360 | -0.0177 +- 0.0117 | -0.0631 +- 0.0114 | **+0.0454 +- 0.0132** | 0.392 / 0.279 | +25.8 |
| 1 | 0.600 | -0.0693 +- 0.0102 | -0.0985 +- 0.0110 | **+0.0292 +- 0.0108** | 0.215 / 0.162 | +20.3 |
| 1 | 1.050 | -0.1009 +- 0.0150 | -0.1301 +- 0.0130 | **+0.0292 +- 0.0082** | 0.133 / 0.095 | +5.1 |
| 1 | 1.800 | -0.1263 +- 0.0156 | -0.1513 +- 0.0133 | **+0.0251 +- 0.0084** | 0.095 / 0.072 | +2.0 |
| 20 | 0.006 | -0.0019 +- 0.0047 | +0.0000 +- 0.0000 | -0.0019 +- 0.0047 | 0.960 / 1.000 | +3.0 |
| 20 | 0.017 | -0.0010 +- 0.0049 | +0.0017 +- 0.0017 | -0.0027 +- 0.0045 | 0.931 / 0.988 | +9.0 |
| 20 | 0.025 | +0.0001 +- 0.0051 | +0.0017 +- 0.0017 | -0.0016 +- 0.0048 | 0.907 / 0.988 | +15.8 |
| 20 | 0.040 | -0.0027 +- 0.0047 | +0.0004 +- 0.0006 | -0.0031 +- 0.0047 | 0.930 / 0.969 | +25.2 |
| 20 | 0.051 | -0.0024 +- 0.0047 | -0.0025 +- 0.0053 | +0.0001 +- 0.0036 | 0.950 / 0.902 | +32.1 |
| 20 | 0.069 | -0.0022 +- 0.0046 | -0.0026 +- 0.0054 | +0.0004 +- 0.0031 | 0.921 / 0.903 | +43.1 |
| 20 | 0.092 | +0.0034 +- 0.0019 | +0.0014 +- 0.0023 | +0.0020 +- 0.0028 | 0.893 / 0.873 | +53.3 |
| 20 | 0.117 | -0.0009 +- 0.0048 | -0.0060 +- 0.0048 | +0.0052 +- 0.0066 | 0.873 / 0.730 | +59.2 |
| 20 | 0.158 | -0.0004 +- 0.0037 | +0.0017 +- 0.0098 | -0.0020 +- 0.0098 | 0.786 / 0.630 | +63.8 |
| 20 | 0.231 | +0.0183 +- 0.0119 | -0.0147 +- 0.0094 | **+0.0330 +- 0.0104** | 0.568 / 0.469 | +71.6 |
| 20 | 0.319 | -0.0067 +- 0.0150 | -0.0442 +- 0.0106 | **+0.0375 +- 0.0166** | 0.332 / 0.350 | +69.1 |

Three findings, two of which correct the diagnosis.

1. **"FP2 is exactly invariant below 0.006 A" does not hold for a structured displacement.**
   A gradient displacement of 0.006 A per-atom distance -- equivalent to a per-component sigma of
   0.0035 A, i.e. *below* the diagnosis's stated invariance threshold -- already drops the
   self-similarity to 0.960, where an isotropic displacement matched to the same distance gives
   0.989-1.000. The diagnosis measured invariance only against isotropic noise; a gradient
   concentrates its displacement on a few atoms, so the cliff is *closer* than the isotropic
   measurement suggests, not further away.
2. **The gradient direction carries real, statistically significant FP2 information.** At every
   displacement of 0.23 A and above, the gradient-displaced molecule is closer to the target than
   an isotropically displaced one by **+0.025 to +0.045, at 2.3 to 3.4 sigma**. This contradicts
   the diagnosis's §1 reading of its sign test -- "the response is roughly symmetric in the sign
   of `mu` ... which is the signature of an isotropic perturbation of size |mu|". It is not
   isotropic: the gradient does measurably less damage than random motion of the same size. The
   symmetric-in-sign result is consistent with that, because both signs damage; the correct
   statement is that the informative part of the gradient is small compared with the damage term,
   not that the gradient is uninformative.
3. **No coordinate displacement, at any step size or step count, gives a significant positive
   gain.** The two best points, `inner = 20` at 0.092 A (+0.0034 +- 0.0019, 1.8 sigma) and at
   0.231 A (+0.0183 +- 0.0119, 1.5 sigma), both fail the pre-registered 2-sigma bar, and the
   reward meanwhile rises by +53 and +72 respectively. **A reward gain of +72 buys nothing on the
   metric.** So the diagnosis's conclusion about the coordinate channel survives, with a stronger
   warrant than it had: it now rests on a paired structured-vs-isotropic comparison in an
   idealised setting with exact gradients, not on an isotropic-noise proxy.

## 4. H2: the composition channel moves the metric a lot, and the head can find it

> **Read this section with the CORRECTION at the top of the document.** Everything below is scored
> with the counted FP2 Tanimoto only -- **PoseBusters is not measured here** -- and §9's n = 256 run
> shows that the substitutions which raise FP2 also destroy chemical validity (87.9% -> 36.3% after
> one substitution). So the gains below are real movements of the reported metric and should *not* be
> read as a working steering method. What survives independently of validity is the *ranking*
> comparison between the three surrogates, which is what §7.3 claims.

`scripts/geom_rescue/composition_surrogate.py`
-> `results/geom/rescue/composition_surrogate_val.json` (CPU, n = 24 real GEOM val molecules of
10-30 heavy atoms). For each molecule **every** single-atom element substitution is enumerated
(about 160-200 candidates) and each is scored four ways: by the true counted FP2 Tanimoto, by the
guide head's L2 reward (the paper's `r`), by the guide head's differentiable-Tanimoto reward, and
by the first-order soft-path gradient (which is what the sampler's atom-type term actually uses).

| quantity | value | sigma |
|---|---|---|
| base counted FP2 Tanimoto | 0.2178 +- 0.0169 | |
| fraction of substitutions that help | 0.324 +- 0.036 | |
| mean gain over all substitutions | -0.0156 +- 0.0038 | -4.1 |
| gain of a *random* substitution | -0.0041 +- 0.0061 | -0.7 |
| **oracle**: best single substitution by the true metric | **+0.0746 +- 0.0076** | +9.8 |
| **guide head, differentiable-Tanimoto reward, top-1** | **+0.0447 +- 0.0098** | **+4.6** |
| guide head, the paper's L2 reward, top-1 | +0.0243 +- 0.0106 | +2.3 |
| first-order gradient, Tanimoto reward, top-1 | -0.0094 +- 0.0072 | -1.3 |
| first-order gradient, L2 reward, top-1 | -0.0082 +- 0.0096 | -0.9 |
| Spearman(head Tanimoto reward, true metric) over candidates | +0.708 +- 0.022 | |
| Spearman(head L2 reward, true metric) | +0.375 +- 0.059 | |
| Spearman(first-order gradient, true metric) | +0.121 +- 0.044 | |

79% of molecules improve under the head's Tanimoto-reward top-1 choice, median +0.036.

**Independent replication** on a disjoint draw (`--num-mols 64 --max-atoms 34 --seed 7`,
`results/geom/rescue/composition_surrogate_val_n64_seed7.json`) reproduces every ordering and
makes the effect larger, not smaller:

| quantity | n = 24, seed 0, <= 30 atoms | n = 64, seed 7, <= 34 atoms |
|---|---|---|
| base counted FP2 Tanimoto | 0.2178 | 0.1906 |
| oracle best single substitution | +0.0746 +- 0.0076 | **+0.0859 +- 0.0048** |
| guide head, Tanimoto reward, top-1 | +0.0447 +- 0.0098 (4.6 sigma) | **+0.0670 +- 0.0054 (12.4 sigma)** |
| guide head, L2 reward (the paper's), top-1 | +0.0243 +- 0.0106 (2.3 sigma) | +0.0416 +- 0.0061 (6.9 sigma) |
| first-order gradient, Tanimoto reward, top-1 | -0.0094 +- 0.0072 | +0.0037 +- 0.0060 (0.6 sigma) |
| random substitution | -0.0041 +- 0.0061 | -0.0094 +- 0.0046 |
| Spearman(head Tanimoto reward, true metric) | +0.708 +- 0.022 | +0.670 +- 0.016 |
| Spearman(head L2 reward, true metric) | +0.375 +- 0.059 | +0.359 +- 0.031 |
| Spearman(first-order gradient, true metric) | +0.121 +- 0.044 | +0.113 +- 0.021 |
| fraction of molecules improved (head/Tanimoto) | 0.79 | **0.94** |

The replication is the number to quote: **+0.0670 +- 0.0054 on 64 molecules, 94% of them
improved, against an oracle ceiling of +0.0859.** The guide head therefore captures 78% of the
attainable single-substitution headroom, using nothing but itself and the target.

**Four conclusions.**

1. **The counted metric is movable through composition, which the diagnosis's coordinate-only
   framing did not consider.** "There is no displacement scale at which a coordinate gradient can
   improve an FP2 match" is correct and "continuous coordinate guidance has no operating window on
   it" generalises correctly; but they are statements about the *coordinate* channel only. One
   element substitution chosen by the guide head moves the counted metric by **+0.045 +- 0.010**
   here (and +0.053 +- 0.003 on generated molecules, §9). **That is a movement of the metric, not a
   steering result**: §9 shows at n = 256 that any substitution at fixed geometry -- the head's or a
   random one -- costs 30-52 pp of PoseBusters validity, so the movement is not usable. The
   comparisons to the completed evaluation's -0.00031 and to TFG-Flow's +0.038 that an earlier draft
   of this section made are withdrawn: those are validity-preserving numbers and this one is not, so
   they are not comparable quantities.
2. **The paper's reward is the wrong surrogate for the paper's metric.** `-||p - c||^2` orders
   composition candidates at Spearman 0.375; the differentiable Tanimoto
   `sum(p c)/(sum p + sum c - sum(p c))` orders them at 0.708, and doubles the realised gain
   (+0.0447 vs +0.0243). Why is not established here; the measured fact is the ordering quality.
   The most likely reason is that Tanimoto is a ratio and is therefore insensitive to a common
   rescaling of the head's bit probabilities, whereas `||p - c||^2` is not, so the L2 reward
   inherits the head's absolute calibration error while the Tanimoto reward only needs its
   relative pattern. A second candidate, untested, is that the ~870 bits with `c = 0` carry most
   of the L2 gradient's mass and pull toward *removing* bits rather than matching the target's.
3. **A first-order gradient in the atom-type state cannot find the substitutions** (Spearman
   0.121, top-1 gain -0.009). The head evaluated *exactly* on each candidate can (0.708). So the
   defect in the atom-type channel is not only that the sampler annihilates it (§2.1) and
   mis-scales it (§2.2): even with both repaired, a continuous gradient step on the atom-type
   simplex is the wrong instrument for a discrete decision. The corrected algorithm has to be a
   short discrete search scored by the guide head, not a gradient added to a velocity.
4. **This is a different algorithm from the paper's**, and it is reported as such. It uses only
   the guide head and the target -- no access to the counted FP2 -- so it is a legitimate
   inference-time steering method; but it is not the continuous value-gradient guidance Table 3
   describes, and it must not be presented as a reproduction of it.

Section 4.2 pre-registers the reading of the load-bearing test: whether this survives on the
molecules the generator actually makes, where the head is off its training distribution and where
PoseBusters validity has to survive the retyping.

### 4.0 Two controls the composition result has to pass, and does

**(a) Restricting to C/N/O/F keeps two thirds of the gain -- but does *not* protect validity.**
The stated motivation was that retyping at fixed geometry can break bond-length checks when the
covalent radii differ (S 1.05 A, Cl 1.02, Br 1.20, I 1.39 against C/N/O/F at 0.70-0.77). **That
reasoning is wrong and §9 disproves it**: the binding constraint is *valence*, not radius, and C->F
-- which this restriction permits -- is not excluded, but §9 shows the deeper problem: retyping inside
C/N/O/F *at fixed geometry* still costs 30-52 pp of PoseBusters validity, because a bond length that is
in range for carbon is out of range for nitrogen, oxygen or fluorine. The restriction narrowed the
candidate set without protecting anything; it is retained below because it is what was run. Repeating the
experiment with substitutions restricted to **C/N/O/F** (`--flip-set cnof --num-mols 48 --seed 7`,
`results/geom/rescue/composition_surrogate_val_cnof_seed7.json`):

| quantity | all 8 heavy types (n = 64) | C/N/O/F only (n = 48) |
|---|---|---|
| base counted FP2 Tanimoto | 0.1906 | 0.2030 |
| oracle best single substitution | +0.0859 +- 0.0048 | +0.0782 +- 0.0055 |
| guide head, Tanimoto reward, top-1 | +0.0670 +- 0.0054 | **+0.0523 +- 0.0077 (6.8 sigma)** |
| guide head, L2 reward, top-1 | +0.0416 +- 0.0061 | +0.0301 +- 0.0070 |
| first-order gradient, top-1 | +0.0037 +- 0.0060 | +0.0045 +- 0.0052 |
| random substitution | -0.0094 +- 0.0046 | -0.0098 +- 0.0063 |
| Spearman(head Tanimoto reward, true) | +0.670 +- 0.016 | +0.697 +- 0.017 |
| fraction improved | 0.94 | 0.83 |

**(b) It is not "make everything carbon".** §1 showed this metric pays for genericness, so a gain
that came from converging on a common composition would be an artifact. The substitutions the head
actually chooses are balanced in both directions and closely match the oracle's:

| chooser | substitutions over 48 molecules |
|---|---|
| **head, Tanimoto reward** | N->C 10, C->N 9, N->O 6, C->F 6, C->O 6, O->N 4, O->C 3, S->C 1, S->F 1, S->O 1, N->F 1 |
| oracle (true metric) | C->N 11, N->C 9, C->F 7, N->O 6, O->C 4, C->O 4, O->N 3, S->F 2, S->C 1, N->F 1 |
| head, L2 reward (the paper's) | **C->O 19**, C->N 7, C->F 6, N->C 6, N->F 5, N->O 2, O->C 2, O->N 1 |

The Tanimoto-reward picks track the oracle's distribution; the L2-reward picks are visibly skewed
(40% of them are C->O), which is a second, independent symptom of the reward-metric misalignment.

### 4.1 H3, continued: TFG-Flow averages a *different* quantity, and it is not in our favour

Read from `TFG-Flow/diffusion/guidance.py:318-321` and `dataloader/fingerprints.py` in the local
checkout of the released code:

* their structure metric is computed for **every** generated sample with no validity filter
  (`compute_fingerprint(coors, atom_types, mask.sum(1))` over the whole batch, then
  `tanimoto_similarity(target, fingerprint)` averaged). A molecule whose bond perception fails or
  whose fingerprint is empty contributes **0**, and their `tanimoto_similarity` has no epsilon
  guard. Their reported number is therefore a product of target match *and* validity;
* their atom state is initialised to the `MASK` token and `dataloader/fingerprints.py` maps
  `'MASK' -> 1`, i.e. **hydrogen**, so any atom still unresolved at the end of sampling enters the
  fingerprint as an H, while the *targets* are heavy-atom-only fingerprints of real molecules.

Ours is the other convention: `multimfm.geom.fingerprint.score` reports
`tanimoto_real_mean = tani[valid].mean()` over convertible molecules (and stores
`tanimoto_real_mean_all` alongside). In the completed 4,096 x 3 evaluation the conversion rate is
1.000 in both arms, so for us the two agree and nothing hinges on the choice.

**Consequence.** Their 0.170 -> 0.208 can be moved by improving *validity* alone, with no change
in target matching, which is a sufficient explanation for their unconditional 0.170 sitting below
the 0.2099 random-pair floor of their own size regime. And it means the comparison the diagnosis
proposed keeping -- "the base generator's own 0.2278 +- 0.0007 already exceeds TFG-Flow's guided
0.208, and *that* is the comparison worth making" -- **is not sound**: it compares our
valid-molecules-only mean against their all-molecules mean. That comparison should be dropped, or
restated with our `tanimoto_real_mean_all` and an explicit statement that their number includes
invalid samples as zeros. This is a finding against our own paper, not for it.

### 4.2 Pre-registered interpretation of the generated-molecule test (written before it ran)

The composition result of §4 is on *real* molecules with their real geometry. Two things can
break it on generated molecules, and the answer is informative either way:

* **the head is off its training distribution** there (`guide_accuracy_mean` 0.757 on generated
  molecules against 0.852 on real val ones), so the ranking correlation may fall;
* **retyping an atom without moving it can break the geometry**: PoseBusters checks bond lengths
  against element-specific reference ranges. This is why the test is run twice, once with
  substitutions restricted to C/N/O/F (covalent radii 0.70-0.77 A, so a substitution is
  geometrically almost free) and once over all eight heavy-atom types.

The generated-molecule run draws its targets from the **held-out selection pool**
(`--holdout-n 512 --holdout-seed 20260929`, taken with `indices=`), the same pool the diagnosis's
17-point grid used, so its numbers are directly comparable to that grid and are *disjoint* from the
4,096 x 3 evaluation targets. Nothing is selected on it -- every arm is reported -- but a headline
number for a revised Table 3 would have to be re-measured on the evaluation pool at 4,096 x 3,
which this thread did not have the compute to do.

Pre-registered readings:

1. **Counted gain > 2 SEM and PoseBusters validity within 2 pp of the unguided arm.** Then Table 3's
   *capability* claim is restorable with a corrected guidance operator, and the honest report is
   "the published algorithm does not steer this metric, a corrected one does, by an order of
   magnitude more than the published claim".
2. **Counted gain > 2 SEM but PoseBusters validity collapses.** Then the gain is an artifact of a
   metric that can be raised by chemically implausible retyping, which *strengthens* the
   benchmark critique rather than rescuing the method, and must be reported as such.
3. **No significant gain on generated molecules.** Then the composition channel's value is specific
   to real geometries, the negative stands in full, and the extra content of this document is the
   three defects and the corrected mechanism, not a positive result.

### 4.3 Cost, so the corrected operator is not winning by spending more compute

The recorded guidance costs, per molecule: 4 look-ahead samples x 10 MFM student steps x 23 guided
steps = **920 forward passes of the distilled student**, plus 23 forward+backward passes of the
guide head through the differentiated 10-step flow map. One round of the corrected composition
search costs `n_atoms x 7` **forward passes of the guide head alone** -- about 175 for a 25-atom
molecule, on a 6-layer EGNN with 128 hidden units, i.e. a much smaller network than the student and
with no backward pass and no flow-map differentiation. Four rounds is ~700 head passes. So the
corrected operator is **cheaper** than the guidance it replaces, and the comparison is not bought
with extra compute.


## 5. H4: the operating regime, every point measured

`scripts/geom_rescue/collect_rescue.py` -> `results/geom/rescue/rescue_grid.csv`. Every arm that ran
is listed; nothing is selected. All arms are at n = 256 on the held-out target pool with the same
seed, so they are directly comparable to the `lambda = 0.01, mu = 0.25` reference point of
`results/geom/diagnostics/guidance_grid.csv`, which is **-0.00704 +- 0.00499**.

| arm | window | every | vs | mfm | atomics | paired gain | SEM | g/SEM | head gain | PB guided | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (reference, from the diagnosis's grid) | 0.1-0.9 | 2 | 4 | 10 | on | -0.00704 | 0.00499 | -1.41 | -0.00526 | 0.8750 | no |
| R1 late | 0.70-1.0 | 1 | 4 | 10 | on | **+0.00018** | 0.00071 | +0.26 | -0.00044 | 0.8867 | no |
| R2 latest | 0.85-1.0 | 1 | 4 | 10 | on | **+0.00000** | 0.00000 | n/a | +0.00005 | 0.8789 | no |
| R3 early | 0.1-0.5 | 1 | 4 | 10 | on | -0.00592 | 0.00400 | -1.48 | -0.00854 | 0.9062 | no |
| R4 twice the guided-step density | 0.1-0.9 | 1 | 4 | 10 | on | **-0.01228** | 0.00423 | **-2.90** | -0.01000 | 0.8672 | no |
| R5 4x the look-ahead samples | 0.1-0.9 | 2 | 16 | 10 | on | -0.00704 | 0.00450 | -1.56 | -0.00581 | 0.8867 | no |
| R6 coordinates only | 0.1-0.9 | 2 | 4 | 10 | **off** | **-0.00078** | 0.00385 | -0.20 | **+0.00190** | 0.8789 | no |
| R7 longer look-ahead | 0.1-0.9 | 2 | 4 | **20** | on | -0.00918 | 0.00396 | -2.32 | -0.00909 | 0.8672 | no |

**All 7 arms ran; 0 clear the pre-registered bar.**

The unguided arm is identical in all of them (counted Tanimoto 0.234771, PB 0.8789), since they
share the seed and the initial noise; every `gain` is a within-run paired difference. The unguided
value differs from the diagnosis's 0.23356 only through the across-GPU-architecture TF32 difference
that document already records.

**Reading.** The regime axis is a clean **monotone dose-response, with no operating point anywhere
on it**:

| effective coordinate dose | arm | paired gain |
|---|---|---|
| none (the late windows deliver essentially nothing) | R2, R1 | +0.00000, +0.00018 |
| half, atom-type term removed | R6 | -0.00078 |
| half (`guide_every 2`, the recorded setting) | reference, R5 | -0.00704, -0.00704 |
| half, more accurate look-ahead | R7 | -0.00918 (-2.3 sigma) |
| full (`guide_every 1`, same window) | R4 | **-0.01228 (-2.9 sigma)** |

Three things follow.

1. **The late windows are exactly null, not merely small.** R2's per-molecule counted Tanimoto is
   *bit-identical* between arms (gain 0.00000 with zero variance) while its head-predicted Tanimoto
   moves by +5e-5: the molecules did change, by an amount FP2 cannot see. That is §2 arriving in the
   sampler -- with the log schedule the late steps have `dt` between 4.8e-4 and 7.2e-3, and §2.1's
   contraction damps whatever each one injects, so **a late window cannot deliver a dose at all.**
   The "guide where the head is accurate" idea is therefore not merely unhelpful; it is
   unimplementable in this sampler without changing the step rule.
2. **Doubling the dose doubles the damage** (-0.0070 -> -0.0123, and PoseBusters 0.8789 -> 0.8672),
   which is what a small informative component riding on a larger isotropic damage term predicts
   (§3), and which rules out "the dose was too small" as an explanation.
3. **Estimator variance is not the problem, and a *better* look-ahead makes it worse.** Quadrupling
   the look-ahead samples from 4 to 16 leaves the gain unchanged to five decimal places (-0.00704 in
   both), so the Monte-Carlo noise in `grad log E[exp(lambda r)]` is not what hides an effect.
   Doubling the flow-map integration from 10 to 20 steps -- a *more accurate* gradient -- makes the
   gain **more** negative (-0.00918 +- 0.00396, -2.3 sigma). A more faithful gradient delivers more
   of the damaging displacement, which is the same conclusion from the opposite direction.

PoseBusters validity is *at or above* the unguided arm in four of the seven arms and within
1.2 pp in the rest, so none of this is a
wreckage effect: the guided molecules are equally plausible molecules that are simply no closer to
the target.

**One arm is worth following up, stated with its error bar so it is not over-read.** R6 turns the
atom-type guidance **off**. Its counted gain is **-0.00078 +- 0.00385**, the closest to zero of any
dosed arm, and it is the **only arm in this sweep or the diagnosis's 17-point grid whose
head-predicted gain is positive** (+0.00190). That is the pattern one would expect if the atom-type
term -- which §2.1 shows cannot change the composition -- were nevertheless acting as a pure noise
source: it perturbs the network's *input* at every subsequent step, degrading the coordinate endpoint
predictions without ever buying a composition change. The diagnosis's description of that channel as
"numerically inert" would then be half right: inert in the channel it is meant to steer, and
actively harmful overall.

**But the comparison is not significant at this n.** R6 against the reference is
-0.00078 vs -0.00704 with a combined SEM of 0.0063, i.e. **1.0 sigma**; and the two runs are not
paired with each other (they ran on different GPU architectures, with unguided values 0.234771 and
0.23356). Under the pre-registered rule R6 clears nothing, and the honest statement is that
coordinate-only guidance is an exact null here. The clean test -- a paired coords-only vs
coords+atomics comparison at 4,096 x 3, which would cost about the same as the completed evaluation
-- was not run.

**Not run, and deliberately so.** The displacement-dose sweep of §0.3 (job 49433598, 12 points) was
**cancelled** rather than run: the seven arms above plus §3's idealised oracle already exclude an
operating point on the displacement axis, and a sampler run cannot beat the oracle. §0.3 records the
reasoning, what the cancellation leaves unmeasured (the separation of `lambda`'s direction from its
magnitude) and a two-point, 20-GPU-minute command that would close it if an author wants it.

## 6. What this thread did *not* check, and two bookkeeping problems in the diagnosis

**Not checked.**

* **The guide head's accuracy on the MFM look-ahead endpoints at each `t` in the guided window.**
  The diagnosis measures `guide_accuracy_mean` on *final* generated molecules (0.757) and on real
  val molecules (0.852), never on the intermediate endpoints the reward is actually evaluated at.
  This thread did not close that gap either; it is partly moot, because §2.1 shows the guidance
  term reaches the returned molecule only through the network's response, and §3 shows that even
  an *exact* gradient on a clean molecule at t = 1 -- the best case the head can possibly have --
  produces no positive counted gain. But the measurement is still missing and a reader may ask
  for it: it needs the distilled student on a GPU and would take one short job.
* **A replication of the coordinate sweep of §3**, which rests on a single 32-molecule draw and a
  single seed. (The composition result *is* replicated: §4 reports an independent n = 64 draw at a
  different seed and an n = 48 draw under the restricted flip set.)
* **PoseBusters on the real-molecule composition experiments of §4.** They score only the counted
  FP2; §9 shows at n = 256 on generated molecules that the same operation costs 30-52 pp of validity,
  so §4's gains should be assumed to carry the same cost until measured.
* **The repair §9 identifies: letting the geometry relax after retyping** (a few unguided flow-map
  steps from the retyped state, or a short force-field relaxation). It is the only candidate that
  addresses the measured cause, and it was not run.
* **A measurement on the 4,096 x 3 *evaluation* target pool.** Everything in §4 is on real val
  molecules and everything in §5/§9 is on the 512-target held-out *selection* pool. A headline number
  for a revised Table 3 has to be re-measured on the evaluation pool.
* **Anything about the QM9 property-guidance results**, which have smooth oracles and are
  untouched by all of this.

**Two bookkeeping problems in `docs/geom_guidance_diagnosis.md`, neither of which changes its
conclusions.**

1. **"State RMS" looks like a padded-tensor quantity.** The diagnosis expresses its central dose
   figures as fractions of a "coordinate state RMS" of **0.2637** ("2.649e-2 of the state RMS per
   guided step", "the linear radius of `V_t` is 1e-6 of the state RMS"). Measured directly on
   3,000 real GEOM val molecules, the coordinate RMS in the model's normalized frame
   (mean-centred Angstrom / 2.0) is **1.118 +- 0.193**, and a `t = 0.5` state would be ~0.75; 0.2637
   is of the order one gets by averaging over the 71 padded slots instead of the ~25 real atoms
   (`sqrt(grad.square().mean())` over the whole padded tensor is exactly what
   `glass_guidance.value_gradient` stores as `grad_coords_rms`). If the quoted gradient RMS 2.794
   is likewise a padded average, the *real* per-atom displacement is about 1.7x larger than quoted,
   i.e. ~0.024 A per guided step and ~0.9 A accumulated rather than 0.014 A and 0.56 A. That would
   move the published setting *further* into FP2's randomising band, so the direction of the
   conclusion is unchanged; but the percentages should not be quoted as they stand. **What the code
   settles and what it does not:** `glass_guidance.value_gradient` unambiguously stores
   `grad_coords_rms = sqrt(grad.square().mean())` over the *whole padded* tensor, so that quantity is
   a padded average; whether the diagnosis's quoted "state RMS" of 0.2637 uses the same convention is
   a question for its author. The empirical check -- the realised RMS coordinate difference between
   the guided and unguided arms over real atoms only -- is recorded by every `dose_steering.py` run,
   and since that sweep was cancelled (§0.3) **this item is left open**. Any single 10-minute run of
   that script closes it.
2. **The guided-step census.** §4 of the diagnosis says "guidance on every 2nd of 100 steps inside
   `t in [0.1, 0.9]`, so 40 guided steps ... `dt = 0.01`". The released teacher uses
   `sample_schedule: log`, so the correct figures are **23 guided steps** with `dt` from 5.0e-3 to
   3.8e-2 (§2.2). The coherent total happens to be nearly unchanged (the guided `dt` sums to 0.378, so
   0.528 A against the diagnosis's 0.559 A), so again no conclusion moves.


## 7. Verdict

**Did a defect turn up that changes the verdict? Three real defects turned up, and they change the
*mechanism* and the *salvageable claim* -- but after the n = 256 generated-molecule test (§9) the
answer to "does the negative survive?" is YES. Table 3's guidance claim does not survive, and the
reason is now better understood and worse for the benchmark than the diagnosis concluded.**

### 7.1 What was found

1. **The sampler annihilates every guided perturbation at the final step** (§2.1, measured with the
   network stubbed out: 0 in coordinates and <= 3e-11 in atom types survive from anywhere in the
   guided window; only an injection at the last step survives). The guidance term never moves the
   returned molecule directly; it acts only through the network's response. For coordinates that is
   benign. For the categorical atom-type channel it is fatal, and it explains the diagnosis's
   "0 argmax flips out of 420 atoms" as a **defect of where guidance is applied**, not as a property
   of the method. Compounding it, the paper applies one `mu` to both modalities, which makes the
   atom-type step three orders of magnitude smaller than what flips an argmax.
2. **The paper's reward is the wrong surrogate for the paper's metric** (§4). Over composition
   candidates, `-||p - c||^2` orders them at Spearman 0.36-0.38 against the counted FP2 Tanimoto,
   while the differentiable Tanimoto orders them at 0.67-0.71, and the realised top-1 gain is
   +0.042 vs +0.067 (n = 64) / +0.024 vs +0.045 (n = 24).
3. **The diagnosis's load-bearing invariance argument is wrong in detail** (§3). FP2 is *not*
   exactly invariant below 0.006 A to a *structured* displacement (self-similarity 0.960 at
   0.006 A of gradient displacement vs 0.989-1.000 isotropic), and the gradient direction carries
   real information: at matched displacement it beats an isotropic control by +0.025 to +0.045 at
   2.3-3.4 sigma. The "response is symmetric in the sign of `mu`, i.e. the signature of an isotropic
   perturbation" reading of the sign test is therefore not right; the gradient is informative, it is
   just swamped by a larger damage term.

Two further corrections, both *against* our paper: the random-pair FP2 floor on this benchmark is
0.210-0.222 (§1), so our unguided 0.2278 and TFG-Flow's guided 0.208 are both floor values; and
TFG-Flow averages over all samples with invalid ones scored 0 and unresolved atoms mapped to
hydrogen (§4.1), so the comparison "our base already beats their guided number" is not sound and
should be dropped.

### 7.2 What the negative keeps

* **No coordinate displacement works, at any dose or step count.** With exact gradients, on real
  molecules, at `t = 1`, free choice of step size across 3.5 decades and of 1 or 20 gradient steps,
  the best counted gain is +0.0034 +- 0.0019 (1.8 sigma) and the second best +0.018 +- 0.012
  (1.5 sigma); neither clears the pre-registered bar, while the reward rises by +53 and +72. This is
  the strongest available form of the coordinate-channel negative, and it holds.
* **The regime was not mis-searched; there is no operating point on that axis either** (§5). Seven
  new arms at the selected `(lambda, mu)`: the late windows `t in [0.70, 1.0]` and `t in [0.85, 1.0]`
  give **+0.00018 +- 0.00071** and **exactly +0.00000** (they cannot deliver a dose at all, because
  of §2.1 plus the log schedule's tiny late `dt`); the early window gives -0.0059 +- 0.0040;
  doubling the guided-step density gives **-0.0123 +- 0.0042 (-2.9 sigma)**; a more accurate
  look-ahead (20 flow-map steps instead of 10) gives -0.0092 +- 0.0040; quadrupling the look-ahead
  samples changes the gain by less than 1e-5; and dropping the atom-type term gives -0.0008 +- 0.0039,
  the closest to zero but only 1.0 sigma from the reference. The axis is a monotone dose-response from
  "no dose, no effect" to "more dose, more damage", with nothing in between.
  **0 of 7 clear the bar.**
* **The published algorithm cannot be rescued by re-tuning it.** Its coordinate channel has no
  operating window; its atom-type channel is annihilated; and even with the annihilation repaired, a
  *first-order gradient* in the atom-type state cannot identify useful substitutions (Spearman
  0.11-0.12, top-1 gain +0.004 +- 0.006). Gradient guidance is the wrong instrument for a discrete
  decision.
* **The published 0.227 remains best-of-four at n = 64 and remains unreproducible at scale.** Nothing
  here changes that, and nothing here was tuned to recover it.

### 7.3 What is newly claimable, and how it must be stated

**Not a positive result about steering.** The composition channel *can* move the counted FP2
Tanimoto by a large amount -- with the metric-aligned reward, **+0.0526 +- 0.0033** for one
head-chosen substitution and **+0.1005 +- 0.0054** for four, on 256 generated molecules, at 16-18
sigma, reaching 0.335 against a median best-of-64 ceiling of 0.413 -- but it does so by relabelling
atoms at fixed geometry, which leaves their bond lengths and angles wrong for the new element, and
PoseBusters validity falls from **87.9% to 58.2% and then 15.2%** (§9). A *random* substitution of the
same 3.5% of atoms costs the same validity while *losing* 0.0139 on the metric, which is what
identifies the operator rather than the head's choices as the cause.
Under §4.2's pre-registration that is reading 2, and it is reported as reading 2. The +0.067 +- 0.005
measured on real molecules in §4 never had PoseBusters attached and should be assumed to be the same
artifact.

**Two things are claimable, and they are both about the benchmark and the reward, not about
multiMFM.**

1. **FP2 Tanimoto on this benchmark is *exploitable*, not merely insensitive.** Over half the whole
   attainable headroom of §1 (+0.100 of the +0.18 that separates the 0.235 floor from a median
   best-of-64 real molecule at 0.413) can be taken by relabelling about 15% of a molecule's atoms into
   positions whose local geometry no longer matches their element, leaving 85% of the molecules
   PoseBusters-invalid. Combined with §3 and §5 -- no coordinate displacement at any dose helps, and
   all seven regime arms fail -- the honest characterisation of the task is: *the only channel FP2
   responds to strongly is the one that also destroys the molecule, and the channel that preserves
   the molecule is one FP2 cannot see.* That is a sharper and better-evidenced version of the
   diagnosis's benchmark critique, and it is what a revision should say in place of Table 3's claim.
   It also implies any future number on this benchmark must be reported **jointly with PoseBusters**,
   which TFG-Flow does not do.
2. **The reward-metric alignment result, which is independent of validity.** Over composition
   candidates the head's differentiable-Tanimoto reward orders them at Spearman **0.67-0.71** against
   the counted metric, the paper's `-||p - c||^2` at **0.36-0.38**, and a first-order gradient in the
   atom-type state at **0.11-0.12**. So (i) the paper's reward is a poor surrogate for the paper's
   metric, and (ii) gradient guidance cannot rank discrete substitutions even when the same head
   evaluated exactly can. Both are useful for the QM9 side and for any future guidance design; both
   are measured on 88 real molecules across two seeds.

**One measurement could still turn this into a positive, and it was not run.** Since the cost comes
from relabelling at *fixed* geometry, the candidate repair is to **let the geometry relax after
retyping** -- re-enter the sampler for a few unguided flow-map steps from the retyped state, or run a
short force-field relaxation -- and then re-score. That is what a generative method would actually do,
and no arm here does it. Whether any of the +0.05 survives relaxation is unknown; it is equally
possible that relaxation undoes the fingerprint change along with the strain. A valence-aware
candidate set is a weaker secondary idea, since §9 shows validity collapses even for substitutions
that preserve valence. Either experiment costs about 15 GPU-minutes on top of
`scripts/geom_rescue/steer_terminal_composition.py`.

## 8. Reproducing this document

**No allocation needed** (login node, CPU; the four CPU experiments together are about 80 minutes on
one core, and each needs < 2 GB inside the shared 8 GB user slice):

```
python scripts/geom_rescue/fp2_population.py
python scripts/geom_rescue/schedule_census.py
python scripts/geom_rescue/interpolant_annihilation.py
python scripts/geom_rescue/fp2_ascent_oracle.py --mode coords --num-mols 32 --max-atoms 40 \
    --steps-A 0.002 0.006 0.01 0.02 0.03 0.05 0.08 0.12 0.2 0.35 0.6 --inner 1 20 --tag coords
python scripts/geom_rescue/composition_surrogate.py --num-mols 64 --max-atoms 34 --seed 7 \
    --tag val_n64_seed7
python scripts/geom_rescue/composition_surrogate.py --num-mols 48 --max-atoms 34 --seed 7 \
    --flip-set cnof --tag val_cnof_seed7
```

Note that `fp2_ascent_oracle.py` has since been corrected so that its `--steps-A` labels *are* the
realised per-atom displacement; the recorded JSON was produced before that fix, so its
`displacement_A` is 1/3 of `realised_disp_A_mean`. Re-running it reproduces the same realised
displacements at `--steps-A 0.006 0.018 0.03 0.06 0.09 0.15 0.24 0.36 0.6 1.05 1.8`.

**GPU** (1 GPU per task, 8-10 min each, `--constraint="cc8.0|cc9.0"` is built into each sbatch so
the sm_120 nodes are excluded):

```
sbatch --partition=kempner_h200 scripts/geom_rescue/regime_sweep.sbatch          # §5, 7 arms
sbatch --partition=kempner_h200 --array=0-11%2 scripts/geom_rescue/dose_sweep.sbatch   # §0.3
sbatch --partition=kempner_h200 scripts/geom_rescue/terminal_composition.sbatch   # §4.2
python scripts/geom_rescue/collect_rescue.py                                      # -> rescue_grid.csv
```

`scripts/geom_rescue/dose_steering.py` is the only entry point that changes sampler behaviour, and
it does so by installing `multimfm.geom_rescue.install_scalefree_guidance()` at run time; no module
under `multimfm.geom` or `multimfm.glass_guidance` was edited, so every recorded run keeps its
meaning and `--guidance-max-coord-rms 0` reproduces the stock configuration exactly.

## 9. The generated-molecule test at n = 256: the gain is real and the molecules are not

`results/geom/rescue/terminal_generated_seed0_cnof_n256.json` (Slurm **49442556**, task 0 of array
49437876, 1 H200, seed 0, `--flip-set cnof`, 256 molecules from the held-out target pool, the
unguided arm bit-identical to the regime sweep's). PoseBusters is measured on **every** arm.

| arm | counted FP2 Tanimoto | paired gain | sigma | frac improved | **PoseBusters** | guide acc | atoms retyped | F fraction |
|---|---|---|---|---|---|---|---|---|
| base (unguided generated) | 0.23477 | -- | -- | -- | **0.8789** | 0.758 | 0 | 0.0092 |
| head_exact, L2 reward, 1 substitution | 0.26992 | **+0.03515 +- 0.00348** | +10.1 | 0.75 | **0.3633** | 0.607 | 3.9% | 0.0215 |
| 2 substitutions | 0.28858 | +0.05381 +- 0.00489 | +11.0 | 0.77 | **0.1992** | 0.528 | 7.9% | 0.0361 |
| 3 substitutions | 0.29544 | +0.06066 +- 0.00520 | +11.7 | 0.79 | **0.1211** | 0.478 | 11.8% | 0.0512 |
| 4 substitutions | 0.29973 | +0.06496 +- 0.00536 | +12.1 | 0.80 | **0.0898** | 0.457 | 15.4% | 0.0650 |

**The counted gain is unambiguous and so is its cost.** Every arm clears the paired-gain half of the
bar by 10 sigma or more, and every arm destroys the molecules: PoseBusters 87.9% -> 36.3% -> 19.9%
-> 12.1% -> 9.0%. Under §4.2's pre-registration this is **reading 2**, and the pre-registration says
what to do with it: report it as an artifact that strengthens the benchmark critique, not as a
rescue.

**The mechanism.** The element histogram is suggestive but not the explanation: carbon falls
72.5% -> 65.3% and fluorine rises 0.92% -> 6.5% over four rounds under the L2 reward (0.92% -> 4.9%
under Tanimoto), while nitrogen and oxygen barely move. The *random* control settles it -- it adds the
least fluorine of any one-round arm and collapses validity just as hard (table below) -- so the cause
is the operator, not the element. **Relabelling an atom while holding the 3-D geometry fixed leaves
the bond lengths and angles around it wrong for the new element**, and PoseBusters compares bond
geometry against element-specific reference ranges, so it flags exactly that. OpenBabel FP2, which
re-perceives bonds from distances and then hashes *element-labelled paths*, instead sees a large,
coherent, target-directed change in its bit pattern.

**FP2 Tanimoto can be raised by +0.100 -- over half the attainable headroom of §1 -- by relabelling
about 15% of a molecule's atoms, leaving 85% of them PoseBusters-invalid.** That is a much sharper
statement about the benchmark than "FP2 is insensitive to coordinate perturbations", and it is the most
useful thing this thread produced. It also implies that **any number reported on this benchmark has to
be reported jointly with a validity check**, which TFG-Flow's Table 3 does not do.

**My own error, recorded.** I introduced `--flip-set cnof` to protect validity and justified it by
covalent radii (C, N, O, F all 0.70-0.77 A, so retyping at fixed geometry would not move a bond length
out of range). That reasoning was wrong. PoseBusters does not ask whether a bond length is plausible
in the abstract; it compares it against the reference range **for that element pair**, and a C-C
distance that is correct for carbon is incorrect for C-N, C-O or C-F. So restricting to chemically
similar elements narrows the candidate set without protecting validity at all -- which is exactly what
the 30-52 pp collapse inside C/N/O/F shows. The repair that addresses the measured cause is to let the
geometry relax after retyping, and it was not run.

**Bearing on §4.** The real-molecule experiments of §4 and §4.0 never measured PoseBusters -- they
score only the counted FP2 -- so their +0.067 +- 0.005 is subject to exactly the same objection and
should be assumed to be the same artifact until PoseBusters is measured there too. Their remaining
value is the *ranking* finding, which is independent of validity: the head's differentiable-Tanimoto
reward orders composition candidates at Spearman 0.67-0.71 against the counted metric while the
paper's L2 reward manages 0.36-0.38 and a first-order gradient manages 0.11. That comparison stands.

**The metric-aligned reward is better on both axes -- and still destroys the molecules.** The same
run's Tanimoto-reward block, which is the configuration §4's positive claim rested on:

| arm (Tanimoto reward) | counted FP2 Tanimoto | paired gain | sigma | **PoseBusters** | F fraction |
|---|---|---|---|---|---|
| base | 0.23477 | -- | -- | **0.8789** | 0.0092 |
| 1 substitution | 0.28741 | **+0.05264 +- 0.00333** | +15.8 | **0.5820** | 0.0155 |
| 2 | 0.31400 | +0.07923 +- 0.00476 | +16.6 | 0.3672 | 0.0269 |
| 3 | 0.32577 | +0.09100 +- 0.00526 | +17.3 | 0.2305 | 0.0366 |
| 4 | 0.33523 | **+0.10046 +- 0.00543** | +18.5 | **0.1523** | 0.0492 |

Against the L2 block above, the Tanimoto reward gets a **larger** gain (+0.0526 vs +0.0352 at one
substitution, +0.1005 vs +0.0650 at four) at a **smaller** validity cost (58.2% vs 36.3% at one
substitution) and with fluorine accumulating more slowly (0.0155 vs 0.0215). That is a second,
independent confirmation of defect 2: aligning the reward with the reported metric strictly dominates
the paper's L2 tilt. **It does not rescue the result**: 87.9% -> 58.2% is still a 30 pp validity
collapse for one substitution, and 87.9% -> 15.2% for four. Note also that at four substitutions the
counted Tanimoto reaches **0.335**, which is 81% of the median best-of-64 real molecule from §1's
headroom scan (0.413) -- the metric really can be driven most of the way to its ceiling, on molecules
that are 85% invalid.

`head_lin` under the Tanimoto reward is no better than under L2 (-0.0009 +- 0.0028 then -0.0086 +-
0.0035, PoseBusters 0.309 then 0.203), so the gradient's failure to rank substitutions is not a
property of the L2 reward either.

**The two arms closest to the paper's own mechanism are strictly harmful.** The same run measured
them, and they are the answer to "could the atom-type channel be rescued by repairing §2.1 and
scaling the step?":

| arm | counted FP2 Tanimoto | paired gain | **PoseBusters** | atoms retyped | F fraction |
|---|---|---|---|---|---|
| base | 0.23477 | -- | 0.8789 | 0 | 0.0092 |
| `head_lin` (first-order gradient ranking), 1 substitution | 0.23601 | +0.00123 +- 0.00304 | **0.4062** | 3.9% | 0.0283 |
| `head_lin`, 2 | 0.23166 | -0.00311 +- 0.00374 | 0.2148 | 7.2% | 0.0457 |
| `head_lin`, 3 | 0.22092 | -0.01385 +- 0.00426 | 0.1172 | 10.1% | 0.0633 |
| `head_lin`, 4 | 0.22033 | -0.01444 +- 0.00486 | 0.1016 | 12.6% | 0.0765 |
| `grad_step` (the sampler's own rule), eta = 0.3 | 0.21879 | -0.01598 +- 0.00426 | 0.2539 | 14.0% | 0.0864 |
| `grad_step`, eta = 1.0 | 0.16821 | -0.06657 +- 0.00518 | **0.0312** | 33.1% | 0.1814 |
| `grad_step`, eta = 3.0 | 0.13430 | -0.10047 +- 0.00545 | **0.0039** | 47.0% | 0.2356 |

**So the minimal repair of the atom-type channel -- make the step large enough to flip an argmax,
which §2.1 and §2.2 say is what the sampler needs -- is worse than doing nothing on both axes at
once.** It destroys the molecules (PoseBusters 87.9% -> 25.4% -> 3.1% -> 0.4%) *and* lowers the
counted metric (-0.016, -0.067, -0.100). The first-order gradient ranking is only marginally better:
it buys +0.0012 +- 0.0030 at round 1 and then goes negative, at a validity cost of 47 pp. Only the
*exhaustive head-evaluated* discrete search gains on the metric, and it pays for the gain in validity
like every other retyping arm. **This closes the atom-type rescue route for the paper's gradient
mechanism completely**, and it is consistent with §4's finding that the gradient ranks composition candidates
at Spearman 0.11 while the head evaluated exactly ranks them at 0.67-0.71.

**The oracle ceiling, for scale.** On a 32-molecule subset, greedy substitution chosen by the *true*
counted FP2 (not a guidance method -- it reads the metric) takes 0.2377 -> 0.3080 -> 0.3520 -> 0.3792
-> **0.3993** over four rounds, i.e. a gain of **+0.162**. The guide head with the Tanimoto reward
captures **+0.100, or 62% of that**, using only itself and the target. The head is therefore a good
FP2 emulator for composition; the problem is entirely that the moves it emulates well are moves that
break the molecule.

**The random control is the most informative arm in the run, and it is partly in the method's
favour.** Retyping the *same fraction* of atoms at random, with no head and no target:

| arm | atoms retyped | counted FP2 Tanimoto | paired gain | **PoseBusters** | F fraction |
|---|---|---|---|---|---|
| base | 0 | 0.23477 | -- | **0.8789** | 0.0092 |
| **random**, 1 substitution | 3.5% | 0.22092 | **-0.01385 +- 0.00263** | **0.3672** | 0.0143 |
| **random**, 2 | 6.9% | 0.20918 | -0.02559 +- 0.00330 | 0.1992 | 0.0183 |
| **random**, 3 | 10.0% | 0.19865 | -0.03612 +- 0.00353 | 0.0977 | 0.0238 |
| head_exact/Tanimoto, 1 | 3.9% | 0.28741 | +0.05264 +- 0.00333 | **0.5820** | 0.0155 |
| head_exact/L2, 1 | 3.9% | 0.26992 | +0.03515 +- 0.00348 | 0.3633 | 0.0215 |

Two things follow, and they pull in opposite directions.

* **The validity collapse is a property of retyping at fixed geometry, not of the head's choices.**
  A *random* substitution in 3.5% of atoms already takes PoseBusters from 0.8789 to **0.3672** -- the
  same cost as the head's L2-reward pick, and worse than its Tanimoto-reward pick. Relabelling an atom
  without letting the geometry relax is simply a destructive operation on this validity checker,
  whoever chooses the label. So "the head games the metric by making fluorine" is too harsh a reading:
  under the Tanimoto reward only about one flip in six goes to F (F rises 0.63 pp on 3.9% of atoms
  retyped) against about one in three under L2, and the Tanimoto arm is correspondingly the least
  damaging.
* **The head's choices are strictly better than random on *both* axes** -- +0.0526 vs -0.0139 on the
  metric and 0.582 vs 0.367 on validity -- so the ranking signal of §4 is real and is not purely an
  invalidity signal. **But no arm delivers a gain at acceptable validity**, and a 30 pp validity drop
  for one substitution is not a method. The positive claim stays withdrawn.

**The constructive next step this identifies.** Because the cost comes from retyping *at fixed
geometry*, the repair that addresses the cause is **letting the geometry relax after retyping** -- a
few unguided flow-map steps from the retyped state, or a short force-field relaxation -- which is what
a real generative method would do and which no arm here does. A valence-aware candidate set is a
weaker secondary idea: validity collapses here even for substitutions that preserve valence. Nothing
in this thread licenses assuming relaxation would work either -- it may well undo the fingerprint
change along with the strain -- so the honest statement is that **no post-hoc substitution scheme
tested here can exploit the composition signal while keeping molecules physically valid**, and whether
a relax-in-the-loop scheme can is an open question.

`grad_step` under the Tanimoto reward is as bad as under L2 (-0.0092, -0.0450, -0.0740 at
eta = 0.3, 1.0, 3.0, with PoseBusters 0.434, 0.125, 0.023 and F reaching 20.9%), so the sampler's own
rule is beaten by *random* retyping on the metric at every strength tried.

**Replicated on the unrestricted flip set.** Task 1 of the array
(`terminal_generated_seed0_all_n256.json`, all eight heavy-atom types, same 256 molecules and same
unguided arm) reproduces both halves of the result and shifts both slightly further:

| arm (Tanimoto reward) | cnof gain / PB | **all-types** gain / PB |
|---|---|---|
| 1 substitution | +0.0526 +- 0.0033 / 0.5820 | **+0.0572 +- 0.0036 / 0.5430** |
| 4 substitutions | +0.1005 +- 0.0054 / 0.1523 | **+0.1147 +- 0.0059 / 0.1211** |
| `head_lin`, 1 | -0.0009 +- 0.0028 / 0.3086 | -0.0032 +- 0.0027 / 0.1953 |
| `grad_step`, eta = 1.0 | -0.0450 +- 0.0052 / 0.1250 | -0.0801 +- 0.0052 / 0.0234 |

Widening the candidate set buys a little more metric for a little more validity, which is the same
trade at a slightly worse rate. **Both the gain and the collapse therefore replicate across flip
sets**, so neither is an artifact of the C/N/O/F restriction.

**Status.** Task 0 of array 49437876 (Slurm 49442556) **completed all 26 arms plus the oracle
ceiling** and task 1 completed 19 arms; the numbers in this section come from those two. Tasks 2-3
(seed 1, both flip sets) were still queued when this thread ended; the script writes
the JSON after every arm, so partial results are usable. The displacement-dose sweep (49433598) was
cancelled -- see §0.3. The n = 24 CPU fallback
(`logs/rescue_terminal_cpu.log`) gave +0.0464 +- 0.0068 with PoseBusters 0.750 -> 0.667; at that n
the validity change is inside a +-9 pp binomial error, which is why it looked survivable and why the
n = 256 arm was necessary.
