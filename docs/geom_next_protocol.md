# Evaluation protocol for a new GEOM-Drugs steering comparison

**Status: design document, written 2026-09-30 before any result exists. Nothing here has been
run.** Owner: the geom-next-protocol agent. Owned path: this file alone. No code was written, no
job was submitted, nothing was committed.

**What this is.** The evaluation protocol for a proposed GEOM-Drugs experiment in which multiMFM is
compared against inference-time-steering baselines on a geometry-sensitive reward. It specifies, in
advance of any measurement, what would have to be true for a dominance claim to survive review: the
comparator set, the compute axis, the guide/oracle separation, the validity requirement, the sample
size, the hyperparameter-selection rule, and the conditions under which the claim is *falsified*.

**What this is not.** It does not choose the task (a parallel thread is surveying candidates) and it
does not establish what the sampler can steer (a third thread is measuring that). It assumes only
that the reward will be a geometry-sensitive function on GEOM-Drugs. Every gate below is stated so
that it can be applied to whatever task those threads return.

**Why it exists.** Three prior threads on this benchmark each found the same class of problem, and
each found it *after* the numbers were in:

* `docs/geom_guidance_diagnosis.md` / `docs/geom_rescue_attempt.md`: the published Table 3 GEOM
  number (0.227, 87.5% PoseBusters) is best-of-a-four-point-grid at n = 64 on one seed, and the
  completed 4,096 x 3 evaluation reverses it (paired gain **-0.00031 +- 0.00065**).
* The same rescue thread found that **over half the attainable headroom on that metric can be taken
  while leaving 85% of the molecules PoseBusters-invalid** (FP2 Tanimoto +0.1005 +- 0.0054 at four
  substitutions, PoseBusters 87.9% -> 15.2%), and that a *random* substitution costs the same
  validity. The metric was exploitable, not merely insensitive.
* The same thread measured the metric's **random-pair floor at 0.2164** (0.2195 for our exact
  pairing) and found our unguided 0.2278 and TFG-Flow's guided 0.208 both sitting on it.
* `docs/nfe_accounting_audit.md` found the published compute axis bills a differentiated forward at
  zero. At the measured backward cost the DNA crossover moves from 582 to **428** forward-equivalent
  NFE and the dMFM's win region collapses from two of five settings to one; **in wall-clock there is
  no regime in which the guided method wins at all.**

Every one of those is a protocol failure, not a modelling failure. They are all avoidable by fixing
the protocol first. That is what this document is for.

---

## 0. The standing rules

These bind the whole protocol and are not negotiable inside it.

**R0.1 — A negative result under this protocol is publishable and must be reported as readily as a
positive one.** If the pre-registered analysis says multiMFM does not dominate the baselines, that
is the result: it is written up, with the same tables, the same error bars and the same prominence a
positive would have received. Pre-registration is worthless if only positives are reported, and the
three prior threads on this benchmark all produced negatives that were more informative than the
positive they replaced. An abandoned task (§2 gate failure) is also reportable: "this benchmark
cannot resolve the effect any method could plausibly produce" is a finding about the benchmark, and
the field currently lacks it.

**R0.2 — Every arm that runs is reported.** No point is selected after the fact. Best-of-grid is not
a result. This is the same rule `docs/geom_rescue_attempt.md` §0 operated under, and it exists
because the published 0.227 was best-of-four at n = 64 and reversed at n = 4,096.

**R0.3 — Nothing is measured on the evaluation targets until the pre-registration form in §1 is
filled in, signed by an author, and committed.** Selection, tuning and pilots run on a disjoint
held-out pool (`--holdout-n`, `--holdout-seed`, `--tune-on-holdout` already implement this split in
`src/multimfm/geom/steering.py`).

**R0.4 — Deviations are logged, not silently absorbed.** Any departure from the committed form goes
in §11 with a date, the reason, and what it changes. A deviation discovered after unblinding
downgrades the claim tier by one (§7.4).

**R0.5 — One device type per panel.** CUDA and CPU do not draw the same samples and do not round
identically. `results/dna/fig4/PROVENANCE.md` records the enforcement: every shard stores
`device_type`, `tf32_matmul`, `slurm_account`, `slurm_partition`, and the collector warns if a
panel's shards disagree. Same rule here. A pilot that only *ranks* hyperparameters may run anywhere,
because it contributes no plotted number.

---

## 1. Pre-registration form

Fill every blank. Commit before the first evaluation run. An unfilled blank is a blocker, not a
default.

```
================ GEOM-DRUGS STEERING PRE-REGISTRATION ================
Version: ____   Date: ____________   Author(s): ______________________
Git commit of the code that will run: ________________________________
Commit of this filled form: __________________________________________

--- 1. TASK AND REWARD -------------------------------------------------
1.1  Reward name:                       ______________________________
1.2  Reward is computed:                [ ] exactly, in closed form, from
                                            coordinates (+ element labels)
                                        [ ] by a learned oracle network
                                        [ ] by an external tool (name: ____)
1.3  Closed-form definition (one line, or file:line):
     _________________________________________________________________
1.4  Per-molecule target c_i is drawn from: ___________________________
     Target pool size: ____  seed: ____  disjoint from selection pool? [ ]
1.5  Higher reward is better?           [ ] yes  [ ] no (state direction)
1.6  Reward is differentiable w.r.t. coordinates?   [ ] yes  [ ] no
     w.r.t. atom types (soft path)?                 [ ] yes  [ ] no
1.7  GUIDE (what the method may use at inference): __________________
1.8  OFFICIAL SCORER (what every number is reported with): __________
     If 1.2 is "exactly", 1.8 MUST be the closed form and 1.7 MUST NOT be.
     If 1.2 is "learned oracle": guide/oracle data split: ____________
     architectures/seeds differ?  [ ] yes   guide-oracle MAE on held-out
     ground truth (§3.3, must be filled BEFORE any steering run): ____

--- 2. GATE 0: FLOOR, CEILING, HEADROOM (§2) ---------------------------
     Run and record these BEFORE any steering run. Task is abandoned if
     any gate fails; abandonment is reported (R0.1).
2.1  Random-pair floor  F      = ________ +- ________   (n pairs: ____)
2.2  Unguided mean      U      = ________ +- ________   (n: ____, seeds: __)
2.3  Per-target paired SD of the reward, sigma_d = ________
2.4  Unguided best-of-N order statistics (bootstrap from one pool):
       N=4 ____  N=16 ____  N=64 ____  N=256 ____  N=1024 ____
2.5  Population ceiling (real val molecules vs target), median best-of-64
     and best-of-4096:              ________ / ________
2.6  Target-information check: unguided reward against OWN target minus
     against a PERMUTED target = ________ +- ________   (expect ~0)
2.7  GATE A  (metric has headroom):   (BoN@64 - U) / SEM(n*) >= 10 ?  [ ]
2.8  GATE B  (metric is not at floor): (U - F) is reported, and the
     headroom in 2.7 is >= 5x (U - F) ?                            [ ]
2.9  GATE C  (an exploit check exists): the cheapest known way to raise
     this reward while destroying the molecule is: ________________
     and it is measured in the arm list of 6.3 ?                   [ ]
2.10 GATE D (sampler can move it): from the steerability thread, the
     realised reward change from a dosed perturbation, at validity
     within 2 pp of unguided: ________ +- ________   nonzero?      [ ]

--- 3. METHODS AND BUDGETS (§4, §5) -----------------------------------
3.1  Method under test:   multiMFM (value_sampler=mfm), MC in {______}
3.2  Baselines that will run in-harness (tick and give the knob grid):
     [ ] best-of-N            N in {____________________}
     [ ] Feynman-Kac          K in {______}  resample_every ____ beta ____
     [ ] DPS                  guide_every in {______}
     [ ] beam search          W ____ L ____ K in {______}
     [ ] MCTS                 I in {______}
     [ ] unguided base        (mandatory)
3.3  REFERENCE BASELINE (the one the primary comparison is against):
     ______________________  chosen because ____________________________
3.4  Published numbers cited INSTEAD of running: ______________________
     For each, the five comparability tests of §4.4 are recorded in:
     ____________________________  (all five must pass, or it is run)
3.5  Primary compute budget B*  = ________ forward-equivalent NFE
     Secondary budgets          = ________ , ________
3.6  Backward cost k, MEASURED on this path (§5.2): k = ______
     measured on GPU model ________ at batch sizes ____________
     file: ______________________________   date: __________
3.7  Guide/head forward cost relative to one teacher forward: ______
3.8  Axis reported as primary:  fwd_equiv at the measured k  [ ] yes
     Sensitivity reported at k in {0, 1, 2, 3}?              [ ] yes
3.9  Wall-clock panel produced, matched batch width, one GPU model,
     all checkpoints resident?                               [ ] yes
     GPU model: ____________

--- 4. SAMPLE SIZE AND PAIRING (§7) -----------------------------------
4.1  Minimum interesting effect delta_min = ________ (reward units)
     stated as a fraction of the 2.7 headroom:  ________ %
4.2  sigma_Delta for the PRIMARY CONTRAST, from the pilot: ________
4.3  n (targets per seed) = ________   seeds = ________  (>= 3)
     Power at delta_min, alpha = 0.05 two-sided: ________ (>= 0.80)
4.4  Pairing: every method sees the same target list in the same order,
     and the same initial noise where the method admits one.   [ ] yes
     Shared-noise exceptions (e.g. best-of-N draws N): ______________
4.5  Primary analysis unit: the TARGET. Paired differences across
     methods, CI from ________ (t / bootstrap over targets).
4.6  Multiplicity: primary comparison is ______ vs ______ at B*.
     All others are secondary, Holm-corrected within the family.

--- 5. ENDPOINTS AND VALIDITY (§6) ------------------------------------
5.1  VALIDITY DEFINITION: PoseBusters, config ________________________
     (all chosen binary checks pass), plus RDKit conversion rate.
5.2  VALIDITY GATE: guided PB rate >= unguided PB rate - ____ pp,
     one-sided 95%. If the gate fails, the reward number is a
     DIAGNOSTIC, not a result, and is labelled as such.        [ ] yes
5.3  Every reported reward number carries its PB rate in the same
     row of the same table.                                    [ ] yes
5.4  Reward is reported under all three conventions:
     (a) all returned samples   (b) PB-valid only
     (c) invalid scored at the null value                      [ ] yes
5.5  PRIMARY ENDPOINT (derangement-controlled, §6.4):
     S(m,i) = R_own(m,i) - R_perm(m,i);  Delta = mean_i [ S(test,i)
     - S(reference baseline,i) ] at B*.
     Permutation is a fixed derangement with seed ______.

--- 6. CLAIM AND FALSIFICATION (§7.4, §8) -----------------------------
6.1  CLAIM TIER being registered (tick ONE, in advance):
     [ ] C1  uniform dominance: multiMFM is better than EVERY baseline
             at EVERY budget in the shared range
     [ ] C2  regime win: multiMFM is better below a crossover budget,
             and the crossover is reported as a number
     [ ] C3  parity/no claim
6.2  THE CLAIM IS FALSIFIED IF ANY OF THESE HOLDS (§8):
     F1  Gate 0 (§2) fails -> task abandoned, reported as such
     F2  Delta <= 0 at B*, or the upper 95% bound on Delta < delta_min
     F3  the validity gate 5.2 fails for the method under test
     F4  Delta loses significance under the derangement control while
         the uncontrolled gain does not (the gain is a population
         shift, not target-specific)
     F5  the sign of Delta is not consistent across all ____ seeds
     F6  the advantage exists at k = 0 but not at the measured k
     F7  under convention 5.4(c) the sign of Delta reverses
6.3  If F1-F7 all pass but the win is only at the cheapest budget,
     the reported tier is C2 and the crossover number goes in the
     caption. Reporting C1 when the data support C2 is a deviation.
6.4  Wall-clock: if there is no wall-clock budget at which the method
     wins, the sentence "at matched wall-clock the method does not
     win at any budget measured" appears in the paper.         [ ] yes

--- 7. SIGN-OFF -------------------------------------------------------
Author:  ______________________  date: __________
Second reader (not the author): ____________  date: __________
Evaluation targets remain untouched until both signatures exist. [ ]
=======================================================================
```

---

## 2. Gate 0: measure the floor and the ceiling before anything else

**Nothing is run on the evaluation pool until this passes.** It is cheap — all of it is CPU except
2.4 and 2.10 — and it is the step whose absence cost three threads their result.

### 2.1 The four quantities

| symbol | what | how |
|---|---|---|
| **F** random-pair floor | the reward between a generated (or real) molecule and an *unrelated* target | 2 x 10^5 random pairs, CPU. Template: `scripts/geom_rescue/fp2_population.py` -> `results/geom/rescue/fp2_population.json` |
| **U** unguided mean | the base generator's reward against its assigned target | one unguided pool, >= 4,096 molecules, the same sampler config the evaluation will use |
| **sigma_d** | per-target SD of the reward | from the same pool |
| **C(N)** ceiling | order statistics of the reward: (a) over the *generator's* own pool, (b) over the real val population | bootstrap from the pool; no new sampling |

**C(N) over the generator's own pool is the best-of-N baseline curve, exactly.** One unguided pool
of M molecules per target gives best-of-N for every N <= M by bootstrap resampling without
replacement. So the entire best-of-N comparator is obtained for the cost of the floor measurement,
before any baseline code is written. This is the single largest cost saving available in this design
and it should be taken.

### 2.2 The gates

* **Gate A, headroom.** `(C_gen(64) - U) / SEM(n*) >= 10`, where `SEM(n*) = sigma_d / sqrt(n* x
  seeds)` at the planned sample size. Rationale: a method that captures 10% of what best-of-64
  already achieves must be detectable at 1 sigma or better. If the whole headroom is worth 3 SEM, no
  method can be distinguished from any other and the experiment cannot produce a claim.
* **Gate B, the metric is not measuring genericness.** Report `U - F`. The FP2 failure was that
  `U - F = 0.008` against a headroom of 0.19 — the metric paid 4% of its dynamic range for
  target information and 96% for being an average drug-like molecule. Require the headroom of Gate A
  to be at least 5x `U - F`, and record both numbers in the paper whatever they are.
* **Gate C, a named exploit.** Write down, in advance, the cheapest way you can think of to raise
  this reward while destroying the molecule, and put it in the arm list (§6.3) as a control. For FP2
  it was element relabelling at fixed geometry. If you cannot name one, say so — but the arm list
  must still contain a random-perturbation control matched to the method's realised displacement.
* **Gate D, the sampler can move it.** From the steerability thread: a dosed perturbation must
  produce a reward change that is nonzero at 2 SEM *at validity within 2 pp of unguided*. The GEOM
  rescue measured the opposite on FP2 — the only channel the metric responded to was the one that
  destroyed the molecule (§9 of that document), and the coordinate channel produced no significant
  gain at any dose across 3.5 decades with exact gradients.

**If any gate fails, the task is abandoned and the failure is written up.** "This benchmark has
`U - F = 0.008` against a headroom of 0.19 and every method's number sits on the floor" is a
publishable statement about the benchmark, and it is currently missing from the literature on this
task.

### 2.3 The target-information check (form item 2.6)

Score each unguided molecule against its own target and against a permuted target. For an
unconditional generator the difference must be ~0; any systematic difference means the target draw
is correlated with the sample (e.g. size-matched targets), which is a confound that would be
credited to the method later. Record it even when it is zero, because §6.4 reuses the machinery.

---

## 3. The reward, and the guide/oracle separation

### 3.1 Prefer a reward that is exact in closed form from coordinates

**This is the single most valuable design choice available, and it removes the evaluator from the
argument entirely.** If the official scorer is `f(coords, elements)` computed in closed form, then:

* there is no guide/oracle gap to argue about, because the oracle is not a model;
* a reviewer cannot object that the evaluator was trained on the generator's distribution;
* the selection-bias failure of `docs/ss_guide_oracle_gap.md` — where best-of-N selection on the
  guide opened a 20x guide/oracle disagreement (21-35 meV guide MAE against 406-419 meV oracle) on
  exactly the same molecules — cannot occur, because selection and scoring use the same function;
* the exploit analysis is tractable: you can say precisely what configurations maximise `f`, and
  whether they are molecules.

Properties the closed form must have, all checkable before any run:

1. **Exact and deterministic** given coordinates and element labels; no conformer search, no
   force-field call, no stochastic embedding.
2. **Defined on invalid molecules too**, so convention 5.4(c) is computable and the validity gate
   does not silently change the population being averaged.
3. **Not invariant** under the displacement scale the sampler can deliver — this is Gate D, and the
   FP2 experience says measure it rather than assume it. FP2 turned out to be exactly invariant
   below ~0.006 A of *isotropic* displacement and randomised by 0.2 A, with the informative band in
   between too narrow to occupy; a smooth geometric functional (an inertia-tensor quantity, a
   radius-of-gyration-like scalar, a shape-overlap integral) has no such cliff, which is the point.
4. **Differentiable w.r.t. coordinates** in closed form, so the guide can be the exact reward and no
   surrogate head is needed for the coordinate channel at all. (This also makes the "is the reward a
   good surrogate for the metric?" question — defect 2 of the GEOM rescue, Spearman 0.36-0.38
   against 0.67-0.71 — disappear by construction.)
5. **Per-molecule targets drawn from a real population**, so the target set is attainable by
   construction (the H5 check of the rescue document).

### 3.2 If a learned oracle is unavoidable

Then the separation must be constructed, not assumed. The QM9 convention in this repo is the
template (`docs/ss_guide_oracle_gap.md` §1): guide and oracle trained on **disjoint data splits**
(there, 0-50k vs 50k-100k), different seeds, and anchored by reporting both networks' MAE on a third
split held out from both. Requirements:

* disjoint training data, disjoint validation data, different initialisation seeds, and preferably a
  different architecture;
* **the anchor table is produced and recorded before any steering run** (form item 1.8): both
  networks' MAE and R^2 on ground truth held out from both. Without it, a later guide/oracle
  disagreement cannot be attributed;
* **every method is scored by the oracle only**, and every method's *guide*-scored number is
  reported alongside, so the guide-oracle gap is visible per method. Methods differ enormously in
  how much selection pressure they put on the guide (best-of-N applies N evaluations; multiMFM-SS
  applied 230 x MC regressor calls, 28x best-of-N's at the QM9 crossover), so the selection bias is
  *not* common-mode and cannot be cancelled;
* the guide-oracle gap on *selected* molecules is reported for each method as a diagnostic column.

### 3.3 Whether baselines may use the exact reward

State it in the form. Two options, and the primary one must be the generous one:

* **Primary (recommended): every method may use anything computable at inference time.** If the
  reward is exact and cheap, best-of-N and FK may score with it directly while the guided method
  needs gradients through the sampler. This is a genuine advantage for the baselines and it is the
  only setting in which a win is not open to the objection that the comparator was handicapped. A
  dominance claim obtained against a deliberately weakened baseline is worth nothing.
* **Secondary (matched-information): every method sees only the guide.** Report it as a second row.
  It isolates "does the guidance mechanism help?" from "is an exact reward available?", which is a
  real and interesting question — but it is not the headline.

---

## 4. The baseline set

### 4.1 What exists, and what porting costs

Two implementations already exist in this repo and they are not equally far from GEOM.

**DNA (`src/dmfm/benchmarks/search.py`, 356 lines, after Didi et al. 2026)** implements best-of-N,
Feynman-Kac, beam search and MCTS against the `BaseFlow` abstraction in
`src/dmfm/benchmarks/core.py` (`prior`, `step` returning `(x, psi)`, `integrate`, `grid`). It is a
sequence model with a token state. Porting *that* file to GEOM means re-implementing the state
handling.

**QM9 (`src/multimfm/baselines.py`, 423 lines)** implements base, best-of-N, DPS and Feynman-Kac
against **exactly the code path the GEOM sampler already uses**:
`multimfm.glass_guidance.production_step_with_optional_guidance`,
`multimfm.glass_posterior.endpoint_prediction`, TensorDict states with
`coords`/`atomics`/`padding_mask`. `src/multimfm/geom/steering.py` imports `guided_sample` from that
same `glass_guidance` module, and `multimfm.nfe.attach_standard` hooks `model.net`, which the GEOM
teacher has. **So the QM9 baselines are the ones to port, and most of the port is swapping the model
loader, the target sampler and the scorer.**

| baseline | port from | what actually changes | estimate |
|---|---|---|---|
| **unguided base** | `baselines.sample_unguided` | model loader + scorer | ~0.5 d |
| **best-of-N** | `baselines.select_best` | **mostly unnecessary**: one unguided pool of M per target gives best-of-N for all N <= M by bootstrap (§2.1). Only the PB-valid-filtered variant needs code | ~0.5 d |
| **DPS** | `baselines.dps_gradient`, `sample_dps` | reward swap only; the one-step endpoint path is shared | ~0.5 d |
| **Feynman-Kac** | `baselines.sample_fk` | per-target repeat of a 2-D target (already handled by `grouped_rewards` in `src/multimfm/geom/reward.py`); resampling indexes a TensorDict | ~1 d + the stochasticity check below |
| **beam search** | `search.beam_search` | needs a ragged re-integration from an arbitrary step index against the GEOM schedule; W x L full roll-outs | ~1.5-2 d |
| **MCTS** | `search.mcts` | tree over trajectory prefixes, data-dependent NFE | ~2-3 d |

Plus ~1-2 d for the shared harness: scorer, PoseBusters on every arm, per-sample CSV, NFE counters,
shard/lock/skip-if-exists (copy `scripts/dna/fig4_all.sbatch`'s work-stealing pattern), and the
collector.

**Total: about one engineer-week for base + best-of-N + DPS + FK + harness; 4-5 days more for beam
and MCTS.**

### 4.2 Are they the right comparators?

* **best-of-N: mandatory.** It is the honest floor for any inference-time method and it is nearly
  free here (§2.1). It is also the comparator that most often wins in practice at large budgets.
* **Feynman-Kac: mandatory.** It was the *best* baseline at every budget on DNA Figure 4, and it is
  the one that sets the crossover. If only one search baseline can be afforded, it is this one, and
  the form should name it as the reference baseline (item 3.3).
* **DPS: mandatory, and it is the most informative single comparison in the whole design.** It uses
  the same gradient machinery as multiMFM but replaces the flow-map look-ahead with a one-step
  endpoint prediction. The difference between them *is* the contribution being claimed. It is
  already implemented for QM9 and costs half a day. Without it, a win over search baselines does not
  establish that the look-ahead is what bought the win.
* **beam search and MCTS: secondary.** On DNA neither was ever the best baseline (FK dominated
  both), MCTS has data-dependent NFE that complicates the axis, and together they are 4-5 days. Run
  them if budget allows; pre-register them as secondary so that omitting them is not a deviation.
* **The unguided base: mandatory**, and it must share the initial noise and targets with the guided
  arm so every gain is a within-run paired difference (this is what `run_steering` already does).

### 4.3 A GEOM-specific hazard for every branching baseline

FK, beam search and MCTS all require a **stochastic** sampler: with a deterministic sampler two
children of one state are identical, resampling collapses the population, and all three degenerate
to best-of-N.

The released GEOM teacher (`checkpoints/geom/teacher/hydra/config.yaml`) uses
`sample_schedule: log`, `SDEMetricInterpolant` with `SampleNoiseSchedule(cutoff=0.9)` and
`white_noise_sampling_scale: 0.01`. The Langevin score term and the injected noise are active for
`t < 0.9` and **switched off for `t >= 0.9`** — and the log schedule puts **51 of 100 steps at
`t >= 0.9`** (`results/geom/rescue/schedule_census.json`). So branching is only meaningful in the
first half of the trajectory, and the entire deterministic refinement phase in which the topology is
finalised admits no diversity at all.

**Required before any branching baseline is reported:** measure the effective sample size of the
particle population at each resampling time, and the pairwise diversity of candidates spawned from
one parent. If ESS collapses to ~1, the baseline is best-of-N wearing a different name and must be
labelled as such rather than reported as a defeated comparator. Pre-register the ESS threshold
(suggest: report FK as a distinct method only if median ESS/K >= 0.3 at the last resampling event).

### 4.4 Which published GEOM-Drugs numbers can be cited instead of run

A published number is citable only if it passes **all five** of these:

1. same base generator, or at minimum the same molecule-size distribution;
2. same target pool and same target-drawing rule;
3. identical reward/metric definition, including the treatment of unresolved atoms;
4. same validity convention (whether invalid samples are averaged in, and at what value);
5. a compute axis that can be converted to ours.

**Result of applying it: essentially nothing is citable, and all comparators must be run in-harness.**

* **TFG-Flow (ICLR 2025), GEOM-Drug rows 0.170 +- 0.001 unconditional / 0.208 +- 0.002 guided**
  (`results/geom/tfgflow_reference.json`) **fails 1, 3, 4 and arguably 2.** Their
  `GEOMDrugDataset.max_length = 30` heavy atoms against our 71; their `MASK` token maps to hydrogen
  in the fingerprint so unresolved atoms enter as H while targets are heavy-atom-only; they average
  over **all** samples with no validity filter and no epsilon guard, so a failed conversion
  contributes 0 and their number is a product of target match *and* validity; and their reported
  0.208 sits at or below the 0.2099 random-pair floor of their own size regime. They release **no**
  structural-fingerprint predictor for either dataset. Cite the paper, do not cite the number as a
  comparator.
* **Didi et al. 2026** is the source of the four search baselines but is protein binder design, not
  GEOM-Drugs. No transferable number.
* **best-of-N, FK, beam, MCTS on GEOM-Drugs with a geometry reward**: no published numbers exist
  that we could find in this repo's reference material. They must be run.

Write this out in the paper rather than leaving it implicit: "no published inference-time-steering
number on GEOM-Drugs passes our comparability test, so every baseline here was run in our harness on
our generator" is a defensible sentence and pre-empts the obvious review question.

---

## 5. The compute axis

### 5.1 The axis, defined in advance

Per returned sample:

```
axis_k  =  (generative forward passes)  +  k * (generative forward passes
                                                inside a differentiated graph)
```

`k = 0` is the published axis and **it is not fair**: it bills a forward-plus-backward as one
forward, so every generative forward the value estimator makes is free on the backward side, while
the search baselines are pure forward and do not move at all. On DNA that convention put the
crossover at 582 NFE; at the measured backward cost it is **428**, and the win region collapsed from
two of five settings to one.

**Pre-registered decisions:**

* **primary axis = `axis_k` at the k measured on this code path** (§5.2);
* **sensitivity at k in {0, 1, 2, 3} reported in an appendix table**, so a reader can read the
  verdict at any convention. On DNA every verdict was identical for all k in [1, 3], which is what
  made the audit robust; if the GEOM verdict *does* turn on k, that fact is the result and must be
  stated;
* **the guide/head is charged**, on a second axis, at its **measured** cost ratio to one teacher
  forward. Do not assume it is cheap. The QM9 reward regressor had 928,263 parameters against the
  base flow net's 338,121 and measured **1.30-1.39x** a base forward; the GEOM structure head is a
  6-layer EGNN and must be timed, not asserted. Methods differ by more than an order of magnitude in
  head calls (multiMFM-SS made 28x best-of-N's at the QM9 crossover), so charging for it can only
  move the crossover, and on QM9 it moved it against the guided method (4,214 -> ~3,450 at 1.0x);
* **counters, not estimates.** `multimfm.nfe` hooks `model.net`, the student and the regressors and
  counts rows per forward call, splitting out the back-propagated subset; `attach_standard` works on
  the GEOM teacher unchanged. Every arm records `nfe_generative`, `nfe_generative_bwd`,
  `nfe_regressor` per returned molecule;
* **one bookkeeping trap to avoid, already found once.** The DNA counter incremented
  `nfe.backward += B` once per (output, guided step) *independent of MC*, so it read 91 at every MC
  from 1 to 16 while the differentiated graph actually contained 91 x MC student forwards — a 16x
  understatement at MC = 16, and the paper sentence built on it ("one backward pass per guided
  step") was literally true and materially misleading. Count back-propagated *forwards*, not
  autograd *calls*.

### 5.2 How the backward cost will be measured

Not chosen, not assumed — measured, on this path, before the evaluation runs, and recorded in the
form (item 3.6). Template: `results/shared/nfe_audit/backward_cost_h200.json` (Slurm 49431669).

Procedure:

1. On the **same GPU model** the evaluation will use, with the real GEOM teacher, student and head
   loaded and all parameters frozen exactly as in production.
2. Time (a) one `model._call_net` forward and (b) one full `value_gradient` call, at **at least
   three production batch sizes**, >= 5 warm-up iterations, >= 20 timed repeats, with
   `torch.cuda.synchronize()` around each.
3. Report the ratio `(forward + backward) / forward`, and derive `k = ratio - 1` per generative
   forward inside the graph. Report it to two significant figures with its spread across batch
   sizes.
4. Separately time the head forward against the teacher forward for the second axis (item 3.7).

Prior measurements for calibration: **DNA value-gradient path, k = 1.1** (ratio 2.01-2.14 across
batch 25-1600 on an H200); **QM9, the whole guided step including its off-axis regressor, 2.89-3.00x
the forwards it is billed for**, which is exactly `axis_k2`. The theory bound is `k >= 1` because
all parameters are frozen and only input gradients are formed; it is a *lower* bound because
attention's activation-times-activation matmuls carry no weights and their backward needs two
matmuls per forward one regardless. Expect GEOM to land between the two — but measure it.

### 5.3 Wall-clock: yes, it is reported, and it is co-primary

**Pre-registered: a wall-clock panel is produced, and if the method loses on wall-clock everywhere
the paper says so.**

The DNA experience is the reason. The stored `elapsed_s` there was unusable (the timer started
before checkpoint loading, and the dMFM ran 25 rows in flight against the baselines' 200-3,200), so
all five methods had to be re-timed with every checkpoint resident and batch widths matched. Batch
matching alone made the dMFM 3.5-3.9x faster — a large correction **in our favour**, which is
exactly why it was worth doing. And even after it, **there was no wall-clock budget at which the
guided method won**: at its cheapest point (0.93 s per 100 outputs) it scored 0.133 where
Feynman-Kac reached 0.035. The measured throughput deficit at matched width (2.4x-3.5x) exceeded
what `k = 1.1` predicts, the residual being launch overhead from 91 autograd calls inside a 96-step
trajectory — worst exactly where the claim lived.

Rules:

* one GPU model, all checkpoints resident before the timer starts, batch widths matched across
  methods, >= 3 timed repeats per point;
* report seconds per returned *valid* molecule as well as seconds per returned molecule, since a
  method that returns 15% valid molecules is not comparable per-sample to one that returns 88%;
* the wall-clock panel is published whatever it shows. A claim that holds on forward-equivalent NFE
  but not on seconds is still a claim — it is a claim about NFE — but it must be stated that way,
  and the seconds must be in the same figure.

---

## 6. Validity

### 6.1 The definition

**PoseBusters, `configs/geom/posebusters_geom.yaml`, all chosen binary checks passing.** That config
is loading + RDKit sanitisation + InChI convertibility + all-atoms-connected + distance geometry
(bond lengths, bond angles, internal clash, `ignore_hydrogens: True`) + aromatic ring flatness +
**non-aromatic ring non-flatness** + double-bond flatness. It is the config Table 3 and Table 21
use. (`src/tabasco/utils/posebusters_no_strain.yaml` is the *training-validation* config, one check
short; do not mix them in one table — `src/multimfm/geom/check_pb_validity.py` documents the
distinction.)

Report alongside it: the **RDKit conversion rate** (fraction of returned samples that yield a
molecule at all), separately, because averaging conventions hinge on it.

### 6.2 The gate

**Guided PB-valid rate >= unguided PB-valid rate - 2 pp, one-sided 95%.** If the gate fails, the
reward number **is not a result**; it is a diagnostic, labelled as such in the table, and no claim
rests on it.

This is not a formality. The GEOM rescue took **+0.1005 +- 0.0054** on the reported metric — over
half the entire attainable headroom, 18 sigma — while PoseBusters fell from **87.9% to 15.2%**. At
n = 24 on CPU the same operation had looked survivable (validity 0.750 -> 0.667, inside a +-9 pp
binomial error); it took n = 256 to see the collapse. Small-n validity numbers do not constrain
anything, which is a second reason for §7's sample sizes.

### 6.3 Mandatory arms

Every evaluation runs these, and all of them are reported:

1. **unguided base**, sharing noise and targets with the guided arm;
2. **the method under test**;
3. **every baseline in the registered set**;
4. **a random-perturbation control**: the same operation as the method, same realised magnitude,
   random direction. This is the arm that settles whether a validity cost belongs to the method's
   *choices* or to its *operator*. In the GEOM rescue it was decisive: random retyping collapsed
   validity exactly as hard as the head's choices (0.367 vs 0.363) while *losing* 0.0139 on the
   metric, which is what identified the operator rather than the element choice as the cause — and
   it also showed the head's ranking signal was real (better than random on **both** axes) even
   though the method was unusable;
5. **the named exploit of Gate C**, as an upper bound on what the metric can be made to do by a
   method that does not care about chemistry.

### 6.4 The derangement control is the primary endpoint

For each returned molecule, score it against **its own target** and against a **permuted target**
(a fixed derangement with a registered seed). The primary endpoint is the difference:

```
S(m, i) = R_own(m, i) - R_perm(m, i)
Delta   = mean_i [ S(method, i) - S(reference baseline, i) ]
```

**Why this and not the raw reward.** The FP2 failure mode was that the metric rewarded
*genericness*: the unguided generator scored 0.008 above two unrelated real molecules simply for
producing an average drug-like molecule, and the composition exploit worked partly by pushing
molecules toward a common composition. A raw-reward gain cannot distinguish "moved toward this
target" from "moved toward the mode of the reward". The derangement control can, it costs one extra
scorer evaluation per molecule (free for a closed-form reward), and it is immune to any global shift
in the reward distribution.

Report both `Delta` (derangement-controlled) and the raw gain. **If the raw gain is significant and
the controlled one is not, the result is a population shift and the claim is falsified (F4).**

### 6.5 The three averaging conventions

Every reward number is reported under all three:

* **(a) all returned samples** — invalid ones scored as returned (possible for a closed-form
  reward);
* **(b) PB-valid only** — the mean over molecules that pass;
* **(c) invalid scored at the null value** — the harshest, and the convention closest to TFG-Flow's
  (which scores unconvertible molecules 0 with no epsilon guard).

They answer different questions and they can disagree in sign. `multimfm.geom.fingerprint.score`
already stores both `tanimoto_real_mean` and `tanimoto_real_mean_all`, so the machinery exists.
**The claim requires the sign to be the same under all three** (F7). This single requirement would
have prevented the "our base beats their guided number" comparison, which compared our
valid-molecules-only mean against their all-molecules mean and had to be withdrawn.

---

## 7. Power, sample size, pairing, and selection

### 7.1 Sample size

The published Table 3 was **n = 64, one seed, best of a four-point grid**, and it **reversed at
n = 4,096 x 3**. That is the specific failure this section prevents.

Procedure:

1. **Pilot on the held-out pool** at n = 256 to measure `sigma_Delta` for the *actual primary
   contrast* (method minus reference baseline, paired by target). Do not assume it; the paired SD
   depends on how much noise the two methods share.
2. **Choose `delta_min` as a fraction of the measured headroom** (Gate A), not out of the air. A
   defensible default: 10% of `C_gen(64) - U`. State it in reward units *and* as a percentage of
   headroom, so a reader can judge whether the effect is worth having.
3. **n per seed from the standard two-sided formula:**
   `n = (z_{0.975} + z_{0.80})^2 * sigma_Delta^2 / delta_min^2 = 7.85 * sigma_Delta^2 / delta_min^2`.

   Calibration from the recorded FP2 runs, where the paired per-target SD of the gain was
   `sigma_d ~ 0.055-0.072` (SEM 0.0033-0.0054 at n = 256; SEM 0.00065 at n = 12,288):

   | delta_min | n per seed at sigma_d = 0.07 (one-sample) | n if sigma_Delta = sqrt(2) x sigma_d |
   |---|---|---|
   | 0.020 | 96 | 193 |
   | 0.010 | 385 | 770 |
   | 0.005 | 1,539 | 3,078 |
   | 0.002 | 9,617 | 19,234 |

   **Read the second column.** A method-versus-baseline contrast is not the same as a
   guided-versus-unguided paired gain: unless the two methods share their noise, the difference
   variance is roughly doubled. Pair where you can, and measure `sigma_Delta` rather than deriving it.

4. **Minimum: n = 4,096 targets per seed, 3 seeds.** This is what it took for the completed FP2
   evaluation to resolve -0.00031 +- 0.00065, it is what the repo's shard infrastructure is already
   sized for (3 seeds x 8 shards x 512 in Slurm array 49274410), and it is the n at which the n = 64
   result reversed. A pilot may be smaller; **no reported number may be.**
5. **Validity needs its own n.** A 2 pp validity gate at 88% base rate needs roughly
   `n >= 0.88 x 0.12 x (1.645 + 0.84)^2 / 0.02^2 ~ 1,630` per arm for 80% power one-sided —
   comfortably inside 4,096, and far outside the n = 24 CPU pilot that made a 30 pp collapse look
   survivable.

### 7.2 Seeds and pairing

* **>= 3 seeds**, and **the sign of the primary effect must agree across all of them** (F5).
  Pooling across seeds hides exactly the instability that broke Table 3. Report per-seed numbers in
  the appendix, always.
* **Pair on the target.** Every method sees the same target list in the same order. The analysis
  unit is the target; CIs come from the distribution of per-target differences (t or bootstrap over
  targets), never from pooling molecules across targets as if independent.
* **Share the initial noise wherever the method admits one.** `run_steering` already draws
  `init_state` once and runs the unguided and guided arms from it after re-seeding, which is why its
  gains are exact within-run paired differences. best-of-N cannot share noise across all N draws;
  register that exception (form item 4.4) rather than pretending otherwise.
* **Seeds are fixed in the form before running**, and a seed that crashes is rerun, not dropped.
  (One recorded replicate has 3 shards instead of 4 because a seed crashed; that asymmetry has to be
  disclosed whenever it happens.)

### 7.3 Hyperparameter selection

The rule that makes a comparison fair, in four parts:

1. **Equal tuning budget.** Every method, including every baseline, gets a grid of the **same
   cardinality**, chosen by the same pilot procedure. A method tuned over 17 points against
   baselines tuned over 1 is not a comparison.
2. **Selection on a disjoint pool.** Tuning runs draw targets from the held-out selection pool
   (`--holdout-n`, `--holdout-seed`, `--tune-on-holdout`); the evaluation excludes that pool by
   construction. Already implemented.
3. **Freeze before evaluating.** The pilot writes the winners to a frozen file and the evaluation
   reads only that file, so a late-finishing pilot shard cannot change the configuration mid-sweep.
   `results/dna/fig4/PROVENANCE.md` documents this exact mechanism
   (`plan.frozen_from_pilot` -> `frozen_hyperparameters.json`, with `pilot_choice.json` recording
   the full grid and the tree the ranking came from). Reuse it.
4. **The frozen point is the only point reported as a result.** The full grid goes in an appendix
   table. Best-of-grid is not a result (R0.2). The pilot may run on a different device from the
   evaluation, because it contributes no plotted number — but the evaluation panel is device-pure
   (R0.5).

### 7.4 Claim tiers

Register one in advance (form item 6.1) and report whichever the data support:

* **C1, uniform dominance** — better than every baseline at every budget in the shared range. This
  is what "dominance" means and it is the hardest to get. Note that on DNA the method's cheapest
  setting sat *below* every baseline's cheapest point on the published axis, producing `NaN`
  comparisons that read as an uncontested win; on the corrected axis the point moved inside the
  baselines' range and the win became real and measurable. **Make sure the budget ladders overlap**,
  or the cheapest point is uncontested by construction and cannot support C1.
* **C2, regime win** — better below a crossover, with **the crossover reported as a number** and the
  interpolation method stated. This is the honest tier for most real results, and it is what the DNA
  result became: "Feynman-Kac overtakes from 824 NFE" had to become "from about 430
  forward-equivalent NFE".
* **C3, parity or no claim** — report it (R0.1).

Reporting C1 when the data support C2 is a deviation and is logged as one.

---

## 8. The falsification criteria, stated once

The claim registered in form item 6.1 is **falsified** if any of the following holds. These are
evaluated in order and the first that fires determines the reported outcome.

| # | criterion | consequence |
|---|---|---|
| **F1** | any Gate 0 check (§2) fails | task abandoned; the gate measurements are written up as a benchmark result |
| **F2** | `Delta <= 0` at the primary budget `B*`, **or** the upper 95% bound on `Delta` lies below `delta_min` | no advantage; report C3 |
| **F3** | the validity gate (§6.2) fails for the method under test | the reward number is a diagnostic, not a result; no claim |
| **F4** | the raw gain is significant but the derangement-controlled `Delta` is not | the gain is a population shift, not target-specific; no claim |
| **F5** | the sign of `Delta` differs across the registered seeds | unstable; report the per-seed numbers and claim nothing |
| **F6** | the advantage holds at `k = 0` but not at the measured `k` | the claim is an artifact of the compute axis; it may only be restated as "at equal forward passes, not charging for backward", which is not a dominance claim |
| **F7** | the sign of `Delta` reverses between averaging conventions 6.5(a), (b), (c) | the result is an artifact of the validity convention; no claim |

**And the converse, stated so that it is equally binding:** if none of F1-F7 fires, the claim stands
at the tier the data support, and the numbers are published as they are — including the wall-clock
panel, including the appendix grid, including the baselines' own error bars, and including every arm
that ran.

---

## 9. What gets reported, in what shape

One table and two figures, all pre-specified:

**Table (the main result).** One row per (method, budget, seed-pooled), columns:

`method | knob | nfe_gen | nfe_gen_bwd | axis_k(measured) | axis_k=0 | seconds/mol | seconds/valid mol | R_own (a) | R_own (b) | R_own (c) | S = R_own - R_perm | Delta vs reference | PB-valid % | conversion % | guide-oracle gap`

No number appears without its PB-valid rate in the same row (form item 5.3).

**Figure 1.** Reward (or error) against `axis_k` at the measured k, one curve per method, error bars
from the per-target bootstrap, with the crossover marked and its numeric value in the caption. An
inset or companion panel repeats it at `k = 0` so the size of the axis correction is visible.

**Figure 2.** The same on wall-clock seconds. Published whatever it shows (§5.3).

**Appendices.** The full hyperparameter grid for every method; per-seed tables; the Gate 0
measurements; the backward-cost measurement; the ESS diagnostic for branching baselines; the
deviation log.

---

## 10. Cost

Rough, from recorded runs, for planning only.

* **Gate 0**: floor + ceiling + order statistics are CPU and cost ~1 hour
  (`scripts/geom_rescue/fp2_population.py` did the FP2 version in ~4 min on one core). The unguided
  pool at M = 4,096 x 3 is the one GPU component and it doubles as the best-of-N baseline.
* **Backward-cost measurement**: one short GPU job, ~10 minutes.
* **Guided evaluation**: recorded GEOM guided arms ran **n = 256 in 8-10 GPU-minutes** at
  `sample_steps 100`, `mfm_steps 10`, `value_samples 4`, both arms — about 2.2 GPU-seconds per
  molecule. So **n = 4,096 x 3 ~ 7-8 GPU-hours per guided configuration**, which matches the
  recorded shape of the completed evaluation (3 seeds x 8 shards x 512, Slurm 49274410).
* **Baselines at matched budget** cost comparable generative forwards by construction, so budget
  roughly the same per configuration, minus the backward.
* **A 5-method x 3-budget x 3-seed evaluation is therefore of order 100 GPU-hours**, plus
  PoseBusters on CPU for every returned molecule in every arm.
* Shard it (512 per task), write per-shard JSON, skip-if-exists, and prefer many short jobs; the
  brief's queue guidance applies.

---

## 11. Deviation log

| date | deviation from the committed form | reason | effect on the claim |
|---|---|---|---|
| | | | |

---

## 12. Sources for every empirical number quoted above

* Table 3's n = 64 / one seed / best-of-four provenance, the 4,096 x 3 reversal (-0.00031 +-
  0.00065), the random-pair floor (0.2164 all-pairs, 0.2195 for our pairing, 0.2099 in TFG-Flow's
  size regime), the ceiling (median best-of-64 0.4127, best-of-4096 0.6350), the composition exploit
  (+0.1005 +- 0.0054 with PoseBusters 87.9% -> 15.2%), the random control (0.367 validity, -0.0139
  on the metric), the interpolant annihilation, the schedule census (51 of 100 steps at t >= 0.9),
  and the n = 24 CPU pilot that looked survivable: `docs/geom_rescue_attempt.md` and
  `results/geom/rescue/`.
* The compute-axis findings — `k = 1.1` measured on DNA, 2.89-3.00x on QM9, crossover 582 -> 428,
  the 928,263-parameter QM9 regressor at 1.30-1.39x a base forward, the backward-counter units bug,
  and the absence of any wall-clock winning regime: `docs/nfe_accounting_audit.md` and
  `results/shared/nfe_audit/`.
* The baseline implementations, the pilot/freeze mechanism, the device-purity rule and the Panel B
  numbers: `src/dmfm/benchmarks/search.py`, `src/multimfm/baselines.py`,
  `results/dna/fig4/PROVENANCE.md`.
* The guide/oracle split convention and the selection-bias result: `docs/ss_guide_oracle_gap.md`.
* TFG-Flow's reported values, protocol and released checkpoints: `results/geom/tfgflow_reference.json`.
* The validity definition: `configs/geom/posebusters_geom.yaml`,
  `src/multimfm/geom/check_pb_validity.py`.
* The GEOM sampler configuration: `checkpoints/geom/teacher/hydra/config.yaml`,
  `src/multimfm/geom/steering.py`, `src/multimfm/glass_guidance.py`.
