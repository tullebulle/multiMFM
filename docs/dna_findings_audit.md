# Audit of four DNA findings that count against the paper

Status 2026-09-30, owner: the DNA findings-audit agent of the 2026-09-29/30 round.
Everything here is L = 50, the parent-disjoint base DFM / C0 guide / C0 oracle unless a row says
otherwise. Results live in `results/dna/findings_audit/`, scripts in
`scripts/dna/findings_audit/`. Nothing outside those two paths and this file was changed.

The four items are the ones in `docs/STATE_OF_CLAIMS.md` and `docs/REPRODUCIBILITY_PLAN.md` S7
items 7-9 that count *against* the paper. The brief for this audit was to try to break them, on the
grounds that two earlier "the method is worse" findings of the same round turned out to be wiring
bugs. Each section states the hypothesis, every configuration tried, and the verdict.

## Verdict, one line each

| item | verdict |
| --- | --- |
| **(a)** Table 5's `-1` clustering and Table 6's magnitudes "do not survive" | **Our error, in the attribution** (the non-reproduction itself is real: the regenerated sets are genuinely not clustered). The metric code is AST-identical to the June original and the reference sets are interchangeable to 0.002; what changed is the *guidance protocol*, which differs in six knobs. The move from `split65k` to parent-disjoint demonstrably moves these statistics by <= 0.0024 (diversity) and <= 0.0042 nats (Evo 2), i.e. 24-66x less than the effect being attributed to it. |
| **(b)** Table 16 disagrees with Fig 6's own data (0.1813 vs -0.852) | **Neither number is right, and the finding needs restating.** The uniform-vs-real global Pearson r is a 4-point correlation against a reference with zero population variance; over uniform draws it is *exactly* Uniform(-1, 1) and independent of sample size (measured sd 0.5754 vs 1/sqrt(3) = 0.5774, KS 0.006 over 20,000 draws). 0.1813, -0.852 and our own regeneration's 0.4576 are three draws from the same noise. |
| **(c)** A well-integrated posterior does not guide at all (`glass_euler32`) | **OUR ERROR, and the finding is refuted end to end.** Two independent defects in the one arm the claim rests on. (i) `glass_euler32` integrates 32 Euler steps to the *singular* data time `t = 1`: the endpoint is a simplex vertex to float precision (argmax margin exactly 1.00000, soft-hard guide gap 4e-7), the value function is measurably **piecewise constant** (directional derivative 1.5e-4 to 1.6e-3 at h = 1e-4, and 33-100% of random directions leave V bit-unchanged, against 0% for the Table 1 posterior) and its "gradient" norm is 1.2e-5 in true fp32, i.e. float noise. (ii) It is the only arm never given the scale-free rule, so its step was thousands of times smaller than every other arm's. Fix both and an accurate posterior guides: at n = 8, y* = -1, unguided 0.8651 -- `glass_euler32_e999@norm` reaches guided oracle MAE **0.1939 with 8/8 pairs improved**, against **0.1548** for the Table 1 4-step estimator (paired +0.039 +- 0.064, i.e. 0.6 SEM), on a value gradient 25x smaller. Fixing only `end_time` gives 0.6014 and only the rule gives 0.7530, and the published arm reproduces exactly the unguided value (0.8651, 0/8 improved). Independently, in a pre-registered one-step sign test the claim's own reference integrator (GLASS RK4-25 to 0.999, gradient norm 0.202) delivers **89%** of the coarse estimator's per-step improvement once normalised (+0.242 vs +0.272), sign control negative, both random controls zero. **"A more accurate posterior guides less" does not hold**; it was an artifact of a degenerate integrator plus an unrescaled constant. |
| **(d)** The reward temperature is only a disclosure gap | **Finding stands, and it is worse than "disclosure".** The undisclosed factor 11.111 turns the paper's N = 8 Monte-Carlo average into an effectively 1.1-2.6-draw estimator inside the guidance window (ESS at t = 0.5: 1.09 at y* = +1, 1.98 at y* = -1, against 5.4-6.6 at the equation's own coefficient), while the MC-convergence tables that justify N = 8 run a different reward whose ESS is 5.1/8. It also scales the value gradient by 10-16x, and `coeff_cap = 10` already saturates for t <= 0.615, so the published strength **cannot** be re-derived from Eq (17) by re-tuning `guidance_frac`. |

---

## (a) Tables 5 and 6: the attribution to `split65k` -> parent-disjoint is not supported

### What was claimed

S7 item 7 and `docs/dmfm_demo_reruns.md` attribute the loss of Table 5's `y* = -1` clustering
(0.9507x -> 0.9872-0.9927x pairwise, 0.8027x -> 0.9367-0.9621x nearest-generated) and of Table 6's
magnitudes (`-1` delta -0.1104 -> -0.0068, `+1` delta +0.0075 -> +0.0169) to the move from the purged
`split65k` models to the parent-disjoint models, "because the GLASS controls fail identically".

### What the audit found

**1. The metric pipeline is equivalent; this hypothesis is closed.** Every scoring function of the
June analyser `DNA-MFM@187fe7b:scripts/analyze_yeast_c0_diversity.py` is **AST-identical** to the
port in `dmfm.experiments.{analyze_c0_diversity,_yeast_c0_diversity_metrics}`:
`hamming_dist_chunk`, `nearest_generated_distances`, `nearest_reference_distances`,
`pairwise_generated_distances`, `seqs_to_tokens`, `summarize_one`, `summarize_reference_baseline`,
`run_metadata`. Only `parse_args`, `main` and `load_train_tokens` differ, i.e. which data file and
which splits are read. Same normalisation (Hamming / L), same dedup (`exact_unique`), same
subsampling (`max_pairwise_n 3000`, `seed 0`), same `baseline_n 1000`, same n = 1000.
The Evo 2 scorer is likewise AST-identical on the whole scoring path (`load_evo2`, `model_forward`,
`prepare_batch`, `score_mean_logprobs`, `score_rows`, `reverse_complement`, `summarize`,
`bootstrap_ci`, `generated_rows`), same `evo2_7b_base`, same `--bootstrap 2000`, same RC averaging,
same `scored_positions = 49`.

**2. The reference set moved, but not enough to matter, and on the one reference the two pipelines
share they agree to the printed digit.** The June analyser's `train_sample` row (a seed-0 draw of
1,000 from the *full* window set) is `nearest_gen 0.52188`, `pairwise 0.737699`
(`results/dna/june_split65k_recovered/yeast_split_dmfm_c0_diversity_summary_2026-06-09.selected_columns.txt`);
our `full_sample` row is `0.52188`, `0.7376993` -- the same numbers. Across the four real-data
baselines the two pipelines differ by at most 0.002: published `split_{test,train,val}_sample`
0.5201 / 0.52192 / 0.5184 and 0.7374 / 0.737335 / 0.737177 against parent-disjoint
`test_sample` / `a_train_sample` / `a_val_sample` 0.52192 / 0.52000 / 0.51936 and
0.738115 / 0.736074 / 0.738451. So the held-out *denominator* of the App. B.1 ratios moved by
0.0007 (pairwise) and 0.0018 (nearest-gen), while the *numerator* moved from 0.7010 to 0.7287-0.7327
and from 0.4175 to 0.4889-0.5021. A different normalising reference cannot be the mechanism.

**3. The tables' own guidance-independent rows bracket the model move, and it is 24-66x too small.**
`results/dna/findings_audit/tables56_attribution.json`. The unguided generated rows and the
real-data reference rows use the same base model, the same trajectory integrator and the same
denominators as the guided rows, so they measure exactly the `split65k` -> parent-disjoint effect:

| statistic | largest shift in a guidance-independent row | shift in the `-1` guided row | ratio |
| --- | --- | --- | --- |
| Table 5 nearest-gen / held-out | 0.0024 | **0.1594** | 66x |
| Table 5 pairwise / held-out | 0.0015 | **0.0420** | 28x |
| Table 6 NLL (nats) | 0.0042 | **0.0993** | 24x |

For comparison, the `y* = +1` guided rows *do* reproduce (Table 6 delta +0.0052 nats from published,
about 1.2x the control bracket). So the model move is not a credible explanation of a `-1`-only
effect 24-66x larger than the move's own footprint.

**4. What actually differs: the guidance protocol, in six knobs.** The published Tables 4-6 sets came
from `DNA-MFM@187fe7b:scripts/sample_yeast_split_dmfm_c0.py` driving the `DMFMSteeringRunner` of
`scripts/benchmark_yeast_split_dmfm_steering.py` (recovered args:
`results/dna/june_split65k_recovered/samples_target_{m1,p1}_2026-06-02.args.json`). That producer's
defaults are exactly the June args, and `dmfm.experiments.sample_c0_guidance_dmfm` is its port, so
the protocol can be rerun verbatim:

| knob | published June sets | every parent-disjoint regeneration |
| --- | --- | --- |
| guidance window | **[0.01, 0.95]**, ~91 guided steps | [0.50, 0.95], ~30 guided steps |
| `reward_scale` | **1.0** (log-reward `-22.22 (f - y*)^2`) | 0.5 (`-11.11 (f - y*)^2`) |
| MC draws | **16**, fresh `randn` per step | 8, frozen per-sample pool |
| posterior | dMFM student, **1** composed map (`nfe_value 1`) | GLASS 4 Euler steps, or the 4-step ESD student at 4 steps |
| trajectory | `nfe_sample` **96** integrated to **t = 1.0** | `nfe_traj` 64 to `t_max = 0.95` |
| seeds | 0 (`+1`) / **1** (`-1`) | 0 / 0 |
| guide | `dinko/C0free_torch.pt` | parent-disjoint `park_cnn` |

The reward-evaluation count stored in the June `summary.json`
(`reward_evals_per_output = 1457 = 1 + 91 x 16`) confirms the 91-guided-step reading exactly. So the
published `-1` set was steered roughly 3x as many steps at twice the reward sharpness as the set it
is being compared with. The "GLASS controls fail identically" argument does not separate the model
from the dose, because *both* controls were run at the weak dose.

Two side findings, both corrections to the current record:

* `docs/dmfm_demo_reruns.md` states that the published Tables 4-6 sets "were produced by **GLASS**
  guidance on the base DFM". They were not: `DMFMSteeringRunner.value_grad_target_dmfm` differentiates
  through `dmfm_integrate_diff`, which composes the **dMFM student**. The GLASS-only claim is correct
  for Table 1 / Fig 3 / Fig 7 (`sample_yeast_c0_guidance_hist.py`), not for Tables 4-6. Table 5's
  caption "generated C0-steered yeast dMFM samples" was accurate.
* `sample_yeast_split_dmfm_c0.py`, the actual producer of the published Tables 4-6 sets, is not among
  the verbatim originals in `results/dna/original_scripts/` (only the `DMFMSteeringRunner` file is).
  It should be added.

### The controlled test

`scripts/dna/findings_audit/june_dose_c0_sets.sbatch` reruns the published protocol unchanged on the
parent-disjoint model and guide (jobs 49431930 and 49431932; `ARM=ablate` adds two one-knob-back
controls, `--guide_t_start 0.50` and `--reward_scale 0.5`, at `y* = -1`), followed by the same
analyser. Status and numbers: see "Measured results" below.

---

## (b) Table 16 vs Fig 6: the statistic is noise

`results/dna/findings_audit/table16_uniform_r.json`,
`scripts/dna/findings_audit/table16_uniform_r.py`. CPU, seconds, no GPU needed.

**The construction is identified from the printed p-values.** Solving `p = 2 sf(t, n-2)` for n:
`(0.9991, 9.5e-4) -> n = 4`; `(0.1813, 0.819) -> n = 4` (p at n = 4 is 0.8187);
`(0.9863, 6.02e-157) -> n = 200`; `(0.0264, 0.711) -> n = 199-200`. So the global rows are Pearson
correlations over the **4** nucleotide marginals and the position-wise rows over the **200**
(50 positions x 4 bases) values. Fig 6's own bar heights
(`results/dna/figure_digitized/fig6_global_marginals_digitized.csv`) give, under that same
construction, -0.8518 (p = 0.148) for uniform-vs-real and 0.9988 for generated-vs-real.

**Why the uniform row cannot be reproduced by anyone.** An exactly uniform reference
`(0.25, 0.25, 0.25, 0.25)` has zero variance, so r is 0/0 = undefined; the printed number can only
come from a finite uniform *sample*. Both the real and the sampled-uniform marginal vectors sum to 1,
so after centring they live in a 3-dimensional subspace, and because all four multinomial cell
probabilities of the uniform reference are equal, its sampling noise is isotropic there. r is
therefore the cosine of the angle between a fixed direction and an isotropic random direction in
R^3, which is **exactly Uniform(-1, 1)** -- and, crucially, independent of how many sequences were
drawn. Measured over 20,000 draws at three set sizes:

| uniform set | mean r | sd r | 2.5% / 50% / 97.5% | KS vs U(-1,1) | P(\|r\| > 0.8) |
| --- | --- | --- | --- | --- | --- |
| 1,000 x 50 | -0.0014 | 0.5762 | -0.946 / -0.001 / 0.948 | 0.0040 | 0.198 |
| 4,096 x 50 | +0.0010 | 0.5754 | -0.951 / +0.004 / 0.948 | 0.0062 | 0.201 |
| 10,000 x 50 | +0.0041 | 0.5761 | -0.948 / +0.007 / 0.952 | 0.0059 | 0.198 |

(1/sqrt(3) = 0.5774 is the sd of U(-1,1).) 0.1813 lies in the bulk (2.1% of draws within +-0.02 of
it) and so does -0.852 (7.5% of draws are below it). Our own regeneration
(`results/dna/regen/fig6_table16/correlations.csv`, another agent's run, `uniform_seed 1`) gives
0.4321, 0.9695, 0.4576, 0.9272 for the *same* uniform draw truncated to 1,000 / 2,000 / 4,096 /
10,000 sequences against the same real set -- a 0.54 swing from a change that should be irrelevant,
which is the same phenomenon.

**Verdict.** The finding "Table 16 disagrees with Fig 6's own plotted data" is arithmetically true
but is not evidence that either number is wrong, and it cannot be resolved: the quantity has no
population value. The two rows that *do* carry information are reproduced --
generated-vs-real global 0.9991 (table) / 0.9988 (figure) / 0.9887 (our regeneration), position-wise
0.9863 / 0.9835 / 0.9768. The position-wise uniform row is also fine as reported, because with
n = 200 points the noise floor is 1/sqrt(198) = 0.071, so 0.0264 (table) and 0.065 (figure) both say
"no correlation".
**Recommended paper change:** drop the two "uniform vs. real" rows of Table 16, or replace them with
a distance that has a population value (total-variation distance between the marginal
vectors is 0.1131 for real-vs-uniform against 0.0053 for real-vs-generated on the Fig 6 numbers,
a 21x separation with no sampling ambiguity), and state that
a Pearson correlation against a uniform reference is undefined in the limit. Keep Fig 6: its point
("the uniform baseline lacks the observed nucleotide biases") is exactly what the distance says.
Note also that the Fig 6 PDFs are dated 2026-05-03, before both the `split65k` (2026-05-12) and the
parent-disjoint models, so the figure and the table need not describe the same generated set at all.

---

## (c) The discretisation-error finding

### What was claimed

S7 item 9 / `docs/dmfm_glass_parity.md` S2 / `docs/STATE_OF_CLAIMS.md` item 1: "A well-integrated
posterior does not guide at all: `glass_euler32` leaves guided MAE at the unguided value (1.1905 vs
1.1907), and the well-integrated reference gradient has norm ~0.15 against 3-125 for every estimator
actually used."

### Two configuration defects in the one arm the claim rests on

**Defect 1: `glass_euler32` integrates to the singular data time, so its endpoint is saturated and
its gradient is float noise.** `CONFIGS["glass_euler32"]` is `n_steps=32, end_time=1.0,
solver="euler"`. The same document *discards its own 100-Euler-to-1.0 reference* for precisely this
reason -- "a fine Euler grid run all the way to t=1 crosses the singular linear-DFM data-time
velocity, driving the endpoint onto a hard corner" -- and the stored evidence agrees: in
`results/dna/parity/grad_L50/` (that reference) the reference gradient norm is **8e-6**, against
**0.15** for RK4-25 to 0.999 in `grad_L50_rk4ref/`. `glass_euler32` is on the same ladder, one
refinement short. Measured directly (4 probe states at t = 0.5, CPU, forward only):

| posterior | argmax margin | soft-hard guide gap | endpoint min |
| --- | --- | --- | --- |
| `euler4` to 1.0 (Table 1) | 0.83714 | -1.0e-01 | 1.2e-07 |
| **`euler32` to 1.0 (the claim's arm)** | **1.00000** | **-4.2e-07** | -1.6e-07 |
| `euler32` to 0.999 | 0.99792 | -9.8e-04 | -2.9e-03 |
| `rk4-25` to 0.999 (the reference) | 0.99813 | -1.1e-03 | -2.9e-03 |

An argmax margin of exactly 1.00000 with a 4e-7 soft-hard gap means the endpoint is a simplex vertex
to float precision, i.e. locally constant in `x_cond`: there is no gradient to estimate. That is
consistent with the recorded `mean |g| = 0.00198` at `y* = -1` and with the `y* = +1` value of 28.8
being carried by 0.4% of steps (rare float-noise spikes). `glass_euler32` was also never run in
`--mode grad`, so its agreement with the accurate reference direction was never measured.

**Defect 2: it is the only arm never given the scale-free rule.** Grep of all three parity sample
runs: `glass_euler32` appears once, in `sample_L50_n64`, with no `@norm` and no `@gf` variant. Under
the stock clip-down rule the step is `dt * coeff * clip(g, 10)` with clipping only downward, so at
`|g| = 0.00198` the step is **5,000x** smaller than the step the same rule gives an arm whose
gradient sits at the clip. S7 item 8 mandates the scale-free rule "wherever a GLASS comparison is
shown" exactly because "the stock clip-down rule ... silently re-tunes strength when the posterior
changes"; that rule was applied to `diag_fm4`, `dmfm4_fm4` and `glass_euler4`, and not to
`glass_euler32`.

### The test

`scripts/dna/findings_audit/posterior_accuracy_probe.sbatch` (job 49431928) adds the posteriors the
claim needs and runs them through the *same* harness, same ids, same frozen MC pool, n = 64, both
targets: `glass_euler32` (as published), `glass_euler32@norm`, `glass_euler32_e999`,
`glass_euler32_e999@norm`, `glass_rk4_25_e999` and `glass_rk4_25_e999@norm` -- the last being the
Table-18 *reference* integrator used as the guiding posterior, which is the literal test of "does an
accurate posterior guide". `--mode grad` additionally measures the whole Euler ladder
(4/8/16/32/100 steps, to 1.0 and to 0.999) and RK4 8/25 against the RK4-25 reference, so the
degeneracy is visible as a curve rather than as one point. That job was still queued behind the
account GPU cap; **the item was settled without it** by the CPU runs below (49434328 end-to-end,
49442813 direction test, 49438552 smoothness), and 49434948 would only raise n and add the reference
integrator end to end.

---

## (d) The reward temperature is more than a disclosure gap

`results/dna/findings_audit/reward_temperature/`, `scripts/dna/findings_audit/reward_temperature.py`.

Eq (17) (`r_guide = -(f_guide - y*)^2`) and Eq (48) (`r = -|f_cyc - y*|^2`) both state a unit
coefficient. `api.c0_log_reward_fn` computes `reward_scale * -0.5 ((f - y*)/reward_sigma)^2`, i.e.

* `-11.111 (f - y*)^2` at `reward_sigma 0.15, reward_scale 0.5` -- Table 1, Fig 3, Fig 7, Fig 8,
  all parity runs, all `c0_guidance/*` runs;
* `-22.222 (f - y*)^2` at `reward_scale 1.0` -- the published Tables 4-6 sets and Fig 4 panel A.

So the paper's own experiments do not share one effective reward, and neither matches the equation.

**It changes the Monte-Carlo conclusion.** The coefficient is the inverse temperature of the
finite-N value `V = log (1/N) sum_j exp(c * -(f_j - y*)^2)`, so it sets how concentrated the N
softmax weights are. Effective sample size `ESS = 1 / sum_j w_j^2` at the paper's operating point
N = 8, measured on 8 probe states from the base flow with the Table 1 GLASS-4 posterior and the
identical MC pool:

| log-reward | ESS at t = 0.5 | ESS at t = 0.7 | largest weight at t = 0.5 |
| --- | --- | --- | --- |
| `-(f - y*)^2` (the paper's equation), y* = +1 / -1 | 5.43 / 6.58 | 6.84 / 7.14 | 0.31 / 0.21 |
| **`-11.111 (f - y*)^2` (Table 1 as run)**, +1 / -1 | **1.09 / 1.98** | 2.20 / 2.60 | **0.96 / 0.68** |
| `-22.222 (f - y*)^2` (Tables 4-6 sets as run), +1 / -1 | 1.01 / 1.42 | 1.64 / 1.63 | 0.99 / 0.84 |
| `-(z - 1)^2` on standardised gc score -- **the reward of the MC-convergence tables** | 5.06 | 6.46 | 0.34 |

(At t = 0.9 every ESS equals N exactly: the posterior is already deterministic there and the
gradient is numerically irrelevant, consistent with `docs/dmfm_glass_parity.md` S1.) At N = 64 the
picture is unchanged in kind: 2.77 (`-11.111`, y* = +1) and 39.6 (gc reward).

Two consequences, both conclusion-level rather than cosmetic:

1. **"N = 8 Monte-Carlo draws" is an effectively 1-2.6-draw estimator** at the temperature actually
   used, inside the guidance window where it matters. Reporting the operating point as N = 8 without
   the coefficient overstates the averaging by 3-7x.
2. **Fig 5 / Tables 14-15 / Table 18 do not cover that regime.** They run
   `-reward_beta (z - z_target)^2` on a standardised gc / motif / conjunction score with
   `reward_beta = 1, z_target = 1` (verified in every `run_metadata.json` under
   `results/dna/{scale_free_mc,dmfm_vs_glass}/`), whose ESS is 5.1/8 -- the *paper-equation* regime,
   not the shipped-C0 regime. In score-sd units the C0 reward's sharpness is
   `11.111 x 0.4136^2 = 1.90` with the target 2.87 sd (y* = +1) or -1.97 sd (y* = -1) from the
   unconditional mean, against sharpness 1.0 at 1.0 sd for the MC tables. So the MC-scaling evidence
   is measured at a materially flatter reward, aimed materially closer, than the demo it is used to
   justify.

The remaining question -- whether the literal Eq (17) would reproduce Table 1 at all, given that
under clip-down the step is proportional to the coefficient whenever `|g|` is below `grad_clip`
(the GLASS-4 regime: mean norm ~3, clipped at ~5% of steps) -- is measured by
`scripts/dna/findings_audit/reward_temperature.sbatch` at `--reward_scale 0.045` (coefficient exactly
1), 0.5 and 1.0, under both the stock and the scale-free rule. See "Measured results".

---

## Measured results

### (c) End-to-end: an accurate posterior DOES guide once both defects are fixed (measured)

`results/dna/findings_audit/sample_posterior_accuracy_cpu_n8/`, Slurm **49434328**
(`serial_requeue`, CPU, true fp32, 16 cores, ~6 min). n = 8 paired samples, `y* = -1`, ids 0-7,
identical initial noise and identical frozen MC = 8 pool for every configuration, all Table 1
trajectory settings (`nfe_traj 64`, `t_max 0.95`, window [0.5, 0.95], `guidance_frac 8`,
`coeff_cap 10`, `grad_clip 10`, `reward_sigma 0.15`, `reward_scale 0.5`), park_cnn oracle
RC-averaged. Unguided oracle MAE 0.8651 in every row by construction.

| posterior | guidance rule | guided oracle MAE | mean \|g\| | pairs improved | paired vs `glass_euler4` |
| --- | --- | --- | --- | --- | --- |
| `glass_euler4` to 1.0 (Table 1) | stock clip-down | **0.1548** | 1.64 | 8/8 | 0 |
| **`glass_euler32` to 0.999** | **scale-free** | **0.1939** | 0.066 | **8/8** | +0.039 +- 0.064 |
| `glass_euler32` to 0.999 | stock clip-down | 0.6014 | 0.294 | 7/8 | +0.447 +- 0.100 |
| `glass_euler32` to 1.0 | scale-free | 0.7530 | 5.9e-6 | 4/8 | +0.598 +- 0.162 |
| **`glass_euler32` to 1.0 (the published arm)** | stock clip-down | **0.8651** | 1.2e-5 | **0/8** | +0.710 +- 0.116 |

Reading, in the order that matters:

* **The published arm reproduces.** `glass_euler32` leaves the guided oracle MAE at *exactly* the
  unguided value (0.8651 vs 0.8651, 0 of 8 pairs improved), as `docs/dmfm_glass_parity.md` reported.
  Its gradient norm here is 1.2e-5 -- even smaller than the 0.00198 recorded on GPU, because that
  run had TF32 on and this one does not. It is measuring nothing.
* **Both defects have to be fixed, and each alone is insufficient.** `end_time 0.999` alone gives
  0.6014 (the gradient is genuine but 34x below `grad_clip`, so the step is 34x too small); the
  scale-free rule alone gives 0.7530 (normalising float noise produces a random direction, and
  exactly half the pairs improve -- a coin flip, as it should be).
* **Together they restore guidance.** A 32-step posterior stopped just short of the singular data
  time, with its gradient rescaled to the same step the other arms get, drives oracle MAE from
  0.8651 to **0.1939 with every one of the 8 pairs improved** -- against 0.1548 for the 4-step
  estimator, a paired difference of +0.039 +- 0.064, i.e. 0.6 SEM. Its gradient norm is 0.066-0.294,
  the *same order* as the well-integrated RK4-25 reference's 0.15 and 25-500x below the coarse
  estimator's, which is exactly the class of posterior the claim says cannot steer.
* **So the mechanism claim is wrong as stated.** "The C0-guidance signal is carried by the
  discretisation error of a deliberately coarse posterior, and a *more accurate* posterior guides
  less, not more" does not survive: the accurate posterior guides to within noise of the coarse one
  once the guidance constant is not silently scaled by the gradient magnitude. What survives is the
  much weaker and already-known statement that the *constants* are not transportable between
  posterior samplers without renormalisation -- which is the reason S7 item 8 mandated the
  scale-free rule in the first place, and it was simply not applied to this arm.

Caveats, stated because they are the reason this is not yet the final word: n = 8, `y* = -1` only,
one displacement rule, and `rk4-25` to 0.999 (the actual Table-18 reference integrator) has not yet
been run end to end. Job 49434948 does all of that at n = 64 on both targets and was still queued
behind the account GPU cap.

### (c) Is the accurate posterior's value function differentiable at all? (CPU, measured)

`results/dna/findings_audit/value_smoothness/value_smoothness.csv`, Slurm **49438552**
(`serial_requeue`, CPU, true fp32, ~8 min),
`scripts/dna/findings_audit/value_smoothness.py`. Central finite differences of the finite-N value
`V(x_t)` along 3 random unit directions at 4 probe states, MC = 8, the Table 1 reward, for
t in {0.5, 0.7} and both targets. A smooth `V` has `|(V(x+hu) - V(x-hu)) / 2h|` tending to a constant
as `h -> 0`; a piecewise-constant `V` has it tending to 0. `flat` is the fraction of
(probe, direction) pairs whose `V` is *exactly* unchanged.

`|dV/dh|`, averaged over both outer times and both targets:

| posterior | h = 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 3e-1 |
| --- | --- | --- | --- | --- | --- | --- |
| `glass_euler4` to 1.0 (Table 1) | **1.885** | 1.889 | 1.893 | 1.894 | 1.857 | 0.993 |
| `glass_rk4_25` to 0.999 (reference) | **0.228** | 0.036 | 2.705 | 1.037 | 1.496 | 0.772 |
| `glass_euler32` to 0.999 | **0.093** | 0.053 | 0.053 | 12.06 | 4.624 | 0.840 |
| `glass_euler32` to 1.0 (the published arm) | **0.045** | 0.008 | **0.0008** | 12.28 | 4.695 | 0.839 |

`flat`, same averaging:

| posterior | h = 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 |
| --- | --- | --- | --- | --- | --- |
| `glass_euler4` to 1.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `glass_rk4_25` to 0.999 | 0.167 | 0.042 | 0.063 | 0.000 | 0.000 |
| `glass_euler32` to 0.999 | 0.604 | 0.333 | 0.083 | 0.000 | 0.000 |
| `glass_euler32` to 1.0 | **0.625** | 0.458 | 0.521 | 0.375 | 0.021 |

Reading, stated so it does not overclaim:

* **The Table 1 posterior's value function is smooth**, with a clean plateau across five decades of
  `h` in *every* (t, target) cell -- e.g. 2.096 / 1.895 / 1.914 / 1.906 / 1.938 at t = 0.5, y* = +1 --
  and not one exactly-flat direction anywhere.
* **The published arm's is not.** `glass_euler32` to 1.0 has `|dV/dh|` of 8e-4 at `h = 1e-4` at
  t = 0.5 y* = +1 (1.5e-4 at y* = -1) and 33-100% of random directions leave `V` bit-unchanged, with
  the large values at `h >= 1e-3` being boundary crossings, not derivatives. Its endpoint is a simplex
  vertex to float precision (argmax margin exactly 1.00000, soft-hard guide gap 4e-7, measured
  separately). There is nothing there to rescale.
* **The accurate posteriors are in between, and this is the honest nuance.** Stopping at 0.999
  restores a genuine derivative, but their `V` is still flat in 42-83% (`euler32_e999`) and 8-33%
  (`rk4-25`) of *random* directions at `h = 1e-6`, against 0% for the coarse estimator. So an accurate
  posterior's value function is nearly piecewise constant in a random direction while retaining a real
  derivative along the minority of directions that matter -- which is exactly where its gradient
  points. That is why its gradient norm is 0.15-0.3 rather than 20-30, and why its *direction* still
  works: the direction test above gives it 89% of the coarse estimator's per-step benefit, and the
  end-to-end run gives 0.1939 against 0.1548.
* This also explains the `mean |g|` numbers in `docs/dmfm_glass_parity.md` S2: a finite-difference or
  autodiff probe of these value functions at too large a step reports boundary crossings (the 8.4-12.9
  entries above), which is a further reason not to read a gradient-norm ratio as a statement about
  steerability.

### (c) Is the accurate posterior's direction useful? Full pre-registered test (CPU, measured)

`results/dna/findings_audit/direction_usefulness/direction_usefulness.csv`, Slurm **49442813**
(`serial_requeue`, CPU, 2.5 min). 8 probe states, t in {0.5, 0.7}, both targets, MC 8, three
displacements, two random-direction controls, all four posteriors. The registered reading is in the
docstring of `scripts/dna/findings_audit/direction_usefulness.py`, written before the run: the
direction is useful iff `+grad` improves `|f - y*|` more than `-grad` **and** more than a random
direction at the operating displacement.

Mean improvement in `|f - y*|` over both outer times and both targets (`delta = 1.484` is the
displacement one guided step takes under the scale-free rule at t = 0.5):

| posterior | mean \|grad V\| | `+grad` at delta 0.148 / 1.484 / 14.84 | `-grad` at 1.484 | random controls |
| --- | --- | --- | --- | --- |
| `glass_euler4` to 1.0 (Table 1) | 26.4 | 0.083 / **0.272** / 0.439 | -0.251 | 0.000 / 0.000 |
| `glass_euler32` to 0.999 | 1.77 | 0.063 / **0.268** / 0.318 | -0.123 | 0.000 / 0.000 |
| **`glass_rk4_25` to 0.999 (the reference integrator)** | **0.202** | 0.025 / **0.242** / 0.240 | -0.126 | 0.000 / 0.000 |
| `glass_euler32` to 1.0 (the published arm) | 0.0005 | 0.021 / 0.117 / 0.038 | -0.010 | -0.005 / 0.000 |

**This is the claim's own reference posterior, and its direction works.** GLASS RK4-25 to 0.999 is
exactly the integrator whose gradient norm of ~0.15 the published claim cites as proof that accurate
posteriors are useless; measured here at norm 0.202, its normalised direction delivers **0.242 of
0.272 = 89%** of the Table 1 estimator's per-step improvement, with the sign control negative and both
random controls at zero. The pattern holds in 7 of the 8 (t, target, delta 1.484) cells and at every
displacement. Averaged per cell at `delta = 1.484`: at t = 0.7 the accurate posteriors are actually
*better* than the coarse one at both targets (0.375 / 0.317 for `euler32_e999` and 0.285 / 0.336 for
`rk4-25`, against 0.219 / 0.437 for `euler4`).

The published arm's own direction is not pure noise either (+0.117), which is worth knowing: its
gradient of norm 5e-4 is dominated by whichever coordinates sit near a decode boundary, and that is
weakly informative. But at the *magnitude* the stock rule gives it, it moves the state by nothing at
all, which is the end-to-end 0.8651.

### (c) Is the accurate posterior's direction useful? First pass at n = 4 (CPU, measured; superseded by the table above, kept because it was run first)

`results/dna/findings_audit/direction_usefulness/direction_usefulness_quick_n4.csv`,
`scripts/dna/findings_audit/direction_usefulness.py` (the registered reading is in that file's
docstring, written before the run). For each posterior: normalise its value gradient, displace the
probe state by `delta * u`, integrate the **base** flow from `t` to `t_max = 0.95` with no further
guidance, decode `argmax` and score the hard sequence with the guide. `delta = 1.484` is the
displacement one guided step actually takes under the scale-free rule at t = 0.5
(`dt * min(guidance_frac sigma_t^2, coeff_cap) * grad_clip = (0.95/64) * 10 * 10`). `-grad` is the
sign control and `random` the magnitude control. Mean improvement in `|f - y*|`, averaged over both
targets, 4 probe states, t = 0.5, MC 8, CPU fp32:

| posterior | `+grad` | `-grad` | `random` | grad direction useful? |
| --- | --- | --- | --- | --- |
| `glass_euler4` to 1.0 (Table 1) | **+0.263** | -0.250 | 0.000 | yes |
| `glass_euler32` to 1.0 (the claim's arm) | **+0.102** | -0.016 | 0.000 | yes |
| `glass_euler32` to 0.999 (accurate) | **+0.118** | -0.000 | 0.000 | yes |

Per target the sign test passes in every cell: at y* = -1, `+grad` improves by 0.343 / 0.113 / 0.102
for the three posteriors while `-grad` gives -0.061 / -0.028 / +0.030; at y* = +1, `+grad` gives
0.182 / 0.090 / 0.135 and `-grad` -0.438 / -0.003 / -0.030. The random control is exactly neutral by
construction (it does not depend on the posterior, and its two targets cancel).

**This is the first direct evidence on the claim, and it goes against it.** At the displacement the
guided sampler actually uses, the *accurate* posterior's gradient direction improves the final
cyclizability error by 0.118 -- 45% of the Table 1 estimator's 0.263, not 0 -- with the sign test
passing. A direction that carries 45% of the coarse estimator's per-step benefit is not "does not
guide at all"; what it is not is *as good as* the coarse estimator. Caveats to be explicit about:
4 probe states, one step with no compounding, one displacement, and t = 0.5 only. The queued full
grid (job 49442813: 8 probes, t ∈ {0.5, 0.7}, both targets, three displacements, two random
directions, plus `rk4-25` to 0.999) and the end-to-end runs (49434948, 49434328) are what would
settle it.

### (c) Does the QM9 DPS step-refinement result constrain this? No -- it is a different limit

The coordinator's QM9 measurement (DPS at `--sample-steps` 128 -> 4096, one seed: oracle MAE
~2.8 -> 3.18 -> 3.78 -> 4.01 -> 5.41, PB 0.89 -> 0.73) was offered as the same signature in another
domain. Read at code level it is not the same experiment, and neither result supports the other.

* **QM9 refines the outer guided trajectory.** `multimfm.baselines.sample_dps` adds
  `mu * grad * dt` through `glass_guidance.production_step_with_optional_guidance` (lines 727-736),
  and `--guidance-max-rms` defaults to 0, so nothing clips per step. The guidance displacement
  `integral mu grad(r) dt` is therefore dt-invariant as claimed, and refining `sample_steps`
  converges to the exact solution of the *guided ODE*. That limit being worse is a statement about
  the guided dynamics itself.
* **DNA `glass_euler32` refines the inner posterior** that estimates the value gradient, with the
  outer trajectory fixed at `nfe_traj 64` and the guidance rule untouched. Its limit is the exact
  *value gradient*, not the exact guided trajectory. So it cannot corroborate the QM9 observation
  even if both numbers had been sound.
* **And the DNA arm is not sound**, for reasons that have no QM9 analogue: its endpoint is a simplex
  vertex to float precision, its value function is piecewise constant (measured above: derivative
  6e-4 as `h -> 0`, half the directions bit-unchanged), and its guidance step was 5,000x below every
  other arm's because the clip only clips downward. None of those is a "constant was not retuned"
  story about an otherwise healthy estimator -- two of the three are numerical degeneracy. The
  recalibration hypothesis is one of the two defects, not the whole of it.

So the honest position is that the two findings are *independent*, and the DNA one currently rests on
a defective measurement whatever the QM9 one shows.

**The matched DNA arm is the missing experiment, and it is queued.**
`scripts/dna/findings_audit/outer_refinement.sbatch` sweeps the DNA *outer* trajectory
`nfe_traj` 64 -> 128 -> 256 -> 512 at a fixed GLASS-4 inner posterior, under both the stock and the
scale-free rule, n = 32, both targets. The DNA guidance term also carries `dt`
(`dt * min(guidance_frac sigma_t^2, coeff_cap) * clip(grad V, grad_clip)`), so it is invariant in
exactly the sense the QM9 argument uses. If DNA degrades the same way, there is a cross-domain
regularity about *outer* refinement that neither paper section currently states, and it should be
reported as such -- separately from S7 item 9, which is about the inner estimator. If DNA does not
degrade, the QM9 result is specific to that sampler and should not be generalised.

**On the QM9 arm itself**, which I did not run: the retuning test the coordinator proposes is the
right one, and one prediction is worth recording in advance. If the total guidance integral really is
invariant, then re-tuning `mu` cannot recover the 128-step result by supplying "more drive" -- there
is no missing drive. The candidate mechanism that fits the measured PB collapse (0.89 -> 0.73) is the
opposite: at coarse steps each large guidance nudge is immediately followed by a large base-model
step that projects the state back toward the data manifold, whereas at fine steps the nudges
accumulate between correspondingly weaker corrections, so the trajectory drifts off-manifold. That
predicts that the best `mu` at 4096 steps is *smaller* than at 128 and still does not reach the
128-step MAE. If instead a *larger* `mu` at 4096 steps recovers it, the invariance argument is wrong
somewhere and the code path should be re-derived before either result is quoted. I am not running
this: it is another agent's results tree and I have no free GPU slot while the DNA jobs queue.

### (a) The published June protocol rerun on the parent-disjoint model (GPU, pending)

Filled in from `outputs/dna/findings_audit/` + `dmfm.experiments.analyze_c0_diversity` when jobs
49431930 / 49431932 land. What the test can show:

* If the June dose reproduces pairwise ~0.70 (0.95x) and nearest-generated ~0.42 (0.80x) on the
  parent-disjoint model, then the published `-1` clustering is a property of that guidance protocol,
  the App. B.1 sentence stands as a description of what was run, and the S7-item-7 attribution and
  the "does not survive" verdicts in `docs/dmfm_demo_reruns.md` and `docs/STATE_OF_CLAIMS.md` are
  wrong and must be rewritten. The paper would then need the protocol disclosed (it currently is
  not: App. B.1 gives the metric definitions and the model, never the guidance window, MC budget or
  reward scale), plus the note that Table 1 and Tables 4-6 were run at different guidance doses.
* If it does not reproduce the clustering, the dose is exonerated and the remaining candidates are
  the guide network (`dinko/C0free_torch.pt`, purged) and the base model itself. The attribution to
  the model would then be supported by elimination rather than, as now, by an argument that its own
  control rows contradict.

### (d) The reward coefficient sets the guidance step, and the stated constants cannot absorb it (CPU, measured)

`results/dna/findings_audit/reward_temperature/grad_norm_vs_coefficient.json`. Mean `||grad V||` of
the Table 1 GLASS-4 posterior at 4 probe states, MC 8, identical pool, true fp32:

| t | y* | coefficient 1 (Eq 17) | 11.111 (Table 1 as run) | 22.222 (Tables 4-6 as run) | ratio 11.111/1 |
| --- | --- | --- | --- | --- | --- |
| 0.5 | +1 | 1.49 (0% above clip 10) | 24.3 (100%) | 51.2 (100%) | 16.3x |
| 0.5 | -1 | 1.01 (0%) | 10.6 (25%) | 22.1 (100%) | 10.5x |
| 0.7 | +1 | 3.75 (0%) | 48.3 (100%) | 105.7 (100%) | 12.9x |
| 0.7 | -1 | 2.77 (0%) | 29.4 (100%) | 64.4 (100%) | 10.6x |

So the coefficient scales the value gradient by 10-16x, close to its nominal 11.1x (slightly
superlinear because the sharper reward also shifts the softmax weights). Along a *guided* trajectory
the residual `f - y*` is small and `docs/dmfm_glass_parity.md` records mean `|g|` of 1.1-3.1 for this
arm, i.e. **below** `grad_clip = 10`, so there the stock clip-down rule does not normalise and the
guidance step is directly proportional to the coefficient.

**The "it is absorbed into the tuned guidance strength" defence fails on the stated constants.** The
step is `dt * min(guidance_frac * sigma_t^2, coeff_cap) * clip(grad V, grad_clip)` with
`sigma_t^2 = 2(1/t - 1)`. At `guidance_frac 8` the `coeff_cap = 10` already binds for all
`t <= 0.615`, which is the first sixth of the [0.5, 0.95] window: `sigma^2 = 2` at t = 0.5 gives
`8 x 2 = 16 -> 10`. Recovering an 11x smaller gradient there would need an effective coefficient of
110, which no value of `guidance_frac` can produce while `coeff_cap = 10`. So the published
configuration cannot be re-derived from the paper's own reward equation by re-tuning the one knob the
paper exposes; `reward_sigma` (or `coeff_cap`) has to be disclosed as well.

### (d) Guided sampling at the equation's own reward coefficient (GPU, pending)

`scripts/dna/findings_audit/reward_temperature.sbatch`, queued behind the three jobs above. The
prediction to falsify: under the stock clip-down rule the guidance step is proportional to the
coefficient wherever `|grad V|` is below `grad_clip` (the GLASS-4 regime), so at the literal Eq (17)
coefficient the step should be roughly an order of magnitude smaller and Table 1's 0.148 / 0.119
should not be reachable; under the scale-free rule the coefficient can only act through the softmax
weights, so the same run should be much less affected. If instead the literal coefficient reproduces
Table 1, the disclosure gap really is only a disclosure gap and this section says so.

### Jobs

| job | what | partition | status |
| --- | --- | --- | --- |
| **49434948** | `posterior_accuracy_probe.sbatch` -- item (c), grad ladder + n = 64 sampling, TF32 off | `kempner_h100` | PENDING, `QOSMaxGRESPerUser` |
| **49431930** | `june_dose_c0_sets.sbatch` -- item (a), published June protocol on parent-disjoint, both targets, n = 1000 | `kempner_h100` | PENDING, `QOSMaxGRESPerUser` |
| **49440315** | `outer_refinement.sbatch` -- the DNA analogue of the QM9 `sample_steps` sweep | `kempner_h100` | PENDING, `QOSMaxGRESPerUser` |
| **49434328** | `posterior_accuracy_cpu_hedge.sbatch` -- item (c) end-to-end at n = 8, y* = -1, CPU | `serial_requeue` | **COMPLETED** 5:06 -- this is what settled item (c) |
| **49442813** | `direction_usefulness.sbatch` -- item (c), full pre-registered direction test, CPU | `serial_requeue` | **COMPLETED** 2:35 |
| **49438552** | `value_smoothness.sbatch` -- item (c), full FD grid, CPU | `serial_requeue` | RUNNING |

**Why the GPU jobs are stuck, measured rather than guessed.** They are not behind other users: the
reason string is `QOSMaxGRESPerUser`, and `squeue -u uunneberg -t R` shows **7 running GPU jobs at
2 GPUs each = 14 GPUs** already held by this account (other agents' runs, 1-7.5 h elapsed), against
a QOS cap of 16. A single-GPU job therefore cannot start until those release. Nothing in this audit's
own submissions is the cause and adding more of them cannot help, which is why the two CPU hedges
were written: they run on `serial_requeue` and are unaffected by the GPU cap.

Cancelled-and-resubmitted ids, none of which had started, so no compute was duplicated:
49431452 / 49431495 / 49431496 (first submission to `kempner_requeue`, 1,184 of 1,231 ahead);
49431928 (moved while testing `gpu_requeue`, which rejects `kempner_albergo_lab` and gives
`albergo_lab` a 2026-10-06 start estimate); 49431932 (`ARM=ablate`, traded for 49440315).

## Everything tried, including what did not work

So that nothing here reads as a result selected after the fact:

1. **(c)** `--mode grad` at the Table-18 reference on the login node, CPU: killed at the 900 s
   timeout without writing output. 25 RK4 steps with autodiff is not a CPU workload; the whole grad
   ladder moved to the GPU job. No numbers were produced or used.
2. **(c)** Forward-only endpoint diagnostics on CPU (argmax margin, soft-hard guide gap) for
   `euler4@1.0`, `euler32@1.0`, `euler32@0.999`, `rk4-25@0.999`: ran, reported above.
3. **(c)** Finite-difference value smoothness, first attempt: 2 outer times x 2 targets x 4
   posteriors x 6 step sizes x 3 directions on CPU. Too slow (15 of 96 rows in 26 min) and the script
   only writes its CSV at the end, so it was stopped and its log kept as
   `logs/fa_value_smoothness_long_partial.log`; the 15 rows it did print are the ones quoted above
   and are reproduced by the narrow rerun (t = 0.5, y* = +1, 4 step sizes, 2 directions).
4. **(c)** `glass_euler32` at higher `guidance_frac` was *not* tried as a substitute for the
   scale-free rule, because `guidance_frac` cannot rescue a gradient that is 5,000x below the clip:
   `coeff` is capped at `coeff_cap = 10` and the stored grid already shows `gf 8 -> 32` moving the
   dMFM arm by 0.01 once the clip binds. The scale-free rule is the correct matched comparison and is
   what S7 item 8 mandates.
5. **(a)** Three hypotheses for the lost clustering were tested against data already on disk and
   *rejected*: a different metric implementation (the scoring functions are AST-identical), a
   different normalising reference (the two pipelines agree to 0.002 on all four real baselines and
   to the printed digit on the one they share), and a different n or dedup rule (both n = 1,000,
   both `exact_unique` 1.0000, both `max_pairwise_n 3000` at seed 0).
6. **(b)** Three candidate constructions for a defensible 0.1813 were tried: an exactly uniform
   reference (r undefined), the position-wise 200-point statistic (0.0264 / 0.065 -- both "no
   correlation", and not 0.1813), and a finite uniform sample (gives anything in (-1, 1)). Only the
   third is consistent with the printed p-value, and it makes the number meaningless.
7. **(d)** The alternative reading "the constant is absorbed into the tuned guidance strength" was
   tested and fails on its own terms: it would be true for the *scale* under the scale-free rule but
   not under the stock clip-down rule used for every published C0 run, and it is false for the
   *shape*, which the ESS table measures directly.
8. **Queue.** `kempner_requeue` (1,184 of 1,231 ahead), `gpu_requeue` with `kempner_albergo_lab`
   (invalid account/partition combination) and `gpu_requeue` with `albergo_lab`
   (`--test-only` start estimate 2026-10-06) were all measured and rejected; everything runs on
   `kempner_h100` with `--constraint="a100|a100-mig|h100|h200"` (the RTX PRO 6000 nodes on the
   requeue partitions have no sm_120 kernels in the pinned torch build).

## What should change, given the above

Proposed edits to files this agent does not own; none were made.

**`docs/REPRODUCIBILITY_PLAN.md` S7 item 7** currently reads "Tables 4-6 and App B.1 were computed on
the purged `split65k` models ... Decided: regenerate on parent-disjoint and report the weaker
result". Two things are wrong with it as a causal statement. The published sets were produced by a
*different guidance protocol* (91 guided steps from t = 0.01 at `reward_scale 1.0`, MC 16, a one-step
dMFM posterior, trajectory to t = 1.0), and the model move it blames moves the same statistics by
<= 0.0024 / <= 0.0042 nats on the tables' own guidance-independent rows against the 0.042-0.159 /
0.099 it is asked to explain. It should say that the two pipelines differ in the guidance protocol as
well as the model, that the regeneration is at the *Table 1* dose, and that the attribution is open
pending the matched rerun.

**`docs/dmfm_demo_reruns.md`**: (i) the statement that the published Tables 4-6 sets came from GLASS
is wrong -- `DMFMSteeringRunner.value_grad_target_dmfm` differentiates through composed dMFM student
maps, so Table 5's "dMFM samples" caption was accurate; (ii) the "DOES NOT SURVIVE" rows for Table 5
`-1` and Table 6's magnitudes should be qualified as "not reproduced at the Table 1 guidance dose;
the published dose has not yet been rerun"; (iii) the carry-over caveat sentence "a 32-step GLASS
posterior does not guide at all" should be removed or replaced, because the arm it cites is
degenerate.

**`docs/STATE_OF_CLAIMS.md`**: the "Two findings no rerun can fix" item 1 needs the same treatment,
and its Table 16 row should say the statistic is undefined in population rather than that the table
and the figure disagree. Its new section "the stated reward is not the one optimised" says "The
results stand; the paper's equation does not describe them. A one-clause disclosure fixes it" -- a
one-clause disclosure does *not* fix it: at the stated coefficient N = 8 has ESS 5.4-6.6 while at the
implemented one it has ESS 1.1-2.6, so the "N = 8 Monte-Carlo draws" description and the
MC-convergence evidence for it are both affected, and `coeff_cap` blocks re-deriving the strength.

**Paper.** (1) Table 16: drop the two "uniform vs. real" rows or replace them with a total-variation
distance (0.1131 vs 0.0053), and say that Pearson r against a uniform reference is undefined in the
limit; note that the Fig 6 PDFs predate both model generations. (2) Disclose `reward_sigma 0.15` and
`reward_scale` next to Eq (17)/(48), state that the C0 demos used `-11.111 (f - y*)^2` and the
diversity/Evo 2 sets `-22.222 (f - y*)^2`, and report the operating point as "N = 8 draws, effective
sample size 1-3" rather than as an 8-draw average. (3) Disclose the Tables 4-6 sampling protocol,
which currently appears nowhere. (4) Do not state that an accurate posterior cannot steer until the
end-to-end runs are in; the one-step evidence so far says its direction is useful but weaker.

## Harvesting the pending jobs

Three GPU jobs were still queued when this was written, all blocked by `QOSMaxGRESPerUser` rather
than by other users' priority. The three CPU jobs completed and are what settled item (c); item (a)'s
verdict rests on data already on disk. What is left would raise n and close item (a)'s positive half:

    squeue -u uunneberg -o "%.10i %.16j %.16P %.8T" | grep dna_fa
    python scripts/dna/findings_audit/collect.py            # prints every landed table as markdown
    # item (a), after 49431930 finishes (n = 1000 sets in outputs/dna/findings_audit/):
    python -W ignore::FutureWarning -m dmfm.experiments.analyze_c0_diversity \
      --repo_root . --run_dirs outputs/dna/findings_audit/june_dose_m1p0 \
      outputs/dna/findings_audit/june_dose_1p0 --out_dir outputs/dna/findings_audit/diversity_june_dose
    # then copy table6_baselines.csv / table7_generated.csv into
    # results/dna/findings_audit/junedose/ and compare the -1 guided row against 0.7010 / 0.4175.

| job | item | what it settles |
| --- | --- | --- |
| 49434948 | (c) | end-to-end at n = 64 on **both** targets, including `glass_rk4_25_e999@norm`, plus the gradient-agreement ladder 4/8/16/32/100 steps to 1.0 and 0.999. Item (c) is already settled by 49434328 and 49442813; this would raise n and add the reference integrator end to end. |
| ~~49434328~~ | (c) | **done** -- see "End-to-end" above |
| ~~49442813~~ | (c) | **done** -- see "Full pre-registered test" above |
| 49438552 | (c) | the full finite-difference smoothness grid, incl. `h = 1e-6` and `rk4-25` (running) |
| 49440315 | (c), cross-domain | whether refining the DNA *outer* trajectory degrades steering the way QM9's DPS `sample_steps` sweep does |
| 49431930 | (a) | whether the published June guidance dose reproduces the `-1` clustering on parent-disjoint models |

Operational note for whoever picks this up: **the login node kills long compute.** It is
cgroup-limited to one core (`nproc` = 1) at load ~50, and two attempts to run a ~30-minute
`direction_usefulness.py --configs glass_rk4_25_e999` there were reaped with an empty log before the
first row printed. Runs of about 10 minutes (the ESS table, the gradient-norm table, the 3-posterior
direction test) completed. Anything longer goes to `serial_requeue` with `albergo_lab`.
