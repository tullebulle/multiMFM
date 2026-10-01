# Is the GEOM-Drugs fingerprint guidance term implemented correctly?

Working document, 2026-09-29. Author: reproducibility agent (GEOM guidance thread).
Owned paths: `src/multimfm/geom/`, `scripts/geom/`, `results/geom/diagnostics/`, this file.
Question put by `docs/REPRODUCIBILITY_PLAN.md` §7 item 1 and
`docs/mismatch_audit_molecules.md` ("rows that look like our bug", item 1):
**before Table 3 is rewritten around a null or negative result, rule out a mis-signed or
mis-scaled guidance term.**

---

## 1. Verdict

**The guidance gradient is correct. It is not mis-signed and it is not mis-scaled. No fix
was applied to it, because none was needed.** The null result at scale is real.

Five independent checks, four numerical and one on the metric itself:

| # | check | what it establishes | result |
|---|---|---|---|
| 1 | central finite difference of `d r / d coords` (§3.2) | the gradient of `r = -\|\|sigmoid(head(X_1)) - c\|\|^2` on the *hard* atom-type path the evaluation scores | ratio to autograd **1.0003** at h = 1.2e-5 (CPU, exact fp32); componentwise median relative error **1.5e-3** over 12 entries |
| 2 | central finite difference of `d r / d atomics` (§3.2) | the soft atom-type relaxation `guide_atomics` differentiates | ratio **1.0026 / 1.0000 / 1.0002 / 0.9977** over four step sizes; on GPU in exact fp32 **1.0008 / 0.9992 / 0.9697** |
| 3 | central finite difference of `grad_x V_t(x)` through the real 10-step MFM flow map, at frozen look-ahead noise (§3.4) | the complete estimator the sampler uses | ratio **1.0058** jointly and **0.9845** in coordinates alone, at a probe step of 1e-6 of the state RMS, against a measured noise floor of 4.8e-7. `fd-check` prints `gradient VERIFIED correct in sign and magnitude` |
| 4 | analytic end-to-end sign test of the production chain (§3.3) | every line between `torch.autograd.grad` and the returned molecule: log-mean-exp, `clip_guidance_rms`, the mu-schedule wrapper, `production_step_with_optional_guidance` | **12 of 12** cases match the closed form to 7 significant figures, for mu in {+0.25, +1, -0.25}, lambda in {0.01, 1}, both mu schedules |
| 5 | **the sign test on the real metric** (§5) | the pre-registered discriminator: under a sign inversion, `-mu` must *improve* Table 3's Tanimoto | at matched \|mu\|, `-mu` is **worse, not better**, at all four matched magnitudes (-0.00818 vs -0.00704 at 0.25; -0.01509 vs -0.00930 at 0.5; -0.01668 vs -0.01476 at 1.0; -0.01102 vs -0.00616 at 0.25 under the paper coefficient), n = 256 paired molecules per point, paired SEM ~0.004. Same in the head-predicted metric |

Check 5 is the one the pre-registration named, and it comes out unambiguously: the response
is roughly symmetric in the sign of `mu` and negative on both sides, which is the signature
of an isotropic perturbation of size \|mu\|, not of a reversed gradient.

**What is *not* part of this evidence: the `grad-sign` probe.** §7 item 1 named it as the
gate, and it has now run (Slurm 49344809), but **its printed verdict "gradient ASCENDS the
guide reward (plumbing correct)" was not supported by its own data and has been removed from
the code.** Read properly, its scan favours `-grad` at every step of 0.1 or larger -- and it
cannot settle the sign at any step size, because at steps small enough to be linear its
Monte-Carlo noise is ~10^4 times its signal, and at steps large enough to beat the noise it is
displacing every atom by 0.34 to 3.4 Angstrom. §2.2 works this through with error bars from
four repeats and reconciles it with checks 1-4. Its across-*draw* uncertainty is still
unmeasured (Slurm 49354725 pending), so **the scan should not be quoted in either direction.**

**A step-size caveat, stated precisely so it is not over-read (§3.4).** The finite
difference matches only in a narrow window: the *single-draw* value function
`V_t(.; frozen noise)` has a linear radius of about **1e-6 of the state RMS**, while the
production step displaces coordinates by **2.6e-2 of the state RMS per guided step**. So each
individual guidance step is a first-order extrapolation about 2.6e4 times beyond the radius
over which that particular draw's gradient describes that particular draw. **This is not by
itself the explanation for the null**, because the sampler re-draws the look-ahead noise at
every guided step, and the noise-*averaged* objective is far smoother than any single draw:
the measured paired `+/-` scan (§2.2) still favours `+grad` out to a displacement of ~6e-2 of
the state RMS, which brackets the production step. The step size is therefore not obviously
wrong. What is wrong is the metric (§4) and the inert atom-type channel (§6.2).

**Why guidance cannot help on this metric either, and that is not a bug (§4).** Table 3's metric
is the OpenBabel FP2 Tanimoto, a *path-based graph* fingerprint whose bonds are re-perceived
from the coordinates. Measured on real GEOM molecules, FP2 is **exactly invariant below
0.006 A** and **as dissimilar from itself as a random other molecule by 0.2 A**. There is no
displacement scale at which a coordinate gradient can improve an FP2 match: too small and the
metric cannot see it, large enough to see and the fingerprint is being randomised. The
production step displaces coordinates by 2.6% of the state RMS *per guided step*
(0.56 A accumulated, measured), which is in the randomising band -- and the measured
dose-response is monotone damage in every metric at once, exactly as that predicts.

**Two genuine defects were found, neither of them the gradient (§6), and neither rescues the
result:** the published runs used a reward scale 20-100x below the paper's own reward
definition (and at the paper's literal `lambda = 1` the method is *much* worse, not better --
measured, §5), and the atom-type half of the guidance is numerically inert.

---

## 2. Why job 49293660 produced nothing, and the grad-sign gate

### 2.1 The job never ran

Nothing crashed and nothing was misconfigured. `sacct -j 49293660` gives
`State=PENDING, Elapsed=00:00:00, Start=Unknown`; `scontrol show job` gives
`Reason=Priority, Dependency=(null)`; `sprio` gives `PRIORITY=55428` of which `AGE=55424`
and `FAIRSHARE=4`. Submitted 16:41, still never started five hours later. There is no log in
`logs/` because it never started. Same for the constant-vs-paper CPU comparison
(49293829 / 49293743 / 49293831). The cause is queue contention against a near-exhausted
fairshare.

Two corrections follow, both now applied to the scripts:

1. **The claim in `scripts/geom/grad_sign.sbatch` and in the `diagnostics` docstring that
   "the login node OOM-kills it, since each checkpoint carries a 1.14M-entry all_smiles
   list" is wrong.** Loading the teacher *and* the student *and* the head on the login node
   takes 2.2 s and peaks at 1.28 GB. The real constraint was recorded nowhere: the login node
   puts the whole *user* in a systemd slice with `MemoryMax=8589934592` (8 GB) and
   `CPUQuotaPerSecUSec=1s` (one core), **shared with every other process the same user runs.**
   With the other reproducibility agents active, 5.5-7.3 GB of the 8 GB was in use throughout
   this session and the four-stage `fd-check` (peak 3.65 GB) was SIGKILLed (exit 137) three
   times. Both docstrings are corrected, with the measured numbers.
2. The diagnostics were repackaged to be runnable at all: `scripts/geom/gradient_audit.sbatch`
   asks for 2 CPUs / 12 GB / 30 min so it backfills (it ran in **1 m 12 s** on one GPU), and
   `fd-check --stages reward soft` skips the teacher and the student entirely (2.05 GB, 2 min)
   so checks 1 and 2 of §1 need no allocation.

### 2.2 grad-sign has now run, and its verdict string was wrong

**Slurm 49344809** (`gpu_test`, 1 m 22 s, COMPLETED) produced
`results/geom/diagnostics/grad_sign_cuda.json`, so §7 item 1's outstanding diagnostic now
exists. Its printed verdict was **"gradient ASCENDS the guide reward (plumbing correct)"**.
**That verdict was not supported by its own data and has been removed from the code.** It was
produced by `max(positive deltas) > 0`, i.e. by comparing the single best probe against
step 0, which is not a sign test at all. Read properly, the scan's own scan *favours the
negative direction* at every step of 0.1 or larger.

The configuration happened to be run four times (seed 0, identical settings, differing only
by GPU nondeterminism and GPU model: Slurm 49344809, 49349903, 49351247, 49352832), which
gives free error bars on the scan. The probe displaces the state by `|step| * grad_rms` with
`grad_rms = 1.715`, against a coordinate state RMS of 0.2637, so the third column is what the
probe actually does to the molecule:

| step | displacement / state RMS | displacement (A) | mean delta over 4 repeats | sd |
|---|---|---|---|---|
| -1.000 | 6.505 | 3.43 | -1.485 | 2.94 |
| -0.300 | 1.952 | 1.03 | +1.603 | 0.45 |
| -0.100 | 0.651 | 0.343 | **+6.311** | 0.17 |
| -0.010 | 0.0651 | 0.0343 | -0.255 | 0.22 |
| -0.001 | 0.00651 | 0.00343 | -1.236 | 0.09 |
| 0 | 0 | 0 | 0 | 0 |
| +0.001 | 0.00651 | 0.00343 | -0.846 | 0.06 |
| +0.010 | 0.0651 | 0.0343 | +0.052 | 0.02 |
| +0.100 | 0.651 | 0.343 | +0.827 | 0.17 |
| +0.300 | 1.952 | 1.03 | -5.265 | 0.24 |
| +1.000 | 6.505 | 3.43 | -39.485 | 3.77 |

The only defensible reading of a step scan is the **paired comparison at matched `|step|`**: a
correctly signed gradient must make `delta(+step) - delta(-step)` positive. Doing that:

| `|step|` | displacement / state RMS | delta(+) | delta(-) | **(+) - (-)** | sd over 4 repeats |
|---|---|---|---|---|---|
| 0.001 | 0.0065 | -0.846 | -1.236 | **+0.391** | 0.156 |
| 0.010 | 0.065 | +0.052 | -0.255 | **+0.308** | 0.218 |
| 0.100 | 0.651 | +0.827 | +6.311 | **-5.484** | 0.245 |
| 0.300 | 1.952 | -5.265 | +1.603 | **-6.868** | 0.268 |
| 1.000 | 6.505 | -39.485 | -1.485 | **-38.000** | 2.766 |

So: **`+grad` wins at the two smallest steps and loses at the three largest.** Both halves are
reproducible. Three things follow.

1. **The crossover is exactly where a curved surface puts it, and exactly where the
   finite-difference check puts it.** `fd-check` measures the FD ratio of `V` as 1.006 at a
   probe step of 1e-6 of the state RMS, 0.74 at 1e-5 and **-0.30 at 1e-4** (§3.4). The scan's
   two "correct" steps are at 6.5e-3 and 6.5e-2 of the state RMS; its three "wrong" steps are
   at 0.65, 2.0 and 6.5 -- that is, **displacing every atom by 0.34, 1.03 and 3.43 Angstrom.**
   A probe that moves every atom by 3.4 A is not measuring a derivative; it is measuring what a
   scrambled molecule scores. (The scan's surface is smoother than `fd-check`'s because it
   averages over 16 *fresh* look-ahead draws, which mollifies the function -- that is why its
   crossover sits at ~0.1 rather than at 1e-5.)
2. **The scan can never settle the sign, at any step size.** Below the curvature scale its
   signal is `2h|grad|^2`, which at a step small enough to be linear is ~1e-4 in reward units,
   while its Monte-Carlo noise from 16 fresh look-aheads is of order 1 -- four orders of
   magnitude larger. Above the noise, it is outside the linear regime. There is no window. This
   is a structural defect of the probe, not bad luck.
3. **The verdict string is gone from the code.** `cmd_grad_sign` now prints the paired `+/-`
   table, the displacement in units of the state RMS, and an explicit "UNDERPOWERED AS A SIGN
   TEST" statement pointing at `fd-check`; the module docstring says the same, and the record
   now also stores `paired_plus_minus` and `state_coords_rms`. The JSON already on disk
   (`grad_sign_cuda.json`, written by Slurm 49352832 before the fix) still contains the old
   `verdict` string, so a `_verdict_note` key was added to that file warning against it. Its
   measurements were not altered.

**Error bars across independent draws.** The sd column above is GPU nondeterminism only: all
four repeats share the same 4 molecules and targets. `scripts/geom/grad_sign_seeds.sbatch`
repeats the scan over 5 independent seeds (different molecules, targets and trajectory) and
reports the mean, sd, SEM and t-statistic of the paired delta per step, plus the paired
`+/-` comparison with a combined SEM. Submitted as Slurm **49354725**; it was still
`PENDING` at the end of this session, so **the across-draw uncertainty on the scan is not yet
measured and the scan should not be quoted at all until it is.** Nothing in §1 depends on it:
the sign is settled by §3.2, §3.3, §3.4 and the §5 sign test, all of which are either exact or
have their own error bars.

## 3. The gradient, verified by hand

### 3.1 Sign convention, stated then measured

The paper's ODE sampler (eq. 1, p. 3) is
`dX_t = (b_t(X_t) + (sigma_t^2/2) grad V_t(X_t)) dt` with
`V_t(x) = log E[exp(lambda r(X_1)) | X_t = x]`, and for the linear interpolant
`sigma_t^2 = 2(1-t)/t`, so `sigma_t^2/2 = (1-t)/t` (paper line 159). The reward is
`R(G) = exp(-||E(G) - c||^2)`, `r(G) = -||E(G) - c||^2` (paper lines 379-380).

`r` is *higher* when the predicted fingerprint is *closer* to the target, so the guided
velocity must add `grad V` with a **positive** coefficient. In the code:

* `multimfm.geom.reward.fingerprint_reward` returns `-((pred - target)**2).sum(-1)`, the
  paper's `r`;
* `glass_guidance.value_gradient` forms `logsumexp(reward_scale * rewards, -1) - log N` and
  returns `torch.autograd.grad(values.sum(), [coords, atomics])` -- the gradient of a scalar,
  an **ascent** direction, with no negation anywhere;
* `glass_guidance.production_step_with_optional_guidance` adds
  `next_coords + mu * grad_coords * dt` -- **plus**, so positive `mu` ascends.

No sign inversion on that path. That is a reading; §3.2-3.4 are the measurements.

### 3.2 Finite differences on the reward

`results/geom/diagnostics/fd_check_reward_only.json`, login node, no allocation, 2 min,
2.05 GB peak:

```
python -m multimfm.geom.diagnostics fd-check --device cpu --stages reward soft \
    --num-mols 4 --components 12 --rel-steps 1e-5 1e-4 1e-3 1e-2 \
    --out results/geom/diagnostics/fd_check_reward_only.json
```

Four real GEOM val molecules (24-30 heavy atoms), one-hot atom types, targets drawn from
*other* molecules -- the real steering objective. The gradient is normalised to unit RMS and
the probe step is a fraction of the coordinate RMS (1.166), so
`(r(x + h g) - r(x - h g)) / (2 h)` must equal `|g|^2` as `h -> 0`:

| stage | rel step | h | FD slope | `|g|^2` | ratio |
|---|---|---|---|---|---|
| `d r / d coords`, hard atom types | 1e-5 | 1.166e-5 | 5.311410e3 | 5.309914e3 | **1.0003** |
| | 1e-4 | 1.166e-4 | 5.335481e3 | 5.309914e3 | **1.0048** |
| | 1e-3 | 1.166e-3 | 5.528523e3 | 5.309914e3 | 1.0412 |
| | 1e-2 | 1.166e-2 | 2.766363e3 | 5.309914e3 | 0.5210 |
| `d r / d atomics`, soft, T = 0.25 | 1e-5 | 3.333e-6 | 1.043701e3 | 1.040985e3 | **1.0026** |
| | 1e-4 | 3.333e-5 | 1.040955e3 | 1.040985e3 | **1.0000** |
| | 1e-3 | 3.333e-4 | 1.041229e3 | 1.040985e3 | 1.0002 |
| | 1e-2 | 3.333e-3 | 1.038629e3 | 1.040985e3 | 0.9977 |

The ratio converges to **+1**, not -1 and not some other constant. Componentwise central
differences on 12 individual `(molecule, atom, axis)` entries agree with autograd to a median
relative error of **1.5e-3**. The same two stages at the *MFM look-ahead endpoint* on GPU in
exact fp32 (Slurm 49349903) give coords ratio 0.9966 at h = 9.4e-5 with componentwise median
relative error 8.2e-3, and atomics 1.0008 / 0.9992 / 0.9697.

Note where the coords row leaves the linear regime: a displacement of 1% of the coordinate
RMS (0.023 A) already halves the ratio. That is §4 arriving early.

Two by-products:

* the soft atom-type relaxation is faithful to the hard path it stands in for:
  `||p_soft - p_hard||_2 = 1.01` over 1024 bits (mean bit probability 0.1573 vs 0.1580);
* the guide head is good. `checkpoints/geom/structure_head/best.txt` records
  `val_tanimoto = 0.8523` at epoch 34 on 146,401 real val molecules, and the newly added
  `guide_accuracy_mean` (§6.3) measures **0.757** on *unguided generated* molecules. The
  guide transfers, so "the guide is useless" is not an available explanation.

### 3.3 End-to-end sign test of the production chain

`scripts/geom/sign_chain_selfcheck.py` runs the *real* `guided_sample` with the two learned
pieces stubbed to functions whose exact answer is known: the look-ahead posterior stubbed to
the identity (`x_1 = x_t`, so `V_t(x) = lambda r(x)` exactly) and the base velocity stubbed
to zero (so the guidance term is the only motion). With `reward_name="first_atom_x"` the
analytic answer is `coords[0,0,0] = sum_guided_steps mu * coeff(t) * lambda * dt * (1 - 1/N)`,
the last factor being the zero-COM projection `mask_and_zero_com` applies to the gradient.
No checkpoints, no GPU, ~40 s:

```
  mu sched     mu  lambda   observed x[0,0,0]        analytic   ok
  constant   0.25    0.01     0.0006666668924 0.0006666666667  YES
  constant   0.25       1       0.06666667014   0.06666666667  YES
  constant      1    0.01      0.002666667569  0.002666666667  YES
  constant      1       1        0.2666666806    0.2666666667  YES
  constant  -0.25    0.01    -0.0006666668924-0.0006666666667  YES
  constant  -0.25       1      -0.06666667014  -0.06666666667  YES
     paper   0.25    0.01      0.001093012514  0.001093012352  YES
     paper   0.25       1        0.1093012467    0.1093012352  YES
     paper      1    0.01      0.004372050054  0.004372049408  YES
     paper      1       1        0.4372049868    0.4372049408  YES
     paper  -0.25    0.01     -0.001093012514 -0.001093012352  YES
     paper  -0.25       1       -0.1093012467   -0.1093012352  YES

PASS: all 12 cases match the analytic value, including the sign.
```

Positive `mu` moves the state along `+grad V`; negative `mu` moves it the other way by
exactly the same amount.

### 3.4 The full chain through the real MFM flow map, finite-differenced

The four-stage `fd-check` also finite-differences `V_t` itself, through the real 10-step MFM
flow map, at frozen look-ahead prior noise. Slurm **49352832** (`gpu_test`, exact fp32 with
`torch.backends.cuda.matmul.allow_tf32 = False`; noise floor over repeated evaluations at the
same `x` measured at **4.768e-7**, so every row is 10^4-10^5 times above the noise):

| direction | rel step | h | FD slope | `|g|^2` | ratio |
|---|---|---|---|---|---|
| coords + atomics | 1e-8 | 4.086e-9 | 1.342011e3 | 1.458457e3 | 0.9202 |
| | 1e-7 | 4.086e-8 | 1.342011e3 | 1.458457e3 | 0.9202 |
| | **1e-6** | 4.086e-7 | 1.466877e3 | 1.458457e3 | **1.0058** |
| | 1e-5 | 4.086e-6 | 1.076877e3 | 1.458457e3 | 0.7384 |
| | 1e-4 | 4.086e-5 | -4.365096e2 | 1.458457e3 | -0.2993 |
| coords only | 1e-7 | 2.637e-8 | 1.229839e3 | 8.799966e2 | 1.3975 |
| | **1e-6** | 2.637e-7 | 8.663127e2 | 8.799966e2 | **0.9845** |
| | 1e-5 | 2.637e-6 | 8.561846e2 | 8.799966e2 | 0.9729 |
| | 1e-4 | 2.637e-5 | 3.241619e2 | 8.799966e2 | 0.3684 |

**`grad_x V_t(x)` is the true gradient of `V_t`: ratio 1.0058 jointly and 0.9845 in
coordinates alone, at a probe step of 1e-6 of the state RMS.** Stage 1 in the same run gives
best ratio **0.9999** and stage 2 **1.0008**, and the printed verdict is
`gradient VERIFIED correct in sign and magnitude`. The entire chain -- reward, soft
relaxation, 10-step flow-map differentiation, log-mean-exp, and the production step -- is
correct.

**The step-size reading, stated carefully.** The ratio is near 1 only in a narrow window: it
degrades to 0.74 at a probe step of 1e-5 of the state RMS and *changes sign* (-0.30) at 1e-4.
So the linear radius of the **single-draw** value function `V_t(.; frozen noise)` is about
**1e-6 of the state RMS**, while

> the production step displaces coordinates by **2.649e-2 of the state RMS per guided step**
> (stage 4, measured),

about 2.6e4 times that radius. **Two readings, and only the weaker one is supported.** The
supported one: any *single* guidance step is a first-order extrapolation far beyond the range
over which that draw's gradient describes that draw, so the per-step displacement is not a
small perturbation of the estimated objective and no local-descent argument applies to it. The
*unsupported* one -- which an earlier draft of this document asserted and which is withdrawn
here -- is that this alone explains the null. It does not, for two reasons. The sampler
re-draws the look-ahead noise at every guided step, so what it effectively follows is a noisy
estimate of the *noise-averaged* objective, and that is a much smoother function: the paired
`+/-` scan of §2.2, which averages 16 fresh look-aheads, still favours `+grad` at a
displacement of 6.5e-2 of the state RMS and only turns over between 6.5e-2 and 0.65. The
production step at 2.6e-2 sits inside that ascent region. So the step size is *not* obviously
mis-chosen, and §4 and §6.2, not this, are the reasons the method fails here.

**A recorded false start, so nobody repeats it.** The *first* GPU run of this stage inherited
production's `torch.set_float32_matmul_precision("high")`, i.e. TF32. TF32 carries ~10 mantissa
bits, the 10-step flow-map rollout compounds that, and `V` was then reproducible only to ~1%
between calls -- so the stage-3 FD slopes scaled as `1/h` across three step sizes, the
signature of pure noise, and stage 1's componentwise relative error was 0.95 instead of 8e-3.
It would have been easy to misread as a sign failure. `fd-check` now forces `"highest"`,
disables TF32, measures and prints its own noise floor, and judges each stage at the probe step
whose ratio is closest to 1 rather than at the smallest step (in fp32 the smallest steps here
give a central difference of exactly 0). The componentwise check uses its own, larger step
(`--component-rel-step`, default 1e-4) for the same reason; the clean CPU value is 1.5e-3
(§3.2) and the GPU value at the same nominal step is 0.10, the difference being fp32
cancellation on a single-component difference.

## 4. Why guidance cannot raise this metric: FP2 is a cliff, not a slope

This is the substantive finding, and it is not a bug in our code. It explains every symptom
the audit flagged, including the one that looked most like a sign defect.

**The metric.** Table 3 reports the Tanimoto similarity of the **OpenBabel FP2** of the
generated molecule to the target. FP2 is a *path-based graph* fingerprint: bonds are
re-perceived from the 3-D coordinates (`ConnectTheDots` + `PerceiveBondOrders`) and only then
are bits set. FP2 is therefore a piecewise-constant function of the coordinates -- blind below
the bond-perception thresholds, discontinuous above them.

`python -m multimfm.geom.diagnostics fp2-sensitivity --num-mols 16 --repeats 3`
(`results/geom/diagnostics/fp2_sensitivity.json`) perturbs 16 real GEOM val molecules with
isotropic coordinate noise and measures each molecule's FP2 Tanimoto **to its own unperturbed
FP2**:

| sigma (normalized) | sigma (Angstrom) | self-Tanimoto | fraction of molecules with any bit changed |
|---|---|---|---|
| 0.001 | 0.002 | 1.0000 | 0.000 |
| 0.003 | 0.006 | 1.0000 | 0.000 |
| 0.01 | 0.02 | 0.9777 | 0.104 |
| 0.03 | 0.06 | 0.7933 | 0.479 |
| 0.1 | 0.2 | **0.2580** | 1.000 |
| 0.3 | 0.6 | 0.0929 | 1.000 |
| 1.0 | 2.0 | 0.0420 | 1.000 |

Read the two ends together. Below **0.006 A** the metric is *exactly* invariant: no
displacement in that range can change Table 3's number in either direction. By **0.2 A** a
molecule's fingerprint is no more similar to itself (0.258) than a random other GEOM
molecule's is to the target (the unguided baseline is 0.2278). The transition occupies **less
than one and a half decades.** There is therefore no displacement scale at which a coordinate
gradient can usefully improve an FP2 match. That is a property of FP2, not of multiMFM, and it
applies equally to TFG-Flow's own numbers on the same benchmark.

**Where the production step actually sits.** `fd-check` stage 4, measured at the as-described
setting (lambda = 0.01, mu = 0.25, dt = 0.01, guidance on every 2nd of 100 steps inside
t in [0.1, 0.9], so 40 guided steps), with the gradient RMS through the real flow map at
t = 0.5:

| quantity | measured |
|---|---|
| coords gradient RMS through the flow map | 2.794 |
| coordinate displacement per guided step | 0.006984 normalized = **2.65% of the state RMS** |
| accumulated over 40 guided steps if coherent | 0.2794 normalized = **0.559 A** |
| atom-state displacement per guided step | 0.00186; accumulated **14.5% of the atom-state RMS** |
| the paper's `(1-t)/t` coefficient at t = 0.5 | x 1.000 |

**0.56 A is past the 0.2 A at which FP2 is fully randomised.** Even the incoherent
(random-walk) accumulation, `sqrt(40) x 0.0070 x 2 = 0.088 A`, sits in the middle of the
0.02-0.2 A degradation band and far above the 0.006 A invariance floor. The flow model does
absorb part of the perturbation -- the guided molecules stay 100% convertible and 83-88%
PB-valid, so they are *different plausible molecules*, not wreckage -- but the direction in
which they differ carries no usable FP2 information.

**This predicts, quantitatively, everything the audit found suspicious.**

* *The monotone dose-response* (§5): every metric degrades together as `|mu|` grows -- counted
  Tanimoto, head-predicted Tanimoto, PoseBusters validity, and the guide's own accuracy.
* *"The head-predicted Tanimoto also falls, which is the signature of a sign defect."* It is
  not, once the geometry is being damaged: the head is an EGNN on the same coordinates, so a
  randomised geometry lowers the head's fingerprint match too. The only clean discriminator is
  the sign test, and §5 runs it.
* *The guide's accuracy on the molecules it produced collapses with dose* -- 0.757 unguided,
  0.754 at lambda = 0.01/mu = 0.25, 0.390 at lambda = 1/mu = 0.25, 0.273 at lambda = 1/mu = 0.5
  -- which is what pushing molecules off the distribution the head was trained on looks like.

### 4.1 The three facts side by side

Because they look contradictory until the numbers are lined up, and because the audit read the
third one as evidence for a sign defect:

| fact | measurement |
|---|---|
| the ascent direction is correct | FD ratio **1.006** for `grad_x V_t` at a probe step of 1e-6 of the state RMS, noise floor 4.8e-7 (§3.4); **1.0003** for `d r / d coords` (§3.2); analytic chain exact to 7 digits (§3.3); and `-mu` is **worse** than `+mu` at all four matched magnitudes (§5.2) |
| it produces no gain in FP2 Tanimoto at 4,096 x 3 | paired gain **-0.00031 +- 0.00065** on 12,288 molecules, 0.5 sigma from zero (§5.3) |
| the *head-predicted* Tanimoto falls too | **-0.00034** in the same run; and monotonically with dose, to -0.095 at lambda = 1, mu = 1 (§5.2) |

There is no contradiction. The third fact was the audit's main reason to suspect a sign error,
and it is fully accounted for without one: the head is an EGNN on the same coordinates the
guidance perturbs, so once the perturbation is large enough to move the geometry off the
distribution the head was trained on, the head's *own* fingerprint match degrades along with
everything else. That is measured directly, not inferred: `guide_accuracy_mean` -- the head's
prediction against the molecule's own FP2 -- is **0.757** unguided, **0.754** at
lambda = 0.01/mu = 0.25, **0.390** at lambda = 1/mu = 0.25 and **0.140** at
lambda = 1/mu = 1 (§5.2). A guide whose accuracy on its own inputs has collapsed to 0.14
cannot be expected to report an improvement, however correctly its gradient is computed.

At the *published* strength (lambda = 0.01, mu = 0.25) the damage is mild -- guide accuracy
0.754 vs 0.757, PoseBusters 88.28% -> 87.50% -- and correspondingly the gain is not negative,
it is **zero**: -0.00031 +- 0.00065. That is the honest description of Table 3's setting. The
strongly negative numbers appear only once `|mu|` or `lambda` is raised enough to visibly
damage the molecules, which §4 shows is unavoidable on a metric that is invariant below
0.006 A and randomised by 0.2 A.

---

## 5. The grid, every point

### 5.1 Pre-registration, verbatim (written before any new point was run)

> **Grid.** All points at `--num-mols 256`, `--seed 0`, drawn from the held-out target pool
> (`--holdout-n 512 --holdout-seed 20260929 --tune-on-holdout`), which is disjoint by
> construction from the 4,096 x 3 evaluation targets. Sampler settings identical to the
> recorded runs and to the five tuning points already on disk: `--sample-steps 100`,
> `--mfm-steps 10`, `--value-samples 4`, `--guide-every 2`, `t in [0.1, 0.9]`,
> `--guide-atomics` on. So the five existing points fold into the same grid.
>
> | arm | purpose | lambda | mu | mu schedule |
> |---|---|---|---|---|
> | A0 | already on disk (the as-described sweep) | 0.01 | +0.25, +0.5, +1.0 | constant |
> | A0 | already on disk | 0.02 | +0.25, +0.5 | constant |
> | A- | **sign test**: if the gradient is mis-signed, negative mu must *improve* | 0.01 | -0.25, -0.5, -1.0 | constant |
> | B | the paper's literal reward `R = exp(-||E-c||^2)`, i.e. lambda = 1 | 1.0 | +0.25, +0.5, +1.0 | constant |
> | B- | sign test at the paper's lambda | 1.0 | -0.25 | constant |
> | C | the paper's coefficient `sigma_t^2/2 = (1-t)/t` | 0.01, 1.0 | +0.05, +0.25 | paper |
> | C- | sign test under the paper's coefficient | 0.01 | -0.25 | paper |
>
> **Reported statistic per point.** The *paired* molecule-level Tanimoto gain (guided minus
> unguided from the same initial noise and the same target), its paired SEM, the head-predicted
> Tanimoto gain, PoseBusters validity, and `guide_accuracy_mean` (head prediction vs the
> molecule's own FP2).
>
> **Selection rule.** *No point is selected.* Every point is reported. A point counts as
> evidence that the guidance term works only if **both** (i) its paired gain exceeds 2 x its
> paired SEM, and (ii) its head-predicted gain is also positive. If no point clears that bar,
> the reported conclusion is "no demonstrated gain", whatever the largest point happens to be.
> Best-of-grid is explicitly *not* a result; the audit already showed that best-of-four at
> n = 64 is how the published 0.227 arose.
>
> **Pre-registered interpretation of the sign test.** If gains at `-mu` are systematically
> *better* than at `+mu` of the same magnitude, the guidance term is mis-signed and must be
> fixed. If gains at `-mu` and `+mu` are both negative and roughly symmetric in magnitude, the
> term is not mis-signed: any perturbation of that size degrades the molecule, and the result
> is a property of the method at this scale, not a bug.

### 5.2 Results: all 16 measured points

Slurm arrays **49345752** (constant mu) and **49345761** (paper coefficient), plus the five
points already on disk. `results/geom/diagnostics/guidance_grid.csv`; regenerate with
`python scripts/geom/collect_guidance_grid.py`. "guide acc" is the guided arm's
`guide_accuracy_mean`; the unguided arm is 0.7569-0.7570 in every run that measured it.

```
    sched  lambda     mu  unguided    guided       gain      SEM  g/SEM  head gain  PB drop  guide acc
------------------------------------------------------------------------------------------------------
 constant    0.01     -1   0.23356   0.21688   -0.01668  0.00506  -3.30   -0.01562  +0.0391     0.7286
 constant    0.01   -0.5   0.23356   0.21847   -0.01509  0.00447  -3.37   -0.01396  +0.0000     0.7564
 constant    0.01  -0.25   0.23356   0.22537   -0.00818  0.00353  -2.32   -0.00806  +0.0234     0.7533
 constant    0.01   0.25   0.23356   0.22652   -0.00704  0.00499  -1.41   -0.00526  +0.0078        n/a
 constant    0.01    0.5   0.23356   0.22426   -0.00930  0.00502  -1.85   -0.01017  +0.0312        n/a
 constant    0.01      1   0.23356   0.21880   -0.01476  0.00524  -2.82   -0.01195  +0.0508        n/a
 constant    0.02   0.25   0.23356   0.22789   -0.00567  0.00429  -1.32   -0.00729  +0.0352        n/a
 constant    0.02    0.5   0.23356   0.21908   -0.01447  0.00529  -2.74   -0.01181  +0.0352        n/a
 constant       1   0.25   0.23356   0.14541   -0.08835  0.00785 -11.26   -0.05847  +0.4336     0.3897
 constant       1    0.5   0.23356   0.11547   -0.11836  0.00784 -15.10   -0.06829  +0.5703     0.2726
 constant       1      1   0.23356   0.08114   -0.15245  0.00765 -19.93   -0.09541  +0.7383     0.1404
    paper    0.01  -0.25   0.23477   0.22375   -0.01102  0.00455  -2.42   -0.01073  -0.0195     0.7544
    paper    0.01   0.05   0.23356   0.22909   -0.00446  0.00348  -1.28   -0.00420  -0.0078     0.7541
    paper    0.01   0.25   0.23356   0.22740   -0.00616  0.00471  -1.31   -0.00531  -0.0195     0.7540
    paper       1   0.05   0.23356   0.19063   -0.04293  0.00699  -6.14   -0.03443  +0.1953     0.5933
    paper       1   0.25   0.23477   0.13489   -0.09988  0.00756 -13.22   -0.06806  +0.4961     0.3332

16 points; 0 clear the pre-registered bar (paired gain > 2 x SEM AND head gain > 0)
```


**One point of the 17 is still queued**: the `constant lambda=1 mu=-0.25` sign test (task 6
of 49345752, held by `JobArrayTaskLimit`). `collect_guidance_grid.py` folds it in when it
lands. Nothing in §1 depends on it: the sign test already has four matched +-`mu` pairs at
lambda = 0.01, and the lambda = 1 trend is monotone across mu = 0.25, 0.5 and 1.0.

PoseBusters validity for the three `lambda = 1` constant-mu points: 88.28% unguided ->
**44.92%** at mu = 0.25, **31.25%** at mu = 0.5, **14.45%** at mu = 1.0, with the guide's
accuracy on its own molecules falling 0.757 -> 0.390 -> 0.273 -> 0.140. The paper's literal
reward does not merely fail to help; at any mu tried it destroys the molecules.

**Three readings, all pre-registered.**

1. **The sign test (the §1 gate).** At matched `|mu|`, `-mu` is *worse*:

   | \|mu\| (lambda = 0.01, constant) | gain at `+mu` | gain at `-mu` | head gain `+mu` | head gain `-mu` |
   |---|---|---|---|---|
   | 0.25 | -0.00704 | **-0.00818** | -0.00526 | **-0.00806** |
   | 0.50 | -0.00930 | **-0.01509** | -0.01017 | **-0.01396** |
   | 1.00 | -0.01476 | **-0.01668** | -0.01195 | **-0.01562** |
   | 0.25 (paper coefficient) | -0.00616 | **-0.01102** | -0.00531 | **-0.01073** |

   Under the pre-registered rule this is the "not mis-signed" branch: both signs of `mu`
   degrade, roughly symmetrically, with the descent direction mildly worse -- which is the
   small residual *correct-sign* signal sitting on top of a much larger isotropic damage term.
2. **The paper's literal reward is far worse, not better.** `lambda = 1` -- the reward the
   paper actually writes down -- at `mu = 0.25` gives a gain of **-0.0884** with PoseBusters
   validity down **43.4 pp** and the guide's own accuracy collapsing 0.757 -> 0.390; at
   `mu = 0.5`, -0.1184 and 57.0 pp. So the configuration the paper describes has now been
   measured, and it does not rescue Table 3; it destroys the molecules. This is the direct
   answer to the audit's "lambda = 1.0 is unmeasured" item.
3. **The paper's `(1-t)/t` coefficient behaves like a milder constant mu, not differently in
   kind.** At `lambda = 0.01` it gives -0.00446 (mu = 0.05) and -0.00616 (mu = 0.25), the least
   negative points in the whole grid, and it is the only arm where PoseBusters validity
   *improves* slightly under guidance (-0.0078, -0.0195). But the gains are still negative,
   still within 1.3 sigma of zero, and the head-predicted gain is negative too, so no point
   clears the bar. The constant-mu substitution therefore is *not* what hid an effect.

**One reproducibility caveat on the table.** 14 of the 16 runs report the unguided arm as
exactly 0.23356; the two that ran on `holygpu8a10201` report 0.23477. The unguided sampling
is bit-reproducible within a GPU architecture but not across, under production's TF32 matmuls
(already noted in `results/geom/table3_asdescribed_selection.json`'s device note). Every
`gain` above is a *paired* within-run difference, so it is unaffected.

### 5.3 The as-described 4,096 x 3 evaluation is complete, and the sign flipped back

Reported here because the audit had to quote a partial pool and the numbers moved.
`sacct -j 49274410`: **all 24 array tasks COMPLETED**; all 24 shard JSONs are in
`results/geom/runs/asdescribed_eval/`. Re-pooling all 24 (12,288 paired molecules,
lambda = 0.01, mu = 0.25, constant mu):

| quantity | 17-shard partial (audit) | **24-shard complete** |
|---|---|---|
| unguided FP2 Tanimoto | 0.226735 | **0.227768** |
| guided FP2 Tanimoto | 0.227392 | **0.227456** |
| molecule-weighted paired gain | +0.000658 | **-0.000312** |
| shard sd / SEM | 0.002933 / 0.000711 | 0.003192 / **0.000651** |
| negative-gain shards | 8 of 17 | **14 of 24** |
| head-predicted Tanimoto gain | +0.000584 | **-0.000343** |
| PB validity | 87.707 -> 86.960 % | **87.752 -> 87.142 %** |

The completed evaluation is **-0.00031 +- 0.00065**: a clean null, 0.5 sigma from zero, in
both the counted and the head-predicted metric. The partial pool's `+0.00066` was the
replicate-2 artifact the audit warned about. `docs/mismatch_audit_molecules.md` "In flight"
and `docs/REPRODUCIBILITY_PLAN.md` §7 item 1 should both be updated to this.

---

## 6. The two defects that *are* real

Neither is the gradient, and neither rescues Table 3.

### 6.1 The published runs used a reward scale 20-100x below the paper's own reward

The paper defines the tilt with no free scale: `R(G) = exp(-||E(G) - c||^2)` (line 379), i.e.
`lambda = 1`. Every recorded GEOM run used `lambda` in [0.01, 0.10] -- the `--reward-lambda`
default is 0.05 for the MFM driver and 0.1 for the GLASS driver, the published Table 3 point
used 0.02, the as-described rerun 0.01. So the tilt applied was 10-100x weaker than the one
the paper writes down, and **the configuration the paper describes had never been run.** It
has now: §5.2 arm B, and it is *much* worse.

Worth recording why `lambda = 1` is not merely a rescaling. With `r` of order -168 and a
spread of ~47 between the 4 posterior samples, `logsumexp(lambda * r)` at `lambda = 1` is
dominated by the single best sample, so the 4-sample log-mean-exp degenerates into "take the
gradient of the best look-ahead" -- a much higher-variance estimator, and the paper gives no
posterior-sample count at which it would be accurate. The measured gradient is also ~100x
larger, which per §4 is squarely in the randomising band; the 43-57 pp PoseBusters collapse is
that.

No code change is warranted: `--reward-lambda` already exposes it and the default reproduces
the recorded runs, which is the convention this repo keeps.

### 6.2 The atom-type half of the guidance is inert

`guide_atomics` is on by default and its gradient is correctly computed (§3.2 verifies it to
4 digits). It nevertheless does nothing:

* at the reward level, a step along the atom-type gradient of **10% of the atom state's own
  RMS** produces **0 argmax flips out of 420 real atoms**, and the hard-path reward is
  unchanged to 3 decimals. Only at 100% of the state RMS do 57 of 420 flip
  (`stage2_soft_atomics.hard_follow`, Slurm 49349903);
* in production the accumulated atom-state displacement is 14.5% of the state RMS (stage 4) --
  the same order as the 10% that already flips nothing;
* and in the finished 4,096 x 3 evaluation, `mean_atoms` is **bit-identical between the guided
  and unguided arms in 20 of 24 shards** (e.g. 25.2402 in both arms of shard 0, over 512
  molecules). The composition really is unchanged.

So Table 3's guidance is in effect coordinate-only. Anything the paper says about steering
composition on GEOM-Drugs is not supported by these runs -- and it is why §4's cliff is
inescapable here: the only channel that is active is the one the metric cannot use.

### 6.3 Things checked and cleared (recorded so they are not re-litigated)

* **Coordinate frame.** The structure head was trained on mean-centred Angstrom coordinates
  divided by `coord_normalizer = 2.0`
  (`third_party/tfg_flow/dataloader/datasets/geom_fingerprint.py` docstring,
  `checkpoints/geom/structure_head/train_args.json`), which is the frame the flow model
  operates in and the frame `fingerprint_prediction` feeds it. `state_to_binary_fps`
  multiplies by 2.0 before OpenBabel, which is right because FP2 needs Angstroms. No mismatch.
* **The soft and hard atom-type paths compute the same function.** `fingerprint_prediction`'s
  soft branch reproduces `Predictor.forward` hunk for hunk (embed -> mask -> EGNN -> mask ->
  masked mean-pool -> linear), differing only in `softmax(atomics/T) @ atom_emb.weight` in
  place of `atom_emb(argmax)`. Measured divergence: `||p_soft - p_hard||_2 = 1.01` over 1024
  bits on real molecules, 1.97 at the look-ahead endpoint.
* **Target/posterior alignment.** `repeat_state` and the fingerprint patch's
  `property_target.repeat_interleave(P, dim=0)` both interleave, and `rewards.view(B, P)`
  matches that layout, so every look-ahead sample is scored against its own molecule's target.
* **The mu-schedule mechanism.** `install_mu_schedule("paper")` pre-multiplies the gradient by
  `(1-t)/max(t, 0.05)`; measured **9.000** at t = 0.1, **1.000** at t = 0.5, **0.111** at
  t = 0.9, matching `(1-t)/t` exactly, and `mu = 0.0` passes through unscaled so the unguided
  arm is untouched. `guided_sample.__globals__ is glass_guidance.__dict__`, so the rebinding
  is visible to the sampler (verified directly). Relative to the paper, a constant `mu = 0.5`
  is 0.056x the paper's coefficient at t = 0.1 and 4.5x at t = 0.9, as
  `multimfm/geom/mu_schedule.py` states.
* **The guide head transfers to generated molecules.** `fp.score` now also returns
  `guide_accuracy_mean` -- the Tanimoto between the head's prediction and the molecule's *own*
  FP2, which is not a similarity to the target at all but a measure of whether `E(G)` tracks
  `FP2(G)` on the molecules the model actually makes. Measured **0.757** on 256 unguided
  generated molecules (GPU, §5.2) and **0.818** on an independent 16-molecule CPU draw
  (`results/geom/diagnostics/guide_accuracy/`, true fp32, login node), against 0.8523 on real
  val molecules. So the guide is informative on generated molecules too, and that is not the
  explanation; guidance then *degrades* it with dose (§4.1). Note in passing that this run is
  also a clean check that `mu = 0.0` makes the guided arm bit-identical to the unguided one:
  every field of the two arms matches exactly.

---

## 7. What this does to Table 3's claim

1. **"multiMFM 0.227" cannot be claimed as an improvement.** The completed as-described
   evaluation at 4,096 x 3 gives a paired gain of **-0.00031 +- 0.00065** on 12,288 molecules:
   indistinguishable from zero. The earlier lambda = 0.02, mu = 0.5 rerun gives
   -0.00457 +- 0.00081. The published 0.227 is the best of a four-point grid at n = 64, whose
   one-draw probability from the 11,260-molecule population is 1.06%, 4.16% as best-of-four
   (`table3_n64_fragility.json`). The defensible statement is **"no demonstrated gain at
   scale"**. The base generator's own 0.2278 +- 0.0007 already exceeds TFG-Flow's guided 0.208,
   and *that* is the comparison worth making, with the audit's caveat that TFG-Flow's reference
   is measured on molecules capped at 30 heavy atoms against our 71.
2. **The negative sign should not be asserted either.** §7 item 1 currently says "guidance was
   negative at every setting tried". At the held-out-selected setting the completed evaluation
   is -0.5 sigma, i.e. null. Only the larger-`|mu|` points are significantly negative, and §4
   explains those as geometry damage rather than as a property of the value gradient.
3. **The gradient defence is closed.** The audit's leading hypothesis -- a mis-signed or
   mis-scaled term hiding a real effect -- is ruled out by §3.2, §3.3 and the §5 sign test. The
   abstract's "In both cases, our steering algorithms improve target attainment" is not
   supportable for GEOM-Drugs, and it cannot be rescued by fixing the implementation, because
   there is nothing to fix. The one configuration that had never been tried, the paper's own
   `lambda = 1`, is now measured and is 100x worse than the published setting.
4. **The honest positive claim available here is a methodological one, and it is worth
   making.** FP2 Tanimoto is a piecewise-constant graph functional of the coordinates,
   *exactly* invariant below 0.006 A and randomised by 0.2 A (§4, measured, not argued), so
   continuous coordinate guidance has no operating window on it, and the production step sits
   in the randomising band (0.56 A accumulated, measured). That is a statement about the
   *benchmark* TFG-Flow introduced; it applies to any method evaluated on it, and it would be a
   genuine contribution in place of the Table 3 claim. A revision could pair it with a
   GEOM-Drugs target that *is* a smooth function of the geometry. None of this touches the QM9
   property-guidance results, whose oracles are smooth.
5. **Table 3's remaining paper-vs-records problems are untouched by this work** and still need
   the author decisions the audit lists: both printed percentages and the Tanimoto come from 64
   molecules while App. F.9 claims 4,096, and the 92.0% base PB validity corresponds to no
   measurement of the released checkpoint (the 4,096 x 3 measurement is 88.02%, 95% CI
   [86.52, 89.56], which excludes 92.0).

---

## 8. Files, commands, and what is outstanding

**Created / changed by this thread** (all inside the owned paths):

| file | what |
|---|---|
| `docs/geom_guidance_diagnosis.md` | this document |
| `src/multimfm/geom/diagnostics.py` | added `fd-check` (deterministic 4-stage FD check with a measured noise floor, exact fp32, `--stages` for a head-only mode that needs no allocation) and `fp2-sensitivity`; corrected the false "login node OOM-kills it" docstring |
| `src/multimfm/geom/reward.py` | `fingerprint_prediction`/`fingerprint_reward` take an optional `soft=` override so the same function can be evaluated under `no_grad` for a finite difference. `soft=None` is the default and keeps the original auto-detecting behaviour, so no recorded run changes meaning |
| `src/multimfm/geom/fingerprint.py` | `score()` additionally returns `guide_accuracy_mean` (head prediction vs the molecule's own FP2); purely additive, older JSONs simply lack the key |
| `scripts/geom/gradient_audit.sbatch` | runs `fd-check` then `grad-sign` in one small (2 CPU / 12 GB / 30 min) allocation; documents the real login-node limits |
| `scripts/geom/sign_chain_selfcheck.py` | model-free end-to-end sign test of the production chain (§3.3) |
| `scripts/geom/grad_sign_seeds.sbatch` | repeats the `grad-sign` scan over seeds and reports the paired delta per step with sd, SEM and a t-statistic, plus the matched `+/-` comparison (§2.2). Submitted, still queued |
| `scripts/geom/collect_guidance_grid.py` | aggregates the grid to `results/geom/diagnostics/guidance_grid.csv` and applies the pre-registered bar. Deliberately does **not** rewrite `table3_asdescribed_{tuning.csv,selection.json}`, which document the selection the completed 4,096 x 3 evaluation was run under |
| `scripts/geom/grad_sign.sbatch` | corrected the false login-node claim; added a warning that its default probe steps cannot support a verdict |
| `results/geom/diagnostics/fd_check_reward_only.json` | §3.2 (CPU, exact fp32) |
| `results/geom/diagnostics/fd_check_t0.5_lam0.01_cuda.json` | §3.4, all four stages PASS (GPU, exact fp32, Slurm 49352832) |
| `results/geom/diagnostics/grad_sign_cuda.json` | §2.2, the §7 item 1 diagnostic. **Its `verdict` field carries the old, unsupported string** because the run predates the code fix; a `_verdict_note` key was added to the file saying so. The measurements in it were not altered |
| `results/geom/diagnostics/fp2_sensitivity.json` | §4 |
| `results/geom/diagnostics/guidance_grid.csv` | §5.2, all 16 measured grid points |
| `results/geom/diagnostics/guide_accuracy/` | the CPU guide-accuracy cross-check of §6.3 |

**Reproduce the whole verdict with no allocation** (login node, ~3 min, needs ~2.5 GB free in
the 8 GB user slice):

```
python scripts/geom/sign_chain_selfcheck.py
python -m multimfm.geom.diagnostics fd-check --device cpu --stages reward soft \
    --num-mols 4 --components 12 --rel-steps 1e-5 1e-4 1e-3 1e-2 \
    --out results/geom/diagnostics/fd_check_reward_only.json
python -m multimfm.geom.diagnostics fp2-sensitivity --num-mols 16 --repeats 3 \
    --out results/geom/diagnostics/fp2_sensitivity.json
python scripts/geom/collect_guidance_grid.py
```

**Outstanding, all queue-blocked, none affecting §1:**

* **the `grad-sign` scan's across-draw error bars** (§2.2): Slurm **49354725**, `gpu_test`,
  still `PENDING`. This is the one measurement a reader might reasonably ask for that is
  missing. Until it lands the scan is characterised only by four repeats that share the same
  4 molecules and targets, so its sd column is GPU nondeterminism, not sampling error, and the
  scan must not be quoted in either direction. It does not affect §1, whose checks are either
  exact (§3.3), deterministic with a measured noise floor (§3.2, §3.4), or carry their own
  paired SEM (§5).
* 1 of the 17 grid points (§5.2): the `constant lambda=1 mu=-0.25` sign test, task 6 of array
  **49345752**, held by `JobArrayTaskLimit`. Four matched +-mu pairs are already measured.
* A CPU cross-check of stage 3 on a second device would be nice to have but is not needed:
  the GPU run already disables TF32, so it is true fp32, and its noise floor is 4.8e-7. The
  CPU copy submitted earlier (49344810) was cancelled because its spooled script predated the
  exact-fp32 fix, and a resubmission hit `QOSMaxSubmitJobPerUserLimit`.

**Compute used.** Six Slurm jobs of mine ran: **49344809**, **49349903**, **49351247** and
**49352832** (all `gpu_test`, 1 GPU, ~1 to 2 minutes each) and the two grid arrays
**49345752** / **49345761** (`kempner_requeue`, 1 GPU per task, 16 of 18 tasks completed at
~6 min each, so ~1.7 GPU-hours in total). Three jobs of mine were cancelled as redundant or
stale (49342672, 49345521, 49344810); 1-2 grid tasks of 49345752 may still be queued. Everything in §3.2, §3.3, §4 and the
mu-schedule checks of §6.3 was computed on the login node inside its 1-core / 8-GB user slice.
**No job belonging to anyone else was touched.** 49274410 was already COMPLETED when this
thread began and was only read. No package was installed, and no shared file was edited.
