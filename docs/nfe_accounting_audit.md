# NFE accounting audit: is the compute axis of the two steering comparisons fair?

Owner: nfe-audit agent, 2026-09-30. Data and scripts: `results/shared/nfe_audit/`.

Both steering comparisons put **"NFE (generative model)"** on the x-axis — forward passes of
the generative network per returned sample:

* **DNA, Figure 4 right** (`results/dna/fig4/panelB_curves.csv`): dMFM against best-of-$N$,
  Feynman–Kac, beam search and MCTS, 100 outputs x 3 repeats per point.
* **QM9 steering figure** (`results/qm9/baselines/baseline_sweep_v2.csv`, plotted by
  `scripts/qm9/plot_steering_nfe_matched.py`; **not in the submission as of 2026-09-29**):
  multiMFM and multiMFM-SS against best-of-$N$, Feynman–Kac and DPS, alpha, 100 molecules x
  3 seeds per point.

The guided methods back-propagate through the generative network to get a value gradient; the
search baselines are pure forward. If a forward-plus-backward is billed as one NFE the guided
methods are flattered; if the backward is billed at a training-style 2x forward they are
penalised. This audit reads the counters, **measures what a backward actually costs on both
code paths** (one H200, Slurm 49431669), rebuilds both figures on a forward-equivalent axis,
reports where the crossovers land under every convention tried, and re-times DNA Figure 4 right
in seconds with the batch widths matched.

---

## Answer up front

1. **The published axis is not fair — it flatters the guided methods.** It bills a
   forward-plus-backward as one forward. Every generative forward that the dMFM /
   multiMFM / multiMFM-SS value estimator makes is inside a differentiated graph, so on any
   forward-equivalent axis their curves shift **right** while the search baselines do not move
   at all: at the **measured** cost of a backward (§5b) that shift is **1.54x–2.03x on DNA**
   and **2.56x–2.98x on QM9**.

2. **The conversion factor is now measured, on an H200, on the real code paths** (Slurm
   49431669, `results/shared/nfe_audit/backward_cost_h200.json`):

   * **DNA:** `(forward + backward) / forward` on the dMFM value-gradient call = **2.01, 2.10,
     2.03, 2.14** at batch 25, 100, 400, 1600, i.e. **$k = 1.1$** (backward $\approx$ 1.1
     forwards). The theory bound $k \ge 1$ is nearly tight here: all parameters are frozen and
     only the input gradient is formed, so no weight gradients are computed, and at $L = 50$
     the weight-free attention matmuls are a small share.
   * **QM9:** the whole guided step, *including* the off-axis reward regressor, costs
     **2.89–3.00x** the generative forwards it is billed for (batch 25 and 100, MC 1/4/16).
     Since `axis_k2` $= \text{gen} + 2\,\text{bwd} = 128 + 3\cdot 460\,MC$, the measured
     multiplier of ~3.0 means **`multimfm.nfe`'s existing `nfe_generative_fwd_equiv` column is
     already the right forward-equivalent axis for QM9** — while the same column over-charges
     the DNA dMFM by 1.4x (it assumes 3.0 where the measurement says 2.1).

   Everything below is reported at $k = 0, 1, 1.1, 1.6, 2, 3$ so the verdicts can be read at
   any convention; the measured ones are $k = 1.1$ (DNA) and $k = 2$ (QM9).

3. **DNA, Figure 4 right: the correction is real and it goes against us.** Best baseline is
   Feynman–Kac throughout.

   | axis | crossover (dMFM loses to the best baseline above this) | dMFM settings that win |
   |---|---|---|
   | published (`nfe_gen`, backward free, $k=0$) | between **460 and 824** NFE (log-log interpolated 582) | MC=2, MC=4 (MC=1 had no baseline coverage) |
   | $k=1$ (theory lower bound) | between 278 and 460 (interp. **431**) | **MC=1 only** |
   | **$k=1.1$ (measured, H200)** | between 287 and 478 (interp. **428**) | **MC=1 only** |
   | $k=1.6$ | between 333 and 569 (interp. 415) | MC=1 only |
   | $k=2$ (`nfe_generative_fwd_equiv`) | between 369 and 642 (interp. 407) | MC=1 only |
   | $k=3$ (upper bound) | Feynman–Kac wins over the whole shared range | none |

   At the measured $k = 1.1$ the crossover is **428** NFE against 582 published, and on any
   $k$ in $[1, 3]$ it is 407–431. The dMFM's win region collapses from two of its five measured
   settings to one. The surviving win is large and significant (at 287 forward-equivalent NFE:
   dMFM 0.133 +- 0.013 against 0.198 interpolated for Feynman–Kac, and against 0.315 +- 0.015
   at FK's nearest *cheaper* measured point, z = -9.5), so "**dMFM achieves low target error at
   low NFEs**" survives — but only at MC = 1, and the recorded statement that Feynman–Kac
   "overtakes from 824 NFE" must become "from about 430 forward-equivalent NFE".

4. **QM9: the crossover moves against us by 1.4x, but the conclusion still stands because the
   curves are statistically tied there.** multiMFM-SS's crossover with best-of-$N$ moves
   4,027 ($k=0$) -> 4,250 ($k=1.1$) -> **2,789 (the measured $k=2$)** -> 2,583 ($k=3$); every
   value sits inside the same 2,048-to-8,192 step of the baseline's budget ladder. Reason: the
   multiMFM-SS curve is
   nearly flat (1.38 -> 1.01 Bohr^3 over a 25x budget range) and non-monotone, so sliding it
   right barely changes where it meets best-of-$N$. In the crossing region the two curves are
   **statistically indistinguishable** under every convention: in the conservative
   no-interpolation comparison best-of-$N$ never beats multiMFM-SS by more than 0.93 standard
   errors anywhere inside the plotted 10,000-NFE range (it does at N = 512, i.e. 65,536 NFE,
   which the figure trims away). "multiMFM-SS wins below ~2,000 NFE" is unaffected;
   "best-of-$N$ catches up by ~8,000" is *looser than the data*, and in the wrong direction:
   the crossover is ~4,000 on the published axis and **~2,800 on the measured forward-equivalent
   axis**, never 8,000.

   **Part of that QM9 shift is the reward regressor, which the axis omits entirely.** It has
   **928,263 parameters against the base flow net's 338,121**, and its forward measures
   **1.30–1.39x** a base-net forward on the H200, so it is not the "cheap EGNN" of
   `multimfm/nfe.py`'s docstring. multiMFM-SS makes **28x as many regressor calls as
   best-of-$N$ at the crossover** (920 against 20 per molecule), so charging for it can only
   move the crossover against the guided methods: at 1.0x per call, 4,214 -> 3,532; at 2.74x,
   -> 2,778.

5. **In measured wall-clock the dMFM has no winning regime at all — and the forward-equivalent
   axis is still generous to it.** The stored DNA `elapsed_s` is unusable (the timer starts
   before `load_dmfm_student`, and the dMFM runs 25 rows in flight against the baselines'
   200–3,200), so all five methods were re-timed on one H200 with every checkpoint resident and
   the dMFM at matched width (§5c). Batch matching alone makes the dMFM **3.5–3.9x faster**
   (MC=1: 3.66 s -> 0.93 s per 100 outputs) — a real and large correction in our favour. Even
   so, at the dMFM's cheapest wall-clock point (0.93 s) Feynman–Kac reaches **0.035** and
   best-of-$N$ **0.097** against the dMFM's 0.133, and the dMFM's measured throughput deficit
   at matched width is **2.4x (MC=16) to 3.5x (MC=1)** against the 2.0x / 1.5x that $k = 1.1$
   predicts. The residual is launch overhead — 91 autograd calls inside a 96-step trajectory —
   and it is worst exactly where the paper's claim lives. On QM9 wall-clock is recorded for the
   guided runs only, so no same-hardware seconds comparison exists there; this audit does not
   invent one.

Nothing below is a convention invented to flatter us. The conversion factor was measured rather
than chosen, every convention tried is reported side by side (§3), and where the measurement
went *our* way — the existing `nfe_generative_fwd_equiv` over-charges the DNA dMFM by 1.4x, and
batch matching makes it 3.5–3.9x faster in seconds — that is recorded too, alongside the two
places where it went against us.

---

## 1. What is actually counted, per method

Read off the counters, not the names: `src/multimfm/nfe.py` (QM9, forward hooks on the
modules) and `dmfm.benchmarks.core.NFECounter` (DNA, manual increments at each call site).

### 1a. DNA (`results/dna/fig4/panelB_runs.csv`, per returned sample)

| method | base fwd | student fwd | on the plotted axis | C0 guide calls | backward |
|---|---|---|---|---|---|
| dMFM, MC | 96 | 91·MC | 96 + 91·MC | 91·MC + 1 | **91** (see below) |
| best-of-$N$ | 96·N | 0 | 96·N | N | 0 |
| Feynman–Kac, K | 96·K | 0 | 96·K | 12·K | 0 |
| beam, W=1 L=4 K | $4\sum_{m<96/K}(96-mK)$ = 576–4,800 | 0 | same | 4·(96/K) = 8–96 | 0 |
| MCTS, I | data-dependent, 192–1,725 (varies across repeats) | 0 | same | I | 0 |

* The axis is `nfe_gen = base + student`. It is **measured**, and it is internally consistent:
  the base DiT (2,912,836 params) and the dMFM student (3,420,484 params) are within 17% of
  each other, so one forward of either costs about the same.
* Feynman–Kac pays **no** generative NFE for its look-ahead: `BaseFlow.step` returns the
  denoiser prediction `psi_t(x) = E[X_1|X_t]` from the forward pass it already made
  (`core.py`), so its potential is free. This is honest and is the best case for FK.
* **The reward model is counted but is off the axis.** `nfe_guide_per_output` is recorded for
  every method. The C0 guide has 88,773 parameters, 3.0% of the base DiT, and the dMFM makes
  far *more* guide calls than the baselines (1,457 at MC=16 against best-of-$N$'s 32), so
  putting the guide on the axis at 3% of a generative forward moves the Feynman–Kac crossover
  from 431.1 to 431.9 forward-equivalent NFE — nothing
  (`results/shared/nfe_audit/reward_model_sensitivity.csv`, also at 10% and 100%).
* **`nfe_backward_per_output` is in different units from everything else.**
  `dmfm_value_and_grad` does `nfe.backward += B` — **one count per (output, guided step),
  independent of MC and of `nfe_value`** — so it reads 91 at every MC from 1 to 16. But that
  one `torch.autograd.grad` call differentiates a graph containing all 91·MC student forwards
  and all 91·MC guide forwards for that output. At MC=16 the number 91 understates the
  back-propagated work by 16x. The paper's caption sentence "the dMFM additionally performs
  one backward pass per guided step" is literally true and materially misleading; see §6.
* A docstring error found while reading, not load-bearing for the axis:
  `dmfm/benchmarks/core.py` calls the base DiT and the student "33 M params" — the loaded
  L=50 models have **2.91 M** and **3.42 M** (checkpoint `model_state_dict` numel;
  `hidden_dim 192`, `dit_blocks 4`, `transformer_heads 6`). The 35 MB checkpoint file is
  weights plus optimizer state, which is presumably where "33 M" came from.

### 1b. QM9 (`results/qm9/baselines/runs_v2/seed*/…`, per generated molecule)

| method | generative fwd | back-propagated | regressor calls (all phases) | PoseBusters checks |
|---|---|---|---|---|
| base | 128 | 0 | 2 (eval) | 0 |
| best-of-$N$ | 128·N | 0 | **N** + 4 | 2.0 |
| Feynman–Kac, K | 128·K + 15·K | 0 | 16·K + 4 | 2.0–2.6 |
| DPS, guide_every=g | 128 + 115/g | **115/g** | 115/g + 2 | 0 |
| multiMFM, MC | 128 + 460·MC | **460·MC** | 115·MC + 10 | — |
| multiMFM-SS, MC | 128 + 460·MC (extra = 0) | **460·MC** | **230·MC** + 10 | 7.7–26 |

* The hooks count *rows per forward call* and mark a call back-propagated when autograd is on
  and an input requires grad. For the guided methods **every** MFM forward is back-propagated
  (460·MC of 460·MC), and 460·MC = 4 inner MFM steps x 115 guided steps x MC.
* Feynman–Kac **does** pay for its look-ahead here (15·K extra base forwards, from 15
  resampling events), unlike on DNA. Counted, on the axis.
* DPS pays one extra back-propagated base forward per guided step: 243 forwards of which 115
  are back-propagated at `guide_every = 1`.
* **The reward model is off the axis and is not cheap.** The QM9 guide regressor has
  **928,263** parameters against the base flow net's 338,121 and the MFM student's 363,465 —
  2.7x the generative network by parameter count, not the "cheap EGNN" of
  `multimfm/nfe.py`'s docstring. best-of-$N$ needs one regressor call per candidate
  (N per molecule, 4–512), multiMFM-SS needs 230·MC (230–7,360) — **28x more at the crossover**
  (N=16: 20 calls; MC=4: 920 calls). So including the reward model can only move the crossover
  *against* the guided methods: at cost ratio 1.0 per call the multiMFM-SS/best-of-$N$
  crossover moves 4,215 -> 3,532, at the parameter-count ratio 2.74 -> 2,778
  (`reward_model_sensitivity.csv`). The measured cost (§5b, H200) is **1.30–1.39x** a base-net
  forward — below the parameter ratio but well above "cheap", so the 1.0x row of that table is
  the closest bracket and the crossover moves 4,214 -> ~3,450.
* **`baseline_sweep_v2.csv`'s `nfe_regressor` column is not comparable across methods.**
  `qm9_results.py:839` takes the *trajectory phase only* for `steer_search` rows but the
  total for the baselines, so the CSV shows 1,840 for multiMFM-SS at MC=16 where the run made
  3,781 regressor calls (1,840 trajectory + 1,931 selection + 10 eval). The audit reads the
  per-phase numbers out of the run JSONs instead. Also: one `--lookahead-source both` job
  produces both the `final_endpoint` (multiMFM) and `best_lookahead_*` (multiMFM-SS) rows, so
  the selection-phase cost in that JSON belongs to the SS variants only.
* multiMFM-SS's entire advantage over multiMFM is bought **off the plotted axis**: identical
  generative NFE (`extra_generative_vs_multimfm = 0.0`), 2x the regressor calls, and 4–13x the
  PoseBusters checks (7.7 against best-of-$N$'s 2.0 per molecule).

---

## 2. The forward-equivalent axis, and what a backward actually costs

Define, per returned sample,

```
axis_k  =  (generative forward passes)  +  k * (generative forward passes inside a
                                               differentiated graph)
```

`k = 0` is the published axis. The columns needed already exist on QM9
(`nfe_generative`, `nfe_generative_bwd`); on DNA the in-graph count is exactly
`nfe_student_per_output`, because `dmfm_value_and_grad` runs the student under
`torch.enable_grad()` from a leaf that requires grad and the 96 base steps run under
`@torch.no_grad()`.

**Why k is between 1 and 2.** The usual "backward costs 2x forward"
applies to *training*, where the backward computes both input gradients (`dL/dx = dL/dy·W^T`)
and weight gradients (`dL/dW = x^T·dL/dy`) — two matmuls per one in the forward. Guidance
needs only the input gradient. And in both codebases every model parameter is explicitly
frozen, so autograd does not build the weight-gradient half at all:

* DNA: `load_dmfm_student` sets `requires_grad_(False)` on all 78 student parameter tensors,
  `load_c0` on all 8 guide tensors; the call is
  `torch.autograd.grad(V.sum(), x_leaf)` (`dmfm_steering.py`).
* QM9: `glass_guidance.main` freezes the base model, `load_mfm_student` the 94 student
  tensors, and the regressor's 97 tensors; the call is
  `torch.autograd.grad(value_sum, [coords, atomics])`.

That argument gives `k = 1` as a *lower bound*: a lower bound rather than the answer because
attention's `QK^T` and `PV` are activation-times-activation matmuls carrying no weights at all,
so freezing parameters saves nothing there and their backward still needs two matmuls per
forward one. Both architectures here are short-sequence (DNA L = 50, QM9 <= 29 atoms), so the
weighted projection and feed-forward matmuls dominate and `k` should sit nearer 1 than 2.
**§5b measures it: k = 1.1 on the DNA value-gradient path** (ratio 2.01–2.14 over batch
25–1600, H200), and on QM9 the whole guided step including its off-axis regressor costs
2.9–3.0x its billed forwards, which is exactly `axis_k2`. Five values are carried through so
any verdict can be read at any convention:

* `k = 0` — the published axis;
* `k = 1` — theory lower bound (every matmul weighted);
* **`k = 1.1` — measured on DNA**;
* `k = 1.6` — a midpoint, kept for continuity with this document's first draft;
* `k = 2` — `multimfm.nfe`'s existing `nfe_generative_fwd_equiv`, and **the measured axis for
  QM9**;
* `k = 3` — upper bound.

**The verdicts in §3 are identical for every `k` in [1, 3] on DNA**, so none of them turns on
the exact value.

Per-point axis values (`results/shared/nfe_audit/{dna,qm9}_axes.csv`):

| DNA dMFM | k=0 | k=1 | **k=1.1 (measured)** | k=1.6 | k=2 | k=3 | MAE |
|---|---|---|---|---|---|---|---|
| MC=1 | 187 | 278 | **287** | 333 | 369 | 460 | 0.1329 +- 0.0125 |
| MC=2 | 278 | 460 | **478** | 569 | 642 | 824 | 0.1206 +- 0.0055 |
| MC=4 | 460 | 824 | **860** | 1042 | 1188 | 1552 | 0.0947 +- 0.0042 |
| MC=8 | 824 | 1552 | **1625** | 1989 | 2280 | 3008 | 0.0736 +- 0.0014 |
| MC=16 | 1552 | 3008 | **3154** | 3882 | 4464 | 5920 | 0.0613 +- 0.0026 |

(the four search baselines are unchanged by k, by construction)

| QM9 multiMFM / -SS | k=0 | k=1 | k=1.1 | k=1.6 | **k=2 (measured)** | k=3 | SS oracle MAE |
|---|---|---|---|---|---|---|---|
| MC=1 | 588 | 1048 | 1094 | 1324 | **1508** | 1968 | 1.381 +- 0.063 |
| MC=2 | 1048 | 1968 | 2060 | 2520 | **2888** | 3808 | 1.290 +- 0.049 |
| MC=4 | 1968 | 3808 | 3992 | 4912 | **5648** | 7488 | 1.069 +- 0.035 |
| MC=8 | 3808 | 7488 | 7856 | 9696 | **11168** | 14848 | 1.094 +- 0.088 |
| MC=16 | 7488 | 14848 | 15584 | 19264 | **22208** | 29568 | 1.006 +- 0.045 |
| MC=32 | 14848 | 29568 | 31040 | 38400 | **44288** | 59008 | 1.006 +- 0.025 |

DPS also moves (143–243 -> 158–358 at k=1), but its budget knob saturates below the shared
ladder, so it never overlaps the search baselines and no crossover is defined for it.

---

## 3. Where the crossovers land

`crossovers.csv` (log-log interpolated), `matched_budget.csv` (every search curve read at each
guided point's budget, the construction of `summary.json -> panelB.matched_nfe_comparison`),
`conservative_measured.csv` (no interpolation at all: each search method at its largest
*measured* budget <= the guided point, so the baseline never gets more compute).

### 3a. DNA, Figure 4 right — matched-budget verdict per dMFM setting

Best baseline at the same budget; `+` = dMFM better.

| dMFM | k=0 (published) | k=1 | **k=1.1 (measured)** | k=2 |
|---|---|---|---|---|
| MC=1 | 187 — *no baseline coverage* | 278 / FK 0.205 → **+** | **287 / FK 0.198 → +** | 369 / FK 0.147 → **+** |
| MC=2 | 278 / FK 0.205 → **+** | 460 / FK 0.113 → − | **478 / FK 0.108 → −** | 642 / FK 0.076 → − |
| MC=4 | 460 / FK 0.113 → **+** | 824 / FK 0.057 → − | **860 / FK 0.054 → −** | 1188 / FK 0.037 → − |
| MC=8 | 824 / FK 0.057 → − | 1552 / FK 0.027 → − | **1625 / FK 0.025 → −** | 2280 / FK 0.016 → − |
| MC=16 | 1552 / FK 0.027 → − | 3008 / FK 0.011 → − | **3154 / beam 0.022 → −** | 4464 / beam 0.015 → − |

Interpolated crossovers, dMFM against each baseline:

| baseline | k=0 | k=1 | **k=1.1 (measured)** | k=1.6 | k=2 | k=3 |
|---|---|---|---|---|---|---|
| Feynman–Kac | 582 | 431 | **428** | 415 | 407 | baseline wins everywhere |
| beam search | 950 | 775 | **763** | 713 | 682 | 654 |
| MCTS | 1154 | 891 | **873** | 794 | 748 | 696 |
| best-of-$N$ | dMFM better over the whole shared range | 1503 | **1463** | 1300 | 1204 | 1034 |

The k=0 column reproduces the published reading: `summary.json` reads the baselines only *at*
the dMFM's five budgets, which is why the recorded statement is "Feynman–Kac overtakes from
824"; the continuous crossover is at 582, between the measured budgets 480 and 864.

Two structural consequences worth stating plainly, both independent of the interpolation:

* **The dMFM loses its exclusive low-budget point.** On the published axis its cheapest
  setting sits at 187 NFE, *below* every baseline's cheapest point (192), which is why
  `summary.json` records `NaN` for all four baselines there and why the "low NFE regime" reads
  as uncontested. On the forward-equivalent axis MC=1 lands at 278, inside the baselines'
  range (287 at the measured k = 1.1), where the comparison can actually be made — and the
  dMFM still wins it, 0.133 +- 0.013 against Feynman–Kac's 0.198. That is a better result for us
  than the `NaN`, and it should be reported that way.
* **The dMFM's ladder has no knob below MC=1.** 96 base steps + one posterior sample at each
  of 91 guided steps is its floor, so "the low-NFE regime" is a single measured point rather
  than a regime. The natural way to extend the curve left is a guidance-frequency knob, which
  DPS already has on QM9 (`guide_every` 1/2/4/8) and the DNA dMFM sweep does not. Adding it
  would make the claim a claim about a region instead of a point.

**No-interpolation check** (each baseline at its largest measured budget not exceeding the
dMFM's, so the baseline is handicapped): at k=1 the dMFM still beats the nearest *cheaper*
FK point at MC=1 (z = -9.5), MC=2 (z = -4.9) and MC=4 (z = -2.1), and loses from MC=8
(z = +10.9). Notice how much of the MC=2/MC=4 win is budget slack: FK's next point up, at
864 forward-equivalent NFE — 5% more budget than the dMFM's MC=4 point at 860 (k = 1.1) —
scores 0.0535 against the dMFM's 0.0947, 43% better. That is why the interpolated crossover
(428 at the measured k) is the honest number and the conservative one is not.

### 3b. QM9 — matched-budget verdict per multiMFM-SS setting

| multiMFM-SS | k=0 (published) | k=1 | k=1.1 | **k=2 (measured)** |
|---|---|---|---|---|
| MC=1 | 588 / 3.553 → **+** | 1048 / 2.261 → **+** | 1094 / 2.203 → **+** | **1508 / 1.810 → +** |
| MC=2 | 1048 / 2.261 → **+** | 1968 / 1.538 → **+** | 2060 / 1.496 → **+** | **2888 / 1.273 → −** |
| MC=4 | 1968 / 1.538 → **+** | 3808 / 1.116 → **+** | 3992 / 1.091 → **+** | **5648 / 1.028 → −** |
| MC=8 | 3808 / 1.116 → **+** | 7488 / 0.986 → − | 7856 / 0.979 → − | **11168 / 0.951 → −** |
| MC=16 | 7488 / 0.986 → − | 14848 / 0.931 → − | 15584 / 0.928 → − | **22208 / 0.882 → −** |
| MC=32 | 14848 / 0.931 → − | 29568 / 0.844 → − | 31040 / 0.838 → − | **44288 / 0.826 → −** |

(the MC=2 and MC=4 flips at k=2 are 1.290-vs-1.273 and 1.069-vs-1.028 — 0.35 and 1.2 SE, i.e.
not resolved; the no-interpolation check below puts multiMFM-SS *ahead* at MC=2 with z = -2.65)

Interpolated crossover with best-of-$N$: **4,027 (k=0) -> 4,215 (k=1) -> 4,250 (k=1.1) ->
3,283 (k=1.6) -> 2,789 (k=2, the measured axis) -> 2,583 (k=3)**. The k=1 value is *above* the k=0 value; that is not an error, it is what
happens when a nearly flat, non-monotone curve (MC=4 1.069, MC=8 1.094 — a 0.3-sigma bump) is
slid to the right against a 2x-spaced baseline ladder. The crossover on QM9 is simply not
resolved by these data:

* best-of-$N$ is never significantly better than multiMFM-SS anywhere inside the plotted
  range, on any axis. In the no-interpolation check, over the settings the figure actually
  shows (MC <= 16, N <= 64) the largest z in best-of-$N$'s favour is **+1.28** (MC=8 at
  k = 1.6 and k = 2); at k = 1 it is +0.60, and at k = 0 best-of-$N$ is not ahead at any
  plotted setting at all (max z = -0.94). It becomes significant only off the right-hand edge:
  z = +4.49 at MC=32 against N=512, i.e. 32,768–44,288 forward-equivalent NFE against the
  figure's 10,000 cut.
* multiMFM-SS is decisively better at the low end on every axis, though how far up that
  extends does depend on k: z = -13.9 / -19.3 / -6.0 at MC = 1 / 2 / 4 for k = 1, and
  -13.9 / -2.6 / -0.11 for k = 1.6 and k = 2. So the cheapest one or two settings are a clear
  multiMFM-SS win under any convention; MC=4 is a win at k <= 1 and a tie above.
* Feynman–Kac on QM9 is beaten by both guided methods over the whole shared range under every
  convention — it is far off this pace here (3.17 Bohr^3 at K=32) unlike on DNA.

So on QM9 the honest statement is a two-sided one, and it holds on the measured axis:
**multiMFM-SS wins clearly below ~2,000 forward-equivalent NFE, and from ~2,800 onward it and
best-of-$N$ are statistically indistinguishable out to the end of the plotted range.**

---

## 4. Memory / batching fairness

Rows in flight at the drivers' defaults (`dna_wallclock.csv` column `rows_in_flight`):

| DNA | width |
|---|---|
| dMFM | `--dmfm-batch 25`: **25** base-flow rows, 25·MC student rows (25–400) |
| best-of-$N$ | `--batch-rows 2048`: `(2048//N)*N` = 200–2,048 |
| Feynman–Kac | `n_outputs·K` in one batch, **no chunking**: 200–3,200 |
| beam | `n_outputs·W·L` = 400 |
| MCTS | `n_outputs` = 100 per rollout, plus a Python loop over 100 trees per iteration |

This is an 8–128x width gap in the baselines' favour, entirely from driver defaults. It does
not touch the NFE counts (they are per-row) but it makes the stored seconds meaningless as a
fairness measure, and it is the reason §5 re-times at matched width.

On QM9 the gap is much smaller and runs the other way at the top of the ladder: best-of-$N$
chunks at `--batch-size 8192`, multiMFM/-SS run `num_samples·MC` = 100–3,200 MFM rows in one
batch (`value_batch_size = 0`). Peak GPU memory is recorded for the guided runs only:
0.227 GB (MC=1) to 6.42 GB (MC=32) on an A100-40GB. best-of-$N$ at N=512 holds 8,192
trajectories x 128 steps of nothing but activations under `no_grad`, so its peak is small; it
is not recorded, which is a gap in the record rather than a finding.

---

## 5. Wall-clock

### 5a. What is recorded

* **DNA: yes** — `panelB_runs.csv` has `elapsed_s` per run (100 outputs, 3 repeats). **But it
  is not usable as it stands:** `run_steering.py:206` sets `t0` *before*
  `load_dmfm_student(...)` on line 208, so every dMFM row includes a 3.4 M-parameter
  checkpoint load from disk that no baseline pays. That is visible in the numbers: the dMFM's
  five points sit at 5.20, 5.43, 5.58, 5.83, 8.71 s while their NFE ranges over 8.3x, i.e.
  the series is dominated by a constant. Taken at face value the stored seconds say the dMFM
  is 3.4k–17.8k generative NFE/s against 44k–61k for the batched baselines. **That reading is
  wrong and is not a finding of this audit**: the checkpoint load and the 25-row batch width
  are both inside it, and the re-timing in §5c shows that removing them makes the dMFM
  3.5–3.9x faster.
* **QM9: only for the guided half.** `steer_search` writes `timing_s`
  (`trajectory` / `selection_and_eval` / `total`) and `cuda_max_memory_gb`;
  `baselines.py` (base, best-of-$N$, DPS, FK) writes no timing at all. So there is no
  same-hardware seconds comparison available from the stored QM9 records, and this audit does
  not guess one. For the record, the guided QM9 totals on an A100-40GB at n = 100 are
  multiMFM-SS 38.96 s (MC=1), 22.41 (MC=4), 24.51 (MC=16), 33.53 (MC=32) — again dominated by
  a term that is not the generative NFE (the MC=1 run spends 24.2 s of its 39 s in
  `selection_and_eval`, i.e. PoseBusters).

### 5b. Measured: what a guidance backward actually costs

`results/shared/nfe_audit/measure_backward_cost.py`, Slurm **49431669**, one NVIDIA H200,
TF32 on, torch 2.5.1+cu124, all checkpoints resident before any timer starts. Raw:
`backward_cost_h200.json`, log `logs/nfe_audit_h200-49431669.out`.

**DNA — `dmfm_value_and_grad` against the same posterior draw under `no_grad`:**

| batch | forward | forward + backward | ratio | implied k |
|---|---|---|---|---|
| 25 | 3.40 ms | 6.83 ms | 2.01 | 1.01 |
| 100 | 3.34 ms | 7.01 ms | 2.10 | 1.10 |
| 400 | 4.17 ms | 8.47 ms | 2.03 | 1.03 |
| 1600 | 12.02 ms | 25.75 ms | 2.14 | 1.14 |

So **k = 1.1 on DNA**, flat in batch width. The theory bound `k >= 1` is nearly tight, as
expected for `L = 50` where the weight-free attention matmuls are a small share of the work.
`multimfm.nfe`'s `nfe_generative_fwd_equiv` (which assumes 3.0) over-charges the DNA dMFM by
1.4x.

**QM9 — `value_gradient` (MFM posterior + regressor + backward through both) against the MFM
posterior forward alone:**

| batch, MC | forward | value_gradient | ratio |
|---|---|---|---|
| 25, MC=1 | 14.26 ms | 41.18 ms | 2.89 |
| 25, MC=4 | 13.83 ms | 41.50 ms | 3.00 |
| 25, MC=16 | 13.73 ms | 40.80 ms | 2.97 |
| 100, MC=1 | 13.87 ms | 41.50 ms | 2.99 |
| 100, MC=4 | 13.69 ms | 40.80 ms | 2.98 |
| 100, MC=16 | 14.61 ms | 57.24 ms | 3.92 |

The whole QM9 guided step — backward *and* the off-axis reward regressor — costs **2.9–3.0x**
the generative forwards it is billed for, flat in MC because the regressor is called once per
posterior sample, the same scaling as the MFM forwards. (The 3.92 at batch 100 / MC=16, the
largest allocation in the sweep, is the one outlier.) Since
`axis_k2 = gen + 2*bwd = 128 + 3 * 460 * MC`, a multiplier of 3.0 means **`multimfm.nfe`'s
existing `nfe_generative_fwd_equiv` column is already the correct forward-equivalent axis for
QM9.**

**QM9 regressor cost, measured:** one regressor forward = **1.39x** (batch 100) / **1.30x**
(batch 400) one base-net forward. Below the 2.74x its parameter count suggests, but far above
"cheap": it is more expensive than a generative forward and it is not on the axis at all.

### 5c. Measured: DNA Figure 4 right re-timed, same GPU, batch matched

100 outputs per call, median of 3 after a warm-up, every checkpoint already on the device.
`dna_wallclock_rerun_h200.csv`.

| method | NFE/out | rows in flight | seconds | NFE/s | MAE |
|---|---|---|---|---|---|
| dMFM MC=1, `--dmfm-batch 25` (shipped) | 187 | 25 | 3.657 | 5,113 | 0.133 |
| **dMFM MC=1, batch 100** | 187 | 100 | **0.933** | 20,034 | 0.133 |
| dMFM MC=4, batch 25 | 460 | 100 | 3.726 | 12,344 | 0.095 |
| **dMFM MC=4, batch 100** | 460 | 400 | **1.056** | 43,541 | 0.095 |
| dMFM MC=16, batch 25 | 1552 | 400 | 4.203 | 36,922 | 0.061 |
| **dMFM MC=16, batch 100** | 1552 | 1600 | **2.631** | 58,998 | 0.061 |
| best-of-N N=2 | 192 | 200 | 0.271 | 70,844 | 0.408 |
| best-of-N N=5 | 480 | 500 | 0.404 | 118,666 | 0.206 |
| best-of-N N=9 | 864 | 900 | 0.668 | 129,246 | 0.134 |
| best-of-N N=32 | 3072 | 2048 | 2.193 | 140,055 | 0.043 |
| FK K=2 | 192 | 200 | 0.302 | 63,678 | 0.315 |
| FK K=5 | 480 | 500 | 0.428 | 112,054 | 0.107 |
| FK K=9 | 864 | 900 | 0.694 | 124,577 | 0.053 |
| FK K=32 | 3072 | 3200 | 2.099 | 146,389 | 0.011 |
| beam K=48 | 576 | 400 | 0.546 | 105,555 | 0.152 |
| beam K=16 | 1344 | 400 | 1.288 | 104,348 | 0.058 |
| beam K=4 | 4800 | 400 | 4.626 | 103,755 | 0.014 |

Three results, in order of size:

1. **Batch matching is worth 3.5–3.9x to the dMFM** (MC=1: 3.657 s -> 0.933 s; MC=4: 3.726 ->
   1.056; MC=16: 4.203 -> 2.631). The stored `panelB_runs.csv` seconds are both
   checkpoint-load-inflated and width-handicapped, so any seconds-based claim built on them
   would have overstated the dMFM's cost by roughly 4x. The naive reading of the stored file
   ("the dMFM is 13x slower per NFE") is wrong and is not reported anywhere in this audit.
2. **The dMFM still has no wall-clock regime where it wins.** Baselines interpolated (log-log)
   to each dMFM point's own seconds:

   | dMFM at batch 100 | seconds | dMFM MAE | FK | best-of-N | beam |
   |---|---|---|---|---|---|
   | MC=1 | 0.933 | 0.1329 | **0.0347** | **0.0974** | **0.0836** |
   | MC=4 | 1.056 | 0.0947 | **0.0290** | **0.0866** | **0.0729** |
   | MC=16 | 2.631 | 0.0613 | — | — | **0.0266** |

   Every baseline is better at every dMFM point. At MC=16 the two search ladders stop short of
   2.631 s, so no interpolation is needed at all: FK's *most expensive* measured point costs
   **2.099 s** for 0.0107 and best-of-N's costs **2.193 s** for 0.0434 — both cheaper in seconds
   and better in error than the dMFM's 2.631 s / 0.0613. And the dMFM cannot be moved left:
   MC=1 is its floor and batch 100 already puts all 100 outputs in one chunk.
3. **The forward-equivalent axis is still generous to the dMFM at low MC.** At matched width
   its throughput deficit against the baselines is **3.5x at MC=1** and **2.4x at MC=16**, where
   `k = 1.1` predicts only 1.5x and 2.0x. The residual is per-call overhead: 91
   `torch.autograd.grad` calls and 91 separate 100-row student forwards inside a 96-step
   trajectory, against one 200-row forward per step for best-of-N. It is worst at low MC —
   exactly the regime the paper's claim rests on. A batched-over-MC-and-step implementation
   would close some of it; nothing in the current code does.

MCTS was left out of the re-timing (its Python per-tree loop makes it the slowest method by far
in the stored data, 26 s at 1,725 NFE, and it is never the best baseline at any budget).

---

## 6. Exactly which sentences change

Owner of the tex file is the tex agent; owner of the canonical Figure 4 wording is
`results/dna/fig4/paper_sentences.md` (dna-fig4). Proposed edits, all DNA:

**(a) Figure 4 caption, the NFE sentence.** Currently (canonical file, §3):

> NFEs count forward passes of a generative network and assume no parallel execution; they are
> not wall-clock, and the dMFM additionally performs one backward pass per guided step.

The final clause is true per autograd call and misleading per unit of work: that one call
differentiates a graph holding all $MC$ posterior-sample forwards of that step. Replace with:

> NFEs count forward passes of a generative network and assume no parallel execution; they are
> not wall-clock. The dMFM additionally back-propagates through **every** posterior sample it
> draws: all $91\,MC$ of its student forwards sit inside the differentiated graph. We measure
> one such backward at $1.1$ forward passes (forward-plus-backward $2.01$–$2.14\times$ forward
> over batch sizes $25$–$1600$; all parameters are frozen and only the input gradient is formed,
> so no weight gradients are computed), which puts the dMFM's forward-equivalent budgets at
> $287$–$3{,}154$ rather than the $187$–$1{,}552$ plotted.

**(b) §3.3 "Steering", the low-NFE claim.** Currently:

> dMFM achieves low target error at low NFEs, the regime most relevant for practical sampling.

Keep the claim, bound it:

> At the lowest budget the dMFM reaches it attains the lowest target error of any method:
> $0.133\pm0.013$ against $0.198$ for the best baseline interpolated to the same
> forward-equivalent budget, and $0.315\pm0.015$ at that baseline's nearest cheaper measured
> point. Above roughly $430$ forward-equivalent NFE Feynman--Kac steering is ahead, and by
> $3{,}000$ it is a factor of five better, so the dMFM's advantage is confined to its cheapest
> setting rather than to a regime. In measured wall-clock on one GPU, with matched batch widths,
> the search baselines are ahead at every budget.

**(c) `docs/STATE_OF_CLAIMS.md`, the "Fig 4 right" row.** It currently reads "Feynman–Kac
overtakes dMFM from 824 NFE up" and calls the low-NFE claim "Survives as stated, but only at
low NFE". On the measured forward-equivalent axis the overtake is at **428** (431 at $k=1$,
407 at $k=2$) and the win region is one setting (MC=1) rather than two; in measured wall-clock
at matched batch width there is no winning region at all. I have not edited that file (not
mine); the change is one sentence and the numbers are in
`results/shared/nfe_audit/crossovers.csv` and `dna_wallclock_rerun_h200.csv`.

**(d) QM9: the figure is not in the submission, so nothing published changes — but if it
ships, two things must.** The plot script's own claim ("multiMFM-SS wins below ~2,000 NFE,
best-of-$N$ catches up by ~8,000") should become "**wins below ~2,000 forward-equivalent NFE,
and is statistically indistinguishable from best-of-$N$ from ~2,800 onward**": the crossover is
~4,000 on the published axis and ~2,800 on the measured one, never 8,000. And the axis should be
labelled `NFE (forward-equivalent)` with the convention stated — here that is exactly the
existing `nfe_generative_fwd_equiv` column, which the measurement vindicates (guided step =
2.9–3.0x its billed forwards). The caption should also say that the reward regressor
(928 k parameters; one forward = 1.3–1.4x a generative forward, measured) is excluded from the
axis and that multiMFM-SS makes ~28x as many regressor calls as best-of-$N$ at the crossover.

---

## 7. Reproducing this

```bash
# stored-data analysis (CPU, seconds); writes every CSV cited above
~/micromamba/envs/multimfm/bin/python results/shared/nfe_audit/recompute_axes.py

# the measurements of §5b/§5c (1 GPU, ~4 min). Job 49431669 (kempner_h200, NVIDIA H200) is
# the one behind every measured number here; its twin 49431647 on kempner_requeue was
# cancelled once 49431669 finished, so backward_cost.json / dna_wallclock_rerun.csv (the
# untagged names) do not exist -- only the _h200 files.
sbatch results/shared/nfe_audit/measure_backward_cost_h200.sbatch   # AUDIT_TAG=_h200
sbatch results/shared/nfe_audit/measure_backward_cost.sbatch        # kempner_requeue, untagged
```

Do **not** run `measure_backward_cost.py` on a login node: the interactive node allocates one
core (`nproc` = 1) at a load average above 30, and the ratios come out unusable — two runs of
the identical DNA measurement there gave 2.60 and 1.33 where the H200 gives a stable 2.01–2.14.

Outputs in `results/shared/nfe_audit/`: `dna_axes.csv`, `qm9_axes.csv`,
`qm9_axes_untrimmed.csv`, `crossovers.csv`, `matched_budget.csv`,
`conservative_measured.csv`, `reward_model_sensitivity.csv`, `dna_wallclock.csv`,
`summary.json`, `recompute_axes_output.txt` (the full printed transcript of the run behind
every table in this document), and from the GPU job `backward_cost_h200.json` and
`dna_wallclock_rerun_h200.csv`.

Conventions tried and reported, in full: backward billed at 0x (the published axis), 1x
(lower bound), 1.6x, 2x (`multimfm.nfe`'s existing `nfe_generative_fwd_equiv`) and 3x a
forward;
reward model billed at 0 / 3% / 10% / 100% of a generative forward on DNA and
0 / 25% / 100% / 274% on QM9; QM9 ladders both trimmed as plotted and untrimmed; crossovers
by log-log interpolation, by reading baselines at the guided budgets, and with no
interpolation at all.
