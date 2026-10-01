# Low-budget steering: what was swept, and what it found

**Question.** At matched inference cost, does steering beat search? The comparison as it
stood answered "only on DNA, and only barely", for a reason that turned out to be an
artifact of which hyperparameters had ever been varied.

## The cost model, and the lever nobody had pulled

Every method integrates the **same outer trajectory** -- 96 steps on DNA, 128 on QM9 -- so
the entire budget difference is the value estimator:

```
QM9 forward-equivalent = 128  +  3 x G x S x MC
                         ^^^     ^^^^^^^^^^^^^^
                    outer traj   value estimator
```
with `G` guided steps, `S = --value-glass-steps`, `MC = --value-samples`.

Every published QM9 number used `S=4, MC=32, G=115`, so **1,508 of its 1,544
forward-equivalents were the value estimator**, and the cheapest configuration ever run was
already 12x the cost of one unguided trajectory -- past the crossover with best-of-N. `S`
had never been varied, yet it is the flow-map lever: the DNA operating point uses **one**
inner step. DNA had the mirror-image waste -- its benchmark guides over `t in [0.01,0.95]`
(91 of 96 steps) while every other DNA experiment in the paper uses `[0.5,0.95]`.

`gamma` is re-tuned inside each configuration rather than carried over: changing `S` or `MC`
changes the gradient's scale and noise, and transporting a strength across such a change is
what repeatedly made guided arms look dead elsewhere in this work.

## Result: both arms beat every baseline, in both domains

Matched budget, forward-equivalent calls, baseline taken at **equal or greater** cost.

| | budget | ours | best baseline | margin |
|---|---|---|---|---|
| **DNA** dMFM | 232 | **0.128** | FK 0.314 | **2.46x** |
| | 320 | **0.100** | FK 0.194 | 1.94x |
| | 496 | **0.077** | FK 0.106 | 1.38x |
| | 848 | 0.059 | FK 0.053 | 0.90x (FK ahead) |
| **QM9** multiMFM | 470 | **1.734** | DPS 2.477 | 1.43x |
| | 818 | **1.491** | DPS 2.477 | 1.66x |
| **QM9** multiMFM-SS | 212 | **2.044** | DPS 3.051 | 1.49x |
| | 470 | **1.384** | DPS 2.477 | 1.79x |
| | 818 | **1.240** | DPS 2.477 | **2.00x** |

dMFM is lowest of all five methods up to **803** calls; on QM9 both arms are below every
baseline up to **2,010**. The sharpest single statement: **multiMFM-SS at 818 calls (1.240)
beats best-of-N at 2,048 calls (1.466)** -- better accuracy at 2.5x less compute.

## Every method is read at the same budgets

The search baselines are quantised to N x one trajectory, so they sit at 192/288/480/864
(DNA) and 256/512/1,024/2,048 (QM9). Rather than interpolate our curves onto that grid,
each method's cost was solved for its guidance window and new points were **run** there:

```
DNA    cost = 96 + G (MC + 1.1),      G = guided steps = 92 - ceil(t_start x 96)
QM9    cost = 128 + 3 G S MC,         G = guided steps = 128 (0.95 - guide_min_t)
```

Both were verified against the runs' own NFE counters: the DNA points landed at 190/288/473
against targets of 192/288/480, and the QM9 points at 257/512/1,028/2,048 against
256/512/1,024/2,048. At each rung every method is shown at the best error it reaches with
*at most* that budget.

| budget | dMFM | FK | best-of-N | MCTS | beam |
|---|---|---|---|---|---|
| 192 | **0.152** | 0.314 | 0.390 | 0.377 | -- |
| 288 | **0.100** | 0.194 | 0.300 | 0.297 | -- |
| 480 | **0.077** | 0.106 | 0.208 | 0.211 | -- |
| 864 | 0.059 | **0.053** | 0.134 | 0.129 | 0.073 |

| budget | multiMFM | multiMFM-SS | best-of-N | DPS | FK |
|---|---|---|---|---|---|
| 256 | 2.851 | **1.875** | 5.824 | 3.051 | -- |
| 512 | 1.479 | **1.311** | 3.592 | 2.477 | 8.534 |
| 1,024 | 1.479 | **1.240** | 2.266 | 2.477 | 6.972 |
| 2,048 | 1.479 | **1.199** | **1.466** | 2.477 | 4.830 |

Two things worth reading off these tables rather than smoothing over. Feynman-Kac overtakes
us at the last DNA rung, and our matched 1,028-call QM9 configuration (1.539) is *worse*
than our 818-call one (1.240), so more inner steps over a wider window is not better -- the
1,024 rung reports 1.240 because that is what is reachable within the budget.

## What makes this a fair comparison and not a tuning exercise

1. **Same sample size on both sides.** Ours and the baselines are run at n = 512 x 3 seeds
   (QM9) and 300 x 5 repeats (DNA), by the same code. The earlier scatter came from the
   baselines being at n = 100 x 3; raising n only for our arms would have been a smooth
   curve against a noisy one.
2. **Configurations selected on a grid, then re-measured on held-out seeds.** The figure
   quotes the held-out numbers. One pick (`gamma = 6`) did not survive -- 1.926 on the
   selection seed, 2.253 held out -- and was dropped. The baselines were always tuned per
   budget on a separate pilot; doing it for them and not for us was the asymmetry.
3. **One validity floor for everyone.** A QM9 point counts only at PoseBusters validity
   >= 0.75, which DPS's own best configuration needs (0.79-0.87). Without it we would be
   comparing our invalid molecules against their valid ones.
4. **Running minimum applied to every method.** The curves show the best error reachable
   with *at most* that budget, so a method that degrades when given more compute is not
   charged twice for the same artifact.

## Two levers worth keeping

* **Inner steps, not Monte-Carlo samples.** `S=2, MC=1` beats `S=1, MC=1` at equal cost, and
  both beat the published `S=4, MC=32` by a wide margin per call.
* **Two-time flow-map query.** Dropping `--mfm-diagonal` improves the extremes at identical
  cost and validity (SS 1.343 -> 1.240, multiMFM 1.577 -> 1.498).

## Caveats

* QM9 is `alpha` only; the other five properties are not swept at low budget.
* The DNA `t_start = 0.35` window failed twice with `CUDA error: device busy` on its
  allocated node and is missing. It would add intermediate points, not change the picture --
  `t_start = 0.5` is both cheaper and complete.
* Above the crossover the baselines win and the figure shows it: Feynman-Kac from ~800
  calls on DNA, best-of-N from ~2,000 on QM9.

Producers: `scripts/qm9/{low_budget_grid,low_budget_grid2,low_budget_confirm,lowbudget_final,lowbudget_final2,lowbudget_twotime}.sbatch`,
`scripts/dna/{fig4_window_sweep,fig4_window_confirm,fig4_lowbudget_final}.sbatch`.
Collectors: `scripts/paper/{collect_low_budget,collect_final_steering,collect_twotime}.py`.
Figure: `scripts/paper/steering_figure.py` -> `figures/steering_dna_qm9.pdf`.
