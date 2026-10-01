# QM9 value-function estimation

The QM9 analogue of the DNA posterior-recovery benchmark, built so both panels of the
steering figure can be QM9. For conditioning states drawn by integrating the base model to
$t \sim \mathcal U(0.05, 0.35)$, it measures how well each estimator recovers

$$V_t(x) = \log \mathbb E\!\left[e^{r(X_1)} \mid X_t = x\right].$$

| estimator | what it is | cost | depends on $N$? |
|---|---|---|---|
| **multiMFM** | log-mean-exp over $N$ flow-map posterior samples | $N$ calls | **yes** |
| **DPS** | $r(\mathbb E[X_1 \mid X_t])$, the one-step denoiser point estimate | 1 call | no |
| **FMAP** | base ODE integrated $t \to 1$, scored at that endpoint | steps to 1 | no |
| *reference* | the same log-mean-exp over 4,096 GLASS posterior samples | — | — |

## Result (32 conditioning states, 4,096-sample reference)

| $N$ | multiMFM | DPS | FMAP |
|---|---|---|---|
| 1 | 27.33 | 21.27 | 29.68 |
| 2 | 15.02 | " | " |
| 4 | 7.59 | " | " |
| 8 | 4.26 | " | " |
| 16 | 2.18 | " | " |
| 32 | 1.58 | " | " |
| 64 | 1.08 | " | " |
| 128 | 0.51 | " | " |
| 256 | 0.40 | " | " |
| 512 | **0.19** | 21.27 | 29.68 |

multiMFM passes both point estimates by $N \approx 2$ and ends **110x** better than DPS.
Both baselines are flat by construction: neither has an $N$ to spend.

## The reference is an estimate too, and here it is good enough

Its own self-split spread -- two disjoint halves of the 4,096 pool scored against each
other -- is **0.062**, below every measured point including the last, so the whole range is
resolved rather than floor-limited.

That took a large pool to achieve, and the smaller runs show why it matters: at 32
reference samples the floor was **7.69** against a DPS error of 5.79, i.e. *above* the
signal, and at 64 it was 2.04. A version of this benchmark run cheaply would have measured
nothing. The reason is the reward: $r = -(\text{pred} - \text{target})^2$ in Bohr$^6$, so
$e^r$ spans many orders of magnitude and the log-mean-exp is high-variance.

## Why FMAP is worse than DPS, in both domains

FMAP scores one sampled molecule, whose property lands wherever the ODE takes it, while
$\log \mathbb E[e^r]$ is dominated by the good tail of the posterior -- a Jensen gap. DPS's
blurred mean prediction scores nearer the average and so sits closer to the reference. The
same ordering holds on DNA (FMAP 4.20, DPS 2.86), which is a check that this
implementation behaves like the one the DNA panel uses.

## Caveats

* `alpha` only, as with the steering panel.
* The reference is a GLASS pool, so the curve measures convergence to the exact posterior
  sampler. A bias in the distilled flow map would show as a plateau above the floor; none
  appears within the range measured.

Producer: `scripts/qm9/value_error.py` (`scripts/qm9/value_error.sbatch`, Slurm 49644002).
Figure: `scripts/paper/qm9_figure.py` -> `figures/qm9_property_steering.pdf`.
