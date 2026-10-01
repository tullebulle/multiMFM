# Preliminary dMFM scale-free numbers (NOT the result)

A 4-state slice run on CPU on 2026-09-29 while the four full routes were queued. Kept because
it already overturns a prediction that was circulating, and because it is the only dMFM
measurement on disk until a full array lands. **Do not put these in the paper.**

* `L50_motif_4states_cells.csv`, `L50_motif_4states_summary.json` - states 0-3 of 32,
  reward `motif`, L=50 only, all nine N values, 8 noise pools per state, the published
  GLASS-2048 reference, dMFM estimates in fp32 on CPU.

## What it shows

Published GLASS cells recomputed over **the same 4 states** so the comparison is
apples-to-apples (the published column in Tables 14-15 pools 32 states, so it is not
directly comparable to a 4-state cell). The recomputation is trustworthy: the same cell
function reproduces the published 32-state cells to 5e-7.

| N | GLASS relL1 | dMFM relL1 | ratio | GLASS cos | dMFM cos | cos diff |
|---|---|---|---|---|---|---|
| 1 | 2.81 | 3.03 | 1.08 | 0.140 | 0.204 | +0.064 |
| 2 | 3.10 | 2.78 | 0.90 | 0.176 | 0.160 | -0.016 |
| 4 | 1.77 | 2.00 | 1.13 | 0.164 | 0.219 | +0.055 |
| 8 | 1.68 | 1.92 | 1.15 | 0.312 | 0.281 | -0.031 |
| 16 | 1.60 | 1.37 | 0.86 | 0.327 | 0.276 | -0.051 |
| 32 | 1.37 | 1.17 | 0.85 | 0.423 | 0.339 | -0.084 |
| 64 | 1.17 | 1.00 | 0.86 | 0.515 | 0.533 | +0.018 |
| 128 | 0.93 | 0.83 | 0.90 | 0.650 | 0.622 | -0.028 |
| 256 | 0.77 | 0.70 | 0.91 | 0.713 | 0.726 | +0.013 |

1. dMFM tracks GLASS within a ratio of 0.85-1.15 on relative L1 and +/-0.06 on cosine at
   matched N. No systematic penalty for substituting the dMFM sampler.
2. The dMFM curve keeps decaying to N=256 (4.32x from N=1, against GLASS's 3.66x). **No
   floor is visible in the measured range.**

## Why the "nonzero floor" prediction was wrong

The prediction was that the dMFM approximation error does not shrink with N, so the curves
would flatten instead of decaying. That is not what happens, and the reason is visible in the
published data alone: the reference is GLASS-**2048**, itself a finite-MC estimate, so the
measured error is dominated by the *reference's* Monte Carlo noise. Over a 256-fold increase
in N the published GLASS-vs-GLASS relative L1 falls only 3.66x where pure 1/sqrt(N) would give
16x, and it is still ~0.77 at N=256. The floor in the published figure is therefore already
mostly reference noise, and the dMFM bias is small compared with that residual.

The corrected statement, which is sharper and agrees with the paper's own l.1727: **this
experiment cannot separate dMFM approximation error from the reference's Monte Carlo error.**
It does answer "does swapping GLASS for dMFM at finite N degrade the gradient estimate?"
(preliminary answer: no, not materially). Measuring the dMFM approximation error itself would
need a reference converged in N - a much larger reference, or common random numbers shared
between the two samplers.

## Caveats

4 of 32 states; `motif` only; L=50 only; dMFM side fp32 on CPU against the published TF32-GPU
reference. That last point does not affect the conclusion: `../precision_bound.json` measures
the CPU-vs-GPU movement of a cell at 0.006 in relative L1 and 0.0005 in cosine, while the
ratios above differ from 1 by 10-15 %.
