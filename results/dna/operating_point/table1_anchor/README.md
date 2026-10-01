# The anchor: where the operating point was chosen

Not a new run. A re-analysis of `results/dna/deficit_rescue/nfe_L50_n200/` (Slurm 49431420, one
H200, n = 200 paired per target, both C0 targets, every configuration under the scale-free rule at
`guidance_frac 8`) into the two files here, so that the choice this round applies everywhere is
documented with the statistics it was made on:

* `nfe_matched_summary.csv` -- guided oracle MAE and guide MAE per configuration, with
  `steps`, `mc` and `nfe_value = steps x mc`, at every NFE level in that sweep.
* `nfe_matched_paired_all.csv` -- every within-NFE paired difference on shared `sample_idx`,
  with SEM and a 2,000-resample `default_rng(0)` bootstrap CI. `mean_diff` is `first - second`
  on the oracle absolute error, so negative favours `first`.

**The choice.** At NFE 32, target `y* = +1`: dMFM's own best split is 1 x 32
(`dmfm4_fm1@norm@mc=32`, 0.1516) against 2 x 16 (0.1694) and 4 x 8 (0.1955); GLASS's own best is
2 x 16 (0.1411) against 4 x 8 (0.1456). `y* = -1` agrees (dMFM 0.1103 / 0.1221 / 0.1717; GLASS
0.1178 / 0.1303). Paired, dMFM 1 x 32 against dMFM 4 x 8 is **-0.0439 +- 0.0150** at `y* = +1`
and **-0.0614 +- 0.0118** at `y* = -1`.

So the operating point carried into every other DNA item is **dMFM 1 composed step x MC 32** and
**GLASS 2 Euler steps x MC 16**, both NFE 32. It is not re-chosen anywhere else.

**What this sweep does not contain**, and my grids therefore add: the 8 x 4 corner on either arm,
and the 1 x 32 corner on the GLASS arm.
