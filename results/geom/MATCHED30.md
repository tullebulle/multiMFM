# GEOM-Drugs fingerprint steering vs TFG-Flow, on a matched protocol

The earlier comparison to TFG-Flow was unsound in two ways at once, both recorded in
`docs/geom_rescue_attempt.md`: it compared our numbers on molecules up to 71 heavy atoms
against theirs on molecules capped at 30, and it compared our valid-molecules-only mean
against their all-molecules mean. This run removes both objections.

## Protocol

* **Atom cap.** `--max-atoms 30`, TFG-Flow's cap. 44 of the val targets are excluded by it.
* **Averaging convention.** Both `tanimoto_real_mean` (valid only, ours) and
  `tanimoto_real_mean_all` (every generated molecule, unconvertible counted as 0, theirs)
  are reported below. **Our conversion rate is 1.0000**, so for us the two conventions give
  the same number and the objection does not bite.
* **Seeds.** 3 seeds (1, 2, 3), 256 molecules each, none used for tuning
  (`tuned_on_holdout: false`). Uncertainties are standard errors over seeds.
* **Floor.** Tanimoto between two *unrelated real* molecules both at <= 30 heavy atoms is
  **0.20991 +- 0.00019** (200,000 pairs; `docs/geom_rescue_attempt.md` section 1). This
  metric pays for genericness, so distance above this floor is the quantity that means
  something, not the raw value and not the relative gain.
* Sources: `results/geom/runs/fp_matched30/*.json`, slurm job 49669211.

## Result

| | unguided / uncond. | guided | guided - floor |
|---|---|---|---|
| TFG-Flow, as reported | 0.170 +- 0.001 | 0.208 +- 0.002 | **-0.002** |
| multiMFM, published config | 0.2270 +- 0.0046 | 0.2212 +- 0.0063 | +0.011 |
| **multiMFM, corrected config** | 0.2270 +- 0.0046 | **0.2516 +- 0.0073** | **+0.042** |

Corrected config: `mu = 2.0`, `lambda = 20.0`, `t in [0.1, 0.9]`, `guide_every = 2`, `MC = 4`,
`mfm_steps = 2`, atom-type channel guided, EMA weights. PoseBusters validity **0.888 +- 0.013**
guided against 0.865 unguided, so the gain is not bought with validity. Mean 25.0 heavy atoms
in every arm.

## What can and cannot be claimed

**Sound.** Our guided arm reaches 0.2516 where theirs reaches 0.208: higher in absolute
similarity, under their own averaging convention, at their own atom cap, and with validity
slightly up rather than down. Measured against the floor the metric itself sets, we clear it by
0.042 while their guided number sits *at* it (-0.002).

**Not sound, and we do not claim it.** That we win on *relative* improvement -- we do not.
Ours is +10.76% +- 0.95 over its own unguided arm; theirs is +22.4%. Relative gain is the wrong
axis here, because it rewards a weak baseline: their unconditional 0.170 is *below* the
random-pair floor of its own size regime, which their validity-inclusive convention is enough to
explain (an unconvertible molecule enters their average as a 0, so 0.170 -> 0.208 can be moved by
improving validity alone, with no change in target matching).

**The configuration, not the method, was the problem.** Run at the published settings in this
same regime, guidance is flat to slightly negative: 0.2212 +- 0.0063 against an unguided
0.2270 +- 0.0046, i.e. **-2.19% +- 1.74**. The corrected settings give **+10.76% +- 0.95** on
seeds never used for tuning. The two differ by `mu` (0.5 -> 2.0), `lambda` (0.02 -> 20.0) and
`mfm_steps` (10 -> 2); the reward is the differentiable (soft) Tanimoto rather than L2, which
ranks candidate substitutions far better (Spearman 0.708 against 0.375).
