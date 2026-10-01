# Is QM9's Table 2 run at the right (posterior steps x MC) split?

Status 2026-09-30, **complete**: both rows, four NFE-matched splits, two properties, the
selection seed and the three held-out evaluation seeds, plus an off-budget probe and the baseline
ladder on one NFE axis. Owner: the QM9 operating-point agent of the 2026-09-30 round.
Companion: `docs/dmfm_deficit_rescue.md`, which found the same class of defect on the DNA C0
experiment (Table 1) and is the reason this document exists.

**Sections 1-2 were written to disk BEFORE any job of this investigation was submitted**
(this file 15:10 EDT, first job 49464797 15:17 EDT). They are the pre-registration: the property, the grid, the selection rule and the
success criterion are fixed in advance so that the answer cannot be a best-of-grid artifact of the
kind that made the published Table 3 fragile (plan section 7, and
`docs/dmfm_deficit_rescue.md` section 3b).

## Verdict

**Yes, 4 x 32 is the right split, and Table 2 does not change.** The DNA finding does not transfer:
on QM9 the MC-heavy end of the trade is much *worse*, not better. On the pre-registered selection
property, at matched generative NFE of 14,848 per molecule, 4 posterior steps x MC 32 is the argmin
of the four splits **on both rows** -- by 2.6 to 5.5 SEM for multiMFM and 2.2 to 6.2 SEM for
multiMFM-SS -- and for the plain row it is simultaneously the argmax of PoseBusters validity. The
ordering reproduces on the held-out evaluation seeds and, in weaker form, on a second property. The
one split the DNA result would have recommended, 1 x 128, is the worst cell of every grid: +1.72 and
+1.30 oracle MAE on alpha, +205 and +186 meV on homo, with PoseBusters validity falling 0.86 -> 0.69
on the unfiltered row and the multiMFM-SS candidate pool falling 0.71 -> 0.18 valid, at 30 times the
wall clock.

The measured reason for the difference is in section 1a and section 3.1: QM9's only MFM student was
distilled *for* a 4-step diagonal rollout (`rollout_steps 4, rollout_diagonal True`), so one Euler
step of size 1.0 is a query it was never trained to answer, whereas on DNA the artifact the
step-matched rule selects at one step is a student distilled for one jump. Averaging 128 draws over
a degenerate posterior does not help.

What this does buy the paper is that the split is no longer an untested default: all 133 rows of
`results/qm9/multimfm_ss_search_grid.csv` record `value_samples = 32`, and
`results/qm9/multimfm_search_grid.csv` has no split column at all across its 60 configurations
because the driver hard-codes `--value-glass-steps 4 --value-samples 32`. Before this grid nothing
in the record had ever varied it. Two
by-products are worth the authors' attention: an off-budget probe shows the estimator is
**budget-saturated** at 4 x 32 (doubling MC to 64 moves multiMFM 1.112 -> 1.062 and SS 0.796 ->
0.805, section 3.5), and the one lead found is a *query-mode*, not a split: the same checkpoint used
as a genuine two-time flow map rather than a diagonal velocity field is +5 points of PoseBusters
validity on alpha at an accuracy cost inside noise (section 3.1, section 4 item 1).

## 1. The question

The DNA finding: the value gradient costs `posterior_steps x MC` network evaluations, so at fixed
NFE those two knobs trade off, and the two posteriors (GLASS Euler vs a flow map) prefer opposite
ends of the trade. Table 1's dMFM row had been run at GLASS's end (4 steps x MC 8); moved to the
flow map's end (1 step x MC 32) at the *same* NFE it went from 0.193 / 0.172 to 0.146 / 0.120 and
the 6-9 SEM deficit against GLASS became a tie.

QM9's Table 2 rows are produced by

* `scripts/qm9/run_multimfm.sh`  -> `python -m multimfm.property_targets` (the multiMFM row)
* `scripts/run_steer_search.sh`  -> `python -m multimfm.steer_search`     (the multiMFM-SS row)

both with `--value-glass-steps 4 --value-samples 32 --value-sampler mfm --mfm-diagonal`
(plan section 3.1: "32 value samples, 4 inner steps"). Measured from
`outputs/qm9/ss_mfm_pilot/alpha_n100/summary.json`, that is exactly

    nfe_generative = 128 (base ODE) + 115 guided steps x 4 steps x 32 MC = 14,848 per molecule
    mfm forward calls per guided step = 4, each of (n_molecules x 32) rows

so 1 x 128, 2 x 64, 4 x 32 and 8 x 16 all cost **14,848 generative NFE per molecule**, exactly. The
question is whether 4 x 32 is the best of those four, and if not, what Table 2 becomes at the
better one at no extra compute.

### 1a. Two differences from DNA that must be stated first

**(a) Which artifact the step-matched-student rule selects: the same one at every step count,
because only one exists.** `checkpoints/mfm_student/student_step_1000.pt` is the only QM9 MFM
student in the repo, and its stored `args` say it was distilled with `rollout_steps 4`,
`rollout_diagonal True`, `eval_diagonal True`, `mfm_steps 4`, `glass_steps 4`, `max_jump 0.5`,
`esd_weight 1.0`, `diag_weight 1.0`, `t_cond_grid 0.1,0.3,0.5,0.7,0.9`, from
`init_ckpt .../mfm_diag_rollout8_forced_.../student_step_3000.pt` (purged, so it cannot be
retrained end to end). So on QM9 the rule is vacuous in the artifact it picks -- but not vacuous in
what it says about the split: the artifact's *own distillation target is the 4-step diagonal
rollout*, which is the split Table 2 already uses. Where DNA's 1-step operating point happened to
select a *different, diagonal* artifact that had been distilled for one jump, QM9 has to ask the
4-step-rollout student to do a one-step job. That is an argument against 1 x 128 here, not for it.

**(b) `--mfm-diagonal` means the student is used as a velocity field, Euler-integrated.** In
`glass_guidance.differentiable_mfm_posterior`, `diagonal=True` forces `u = s`, so
`--value-glass-steps k` is *k Euler steps of the diagonal MFM velocity*, not k composed flow-map
jumps. The published QM9 posterior is therefore the analogue of DNA's `diag_fm4`, and 1 x 128 the
analogue of `diag_fm1@mc=128`. (On DNA, `diag_fm1@mc=32` measured 0.154 / 0.115 against
`diag_fm4`'s 0.188 / 0.175 -- i.e. this change helped there.) Dropping `--mfm-diagonal` would query
the same checkpoint as a genuine two-time map; at 2 steps the gaps are exactly 0.5 = its
`max_jump`, at 1 step the gap is 1.0, outside anything it was trained on. That mode is a **labelled
diagnostic below, deliberately outside the pre-registered selection**, because the published
configuration uses the flag and changing both the split and the query mode at once is two knobs.

**(c) QM9 has no gradient-normalisation rule.** `--guidance-max-coord-rms` and
`--guidance-max-atom-rms` default to 0.0 and neither driver passes them, so the guidance is
`mu * grad * dt` with the raw gradient. DNA's comparison was made scale-free (`@norm`) precisely so
that a change of estimator could not smuggle in a change of dose. Here it can, so every table below
also reports the mean `grad_coords_rms` from `guided_step_logs.csv` / the step logs, and any
improvement that comes with a *larger* gradient RMS is flagged as confounded with dose. This is the
same trap that forced the DPS withdrawal in this repo (commit bc57c94, "the sampler shrinks the
dose as 1/N"). The pre-registered change holds `mu` and `reward_scale` at their published
per-property values and moves the split only: one knob.

**(d) For multiMFM-SS the MC count is also the candidate-pool size.** With
`--lookahead-source mfm` (Algorithm 3) the look-aheads that SS selects among *are* the value
estimator's posterior samples, so candidates per molecule = `guided_steps x MC` =
115 x MC: 14,720 at MC 128, 3,680 at MC 32, 1,840 at MC 16. Those samples are already paid for on
the generative NFE axis (that is Algorithm 3's point), but the selection re-scoring is regressor
NFE, which grows with MC and is *not* on the NFE axis. So an SS improvement at an MC-heavy split
has two possible mechanisms -- a better value gradient and a four-times-larger free candidate pool
-- and the tables must report both the generative NFE (matched) and the regressor NFE (not matched).

## 2. Pre-registration

**Selection property: `alpha`** (polarisability, Bohr^3). Chosen because it is the only property
for which the existing matched-budget baseline ladder in `results/qm9/baselines/baseline_sweep_v2.csv`
(best-of-N, DPS, Feynman-Kac, and an SS MC ladder at 4 steps) was run, so the operating point is
chosen where the matched-budget comparison against the baselines already exists. Fixed before any
job was submitted.

**Selection seeds: seed 8 / target-seed 2026, n = 100.** This is the set on which every published
guidance hyperparameter was already selected, and it is disjoint from the seeds 1-3 /
target-seeds 2027-2029 on which Table 2 is reported. Choosing the split here therefore does not
touch the evaluation seeds.

**Grid (both rows, `--mfm-diagonal`, everything else at the published per-property values):**

| (steps, MC) | generative NFE/mol | note |
|---|---|---|
| 1 x 128 | 14,848 | most MC-heavy |
| 2 x 64  | 14,848 | |
| 4 x 32  | 14,848 | **the published operating point** |
| 8 x 16  | 14,848 | most step-heavy |

**Selection rule, fixed in advance:** the split with the **lowest guided oracle MAE subject to
PoseBusters validity >= 0.90**, applied to each row separately. This is not a new rule: it is the
rule the published per-property `(mu, reward_scale)` were themselves chosen by
(`scripts/qm9/run_multimfm.sh` header: "the lowest oracle MAE with PB-valid >= 0.90 over a 5-round
search"). Ties within 1 SEM are broken in favour of the published 4 x 32, so the burden of proof is
on the change.

**Then the chosen split is applied unchanged to all six properties.** No per-property re-selection.
If the chosen split is worse than 4 x 32 on some property, that is reported as a loss, not
re-optimised away.

**Success criterion:** the change is reported as a real improvement only if, on the held-out
seeds 1-3 at n = 1000, the paired per-target difference against the published 4 x 32 is negative by
more than 2 SEM on the selection property and is not positive by more than 2 SEM on a majority of
the other five.

**Reference kept alongside:** the published 4 x 32 numbers, from
`outputs/qm9/table2_3seed/` / `results/qm9/rerun_3seed/rerun_3seed_summary.csv`
(3 seeds x n = 1000, jobs 49276185 and successors), which are the same seeds/targets, so the
comparison is paired per `sample_idx`.

## 3. Results

### 3.1 The multiMFM row on the selection property: 4 x 32 wins, and it is not close

`results/qm9/operating_point/grid_alpha_n100_s8/`, Slurm 49464797 (`kempner`,
`holygpu8a19102`, NVIDIA A100-SXM4-40GB). One card, one node, alpha, seed 8 / target seed 2026,
n = 100 paired ids, everything except `(steps, MC)` at the published value. Every configuration
has generative NFE 14,976 per molecule (14,848 in the guided phase plus 128 in the unguided
control arm). Control: the *unguided* arm is shared by construction -- `guided_sample` captures
and restores the torch RNG around the value gradient -- and measures a maximum per-id spread of
**2.4e-3 on a scale of 8.30** (3e-5 relative) across all six configurations, so the whole
difference below is the guidance term.

`vs 4x32` is the per-sample paired difference in guided oracle absolute error against the
published split; positive means the split is **worse** than what Table 2 already uses.

| split (steps x MC) | mode | guided oracle MAE [95% CI] | guide MAE | PB valid | unique | vs 4 x 32 | frac. improved |
|---|---|---|---|---|---|---|---|
| **4 x 32 (published)** | diagonal | **1.112** [0.926, 1.324] | **0.700** | 0.86 | 1.00 | 0 | -- |
| 4 x 32 | flow map | 1.267 [1.036, 1.511] | 0.754 | **0.91** | 0.96 | +0.155 +- 0.147 | 0.47 |
| 2 x 64 | flow map | 1.433 [1.206, 1.677] | 0.864 | 0.86 | 0.98 | +0.321 +- 0.154 | 0.43 |
| 2 x 64 | diagonal | 1.583 [1.339, 1.845] | 0.822 | 0.82 | 1.00 | +0.471 +- 0.163 | 0.41 |
| 8 x 16 | diagonal | 1.734 [1.343, 2.161] | 0.853 | 0.78 | 0.99 | +0.622 +- 0.239 | 0.47 |
| 1 x 128 | diagonal | 2.835 [2.292, 3.478] | 1.367 | 0.69 | 0.97 | +1.723 +- 0.311 | 0.21 |

("diagonal" = the published `--mfm-diagonal` usage; "flow map" = the same checkpoint queried as a
genuine two-time map, the labelled diagnostic of section 1a(b), not part of the selection.)

**The published 4 x 32 is the argmin of the grid, by 2.6 to 5.5 SEM over the other three splits,
and it is simultaneously the argmax of PoseBusters validity.** The pre-registered rule -- lowest
oracle MAE subject to PB >= 0.90 -- does not even have to arbitrate: at n = 100 no split reaches
0.90 (the n = 1000 reference measures 0.905 for this row, so 0.86 +- 0.03 here is the same rate
with a 100-sample error bar), and 4 x 32 is the unconditional argmin of MAE *and* the
unconditional argmax of PB, so it wins under every reading of the rule. **The answer on this row
is that Table 2 is already at the right operating point, and there is no free improvement to
take.**

Three things worth recording about *why*, since the DNA result went the other way:

* **The direction is the opposite of DNA's, and the mechanism is visible in the guide column.**
  The function the sampler actually optimises is twice as badly served at one step
  (guide MAE 1.367) as at four (0.700). Moving MC from 32 to 128 buys nothing if the posterior it
  averages over is a bad endpoint predictor, and on QM9 it is: the only student that exists was
  distilled with `rollout_steps 4, rollout_diagonal True` (section 1a(a)), so a single diagonal
  Euler step of size 1.0 is a query it was never trained to answer. DNA's one-step operating point
  worked because the artifact the step-matched rule selects there *is* a one-jump student.
* **It is not a dose artifact, and the dose confound runs against the published point if
  anything.** QM9 applies no gradient-normalisation rule (section 1c), so the mean guidance
  gradient RMS is reported: 1.41 (1 x 128), 0.75 (2 x 64), **1.49 (4 x 32)**, 5.44 (8 x 16). It is
  not monotone in the split, 1 x 128 gets essentially the same dose as 4 x 32 and is 1.7 worse,
  and the one split with a much *larger* dose (8 x 16, 3.6x) is also worse. So no split is winning
  or losing by being quietly driven harder.
* **The posterior's own mean reward is monotone in the step count and points the wrong way, which
  is informative.** `posterior_reward_mean` (the unscaled reward of the draws the estimator
  scored) is -16.5, -24.6, -36.4, -45.4 at 1, 2, 4, 8 steps: the *fewer* steps, the closer to
  target the posterior draws look. Yet fewer steps guide worse. That is the signature of a
  degenerate posterior -- a one-step endpoint that is not a molecule the guide can discriminate,
  so the guide regresses everything towards the dataset mean, which sits near the target and
  yields a flat, uninformative value landscape. The 0.69 PoseBusters rate at 1 x 128 is the same
  story from the other end: the guidance built from that posterior drives the trajectory off the
  data manifold.

The flow-map diagnostic is the one place the published configuration is arguably leaving something
on the table, and only for validity: querying the same checkpoint as a genuine two-time map at
4 x 32 (gaps of 0.25, inside its `max_jump 0.5`) gives **PB 0.91 against 0.86** at a +0.155 +-
0.147 MAE cost, i.e. a validity gain of about 5 points for an accuracy difference that is 1.1 SEM,
inside noise. At 2 steps (gaps of exactly 0.5) the flow-map mode also beats the diagonal mode on
both metrics (1.433 vs 1.583, PB 0.86 vs 0.82). This is a *query-mode* knob, not a split knob, it
was deliberately outside the pre-registration, and one property at n = 100 is not enough to act
on; it is recorded as the one lead this investigation found.

### 3.2 The multiMFM-SS row, same grid, same conclusion

Same run root, Slurm 49464822. The reported SS row is `best_lookahead_pbvalid`, Algorithm 3's
selection among the value estimator's own MFM look-aheads.

| split | mode | SS oracle MAE [95% CI] | SS guide MAE | PB valid | vs 4 x 32 | unfiltered pool PB | candidates/mol | regressor NFE/mol | wall s |
|---|---|---|---|---|---|---|---|---|---|
| **4 x 32 (published)** | diagonal | **0.796** [0.680, 0.931] | 0.001 | 1.000 | 0 | 0.71 | 3,680 | 7,471 | **33** |
| 4 x 32 | flow map | 0.973 [0.815, 1.160] | 0.002 | 1.000 | +0.177 +- 0.098 | 0.67 | 3,680 | 7,471 | 35 |
| 8 x 16 | diagonal | 1.035 [0.863, 1.218] | 0.008 | 1.000 | +0.238 +- 0.106 | 0.59 | 1,840 | 3,791 | 37 |
| 2 x 64 | flow map | 1.120 [0.925, 1.324] | 0.003 | 1.000 | +0.324 +- 0.116 | 0.42 | 7,360 | 14,831 | 51 |
| 2 x 64 | diagonal | 1.341 [1.132, 1.568] | 0.014 | 1.000 | +0.545 +- 0.122 | 0.38 | 7,360 | 14,831 | 62 |
| 1 x 128 | diagonal | 2.100 [1.735, 2.564] | 0.377 | 0.980 | +1.304 +- 0.211 | 0.18 | 14,720 | 29,551 | **1,003** |

**4 x 32 is the argmin here too, by 2.2 to 6.2 SEM.** So the pre-registered choice is the published
split on both rows, and there is nothing to apply to the other five properties that is not already
there.

Two things this row shows that the plain row cannot:

* **PoseBusters validity is not what the split trades against, for SS -- the *pool* is.** As the
  brief anticipated, SS's reported PB is 1.000 at every split (0.98 at 1 x 128). But the
  `best_lookahead_raw` pool it selects from is 0.71 valid at 4 x 32 and **0.18 at 1 x 128**: the
  MC-heavy split does not produce a worse answer because the filter fails, it produces a worse
  answer because there is almost nothing valid in a four-times-larger pool to find. The plain
  multiMFM row, which has no filter, prints that same collapse directly as PB 0.86 -> 0.69.
* **The MC-heavy splits are matched on generative NFE and nowhere else.** The regressor calls scale
  with MC alone (7,471 -> 29,551 rows per molecule from 4 x 32 to 1 x 128) and the lazy
  PoseBusters search has to descend a pool four times as long and four times as invalid. The
  measured wall clock is **33 s against 1,003 s, a factor of 30**, on the same card at n = 100.
  Had 1 x 128 *won* on accuracy, it would still not have been free; as it is, it is worse on every
  axis at once.

### 3.3 Held-out seeds, and a second property: the same ordering

**Evaluation seeds 1-3, alpha, n = 100** (`results/qm9/operating_point/evalseeds_alpha_n100/`,
Slurm 49464799 and 49471391) -- the seed / target-seed / n set of
`results/qm9/baselines/baseline_sweep_v2.csv`, so these arms are paired with every baseline.
Oracle MAE as mean +- SEM over seeds; `vs 4 x 32` is the per-sample paired difference.

| split | multiMFM MAE | multiMFM PB | multiMFM vs 4x32 | SS MAE | SS PB | SS vs 4x32 | SS pool PB |
|---|---|---|---|---|---|---|---|
| **4 x 32** | **1.215 +- 0.106** | **0.887** | 0 | **1.020 +- 0.074** | 1.000 | 0 | 0.743 |
| 2 x 64 | 1.689 +- 0.081 | 0.823 | +0.475 +- 0.196 | 1.219 +- 0.019 | 1.000 | +0.199 +- 0.133 | 0.397 |
| 8 x 16 | 2.003 +- 0.391 | 0.763 | +0.788 +- 0.392 | 1.196 +- 0.038 | 1.000 | +0.176 +- 0.156 | 0.450 |
| 1 x 128 | 2.787 +- 0.154 | 0.700 | +1.573 +- 0.346 | 2.316 +- 0.186 | 0.973 | +1.297 +- 0.243 | 0.157 |

The selection reproduces on seeds it was not made on: 4 x 32 is the per-seed argmin on both rows at
**3 of 3 seeds** (multiMFM 1.365 / 1.011 / 1.267, SS 1.117 / 1.066 / 0.876), and on the plain row it
is also the argmax of PB. For the SS row 2 x 64 and 8 x 16 are 1.1-1.5
SEM behind rather than decisively worse, so on that row the honest statement is that 4 x 32 is the
best of the grid and not beaten by any of it; 1 x 128 is 5.3 SEM worse. (Unguided control: maximum
per-id spread 1.7e-3 to 3.3e-3 across splits at each seed, on a scale of 8.6-9.7.)

**A second property, `homo`, seed 8, n = 100** (`results/qm9/operating_point/grid_homo_n100_s8/`,
Slurm 49465288 / 49476040), run purely as a check that the shape of the curve is not specific to
alpha. It is not part of the selection and was not allowed to change it. Oracle MAE in meV;
`vs 4 x 32` paired per sample.

| split | mode | multiMFM MAE [CI] | PB | vs 4x32 | SS MAE [CI] | SS PB | vs 4x32 | SS pool PB |
|---|---|---|---|---|---|---|---|---|
| **4 x 32** | diagonal | **238.6** [195, 285] | **0.96** | 0 | 259.5 [221, 299] | 1.000 | 0 | 0.60 |
| 4 x 32 | flow map | 237.6 [196, 282] | 0.94 | -1.1 +- 29.7 | 258.2 [217, 303] | 1.000 | -1.3 +- 30.0 | 0.65 |
| 8 x 16 | diagonal | 281.7 [230, 338] | 0.88 | +43.1 +- 37.4 | **233.9** [194, 276] | 1.000 | -25.6 +- 27.9 | 0.80 |
| 2 x 64 | diagonal | 255.0 [213, 299] | 0.90 | +16.4 +- 29.5 | 257.7 [214, 305] | 1.000 | -1.8 +- 32.8 | 0.38 |
| 2 x 64 | flow map | 280.0 [238, 324] | 0.90 | +41.3 +- 30.2 | 248.2 [210, 290] | 1.000 | -11.4 +- 29.4 | 0.43 |
| 1 x 128 | diagonal | 443.3 [368, 530] | 0.78 | +204.7 +- 48.8 | 446.0 [377, 520] | 0.920 | +186.5 +- 44.1 | 0.31 |

On homo the curve is **flat except for the collapse at one step**: for the plain row 4 x 32 is again
the argmin of the diagonal splits and again the argmax of PB, and nothing beats it by more than
1.5 SEM; for the SS row 8 x 16 is nominally best but by 0.9 SEM, well inside noise, and 1 x 128 is
4.2 SEM worse. The one feature that reproduces at full strength on both properties and both rows is
that the MC-heavy end is a disaster, which is the opposite of what the DNA result would predict.
(Unguided control on homo: maximum per-id spread 0.68 meV on a scale of 723 meV, 9e-4 relative.)

### 3.4 The verdict, and what Table 2 becomes

**4 x 32 is the right split. Table 2 does not change, and no Table 2 rerun was needed** -- the
pre-registered procedure was "choose the split once on alpha, apply it unchanged to the other
five", and the split it chooses is the one already published, so the row that the procedure
produces is `results/qm9/rerun_3seed/rerun_3seed_summary.csv` as it stands. Reproduced here for the
record (3 seeds x n = 1000, seeds 1-3 / target seeds 2027-2029, mean +- SEM over seeds, A100):

| property | unit | unsteered | multiMFM (4 x 32) | PB | multiMFM-SS (4 x 32) | PB | paper's multiMFM | paper's SS |
|---|---|---|---|---|---|---|---|---|
| Cv | cal/(mol K) | 3.448 | 0.444 +- 0.004 | 0.903 | 0.414 +- 0.009 | 1.000 | 0.417 | 0.398 |
| mu | D | 1.721 | 0.818 +- 0.033 | 0.906 | 0.528 +- 0.016 | 1.000 | 0.843 | 0.686 |
| alpha | Bohr^3 | 8.882 | 1.166 +- 0.017 | 0.905 | 1.010 +- 0.024 | 1.000 | 1.120 | 1.104 |
| Delta-eps | meV | 1872.96 | 592.3 +- 7.2 | 0.908 | 413.3 +- 3.7 | 1.000 | 599.6 | 462.3 |
| eps_HOMO | meV | 757.33 | 270.1 +- 4.3 | 0.915 | 243.1 +- 5.1 | 1.000 | 262.3 | 252.2 |
| eps_LUMO | meV | 1479.26 | 395.3 +- 9.2 | 0.909 | 264.7 +- 6.9 | 1.000 | 391.9 | 315.7 |

(The 3-seed column is not this investigation's product and is not changed by it; it is the existing
reference, and the last two columns are the published values for contrast. The SS row's
improvement over the paper on mu / Delta-eps / eps_LUMO is the Algorithm-3 selection replacing the
base-endpoint harvest, a separate decision recorded in `results/qm9/PROVENANCE.md`.)

The four-way split grid is therefore a **negative result on the transfer of the DNA finding**, and
the honest form of it is: the deficit DNA had was a wrong operating point; QM9's operating point is
right, and this is the measurement that establishes it rather than an assumption. It matters
because nothing in the published record had ever varied it: all 133 rows of
`results/qm9/multimfm_ss_search_grid.csv` record `value_samples = 32`, and the 60 rows of
`results/qm9/multimfm_search_grid.csv` carry no split column at all, the driver having hard-coded
`--value-glass-steps 4 --value-samples 32`. Before this grid the split was an untested default that
happened to be right.

### 3.5 The baselines, on the same NFE axis, with their own knobs already swept

`results/qm9/operating_point/baselines_alpha_matched.csv`. Alpha, n = 100, seeds 1-3, target seeds
2027-2029, every arm paired on `sample_idx`. `paired_vs_ss_4x32` is against the multiMFM-SS arm at
the published split. The baselines' budget knobs were already swept by the existing sweep
(`results/qm9/baselines/baseline_sweep_v2.csv`, jobs 49429598 and predecessors): best-of-N over
N in {4, 8, 16, 32, 64, 128, 256, 512}, DPS over mu in {0.03, 0.1, 0.3, 1.0}, `guide_every` in
{1, 2, 4, 8} and `sample_steps` in {128 ... 8192}, Feynman-Kac over K in {4 ... 128} and lambda in
{0.003, 0.01, 0.03}; each is quoted at its own best point near the matched budget.

| arm | best config | oracle MAE | PB | generative NFE/mol | regressor NFE/mol | vs SS 4 x 32 |
|---|---|---|---|---|---|---|
| best-of-N | N = 128 | **0.924 +- 0.084** | 1.000 | 16,384 | 132 | **-0.248 +- 0.090** |
| best-of-N | N = 64 | 0.973 +- 0.032 | 1.000 | 8,192 | 68 | -0.141 +- 0.093 |
| **multiMFM-SS** | **4 x 32** | **1.020 +- 0.074** | 1.000 | 14,848 | 7,471 | 0 |
| multiMFM-SS | 8 x 16 | 1.196 +- 0.038 | 1.000 | 14,848 | 3,791 | +0.176 +- 0.156 |
| multiMFM-SS | 2 x 64 | 1.219 +- 0.019 | 1.000 | 14,848 | 14,831 | +0.199 +- 0.133 |
| **multiMFM** | **4 x 32** | 1.215 +- 0.106 | 0.887 | 14,848 | 3,684 | +0.097 +- 0.112 |
| Feynman-Kac | K = 128, lambda = 0.01 | 1.465 +- 0.043 | 1.000 | 18,304 | 2,052 | +0.356 +- 0.131 |
| multiMFM-SS | 1 x 128 | 2.316 +- 0.186 | 0.973 | 14,848 | 29,551 | +1.297 +- 0.243 |
| multiMFM | 1 x 128 | 2.787 +- 0.154 | 0.700 | 14,848 | 14,724 | +1.573 +- 0.346 |
| DPS | best anywhere (mu 0.3) | 2.499 +- 0.157 | 0.817 | 243 | 117 | -- |
| best-of-N | N = 512 (4 x the budget) | 0.819 +- 0.048 | 1.000 | 65,536 | 516 | -0.242 +- 0.096 |

Two things follow, and neither is changed by the split:

* **On alpha, best-of-N is ahead of both our rows**, by 2.7 SEM against SS at the nearest budget
  above ours (N = 128, 16,384 NFE against our 14,848, so 10% *more* compute -- but it is already
  1.5 SEM ahead at N = 64, which is *half* the budget), and it pays 57 times fewer regressor calls
  to do it. That is a pre-existing fact of `baseline_sweep_v2.csv`, not a new one; what this grid
  adds is that **no split of the budget rescues it.** 4 x 32 is the best our estimator can do with
  14,848 generative NFE, so the gap to best-of-N is not an operating-point artifact on our side
  either. (Alpha is one property, chosen because it is the one the baseline sweep covers; the
  baselines were never run on the other five, so this is not a Table 2-wide statement.)
* **The budget is saturated, not the split.** `results/qm9/operating_point/budget_alpha_n100_s8/`
  (Slurm 49474230 / 49476059) runs the off-budget point 4 steps x MC 64 -- twice the NFE, 29,696 --
  on both rows at seed 8: multiMFM 1.062 [0.899, 1.225] against 1.112 at 4 x 32, and SS
  **0.805** [0.693, 0.933] against **0.796**. Doubling the estimator budget buys nothing
  measurable. The reason is visible in the SS guide column of section 3.2: at 4 x 32 the SS
  selection already drives the *guide* error to 0.001, i.e. it has already found look-aheads the
  guide considers exact, and the 0.80 that remains is the guide-oracle gap of
  `docs/ss_guide_oracle_gap.md`, which no amount of inference-time search can close. So the right
  statement for the paper is not "multiMFM-SS needs a bigger budget" but "multiMFM-SS is
  guide-limited at the budget it already uses".

## 4. What is not established

1. **The flow-map query mode is a live lead and was not pursued.** Dropping `--mfm-diagonal` at
   4 x 32 gave PB 0.91 vs 0.86 on alpha (MAE +0.155 +- 0.147, inside noise) and, on homo, was the
   argmin for the plain row (237.6 vs 238.6) with an unfiltered SS pool 5 points more valid. Two
   properties at n = 100, one seed, so it is a hypothesis, not a result. Testing it properly means
   the same 3 seeds x 6 properties x n = 1000 that Table 2 uses, which
   `scripts/qm9/operating_point/table2_split.sbatch` would run unchanged with `MFM_DIAGONAL=0`.
   It also raises a question this investigation cannot answer: a single 1-step flow-map query has
   gap 1.0, twice the student's `max_jump 0.5`, so the mode only makes sense at 2 steps or more.
2. **`mu` was deliberately not re-tuned at any split.** The pre-registration holds the published
   guidance strength fixed so that the split is the only knob, which means the losing splits were
   never given the chance to be rescued by a different dose. The gradient-RMS column argues this is
   not what is happening -- 1 x 128 gets the same dose as 4 x 32 and is far worse, while 8 x 16 gets
   3.6 times the dose and is also worse -- but a full (split, mu) grid was not run.
3. **No split was taken to n = 1000.** It was not needed: the pre-registered choice is the
   published split, whose n = 1000 numbers already exist at 3 seeds. Had another split won, stage B
   (`table2_split.sbatch`) was written and ready.
4. **Only two of six properties were gridded.** The pre-registration fixes the split on alpha and
   forbids per-property choice, so the other four were never a decision point; homo was run as a
   shape check. If an author wants the grid on all six, `grid_alpha.sbatch` takes `PROP`.
5. **One card only.** Every number here is an NVIDIA A100-SXM4-40GB on `--partition=kempner`, the
   same card as the reference runs, so nothing crosses card generations. Guided sampling is not
   bit-reproducible, so the n = 100 numbers carry ~0.1 Bohr^3 / ~30 meV of run-to-run noise, which
   is why every claim above is a paired difference with a SEM and not a difference of means.

## 5. Files and runs

* `docs/qm9_operating_point.md` (this file)
* `scripts/qm9/operating_point/run_split.sh` -- one run at an explicit (steps, MC) split; the
  published Table 2 arguments are copied into it, and `scripts/qm9/run_multimfm.sh`,
  `scripts/run_steer_search.sh` and `src/multimfm/*.py` are **not modified**, so no stored result
  can change
* `scripts/qm9/operating_point/grid_alpha.sbatch` (`PROP`/`N`/`SEED`/`SPLITS`/`OUT_ROOT`),
  `grid_eval_seeds.sbatch`, `table2_split.sbatch` (written, not needed), `collect.py`,
  `baselines_matched.py`
* `results/qm9/operating_point/{grid_alpha_n100_s8,grid_homo_n100_s8,evalseeds_alpha_n100,budget_alpha_n100_s8}/`
  -- `summary.csv`, `paired.csv`, `unguided_control.csv`; plus `baselines_alpha_matched.csv` and
  `PROVENANCE.md`

Slurm: 49464797 + 49464822 (alpha grid, seed 8), 49465288 + 49476040 (homo grid, seed 8),
49464799 + 49471391 (alpha, evaluation seeds 1-3), 49474230 + 49476059 (the off-budget 4 x 64
probe). All on `--partition=kempner`, A100-SXM4-40GB.

## 6. Consequences for the paper, for the authors to decide

* **Nothing in Table 2, 21 or 22 changes.** The pre-registered split is the published one. What can
  be added is a sentence that the split was *checked*: at matched value-gradient NFE, 4 posterior
  steps x 32 posterior samples is the best of 1 x 128, 2 x 64, 4 x 32 and 8 x 16 on both rows, and
  the MC-heavy end degrades both accuracy and PoseBusters validity. That closes the hole
  `docs/dmfm_deficit_rescue.md` section 4c opened -- "the paper must state the split, not just the
  NFE, because the two posteriors prefer opposite ends of that trade" -- for the molecular side, and
  it now has a measurement behind it on both datasets rather than one.
* **The two datasets genuinely prefer opposite ends, and the reason is the artifact, not the
  domain.** DNA's best operating point is one composed jump with MC 32 because the artifact the
  step-matched-student rule selects at one step there was distilled for one jump; QM9's is four
  steps with MC 32 because its only student was distilled for a four-step diagonal rollout
  (`rollout_steps 4`, `rollout_diagonal True`, `max_jump 0.5`). If the paper says anything general
  about how to spend a value-gradient budget, the invariant is **"match the split to the split the
  student was distilled for"**, not "fewer steps and more draws".
* **A wording precision, not a mismatch.** Section 7's "QM9 uses four inner steps and 32 posterior
  samples per outer step" is correct, but the four QM9 inner steps are **Euler steps of the
  *diagonal* meta velocity** (`--mfm-diagonal` sets `u = s` in
  `glass_guidance.differentiable_mfm_posterior`), whereas the ten GEOM inner steps are described in
  the same sentence as "flow-map inner steps" and are composed two-time jumps. Both are the meta
  flow map, but they are two different queries of it, and the QM9 one is the one Proposition 1's
  diagonal identity licenses. Worth one clause, because a reader who takes "inner steps" to mean
  composed jumps in both places will mis-picture the QM9 estimator.
* **multiMFM-SS is guide-limited, not budget-limited.** At the published operating point the SS
  selection already drives the guide error to 0.001 Bohr^3 on alpha; doubling the estimator budget
  to 4 x 64 moves the oracle MAE from 0.796 to 0.805, i.e. not at all. Any claim that more
  inference-time compute would improve multiMFM-SS is not supported; the binding constraint is the
  guide-oracle gap of `docs/ss_guide_oracle_gap.md`. This is directly relevant to the
  best-of-N comparison, where best-of-N is ahead on alpha at the nearest budget above ours
  (2.7 SEM) and already ahead at half of it (1.5 SEM, section 3.5), and cannot be caught by
  re-splitting or by spending more.
* **One lead, not a result: the flow-map query mode.** Same checkpoint, no `--mfm-diagonal`, at
  4 x 32: PoseBusters 0.91 vs 0.86 on alpha and the argmin on homo's plain row, at an MAE cost
  inside noise. If validity is the metric under pressure (Table 21), this is the cheapest thing left
  to try, and `scripts/qm9/operating_point/table2_split.sbatch` runs it at Table 2 scale with
  `MFM_DIAGONAL=0`. It was deliberately excluded from the pre-registered selection, so treating it
  as a result would be exactly the best-of-grid move this document was built to avoid.
