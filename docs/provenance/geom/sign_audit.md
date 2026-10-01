# End-to-end audit of the guidance sign, in the code path the GEOM driver uses

Done by hand on 2026-09-29, after the held-out sweep and the 4,096x3 rerun both showed that
steering lowers *the guide's own* head-predicted Tanimoto as well as the counted FP2 one
(`results/geom/head_tanimoto_direction.json`). Path:
`multimfm.geom.mfm_steering` -> `multimfm.geom.steering.run_steering` ->
`multimfm.glass_guidance.guided_sample` -> `value_gradient` ->
`production_step_with_optional_guidance`.

| step | code | orientation |
|---|---|---|
| reward | `multimfm/geom/reward.py:fingerprint_reward` returns `-((pred - target)**2).sum(-1)` with `pred = sigmoid(head(x1))` | higher = closer to the target `c`; this is `r(G) = -||E(G)-c||^2` as the paper states |
| tilt | `glass_guidance.py:647` `scaled_rewards = reward_scale * rewards` | `reward_scale` is lambda >= 0, so monotone increasing in `r` |
| value | `:648` `values = logsumexp(scaled_rewards, -1) - log(N)` | `= log (1/N) sum_i exp(lambda r_i)`, the finite-sample `V_t`; increasing in every `r_i` |
| gradient | `:651` `grads = autograd.grad(value_sum, [coords, atomics])` | **`+grad V`**, not `-grad`: there is no loss negation, no `backward()` on a minimised objective, and no `-` anywhere between the reward and the step |
| projection | `mask_and_zero_com(grad_coords, ...)`, `apply_mask(grad_atomics, ...)` | linear projections (zero out padding, remove the centre-of-mass component): sign-preserving |
| clipping | `clip_guidance_rms` scales by `min(max_rms/rms, 1) in (0, 1]` (and is disabled by default, `--guidance-max-*-rms 0.0`) | non-negative scale: sign-preserving |
| step | `:729` `next_coords = base_step(x_t) + mu * grad_coords * dt` (same for atomics) | forward-Euler step of `dX = (b_t + mu grad V_t) dt`: **ascent** on `V` for `mu > 0` |
| driver | unguided arm `mu=0.0, value_samples=0`; guided arm `mu=args.mu > 0`, same `init_state`, RNG captured before / restored after the gradient so the base step consumes identical randomness | properly paired comparison |

**Conclusion of the audit: there is no sign inversion in this code path.** Every link between
the reward and the state update is orientation-preserving, and the guided arm ascends `V`.

Two caveats on what an audit can and cannot settle:

1. Monotonicity does not discriminate. Larger `mu` makes the results monotonically worse
   (`results/geom/table3_asdescribed_tuning.csv`), which is equally consistent with a flipped
   sign and with a correctly signed but badly scaled or noise-dominated direction. Only the
   code path settles the sign, and it says the sign is right.
2. What inspection cannot check is whether the **estimated** gradient points uphill at all.
   It is a 4-sample log-mean-exp differentiated through a 10-step flow map; if the estimate is
   dominated by one sample or by noise, the update is a random perturbation with the correct
   *formula* and no useful direction. That is exactly what
   `python -m multimfm.geom.diagnostics grad-sign` measures: the guide reward of fresh
   look-aheads at `x` versus `x +- eps * grad`.

## Scale mismatch worth reporting regardless of the diagnostic

The paper's sampler (eq. ODE / multimodal steering) carries the coefficient
`sigma_t^2/2 = (1-t)/t` in front of `grad V`; the implementation uses a **constant** `mu`.
With the recorded window `t in [0.1, 0.9]` that substitution is wrong by a large, *t*-dependent
factor:

| t | paper coefficient `(1-t)/t` | code (`mu = 0.5`) | code / paper |
|---|---|---|---|
| 0.1 | 9.0 | 0.5 | 0.06x (18x under-guided) |
| 0.3 | 2.33 | 0.5 | 0.21x |
| 0.5 | 1.0 | 0.5 | 0.5x |
| 0.67 | 0.5 | 0.5 | 1.0x |
| 0.9 | 0.111 | 0.5 | 4.5x (over-guided) |

So the implemented sampler under-guides early -- where composition and scaffold are decided --
and over-guides late, where perturbing an almost-final geometry is what breaks PoseBusters
validity and the counted fingerprint. This is a plausible mechanism for "PB drops and both
similarities drop", and it is a *paper-vs-code* discrepancy independent of the diagnostic's
outcome (either the equations should say that `mu` absorbs `sigma_t^2/2`, or the sampler should
apply `(1-t)/t`).

## Implemented: the paper's coefficient as an option

`--mu-schedule paper` pre-multiplies the value gradient by `(1-t)/max(t, 0.05)` before the
constant `mu`, which is algebraically the paper's `sigma_t^2/2` with `mu` as an overall scale
(the step is linear in the gradient). Numerically verified: at `t = 0.1` the guided update is
exactly 9.000x the constant-`mu` update, matching `(1-t)/t`; the installer is idempotent and
reverting to `constant` restores the original engine function object. The clamp at `t = 0.05`
caps the coefficient at 19 and never binds inside the recorded window `t in [0.1, 0.9]`.

## Ruled out by inspection

* **Padded-length mismatch.** The head was trained with `max_len=82` and the driver builds it
  with `max_len=71`. `diffusion/predictor.py:Predictor.forward` stores `max_len` but never uses
  it (masked mean-pool over the node features), so this cannot affect the prediction.
* **Coordinate frame.** The head was trained on Angstrom/2.0 and is fed
  `state["coords"]`, which is the model's /2.0 frame; the FP2 recount multiplies back by the
  normalizer before calling OpenBabel.
* **Target/prediction alignment.** `grouped_rewards` repeats a 2-D target along the batch axis
  (`repeat_interleave(posterior_samples, dim=0)`), matching the look-ahead layout.

## Remaining hypotheses and their one-job tests

| hypothesis | one-job test |
|---|---|
| estimator variance: 4 look-aheads make `grad log mean exp` a near-single-sample gradient (at lambda = 1, the paper's literal reward, `lambda*|r| ~ 100` and the log-mean-exp is a hard max) | `VALUE_SAMPLES=16 TAG_SUFFIX=_vs16 GRID_OVERRIDE="0.01:0.25" sbatch scripts/geom/tune_lambda_mu.sbatch` (and 32); positive gain with more samples = estimator variance |
| constant `mu` instead of `sigma_t^2/2` | **implemented and queued**: `multimfm.geom.mu_schedule` adds `--mu-schedule {constant,paper}` (default `constant`, so nothing already run changes meaning). Held-out sweep with the paper coefficient: Slurm array **49293831** (CPU, the only route; the GPU copy 49281926 was cancelled under the one-route rule), lambda = 0.01 x mu-scale in {0.05,0.15,0.5} and lambda = 0.02 x 0.15, n=256, against a matched constant-mu arm on the same device (49293829, 49293743) |
| soft-atomics guidance hurting composition (softmax temperature 0.25 makes the atom gradient nearly zero except near ties; the diagnostic prints `grad_atomics_rms` vs `grad_coords_rms`) | `EXTRA_ARGS="--no-guide-atomics" TAG_SUFFIX=_coordsonly GRID_OVERRIDE="0.01:0.25" sbatch scripts/geom/tune_lambda_mu.sbatch` |
| genuine defect in the flow-map look-ahead itself (the student is a 10-step map used at `t_cond` up to 0.9 while trained to `t_max = 0.95`) | `grad-sign` at several `--t` values (0.2, 0.5, 0.8) |
