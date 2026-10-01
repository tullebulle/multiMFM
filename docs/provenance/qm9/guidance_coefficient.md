# QM9 steering uses a constant guidance coefficient, not the paper's sigma_t^2/2

Established 2026-09-29 by reading the code (no rerun; nothing was changed on account of this).

## The fact

The paper's guided ODE, Eq. `eq:multimodal steer` (§2, "Multimodal Steering"), is

    dY_t = ( b^y_t + (sigma_{y,t}^2 / 2) * d_y V_t ) dt
    dA_t = ( b^a_t + (sigma_{a,t}^2 / 2) * d_a V_t ) dt

and the text immediately after it states "For the common linear schedule,
sigma_{y,t}^2 = sigma_{a,t}^2 = 2(1-t)/t" and "These ODEs are used for the reported
steering experiments". So the stated guidance coefficient is **time dependent**:

    sigma_t^2 / 2 = (1 - t) / t

Every QM9 steering run instead applies a **constant** coefficient. The guidance term
enters the state update in exactly one place, `production_step_with_optional_guidance`
in `src/multimfm/glass_guidance.py`:

```python
next_coords  = model.coords_interpolant.step(state, pred, t, dt)
next_atomics = model.atomics_interpolant.step(state, pred, t, dt)
if grad_coords is not None:
    next_coords  = mask_and_zero_com(next_coords  + mu * grad_coords  * bcast(dt, next_coords),  ...)
if grad_atomics is not None:
    next_atomics = apply_mask(       next_atomics + mu * grad_atomics * bcast(dt, next_atomics), ...)
```

i.e. `x_{t+dt} = base_step(x_t, t, dt) + mu * grad V * dt` with `mu` a scalar CLI
argument (`--mu`), constant over the whole trajectory. Both molecular entry points use
this one function: `guided_sample` (`property_targets.py`, the multiMFM row) and
`guided_sample_collect` (`steer_search.py`, the multiMFM-SS row). There is no `sigma_t`,
no `(1-t)/t`, and no other t-dependent factor anywhere in the guidance path. (The one
`(1.0 - t_b)` in `glass_guidance.py` is inside `differentiable_joint_glass_posterior`'s
`teacher_v`, which converts an endpoint prediction into a velocity for the *GLASS value
sampler*; QM9 runs use `--value-sampler mfm` and never reach it.)

This is the same discrepancy the GEOM agent found in the GEOM steering code. It applies
to **both** QM9 rows of Table 2 and therefore to Tables 21 and 22 as well, and to the
2026-09-29 3-seed reruns, which deliberately keep the published sampler.

## What the tuned constants correspond to

`ratio(t) = mu / ((1-t)/t) = mu*t/(1-t)` is the factor by which the code's coefficient
exceeds the paper's. It is 1 at `t* = 1/(1+mu)`. Guidance is active on t in [0.05, 0.95]
(`--guide-min-t` / `--guide-max-t`).

| property | row | mu | ratio at t=0.05 | at t=0.5 | at t=0.95 | t* (ratio = 1) |
|---|---|---|---|---|---|---|
| Cv | multiMFM | 2.80 | 0.147 | 2.80 | 53.2 | 0.263 |
| Cv | SS | 5.0 | 0.263 | 5.00 | 95.0 | 0.167 |
| mu | multiMFM | 0.78 | 0.041 | 0.78 | 14.8 | 0.562 |
| mu | SS | 3.0 | 0.158 | 3.00 | 57.0 | 0.250 |
| alpha | multiMFM | 3.40 | 0.179 | 3.40 | 64.6 | 0.227 |
| alpha | SS | 11.0 | 0.579 | 11.00 | 209.0 | 0.083 |
| gap | multiMFM | 0.17 | 0.009 | 0.17 | 3.2 | 0.855 |
| gap | SS | 1.0 | 0.053 | 1.00 | 19.0 | 0.500 |
| HOMO | multiMFM | 0.50 | 0.026 | 0.50 | 9.5 | 0.667 |
| HOMO | SS | 1.5 | 0.079 | 1.50 | 28.5 | 0.400 |
| LUMO | multiMFM | 0.44 | 0.023 | 0.44 | 8.4 | 0.694 |
| LUMO | SS | 1.5 (3.0 in the rerun) | 0.079 (0.158) | 1.50 (3.00) | 28.5 (57.0) | 0.400 (0.250) |

So the constant-mu sampler **under-weights guidance early** (1-58% of the stated
coefficient at t = 0.05) and **over-weights it late** (3-209x at t = 0.95), crossing the
paper's coefficient once, at t* between 0.08 and 0.86 depending on the property.

## Why this does not invalidate the QM9 numbers

`mu` was tuned per property against the oracle (a 5-round grid for the multiMFM row, the
`v*`/`k_*` sweeps for SS), so the reported numbers are those of a tuned constant-coefficient
sampler, and they are reproduced by the reruns. What is wrong is only the *claim* that the
reported experiments integrate Eq. `eq:multimodal steer`: they do not. Note also that the
constant-mu form is not merely a rescaling of the paper's ODE - it changes the relative
weight of the guidance across t, so it is a different sampler, not the same sampler with a
different step size.

## Recommendation for the text

Either (a) state plainly that the molecular experiments use a constant guidance strength
`mu` per property (tabulated), and present Eq. `eq:multimodal steer` as the exactness
result that motivates the method rather than as the sampler that was run; or (b) rerun with
the `(1-t)/t` coefficient and retune. The reruns here take (a): they reproduce the published
sampler so that the 3-seed numbers stay comparable to the submission.
