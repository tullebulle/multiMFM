# A second GEOM-Drugs steering experiment: candidates and design

Draft 2026-09-30, written directly rather than by agents (the org spend limit stopped three
brainstorming agents mid-run). Nothing here is implemented or measured yet — this is a proposal to
argue with, and every number marked *to measure* is exactly that.

## What killed the first GEOM task, stated as a design constraint

Table 3 steered an FP2 structural fingerprint. It failed for a reason that was knowable before any
GPU time was spent, and the reason generalises into a filter:

1. **FP2 is a graph-topology fingerprint.** Measured: exactly invariant to coordinate perturbations
   below 0.006 Å, randomised by 0.2 Å. There is no displacement scale at which it responds
   *smoothly*, so a coordinate gradient has nothing to descend.
2. **The metric barely discriminated.** Random-pair Tanimoto on that benchmark is 0.2164; our
   unguided 0.2278 and TFG-Flow's guided 0.208 both sat at the floor.
3. **Guidance is annihilated at the end of the trajectory.** `CenteredMetricInterpolant.step` is a
   contraction onto the endpoint prediction, so an injected perturbation reaches the endpoint scaled
   by ~`dt/(1-t)` — measured as **0** for coordinates with the network stubbed out. Guidance acts
   only *through the network's response*, never directly.
4. **The discrete channel was effectively unsteerable**: one shared `mu` for both modalities put the
   atom-type step at 0.19% of state RMS against the ~1 needed to flip an argmax.

**Filter for any replacement.** The reward must (a) be a smooth function of the coordinates with a
non-vanishing gradient at *small* displacement, (b) have measured headroom well above its own noise
floor, (c) be steerable through the network's endpoint response rather than by direct perturbation,
and ideally (d) be computable **exactly**, so no learned guide sits between the claim and the
evidence.

## Candidates, ranked

### 1. Radius of gyration (and the gyration tensor more generally) — *recommended*

$R_g^2 = \frac{1}{N}\sum_i \lVert x_i - \bar{x}\rVert^2$.

- **Exactly computable, exactly differentiable**, no learned guide, no oracle gap to defend. The
  evaluator disappears from the argument entirely — which is the single biggest structural weakness
  of Table 3 (our guide, our metric, both at the floor).
- **No threshold behaviour.** Moving one atom by δ changes $R_g$ by $O(\delta/N)$ — linear, smooth,
  responsive at *any* scale. This is precisely the property FP2 lacked.
- Chemically meaningful: $R_g$ is molecular compactness, it correlates with permeability and
  packing, and GEOM-Drugs has a broad natural distribution of it.
- The gradient is dense (touches every atom) and well conditioned, unlike a fingerprint's.
- **Extends naturally**: the full gyration tensor gives principal moments, asphericity, and the
  NPR1/NPR2 shape descriptors standard in cheminformatics — a 2-D shape target rather than a scalar.

### 2. Gaussian shape overlap with a reference conformer (ROCS-style)

Smooth differentiable overlap between atom-centred Gaussians. This is a *real* drug-discovery task
(shape-based virtual screening), is geometry-native, and targets can be drawn from GEOM itself.
Costlier than $R_g$ and needs alignment handling, but it is the most publishable framing if it works.

### 3. Force-field strain / MMFF94 energy

RDKit computes it with analytic gradients. Attractive because steering toward low strain should
*improve* PoseBusters validity rather than destroying it — the opposite of the composition-edit
failure. Risk: the base model already produces low-strain conformers, so headroom may be small.

### Not recommended

- **Anything fingerprint- or graph-derived.** Same failure mode as FP2.
- **Dipole moment** — needs partial charges; with fixed Gasteiger charges it is differentiable in
  coordinates, but the charge model becomes an unstated part of the reward.
- **Docking scores** — too expensive per evaluation for Monte-Carlo value estimation.

## The design choice that decides whether this is worth running

For a cheap scalar target, **best-of-N is a very strong baseline**: generate N, keep the closest.
Guidance has to beat it at matched compute, and on DNA it only did so at low NFE.

The principled response is to pick targets **in the tail of the base distribution**, where rejection
sampling is exponentially expensive and steering is actually the point. This is a defensible
experimental design — steering matters precisely when the target is rare — and it must be stated as
the design rationale *before* running, not discovered afterwards. Targets at the base distribution's
median would favour best-of-N and would be the wrong experiment, not a worse result.

Concretely: measure the base model's $R_g$ distribution first, then set targets at roughly ±1, ±2
and ±3 SD, and report the whole ladder. If guidance only wins in the far tail, that is the finding
and it should be reported as such.

## Pre-flight test, to run before committing anything

The test that would have killed FP2 in an afternoon:

1. Sample ~64 molecules from the base model.
2. Compute the reward's finite-difference sensitivity at the displacement magnitudes guidance
   actually delivers (to be measured — the annihilation work gives the scale), and compare it with
   the reward's own noise floor across independent samples.
3. **Pass condition**: the reward responds monotonically and well above noise at those magnitudes.
   FP2 fails this outright; $R_g$ should pass trivially, which is the point of running it on both —
   $R_g$ as the positive control, FP2 as the known negative.
4. Also measure floor and ceiling: random-pair reward spread, and best-of-4096 as the attainable
   ceiling. **Abandon the task if headroom cannot be resolved at the planned n.**

## Evaluation protocol (sketch — to be filled in before running)

- **Baselines**: best-of-N, Feynman–Kac, beam, MCTS (port from `src/dmfm/benchmarks/search.py`),
  plus DPS from `src/multimfm/baselines.py`. Best-of-N is the one that matters.
- **Compute axis**: forward-equivalent, with the backward cost *measured* on this code path, not
  assumed. Our audit found backward costs 1.1 forwards on DNA and 2.89–3.00× for the whole QM9
  guided step. Report wall-clock too — on DNA there was no wall-clock regime where guidance won.
- **Validity jointly with every number**, PoseBusters. Non-negotiable: the GEOM rescue showed over
  half the attainable headroom on the fingerprint task could be taken while leaving 85% of molecules
  invalid.
- **Pre-register** n, seeds, the paired structure, the hyperparameter selection rule, and what
  result would falsify the claim — all before running. Table 3 was best-of-four-grid at n=64, one
  seed, and reversed at n=4096.
- A negative result under this protocol is publishable and gets reported as readily as a positive.

## Open questions for the authors

1. $R_g$ or shape overlap? $R_g$ is safer and cleaner; shape overlap is the better story if it works.
2. Is a tail-target design acceptable as the headline framing, with the full ladder reported?
3. Is an exactly-computable reward preferable to a learned guide here? I think yes — it removes the
   evaluator from the argument — but it does make the task less like TFG-Flow's setup.
