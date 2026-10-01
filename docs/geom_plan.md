# GEOM: what to run, and why not the fingerprint task

Written 2026-10-01, after the QM9 steering figure settled. This is the plan for the GEOM
experiment and the argument for changing the reward rather than repairing the old one.

## The fingerprint task is not repairable

Three problems, measured:

1. **Guidance is annihilated at the last step** for the *categorical* channel. Both
   interpolants contract onto the endpoint prediction with coefficient 1 at $t=1$, so a
   perturbation injected anywhere in the guided window survives with magnitude 0 in
   coordinates and $\le 3\times10^{-11}$ in atom types, measured with the network replaced
   by a stub. Guidance reaches the molecule only through the network's response.
2. **The reward is the wrong surrogate.** Spearman between $-\lVert p-c\rVert^2$ and the
   counted FP2 Tanimoto is 0.375; for a differentiable Tanimoto it is 0.708.
3. **The benchmark sits at its own floor.** Random molecule pairs score **0.2164**; our
   unguided generator 0.2278 and TFG-Flow's guided 0.208. Every number anyone reports on
   this task is within noise of drawing two molecules at random.

(1) and (2) are fixable. (3) is not, and it is the same failure we hit on DNA, where the
guide saturates at the guide-oracle disagreement and no method can be distinguished from
any other. Worse, the one lever that *does* move the metric -- editing composition -- takes
over half the attainable headroom while leaving **85% of molecules PoseBusters-invalid**,
and random substitution destroys validity just as fast, so the movement is not chemistry.

A task whose metric cannot separate methods cannot show that one method is better.

## What to run instead: radius of gyration

$R_g^2 = \frac{1}{N}\sum_i \lVert x_i - \bar x\rVert^2$, steered to a target value.

| problem above | with $R_g$ |
|---|---|
| learned guide, hence a guide-oracle gap that saturates | **no guide at all**: the reward is computed exactly from coordinates |
| reward correlates 0.375 with the metric | the reward **is** the metric |
| categorical channel annihilated by the interpolant | $R_g$ is a coordinate target, which the measurement showed is unaffected |
| metric pinned at the random-pair floor | smooth, unbounded, and the target is ours to place |

It is already supported: `radius_gyration` is a registered property, so the existing
`target_property` reward $-(R_g-c)^2$ works without new reward code. It is chemically
meaningful -- molecular compactness, which relates to packing and permeability -- and it
lets the whole QM9 apparatus carry over: the value-estimation panel and the matched-NFE
comparison against DPS, best-of-$N$ and Feynman--Kac.

## The design decision that has to be made before running

For a cheap scalar target, **best-of-$N$ is a very strong baseline**: draw $N$, keep the
closest. Steering has to beat it at matched compute. The honest way to make the task
meaningful is to place targets in the **tail** of the base distribution, where rejection
sampling is exponentially expensive and steering is the point -- and to say so in advance
rather than discover it afterwards. Targets at the median would favour best-of-$N$ and
would be the wrong experiment, not a worse result.

## Pre-flight, which decides whether this is worth running at all

`scripts/geom/rg_preflight.py`: sample unguided molecules from the GEOM teacher and measure

* the $R_g$ distribution (mean, spread, quantiles) -- this sets what "ordinary" and "rare" mean;
* **how close best-of-4/16/64 gets to targets at $+1$, $+2$, $+3$ standard deviations.**

That last number is the bar. Where best-of-$N$ already lands on target there is nothing to
win, and the task should be abandoned at that budget rather than run and written up. This
is the check that would have killed the fingerprint task in an afternoon.

## If it passes

1. Wire `--reward radius_gyration` into the GEOM steering CLI (targets per molecule, drawn
   from the declared tail).
2. Value-estimation benchmark for $R_g$, as on QM9: multiMFM at $N$ posterior samples
   against DPS and FMAP, with a large reference and its own self-split floor plotted.
3. Matched-NFE steering against best-of-$N$, DPS and Feynman--Kac, with PoseBusters
   validity reported jointly with every number -- non-negotiable after what the composition
   rescue showed.
4. Pre-register $n$, seeds, the pairing, the selection rule and what result would falsify
   the claim, before running the evaluation.

A negative result under this protocol is publishable and gets reported as readily as a
positive one.
