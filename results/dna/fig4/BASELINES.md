# Which baselines the steering comparison shows, and why

## The two panels used different baselines, for no good reason

The DNA comparison ran best-of-$N$, Feynman--Kac, beam search and MCTS; the QM9 one ran
best-of-$N$, Feynman--Kac and DPS. That is an artifact of two independently written
sweeps, not a decision: the DNA harness had no DPS steering path and the QM9 one offers no
tree search. Both panels now show **best-of-N, Feynman-Kac and DPS**.

* **DPS was added to DNA** (`src/dmfm/benchmarks/dps_steering.py`). It is the standard
  gradient-guidance baseline and the arm the dMFM most directly replaces, and the DNA
  value-estimation panel already scores the same estimator -- this only turns it into a
  sampler. Everything but the value estimator is shared with the dMFM arm.
* **MCTS and beam moved to this appendix.** Neither is the best baseline at any budget
  (Feynman--Kac is, at all four), so removing them from the figure does not widen our
  margin by a single point; the margin is measured against Feynman--Kac either way.

## MCTS is best-of-N with a shared prefix

At matched *rollout count* the two are statistically indistinguishable:

| rollouts | MCTS | best-of-N | diff | z | MCTS NFE | best-of-N NFE | saving |
|---|---|---|---|---|---|---|---|
| 2 | 0.37723 | 0.39029 | $-0.013$ | $-1.43$ | 192 | 192 | 0% |
| 3 | 0.29728 | 0.29987 | $-0.003$ | $-0.57$ | 272 | 288 | 6% |
| 5 | 0.21097 | 0.20760 | $+0.003$ | $+0.52$ | 428 | 480 | 11% |
| 9 | 0.12920 | 0.13352 | $-0.004$ | $-0.89$ | 695 | 864 | 20% |
| 16 | 0.07674 | 0.07671 | $+0.00003$ | $+0.01$ | 1{,}076 | 1{,}536 | 30% |
| 32 | 0.04290 | 0.04170 | $+0.001$ | $+0.64$ | 1{,}553 | 3{,}072 | 49% |

**This is a property of the task, not a defect in the implementation.** Reading
`search.mcts`: each iteration descends by UCT to a node that is not yet fully expanded,
extends it by `checkpoint_every` steps, rolls that child out to completion, and backs up
`exp(r)`; the best completed rollout is returned. So $I$ iterations produce $I$ complete
samples and keep the best -- structurally best-of-$I$, with the samples correlated through
their shared prefixes and cheaper for the same reason.

The tree buys nothing because **the reward is terminal-only**: there is no intermediate
signal for UCT to exploit, so the search can only decide which prefix to extend, and with
six levels (96 steps, `checkpoint_every` 16) and at most 64 iterations it never amortises.
Its one real gain is the shared prefix, worth 0--49% of the network calls, which is why its
curve sits slightly left of best-of-$N$ rather than below it.

The configuration shown (`children=2, checkpoint_every=16, c_uct=0.5`) was chosen on a
separate pilot of 24 settings at matched NFE, so this is MCTS at its tuned setting here,
not a strawman.
