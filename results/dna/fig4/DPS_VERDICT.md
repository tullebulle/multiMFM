# DNA steering: dMFM does not beat DPS, and why

Adding DPS to the DNA comparison (it was only ever in the QM9 one) changed the picture, and
not in our favour. This records what was tried and what the honest conclusion is.

## The result

Held-out oracle MAE at the rare target $y^\star = +1$, forward-equivalent calls:

| budget | dMFM (guide) | DPS (guide) | Feynman--Kac | best-of-N |
|---|---|---|---|---|
| 192 | 0.181 (0.121) | **0.153** (0.039) | 0.981 | 0.993 |
| 288 | 0.177 (0.102) | **0.155** (0.039) | 0.623 | 0.872 |
| 480 | 0.162 (0.076) | **0.152** (0.040) | 0.410 | 0.726 |
| 864 | 0.162 (0.056) | **0.156** (0.045) | 0.270 | 0.551 |

DPS wins at every budget, at both targets ($C_0 = 0.30$ and $y^\star = +1$).

## What was tried first, in case the gap was ours

1. **The scale-free gradient rule.** Figure 4's sampler clipped the gradient down only, while
   every other DNA result in the paper rescales it to exactly `grad_clip`. That is worth
   +0.109 -> +0.124 paired on Table 13, and this benchmark was the one place not using it.
   Fixing it moved us 0.190 -> 0.162 at the lowest budget. **Real defect, now fixed.**
2. **Guidance strength.** The dMFM ran at the harness default of 8 while DPS's strength was
   pilot-tuned. Giving ours the same pilot moved it 0.190 -> 0.184, and the first grid was
   boundary-limited (it chose 32, the top), so the grid was widened.
3. **The (window, MC) split.** Our cost is $96 + G(\text{MC} + 1.1)$, so a budget can be spent
   as a wide guidance window with few posterior samples or a narrow one with many -- the knob
   that moved Table 1 from 0.193 to 0.153 at fixed cost. Sweeping it gained 0.177 -> 0.162 at
   480 and nothing elsewhere.

After all three, DPS is still ahead by 0.006--0.028 at every budget.

## Why, and what it means

**The task is guide-limited.** Every arm steers with the same guide and is scored by a
held-out oracle. DPS drives the guide to 0.039 -- essentially perfect -- and lands at oracle
0.153. That 0.153 is the guide--oracle disagreement, and no method that only sees the guide
can go below it. DPS reaches that ceiling at the cheapest budget and is then **flat across a
4.5x budget range**; the dMFM walks down toward it from above and runs out of room.

**The one thing that does favour the dMFM** is transfer: our guide-to-oracle degradation is
0.060--0.106, against DPS's 0.111--0.114. We lose less of what we gain. But DPS optimises the
guide so much harder that it still wins in absolute terms. A method combining DPS's guide
optimisation with our transfer would land near 0.10; neither method is that.

**What survives for the paper.** At the rare target both guidance methods beat every search
baseline by 1.6--6x (0.15--0.18 against 0.27--0.99), where at $C_0 = 0.30$ they were
competitive -- so the case for guidance over search is strongest exactly where the target is
rare. And Figure 4's left panel, which measures value estimation directly, is unambiguous:
the dMFM falls 4.80 -> 0.247 with $N$ while DPS is flat at 2.863. Better value estimation is
demonstrated there; on this reward it does not convert into a better steered sample, because
the guide runs out first.

**Recommendation.** Lead the steering comparison with QM9, where the oracle does not saturate
(best-of-N moves 5.82 -> 1.09 across the same budget range) and multiMFM beats DPS
like-for-like by 1.07--2.21x. Report the DNA steering comparison as guidance versus search,
with DPS included and this tie stated rather than omitted.

Producers: `scripts/dna/{dps_ladder,dmfm_strength_pilot,dmfm_scalefree_pilot,fig4_hard_target,fig4_split_sweep}.sbatch`,
`scripts/dna/run_dps_steering.py`, `src/dmfm/benchmarks/dps_steering.py`.
