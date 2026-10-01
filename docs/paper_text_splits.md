# Data splits: text to add to the paper

The parent-disjoint design is one of the paper's genuine strengths and it is currently buried in
App. F.1. It should be stated once, early, with the counts. All figures verified from
`data/dna/yeast_parent_disjoint/parent_split_seed0.json` and the checkpoint metadata on
2026-09-30.

## For §3.1, where the DNA experiment is introduced

> The loop-seq assay provides 100-bp records, so a central 50-bp yeast window flanked by fixed
> 25-bp adapters carries a measured C0 value. Consecutive central windows overlap by 43 bp (stride
> 7), so they reconstruct into 576 contiguous 1,044-bp parent sequences. **We split the 576 parents
> once, before any window is extracted, into two disjoint halves of 230 training and 29 validation
> parents each, plus 58 held-out test parents (8,294 windows). The generator and the guide
> predictor $f_{\text{guide}}$ are trained on the first half; the oracle $f_{\text{oracle}}$ on the
> second; and every number we report is evaluated on the test parents, which none of the three has
> seen.** Because the split precedes window extraction, no window in one partition overlaps a window
> in another --- a split applied after extraction would place overlapping 43-bp neighbours on both
> sides.

Then the existing sentence about guide and oracle accuracy follows naturally, and is worth keeping
adjacent because the near-equality is the evidence the design is honest:

> On the 58 held-out test parents the guide attains Pearson $r = 0.877$ with MAE 0.159 and the
> oracle $r = 0.872$ with MAE 0.162. Their independence is in the training data --- disjoint parents
> and different seeds --- not in the architecture.

**Why state it early.** The force of the DNA result is that target attainment transfers to a
predictor trained on parents the generator never saw. A reader who meets the split only in an
appendix cannot weigh the main claim when they read it.

## For QM9

We do **not** reverse the EDM/QM9 split convention. The generator is trained on `train_flow`, while
the evaluation oracle is trained on `train_classifier`, matching the usual independent-evaluator
protocol. Our additional guide is trained on the generator half, so the reported QM9 errors are
measured by a predictor that did not provide the steering signal. This makes the evaluation less
susceptible to guide-specific reward hacking.

> The QM9 generator is trained on the first half of the dataset and the evaluation oracle on the
> second, following the standard protocol. The guide used for steering is a separate predictor
> trained on the generator's half, so the errors we report are measured by a model that supplied no
> part of the steering signal.

**Two claims retracted here (2026-09-30), both unverified when written.** An earlier draft said our
assignment was "the reverse of TFG-Flow's convention", which is not supported by anything in this
repository --- we vendored TFG-Flow's EGNN, predictor and constants, none of its split code. A
follow-up draft then asserted that TFG-Flow "optimises and scores with the same model"; that is also
unsupported, and TFG-Flow appears from its paper to use an independent oracle predictor for QM9
evaluation. Do not make a comparative claim about TFG-Flow's setup without citing the exact
config, command or table where it is specified.

## What to retire

The earlier `split65k` partition --- a random 65k/8k/9,368 split applied **after** window extraction,
so overlapping windows straddle train and test --- is behind the published Tables 4--6. Those are
being regenerated on the parent-disjoint split, and `split65k` should not appear in the paper. Its
models were purged in any case.

## The principle, for the appendix

Different datasets need different splits, but one rule should hold across all three experiments and
is worth stating once: **the evaluator never sees what the generator or the guide saw.** DNA
satisfies it by construction; QM9 satisfies it for the oracle; and on GEOM an exactly-computable
reward would satisfy it trivially, since there is then no learned evaluator at all.

## Table 16 replacement

Drop the Pearson rows (on 4 constrained numbers vs a sampled uniform, r is exactly Uniform(-1,1)).
Replace with total-variation distance of global base composition:

| comparison | TV |
|---|---|
| two disjoint real splits (floor) | 0.0101 |
| generated vs held-out real | 0.0144 |
| uniform vs held-out real | 0.1064 |

Caption: "Generated marginals lie close to the real-vs-real floor; the uniform baseline is an order
of magnitude further." Source: `results/dna/base_model/table16_tv.json`, `scripts/dna/table16_tv.py`.
