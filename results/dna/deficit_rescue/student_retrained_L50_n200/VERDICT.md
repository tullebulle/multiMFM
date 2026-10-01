# Does retraining the dMFM student help? No.

**Question.** The shipped `dmfm_4step/L50` artifact is `epoch=29-step=15000.ckpt`, selected by step
count. If the student were simply undertrained, or selected at the wrong step, the DNA guidance
numbers should improve with a longer run and a principled selection rule. That would be the honest
route to better numbers, so it was worth testing before concluding anything about the method.

**What was run.** `scripts/dna/rescue/train_dmfm4_longer.sbatch` (Slurm 49471652, COMPLETED):
60,000 steps, ESD, with checkpoints kept by **validation loss** rather than by step count ->
`outputs/dna/deficit_rescue/rescue_L50_dmfm_4step_esd_upto025_by_val_loss/`. Then
`eval_retrained_student.sbatch` (Slurm 49519367, both array tasks COMPLETED) sampled n = 200 paired
guided/unguided sequences per target at the operating point, scoring every config against the
parent-disjoint oracle.

**Answer: a tie, and the shipped checkpoint stays.** Paired on the *same* 200 samples, at the same
1 step x MC 32 operating point, retrained-minus-shipped oracle MAE is

| target | paired difference | SEM |
|---|---|---|
| $y^\star = +1$ | **+0.0020** | 0.0133 (0.15 SEM) |
| $y^\star = -1$ | **+0.0009** | 0.0114 (0.07 SEM) |

Positive means the retrained student is *worse*, but both differences are well inside noise. Four
times the training and a better selection rule move nothing.

**Consequence.** Nothing in the paper changes. The DNA numbers are not limited by student training,
which rules out the most obvious alternative explanation for them, and the shipped checkpoint is
kept so every reported DNA result continues to come from one artifact.

Full per-config means (n = 200, oracle MAE, all at `@norm`):

| config | $y^\star{=}{+}1$ | $y^\star{=}{-}1$ |
|---|---|---|
| `diag_fm1@mc=32` (shipped, the operating point) | 0.1614 | 0.1267 |
| `dmfm4_fm1@mc=32` (retrained) | 0.1634 | 0.1275 |
| `dmfm4_fm1` (MC 8) | 0.1714 | 0.1352 |
| `dmfm4_fm2` | 0.1627 | 0.1581 |
| `dmfm4_fm4` | 0.1826 | 0.1660 |
| `glass_euler2` | 0.1467 | 0.1222 |
| `glass_euler4` | 0.1476 | 0.1239 |

The n = 200 absolute values sit above the n = 1,000 headline (0.153 / 0.117) because they are the
first 200 of the same stream, not a different configuration; only the paired column above is a
like-for-like comparison.
