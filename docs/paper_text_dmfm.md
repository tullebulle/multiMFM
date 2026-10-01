# Paper text for the dMFM DNA reruns

Status 2026-09-29. Owner: the paper-text agent of the 2026-09-29 reproducibility round.
**This file is a proposal, not an edit.** `iclr2027_conferencemod.tex` is deliberately untouched:
the authors are mid-submission and must control what lands in the manuscript. Everything below is
written so that an author can apply it by search-and-replace on the quoted anchor.

## What this file is for

REPRODUCIBILITY_PLAN.md §7 item 8 records the authors' decision: **do not relabel the paper to say
GLASS -- rerun every DNA demo with dMFM.** That decision fixes the *numbers*. It does not fix the
*prose*, and in one place the prose contradicts itself. This file is the prose side of that
decision:

1. the App. F.2 / App. F.6 / Figure 5 contradiction, resolved in the direction the reruns make true;
2. the configuration that §3.1 and App. F.4 state nowhere -- which student, how many composed jumps,
   and which gradient-scale rule -- now mandatory per §7 item 8's two conventions;
3. an honest paragraph for the discretisation-error caveat of §7.9, stated as what was measured;
4. App. B.1's description of Tables 4-6;
5. a sweep of every other place the DNA experiments are described as dMFM, as GLASS, or with a
   posterior / step-count / strength claim, each with a verdict.

**Count: 16 proposed edits** (E1-E16), of which 7 are pure wording with no number attached, 6 remove
or replace an existing `[TODO rerun]` marker, and 3 re-open something a previous round closed. A
further **12 DNA locations were checked and need no change**; they are in
[the sweep](#sweep-every-dna-location-checked). E16 is a discrepancy this round found that no
existing audit records.

No number that has not landed is written anywhere below. Placeholders are in `[[DOUBLE_BRACKETS]]`
and every one of them is listed in [the placeholder table](#placeholder-table) with the results file
that fills it. The tex's own marker convention is `\textbf{[TODO rerun]}`; where an edit replaces an
existing marker I keep that convention and put the `[[KEY]]` beside it so the two indexes line up.

## Evidence base

| evidence | what it establishes |
|---|---|
| `docs/REPRODUCIBILITY_PLAN.md` §7 item 8 (incl. "Resolution chosen by the authors") and §7.9 | the decision, the two mandatory conventions, the per-demo status, the carry-over caveat |
| `docs/dmfm_glass_parity.md` §§1-3b, 6 | the two defects, the $n=32/64$ dMFM-vs-GLASS comparison (**superseded end-to-end**, see the next row), the gradient-norm and clip-binding statistics, the `glass_euler32` result, the reference gradient norm, the guidance-strength sweep |
| `results/dna/c0_guidance/table1_dmfm_PRELIMINARY_partial.json` (read 2026-09-29 22:54) | **the governing dMFM-vs-GLASS measurement.** $y^\star=+1$ dMFM arm final at $n=1000$: unguided 1.174, guided 0.1933 [0.1836, 0.2036], paired improvement 0.980, 97.5% of pairs improved. Matched GLASS-4 arm at $n=800$ under the identical scale-free rule: guided 0.1517 [0.1436, 0.1597], 98.75% improved. Per-sample paired difference on the 800 shared `sample_idx`: dMFM $-$ GLASS $= +0.0403$, SEM 0.0072, CI [0.0260, 0.0549] -- about 5.6 SEM, with dMFM better on only 44.5% of pairs. $y^\star=-1$ dMFM is $n=200$ (guided 0.1844) with no matched GLASS arm yet. The withdrawn mis-wired configuration is 0.6103/0.4081 at $n=1000$ |
| `docs/paper_changes.md` §1.9-1.11, §1.14, §1.23, §3.1, §3.2, §3.18, §4.7, §4.7b | what is already marked in the tex, what a previous round closed, and the reference-noise finding |
| `outputs/dna/table1_dmfm4_norm_n1000/*/shards/*/args.json` and `outputs/dna/table1_glass_norm_n1000/...` | the in-flight Table 1 reruns' actual configuration (read 2026-09-29): 4-step ESD student, `nfe_value 4`, `grad_normalize True`, `guidance_frac 8`, `coeff_cap 10`, `grad_clip 10`, `mc 8`, window `[0.50, 0.95]`, `nfe_sample 64`, `t_max 0.95`, matched on both arms |
| `src/dmfm/api.py:383-449` (`guided_sample`) | the exact guidance step, and that `grad_normalize` rescales to *exactly* `grad_clip` |
| `src/dmfm/scalefree/run_shard.py` (`--student dmfm`, `--dmfm_steps 1`, `--t_eval 0.50`) and `scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch` | the Fig 5 / Tables 14-15 rerun uses the **diagonal** student as **one** map, not four |
| `scripts/dna/table13_reward_attainment.sbatch`, `scripts/dna/table17_posterior_diversity.sbatch`, `scripts/dna/table18_dmfm4_gradient_mc.sbatch` | the step count and student behind Tables 13, 17 and 18 |
| `scripts/dna/regen/fig7_guidance_sweep.sbatch`, `scripts/dna/regen/fig8_guidance_trace.sbatch`, `src/dmfm/experiments/sample_c0_guidance.py:246`, `src/dmfm/regen/guidance_trace.py:92` | the in-flight Fig 7 and Fig 8 regenerations call **GLASS**, not dMFM |
| `scripts/dna/tables4_5_c0_diversity.sh` defaults + `results/dna/c0_guidance/table1_n1000_summary.json` `_sampler` | the Tables 4-5 "final" numbers were computed on the **GLASS** n=1000 sets |
| `src/dmfm/experiments/analyze_c0_diversity.py:403-427` (`summarize_reference_baseline`) | Table 4 is computed from real reference windows only, so it is sampler-independent |

### Revision 2026-09-29: the dMFM-vs-GLASS comparison, settled against dMFM

**This supersedes the "tie" reading that an earlier draft of this file carried.** Read this before
writing any comparative sentence about the DNA experiments.

The Table 1 rerun has measured dMFM against a matched GLASS arm on **800 shared `sample_idx`** under
the identical scale-free rule, both arms sharing initial noise per id
(`results/dna/c0_guidance/table1_dmfm_PRELIMINARY_partial.json`):

| $y^\star=+1$, guided oracle MAE | value | $n$ |
|---|---|---|
| dMFM, 4-jump ESD student at 4 jumps | **0.1933** [0.1836, 0.2036], paired improvement 0.980, 97.5% of pairs improved | 1000 (**final**) |
| matched GLASS-4, identical rule | **0.1517** [0.1436, 0.1597], paired improvement 1.031, 98.75% improved | 800 |
| per-sample paired difference, dMFM $-$ GLASS | **$+0.0403$**, SEM 0.0072, CI [0.0260, 0.0549] $\approx$ 5.6 SEM; dMFM better on 44.5% of pairs | 800 shared ids |
| withdrawn mis-wired configuration, for scale | 0.6103 | 1000 |

$y^\star=-1$ is $n=200$ on the dMFM side (guided 0.1844) with **no matched GLASS arm yet**, so that
target's comparison is unmeasured.

This is compatible with the $n=32/64$ parity runs -- those were simply too noisy to resolve a
$0.04$ gap -- and the tighter measurement governs. Three consequences for the prose:

1. **The supportable claims are:** dMFM works as a demonstration (oracle MAE 1.174 $\to$ 0.193,
   97.5% of matched pairs improved, evaluated on a parent-disjoint oracle), and where dMFM and GLASS
   are compared, **dMFM is modestly but clearly behind**. Never parity, never superiority. This is
   claim 4 of [the caveats list](#claims-that-cannot-be-made-even-after-the-reruns).
2. **Table 1's cells move, and must not be filled from the published values.** The published
   $y^\star=+1$ guided MAE of 0.157 was a GLASS number; the dMFM value is $\approx 0.193$ and the
   paired improvement falls with it (1.013 $\to$ 0.980). Anyone substituting must take every cell from
   `table1_dmfm_n1000_summary.json` when the collector finishes, not from the PDF and not from
   `table1_n1000_summary.json`.
3. **One §3.1 sentence changes character**, although its wording survives: "The guided errors of
   [TODO] and [TODO] are comparable to the oracle's own MAE of 0.16 against measured C0 on the
   held-out parents." Under GLASS the guided error sat *below* the oracle's own error (0.157 vs 0.16);
   under dMFM it sits *above* it (0.193 vs 0.16). "Comparable" remains fair and the following
   sentence's argument -- that the experiment shows guidance reaches the target region rather than
   attaining it to within that figure -- is *strengthened*, not weakened. No edit is needed, but the
   "to within" figure must be the dMFM one. Flagged here because the sentence is easy to substitute
   mechanically and get backwards.

### One thing the authors must know before applying any of this

Three of the in-flight reruns do **not** produce dMFM output, so three of the edits below are written
as markers rather than as final text:

* **Figure 7** (`scripts/dna/regen/fig7_guidance_sweep.sbatch`, job 49298146) calls
  `dmfm.experiments.sample_c0_guidance`, whose posterior is `glass_integrate_diff`
  (`src/dmfm/experiments/sample_c0_guidance.py:246`). No student is loaded.
* **Figure 8** (`fig8_guidance_trace.sbatch`, job 49298115) calls `dmfm.regen.guidance_trace`, which
  builds `api.glass_posterior_fn(base, n_steps=nfe_value)` (`src/dmfm/regen/guidance_trace.py:92`).
* **Tables 5 and 6 and the App. B.1 diversity paragraph** were recomputed on
  `results/dna/c0_guidance/n1000/`, which `table1_n1000_summary.json` records as
  "GLASS-posterior guidance on the parent-disjoint L=50 base flow". `docs/paper_changes.md` §1.14
  calls those numbers "final"; against §7 item 8 they are not, because item 8 requires the
  generated-sequence sets behind Tables 4-6 to come from dMFM samples.

§7 item 8's per-demo status lists all three as owing a dMFM rerun. Until they are rerun, App. F.4's
"tilting the dMFM in Figure 7", Figure 8's "guidance using dMFM" and App. B.1's "C0-steered yeast
DNA dMFM samples" stay false. E14, E15 and E10/E11 are written accordingly.

---

# The edits

## E1. §3.1 -- say that each experiment uses the step-matched student

**Location.** §3.1 (`sec:dna_experiments`), first paragraph, last two sentences.

**Anchor / current text.**

> While the dMFM works as a one-shot sampler, empirically we find using 2-4 steps gives increased performance. Architectural details are given in Appendix~\ref{dMFM_archi}.

**Proposed replacement.**

> While the dMFM works as a one-shot sampler, empirically we find using 2--4 steps gives increased
> performance. Each experiment uses the student distilled for the number of composed jumps it
> applies; Appendix~\ref{molecule train results} records which student and how many jumps each DNA
> result uses, together with the guidance settings. Architectural details are given in
> Appendix~\ref{dMFM_archi}.

**Justification.** §7 item 8 convention 1 makes the student-to-step-count pairing a reportable
convention, and `docs/dmfm_glass_parity.md` §1 measures the cost of getting it wrong (cosine 0.430
for the matched pairing against 0.171 for the mismatched one). The sentence is otherwise already
true of the reruns and is left alone.

*Note, not an edit:* "2--4 steps gives increased performance" **is** supported once the artifacts are
step-matched -- `docs/dmfm_glass_parity.md` §1 gives cosine 0.430 (t=0.5) for the four-jump student
at four jumps against 0.261 for the diagonal student as one map, with two jumps best at 0.591. It is
*not* established end-to-end: §2 of the same file gives guided oracle MAE 0.228/0.153 for the
four-jump student under the scale-free rule against 0.200/0.180 for the one-jump diagonal student,
which is a wash with the sign differing by target -- and not a clean comparison either, because the
one-jump configurations were only run under the stock clip-down rule. No claim in the paper rests on
the end-to-end reading, so nothing more is needed here. (Those two rows are $n=64$; the $n=800$
measurement of claim 4 covers only the four-jump configuration, so it does not sharpen this
comparison.)

## E2. §3.1 -- the clustering sentence has to be re-opened

**Location.** §3.1, final paragraph.

**Anchor / current text.**

> The separate diversity analysis shows slightly more local clustering for the negative target despite complete sample uniqueness (Appendix~\ref{app:dna-diversity}).

**Proposed replacement.**

> The separate diversity analysis shows \textbf{[TODO rerun: [[B1_CLUSTERING_DIRECTION]] ---
> the sentence rests on Table~\ref{tab:yeast_generated_diversity}, whose generated sets are the
> GLASS-guided $n=1000$ runs and not dMFM output; see Appendix~\ref{app:dna-diversity}]} despite
> complete sample uniqueness (Appendix~\ref{app:dna-diversity}).

**Justification.** `docs/paper_changes.md` §1.14 closed this sentence against
`results/dna/diversity/regen/table5_generated.csv`, but that recompute ran on
`results/dna/c0_guidance/n1000/`, which `table1_n1000_summary.json` `_sampler` records as GLASS. Per
§7 item 8 the sets behind Tables 4-6 must be dMFM, so the direction is not yet established for the
runs the paper will describe. The parent-disjoint switch already shrank the effect four-fold
(0.9507x to 0.9872x of held-out, `paper_changes.md` §3.2), so the direction is exactly the kind of
claim a sampler change can flip.

## E3. §3.3 -- name the estimator in the sentence that introduces Figure 5

**Location.** §3.3, "Scale-free Monte Carlo convergence" paragraph.

**Anchor / current text.**

> Figure~\ref{fig:dna-scaling} compares finite-$N$ MC samples from dMFM evaluation with an independent GLASS reference using 2,048 samples, over 32 conditioning states, obtained by integrating the base flow to $t=0.5$, and eight noise pools per state at each length.

**Proposed replacement.**

> Figure~\ref{fig:dna-scaling} compares finite-$N$ MC samples from the dMFM posterior, applied as a
> single flow-map evaluation of the per-length diagonal student, with an independent GLASS reference
> using 2,048 samples, over 32 conditioning states, obtained by integrating the base flow to
> $t=0.5$, and eight noise pools per state at each length.

**Justification.** `src/dmfm/scalefree/run_shard.py` defaults to `--student dmfm --dmfm_steps 1`
and `scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch` keeps that default, so the rerun is the
one-jump diagonal student -- the *other* pairing from Table 1's. With two different dMFM
configurations in one section, "the dMFM sampler" no longer identifies which. This is convention 1
applied to this demo: one jump resolves to the diagonal student
(`paths.dmfm_student_kind(1) == "dmfm"`).

## E4. §3.3 -- the $N=8$ operating point, stated as what it is

**Location.** §3.3, last paragraph.

**Anchor / current text.**

> We found $N=8$ is an effective end-to-end operating point for DNA experiments. At $N=8$, relative errors are \textbf{[TODO rerun]}--\textbf{[TODO rerun]} and cosine similarities are \textbf{[TODO rerun]}--\textbf{[TODO rerun]} \textbf{[TODO rerun: these ranges, and the claim that $N=8$ is an effective operating point, are awaiting the dMFM rerun of Figure~\ref{fig:dna-scaling} and Tables~\ref{tab:scale-free-summary}--\ref{tab:scale-free-full}. On a controlled 4-state slice the dMFM estimates are comparable to the GLASS-only measurement at matched $N$, so the substituted ranges are expected to be close to those printed; no claim either way until the full arrays land.]}. Larger parallel budgets provide higher fidelity.

**Proposed replacement.**

> $N=8$ is the posterior budget used in every end-to-end DNA experiment reported here. At $N=8$,
> relative errors are \textbf{[[SF_RELL1_N8_MIN]]}--\textbf{[[SF_RELL1_N8_MAX]]} and cosine
> similarities are \textbf{[[SF_COS_N8_MIN]]}--\textbf{[[SF_COS_N8_MAX]]}. Larger parallel budgets
> give a more accurate gradient, monotonically in $N$ over the measured range; we do not claim that
> they give better steering, and Appendix~\ref{molecule train results} reports a measurement in
> which they do not.

**Justification.** Two changes, both required by measurement. "Effective operating point" implies
$N=8$ was selected on evidence; nothing in `results/dna/` selects it, so the defensible statement is
that it is the budget used. And "Larger parallel budgets provide higher fidelity" reads as a claim
about steering, which `docs/dmfm_glass_parity.md` §2 contradicts for the posterior axis
(`glass_euler32` guides not at all); splitting fidelity from steering keeps the true half and drops
the implication. The marker's hedge about the 4-state slice goes away because the full arrays supply
the four numbers.

## E5. §4 Discussion -- one sentence on the operating point (do not bury it)

**Location.** §4 Discussion, after "We derive the dMFM as one-shot posterior sampler in the discrete
domain, and showcase its scientific potential through a DNA design application."

**Anchor / current text.** (insertion, nothing replaced)

> We derive the dMFM as one-shot posterior sampler in the discrete domain, and showcase its scientific potential through a DNA design application.

**Proposed insertion, immediately after that sentence.**

> One limitation of that application is worth stating plainly. The DNA guidance signal is carried by
> a deliberately coarse posterior: at $L=50$ the value gradient of a well-integrated posterior is
> about two orders of magnitude smaller than the four-step estimate we steer with, and refining the
> GLASS posterior from four Euler steps to thirty-two removes the steering effect entirely rather
> than improving it (Appendix~\ref{molecule train results}). The reported operating point is
> therefore $N=8$ posterior samples at four posterior steps, and the guidance constants are
> calibrated against that discretisation error; they are not an asymptotically accurate
> value-gradient estimate and do not transport between posterior samplers unchanged.

**Justification.** §7 item 8's closing paragraph: "Carry-over caveat that no rerun fixes: the
guidance signal rides on the 4-step posterior's discretisation error (§7.9), so the operating point
must be reported as N=8 at 4 steps rather than as an asymptotically accurate value-gradient
estimate." Two sentences in the main text with a pointer to the measurement is the minimum that is
not burying it; the numbers themselves are in E12 so the Discussion does not grow a table.

## E6. Figure 5 caption -- configuration in, marker out

**Location.** caption of `fig:dna-scaling`.

**Anchor / current text.**

> Finite-sample dMFM gradient estimation from 50 to 400 bp. Top: pooled relative $\ell_1$ error. Bottom: mean cosine similarity. Each estimate is compared with an independent GLASS-2048 reference. Conditioning states are obtained by integrating the base flow to $t=0.5$. \textbf{[TODO rerun: both panels are awaiting the dMFM rerun --- the finite-$N$ estimates recorded for the submitted version were produced with GLASS posterior samples on the base discrete flow, not with the dMFM.]}

**Proposed replacement.**

> Finite-sample dMFM gradient estimation from 50 to 400 bp. The finite-$N$ estimate is the dMFM
> posterior applied as a single flow-map evaluation of the per-length diagonal student. Top: pooled
> relative $\ell_1$ error. Bottom: mean cosine similarity. Each estimate is compared with an
> independent GLASS-2048 reference. Conditioning states are obtained by integrating the base flow to
> $t=0.5$. Because that reference is itself a finite-sample estimate, the comparison measures
> whether substituting the dMFM for GLASS at matched $N$ degrades the gradient estimate, and not the
> dMFM's approximation error; see Appendix~\ref{app:scale-free-protocol}.

**Justification.** This is one of the two places `docs/paper_changes.md` §1.23(b) identifies as
overclaiming against App. F.6's correct sentence. The rerun makes the "dMFM" label true; it does
**not** make "dMFM approximation error" measurable, because the reference stays a GLASS-2048 finite
sample (`paper_changes.md` §3.18: over a 256-fold increase in $N$ the published GLASS-versus-GLASS
relative $\ell_1$ falls only 3.66x against the ~16x of independent sampling). So the caption gains
the limitation rather than losing the marker for nothing.

## E7. App. F.2 -- the two sentences that carry the contradiction

**Location.** App. F.2 (`app:scale-free-protocol`), opening sentence and the conditioning-state
sentence.

**Anchor / current text (opening).**

> Let $g^\star$ be an independent GLASS estimate using 2,048 terminal-noise samples, and let $\widehat g_N$ be an estimate using $N$ independent terminal samples from the dMFM sampler.

**Proposed replacement.**

> Let $g^\star$ be an independent GLASS estimate using 2,048 terminal-noise samples, and let
> $\widehat g_N$ be an estimate using $N$ independent terminal samples from the dMFM posterior,
> applied as a single flow-map evaluation of the per-length diagonal student -- the same posterior
> sampler as Tables~\ref{tab:exact-dna-reward-improvement} and
> \ref{tab:dna_posterior_diversity_scaling}, and not the four-jump sampler of
> Table~\ref{tab:raw-gradient-mae}.

**Anchor / current text (second part, end of the subsection).**

> The pooled normalization avoids unstable ratios for individual states whose reference gradient is nearly zero.

**Proposed addition, as a new sentence immediately after it.**

> Because $g^\star$ is itself a finite-sample estimate, this comparison does not separate the
> posterior sampler's approximation error from the reference's own Monte Carlo error: over the
> $256$-fold range of $N$ the reference's error falls by only $3.66\times$, against the $16\times$
> expected of independent sampling, so the apparent floor is largely the reference's own noise. What
> the comparison measures is how far the dMFM estimate sits from the GLASS estimate at matched $N$:
> \textbf{[[SF_RELL1_RATIO_RANGE]]} on pooled relative $\ell_1$ and \textbf{[[SF_COS_ABS_GAP]]} on
> cosine. Isolating the dMFM approximation error would require a reference converged in $N$, or
> common random numbers shared between the two samplers.

**Justification.** This is the resolution of the F.2 / F.6 contradiction on the F.2 side. The
template is `docs/paper_changes.md` §4.7b, with the three derivable numbers (256-fold, 3.66x, ~16x)
filled from `paper_changes.md` §3.18, which derives them from the *published* data and so needs no
rerun; the two comparison ranges are placeholders because §3.18 is explicit that they must come from
the 32-state arrays and not from the 4-state slice, whose numbers are marked not-for-paper.

**Revised 2026-09-29 after the $n=800$ paired measurement.** My first draft of this passage wrote
"and it does not: the two agree within ...", i.e. it asserted gradient-level parity before the arrays
had landed. That is exactly the error the $n=800$ end-to-end result punishes: on the guided-sampling
axis the dMFM arm is now clearly behind GLASS (claim 4 below), so no parity may be asserted on the
gradient axis either until the 32-state arrays are in. The sentence is now a neutral statement of what
is measured, and whoever substitutes the two ranges must write the direction the numbers show. The
same applies to E6's and E9's phrasing of the question, which were already neutral and are
unchanged.

## E8. Tables 14 and 15 captions -- the same marker, replaced twice

**Location.** captions of `tab:scale-free-summary` and `tab:scale-free-full`. The marker text is
identical in both; apply the same replacement to each.

**Anchor / current text (in both captions).**

> \textbf{[TODO rerun: every entry in this table is awaiting the dMFM rerun --- the finite-$N$ estimates recorded for the submitted version were produced with GLASS posterior samples on the base discrete flow, not with the dMFM.]}

**Proposed replacement (both captions).**

> The finite-$N$ estimate is the dMFM posterior applied as a single flow-map evaluation of the
> per-length diagonal student; the reference is an independent finite-sample GLASS estimate, so these
> entries are not a measure of dMFM approximation error (Appendix~\ref{app:scale-free-protocol}).
> \textbf{[TODO rerun: [[SF_TABLES_14_15_CELLS]] --- 18 cells in
> Table~\ref{tab:scale-free-summary} and 108 in Table~\ref{tab:scale-free-full}, from
> \texttt{results/dna/scale\_free\_mc/dmfm/}.]}

**Justification.** `docs/paper_changes.md` §1.10 deliberately did not placehold the 126 individual
cells, on the grounds that the caption markers are unmissable; that choice is kept. The added first
sentence is what makes the tables consistent with App. F.6 after the rerun, and the marker now names
the file and the cell count so the substitution is mechanical.

## E9. App. F.6 -- the corrected passage (the contradiction, resolved)

**Location.** App. F.6 (`app:raw-gradient-mae`), first sentence and the last sentence of the second
paragraph.

**Anchor / current text (first sentence).**

> As a separate diagnostic, we compare gradients obtained using the selected four-map dMFM posterior sampler against a GLASS reference using 128 posterior samples.

**Proposed replacement.**

> As a separate diagnostic, we compare gradients obtained using the four-jump dMFM student -- the ESD
> student distilled and checkpoint-selected for a $0.25$ jump, applied as four composed flow-map
> steps -- against a GLASS reference using 128 posterior samples.

**Anchor / current text (the sentence that carried the concession, plus its marker).**

> Because per-coordinate MAE depends on gradient scale, this table is not used to compare scaling across sequence lengths; the scale-free analysis in Table~\ref{tab:scale-free-full} compares GLASS Monte Carlo error across lengths, without measuring dMFM approximation error. \textbf{[TODO rerun: this sentence is the one that states the limitation correctly and should be kept. The reference in Tables~\ref{tab:scale-free-summary}--\ref{tab:scale-free-full} is itself a finite-sample GLASS estimate, so that comparison cannot isolate dMFM approximation error however the finite-$N$ side is sampled; it is Appendix~\ref{app:scale-free-protocol} and the caption of Figure~\ref{fig:dna-scaling} that overclaim, and they are the ones to reconcile with it once the dMFM arrays land.]}

**Proposed replacement.**

> Because per-coordinate MAE depends on gradient scale, this table is not used to compare scaling
> across sequence lengths. Neither does the scale-free analysis of
> Tables~\ref{tab:scale-free-summary}--\ref{tab:scale-free-full} isolate the dMFM approximation
> error, although its finite-$N$ side is a dMFM estimate: its reference is itself a finite-sample
> GLASS estimate, so the two error sources are not separable there
> (Appendix~\ref{app:scale-free-protocol}). The two diagnostics answer different questions. That one
> asks whether substituting the dMFM for GLASS at matched $N$ degrades the gradient estimate, across
> lengths. This one asks how far the four-jump student's gradient sits from a more heavily integrated
> $128$-sample reference at a single length, and it mixes finite-sample and approximation error in
> doing so.

**Justification.** This is the item the task calls the F.2 / F.6 contradiction. The submitted
sentence is correct about the runs as submitted and *becomes wrong in the other direction* once the
finite-$N$ side is dMFM, because it then says the scale-free tables measure GLASS Monte Carlo error
when they no longer do. The limitation it states survives the rerun for a different reason -- the
*reference* is the finite-sample GLASS estimate, not the estimator -- and the replacement says that,
which is exactly the reframing `docs/paper_changes.md` §1.23(b) recorded. App. F.6's "32 held-out
states" is correct and untouched: `paper_changes.md` §1.23 verified that Table 18 comes from
`ablate_dmfm_one_step_gradient_mc.py`, whose probe states really are noised held-out test sequences.

## E10. App. B.1 opening -- which tables describe what

**Location.** App. B.1, first paragraph.

**Anchor / current text (the marker only; the sentences around it stay).**

> \textbf{[TODO rerun: Tables~\ref{tab:yeast_real_diversity_baselines} and \ref{tab:yeast_generated_diversity} are the parent-disjoint recomputation and match this description; Table~\ref{tab:yeast_evo2_plausibility} is still the June 2026 \texttt{split65k} version and is awaiting its Evo~2 scoring run on the same parent-disjoint samples.]}

**Proposed replacement.**

> \textbf{[TODO rerun: [[B1_TABLES_SCOPE]]. Table~\ref{tab:yeast_real_diversity_baselines} is
> computed from real reference windows only and is final --- it does not depend on the generator or
> the posterior sampler. Table~\ref{tab:yeast_generated_diversity} is the parent-disjoint
> recomputation and matches this appendix's description of the models and splits, but the generated
> sets it summarises are the GLASS-guided $n=1000$ runs
> (\texttt{results/dna/c0\_guidance/n1000}), not dMFM output, so its four generated rows must be
> recomputed on \texttt{results/dna/c0\_guidance/dmfm\_n1000}. Table~\ref{tab:yeast_evo2_plausibility}
> needs both: its four \texttt{generated} rows are still the June 2026 \texttt{split65k}
> random-window split and await the dMFM samples and a new Evo~2 scoring run; its four
> \texttt{reference} rows are real data and change only their labels.]}

**Justification.** This is task item 4. App. B.1 describes Tables 4-6 as "C0-steered yeast DNA dMFM
samples" from the parent-disjoint model; `docs/paper_changes.md` §1.14 delivered the parent-disjoint
half but ran `scripts/dna/tables4_5_c0_diversity.sh` on `results/dna/c0_guidance/n1000/`, whose
`_sampler` field is "GLASS-posterior guidance on the parent-disjoint L=50 base flow". §7 item 8
requires the generated-sequence sets behind Tables 4-6 to be dMFM. The split by table is from
`src/dmfm/experiments/analyze_c0_diversity.py:403-427`: `summarize_reference_baseline` takes only
reference windows, so Table 4 is genuinely final and re-opening it would be wrong.

## E11. Table 5 caption and the App. B.1 diversity paragraph -- re-opened

**Location.** caption of `tab:yeast_generated_diversity`, and the paragraph beginning "All generated
samples are exact-unique".

**Anchor / current text (caption, first clause).**

> Diversity statistics for generated C0-steered yeast dMFM samples, from the parent-disjoint L = 50 model.

**Proposed replacement.**

> Diversity statistics for generated C0-steered yeast dMFM samples, from the parent-disjoint L = 50
> model with the four-jump student. \textbf{[TODO rerun: [[TABLE5_GENERATED_ROWS]] --- the four
> generated rows are awaiting recomputation on the dMFM $n=1000$ sets
> (\texttt{results/dna/c0\_guidance/dmfm\_n1000}); the rows printed here were computed on the
> GLASS-guided sets of the same parent-disjoint models.]}

**Anchor / current text (the paragraph).**

> All generated samples are exact-unique and exact-novel relative to both \texttt{a\_train} and the held-out test parents. However, exact novelty alone is a weak diversity criterion, so the main comparison is against the held-out \texttt{test\_sample} pairwise and nearest-neighbor baselines. The \texttt{target\_c0 = +1} guided set preserves held-out-level global diversity: its mean pairwise Hamming fraction is $0.7348$, nearly identical to the held-out baseline of $0.7381$ and corresponding to $0.9956\times$ the held-out value. Its nearest-generated distance is lower than held-out, $0.5091$ versus $0.5219$, indicating mild local clustering, but the global pairwise diversity remains effectively held-out-level. The \texttt{target\_c0 = -1} guided set clusters slightly more, with pairwise diversity $0.7287$ or $0.9872\times$ the held-out baseline and nearest-generated distance $0.4960$ or $0.9504\times$ the held-out baseline. Both guided sets therefore sit below the held-out baseline on the nearest-neighbour statistic and the negative target sits further below, but the gap is small: a $1.3\%$ deficit in pairwise diversity and a $5.0\%$ deficit in nearest-generated distance.

**Proposed replacement.** Keep the two structural sentences and placehold the nine numbers; the
qualitative shape of the conclusion is not safe to assume, so it is placeheld too.

> All generated samples are exact-unique and exact-novel relative to both \texttt{a\_train} and the
> held-out test parents. However, exact novelty alone is a weak diversity criterion, so the main
> comparison is against the held-out \texttt{test\_sample} pairwise and nearest-neighbor baselines.
> The \texttt{target\_c0 = +1} guided set has mean pairwise Hamming fraction
> \textbf{[[B1_PW_P1]]} against the held-out baseline of $0.7381$, i.e.
> \textbf{[[B1_PW_RATIO_P1]]} of the held-out value, and nearest-generated distance
> \textbf{[[B1_NG_P1]]} against $0.5219$. The \texttt{target\_c0 = -1} guided set has pairwise
> diversity \textbf{[[B1_PW_M1]]} or \textbf{[[B1_PW_RATIO_M1]]} of the held-out baseline and
> nearest-generated distance \textbf{[[B1_NG_M1]]} or \textbf{[[B1_NG_RATIO_M1]]}.
> \textbf{[TODO rerun: [[B1_DIVERSITY_CONCLUSION]] --- state the direction and the magnitude of the
> deficit from the dMFM numbers. Do not carry over the GLASS reading: switching from the
> \texttt{split65k} to the parent-disjoint models already shrank the negative target's deficit from
> $4.9\%$ to $1.3\%$ (pairwise) and from $19.7\%$ to $5.0\%$ (nearest-generated), so a sampler change
> can plausibly move or reverse it.]}

**Justification.** Same as E10. The two held-out reference figures (0.7381, 0.5219) are kept as
numbers because they come from Table 4, which is sampler-independent. `docs/paper_changes.md` §3.2
supplies the split65k-to-parent-disjoint shrinkage that justifies refusing to assume the direction.

## E12. App. F.4 -- new paragraph: the guidance configuration

**Location.** App. F.4 (`molecule train results`), immediately after the sentence ending "so that
$f_{\text{cyc}}$ is well-defined over a superposition of discrete states." and before the
search-baseline paragraph.

**Anchor / current text.** (insertion, nothing replaced)

> To make it differentiable with respect to probabilities (a point on the simplex), we removed the initial one first one-hot encoded layer, so that $f_{\text{cyc}}$ is well-defined over a superposition of discrete states.

**Proposed insertion.**

> \paragraph{Posterior sampler, step count and guidance settings.} The cyclizability results of
> Table~\ref{tab:dna-independent-oracle} and Figures~\ref{fig:histograms} and
> \ref{fig:guidance_demo} use the $L=50$ dMFM student distilled for four composed jumps -- the ESD
> student checkpoint-selected on its $0.25$-gap probe -- applied as four flow-map steps of
> $\Delta t = 0.25$, with $N=8$ posterior samples per guided step drawn from a per-sample noise pool
> frozen across the guided and unguided arms of each pair. We pair each student with the jump size it
> was distilled for. The one-jump results (Tables~\ref{tab:exact-dna-reward-improvement} and
> \ref{tab:dna_posterior_diversity_scaling} and Figure~\ref{fig:dna-scaling}) use the diagonal
> student as a single map; composing that student four times instead, which is not a configuration we
> report, degrades the value-gradient direction by a factor of $2.5$--$3$ in cosine similarity
> against a well-integrated reference, and forcing the four-jump student to eight jumps degrades it
> as well. Guidance is applied on $t\in[0.5,0.95]$ of a $64$-step trajectory that stops at
> $t_{\max}=0.95$: at each such step the state receives
> $dt\,\min\!\left(8\sigma_t^2,\,10\right)\widehat g$ with $\sigma_t^2 = 2(1/t-1)$, and the reward is
> $r_{\mathrm{guide}}$ of \eqref{eqn:guidance reward dna} at the inverse temperature of E16 below,
> with the value estimated as a softmax over the $N$ look-aheads. Note that
> $\min(8\sigma_t^2,10)$ is $10$ to $16$ times the coefficient $\sigma_t^2/2$ of \eqref{eq:ode}
> across this window, exactly $16$ wherever the cap does not bind ($t\ge 0.615$); as for the
> molecular experiments, the reported guidance coefficient is a tuned constant times the stated one.
>
> \paragraph{The guidance step is scale-free in the gradient.} $\widehat g$ is the estimated value
> gradient rescaled per sample to norm exactly $10$, rather than only clipped down to $10$. This is
> necessary rather than cosmetic, and because it is a property of the comparison and not of the dMFM
> we apply it to the GLASS arm as well wherever the two are compared. Along the guided trajectory the
> dMFM posterior's value gradient is heavy-tailed where the four-step GLASS posterior's is not: mean
> norm $8$--$65$ with the clip binding on $30$--$66\%$ of guided steps, against $1.1$--$7$ binding on
> $2$--$17\%$. Under a clip-down rule the dMFM trajectory is therefore dominated by a minority of
> saturated steps, and that is not repaired by reducing the strength: holding the clip at $10$ and
> lowering the coefficient from $8\sigma_t^2$ through $4\sigma_t^2$ and $2\sigma_t^2$ to $\sigma_t^2$
> makes the guided oracle error monotonically worse, $0.73 \to 0.79 \to 0.90 \to 1.11$ at
> $y^\star=+1$, and lowering the clip from $10$ to $4$ to $2$ does the same. Rescaling removes the
> tail, so every guided step contributes an equal-magnitude nudge, and only then does more strength
> help. The rule is also what makes a reported strength meaningful across posterior samplers: under
> clip-down the effective strength depends on the gradient magnitude of whichever posterior is
> plugged in, so the same three constants silently re-tune the sampler when the posterior changes.

**Justification.** This is task item 2, and §7 item 8 states both conventions "must be stated in the
paper". Every number is read from `docs/dmfm_glass_parity.md` (§2's norm and clip-binding columns,
§6's monotone-worsening sweep, §1's cosine ratio) or from the in-flight runs' own `args.json`
(`outputs/dna/table1_dmfm4_norm_n1000/*/shards/*/args.json`: `grad_normalize True`,
`guidance_frac 8.0`, `coeff_cap 10.0`, `grad_clip 10.0`, `mc 8`, window `[0.50,0.95]`,
`nfe_sample 64`, `t_max 0.95`, `reward_sigma 0.15`, `reward_scale 0.5`), with the step written as
`src/dmfm/api.py:438-444` implements it. The $10$--$16\times$ ratio is arithmetic on
$\min(8\cdot 2(1/t-1),10)$ against $(1/t-1)$ over $t\in[0.5,0.95]$; it is the DNA counterpart of the
coefficient discrepancy already marked at \eqref{eq:multimodal steer} for the molecular runs
(`docs/paper_changes.md` §3.15), which records that finding as molecular-only.

## E13. App. F.4 -- new paragraph: the discretisation-error caveat

**Location.** App. F.4, immediately after E12's second paragraph.

**Anchor / current text.** (insertion, nothing replaced)

**Proposed insertion.**

> \paragraph{The guidance signal rides on the discretisation error of a coarse posterior.} The
> quantity we differentiate is not an approximation to the value gradient of a converged posterior,
> and the operating point should be read as $N=8$ Monte Carlo samples at four posterior steps rather
> than as an asymptotically accurate value-gradient estimate. Two measurements at $L=50$, on the
> setting of Table~\ref{tab:dna-independent-oracle}, establish this. First, the same value gradient
> computed from a well-integrated posterior -- GLASS with $25$ RK4 steps to $t=0.999$ -- has mean norm
> $0.15$ at $t=0.5$ and $0.17$ at $t=0.7$, where the four-step GLASS estimator has $17.5$ and $38.8$
> and the four-jump dMFM student $27.6$ and $8.7$: two orders of magnitude larger at every probe time
> we measured. Along the guided trajectory the same contrast holds, with mean norms of $1.1$--$3.1$
> and $7.6$--$15.5$ respectively. Second, refining the posterior does not sharpen the steering but removes it: replacing the
> four-step GLASS posterior by a $32$-step one leaves the guided oracle error at its unguided value,
> $1.1905$ against $1.1907$ at $y^\star=+1$ and $0.8098$ against $0.8098$ at $y^\star=-1$, with the
> clip binding on $0.4\%$ of guided steps. The guidance constants of the previous paragraph are
> therefore calibrated against the discretisation error of a deliberately coarse posterior. Three
> consequences: they cannot be transported between posterior samplers without the renormalisation
> above; "use a more accurate posterior sampler" is not, in this setup, a route to better steering,
> and we measured the opposite; and the strength sweep of Figure~\ref{fig:histograms} is a sweep of
> that discretisation error times the coefficient. This is a property of the experiment rather than
> of the dMFM -- both measurements above are on the GLASS posterior -- but it bounds what the
> cyclizability results can be read as showing.

**Justification.** This is task item 3 and §7.9's carry-over caveat. Every number is from
`docs/dmfm_glass_parity.md`, and deliberately only from the two configurations the paper reports
rather than from all nine that were tried: the reference norms 0.151/0.166 from the §1 table's
`glass rk4-25 @0.999` row; 17.5/38.8 from its `glass_euler4` row and 27.6/8.65 from its
`dmfm4_fm4` row; 1.1--3.1 and 7.6--15.5 from §2's `mean |g|` columns for `glass_euler4`/
`glass_euler4@norm` and `dmfm4_fm4@norm`; the `glass_euler32` rows of both §2 tables for
1.1905/1.1907, 0.8098/0.8098 and the 0.4% binding fraction. (§1's widest range across all nine
configurations is 0.055--125, which includes step counts no student was distilled for; quoting it
would overstate the case.) The paragraph deliberately says what was measured and no more: it does not claim the dMFM
posterior is or is not in the same position, because the 32-step experiment was run on GLASS.

## E14. App. F.4 -- the Figure 7 / Figure 8 pointer

**Location.** App. F.4, the sentence introducing Figures 6-8.

**Anchor / current text.**

> We verified the validity of our generative backbone by distribution comparison of both cyclyzability and correlation metrics, as in Figure \ref{fig:yeast_generation_eval}. One can further inspect the effect of tilting the dMFM in Figure \ref{fig:histograms} as a function of guidance strength, and an example of a guided sample in Figure \ref{fig:guidance_demo}.

**Proposed replacement.**

> We verified the validity of our generative backbone by distribution comparison of both
> cyclizability and correlation metrics, as in Figure~\ref{fig:yeast_generation_eval}. One can
> further inspect the effect of tilting the dMFM in Figure~\ref{fig:histograms} as a function of
> guidance strength, and an example of a guided sample in Figure~\ref{fig:guidance_demo}.
> \textbf{[TODO rerun: [[FIG78_SAMPLER]] --- the regeneration jobs queued for both figures
> (\texttt{scripts/dna/regen/fig7\_guidance\_sweep.sbatch},
> \texttt{scripts/dna/regen/fig8\_guidance\_trace.sbatch}) call the GLASS posterior on the base
> discrete flow, not the dMFM; both must be rerun with the four-jump student before this sentence and
> Figure~\ref{fig:guidance_demo}'s caption are true.]}

**Justification.** §7 item 8's per-demo status requires Fig 7 and Fig 8 to be regenerated from dMFM
samples. They are not being: `sample_c0_guidance.py:246` integrates `glass_integrate_diff` and
`regen/guidance_trace.py:92` builds `api.glass_posterior_fn`. Also fixes the two spellings of
"cyclyzability" in the same sentence.

## E15. Figure 8 caption -- the dMFM claim

**Location.** caption of `fig:guidance_demo`.

**Anchor / current text (first clause).**

> Demonstration of guidance using dMFM, for one guided and one unguided DNA sample from the parent-disjoint $L=50$ model.

**Proposed replacement.**

> Demonstration of guidance using dMFM, for one guided and one unguided DNA sample from the
> parent-disjoint $L=50$ model with the four-jump student.
> \textbf{[TODO rerun: [[FIG8_SAMPLER]] --- see Appendix~\ref{molecule train results}; the queued
> regeneration uses the GLASS posterior, and its guidance settings differ from
> Table~\ref{tab:dna-independent-oracle}'s ($N=128$, 12 posterior steps, coefficient
> $\min(\sigma_t^2,10)$, guidance from $t=0$, $y^\star=0.3$), which the caption should state once the
> figure is final.]}

**Justification.** Same evidence as E14, plus `scripts/dna/regen/fig8_guidance_trace.sbatch`'s
defaults (`--mc 128 --nfe_value 12 --guidance_frac 1.0 --guide_t_start 0.0 --target_c0 0.30`), which
are not the Table 1 settings; a reader would otherwise take the trace as illustrating the reported
configuration. The rest of this caption (`docs/paper_changes.md` §1.15: "must not claim the published
trajectory") is already correct and untouched.

## E16. The DNA reward as stated is not the reward that was run

**Location.** §3.1's terminal reward \eqref{eqn:guidance reward dna} and its restatement in App. F.4
(the equation beginning `r(x)= - |f_{\text{cyc}}(x) - y^\star |^2`). One clause, in App. F.4 only, so
the main text keeps the clean form.

**Anchor / current text (App. F.4, the sentence after that equation).**

> where $y^\star$ is a target cyclyzability and $f_{\text{cyc}}$ is a cyclyzability predictor (CNN) taking in a discrete sequence.

**Proposed replacement.**

> where $y^\star$ is a target cyclizability and $f_{\text{cyc}}$ is a cyclizability predictor (CNN)
> taking in a discrete sequence. In every run reported here this reward carries an inverse
> temperature: the quantity exponentiated inside the value function is
> $-\tfrac{1}{2}\lambda\,\sigma_r^{-2}\,(f_{\text{cyc}}(x)-y^\star)^2$ with $\lambda = 0.5$ and
> $\sigma_r = 0.15$, i.e. $-11.1\,(f_{\text{cyc}}(x)-y^\star)^2$. Because
> $V_t = \log\mathbb E[e^{r(X_1)}\mid X_t]$, that factor is not a reparameterisation: it sets how
> sharply the $N$ look-aheads are weighted, and so it changes the direction of the estimated gradient
> as well as its magnitude.

**Justification.** **New discrepancy, recorded in no existing audit.** `src/dmfm/api.py:315-330`
(`c0_log_reward_fn`) returns `reward_scale * (-0.5 ((f - y*)/reward_sigma)^2)`, its own docstring
gives `= -11.1 (f - y*)^2` at the defaults, and `reward_sigma 0.15 / reward_scale 0.5` are the values
in every DNA C0 `args.json` -- the published pilot, the GLASS $n=1000$ run, both in-flight reruns and
the Figure 7 sweep. \eqref{eqn:guidance reward dna} and the F.4 equation both state the unit-weight
form. A constant inside a log-expectation of an exponential is an inverse temperature, so unlike a
constant in front of $\nabla V$ it cannot be absorbed into the tuned guidance strength. This is
independent of the dMFM reruns -- it is equally true of the submitted numbers -- so it is a
disclosure rather than a number, and it belongs beside E12 because a reader reproducing the runs from
the paper alone would build a different value function.

---

# Placeholder table

Every placeholder introduced above, and the results file that fills it. **Nothing in this table may
be guessed.** Where a file does not exist yet the row says so; the path is the one the collection
script writes, read from the script.

| placeholder | in | filled from | exists today |
|---|---|---|---|
| `[[B1_CLUSTERING_DIRECTION]]` | E2, §3.1 | the dMFM recomputation of `table5_generated.csv` (see `[[TABLE5_GENERATED_ROWS]]`) | no |
| `[[SF_RELL1_N8_MIN]]`, `[[SF_RELL1_N8_MAX]]`, `[[SF_COS_N8_MIN]]`, `[[SF_COS_N8_MAX]]` | E4, §3.3 | `results/dna/scale_free_mc/dmfm/` (12 shards; `bash scripts/dna/scalefree/collect.sh`), the $N=8$ rows across the three rewards and four lengths | no -- jobs 49298891 (GPU) / 49297229 (CPU) |
| `[[SF_RELL1_RATIO_RANGE]]`, `[[SF_COS_ABS_GAP]]` | E7, App. F.2 | the same directory, as the dMFM-to-GLASS ratio and absolute cosine gap at matched $N$ over the 32-state arrays. **Must not** be taken from `results/dna/scale_free_mc/dmfm/preliminary/`, which is 4 states, one reward, one length and marked not-for-paper | no |
| `[[SF_TABLES_14_15_CELLS]]` | E8, Tables 14-15 | the same directory: 18 cells for Table 14, 108 for Table 15 | no |
| `[[B1_TABLES_SCOPE]]` | E10, App. B.1 | bookkeeping only; resolved by E11's and Table 6's rows landing | n/a |
| `[[TABLE5_GENERATED_ROWS]]` | E11, Table 5 | `scripts/dna/tables4_5_c0_diversity.sh results/dna/c0_guidance/dmfm_n1000/target_m1p0 results/dna/c0_guidance/dmfm_n1000/target_1p0`, written to a `results/dna/diversity/` sibling of `regen/`. **No such run is queued**; the existing `results/dna/diversity/regen/table5_generated.csv` is the GLASS recompute | no |
| `[[B1_PW_P1]]`, `[[B1_PW_RATIO_P1]]`, `[[B1_NG_P1]]`, `[[B1_PW_M1]]`, `[[B1_PW_RATIO_M1]]`, `[[B1_NG_M1]]`, `[[B1_NG_RATIO_M1]]`, `[[B1_DIVERSITY_CONCLUSION]]` | E11, App. B.1 | the same file | no |
| `[[FIG78_SAMPLER]]`, `[[FIG8_SAMPLER]]` | E14, E15 | a dMFM rerun of `scripts/dna/regen/fig7_guidance_sweep.sbatch` and `fig8_guidance_trace.sbatch`; neither script has a dMFM posterior path today | no |

**Placeholders this file does not introduce, because the tex already carries an equivalent marker
and the owner is recorded in `docs/paper_changes.md` §1.12.** Listed so the two indexes reconcile:
Table 1's ten cells and §3.1's five prose figures (`results/dna/c0_guidance/table1_dmfm_n1000_summary.json`,
jobs 49352242/49357778 for the dMFM arm and 49352244/49357782 for the matched GLASS arm -- the partial
`table1_dmfm_PRELIMINARY_partial.json` already has the $y^\star=+1$ dMFM arm final at $n=1000$, and
its own `_status` says not to cite it as Table 1; **do not** fill these cells from the PDF or from
`table1_n1000_summary.json`, which are GLASS); Table 6's
56 cells and App. B.1's six Evo 2 numbers (`results/dna/regen/`, job 49298971 -- and now also owing
the dMFM samples); Figure 7's two $\gamma$ values (`results/dna/regen/fig7/index.csv`); Figure 4's
`N_COND` / `N_OUT` / `N_REP` and its §3.3 claim (`results/dna/fig4/summary.json`).

---

# Sweep: every DNA location checked

Every place in `iclr2027_conferencemod.tex` that describes the DNA experiments as dMFM, as GLASS, or
that makes a posterior / step-count / guidance-strength claim. "Already correct" means true of the
dMFM reruns as written, and **nothing should be changed there**.

| location | claim | verdict |
|---|---|---|
| Abstract, "We demonstrate dMFM by designing DNA sequences with a target mechanical property." | dMFM | **already correct** after the Table 1 rerun. No edit. |
| Contribution bullet, "dMFM enables inference-time optimization of generated DNA sequences toward prescribed predicted cyclizability values." | dMFM | **already correct** after the Table 1 rerun. No edit. |
| §3 Experiments, "we train an unconditional DNA sequence generator and distill it into a dMFM" | dMFM | **already correct**. No edit. |
| §3.1, "While the dMFM works as a one-shot sampler, empirically we find using 2--4 steps gives increased performance." | step count | **needs the new text** (E1). The claim itself holds on the step-matched gradient diagnostic; what is missing is which student each result uses. |
| §3.1, "We evaluate dMFM for DNA design guided by predicted cyclizability." | dMFM | **already correct** after the rerun -- this is the sentence §7 item 8 says the rerun makes true as written. No edit. |
| §3.1, Table 1 body and the four oracle-MAE figures | numbers | **needs a number**; already marked, owner dna-core. No new edit -- but see [the 2026-09-29 revision](#revision-2026-09-29-the-dmfm-vs-glass-comparison-settled-against-dmfm): every cell must come from the dMFM artifact, not from the published GLASS values (0.157 becomes $\approx$ 0.193 at $y^\star=+1$). |
| §3.1, "The guided errors of [TODO] and [TODO] are comparable to the oracle's own MAE of 0.16" and the "to within [TODO]" figure | numbers | **already correct as written**, and the argument is strengthened rather than weakened -- but the guided error moves from below the oracle's own 0.16 to above it, so the substitution must not be made from the published values. See the 2026-09-29 revision. No edit. |
| §3.1, "For each target, we generate 1,000 guided--unguided pairs of 50-bp sequences, using identical initial noise within each pair." | protocol | **already correct**: the queued reruns are `--paired`, `n_samples 1000`, `seed 0`. No edit. |
| §3.1, "Across lengths $L\in\{50,100,200,400\}$, guidance improves around 95\% of matched guided-unguided pairs." | dMFM, strength | **already correct**, and unaffected: Table 13's runs loaded the diagonal student as one map (`scripts/dna/table13_reward_attainment.sbatch`). No edit. |
| §3.1, "The separate diversity analysis shows slightly more local clustering for the negative target" | number | **needs a number** (E2); a previous round closed this against GLASS-derived numbers. |
| §3.3 Figure 4 caption and the Posterior Recovery / Steering paragraphs | dMFM NFE=1, baselines | **already correct as far as this task goes**, and separately owned: Figure 4's left panel uses a synthetic reward and a teacher-SDE reference, its dMFM series is `nfe_value 1` on the diagonal student, and its open markers belong to dna-fig4. Its guidance rule is *not* the scale-free one; see the caveats list. |
| §3.3, "Posterior pairwise Hamming distances remain close to the GLASS reference from 50 to 400 bp" | dMFM vs GLASS | **already correct**: Table 17 loaded the diagonal student as a one-step flow map against a GLASS-100 reference. No edit. |
| §3.3, "The raw dMFM-versus-GLASS comparison is reported separately in Appendix~\ref{app:raw-gradient-mae}." | dMFM vs GLASS | **already correct**. No edit. |
| §3.3, the sentence introducing Figure 5 | which posterior | **needs the new text** (E3): after the rerun there are two dMFM configurations in the section and "the dMFM sampler" no longer identifies one. |
| §3.3, the $N=8$ operating-point sentence | strength / budget | **needs the new text and four numbers** (E4). |
| §4 Discussion, DNA sentence | -- | **needs the new text** (E5): the discretisation caveat must appear in the main text. |
| Figure 5 caption | dMFM | **needs the new text** (E6). |
| Tables 14, 15 captions | GLASS-on-base | **needs the new text and the cells** (E8). |
| App. B.1 opening and Tables 4, 5, 6 | dMFM samples | Table 4 **already correct and final** (sampler-independent); Table 5 and the diversity paragraph **need numbers** (E10, E11); Table 6 **needs numbers** and its existing marker extended to cover the sampler (E10). |
| App. F.1 / Table 13 | dMFM, exact rewards | **already correct**, but see the caveats list: these runs use the one-jump diagonal student under the *stock clip-down* rule, so the DNA section will carry two gradient-scale rules. E12 states Table 13's student; whether to unify the rule is an authors' decision, not a text fix. |
| App. F.2 | "terminal samples from the dMFM sampler" | **needs the new text** (E7) -- one of the two overclaiming sites. |
| App. F.4, the reward and the differentiability trick | -- | **already correct**. No edit. |
| App. F.4, the search-baseline paragraph ("use the SDE form of the same base generator ... with $V\equiv0$") | posterior | **already correct**. No edit. |
| App. F.4, configuration | student / steps / scale rule | **needs the new text** (E12) -- none of the three is stated anywhere in the paper today. |
| App. F.4, the discretisation caveat | -- | **needs the new text** (E13). |
| App. F.4, "the effect of tilting the dMFM in Figure~\ref{fig:histograms}" | dMFM | **needs a marker** (E14): the queued Figure 7 regeneration is GLASS. |
| App. F.5 / Table 17 | dMFM / GLASS Hamming | **already correct**; E12's cross-reference names its student, so no separate edit. |
| App. F.6 | four-map dMFM; "compares GLASS Monte Carlo error" | **needs the new text** (E9) -- the concession becomes wrong in the other direction. |
| Figure 6 caption, Table 16 | base-model marginals | **already correct**: base-model marginals, unaffected by the posterior (§7 item 8: "Fig 6/Table 16 are base-model marginals and unaffected"). No edit. |
| Figure 7 sub-captions | guidance strength $\gamma$ | **needs numbers**; already marked, owner dna-regen. E14 adds the sampler caveat. |
| Figure 8 caption | "guidance using dMFM" | **needs a marker** (E15). |
| Table 9 (`tab:dna-per-length`), Table 10 (`tab:parent-split`), Table 11 (`tab:c0-regressor-accuracy`), Figure 3 caption | training, splits, regressor accuracy | **already correct**: all sampler-independent. Figure 3 is a schematic whose only numbers are the two regressor accuracies. No edit. |
| **Algorithm 3** (`alg:mol mfm steer search`) and Algorithm 2 (`alg:mfm steering`) | steer-search look-aheads | **does not touch the DNA experiments.** Both are multimodal/molecular: their `REQUIRE` lists are the continuous MFM $\Phi^c$ plus the discrete dMFM $\m X^d$, they advance with \eqref{eq:multimodal steer}, and steer-search's PB-validity filter has no DNA analogue. §7 item 5's code change (harvest the MFM look-aheads rather than the base model's `endpoint_prediction`) is a QM9 change. The paper contains **no** algorithm for dMFM *steering* -- only Algorithm 1, dMFM *training* -- which is why E12 has to carry the DNA guidance step. No edit. |
| §3.1 \eqref{eqn:guidance reward dna} and App. F.4's restatement of it | reward / strength | **needs the new text** (E16): both state $-(f-y^\star)^2$ while every run exponentiates $-11.1\,(f-y^\star)^2$. New discrepancy, in no existing audit. |
| \eqref{eq:ode} / \eqref{eq:multimodal steer} and the marker at \eqref{eq:multimodal steer} | guidance coefficient | **needs one clause**: `docs/paper_changes.md` §3.15 records the coefficient substitution as molecular-only, but the DNA sampler also departs from \eqref{eq:ode}, by $10$--$16\times$ plus gradient renormalisation. E12's last sentence states it in App. F.4; if the authors want it at the equation too, add "the DNA runs depart from it as well, by a factor of $10$--$16$ over their guidance window and additionally by rescaling the gradient to a fixed norm; see Appendix~\ref{molecule train results}" to the existing marker at line 260. Counted inside E12, not as a separate edit. |

---

# Claims that cannot be made even after the reruns

1. **That the scale-free tables measure the dMFM's approximation error.** The reference is a
   finite-sample GLASS-2048 estimate, and over a 256-fold increase in $N$ its own GLASS-versus-GLASS
   relative $\ell_1$ falls only 3.66x against the ~16x of independent sampling
   (`docs/paper_changes.md` §3.18). Making the finite-$N$ side dMFM does not separate the two error
   sources. Isolating the approximation error needs a reference converged in $N$, or common random
   numbers shared between samplers. This is what E7 and E9 say.
2. **That a more accurate posterior gives better steering, on this experiment.** Measured false:
   `glass_euler32` leaves the guided oracle MAE at the unguided value (1.1905 vs 1.1907; 0.8098 vs
   0.8098), and the well-integrated reference value gradient has norm 0.15--0.17 against 8.7--38.8 at
   the same probe times for the two configurations the paper reports. No rerun changes this, because it is a property of the experiment as designed.
3. **That the guidance constants are transportable.** `guidance_frac 8`, `coeff_cap 10`,
   `grad_clip 10` were tuned to the GLASS-4 gradient magnitude; under clip-down the effective
   strength depends on the posterior plugged in. Only the scale-free rule makes a reported strength
   meaningful across samplers, and even then the *optimal* `guidance_frac` differs by arm (16 for
   dMFM against 4--8 for GLASS, `docs/dmfm_glass_parity.md` §6).
4. **That dMFM matches GLASS on DNA C0 guidance -- nor, of course, that it beats it.**
   **Revised 2026-09-29; this replaces an earlier "tie" reading of mine that the data no longer
   support.** The $n=32/64$ parity runs of `docs/dmfm_glass_parity.md` looked like a tie
   (0.144 +- 0.020 vs 0.150 +- 0.023 best-of-grid at $y^\star=+1$). The Table 1 rerun has since
   measured the same comparison on 800 shared `sample_idx` under the identical scale-free rule on both
   arms, and the tighter measurement governs: at $y^\star=+1$ the per-sample paired difference is
   **dMFM $-$ GLASS $= +0.0403$, SEM 0.0072, CI [0.0260, 0.0549]**, about 5.6 SEM, with dMFM better on
   only 44.5% of pairs (guided oracle MAE 0.1933 [0.1836, 0.2036] at $n=1000$ against GLASS's
   0.1517 [0.1436, 0.1597] at $n=800$). The two measurements are statistically compatible; the $n=64$
   one was simply too noisy to resolve the gap. Source:
   `results/dna/c0_guidance/table1_dmfm_PRELIMINARY_partial.json`, whose $y^\star=+1$ dMFM arm is
   final at $n=1000$ while the GLASS arm is $n=800$ and $y^\star=-1$ is $n=200$.

   **What is supportable.** (a) The demonstration: dMFM guidance moves the parent-disjoint oracle MAE
   from 1.174 to 0.193 and improves 97.5% of matched pairs, against 0.610 for the mis-wired
   configuration that was withdrawn -- which is the claim §3.1 and Table 1 actually make, since
   Table 1 is unguided-versus-guided and contains no GLASS row. (b) Where the two *are* compared:
   dMFM is **modestly but clearly behind** GLASS. **Never parity, never superiority.** If a GLASS row
   is added to Table 1 from the matched `table1_glass_norm_n1000` arm, it must read as a deficit of
   about $0.04$ in oracle MAE at $y^\star=+1$, with the $y^\star=-1$ comparison not yet measured, and
   the numbers must come from the final artifact rather than from the published GLASS values.
5. **That \eqref{eq:ode} describes the DNA runs.** The implemented coefficient is $10$--$16\times$
   the stated $\sigma_t^2/2$ over $t\in[0.5,0.95]$, and with the scale-free rule the step's magnitude
   no longer depends on $\|\nabla V\|$ at all -- only its direction enters. This is the DNA
   counterpart of `docs/paper_changes.md` §3.15 and, like it, is an authors' decision rather than a
   number.
6. **That the four-jump student is better than the one-jump student end-to-end.** Guided oracle MAE
   0.228/0.153 at four jumps against 0.200/0.180 at one (`docs/dmfm_glass_parity.md` §2) -- a wash,
   with the sign differing by target, and the two rows are not even under the same gradient-scale
   rule, since no one-jump configuration was run with the scale-free rule. The step-matched
   *gradient-direction* claim (cosine 0.430 against 0.261 at $t=0.5$) does hold; the paper should not
   slide from one to the other. Note that the $n=800$ measurement behind claim 4 does **not** speak to
   this axis: only the four-jump configuration was rerun at scale, so the student/step-count
   comparison still rests on the $n=64$ parity run and stays a wash by default rather than by
   measurement.
7. **That Figure 4's right panel is a like-for-like comparison under the new convention.** Its dMFM
   series predates the scale-free rule, so its guidance strength is calibrated against the
   clip-down rule while Table 1's is not. Whether that matters is dna-fig4's and the authors' call;
   it is not repaired by any rerun listed in §7 item 8.
8. **That Table 13's exact-reward runs and Table 1 use the same guidance rule.** Table 13 is the
   one-jump diagonal student under clip-down; Table 1 is the four-jump student under the scale-free
   rule. Convention 2 only mandates the scale-free rule "wherever a GLASS comparison is shown", and
   Table 13 shows none, so both are defensible individually -- but the paper cannot describe a single
   DNA guidance configuration. E12 states both; unifying them would need a rerun nobody has queued.
9. **That the paper's stated reward is the reward that was optimised.** Every DNA C0 run
   exponentiates $-11.1\,(f_{\text{cyc}}-y^\star)^2$, not the $-(f_{\text{cyc}}-y^\star)^2$ of
   \eqref{eqn:guidance reward dna} (E16). No rerun changes this and it applies to the submitted
   numbers too; it is a disclosure the reruns do not remove.
10. **That the reruns as currently queued satisfy §7 item 8.** Figure 7, Figure 8 and the generated
   sets behind Tables 5 and 6 are still GLASS. Their claims stay false until those three are rerun
   with the four-jump student.
