# Figure 4 — canonical paper text (dna-fig4)

Durable home for the caption and §3.3 sentences that belong to Figure 4, so they do not
live only inside an agent report. **This file is the source of truth for the wording; the
tex agent owns `iclr2027_conferencemod.tex` and applies it there.**

Everything factual below is checked against, in order of authority:

1. `results/dna/original_scripts/benchmark_yeast_split_value_mc.py` — the *recovered original*
   producer of the left panel (verbatim from `tullebulle/DNA-MFM@187fe7b`);
2. `src/dmfm/benchmarks/value_error.py` — the port of it that the rerun executes;
3. `src/dmfm/benchmarks/{search,dmfm_steering,plan}.py` — the right panel;
4. `src/dmfm/benchmarks/worker.py` — the settings the queued jobs actually use.

Placeholders in `[[...]]` are filled from `results/dna/fig4/summary.json` once
`bash scripts/dna/fig4_collect.sh` has run. **Do not invent them.**

---

## 1. Review of the caption currently in the tex file (lines 631–643)

Factually correct, and can stand as written: the synthetic smooth sequence reward rather
than $f_{\mathrm{cyc}}$; conditioning states as held-out test sequences noised to their own
$t\sim\mathcal U(0.05,0.35)$; DPS using the student's denoiser; FMAP as the teacher ODE
integrated to $t=1$ and therefore independent of $N$; and the reference $V_t$ being a
finite-sample teacher-SDE estimate that bounds the comparison from below. Three corrections:

### Correction 1 (substantive) — an invented rationale in the "Posterior Recovery" paragraph

Line 647 currently reads:

> This benchmark uses a synthetic smooth sequence reward rather than the cyclizability
> predictor of Section~\ref{mechanical_prop}, **so that the reward is exactly computable on
> the conditioning states**, which are held-out test sequences noised to their own outer time
> $t\sim\mathcal U(0.05,0.35)$.

The emphasised clause is not supported by anything. Both rewards are exactly computable —
$f_{\mathrm{cyc}}$ is a small CNN evaluated in closed form — and the original script records
no motivation for the choice. Replace with:

> This benchmark uses a synthetic smooth sequence reward rather than the cyclizability
> predictor of Section~\ref{mechanical_prop}; the conditioning states are held-out test
> sequences noised to their own conditioning time $t\sim\mathcal U(0.05,0.35)$.

### Correction 2 (terminology) — "outer time"

The quantity is the *conditioning* time $t$ of $p_{1|t}(\cdot\mid x_t)$ (`t_cond` in the
code). Replace "outer time" with "conditioning time" in both the caption and the paragraph.

### Correction 3 (precision) — "bounded below"

"every curve is bounded below by the reference's own sampling error" overstates a per-curve
guarantee. The correct statement is about resolution:

> no curve can be resolved below the reference's own Monte Carlo spread

---

## 2. The two questions

### Q1. Should the published panel name the `sde_mc` reference-noise-floor series?

**Yes — plot it and name it.** The original script computes it and the published figure drops
it. Three reasons to put it back:

* it costs nothing (already computed, from the same samples as the reference);
* it is the honest "exact posterior sampler at budget $N$" line, so the dMFM curve can be read
  against the estimator it is distilled from rather than only against two flat lines;
* without it a reader cannot separate dMFM estimator bias from Monte Carlo noise in the
  reference, which is exactly the first question a reviewer asks of a panel whose reference is
  itself an estimate.

It may well sit below the dMFM curve at every $N$. That is the honest comparison and should be
shown; the paper's claim is about scaling in $N$, which the series does not contradict.
Suggested legend entry: `SDE MC (reference's own finite-$N$ error)`.

### Q2. Final conditioning-state count?

**32.** `src/dmfm/benchmarks/worker.py::PANEL_A = {"n_conditions": 32, "block": 4}` is what the
queued arrays run (8 shards of 4).

Two things the caption should reflect:

* The original's default — in both the script and its launcher — is **8**. No log of the
  plotted run survives, so 8 is an assumption about the submitted figure, not evidence.
* Conditions are drawn in order from one seeded stream, so **conditions 0–7 are exactly the
  original protocol** and 8–31 are additional independent draws added only to get error bars.
  `panelA_per_condition.csv.gz` keeps the per-condition numbers, so any subset (including just
  the first 8) can be recovered without rerunning.

---

## 3. Canonical caption

```latex
\caption{DNA inference-time steering on the parent-disjoint $L=50$ models.
\textbf{Left:} value-function estimation error $|\hat V_t(x)-V_t(x)|$ against the number of
posterior samples $N$, for the dMFM at one map evaluation per sample (NFE $=1$), the DPS
denoiser approximation, and a one-step flow map (FMAP). The reward here is a synthetic smooth
sequence reward, \emph{not} the cyclizability predictor of Section~\ref{mechanical_prop}; the
conditioning states are held-out test sequences noised to their own conditioning time
$t\sim\mathcal U(0.05,0.35)$. DPS evaluates the student's denoiser and FMAP integrates the
teacher ODE from $t$ to $1$; both are deterministic point estimates computed once per
conditioning state and are therefore independent of $N$. The reference $V_t$ is itself a
finite-sample Monte Carlo estimate (teacher SDE, 512 samples), so no curve can be resolved
below its own sampling error, which is shown as the \textsc{sde mc} series. Averaged over
[[N_COND]] conditioning states.
\textbf{Right:} target error $|f_{\mathrm{cyc}}(x)-y^\star|$ against generative-model NFE per
returned sample, for best-of-$N$, Feynman--Kac steering, beam search and MCTS implemented as in
\citet{didi2026binder}, each swept over its own budget. Every method uses the same base
generator, guide, target, seeds and evaluation, and each baseline's hyperparameters were chosen
on a separate pilot at matched NFE; [[N_OUT]] samples per point, error bars are the standard
error over [[N_REP]] repeats. NFEs count forward passes of a generative network and assume no
parallel execution; they are not wall-clock, and the dMFM additionally performs one backward
pass per guided step.}
```

Placeholders: `[[N_COND]]` = 32 unless the runs are cut short (`summary.json →
panelA.args.n_conditions`); `[[N_OUT]]` = 100 (`panelB.n_outputs`); `[[N_REP]]` = 3
(`panelB_curves.csv → n_repeats`).

---

## 4. §3.3 sentences

### "Posterior Recovery" — replacement for the two sentences added at line 647

> This benchmark uses a synthetic smooth sequence reward rather than the cyclizability
> predictor of Section~\ref{mechanical_prop}; the conditioning states are held-out test
> sequences noised to their own conditioning time $t\sim\mathcal U(0.05,0.35)$. The reference
> $V_t$ against which all three estimators are scored is itself a Monte Carlo estimate,
> obtained by sampling the teacher SDE from $t$ to $1$, so the comparison does not resolve
> differences beneath the reference's own sampling error; we plot that error as its own series.

### "Steering" — sentence to add (already present in substance at line 653, keep it)

> Every method uses the same base generator, the same guide, the same target and the same
> evaluation, and each baseline's hyperparameters were chosen on a separate pilot at matched
> NFE.

### "Steering" — the low-NFE claim

> dMFM achieves low target error at low NFEs, the regime most relevant for practical sampling.

**Currently unsupported.** It must be re-checked against
`results/dna/fig4/summary.json → panelB.matched_nfe_comparison`, which reads every baseline
curve at each dMFM point's NFE, and `panelB.dmfm_wins_at_lowest_nfe`. In the submitted version
the four baseline curves were never measured — they were hard-coded in the plotting script with
a reported standard error of zero at three repeats — so nothing yet supports this sentence.
Rewrite it to whatever the collected numbers show.

### Method note worth one sentence in the appendix

> The search baselines require a stochastic sampler in order to branch, and use the SDE form of
> the same base generator (Eq.~2 with $V\equiv0$), which has the same marginals as the ODE
> sampler; best-of-$N$ and the dMFM use the deterministic sampler.

Without it, a reader cannot tell why beam search, MCTS and Feynman--Kac do not collapse to
best-of-$N$ on a deterministic flow. Measured evidence that the two samplers agree is in
`results/dna/fig4/calibration.json`.

---

## 5. Where the numbers come from

| claim | file |
|---|---|
| left-panel curves | `results/dna/fig4/panelA_value_error.csv` (per condition: `panelA_per_condition.csv.gz`) |
| right-panel curves | `results/dna/fig4/panelB_curves.csv` (per run: `panelB_runs.csv`) |
| matched-NFE comparison, low-NFE claim | `results/dna/fig4/summary.json` |
| baseline hyperparameters and how they were chosen | `results/dna/fig4/pilot_choice.json`, `panelB_pilot.csv` |
| ODE/SDE sampler agreement | `results/dna/fig4/calibration.json` |
| what the submitted figure plotted | `results/dna/figure_digitized/fig4_{left,right}_*.csv` |
| surviving 2026-05-13 dMFM rerun (split65k) | `results/dna/benchmarks/recovered_2026-05-13/` |
| job IDs, models, settings | `results/dna/fig4/PROVENANCE.md` |

**The rerun is not comparable to the submitted figure**: different base model, different dMFM,
different C0 guide, and baselines that are now measured rather than hard-coded. Any old→new
comparison in the paper must say so.

---

## 6. Device policy (affects nothing in the text, but explains the two shard trees)

GPU time was unavailable for hours, so the authors authorised CPU on `serial_requeue`
(account `albergo_lab`). **A plotted panel is produced entirely on one device type.** CUDA
uses TF32 matmuls and a CUDA RNG stream, CPU true fp32 and a CPU RNG; they do not draw the
same samples and there is no A/B run showing the difference sits below the error bars, so a
mixed panel would carry a confound that could not be ruled out. GPU shards live in
`outputs/dna/fig4/`, CPU shards in `outputs/dna/fig4_cpu/`, each shard records its device,
account and partition, and `collect` warns on any directory that mixes the two.

The pilot is the one part that may run on either device — it only *ranks* baseline
hyperparameter families and no pilot shard is ever plotted — and its ranking is still taken
from a single tree. **Expect the plotted GPU sweep to have been ranked by a CPU pilot**,
because the CPU queue freed up first; the chosen configuration is then measured on the same
device as every other point in its panel, so the comparison the figure makes is unaffected.
If both a CPU and a GPU panel A end up complete, that is a free A/B and should be reported:
it would settle empirically whether the arithmetic matters here.

GPU shards may come from either `kempner_requeue` (H100) or the general `gpu_requeue` pool
(mixed A100/H100/H200/RTX Pro, account `albergo_lab` — `kempner_albergo_lab` is not a valid
association there). Both are CUDA with TF32 and the same seeded Philox stream, so they share
one tree; each shard records `device_name` if the exact GPU model is ever needed.

**Nothing in this file changes because of the device.** If a panel was produced on CPU, say
so once in the appendix and move on; no paper claim depends on it.

## 7. Notes for other agents (recorded here so they are not lost)

Resolved on 2026-09-29, kept only so the trail is readable — no action left in any of them:

* **A complete clone of `tullebulle/DNA-MFM@187fe7b` is on non-purged holylabs** at
  `/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/DNA-MFM_187fe7b`.
  No GitHub access is needed by anyone. *Done:* `results/dna/original_scripts/_sources.json`
  now points there and all eight originals verify against it.
* **`scripts/score_yeast_c0_evo2.py` exists as source** in that clone (13,706 B, sha256
  `059498df13c607cee8736fabb41f64d1b6ddfecab3f1d758c761f76787fafbe9`), not only as the `.pyc`
  the plan recorded. *Done:* the Evo 2 scorer was re-derived from it, 13 of 18 functions
  AST-identical across the scoring path.
* **`docs/REPRODUCIBILITY_PLAN.md` §2** *Done:* the Fig 4 left row now reads "Recovered" and
  names the ported module. Only the models and the plotted run's `n_conditions`/`n_list`
  remain lost.

Still open:

* **The Fig 8 producer and the Table 16 correlation code are in neither commit** of that repo
  (2 commits total; full-history grep for "Gradient Norm", "predicted endpoint", "steering.png",
  "pearson", "position-wise" is empty). Both must ship as labelled reimplementations.
* **NFE accounting is settled** (dna-core, 2026-09-29): the double network evaluation in
  `gaussian_denoiser_flow_step` was present in the original library too, and the published NFE
  counts stand because they always counted one call per base-flow step. Counting one NFE per
  base step, as `dmfm.benchmarks.core.NFECounter` does, is correct. The paper reports no
  measured DNA sampling times, so no timing footnote is needed.
