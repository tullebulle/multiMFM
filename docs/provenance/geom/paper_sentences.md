# GEOM-Drugs: exact sentences the tex agent needs to change

Source: `iclr2027_conferencemod.tex` (line numbers as of 2026-09-29). Placeholders in
`<ANGLE BRACKETS>` are filled from `results/geom/table3_asdescribed.json` once the
as-described rerun is collected (`bash scripts/geom/collect_table3_asdescribed.sh`);
`docs/provenance/geom/README.md` and `results/geom/PROVENANCE.md` hold the provenance.

Notation used below:
`<G_TANI> ± <G_SE>` = `multimfm.tanimoto_mean ± tanimoto_se`,
`<B_TANI> ± <B_SE>` = `base_generator_unguided.tanimoto_mean ± tanimoto_se`,
`<G_PB>`, `<B_PB>` = the same objects' `pb_valid_mean` in percent,
`<LAM>`, `<MU>`, `<EVERY>`, `<TMIN>`, `<TMAX>` = the selected guidance settings from
`results/geom/table3_asdescribed_selection.json`.

## 0. Interim numbers, if the as-described rerun is not finished in time

The 4,096 x 3 rerun of the *submitted* configuration (Slurm 47833756, already in
`results/geom/table3_n4096_replicates.csv`) is the best available 4,096-molecule measurement
and can fill the placeholders until the as-described run lands. Mean +- SE over its three
seeds of 4,096 molecules:

| quantity | value |
|---|---|
| multiMFM Tanimoto | 0.2241 +- 0.0001 |
| base generator (unguided, same noise and targets) | 0.2287 +- 0.0005 |
| paired gain | -0.0046 +- 0.0008 (per-molecule SEM over 11,260 pairs) |
| multiMFM PB validity | 85.0% +- 0.6 |
| base PB validity | 88.0% +- 0.4 |

Its (lambda, mu) = (0.02, 0.5) was not selected on held-out data, which is the only reason the
as-described rerun (lambda, mu selected on 512 held-out targets) exists.

## 1. The guide predictor is not off-the-shelf (§3.2, line 593) -- independent of any rerun

Current:

> Our $\mathcal E(G)$ is off-the-shelf from TFG-Flow.

This is incorrect. TFG-Flow releases only its QM9 base flow and its six QM9 property
predictors (`storage/flow_model.pth`, `guide_clf_ckpt.zip`, `oracle_clf_ckpt.zip`); it
publishes no structural-fingerprint predictor for QM9 or GEOM-Drugs. Our head was trained by
the authors in four legs (`checkpoints/geom/structure_head/train_args.json`), following
TFG-Flow's own guide architecture and fingerprint definition. Evidence:
`results/geom/tfgflow_reference.json`. Suggested replacement:

> Our $\mathcal E(G)$ follows TFG-Flow's guide predictor for this task---a six-layer
> EGNN with mean pooling and a linear head, trained with binary cross-entropy against the
> same 1024-bit OpenBabel FP2 fingerprint---which we train ourselves on GEOM-DRUGS, since
> TFG-Flow releases only its QM9 property predictors.

## 2. Table 3 caption (line 597)

Current:

> GEOM-DRUGS structural fingerprint guidance. The two reference values are from TFG-Flow.
> multiMFM yielded 87.5\% PB-valid structures, compared with 92.0\% for its base generator.

Replacement:

> GEOM-DRUGS structural fingerprint guidance. The two reference values are TFG-Flow's
> GEOM-DRUGS rows, averaged over three seeds; their generator is capped at 30 heavy atoms
> and ours at 71, so they set the scale of the task rather than a matched baseline. Our
> values are the mean $\pm$ standard error over three seeds of 4{,}096 molecules each.
> multiMFM yielded $<G_PB>\%$ PB-valid structures, compared with $<B_PB>\%$ for its base
> generator on the same molecules.

## 3. Table 3 body (lines 604-606)

Current:

> Unconditional generation (TFG-Flow ``Upper bound'') & $0.170 \pm 0.001$ \\
> TFG-Flow \citep{lin2025tfg} & $0.208 \pm 0.002$ \\
> multiMFM & \textbf{0.227} \\

Replacement (the two TFG-Flow rows are verified correct as their GEOM-Drug rows; add our own
unguided baseline, which the table currently lacks, and give our value an error bar):

> TFG-Flow unconditional (their ``Upper bound'') & $0.170 \pm 0.001$ \\
> TFG-Flow \citep{lin2025tfg} & $0.208 \pm 0.002$ \\
> Our base generator (unguided) & $<B_TANI> \pm <B_SE>$ \\
> multiMFM & $<G_TANI> \pm <G_SE>$ \\

## 4. The claim of improvement (§3.2, line 593, last sentence)

Current:

> Our similarity improves on the baseline TFG-Flow, although still moderate.

Two cases; the rerun decides which:

* if the paired gain is positive and separated from zero
  (`paired_gain.mean > 2 * paired_gain.se`):

  > Steering raises the directly counted fingerprint similarity from $<B_TANI>$ to
  > $<G_TANI>$ (paired gain $<GAIN> \pm <GAIN_SE>$ over $4{,}096\times3$ molecules), and both
  > exceed the TFG-Flow reference values, although the absolute similarity remains moderate.

* if it is not (this is what the 4,096$\times$3 run of the submitted configuration showed,
  gain $-0.0046 \pm 0.0008$):

  > Our unconditional base generator already exceeds the TFG-Flow reference values on this
  > metric ($<B_TANI>$ versus $0.208$), but steering does not improve the directly counted
  > fingerprint similarity at this scale: the paired change over $4{,}096\times3$ molecules is
  > $<GAIN> \pm <GAIN_SE>$, while PoseBusters validity drops from $<B_PB>\%$ to $<G_PB>\%$.
  > We report it as a negative result for this benchmark.

  **CORRECTION (2026-09-29).** An earlier version of this file added "The guide's own
  (head-predicted) similarity does improve, so the gap is a guide-quality and not an
  estimator problem." **That is false and must not be used.** I had taken it from the n=64
  grid point; at scale the head-predicted Tanimoto *falls* as well --
  `results/geom/table3_n4096_shards.csv`: pooled $0.2338 \to 0.2308$ ($-0.0030$), negative in
  9 of 11 shards, per replicate $0.23500\to0.23129$, $0.23259\to0.23031$,
  $0.23376\to0.23078$; the held-out sweep shows the same sign at every $(\lambda,\mu)$.
  Guidance therefore fails to improve even the quantity it optimises, which points at the
  guidance strength/scale, the estimator, or an implementation defect rather than at guide
  quality. Until `multimfm.geom.diagnostics grad-sign` has run on a GPU, the paper should
  state the measured numbers and *not* offer a mechanism.

## 5. Guidance settings are unstated (App. F.8, "Use during guidance", line ~2346)

The appendix states the inner steps, posterior samples and outer steps but no guidance
strength, and the implementation guides only on a subset of steps. Add after the existing
GEOM sentence:

> For GEOM-DRUGS the reward was tilted with $\lambda=<LAM>$ and the value gradient scaled by
> $\mu=<MU>$, applied every $<EVERY>$ outer step(s) for $t\in[<TMIN>,<TMAX>]$; these were
> selected on a held-out set of 512 target fingerprints disjoint from the evaluated targets.

Also note for the authors (no wording proposed, it is a modelling choice): equations
\eqref{eq:ode}/\eqref{eq:multimodal steer} carry the coefficient $\sigma_t^2/2=(1-t)/t$ in
front of $\nabla V_t$, whereas the implementation that produced every reported number
multiplies the estimated gradient by a **constant** $\mu$. Either the equations should note
that $\mu$ absorbs this factor, or the sampler should apply $(1-t)/t$.

## 6. App. F.9 sample-size sentence (line 2359)

Current:

> The GEOM-Drugs evaluation generated $4{,}096$ molecules; its reported Tanimoto similarity
> and PB validity come from the same run.

This was not true of the submitted numbers (they came from 64 molecules, one seed, the best
of a four-point grid). After the rerun it becomes true if written as:

> The GEOM-DRUGS evaluation generated $4{,}096$ molecules for each of three seeds; the
> reported Tanimoto similarity and PB validity come from those runs, and the unsteered
> column is the unguided arm of the same runs (identical noise and identical targets).

## 7. Table 21 GEOM row (line 2381) and the paragraph after it (line 2386)

Current:

> GEOM-Drugs & Structural fingerprint & $4{,}096$ & $92.0$ & $87.5$ & --\\

and

> For the selected GEOM-Drugs fingerprint-guidance run, validity was $87.5\%$.

The $92.0$ appears in no measurement of this base generator: the closest record is its
training-time validation PB (100 molecules, no-strain configuration) at epochs 26/28/37,
while the released epoch-38 checkpoint logged $98.0\%$ there;
`results/geom/teacher_training_pb_validity.csv`. Replacement:

> GEOM-Drugs & Structural fingerprint & $4{,}096\times3$ & $<B_PB>$ & $<G_PB>$ & --\\

and

> For the GEOM-DRUGS fingerprint-guidance runs, validity was $<G_PB>\%$, against $<B_PB>\%$
> for the unsteered base generator on the same molecules.

## 9. App. F.8 paragraph the tex agent asked for: the fingerprint guide predictor

Read out of `checkpoints/geom/structure_head/train_args.json` (reconstructed from the run logs
and the launcher) and `third_party/tfg_flow/dataloader/datasets/geom_fingerprint.py`.

> **Fingerprint guide predictor.** The guide $\mathcal E$ maps a heavy-atom molecule
> (coordinates and atom types) to a 1024-bit structural fingerprint. It follows TFG-Flow's
> predictor for this task: a 6-layer E(3)-equivariant graph network
> (128-dimensional node features, 128-dimensional edge features,
> attention, SiLU) whose node features are mean-pooled over the real atoms and mapped by a
> linear head to 1024 logits (1.06M parameters). Atom types are embedded from the
> GEOM-DRUGS heavy-atom vocabulary (C, N, O, F, S, Cl, Br, I and a dummy class) and coordinates
> are consumed in the generator's frame (Angstrom divided by 2.0). Targets
> are the OpenBabel FP2 path fingerprint (1024 bits) of the molecule, with bonds re-perceived
> from the 3-D coordinates, which is TFG-Flow's fingerprint definition. Training minimises the
> binary cross-entropy of the 1024 bits on 1,172,182 GEOM-DRUGS molecules, with
> 146,401 held out for validation and model selection by validation Tanimoto
> similarity (predictions thresholded at 0.5). We used AdamW with gradient clipping 1.0 and
> trained in four legs, warm-starting each from the previous: one single-GPU leg at learning
> rate $10^{-4}$, then two three-GPU legs (per-GPU batch 256, global 768) at $3\times10^{-4}$,
> and a final three-GPU leg of 40 epochs at peak learning rate $10^{-3}$ with
> 500 warmup steps and cosine decay. The selected checkpoint is epoch
> 34 of the last leg, with validation Tanimoto 0.852 and
> validation BCE 0.0686.

Validation Tanimoto by leg, if a sentence about the progression is wanted: 0.764 -> 0.788 -> 0.817 -> 0.852.

This paragraph is also what makes the old "off-the-shelf from TFG-Flow" sentence (item 1)
impossible to keep.

## 10. The abstract's "in both cases" claim (flagged by the tex agent)

Current abstract sentence: "In both cases, our steering algorithms improve target attainment
using one reward-independent MFM per base generator." On GEOM-DRUGS the current data support
no version of "improve target attainment":

| quantity | base -> steered | source |
|---|---|---|
| counted FP2 Tanimoto (the reported metric) | 0.2287 -> 0.2241, paired -0.0046 +- 0.0008 | `table3_n4096_replicates.csv` (4,096x3) |
| the guide's own head-predicted Tanimoto | 0.2338 -> 0.2308 (9/11 shards negative) | `head_tanimoto_direction.json` |
| PoseBusters validity | 88.0% -> 85.0% | same run |
| every held-out (lambda, mu) tried | negative on both similarities | `table3_asdescribed_tuning.csv` |

What the GEOM-DRUGS experiment does support, on the same data:

* the pipeline scales to molecules of up to 71 heavy atoms (vs 9 for QM9), which is the
  "We scale multiMFM to GEOM-DRUGS" sentence and is unaffected;
* the base generator is strong on this benchmark in absolute terms -- its unguided Tanimoto
  (0.2287 +- 0.0005) is above the guided 0.208 that TFG-Flow reports -- but that is a
  base-model comparison, not a steering effect, and the two generators differ in the molecule
  sizes they cover (30 vs 71 heavy atoms), so it should be stated as context, not as a win.

So the honest options for the abstract are to restrict the improvement claim to QM9 (and the
DNA experiments), or to keep "both cases" only if it is about scaling rather than attainment.
This is the authors' call; the numbers above are what it has to be consistent with. If
`multimfm.geom.diagnostics grad-sign` shows the estimated gradient does not ascend the guide
reward, the GEOM row is an implementation defect rather than a property of the method, and the
claim could be restored after a fix -- but not on the strength of the present runs.

## 8. Unchanged

Tables 19 and 20 (GEOM columns) match the checkpoints exactly
(`results/geom/table19_geom_hparams.json`, `table20_geom_hparams.json`), with one omission
worth adding to Table 20's caption or body: the selected phase also used
`max_jump` 0.15, `anneal_frac` 0.2, ESD clips 8/10 and `diag_weight` 2, and the student was
distilled on 150k of the ~1.17M molecules.
