# The multiMFM-SS guide/oracle gap: selection bias, not a calling bug

**Verdict: (A), selection bias — decisively, and with no code defect anywhere in the
guide/oracle path.** The 20x disagreement the molecular audit flagged
(`docs/mismatch_audit_molecules.md`, "Rows that need an author decision", item 3:
Δε guide MAE 21–35 meV against oracle MAE 406–419 meV) appears **only** where a
best-of-N selection on the guide has been applied. On unselected samples the two regressors
agree to within 4 % (18 000 trajectory endpoints, §4) and to within 13 % (unselected
look-aheads drawn from the very pool SS selects from, §7 below). No number in Table 2 changes; what
changes is the account of what multiMFM-SS does.

Written 2026-09-29. Everything below is reproducible from `results/qm9/ss_diagnostics/`
(four scripts, thirteen JSON/CSV outputs, `PROVENANCE.md`; commands in §9). The two
verdict-carrying measurements (§§1 and 4) need no GPU.

---

## 1. The anchor: both regressors on held-out ground-truth QM9

`results/qm9/ss_diagnostics/regressor_anchor.py` scores real QM9 molecules in exactly the
representation `multimfm.glass_guidance.tfg_property_prediction` hands the regressors
(heavy atoms only, `max_len = 9`, coordinates centred, TFG atom order `C,H,N,O,F,P,S,Cl,*`,
argmax atom types, `padding_mask`). MAE against the dataset's own property value:

| partition | n | Cv [cal/(mol K)] | μ [D] | α [Bohr³] | Δε [meV] | ε_HOMO [meV] | ε_LUMO [meV] |
|---|---|---|---|---|---|---|---|
| **test** (held out from both) | 2000 | 0.065 / 0.064 | 0.236 / 0.214 | 0.147 / 0.170 | 88.4 / 91.5 | 57.0 / 64.3 | 57.8 / 55.0 |
| **validation** (held out from both) | 1800 | 0.062 / 0.057 | 0.212 / 0.190 | 0.132 / 0.135 | 83.9 / 83.5 | 55.1 / 59.7 | 50.0 / 50.8 |
| train_flow (0–50k) | 2000 | 0.032 / 0.062 | 0.025 / 0.195 | 0.079 / 0.146 | 15.8 / 87.5 | 9.6 / 61.2 | 10.8 / 52.1 |
| train_classifier (50k–100k) | 2000 | 0.062 / 0.027 | 0.235 / 0.031 | 0.150 / 0.059 | 89.9 / 15.8 | 57.9 / 12.2 | 53.2 / 10.6 |

Cells are **guide / oracle**. R² on `test` is 0.92–0.999 for both (worst case μ: guide
0.919, oracle 0.932; Δε: 0.986 / 0.988).

**The oracle is not mis-called.** On the two splits that are held out from both networks the
two MAEs are within 1–16 % of each other, and neither is systematically worse (the oracle is
better on three of six properties on `test`, three of six on `validation`). An oracle
that was being fed a wrong representation, wrong units or an unfinalised state could not
reach 91.5 meV / R² = 0.988 on ground truth. The 406–419 meV it reports on the *selected*
SS molecules is therefore not an artefact of how it is called.

Two by-products of the anchor, both worth recording:

* **The loss-string → property mapping in `qm9_data.REGRESSOR_LOSSES` is correct for both
  kinds.** `load_tfg_regressor` picks a weight file inside `{kind}_clf_ckpt.zip` by a
  hard-coded `best_loss=<float>.pth` filename, which is fragile. The full 6 × 6
  cross-property table (`regressor_anchor.json` → `cross_property`) is cleanly diagonal:
  each checkpoint's MAE is minimal for its own property by 1–4 orders of magnitude (e.g. the
  guide's Δε head: 88.4 meV on Δε, 13 357 meV on ε_HOMO). No checkpoint is swapped.
* **The two regressors are trained on disjoint splits, and the guide's split is the
  generator's.** The guide memorises `train_flow` (indices 0–50 000: Δε MAE 15.8 meV there
  vs 88.4 on test); the oracle memorises `train_classifier` (50 000–100 000: 15.8 vs 91.5).
  So the held-out-oracle construction is sound — but the *guide* shares its training set
  with the QM9 base flow model, which is the reverse of the TFG-Flow convention where the
  classifier is trained on the second 50k precisely to keep it off the generator's data.
  This does not create the gap analysed here (it would, if anything, make the guide better
  matched to the generated distribution), but it is a provenance fact the paper does not
  state. **New discrepancy; not resolved here.**

## 2. What each regressor receives: identical tensors, verified twice

**By code.** `steer_search.evaluate_state` (`src/multimfm/steer_search.py:400–420`) calls

```python
guide  = predicted_property(state, property_name=..., property_regressor=guide_reg)
oracle = predicted_property(state, property_name=..., property_regressor=oracle_reg)
```

on the *same* `state` object, two lines apart, inside one `@torch.no_grad()`. There is no
second construction path, no re-centring between the two calls, no separate conversion.
`predicted_property` → `tfg_property_prediction` (`glass_guidance.py:278–305`) then does the
same thing for both: `coords.float()`, `mask = (~padding_mask).long()`,
`atom_types = atomics.argmax(-1).long()`. Every candidate — trajectory endpoint or
look-ahead — reaches `evaluate_state` through `qm9_eval.make_state`
(`steer_search.py:557`), which applies `mask_and_zero_com` + `apply_mask` **before** either
regressor sees it. Item by item against the hypotheses in the brief:

| candidate difference | finding |
|---|---|
| coordinate centring | identical, and both are zero-CoM (`make_state` → `mask_and_zero_com`; measured |CoM| ≤ 7e-6). EGNN is translation-invariant anyway. |
| atom-type representation | identical: both get `atomics.argmax(-1)` as `long` indices. The soft-atom relaxation (`softmax(atomics/T) @ atom_emb`) is used *only* inside the differentiable value estimator, never in `evaluate_state`, and never for the oracle. |
| atom-type ordering | one vocabulary, `TFG_QM9_ATOM_NAMES`, from the checkpoint's `data_stats`; both heads are `nn.Embedding(9, 128)` over it. |
| masking | identical `padding_mask`, same `(~mask).long()`. |
| hydrogens | heavy-only for both, everywhere (`include_hydrogens=False`, `max_len=9`). |
| dtype / device | both float32 on the same device. |
| normalisation / units | neither head normalises: `Predictor.forward` ends in a bare `nn.Linear`, and `Predictor.get_loss_batch` takes MSE against the raw dataset value. Both are trained and read in raw QM9 units (Hartree for Δε/HOMO/LUMO), converted once, downstream, by `qm9_results.HARTREE_TO_MEV`. Nothing per-kind. |
| t < 1 state vs finalised molecule | both get the *same* state; see §3. |

**By measurement.** `results/qm9/ss_diagnostics/ss_input_identity.py` runs a tiny CPU
steer-search (`--lookahead-source both`) with a forward-pre-hook on both regressors that
SHA-256s every `(coords, atom_types, mask)` triple. Result
(`ss_input_identity.json`): 18 guide calls, 5 oracle calls;
`oracle_phases = ["evaluate_state"]` only; and
**`every_oracle_input_also_given_to_guide = true`** — every tensor the oracle ever saw is
byte-identical to one the guide saw. This also re-confirms the audit's "no evaluator leak"
finding from the other direction: the oracle is never called during selection or guidance.

## 3. The t ≈ 0.7 question: the look-ahead *is* a clean-sample estimate

`selected_t_mean ≈ 0.60–0.74` is the **trajectory time t_k at which the look-ahead was
harvested**, not the state's own noise level. `steer_search.guided_sample_collect:375`
stores `float(t_value)`, the outer ODE schedule time; `gather_candidates:299` writes 1.0 for
the endpoint. The stored state itself is the MFM student's *one-shot posterior sample*
X̂₁ ~ p(X₁ | X_{t_k}), produced by `differentiable_mfm_posterior`
(`glass_guidance.py:437–480`), which integrates the student's internal flow map s: 0 → 1
conditioned on `(x_{t_k}, t_k)` and returns a t = 1 estimate, zero-CoM and masked. So no
intermediate noisy state is handed to the oracle raw, and no finalisation step is missing.

What *is* true, and is the audit's legitimate point, is that these are one-shot distilled
estimates rather than 128-step ODE endpoints, so their marginal quality differs: the
unfiltered guide-argmin look-ahead is PoseBusters-valid only 68.4 % of the time (Δε,
`best_lookahead_raw`) against 75.3 % for the SS trajectory endpoint and 98.4 % for an
unguided endpoint. PB filtering inside the selection is what lifts the returned set to
100 %.

## 4. The decisive test: dose-response in the amount of selection

`results/qm9/ss_diagnostics/selection_bias.py` re-reads the per-molecule
`guide_prediction` / `oracle_prediction` columns that the 3-seed rerun already wrote
(`results/qm9/rerun_3seed/runs/{ss,multimfm}/<prop>_s<seed>/per_sample.csv.gz`), so both
regressors are compared on identical molecules by construction — 6 properties × 3 seeds ×
1000 molecules per row. Δε, seed-averaged (full table:
`guide_oracle_dose_response.csv`, per-seed: `guide_oracle_paired.csv`):

| selection applied | guide MAE | oracle MAE | ratio | mean \|guide − oracle\| | r(errors) | PB | sel. t |
|---|---|---|---|---|---|---|---|
| none: unguided base endpoints | 1934.3 | 1873.0 | **0.97** | 266.3 | 0.971 | 0.984 | 1.00 |
| guided ODE only (multiMFM row) | 408.7 | 592.3 | 1.45 | 390.6 | 0.719 | 0.908 | 1.00 |
| guided ODE only (SS's own endpoint) | 390.2 | 468.8 | 1.20 | 361.8 | 0.689 | 0.753 | 1.00 |
| best of ≈89 base endpoints, PB-filtered | 147.7 | 473.3 | 3.21 | 386.4 | 0.586 | 0.969 | 0.85 |
| **best of ≈3680 look-aheads, PB-filtered (the SS row)** | **26.8** | **413.3** | **16.0** | **392.2** | 0.316 | 1.000 | 0.69 |
| best of ≈3680, no PB filter | 11.8 | 439.4 | 40.9 | 431.6 | 0.214 | 0.684 | 0.66 |

Three things settle it, and they hold for all six properties:

1. **With no selection the two regressors agree.** `oracle_MAE / guide_MAE` on unguided
   endpoints is 0.968 (Δε), 0.960 (HOMO), 1.003 (LUMO), 1.000 (Cv), 0.979 (α), 1.042 (μ) —
   the oracle is the *lower* of the two in four of six — with error correlation
   r = 0.909–0.993. Two comparable regressors behave exactly as the brief says they should.
   The 20x cannot be a representation bug, because the same two regressors on the same
   pipeline, in the same script, differ by 3 % one row up.
2. **The ratio is monotone in the amount of selection, and only the guide's arm moves.**
   Guide MAE 1934 → 409 → 148 → 27 → 12; oracle MAE 1873 → 592 → 473 → 413 → 439. The
   oracle arm is flat-to-slightly-improving while the guide arm falls by two orders of
   magnitude. Selection pressure, not a fixed offset.
3. **The winner's-curse identity holds numerically.** If selection drives the guide's
   prediction onto the target, then |oracle − target| must collapse onto |oracle − guide|.
   At the SS row:

   | | Cv | μ | α | Δε | HOMO | LUMO |
   |---|---|---|---|---|---|---|
   | oracle MAE of the selected molecule | 0.414 | 0.528 | 1.009 | 413.3 | 243.1 | 264.7 |
   | mean \|guide − oracle\| on those same molecules | 0.404 | 0.507 | 1.002 | 392.2 | 240.7 | 259.7 |

   Agreement to 1–5 %. The "oracle error" of an SS molecule *is* the two regressors' mutual
   disagreement, re-labelled — which is the signature of (A) and of nothing else.
   Correspondingly, the error correlation falls from 0.91–0.99 (no selection) to 0.19–0.48
   (SS row) and to −0.05–0.34 with the PB filter removed: selection strips out exactly the
   shared, informative component of the two predictions and returns the guide-specific
   residual.

The one honest caveat: on *generated* molecules neither number is a true property. The two
regressors disagree by 94 meV on real held-out molecules (Δε) but by 266–432 meV on generated
ones — both degrade ≈ 3–4.6x off-distribution. Most of that degradation is *shared* (which is
why the unselected MAEs still match, r = 0.97 on errors), but it carries an independent
per-network component of roughly 400 meV, and a best-of-3680 argmin on one network isolates
exactly that component. The oracle MAE is the better of the two available estimates — it is
the one the selection did not optimise — not ground truth. Closing the remaining gap needs
DFT, not a third regressor.

## 5. Fix: none. No behaviour changed, no flag added

No defect was found in `src/multimfm/steer_search.py` or `src/multimfm/qm9_data.py`, so
**neither file was modified** and no stored result is invalidated. The one fragile thing
found — `load_tfg_regressor` selecting weights by a hard-coded loss-value filename — was
verified correct by the 6 × 6 cross-property table rather than changed, since changing it
would risk the numbers it currently reproduces.

**Effect on the Table 2 multiMFM-SS row: none.** The reported quantity is the held-out
oracle MAE, and it stands, both as submitted (0.40 / 0.69 / 1.10 / 462 / 252 / 316, with the
replicate-mixing caveat of `mismatch_audit_molecules.md`) and as rerun on 3 held-out seeds
(`REPRODUCIBILITY_PLAN.md` §7 items 4 + 5: 0.414 / 0.528 / 1.009 / 413.3 / 243.1 / 264.7). What must not be reported
as an accuracy is the *guide* MAE of the selected molecule, which `summary.json` also
stores: for the SS rows it is the selection objective evaluated on its own argmin, an
in-sample quantity that a large enough candidate pool drives arbitrarily close to zero. It
is not in Table 2; it should not enter the revision either.

## 6. What multiMFM-SS's advantage actually is

Paired, on the same guided trajectories, the same 3 held-out seeds, n = 1000 each
(`ss_advantage_decomposition.json`). All entries are held-out **oracle** MAE except the last
column, which is the *guide* MAE of the selected molecule with, in brackets, its change against
the guide MAE of the same trajectory's endpoint.

| property | unit | multiMFM endpoint | SS's own endpoint | SS selected | selected vs SS endpoint | selected vs multiMFM | guide MAE of the selected |
|---|---|---|---|---|---|---|---|
| Cv | cal/(mol K) | 0.444 | 0.514 | 0.414 | **−19.5 %** | −6.8 % | 0.0126 (−96.6 %) |
| μ | D | 0.818 | 0.720 | 0.528 | **−26.7 %** | −35.5 % | 0.0235 (−94.8 %) |
| α | Bohr³ | 1.166 | 1.324 | 1.009 | **−23.8 %** | −13.4 % | 0.0101 (−98.7 %) |
| Δε | meV | 592.3 | 468.8 | 413.3 | **−11.8 %** | −30.2 % | 26.83 (−93.1 %) |
| ε_HOMO | meV | 270.1 | 268.0 | 243.1 | **−9.3 %** | −10.0 % | 3.30 (−98.5 %) |
| ε_LUMO | meV | 395.3 | 320.1 | 264.7 | **−17.3 %** | −33.0 % | 7.07 (−96.5 %) |

The "selected vs SS endpoint" column is a *paired* comparison — same trajectories, same
targets, same molecules' initial noise — and it is significant for all six properties on the
3 held-out seeds: Cv −0.100 ± 0.015 (t = 6.9), μ −0.192 ± 0.012 (16.3), α −0.315 ± 0.009
(36.9), Δε −55.5 ± 9.8 (5.7), HOMO −24.9 ± 5.5 (4.5), LUMO −55.4 ± 11.1 (5.0). So the gain is
real; it is its *size* that the guide metric misrepresents. (The "selected vs multiMFM" column
is not paired — multiMFM and SS use different μ, `REPRODUCIBILITY_PLAN.md` §7 item 9 — and is quoted only because it is
the comparison Table 2 makes.)

As a check that this re-derivation is faithful, the seed-averaged selected oracle MAEs
recomputed here from the per-molecule files reproduce `rerun_3seed_summary.csv`'s
`oracle_mae_mean` to 10 significant figures for all six properties.

So the defensible claim is:

* Selecting the PB-valid look-ahead with the lowest guide-predicted error improves the
  held-out oracle MAE of the returned molecule by **9–27 %** over the endpoint of the same
  guided trajectory, at no extra generative NFE.
* It improves the *guide* metric by 93–99 % on the same molecules. So **roughly four fifths to
  nineteen twentieths of the apparent improvement is selection on the scorer** and does not
  transfer to an independent evaluator. Any statement of SS's benefit that quotes a guide-side
  number, or that reads the 100 % PoseBusters validity as a validity measurement rather than
  as a selection criterion optimised over ≈3680 candidates, overstates the method by that
  factor.
* The honest framing of the mechanism: SS buys a real but modest accuracy gain by trading
  *guide* evaluations for target attainment, and its returned molecules are one-shot MFM
  posterior samples harvested at t ≈ 0.7, not draws from the t = 1 generative distribution
  (§3). Both facts belong in §3.2 / Appendix Algorithm 3.

## 7. Pool-level check: unselected *look-aheads* also agree

§4's dose-response compares selected look-aheads against *endpoints*. The one remaining
version of (B) that survives that comparison is "look-ahead states are a representation the
oracle mishandles", so it has to be tested on look-aheads themselves, unselected.
`results/qm9/ss_diagnostics/lookahead_pool.py` reruns a steer-search trajectory with the
stored rerun's own arguments (`--lookahead-topk 0`, whole pool kept) and then scores, with
both regressors on identical tensors: the endpoint, a uniformly random look-ahead, a random
*PoseBusters-valid* look-ahead, and Algorithm 3's guide-argmin.

Run at the stored configuration for Δε, α and μ (n = 64 molecules, 128 ODE steps, 32
posterior samples, **3680 look-aheads per molecule**, 24 of them sampled at random per
molecule for the pool statistics; Slurm **49348724**, A100, `lookahead_pool_{gap,alpha,mu}.json`):

| candidate | Δε [meV] guide / oracle | ratio | α [Bohr³] guide / oracle | ratio | μ [D] guide / oracle | ratio | PB |
|---|---|---|---|---|---|---|---|
| trajectory endpoint | 401.5 / 545.4 | 1.36 | 0.694 / 1.150 | 1.66 | 0.527 / 0.631 | 1.20 | 0.70–0.75 |
| **random look-ahead** | **1255.1 / 1345.4** | **1.07** | **3.806 / 3.990** | **1.05** | **1.042 / 1.061** | **1.02** | 0.58–0.61 |
| **random PB-valid look-ahead** | **1292.1 / 1461.1** | **1.13** | **3.174 / 3.176** | **1.00** | **1.139 / 1.055** | **0.93** | 1.000 |
| guide-argmin, PB-valid (Algorithm 3) | 27.9 / 432.5 | **15.5** | 0.003 / 0.867 | **289** | 0.004 / 0.493 | **123** | 1.000 |

The two middle rows are the point: on **unselected look-aheads** — the exact class of state SS
returns, at the same harvest times (mean t ≈ 0.50) — the guide and the oracle agree to within
0.93–1.13x, the oracle being the lower of the two for μ in the PB-valid row. Only the last
row, which differs
from them by *nothing but the choice of candidate*, shows the 15–289x. Across the pool the
two regressors' errors correlate at r = 0.77–0.94. (The n = 64 guide-argmin row reproduces the
n = 1000 3-seed rerun it copies its arguments from: Δε 27.9 / 432.5 against 26.8 / 413.3,
α 0.003 / 0.867 against 0.010 / 1.009, μ 0.004 / 0.493 against 0.023 / 0.528, PB 1.000.)

Two further readings of the same pool:

* **The pool is not the binding constraint; the guide's off-distribution error is.** Taking the
  *best* PB-valid candidate out of just 24 random draws by oracle error reaches 258 meV / 0.270
  Bohr³ / 0.138 D, against 433 / 0.867 / 0.493 for the guide's argmin over all 3680. That
  number is reported here **only** as a diagnostic upper bound and is itself optimistically
  biased (it is a best-of-24 on the evaluator); selecting on the evaluator is never used for
  any reported number. Its meaning is that a better scorer, not a bigger pool, is where the
  remaining headroom is.
* **Diminishing returns for the evaluator, none for the scorer.** Within a 24-candidate
  subsample the guide-argmin drives the guide error to 130 meV (Δε) while the oracle error at
  that same candidate sits at 413 meV; going to the full 3680 candidates drives the guide error
  to 28 meV and leaves the oracle error at 433 meV. The scorer's metric keeps improving with
  pool size; the evaluator's does not.

A reduced-scale CPU rerun of the same script (`lookahead_pool_gap_cpu_reduced.json`: n = 16,
32 ODE steps, 8 posterior samples, 232 candidates, much weaker guidance, so absolute levels do
not transfer) reaches the same conclusion independently of the GPU: random look-ahead
1380.2 / 1358.5 (ratio 0.98), random PB-valid 1554.2 / 1512.2 (0.97), guide-argmin
306.4 / 701.4 (2.29).

## 8. Summary for the revision

| question | answer |
|---|---|
| Is the oracle mis-called or mis-fed? | No. Identical tensors (hash-verified), correct checkpoint (6 × 6 cross table), 91.5 meV / R² 0.988 on held-out ground-truth Δε. |
| Is the gap selection bias? | Yes. Ratio 0.97 with no selection, 1.07 on *unselected look-aheads from the very pool SS selects from*, 3.2 at ≈89 candidates, 16 at ≈3680; and the oracle MAE of the selected molecule equals the two regressors' mutual disagreement to 1–5 %. |
| Any code change? | None. `steer_search.py` and `qm9_data.py` untouched; every stored result stands. |
| Table 2 SS row | Unchanged. Report the oracle MAE, never the guide MAE of the selected molecule. |
| What SS actually buys | −9 % to −27 % held-out oracle MAE against the endpoint of the same trajectory, at no extra generative NFE — against a −93 % to −99 % improvement in the guide metric. |
| New discrepancy found | The guide regressor is trained on `train_flow` (0–50k), the same split as the QM9 base flow model; the oracle is on `train_classifier` (50k–100k). The held-out-evaluator construction is sound, but the guide/oracle split assignment is the reverse of TFG-Flow's convention and is not stated in the paper. |
## 9. Reproduction

From the repo root, in the `multimfm` env. Everything except the last item is CPU and takes
minutes.

```bash
PY=~/micromamba/envs/multimfm/bin/python
# 1. anchor both regressors on ground truth (+ the 6x6 cross-property table)
$PY results/qm9/ss_diagnostics/regressor_anchor.py --limit 2000 --partition test --cross \
    --out results/qm9/ss_diagnostics/regressor_anchor.json
for P in validation train_flow train_classifier; do
  $PY results/qm9/ss_diagnostics/regressor_anchor.py --limit 2000 --partition $P \
      --out results/qm9/ss_diagnostics/regressor_anchor_$P.json
done
# 2. paired guide-vs-oracle dose-response from the stored 3-seed rerun (no model needed)
$PY results/qm9/ss_diagnostics/selection_bias.py
# 3. prove the two regressors get identical tensors inside steer-search
$PY results/qm9/ss_diagnostics/ss_input_identity.py
# 4. pool-level decomposition (GPU, ~30 s per property; or add --sample-steps/--value-samples for CPU)
$PY results/qm9/ss_diagnostics/lookahead_pool.py --property gap --num-samples 64 \
    --pool-subsample 24 --device cuda --pb-workers 12
```

Outputs: `regressor_anchor*.json`, `guide_oracle_paired.csv`,
`guide_oracle_dose_response.{csv,json}`, `ss_advantage_decomposition.json`,
`ss_input_identity.json`, `lookahead_pool_<prop>*.json`, all in
`results/qm9/ss_diagnostics/`.

