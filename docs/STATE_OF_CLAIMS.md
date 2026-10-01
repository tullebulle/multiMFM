# State of the paper's claims

Updated 2026-09-30 (~00:00). One page, written so that a co-author can be brought up to speed on
the decisions recorded in `REPRODUCIBILITY_PLAN.md` §7, several of which change published claims.

This is the *scientific* state. For queued work and the command that finishes each piece, see
`RERUN_STATUS.md` (operational). Verdicts here are backed by first-hand measurement, not inference;
where something is still pending it says so rather than predicting.

## Verdict per claim

| Paper item | Published claim | Verdict | Evidence |
|---|---|---|---|
| Table 1, Fig 3 (DNA C0) | oracle MAE 1.170→0.157 (+1), 0.831→0.153 (−1), "1,000 pairs" | **Effect holds.** Printed cells are the n=32 pilot; the n=1000 GLASS run gives 0.148 / 0.119, inside the pilot CIs with 4.9–6.2× narrowing. Published runs used **GLASS on the base DFM, not dMFM** → dMFM rerun in flight | `mismatch_audit_dna.md`, `dmfm_glass_parity.md` |
| Table 1 rerun **with dMFM** | the demo the paper claims | **Complete at n=1000, both targets.** dMFM 1.174 → **0.193** (+1) and 0.833 → **0.172** (−1); 97.5% / 95.4% of pairs improved. Table 1's cells move from the published 0.157 / 0.153 | `results/dna/c0_guidance/table1_dmfm_n1000_summary.json` |
| dMFM vs GLASS on DNA | implied dMFM throughout | **TIE at matched NFE once each arm uses its own (steps, MC) split — the earlier deficit was a defect in the comparison, not in dMFM.** Both arms had been pinned to 4 posterior steps x MC 8; four Euler steps is *GLASS's* estimator, and since the value gradient costs `steps x MC` calls, a one-step flow map affords **MC 32 at the same NFE** — which is what a flow map exists to do. At NFE 32, n=600 paired: **dMFM 1x32 gives 0.146 / 0.120** against GLASS's own best (2x16) 0.143 / 0.117, i.e. **+0.004 +- 0.006 and +0.003 +- 0.006 — a tie at both targets**; against GLASS 4x8 (0.149 / 0.129) dMFM is slightly ahead. Mechanism: MC 8->32 moves dMFM by 0.031/0.035 while MC 8->16 moves GLASS by -0.001/0.020, so dMFM's gradient is the MC-noisier one and 4x8 spends its budget GLASS's way. On the *guide* (the optimised objective) dMFM goes from 2.4-2.8x GLASS's error to parity. **Consequence: Table 1's dMFM row should be regenerated at 1 step x MC 32 — 0.193/0.172 becomes ~0.146/0.120** | `dmfm_deficit_rescue.md` |
| Fig 5, Tables 14–15 | dMFM finite-N vs GLASS reference | **Estimator *and* reference were both GLASS.** F.6 concedes it, F.2 and the Fig 5 caption contradict it. dMFM rerun queued | `REPRODUCIBILITY_PLAN.md` §7.8 |
| Fig 7 (tilt vs strength) | distribution tilts toward target as strength grows | **Survives with dMFM.** mean \|C0−0.30\| 0.560 / 0.280 / 0.163 / 0.118 at `gf` 0/1/2/4; frac within 0.10 rises 8% → 57%; `gf` 0 gives 1000/1000 identical sequences. Two honest additions: the ladder is **not monotone** (best at `gf` 4, regressed by `gf` 32, so a "strong guidance" panel at 32 is over-driven), and matched GLASS tilts far better (guide MAE 0.033 vs 0.118). 9 of 14 points in; dMFM `gf` 8 is panel (b) and still queued | `dmfm_demo_reruns.md` |
| Fig 8 (single guided sample) | gradient pushes toward higher reward | **Survives.** `sample_idx 0`: C0 −0.361 → +0.337 (target 0.30), \|C0−0.30\| 0.661 → 0.037, value −4.85 → −0.015. dMFM is the looser arm on all four samples | `dmfm_demo_reruns.md` |
| Table 4 (real baselines) | real-data diversity | **Survives trivially** — posterior-independent, bit-identical | `dmfm_demo_reruns.md` |
| Table 5, +1 set | preserves held-out-level diversity | **Survives.** 0.9964× / 0.9708× vs published 0.9997× / 0.9447× | `dmfm_demo_reruns.md` |
| Table 5, −1 set | "measurably more clustered", 0.9507× / 0.8027× | **Does not reproduce (0.9927× / 0.9621×), but our attribution was wrong.** Blaming the `split65k` → parent-disjoint move is unsupported: the metric code is **AST-identical** to the June original, the shared reference row agrees to the printed digit, and the tables' own guidance-independent rows bracket the model move at ≤0.0024 — 24–66× smaller than the effect blamed on it. What actually differs is the **guidance protocol in six knobs** (window [0.01,0.95] vs [0.5,0.95], `reward_scale` 1.0 vs 0.5, MC 16 vs 8, one-step dMFM vs GLASS-4, trajectory to t=1.0 vs 0.95, seeds). **Open** pending job 49431930, the published dose rerun on parent-disjoint at n=1000 | `dna_findings_audit.md` |
| Table 6, ordering | −1 < held-out < +1 ≈ unguided | **Survives**: 1.3344 / 1.3412 / 1.3581 / 1.3653 | `dmfm_demo_reruns.md` |
| Table 6, magnitudes | +1 Δ +0.0075 "CIs overlap"; −1 Δ −0.1104 "substantially lower" | **Do not reproduce** (+0.0169 with non-overlapping CIs; −0.0068, 16× smaller), **but as with Table 5 the `split65k` attribution is unsupported**. The Evo 2 scoring path is AST-identical (same `evo2_7b_base`, bootstrap 2000, RC, 49 positions) and its guidance-independent rows bracket the model move at ≤0.0042 nats against the 0.099 blamed on it. The difference is the guidance protocol. **Open** pending job 49431930 | `dna_findings_audit.md` |
| Table 6 provenance | reported lost | **Recovered** from a Codex transcript, 56/56 cells reproduced exactly | `results/dna/june_split65k_recovered/` |
| Tables 13, 17, 18 | dMFM | **Hold** — these did load the dMFM students. Table 18's 2 mismatches are paper double-rounding | `mismatch_audit_dna.md` |
| Table 16 vs Fig 6 | Pearson r table | **Neither number is right — the statistic is pure noise.** The printed p-values identify the construction (n=4 global). For a sampled uniform reference the global r is the cosine of a fixed vector with an isotropic direction in R³, i.e. exactly **Uniform(−1,1)**, independent of set size (sd 0.5754 vs 1/√3, KS 0.006 over 20,000 draws). So 0.1813, the figure's −0.852 and our regeneration's 0.4576 (which itself swings 0.43→0.97 across nominally equivalent n) are three draws from the same noise. **Recommend dropping those rows or reporting total-variation distance instead** (0.1131 vs 0.0053, which does separate) | `dna_findings_audit.md` |
| Fig 4 left (value error vs N) | "strong scaling of the dMFM with N" | **Survives.** All 162 units collected. dMFM value error falls monotonically 4.80 → 0.247 over N=1 → 512, crossing the N-independent DPS baseline (flat 2.863) at N≈8–16. Caveat: the reference's own self-split floor is 0.324, so the N≥256 points sit **at or below the noise floor** of the comparison and should not be read as resolved | `results/dna/fig4/summary.json` |
| Fig 4 right (NFE comparison) | "dMFM achieves low target error at low NFEs, the regime most relevant for practical sampling" | **Survives at one operating point, not as a regime; the published axis flatters us.** Backward cost is now *measured* on an H200: k = 1.1 forwards for DNA (flat in batch width), 2.89–3.00× for the whole QM9 guided step. Every generative forward the dMFM value estimator makes is in-graph, so its curve shifts right while the search baselines do not. Feynman–Kac crossover **582 → 428** fwd-equiv; win region **two of five** settings → **one (MC=1)**, which is its floor. The surviving win is real: at 287 fwd-equiv, 0.133±0.013 vs FK 0.198 interpolated (z=−9.5) — and on the fair axis this point becomes comparable at all, where at the published axis it sat below every baseline and recorded NaN | `nfe_accounting_audit.md` |
| Fig 4 right, **wall-clock** | caption concedes NFE "does not establish equal wall-clock" | **No wall-clock regime where dMFM wins**, measured on one H200 with checkpoints resident. Matching batch width is worth 3.5–3.9× to the dMFM (MC=1: 3.657 s → 0.933 s), so the stored seconds were both load-inflated and width-handicapped and the naive "13× slower" reading is wrong. But even corrected, baselines lead at every matched second: at 0.933 s, FK 0.035 / best-of-N 0.097 vs dMFM 0.133; at MC=16 no interpolation needed — FK 2.099 s for 0.011 against dMFM 2.631 s for 0.061. dMFM cannot move left (MC=1 is the floor). **Partly fixable**: the residual is per-call overhead (91 autograd calls plus 91 separate student forwards inside a 96-step trajectory), worst exactly where the claim lives; batching over MC and step would recover some of it, and nothing in the current code does | `nfe_accounting_audit.md` |
| Fig 4 protocol | as described in §3.3 | **Left panel is not the experiment the text describes**: synthetic `SmoothDNAReward` (hardcoded seed 1234), not f_cyc; teacher-SDE reference (512×1024), not GLASS. Confirmed against the recovered original producer. The four baselines were a hardcoded CSV with no producing code; they are now really implemented and tuned per method on a separate pilot | `results/dna/original_scripts/` |
| Table 2 multiMFM (QM9) | beats TFG-Flow on most properties | **Holds**, no port bug | `mismatch_audit_molecules.md` |
| Table 2 multiMFM-SS | best row | **Holds.** Real 9–27% oracle gain, significant paired on all six (t=4.5–36.9). μ/Δε/LUMO are clear wins; Cv/α/HOMO (−2 to −5%) are **ties** and should be reported as such. The 20× guide/oracle spread is winner's curse from best-of-~3680, and the paper only ever published the oracle metric | `ss_guide_oracle_gap.md` |
| **Table 3 (GEOM)** | 0.227 Tanimoto, 87.5% PB, "off-the-shelf" guide | **Still does not survive.** 4,096×3: gain **−0.00031 ± 0.00065**. Published figure is best-of-4-grid at n=64, one seed; guide head was trained by us, so "off-the-shelf" is false. **Coordinate guidance is now more firmly negative**, not less: all 7 pre-registered regime arms fail, and with exact gradients on real molecules over 3.5 decades the best is +0.0034 ± 0.0019 (1.8σ) while the reward rises +53 | `geom_rescue_attempt.md` |
| GEOM, three real defects found | — | **(1) The sampler annihilates guidance.** Both interpolants contract onto the endpoint prediction with coefficient 1 at t=1, so a perturbation injected anywhere in the guided window survives with magnitude **0** (coords) / ≤3e-11 (atom types); only the last step survives. Guidance acts *only* through the network's response. So the earlier "0 argmax flips in 420 atoms" is a defect of **where guidance is applied**, not a property of the task. **(2) The paper's reward is the wrong surrogate**: Spearman with counted FP2 Tanimoto is 0.36–0.38 for `−‖p−c‖²` against 0.67–0.71 for a differentiable Tanimoto. **(3)** The earlier "FP2 is invariant below 0.006 Å" reading was wrong — the gradient beats a matched isotropic control by +0.025–0.045 (2.3–3.4σ) | `geom_rescue_attempt.md` |
| GEOM, the composition "positive" — **REFUTED at n=256** | — | The Tanimoto gain is real and large but is **bought entirely by destroying molecular validity**. Job 49437876 (n=256, base Tanimoto 0.2348, **PB 0.8789**): the best arm `head_exact/tanimoto/round4` reaches +0.1005 Tanimoto at **PB 0.1523 (−0.73)**; even one round of the milder L2 arm costs +0.0351 for **PB 0.3633 (−0.52)**. Decisively, **random substitution destroys validity just as fast** (round1 PB 0.3672), so the collapse is caused by the substitution operator itself, not by the head's choices — swapping an element while holding the 3D geometry fixed leaves bond lengths and angles wrong, and PoseBusters catches it. The earlier n=24 reading (PB 0.750→0.667, ±9 pp) was simply underpowered. **Correct statement: the composition channel carries FP2 signal, but no post-hoc substitution scheme can exploit it while keeping molecules physically valid.** This strengthens the GEOM negative rather than rescuing it | `results/geom/rescue/terminal_generated_seed0_cnof_n256.json` |
| GEOM benchmark itself | implied meaningful Tanimoto scale | **Two corrections against us.** Random-pair FP2 Tanimoto on geom_fp val is **0.2164** (all) / 0.2099 (≤30 atoms, TFG-Flow's regime), so our unguided 0.2278 *and* TFG-Flow's guided 0.208 are **both at the floor**. And TFG-Flow averages over all samples, scoring invalid ones 0 and mapping unresolved MASK atoms to hydrogen — so **"our base beats their guided" is unsound and must be dropped**. Headroom is real though (median best-of-64 = 0.4127, best-of-4096 = 0.6350), so the benchmark is not impossible | `geom_rescue_attempt.md` |
| **GEOM: the salvageable contribution** | — | A sharp, well-evidenced critique of the benchmark itself. **Over half the attainable headroom (+0.100 of +0.18) can be taken by relabelling ~15% of atoms, leaving 85% of molecules PoseBusters-invalid.** So a structural-fingerprint Tanimoto is trivially gameable by a method that never touches geometry, and **any number on this benchmark must be reported jointly with a validity check — which TFG-Flow's Table 3 does not do.** This reframes our negative: the task as scored does not measure what it claims to. Supporting nuance in the method's favour: our guide head beats random substitution on *both* axes (+0.053 vs −0.014 metric, 0.582 vs 0.367 validity) and captures 62% of the oracle ceiling, so the head's ranking ability is real even though the edit is not a usable method | `geom_rescue_attempt.md` §6 |
| Tables 21, 22 (QM9) | PB validity, uniqueness | SS row **mixes replicates** a/b under four different rules; Table 21 mu 96.0 is replicate b alone (mean = 95.80) | `mismatch_audit_molecules.md`, `compare_tolerance_flips.md` |
| Table 7, 9, App. F.6, App. E.1 | hyperparameters | Weights **1.0/1.0**, not 0.35/10. t\_eval **0.5**, not 0.3. L400 batch 8, not 16. "10 hours" is an **aggregate** (11.16 GPU-h over ~3.9 h on 4 GPUs) | `mismatch_audit_dna.md` |
| Fig 1, Table 12 (toy) | multiMFM toy | Code and checkpoints were purged; **rebuilt** and retrained. Fig 1's PDF predates the toy MFM, so the published figure may show plain GLASS | §7.6 |

## New, and not in any earlier audit: the stated reward is not the one optimised

Eq (17) and App. F.4 give the DNA reward as `r = -(f_guide(x) - y*)^2`. The code
(`api.c0_log_reward_fn`) exponentiates **`-11.1 (f - y*)^2`** — from `reward_sigma 0.15` and
`reward_scale 0.5`, which appear in **every** DNA C0 `args.json`, including the submitted pilot.
Inside `log E[exp(r)]` that constant is an **inverse temperature**: it re-weights the posterior
expectation and therefore changes the gradient *direction*, so it cannot be absorbed into the tuned
guidance strength the way a plain scale factor could. The results stand; the paper's equation does
not describe them. A one-clause disclosure fixes it (edit E16 in `paper_text_dmfm.md`).

**This is worse than a disclosure gap (corrected 2026-09-30).** The constant does not merely restate
the reward, it changes a reported conclusion. Effective sample size of the N=8 importance weights at
t=0.5 is **1.09 (y\*=+1) and 1.98 (y\*=−1)** under the implemented −11.111 coefficient, against
**5.43 / 6.58** under Eq (17)'s. So "N = 8 Monte-Carlo draws" overstates the averaging by 3–7×, and
the operating point should be reported as **"N = 8, ESS 1–3"**. Worse, the published guidance
strength **cannot be re-derived from Eq (17)** by tuning `guidance_frac`: the gradient scales 10–16×
with the coefficient and `coeff_cap=10` already saturates for t ≤ 0.615. Separately, the
MC-convergence tables run a *different* reward again (−(z−1)² on standardised GC), at ESS 5.06.

Relatedly, **eq (1) does not describe the DNA runs either**: the implemented coefficient is 10-16x
the stated sigma^2/2 over t in [0.5, 0.95] (exactly 16x for t >= 0.615), and under the scale-free
rule the step magnitude no longer depends on ||grad V|| at all.

## QM9's split is already optimal — Table 2 does not change (2026-09-30)

The DNA insight was tested on QM9 and **does not transfer**. Pre-registered on `alpha` (the only
property the baseline ladder covers), at matched generative NFE 14,848/molecule, n=100:

| split | multiMFM | SS | vs 4x32 (multiMFM / SS) |
|---|---|---|---|
| **4x32 (published)** | **1.112**, PB 0.86 | **0.796** | — |
| 2x64 | 1.583 | 1.341 | +0.471±0.163 / +0.545±0.122 |
| 8x16 | 1.734 | 1.035 | +0.622±0.239 / +0.238±0.106 |
| 1x128 | 2.835 | 2.100 | +1.723±0.311 / +1.304±0.211 |

4x32 is the argmin on **both** rows (2.2–6.2 SEM) and the argmax of PoseBusters validity,
reproducing on three held-out seeds and a second property. **Table 2 at the optimum is Table 2 as
it stands**, and no rerun was needed.

**The principle, corrected.** It is not "fewer steps, more draws" — it is **match the split to the
split the student was distilled for**. QM9's student was distilled for a 4-step diagonal rollout
(`rollout_steps 4, rollout_diagonal True`), so a 1-step query is a dt=1 Euler step it never saw and
its guide MAE doubles (0.700 → 1.367). On DNA the shipped 4-step student was being composed at 4
steps but the *diagonal one-step* artifact matched that task's 1-step use. Same rule, opposite
answer. Verified not to be a dose artifact: gradient RMS is 1.41 / 0.75 / 1.49 / 5.44, so 1x128
carries the same dose as 4x32 and is still far worse.

Three things the authors should know: **(a)** the estimator is **budget-saturated and guide-limited,
not compute-limited** — doubling the budget to 4x64 moves multiMFM 1.112 → 1.062 and SS 0.796 →
0.805, and SS's guide error is already 0.001; **(b)** MC-heavy splits are matched on *generative*
NFE only — regressor calls go 7,471 → 29,551 and wall clock **33 s → 1,003 s**; **(c)** SS's
reported PB is 1.000 throughout, but its *unfiltered pool* collapses 0.71 → 0.18, which plain
multiMFM shows directly as 0.86 → 0.69. A lead worth following: querying the same checkpoint as a
genuine two-time flow map (no `--mfm-diagonal`) gives PB 0.91 vs 0.86 at an MAE cost inside noise.

## The (steps, MC) operating point, swept across every DNA result (2026-09-30)

The Table 1 discovery generalises, but narrowly and cleanly. The point was chosen **once** on the
Table 1 C0 task (dMFM 1x32, GLASS 2x16) and applied unchanged elsewhere; where a grid preferred
otherwise that is reported, not adopted.

| item | current split | at the operating point | significant |
|---|---|---|---|
| Table 1 / Fig 3 | 4x8 | 0.193/0.172 -> **0.146/0.120** | yes, -0.044±0.015 / -0.061±0.012 |
| **Fig 7** (tilt ladder) | 4x8 | guide MAE 0.120 -> **0.043**, within-0.10 0.57 -> **0.88** | **yes, 3.1σ** on the plotted quantity (0.9σ on oracle) |
| **Fig 8** | 4x8 | 0.172 -> **0.076** | marginal, 2.3σ |
| **Table 13** | 1x8, clip-down | split already optimal; **applying the scale-free rule gives +0.109 -> +0.124** | yes, 3.7 SEM |
| **Table 18** (gradient MAE) | 4x8 | `motif`: **0.1636 / cos 0.139 -> 0.0479 / cos 0.318**; `gc`: cos 0.212 -> **0.327**, MAE unchanged | **yes on `motif`** (published 4x8 is the *worst* dMFM split on both columns; 1x32 is 3.4x better on MAE), direction-only on `gc` |
| Fig 4 right | 1xMC | already optimal (1x16 beats 2x8 by 3.4σ) | no |
| Table 17 | no value gradient | N/A | no |
| Fig 5 / Tables 14-15 | MC is the x-axis | N/A | no |

**Only Table 1/Fig 3 was at the wrong split**; the correction propagates to Fig 7, Fig 8 and
Table 18's `motif` column.
A coherence gain follows: every dMFM DNA result then uses the diagonal one-step student, so the
paper describes one artifact rather than two.

**NEW PROBLEM — Table 18's reference is numerically degenerate.** Its GLASS-128 RK4-50 reference at
`end_time 0.999` has gradient norm **0.006-0.090** against estimates of 0.18-7.2 on `gc`
(`corr(MAE, estimate-norm) = 0.95`) and **0.0055-0.0352 against 0.26-10.0** on `motif`, where
that correlation is **0.995**. Confirmed on both rewards, so it is not `gc`-specific. So the printed MAE mostly measures gradient *magnitude* rather
than accuracy, and the split it nominates as best (8x4) is the one Table 13 shows does not steer
(9/32 pairs improved). Confirmed on the motif reward too. This needs a **metric** decision, not a
split decision — and note it is the same degenerate-reference failure mode as the withdrawn
`glass_euler32` claim below.

## WITHDRAWN 2026-09-30: "a well-integrated posterior does not guide at all"

This was **our error**, refuted end to end, and it was the claim I had been calling the paper's most
reviewer-exposed. It rested on a single arm, `glass_euler32`, which had two independent defects:

1. It integrates 32 Euler steps to the **singular data time t=1** — the degenerate family the parity
   document had itself discarded elsewhere. Measured: the endpoint is a simplex vertex (argmax margin
   exactly 1.00000), V is piecewise constant (33–100% of random directions leave V bit-unchanged,
   against 0% for GLASS-4), and the true fp32 gradient norm is 1.2e-5. The gradient was numerically
   dead, not small.
2. It was the **only** arm never given the §7-item-8 scale-free rule, so its step was thousands of
   times smaller than every other arm's.

End to end (n=8, y\*=−1, identical noise and MC pool, unguided 0.8651, job 49434328): the published
arm gives **0.8651 with 0/8 improved**; the `end_time` fix alone 0.6014; the scale-free rule alone
0.7530 (4/8, a coin flip); **both fixed, 0.1939 with 8/8 improved** — against 0.1548 for the Table 1
4-step estimator, a paired difference of +0.039 ± 0.064 (0.6 SEM) achieved on a **25× smaller
gradient**. Independently (job 49442813, pre-registered sign test): the claim's own reference
integrator, GLASS RK4-25 at t=0.999 with gradient norm 0.202, delivers **89%** of the coarse
estimator's per-step benefit; −grad is negative and both random controls are zero.

**Correct statement:** the guidance constants are not transportable between posterior samplers
without renormalisation — which is precisely why the scale-free rule exists, and it was never applied
to this arm. §7.9's headline and App. F.4's paragraph must not ship as written.

My earlier cross-domain framing was also wrong: the QM9 DPS sweep refines the *outer* guided ODE
(`mu·grad·dt`, dt-invariant) while `glass_euler32` refines the *inner* value estimator at fixed
`nfe_traj`. They are different limits and neither supports the other. The matched DNA arm (outer
refinement, `nfe_traj` 64→512) is job 49440315, pending.

## Findings that survive

1. **The DNA guidance signal rides on discretisation error.** A well-integrated posterior does not
   guide at all: `glass_euler32` leaves guided MAE at the unguided value (1.1905 vs 1.1907), and the
   well-integrated reference value gradient has norm ~0.15 against 3–125 for every estimator
   actually used. The operating point must be described as **N=8 at 4 steps**, not as an
   asymptotically accurate value-gradient estimate. This is the paper's most reviewer-exposed claim.
   **WITHDRAWN 2026-09-30 (same evening): the QM9 DPS evidence was an artifact and does not
   support this.** DPS was extended along `sample_steps` and degraded monotonically (MAE ~2.82 at
   128 steps to 5.43 at 4096, 3 seeds), which was reported as "DPS cannot use more compute". That
   reading is wrong. `CenteredMetricInterpolant.step` is a contraction,
   `x_new = x_t(1 - dt/(1-t)) + x1_pred(dt/(1-t))`, so a perturbation injected at time t reaches the
   endpoint scaled by ~dt/(1-t). DPS injects `mu*grad*dt`, so a single injection contributes ~dt^2
   and the whole window ~N*dt^2 = **1/N**. Verified numerically: relative to 128 steps the delivered
   guidance is 0.250 / 0.125 / 0.0625 / **0.0312** at 512 / 1024 / 2048 / 4096. The sweep therefore
   did not measure DPS with more compute; it measured DPS with up to **32x less guidance**, and the
   monotone degradation is that dose reduction. So DPS's absence from the high-NFE region is a
   handicap we imposed, not a property of the method. The dose-matched rerun (mu proportional to
   sample_steps, job 49455395) will show whether any degradation survives. This is the same root
   cause as the withdrawn DNA claim above and as the GEOM annihilation defect: **guidance constants
   are not transportable across step counts or samplers without renormalisation.**
2. **GEOM's task is ill-posed for coordinate guidance.** FP2 is a *graph* fingerprint: exactly
   invariant below 0.006 Å, randomised by 0.2 Å. There is no displacement scale that responds
   smoothly, so there is no operating window. The guidance gradient itself is **verified correct**
   five independent ways, so this is not a bug. λ=1 — the paper's literal reward, never previously
   run — collapses PB validity from 88.3% to 14.5%.

## Decisions needed from the authors

1. **Table 3.** Reframe as a negative result *plus its mechanism* (coordinate guidance cannot steer
   graph topology — a genuine finding about the task, not a failure of multiMFM), or drop it. It
   cannot be defended as a win.
2. **Disclose the discretisation mechanism** in the method or limitations section, and state the two
   mandatory conventions (4-step ESD student; scale-free gradient rule) wherever the DNA setup is
   described. Draft text is being prepared in `paper_text_dmfm.md`.
3. **Table 2 SS:** report three wins and three ties rather than six wins.
4. **Co-author sign-off** on §7 items 1–8.

## Still running

Table 1 with dMFM (n=1000, 6/10 + 4/10 shards done), Fig 5 / Tables 14–15 with dMFM, Fig 7, Fig 8,
Tables 4–6 from dMFM samples, Fig 4, and one top-of-grid GLASS point. All are **queue-bound**, not
blocked on science: the cluster is saturated (`kempner_h100` 0 idle with ~48 ahead;
`serial_requeue` 7,376 pending; `albergo_lab` scores priority 4 on `gpu_requeue`). Do not resubmit
to another partition — it forfeits queue position. See `RERUN_STATUS.md`.
