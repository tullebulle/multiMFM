# DNA (dMFM) port: verification record

Date 2026-09-29. Everything below can be re-run from the repo. The commands are listed with each
check. "Stored" means the result files of the original runs: copies are under `results/dna/`, and
the originals are in the holylabs backup `paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir/`.

## Summary

| Check | Result |
|---|---|
| Every `dmfm` module imports; every checkpoint loads on CPU with no missing/unexpected keys | pass (`tests/dna/test_dmfm_port.py`) |
| Table 11: C0 guide/oracle recomputed from the shipped weights on the 58 held-out parents (8,294 windows) | matches stored `test_metrics` and the paper to <= 2e-6 |
| Forward-pass parity vs the original code (seq_h100: Python 3.9, torch 2.1) on CPU, 96 tensors | single network evaluations agree to <= 1.2e-6 abs (fp32 round-off); chained samplers to <= 3e-5 rel; see note on 4-step RK4 GLASS |
| Unguided Table 1 samples vs the stored n=1000 run | token-identical: 8/8 on CPU, 8/8 on GPU (guide score 5e-6, oracle 1e-6); all 1000 via the dMFM Table 1 run (below) |
| Unguided Table 13 samples + calibration vs stored | 4/4 sequences identical, scores identical, calibration draws identical (GPU) |
| `dmfm.api` equals the experiment code paths | exact (`tests/dna/test_api.py`) |
| `sample_c0_guidance_dmfm --paired --posterior glass` equals the Table 1 sampler | identical sequences and scores (CPU) |
| Every run wrapper parses and reproduces the original run's arguments | identical for Tables 1, 11, 13, 17, 18, Fig 5 / Tables 14-15, all training runs, data prep (`tests/dna/test_script_args.py`, `test_dmfm_port.py::test_training_wrappers_reproduce_original_args`) |
| Table 18 MAE rescore (`rescore_gradient_pairs_mae`) on the stored gradient pairs | reproduces the stored CSVs exactly (max \|diff\| 0, 3 rewards x 4 lengths) |
| GPU smoke: guided sampling + oracle, 20-step training of base DFM / dMFM / 4-step student / C0, Table 13, Table 17, Fig 4 dMFM half | all ran (jobs 49264546, 49266706) |
| dMFM training objective (one `step_both`) vs the original code, CPU | loss and sampled times bit-identical, gradients <= 1.4e-6 (section 5b) |
| Ported library vs DNA-MFM@187fe7b (durable clone), per file and per function | reconstruction byte-identical except 2 documented files; port = commit up to packaging / dead-branch removal (section 5c) |
| Table 1 dMFM rerun: 4-step student at 4 steps, both arms scale-free | complete at n=1000: dMFM guided MAE 0.193 (+1) / 0.172 (-1) vs matched GLASS 0.153 / 0.124, paired +0.041/+0.049 +- 0.006. The first attempt (49277977, diagonal student at 4 steps + stock clip) is **withdrawn**, 0.610/0.408 (section 6) |
| GPU gradient check vs stored Fig 5 / Table 18 gradients | Fig 5 agrees to 0.3-2% (better with `glass_end_time=1.0`); Table 18 GLASS-128 RK4 reference differs by 33% (job 49264548); explanation below |
| `checkpoints/MANIFEST.json` group dna | 41 files verified (`scripts/manifest.py verify --group dna`) |

## 1. Table 11 / Fig 3 caption: C0 guide and oracle

`python tests/dna/test_dmfm_port.py` (`test_c0_regressor_metrics_match_training_metadata`) loads
`checkpoints/dna/c0/{guide,oracle}/best_state.pt` with `dmfm.regressors.load_c0_regressor`. It scores the
parent-disjoint `test` split of `data/dna/yeast_parent_disjoint/yeast_parent_L50.pt` (58 parents,
8,294 windows) with reverse-complement averaging and compares against the measured C0:

| | MAE | RMSE | Pearson | stored `metadata.json` (MAE / RMSE / r) | paper Table 11 |
|---|---|---|---|---|---|
| guide | 0.158771 | 0.230643 | 0.877414 | 0.158772 / 0.230645 / 0.877412 | 0.1588 / 0.2306 / 0.8774 |
| oracle | 0.161985 | 0.234695 | 0.872416 | 0.161985 / 0.234694 / 0.872417 | 0.1620 / 0.2347 / 0.8724 |

The remaining differences are fp32 round-off from CPU vs the original GPU. The dna-notebooks agent
independently recomputed the same values from `test_predictions.csv` (notebook 02).

## 2. Forward-pass parity against the original code

Original side: the DNA-MFM@187fe7b library from `scratch_dfm_recon_20260925` (model, utils, mfm,
lightning_modules), overlaid with the surviving July files (`utils/mfm_diag_distill.py`,
`utils/parsing.py`, `dinko/c0_regressors.py`) and the surviving original scripts
(`ablate_glass_gradient_mc.py`, `ablate_dmfm_one_step_gradient_mc.py`, `probe_dna_mfm_glass.py`,
`sample_yeast_c0_guidance_hist.py`). It ran in the original env `seq_h100` (torch 2.1.0). Port side:
`dmfm` in `multimfm` (torch 2.5.1). Both ran on CPU with the same seeded inputs and the shipped checkpoints.
Harness: `tests/dna/parity/` (`build_orig_tree.sh` assembles the original tree from the holylabs
backup; its header has the commands; the raw comparison is `result_2026-09-29.txt`). 96 tensors were compared:

| Quantity (L = 50, 100, 200, 400) | max \|orig - port\| | max rel |
|---|---|---|
| base DFM logits, drift `v`, one flow step | 1.2e-6 | 4e-7 |
| dMFM and 4-step dMFM: velocity `v(s,t)`, two-time map | 9.5e-7 | 2.2e-7 |
| dMFM: one-step map, 4-step composed flow map (end 0.999) | 3.0e-5 | 3.0e-5 |
| GLASS posterior, 4 Euler steps (end 0.999) | 2.7e-6 | 2.7e-6 |
| GLASS value + gradient (gc / motif / conjunction rewards, MC 3) | 2.7e-6 | 2.0e-6 |
| dMFM value + gradient (motif, 4-step flow map, MC 3) | 2.6e-4 | 9.7e-5 |
| C0 guide / oracle forward | 6e-8 | 1e-7 |
| Table 1 value + gradient (`estimate_v_and_grad`, MC 8, 4-step GLASS) | 4.9e-5 | 3.0e-6 |
| GLASS posterior, **4 RK4 steps to 0.999** | 2.3e-2 (L400) | 1.1e-3 |

Every single network evaluation agrees to fp32 round-off (torch 2.1 vs 2.5 kernels). Composed
samplers amplify that round-off. The coarse 4-step RK4 GLASS integrator ending at 0.999 amplifies it
most (~1e3x). This happens because the GLASS velocity becomes stiff as s -> 1 (the terms scale like 1/(1-s)).
This is float sensitivity, not a code difference. The same code gives the same numbers to ~1e-7
wherever the computation is well conditioned.

The C0 guide class of the original runs was `dinko.torch_model.C0Torch` applied to
`dna_probs_to_park_input(x)`. The recon copy of `C0Torch` has the same layers and forward as
`ParkC0Regressor`, which folds the ACGT->ATGC permutation into `forward`. Rescoring the 64 sequences of
the surviving dMFM C0 run (`yeast_parent_dmfm_matched_*`) with `ParkC0Regressor` reproduces its stored
C0Torch guide scores to <= 1.1e-3, i.e. TF32 precision.

The port's `sample_c0_guidance.py` also matches a verbatim copy of the 187fe7b file
(`results/dna/original_scripts/sample_yeast_c0_guidance_hist.py`, from a DNA-MFM clone) up to packaging
and that same C0 class substitution. Per-module origins and sha256s: `docs/provenance/dna/port_sources.json`.

## 3. Unguided sample statistics vs stored runs

- **Table 1**: the unguided branch is deterministic (per-sample CPU-seeded noise). Samples 0-7 of the
  stored n=1000 run are reproduced token for token on CPU (`tests/dna/test_api.py`; guide score <= 7.8e-4,
  which is CPU vs GPU-TF32) and on GPU (job 49264546: 8/8 identical, guide score 5.4e-6, oracle 1.4e-6).
  The dMFM Table 1 run regenerates all 1,000 unguided samples per target. The collector
  compares them with the stored GLASS n=1000 run (`_unguided_vs_stored_glass_n1000_run` in
  `results/dna/c0_guidance/table1_dmfm_n1000_summary.json`), with the result in section 6.
- **Table 13**: GPU smoke (job 49266706, L=50 motif, 4 pairs): unguided sequences 4/4 identical to
  `rebuttal_reward_attainment_motif_20260731/L50`, unguided scores identical (max \|diff\| 0), and the 64
  unconditional calibration draws identical to the stored `unconditional_score_calibration.npy`. The
  target score differs only because the smoke used 64 instead of 1,024 calibration draws.

## 4. GPU smoke test (gpu_test)

`scripts/dna/smoke_gpu.sbatch` with `PARTS` split over jobs (logs in `logs/dna_smoke_{a,b,c}-*`):

- 49264546 (`sample train nfe`, A100 MIG): C0-guided sampling at L=50, 8 pairs, target +1. Oracle:
  unguided MAE 1.135 -> guided 0.147, paired improvement 0.988 (8/8 improved). Unguided check
  `UNGUIDED_CHECK_OK`. 20 training steps each of the base DFM (loss 1.386 -> 1.298), the dMFM (diag +
  ESD), the 4-step student from the shipped L50 dMFM (loss ~0.6) and the C0 regressor (2 epochs). All
  finite. Fig 4 dMFM half: ran.
  The `reward` and `posterior` parts failed on the old `data/yeast_parent_disjoint` path bug. The bug
  was fixed in `sample_reward_attainment.py` and `eval_posterior_diversity.py`, and both parts reran in 49266706.
- 49266706 (`reward posterior`): Table 13 (L=50 motif, 4 pairs) and Table 17 (L=50, 2x4) complete.
- 49264548 (`gradmc`): see section 5.

## 5. Gradient-MC reruns against the stored gradients

`tests/dna/check_gradient_mc_reference.py` (job 49264548) reruns the first two probes of stored runs on
GPU with their recorded arguments:

| case | reference-gradient rel. error | estimate rel. error |
|---|---|---|
| Fig 5 motif L50, `glass_end_time=1.0` | 0.0063 | 0.022 |
| Fig 5 motif L50, `glass_end_time=0.999` | 0.0118 | 0.042 |
| Fig 5 gc L50, 1.0 | 0.0033 | 0.0028 |
| Fig 5 gc L50, 0.999 | 0.0057 | 0.026 |
| Table 18 motif L50 (GLASS-128 RK4-50 reference / 4-step dMFM) | 0.33 | 0.020 |

Calibration statistics and probe states match exactly (score center/std rel <= 4.6e-4, identical
probe indices). The Fig 5 runs are consistently closer with `glass_end_time=1.0`, which supports the
pre-flag behaviour the wrapper uses. The residual 0.3-2% and the Table 18 reference gap come from
GPU/TF32 round-off amplified by the samplers. Section 2 shows the same code agrees to 1e-6 on CPU, and
that RK4-GLASS to 0.999 amplifies round-off ~1e3x. The Table 18 reference chains 50 RK4 steps
(200 network calls), and its motif reward (score std 0.1, softmax over 128 samples) is sharply
peaked. Job 49278666 runs the original code and the port on the same GPU, with TF32 on and off, to
separate "code" from "hardware". Result: see section 7.

## 5b. dMFM training objective: original vs port

`tests/dna/parity/parity_train.py` runs one `GaussianMfmDiagDistiller.step_both` (diagonal + ESD
objective, teacher = shipped base DFM) on a fixed batch with the args of the shipped L50 dMFM, the
L50 4-step student and the L200 dMFM. It uses the original code (seq_h100) and the port, on CPU.
The loss, the sampled times (s, u, t_cond) and the diagonal/ESD loss terms are **bit-identical**, and
the student gradients agree to <= 1.4e-6 (rel 2.7e-7). So the objective, and its order of random-number
draws, is unchanged (`result_train_2026-09-29.txt`). `train_dmfm.main` differs from the July
`scripts/train_dna.py` only in packaging (argv, W&B defaults, `torch_load`); the trainer, callbacks and
data split are the same.

## 5c. Per-file comparison with DNA-MFM@187fe7b

Reference: the durable read-only clone `paper_backup_dmfm_20260929/DNA-MFM_187fe7b` (`git log`:
`187fe7b Add yeast diversity and Evo2 validation`).

**The reconstruction (`scratch_dfm_recon_20260925`) vs the commit (sha256, first 16 hex):**

| file | commit | recon |
|---|---|---|
| model/dna_models.py | 0b9f9871455bd5ce | identical |
| model/dit_seq.py | a7926feae9cb89a2 | identical |
| lightning_modules/dna_module.py | 13787330fbefb8fb | identical |
| lightning_modules/general_module.py | 69ad076a7873d1be | identical |
| utils/flow_utils.py | 647d813f4bcfa56d | identical |
| utils/dataset.py, esm.py, logging.py, yeast_splits.py, mfm_lib.py | 553c6f84..., 22376a0b..., a6e24ef9..., ed0d9784..., 1c4780cc... | identical |
| utils/mfm_diag_distill.py, utils/parsing.py | b8e18d37..., e5551db9... | identical (the ported files are the newer July versions, 407e7e29... / 2421505a...; expected) |
| mfm/src/mfm/losses/losses.py, losses/utils.py | 1c7bdc7e..., 1a278d89... | identical |
| dinko/torch_model.py (C0Torch) | d8b3b1be9737f940 | identical |
| scripts/sample_yeast_c0_guidance_hist.py | 866c189750f94270 | **differs** (36fb0eba...): the recon swaps `C0Torch` + `dna_probs_to_park_input` for `ParkC0Regressor` (5 lines). The architecture and forward are identical, see section 2 |
| mfm/src/mfm/models/base_model.py | **not tracked in the commit** | 3587dbe93c8bd9ea = byte-identical to the upstream MFM `mfm/src/mfm/models/base_model.py` in this repo (the commit relies on the vendored upstream library for it) |

So the plan's claim "reconstruction = 187fe7b file for file" holds for every library file except
the two above.

**The port (`src/dmfm`) vs the commit, compared function by function** (AST, ignoring docstrings,
comments and type annotations):

| port module | original | functions identical / in both | differences |
|---|---|---|---|
| models/dna_models.py | model/dna_models.py | 41/42 | `_posterior_velocity`: calls the vendored `extract_posterior_velocity` directly instead of via `utils.mfm_lib` import helper |
| models/dit_seq.py | model/dit_seq.py | 25/25 | - |
| lightning/dna_module.py | lightning_modules/dna_module.py | 8/19 | 11 unreachable methods dropped (Dirichlet/Riemannian/AR/distill inference, classifier guidance, fly-brain); the dMFM branch of `general_step` moved verbatim into `_mfm_step` (minus two duplicate debug-print channels); `load_model`/`__init__`/epoch hooks lose the dropped modes' branches. Numerics confirmed by 5b and the GPU smoke |
| lightning/general_module.py | lightning_modules/general_module.py | 4/8 | `wandb` imported lazily (`_wandb()`); logic unchanged |
| utils/flow_utils.py | utils/flow_utils.py | 3/4 | 27 non-Gaussian helpers dropped; `gaussian_denoiser_flow_step` single network evaluation (bit-identical outputs; `legacy_double_eval=True` = original) |
| utils/dataset.py, yeast_splits.py | same | 2/3, 1/2 | `torch.load` -> `torch_load(weights_only=False)` |
| utils/esm.py | utils/esm.py | 1/1 | - |
| utils/log_utils.py | utils/logging.py | 3/5 | no stdout `dup2` / `workdir/default/log.out` on import |
| _mfm/losses.py | mfm/.../losses/losses.py + utils.py | 1/2 + 5/5 | `extract_posterior_velocity`: the image-only CFG branches (`dmf`/`sit` with class labels) raise instead; the DNA path (`labels is None`) is unchanged |
| _mfm/base_model.py | upstream mfm base_model.py (sha 3587dbe9) | 13/13 | - |
| experiments/sample_c0_guidance.py | scripts/sample_yeast_c0_guidance_hist.py (866c1897) | 15/18 | `parse_args`/`main` (argv, repo-relative defaults, `torch_load`), `cyclizability_reward` (C0 class, identical forward) |
| utils/mfm_diag_distill.py | July utils/mfm_diag_distill.py (407e7e29) | 12/14 | `_sample_s_u` adds the opt-in `--mfm_legacy_gap_or_default` (default = July behaviour); `load_teacher` uses `torch_load` |
| utils/parsing.py | July utils/parsing.py (2421505a) | 0/1 | parser split into `build_train_parser()` + `parse_train_args(argv)`; flags/defaults verified against every shipped `args.json` |
| regressors/c0.py | July dinko/c0_regressors.py (b9c1648a) | 5/5 | + `load_c0_regressor` helper |

Experiment modules vs their July originals (the files the paper runs executed; they are newer than
the commit, and e.g. `train_dna.py`, `train_dna_dfm_dit.py`, `probe_dna_mfm_glass.py` differ from
the commit's versions): only `parse_args`/`main` (argv, repo-relative defaults, `torch_load`)
differ, except for `ablate_glass_gradient_mc.glass_integrate_diff` (also accepts `end_time=1.0` for
Euler, the pre-flag behaviour), `eval_posterior_diversity`/`sample_reward_attainment` (data-path fix),
`rescore_gradient_pairs_mae` (split into functions + `--lengths`; outputs identical) and
`analyze_c0_diversity` (the vendored-module loader replaced by a package import). No module needed
re-porting. Origins and sha256s of every source file: `port_sources.json`.

## 6. dMFM Table 1 (the run the paper now reports)

**Withdrawn: job 49277977.** The first n=1000 "Table 1 with dMFM" run composed **four** posterior
flow-map steps of `checkpoints/dna/dmfm/L50`, the *diagonal* student that the rest of the paper uses
as a **one**-step map, instead of `checkpoints/dna/dmfm_4step/L50` (distilled on gaps [0, 0.25],
selected on `val_mfm_probe_esd_gap_0.25`, which is Table 18's 4-step artifact). It also used the
stock clip-down guidance rule, under which a dMFM posterior is driven 5-12x harder than GLASS,
because `grad_clip` only clips down and Table 1's constants are calibrated to the GLASS-4 gradient
norm. Oracle-scored at the full n=1000 (collected 2026-09-29) it gives guided MAE **0.610 (+1)** and
**0.408 (-1)**, against the GLASS n=1000 run's 0.148 / 0.119, confirming the parity agent's n=64
estimate (0.632 / 0.380). Its arrays stay on disk in `outputs/dna/table1_dmfm_n1000` as provenance
and are recorded under `_withdrawn_runs` in the summary; they must not be printed as Table 1
(`REPRODUCIBILITY_PLAN.md` §7 item 8, `docs/dmfm_glass_parity.md`).

A useful by-product of that collection: **1000/1000 unguided sequences per target are identical** to
the stored GLASS n=1000 run (guide score max |diff| 2.9e-4). The unguided branch does not depend on
the posterior sampler, so this confirms the harness and isolates the difference to the guidance term.

**The rerun.** Both arms use the 4-step artifact where a student is involved and the *same*
scale-free rule (`grad_normalize`: rescale the value gradient to exactly `grad_clip` rather than only
clipping it down), with `guidance_frac 8` and everything else at the Table 1 values:

```bash
GRAD_NORMALIZE=1 POSTERIOR=dmfm NFE_VALUE=4 sbatch --array=0-9%4 scripts/dna/table1_c0_guidance_dmfm.sbatch
GRAD_NORMALIZE=1 POSTERIOR=glass            sbatch --array=0-9%4 scripts/dna/table1_c0_guidance_dmfm.sbatch
bash scripts/dna/collect_table1_dmfm.sh <dmfm jobid> <glass jobid>
```

Protocol checks that still hold: `--posterior glass` through this sampler reproduces the Table 1
sampler exactly on CPU (`tests/dna/test_api.py`), and the wrapper's arguments equal the stored GLASS
`args.json` for both targets (`tests/dna/test_script_args.py`). Both conventions are now the wrapper's
defaults and are asserted by `test_script_args.py::test_table1_dmfm_guidance_rule_defaults_safe`:
omitting `GRAD_NORMALIZE` gives the scale-free rule with the 4-step student, `GRAD_NORMALIZE=0` warns,
and `LEGACY_DIAG_STUDENT=1` reproduces withdrawn job 49277977 in full.

**Final result** (2026-09-30, all 20 shards complete, n = 1000 per target per arm;
`results/dna/c0_guidance/table1_dmfm_n1000_summary.json`, `_status: complete`, oracle-scored):

| arm | y* | unguided MAE | guided MAE | paired improvement | frac improved |
|---|---|---|---|---|---|
| **dMFM, 4-step student, scale-free** | **+1** | 1.174 [1.149, 1.200] | **0.193 [0.184, 0.204]** | 0.980 [0.953, 1.009] | 0.975 |
| **dMFM, 4-step student, scale-free** | **-1** | 0.833 [0.806, 0.857] | **0.172 [0.164, 0.180]** | 0.660 [0.633, 0.687] | 0.954 |
| matched GLASS-4, scale-free | +1 | 1.174 | 0.153 [0.145, 0.160] | 1.021 [0.994, 1.048] | 0.987 |
| matched GLASS-4, scale-free | -1 | 0.833 | 0.124 [0.117, 0.130] | 0.709 [0.682, 0.735] | 0.965 |

Per-sample paired difference, dMFM minus matched GLASS, n = 1000 at each target:
**+0.041 (SEM 0.006)** at y*=+1, CI [0.029, 0.053], dMFM better on 43.7% of pairs; **+0.049 (SEM 0.006)**
at y*=-1, CI [0.038, 0.060], dMFM better on 40.6%. So **dMFM is consistently ~0.04-0.05 behind the
matched GLASS arm at both targets** (6-9 SEM), not at parity: the n=64 reading of a ~1 SEM tie, and the
preliminary y*=+1-only figure, were the small-sample versions of the same effect. For scale, both arms
reduce oracle MAE from 1.17 / 0.83 to 0.12-0.19, and the published stock-rule GLASS n=1000 gives
0.148 / 0.119.

Controls that held at full n: every arm's unguided column is bit-identical to the stored GLASS n=1000
run (**1000/1000 sequences at both targets**, guide score max |diff| 2.9e-4), and
`unguided_identical_across_arms` is true on both paired intersections, so the whole difference is
attributable to the guidance term. All dMFM and GLASS shards came from the kempner_h100 arrays
(49352242 / 49352244); the kempner_requeue hedges never started and cost nothing.

**The reward is not eq (17).** Every DNA C0 run, including the submitted pilot, uses
`reward_scale * -0.5((f - y*)/reward_sigma)^2 = -11.111 (f - y*)^2` at `reward_sigma 0.15`,
`reward_scale 0.5`. Inside `V(x) = log mean_j exp(r_j)` that constant is an inverse temperature: it
reweights the MC average over posterior samples and therefore changes the gradient direction, so it
cannot be absorbed into the tuned guidance strength. A disclosure gap in eq (17), not a defect in the
runs; the summaries carry it as `_reward`.

**Caveat to carry into any statement of the result.** The guidance signal rides on the coarse
posterior's discretisation error, and the strength constants are calibrated against that error: a
*more accurate* posterior guides less. GLASS with 32 Euler steps leaves the guided oracle MAE at the
unguided value (1.1905 vs 1.1907 at y*=+1), and the well-integrated reference gradient (GLASS-25 RK4
to 0.999) has norm ~0.15 against 3-125 for every estimator used for guidance. This is a property of
the experiment as published, not of dMFM, but it means "use a better posterior sampler" is not a
route to better steering here, and the constants cannot be transported between samplers without
renormalisation. The summary carries this as `_caveat_posterior_accuracy`.

## 7. Pending at the time of writing (2026-09-29 17:50)

- (Cleared 2026-09-30.) The dMFM Table 1 rerun is complete: all 20 shards of
  `outputs/dna/table1_dmfm4_norm_n1000` (job 49352242) and `outputs/dna/table1_glass_norm_n1000`
  (49352244) landed overnight and were collected into
  `results/dna/c0_guidance/table1_dmfm_n1000_summary.json` (section 6). Re-collect at any time with
  `bash scripts/dna/collect_table1_dmfm.sh 49352242 49352244 49357778 49357782`; the shard tree, not a
  job id, is authoritative, and `run_env.json` records which array, partition and device produced each
  shard. The superseded `table1_dmfm_PRELIMINARY_partial.json` was deleted once this landed.
  Superseded and cancelled while pending: 49293510 (CPU route, would have tripped the new gap guard),
  49308261 and 49298862 (gpu_requeue duplicates of the withdrawn configuration).
- 49278666: same-GPU parity of the original code vs the port for the Fig 5 / Table 18 gradient cases
  (TF32 on/off). Outputs go to `outputs/dna/gpu_parity/49278666/{port_tf32,port_notf32,orig_tf32}`; compare each
  `*/table18_motif/L50/gradient_pairs.pt` and `*/fig5_motif/L50/gradient_pairs.pt` against each other and
  against `tests/dna/reference/gradient_mc_L50.json`.
