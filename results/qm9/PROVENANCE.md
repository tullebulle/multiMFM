# results/qm9 — provenance

QM9 property guidance: paper Table 2 (multiMFM and multiMFM-SS rows), Tables 19–22 (QM9
columns/rows), App. F.8–F.9. Notebook: `notebooks/07_qm9_property_guidance.ipynb`.
Collected on 2026-09-29 by `python -m multimfm.qm9_results collect|hparams|baselines`.

## Sources

All original records are in the purged `tabasco` repo; the copies used here were taken from the
holylabs backup (identical to the surviving netscratch copies; `diff -rq` clean on 2026-09-29):

- backup root `B = /n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929`
- original root `/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco` (`T` below)

The original drivers, manifests, collect scripts and the authors' notes are copied to
`docs/provenance/qm9/` (`codex_scratch/`, `original_src/`, `tabasco_docs/`).

Code equivalence (checked with `diff`): `src/multimfm/property_targets.py` =
`T/src/qm9_length_conditioned_guidance_experiment.py` (mtime 2026-06-19, i.e. unchanged since
before the 2026-07-25 runs) and `src/multimfm/steer_search.py` =
`T/codex_scratch/memory/best_of_lookahead.py` (mtime 2026-07-25 01:36, before the 14:18 launch of
the `k_*` runs), apart from import paths and repo-relative defaults (+ docstrings). Likewise
`glass_guidance.py` = `T/src/glass_guided_sampling_experiment.py` apart from imports. The other
modules they import (`qm9_dfm_smoke.py`, `train_pb_valid_curve.py`,
`glass_posterior_norway_style.py`, the tabasco library) were purged and cannot be diffed; the
per-molecule agreement of unguided samples in the reruns below is the evidence that they are
equivalent.

Checkpoints: `checkpoints/base_flow/model_step_100000.pt` (original
`T/outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt`),
`checkpoints/mfm_student/student_step_1000.pt` (original
`T/outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt`),
`checkpoints/regressors/{guide,oracle}_clf_ckpt.zip` (TFG-Flow QM9 regressors). Target histograms:
`cache/histograms/hist_<prop>_train_flow_100.json`, byte-identical to `T/cache/hist_*.json` (used by
the SS runs); the non-SS runs read `T/cache/qm9_property_histograms/<prop>_train_flow_bins100.json`,
of which only the alpha file survives (byte-identical); for all six properties the stored targets of
the non-SS and SS runs are identical (checked 2026-09-29), so the content is the same.

## multiMFM (non-SS) row — `runs/multimfm/<prop>/`

Full outputs (`summary.json`, `summary.csv`, `per_sample.csv`, `guided_step_logs.csv`) of the six
runs behind the printed row, from `B/tabasco/codex_scratch/check/search/<tag>/`:

| prop | tag | mu | reward_scale | finished (2026-07-25) | Slurm allocation | host |
|---|---|---|---|---|---|---|
| cv | r2_cv_mu2p80 | 2.80 | 1.5 | 00:35 | 34989466 | holygpu8a06502 |
| mu | r3_mu_mu0p78 | 0.78 | 8.638 | 00:42 | 34989311 | holygpu8a03203 |
| alpha | r5_alpha_mu3p40 | 3.40 | 0.3 | 00:59 | 34985247 | holygpu8a12301 |
| gap | r5_gap_mu0p17 | 0.17 | 12000 | 00:58 | 34988184 | holygpu8a12601 |
| homo | r3_homo_mu0p50 | 0.50 | 41500 | 00:43 | 34989391 | holygpu8a03504 |
| lumo | r2_lumo_mu0p44 | 0.44 | 12000 | 00:36 | 34989345 | holygpu8a03504 |

Command: `codex_scratch/check/search/run_prop.sh <prop> <mu> <rs> <tag>` launched by
`dispatch.sh round<k>.manifest` as `srun --jobid=<allocation> --overlap` steps (H100s). Fixed
arguments (all verified in each `summary.json["args"]`): seed 8, target-seed 2026, n = 1000,
128 sample steps, value sampler `mfm` (diagonal, 4 inner steps, 32 samples, value batch 0),
guide every step in t ∈ [0.05, 0.95], guide atomics, eps 1e-6, soft-atom temperature 0.25,
partition `train_flow`, 100 histogram bins, `--compute-pb` with `posebusters_no_strain.yaml`.
In-repo replay: `scripts/qm9/run_multimfm.sh <prop>` (identical arguments, in-repo checkpoints).

Selection: the (mu, reward_scale) grid (`multimfm_search_grid.csv`, 60 runs in rounds r1–r5, all on
the same seed/targets) was searched per property; the printed run is the lowest oracle MAE with
PB-valid ≥ 0.90 (`collect.py`, `FINAL_SUMMARY.txt`; the paper's gap entry uses r5_gap_mu0p17 —
r3_gap_mu0p16 has 599.63 meV at PB 0.918, also "600"; the same-config repeat r5_gap_mu0p16_rep gave
615.82 meV at PB 0.892). The authors' `tabasco_property_guidance_pb_validity.tex` lists the same six
runs. The reward_scale values were fixed from earlier (June) tuning; only mu was searched here.

The `unguided` rows (Table 21/22 "Unsteered") come from the same six jobs (identical across them:
PB 0.987, 665 unique / 992 SMILES).

## multiMFM-SS row — `runs/multimfm_ss/k_<prop>_<a|b>/summary.json`

From `B/tabasco/codex_scratch/memory/search/k_<prop>_<rep>/summary.json` (steer_search writes only
this file). Launched 2026-07-25 14:18 by `dispatch_bol.sh bol1k.manifest` ("FINAL n=1000
confirmation: 6 winners x 2 replicates, identical hyperparameters") → `run_bol.sh` → 
`best_of_lookahead.py`; Slurm allocations per run in `bol1k.manifest` (e.g. k_cv_a 35049269) and
in `multimfm.qm9_results.SS_ALLOCATION`. Finished between 15:58 (k_gap_a, k_lumo_b) and 17:08
(k_gap_b). `k_lumo_a` (allocation 35049197) was cancelled by Slurm at 17:16:01 (step killed) and
left no summary. Both replicates of a property have **identical** arguments (seed 8, target seed
2026); they differ only through nondeterministic CUDA kernels.

| prop | mu | reward_scale | select_min_t |
|---|---|---|---|
| cv | 5.0 | 1.5 | 0.5 |
| mu | 3.0 | 8.638 | 0.3 |
| alpha | 11.0 | 0.3 | 0.3 |
| gap | 1.0 | 12000 | 0.3 |
| homo | 1.5 | 41500 | 0.3 |
| lumo | 1.5 | 12000 | 0.3 |

Other fixed arguments as for the non-SS row, plus `--pb-candidates 0` (harvest every step with
t ≥ select_min_t), `--pb-workers 16`. In-repo replay: `scripts/run_steer_search.sh <prop>` (default
n = 1000). The configs came from the n = 100 sweeps `v1`–`v9` (`bolv*.manifest`), all summarised in
`multimfm_ss_search_grid.csv`.

**Replicate mixing (paper Table 2 / Table 21).** The printed SS numbers match: cv = a (MAE and PB),
mu = MAE a or b (both 0.69) and PB b, alpha = mean(a, b) (MAE and PB), gap = a, homo = MAE
mean(a, b) and PB b (a 97.9, b 97.8, mean 97.85), lumo = b (a missing). Encoded in
`multimfm.qm9_results.SS_PUBLISHED_SOURCE`; `table2.csv` / `table21.csv` carry a, b, the mean of
available replicates and the as-published pick.

## Derived tables

| file | content | built by |
|---|---|---|
| `table2.csv` | per property: multiMFM oracle MAE (paper units) + run/config; SS oracle MAE a, b, mean, as-published | `qm9_results tables` from `runs/` |
| `table21.csv` | PB-valid: unsteered, multiMFM, SS a/b/mean/as-published, SS final-endpoint PB | same |
| `table22.csv` | SMILES uniqueness U, counts, U_all for unsteered and multiMFM | same |
| `multimfm_search_grid.csv` | one row per non-SS search run (60) | `qm9_results collect` |
| `multimfm_ss_search_grid.csv` | one row per steer_search sweep run (b1, v1–v9, k, smoke) | same |
| `table19_base_training.json` | base-model hyperparameters: checkpoint `args`, `model_args`, `stats`, optimizer `param_groups`, `step` + marked code constants | `qm9_results hparams` |
| `table20_mfm_training.json` | final MFM phase: checkpoint `args`, `student_cfg`, `step` + marked code constants | same |
| `paper_values.json` | numbers printed in `dMFM (40).pdf` (Tables 2, 19, 20, 21, 22 + text claims) | transcribed |

Energies: Hartree → meV with 27211.386245988 (as in the original `collect.py`). All MAEs are over
all generated molecules (see "Discrepancies").

## Training provenance (Tables 19–20)

- Base model: trained by the QM9 trainer now at `multimfm.qm9_data.main` (originally
  `qm9_dfm_smoke.py`), arguments in the checkpoint `args`; replay: `scripts/qm9/train_base.sbatch`.
  Original training logs/curves were purged. Not bit-reproducible.
- MFM student: final phase = `--stage 1 --rollout-condition-source generated` (Slurm job 21926337,
  2026-06-12, per `tabasco_docs/mfm_guidance_research.md`), initialised from
  `outputs/mfm_diag_rollout8_forced_20260611_long_spr4/stage4_20260611-171509/student_step_3000.pt`
  (job 21603965, stage-4 forced 8-step diagonal rollout), itself from earlier runs (`sens2000`, …)
  that the notes do not fully reconstruct. **All ancestors are purged and not in the backup**, so the
  MFM cannot be retrained end to end; `scripts/qm9/train_mfm.sbatch` replays the final phase given an
  `INIT_CKPT`. Replaying its command reproduces the stored argument dict exactly (checked by the
  training smoke job below; the only differences are the init path and output dir).

## Same-backbone baseline sweep — `baselines/`

Not part of the paper (as of 2026-09-29). `baselines/runs/{pilot,sweep}/...json` are copies of
`B/MultiMFM/outputs/{pilot,sweep}` (= this repo's gitignored `outputs/`), written by
`src/multimfm/baselines.py` and `steer_search.py`:

- pilot: `scripts/pilot_baselines.sbatch`, Slurm job 48045913, 2026-09-23 (log `B/MultiMFM/logs/bl_pilot-48045913.out`)
- sweep: `scripts/sweep_nfe_benchmark.sbatch`, Slurm job 48559075, 2026-09-25 (log `B/MultiMFM/logs/nfe_sweep-48559075.out`)

α only, n = 100, seed 8 (same targets as above). `baseline_runs.csv` flattens them; NFEs are
measured for the baselines only (`nfe_measured`); steer_search runs have no NFE count.

**Flag:** `B/MultiMFM/outputs/pilot/figures/make_figs.py` (and its fig1/fig2 PNG/PDFs) hardcodes the
pilot numbers and plots best-of-N at N = 32/64/128 from a power law fitted to N = 4/8/16
(0.890/0.585/0.384) plus a "4.8× less compute" crossover. Those points were never measured; the
sweep later measured 1.069/0.918/0.906 (PB-valid-filtered). It is deliberately not copied here.

## Verification reruns (2026-09-29) — `verification/`

- `run_gen_gpu_{A,B}/`: the authors' determinism check (`codex_scratch/check/run_gen.sh`: alpha,
  mu 0.45, rs 0.3, n = 32, no PB), run twice on two H100s on 2026-07-24.
- `ss_{alpha,cv}_n100_<tag>/`: n = 100 replicates of the paper SS configs from the original sweep
  (alpha: v6_alpha_mu11_s30, v7_alpha_mu11_{a,b}, v8_alpha_mu11_c; cv: v3_cv_mu5, v4_cv_mu5_{a,b},
  v5_cv_mu5_{c,d}).
- `rerun_20260929/`: outputs of `sbatch scripts/qm9/smoke_test.sbatch sampling` (Slurm job
  49255662, gpu_test, one A100-40GB) with the in-repo code and checkpoints: `rungen_alpha_n32`
  (run_gen config), `multimfm_alpha_n100`, `multimfm_cv_n100` (`run_multimfm.sh <prop> 100`),
  `ss_alpha_n100` (`run_steer_search.sh alpha 100`). Compared in notebook 07.
- Training smoke (not stored; outputs in gitignored `outputs/qm9/smoke/`): Slurm job 49255664 ran
  `train_base.sbatch` for 200 steps and `train_mfm.sbatch` for 20 updates (from the shipped student);
  both completed and wrote checkpoints with the same format/argument keys as the shipped ones.

## Discrepancies found (not resolved here)

1. Table 2 SS row mixes replicates (above); the consistent mean-of-replicates row is
   0.40 / 0.69 / 1.10 / 471 / 252 / 316 (PB 97.35 / 95.8 / 97.5 / 96.3 / 97.85 / 97.8).
2. Table 2 caption: "Our rows report MAE over PB-valid molecules." The printed MAEs are over all
   1,000 generated molecules (PB-invalid included); per-molecule PB flags were not stored.
3. Table 20 lists the QM9 MFM objective as "Diagonal + ESD"; the stored final phase is stage 1
   (diagonal + direction + condition-sensitivity), with no ESD term (also stated in the authors'
   `training_tables_molecular.tex` / `tables_to_fix.tex`).
4. Guidance hyperparameters (both rows) were selected on the same seed/target set that is reported.
5. The SS "look-aheads" are the base model's one-step `endpoint_prediction` (extra forward pass),
   not the MFM samples (Algorithm 3). Resolved in code on 2026-09-29 (authors' decision 5, see
   below): `steer_search --lookahead-source mfm` (default) implements Algorithm 3; the submitted
   behaviour is `--lookahead-source base_endpoint`.
6. §3.2 says steer-search "integrates the same guided ODE as multiMFM"; the SS runs use a 2-6x
   larger guidance strength mu than the multiMFM row (same reward_scale).

## 2026-09-29 reruns (authors' decisions 4, 5, 8)

Everything below was produced with the in-repo code on Kempner A100-40GB GPUs (partition
`kempner`) and is collected by `bash scripts/qm9/collect_table2_3seed.sh` (which also
re-executes notebook 07). Raw outputs: gitignored `outputs/qm9/{ss_mfm_pilot,
ss_mfm_reselect,table2_3seed,baselines_v2}/`.

### Code changes

* `steer_search.py`: `--lookahead-source {mfm,base_endpoint,both}`, default `mfm` = Algorithm 3:
  the look-aheads are the 32 MFM posterior samples the value estimator draws at every guided
  step (`value_gradient(..., keep_posterior=True)` returns them, detached); the PB-valid
  candidate with the lowest guide error among them and the endpoint is returned, else the
  endpoint. Error = guide on the look-ahead molecule (argmax atoms, `--lookahead-score
  guide_hard`, one regressor forward per look-ahead) or the tilt's own soft-atom guide values
  (`value`). Candidates live on CPU (top-K per molecule, `--lookahead-topk`, 0 = all);
  PoseBusters runs lazily in error order (identical choice to checking everything;
  `n_truncated_uncertain` counts molecules where a finite K could change it). The base-endpoint
  harvest of the submitted runs is `base_endpoint`; `both` runs the two selections on one
  trajectory. Per-sample CSV (PB flag, SMILES) and hook-measured NFE (`multimfm.nfe`) added.
* `property_targets.py`: NFE counters; `--per-sample-pb` stores per-molecule PB flags + SMILES
  (and `oracle_mae_pb_valid_only`).
* `baselines.py`: hook-measured NFE, lazy PB selection, chunked best-of-N, per-sample CSV.
* `qm9_eval.py` (PB worker pool, per-molecule records), `qm9_figures.py` (baseline figure from
  `results/qm9/baselines/baseline_sweep_v2.csv` only), `qm9_results.py` collectors
  (`rerun3`, `pilot-ss`, `ss-reselect`, `baselines-v2`).

NFE check (measured, per molecule, 128 steps, 115 guided steps, 32 samples, 4 MFM
steps): multiMFM = 128 base + 14,720 MFM forward passes (all 14,720 back-propagated), 14,848 generative passes in total; multiMFM-SS
with `mfm` look-aheads = identical (extra generative passes 0); the submitted `base_endpoint`
harvesting adds one base forward per harvested step (89 for select_min_t 0.3, 64 for 0.5).

### Seed-8 pilot (tuning seed, n = 100) — `ss_mfm_pilot/`

Slurm job 49275062 (`scripts/qm9/pilot_ss_mfm.sbatch`, array 0-2; earlier submissions
49269381, 49271305, 49272749 were cancelled before producing output). First implementation
defaults (`--lookahead-score value`, K = 128), all six properties with `--lookahead-source
both --diagnostic-variants`, + `alpha_n100_topk0`, + non-SS alpha with per-sample PB.
Outcome: ranking the MFM look-aheads by the guide on the look-ahead molecule
(`mfm_hardscore`) beat the soft-atom ranking in 5/6 properties on the same candidates ->
`guide_hard` became the default; the old select_min_t window made no consistent difference
(kept: every guided step, as Algorithm 3); K = 128 left 1-2/100 molecules uncertain -> the
3-seed runs keep every look-ahead (`TOPK=0`, ~1.5 GB CPU per variant at n = 1000).

### SS guidance strength on the tuning seed (n = 300) — `ss_mfm_reselect/`

Slurm jobs 49276093 (tasks aborted after one run each: `run_steer_search.sh` was edited while
running) and 49276728 (`scripts/qm9/ss_mu_reselect.sbatch`). Grid per property: multiMFM mu,
geometric midpoint, submitted SS mu, 2 x SS mu. Pre-registered rule: keep the submitted SS mu
unless another grid mu lowers the Algorithm-3 oracle MAE by > 2 SE of the difference with
PB >= 0.95. Result (`ss_mu_decision.csv`): keep for cv, mu, alpha, gap, homo; **lumo 1.5 -> 3.0**
(297.5 +- 18.6 -> 245.2 +- 13.8 meV).

### Table 2 / 21 / 22 on 3 seeds x n = 1000 — `rerun_3seed/`

`scripts/qm9/table2_3seed.sbatch`: seeds 1, 2, 3 with target seeds 2027-2029 (held out from
all tuning). multiMFM row: the submitted per-property (mu, reward_scale), per-sample PB.
SS row: submitted SS (mu, reward_scale) except lumo mu 3.0, `--lookahead-source both
--diagnostic-variants`, TOPK=0 (exact Algorithm 3); the paired submitted-harvesting selection
runs on the same trajectory. `VALUE_BATCH_SIZE=250` (fits 40 GB; only re-partitions the MFM
prior noise). Slurm jobs: 49276185 (multiMFM, tasks 0-17; tasks 12 and 16 failed on a node
with a busy GPU, holygpu8a19102 -- "CUDA-capable device(s) is/are busy or unavailable" --
and were resubmitted as 49279656), 49279470 (SS, tasks 18-35). **All 36 runs COMPLETED**
(2026-09-29 15:51); ~5-6 min and 16 GB GPU each. Every SS run stored all 3,680 look-aheads
per molecule (32 samples x 115 guided steps, 1.5 GB CPU) with `n_truncated_uncertain = 0`,
i.e. the returned molecule is exactly Algorithm 3's; the lazy PoseBusters walk needed only
3-14 checks per molecule instead of 3,680. `rerun_3seed_long.csv` (one row per variant/property/seed), `rerun_3seed_summary.csv`
(mean, SE = sd/sqrt(3)), `runs/` (slim summary.json + gzipped per_sample for key rows).

### Fixed same-backbone baseline sweep — `baselines/baseline_sweep_v2.csv`, `fig_qm9_steering_nfe.*`

`scripts/qm9/baseline_sweep_v2.sbatch`, Slurm job 49278312 (tasks 5 and 10 failed on
holygpu8a19102, resubmitted as 49279522): alpha, n = 100 per seed, seeds 1-3; base, best-of-N
N = 4..512, FK K = 4..32 x lambda {0.003, 0.01, 0.03}, DPS mu {0.03..1} and guide_every {2,4,8},
multiMFM (mu 3.4) and multiMFM-SS (mu 11) at MC = 1..32. Every point measured; figure from the
CSV only (`python -m multimfm.qm9_figures baselines`). All 15 tasks COMPLETED (120 runs,
324 rows, 3/3 seeds). The old `baseline_runs.csv` (seed-8
pilot/sweep of 2026-09-23/25, analytic multiMFM NFE) is kept for reference only.

### Verification of the Algorithm-3 selection

* `scripts/qm9/check_lazy_select.py` (CPU, seconds): the lazy PoseBusters walk returns the
  same molecule as a literal sequential implementation of Algorithm 3 (running arg-min in
  time order, strict "<") on 200 x 40 random cases including exact ties; with a finite
  top-K store every molecule whose choice could differ is flagged by
  `n_truncated_uncertain`.
* NFE (measured by forward hooks, per generated molecule, all 18 SS runs): multiMFM guided
  = 14,848 generative forwards (128 base + 14,720 MFM, all MFM back-propagated);
  multiMFM-SS with `--lookahead-source mfm` = **14,848, extra = 0**; with the submitted
  `base_endpoint` harvesting = 14,912 (cv, select_min_t 0.5) / 14,937 (others, 0.3),
  i.e. +64 / +89 base forwards.
* `scripts/qm9/test_notebook_rerun.sbatch` (Slurm 49281680, 2 min): executes notebook 07
  with `RERUN = True` on one GPU; 0 errors, and reproduces extra = 0 at n = 32.

### Guidance coefficient

`docs/provenance/qm9/guidance_coefficient.md`: the QM9 runs (both rows, submitted and
rerun) apply a **constant** guidance strength `mu`, not the paper's time-dependent
`sigma_t^2/2 = (1-t)/t`. Established by code reading; nothing was changed, so the reruns
stay comparable to the submission.
