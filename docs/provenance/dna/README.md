# DNA (dMFM) experiments: paper item -> code -> checkpoint

Everything for the DNA half of the paper lives in

| Where | What |
|---|---|
| `src/dmfm/` | the packaged DNA-MFM library + the July 2026 experiment scripts |
| `scripts/dna/` | parameterized Slurm/CLI wrappers that reproduce the full-size paper runs |
| `checkpoints/dna/`, `data/dna/` | the artifacts (see `checkpoints.md`) |
| `results/dna/` | the small CSV/JSON behind each table (owned by the results notebooks) |
| `tests/dna/` | CPU checks of the port (pytest), GPU checks against stored runs, and the check against the original library |
| `original_runs.md` | the original command line, Slurm job and result directory of every run |

Package layout:

- `dmfm.models` - `DiTSequenceModel` (base DFM mean denoiser), `DNAMFMStudent` (dMFM),
  `DNAMFMTeacherAdapter`; `dmfm.models.dit_seq` is the DiT backbone.
- `dmfm.lightning` - `DNAModule` / `GeneralModule`, the Lightning loop used for dMFM training.
- `dmfm.utils` - interpolant helpers (`flow_utils`), trainer flags (`parsing`), the dMFM objective
  (`mfm_diag_distill`), dataset/split helpers, and `model_loading` (checkpoint loaders shared by the
  experiments).
- `dmfm.regressors` - the C0 guide/oracle CNNs.
- `dmfm._mfm` - the four objects the DNA code used from the upstream MFM library
  (`BaseModel`, `LossWeightingNetwork`, `extract_posterior_velocity`, `compute_loss`), vendored from
  the copy inside DNA-MFM@187fe7b. It is *not* the repo's top-level `mfm/`: that copy passes zeros
  where this one passes `(t_cond, xt_cond)` to the teacher. For the DNA teacher the two are
  numerically identical (its `v` ignores those arguments), but vendoring keeps the DNA code
  independent of `mfm/` and of `timm`.
- `dmfm.experiments.<name>` - one module per experiment, each runnable as
  `python -m dmfm.experiments.<name> --help`.
- `dmfm.paths` - the repo-relative default location of every checkpoint and data file.
- `dmfm.rerun` - one function per experiment that reruns it at small n with the exact paper
  flags (section "Rerunning at small n" below).
- `dmfm.api` - a documented facade over the experiment code for notebooks and new experiments
  (see `API.md`; written alongside this port by the notebook/API work).

Every experiment module exposes `main(argv=None)`, so a notebook can call
`dmfm.experiments.<name>.main([...flags...])` with the same flags as the command line.

Source: GitHub `tullebulle/DNA-MFM@187fe7b`, recovered from the 2026-09-25 reconstruction
`scratch_dfm_recon_20260925/` (which matches that commit file for file), plus the three newer July
files from the surviving `dirichlet-flow-matching` checkout (`utils/mfm_diag_distill.py`,
`utils/parsing.py`, `dinko/c0_regressors.py`). Each module's docstring names the original script.

See also: `API.md` (stable `dmfm.api` cheat sheet), `verification.md` (Table 11, forward-pass parity
vs the original code, unguided reproducibility, GPU smoke), `port_sources.json` (origin + sha256 of
every ported module).

## Paper item -> producer -> checkpoint

| Paper item | Module | Wrapper | Checkpoints | Status |
|---|---|---|---|---|
| Table 9 (base generators) | `dmfm.experiments.train_base_dfm` | `scripts/dna/train_base_dfm.sbatch` | produces `checkpoints/dna/base/L*/best.pt`; the reported metrics are inside `best.pt["metrics"]` | reproducible |
| Table 9 (dMFM rows) | `dmfm.experiments.train_dmfm` | `scripts/dna/train_dmfm.sbatch`, `train_dmfm_l400_finetune.sbatch` | produces `checkpoints/dna/dmfm/L*` | reproducible (L400 needs the two-stage recipe) |
| Table 10 (parent-disjoint split) | `dmfm.experiments.prepare_yeast_parent_splits` | `scripts/dna/prepare_data.sbatch` | none (data) | reproducible only with the raw loop-seq table (below); the derived files are shipped |
| Table 11 (C0 regressors) | `dmfm.experiments.train_c0_regressor` | `scripts/dna/train_c0_regressors.sbatch` | produces `checkpoints/dna/c0/{guide,oracle}`; the paper numbers are `metadata.json["test_metrics"]` | reproducible; **recomputed from the shipped weights to ~1e-5** by `tests/dna/test_dmfm_port.py` |
| Table 1, Fig 3 (C0 guidance) **as submitted** | `dmfm.experiments.sample_c0_guidance` + `score_c0_oracle` | `scripts/dna/table1_c0_guidance.sbatch` | `base/L50/best.pt`, `c0/guide`, `c0/oracle` | GLASS guidance on the base DFM, **no dMFM loaded**; printed = n=32 pilot, an n=1000 rerun exists. Removed from the paper (authors' decision 2026-09-29), kept as provenance; guided samples are not bit-reproducible, unguided are |
| Table 1 **(new: dMFM)** | `dmfm.experiments.sample_c0_guidance_dmfm --paired --posterior dmfm` + `score_c0_oracle`, collected by `dmfm.experiments.collect_table1_c0` | `scripts/dna/table1_c0_guidance_dmfm.sbatch` (2 targets x 5 shards) + `scripts/dna/collect_table1_dmfm.sh` | `base/L50`, `dmfm/L50`, `c0/guide`, `c0/oracle` | same protocol as the GLASS run except the posterior sampler; results in `results/dna/c0_guidance/dmfm_n1000/` and `table1_dmfm_n1000_summary.json` |
| Tables 4-5 (diversity) | `dmfm.experiments.analyze_c0_diversity` (+ vendored `_yeast_c0_diversity_metrics`) | `scripts/dna/tables4_5_c0_diversity.sh` | `c0/oracle` (re-score check only) | **printed numbers not reproducible**: they come from the purged split65k pipeline. The parent-disjoint recompute (2026-09-25) **is reproduced exactly** (all 8 output CSVs, max diff 0) from `results/dna/c0_guidance/n1000/*` |
| Table 6 (Evo2 NLL) | `dmfm.experiments.score_yeast_c0_evo2` (reconstructed from bytecode) | `scripts/dna/score_evo2.sbatch` (`MODE=c0`), env `scripts/dna/evo2_environment.yml` | Evo2 7B (public) | code recovered; **inputs and outputs purged**, and they were split65k samples |
| Table 13 (exact rewards) | `dmfm.experiments.sample_reward_attainment` (+ `score_parent_reward_attainment_evo2` for the F.1 NLL) | `scripts/dna/table13_reward_attainment.sbatch` | `base/L*/best.pt` + `dmfm/L*` (diagonal students; now the module default) | reproducible |
| Fig 5, Tables 14-15 (MC scaling; **as submitted: GLASS finite-N vs GLASS-2048 on the base DFM, no dMFM loaded**; dMFM rerun: dna-scalefree) | `ablate_glass_gradient_mc`, then `rescore_gradient_pairs_scale_free`, `summarize_gradient_accuracy_metrics`, `plot_gradient_accuracy_required_mc` | `scripts/dna/fig5_tables14_15_gradient_mc.sbatch` + `fig5_tables14_15_rescore.sh` | `base/L*/best.pt` only (no dMFM) | reproducible; the rescore/summary step reproduces the stored Tables 14-15 CSVs exactly from the stored gradient tensors; **no script draws Fig 5 itself** (only its inputs); `--glass_end_time`: see flagged item 5 |
| Table 17 (posterior diversity) | `dmfm.experiments.eval_posterior_diversity` (+ `score_parent_posterior_evo2`) | `scripts/dna/table17_posterior_diversity.sbatch` | `base/L*/best.pt`, `dmfm/L*` | reproducible |
| Table 18 (dMFM vs GLASS-128) | `dmfm.experiments.ablate_dmfm_one_step_gradient_mc`, then `rescore_gradient_pairs_mae` (the table's MAE) | `scripts/dna/table18_dmfm4_gradient_mc.sbatch` | `base/L*/best.pt`, `dmfm_4step/L*` | reproducible up to GPU precision: the GLASS-RK4 reference gradient is precision-sensitive (see `original_runs.md` section 10) |
| Fig 4 right (NFE benchmark) | original: `benchmark_yeast_split_dmfm_steering.py` (dMFM half only); now `dmfm.benchmarks` via `dmfm.experiments.benchmark_nfe_steering` (regeneration with real baselines, separate work) | `scripts/dna/fig4_panelB.sbatch` | original models purged; regeneration uses `base/L50`, `dmfm/L50`, `c0/guide` | **not reproducible as published** (see below) |
| Fig 4 left (value error vs N) | original: none survives; regeneration `dmfm.benchmarks.value_error` (separate work) | `scripts/dna/fig4_panelA.sbatch` | (purged May models) | **no surviving producer** |
| Fig 6, Table 16 (base marginals) | none | none | (purged May models) | **no surviving producer** |
| Fig 7 (tilt histograms) | `dmfm.experiments.sample_c0_guidance` writes exactly this `histogram.png` | `scripts/dna/table1_c0_guidance.sbatch` | `base/L50`, `c0/guide` | figure *type* reproducible; the published panels came from May runs with unknown guidance strengths |
| Fig 8 (guided sample picture) | none | none | - | **producer unknown** (exported on a Mac) |
| Tables 19-20 (training hyperparameters) | - | - | stored in each checkpoint's `args.json` | verifiable from the artifacts |

`original_runs.md` has, for each of these, the exact original command, the Slurm job ids and
resources, and the result directory.

## Flagged items (read before rerunning)

1. **Fig 4 left, Fig 6 / Table 16, Fig 8: no producer survives.** Fig 4 left ("value error vs N")
   and the Fig 6 / Table 16 base-marginal statistics have no script and no data; the figure PDFs
   predate the parent-disjoint models. Fig 8's producer is unknown.
2. **Fig 4 right cannot be reproduced as published.** The original script ran only the dMFM half,
   on purged models, and its four baselines (best-of-N, Feynman-Kac, beam, MCTS) are a
   hard-coded CSV inside the script - no code for them ever existed in the DNA repo, all rows have
   `mae_sem = 0` (kept as `OLD_NON_DMFM_BASELINES` in `dmfm.experiments.benchmark_nfe_steering`).
   Both Fig 4 panels are being regenerated on the parent-disjoint models with real baseline
   implementations in `dmfm.benchmarks` (separate work; not verified here).
3. **Tables 4-5 (and Table 6)** describe parent-disjoint samples but were computed on the purged
   split65k pipeline with the old TF-converted guide. The shipped analyzer recomputes them on the
   parent-disjoint n=1000 sets, which gives different numbers.
4. **Table 1 / Fig 3 / Fig 5 use the base DFM, not the dMFM** (GLASS posterior value gradients with
   MC=8). No dMFM checkpoint is loaded in those runs.
5. **`--glass_end_time` (Fig 5 / Tables 14-15) is 1.0.** The option did not exist when the runs
   were made; the surviving script defaults to 0.999. Recomputing the stored L=50 reference
   gradients (first two probes, motif and gc, original seeds) gives 0.3-0.6 % relative error with
   end time 1.0 and 1.2-1.3 % with 0.999 on every probe, against a TF32 precision noise of
   0.2-0.4 %; so the runs integrated to 1.0 (Euler never evaluates the singular drift at t=1).
   The port accepts `--glass_end_time 1.0` for the Euler solver, and the wrapper and
   `dmfm.rerun` use it. (Table 18's reference used 0.999 with RK4, as recorded.)
6. **Table 13 uses the diagonal students.** The original script imported `DEFAULT_DMFMS` from
   `ablate_dmfm_one_step_gradient_mc`, which was repointed at the 4-step students the day after
   Table 13 ran. The port's `sample_reward_attainment` defaults to the diagonal students
   (`TABLE13_DMFMS`, confirmed by every Table 13 `metadata.json`), and the wrapper also passes them.
7. **The dMFM gap-sampling fix.** `dmfm.utils.mfm_diag_distill` uses the corrected July gap sampling
   (a lower gap bound of 0 is honoured). The 4-step students need it; the diagonal students are
   unaffected (`gap_mode=random`). `--mfm_legacy_gap_or_default` restores the old behaviour.
8. Further paper-vs-record mismatches (base `best.pt` is a 45k-55k-step checkpoint, not 200k;
   Table 7 describes a different finetune; App. E.1 training time; the reward scale factor) are
   listed in `original_runs.md` section 13. None of them were "fixed" here.

## Rerunning at small n (notebook `RERUN = True` cells)

`dmfm.rerun` runs each experiment with exactly the flags of its full-size wrapper
(`dmfm.rerun.PAPER_FLAGS`, checked against `scripts/dna/*.sbatch` by
`tests/dna/test_dmfm_port.py::test_rerun_paper_flags_match_wrappers`), overriding only sample
counts. Each returns its output directory, which holds the same files as the full run. One GPU
(rough times on an A100 MIG slice; everything but Table 18 also runs, slowly, on CPU);
`source scripts/env.sh` or kernel `multimfm`.

```python
from dmfm import rerun
R = "outputs/dna/rerun"
rerun.table11()                                          # CPU, 10-60 s: Table 11 metrics from the shipped C0 weights
rerun.table1(f"{R}/t1_p1", target_c0=1.0, n_samples=32)  # Table 1/Fig 3/Fig 7 (+ oracle); ~1 min; -1.0 for the other row
rerun.tables4_5(f"{R}/t45")                              # CPU, ~3 min, on the shipped n=1000 sets (or pass two run dirs)
rerun.table13(f"{R}/t13_motif_L50", length=50, reward="motif", n_samples=16)   # ~1 min
rerun.fig5(f"{R}/f5_gc", reward="gc", lengths=(50,), n_probes=4, n_repeats=2,
           mc_values=(1, 2, 4, 8, 16, 32), reference_mc=2048)                  # + Tables 14-15 rescore; ~2 min
rerun.table17(f"{R}/t17", lengths=(50,), n_sources=4, n_futures=16)          # ~1 min
rerun.table18(f"{R}/t18_gc_L50", reward="gc", length=50, n_probes=4, n_repeats=2)  # + MAE rescore; ~1 min
```

Outputs to read back: Table 1 `sample_scores.csv`, `summary.json`, `oracle/oracle_summary.csv`;
Tables 4-5 `table6_baselines.csv`, `table7_generated.csv`; Table 13
`L<L>/reward_attainment_summary.csv`; Fig 5 / Tables 14-15
`gradient_scale_free_summary_all_lengths.csv`; Table 17 `posterior_diversity_summary.csv`;
Table 18 `L<L>/gradient_mae_summary.csv`. Any other flag can be overridden by keyword
(e.g. `rerun.table1(..., guidance_frac=4.0)`); the modules can also be called directly as
`dmfm.experiments.<name>.main([...])`. Training at small n:
`dmfm.experiments.train_base_dfm.main([...])` / `dmfm.experiments.train_dmfm.main([...])` with
the flags of `scripts/dna/train_*.sbatch` and a small `--steps` / `--max_steps` (see
`scripts/dna/smoke_gpu.sbatch`, part `train`). Fig 4 lives in `dmfm.benchmarks`, Figs 6-8 in
`dmfm.regen` (separate work).

## How the port was verified (2026-09-29)

All in the `multimfm` env (Python 3.11, torch 2.5.1); the originals ran in `seq_h100`
(Python 3.9, torch 2.1).

| Check | How | Result |
|---|---|---|
| every module imports; every experiment CLI has `--help` | `tests/dna/test_dmfm_port.py` | pass |
| every shipped checkpoint loads (strict / no missing keys) and runs a forward pass on CPU | same | pass (4 base, 4 dMFM, 4 four-step, 2 C0) |
| Table 11: held-out C0 metrics recomputed from the shipped weights and data | same | guide MAE 0.158771 / r 0.877414 vs stored 0.158772 / 0.877412; oracle 0.161985 / 0.872416 vs 0.161985 / 0.872417 (paper: 0.159 / 0.88, 0.162 / 0.87) |
| training wrappers reproduce the original arguments | dry-run of `scripts/dna/train_*.sbatch`, parsed and compared with each shipped `args.json` (same test file) | identical for base L50-400, dMFM L50-200, L400 finetune, 4-step L50-400 (after fixing the base batch sizes 256/128/64 for L100/200/400) |
| dMFM training objective is the original one | one train step (loss + every gradient) and one validation step (all probe metrics) of the original `lightning_modules/dna_module.py` (recon library + July `mfm_diag_distill.py`) vs `dmfm.lightning.dna_module`, same weights and seed, CPU; diagonal+ESD (L50 args, global step 10 and 5000) and 4-step (gap_mode long) configs | **bitwise identical** (max diff 0 in loss, gradients, probe metrics; same optimizer groups) |
| dMFM objective (re-checkable) | `tests/dna/check_against_original_library.py --original <recon dir> --july_distill <...>/utils/mfm_diag_distill.py --kind {dmfm,dmfm4}` | `ORIGINAL_EQUIVALENCE_OK` for both |
| Table 1 sampler vs the original script | `check_against_original_library.py --kind c0_sampler` (CPU, where the guided branch is deterministic): original recon `sample_yeast_c0_guidance_hist.py` vs `dmfm.experiments.sample_c0_guidance`, 2 pairs | guided and unguided sequences and scores **identical** |
| Table 1 sampler vs the stored run | GPU smoke: 8 pairs at L=50, target +1, compared with the stored n=1000 run | unguided sequences 8/8 identical, guide scores max diff 5e-6, oracle scores 1e-6; guided differs (nondeterministic on GPU, as expected) |
| Tables 4-5 recompute | `analyze_c0_diversity` on `results/dna/c0_guidance/n1000/*` vs `workdir/rebuttal_parent_c0_diversity_n1000_20260925` | all 8 CSVs identical (max diff 0) |
| Tables 14-15 post-processing | `rescore_gradient_pairs_scale_free` + `summarize_gradient_accuracy_metrics` on copies of the stored motif/conjunction runs | all per-state/summary CSVs identical (max diff 0) |
| Table 6 scorer | reconstruction compiled with CPython 3.9 vs the surviving `.pyc` | all 22 code objects identical (bytecode, constants, names, line tables) |
| Fig 5 / Tables 14-15 and Table 18 gradients vs the stored runs | `tests/dna/check_gradient_mc_reference.py` (L=50, probes 0-1, original seeds; GPU jobs 49264548 and 49283401 (`GRADMC_CHECK_OK`), diagnostics 49272897) | calibration exact (<5e-4); Fig 5 reference gradients within 0.3-0.6 % with end time 1.0; Table 18 probe states/noise/calibration identical, 4-step dMFM estimates within 2 %, GLASS-RK4 reference only directionally (cos >= 0.99: it moves 35 % with TF32 on/off) |
| `dmfm.rerun` cheat-sheet calls | every function at tiny n on GPU (job 49283401, 81 s total) and all but Table 18 on CPU | all run |
| GPU runs of every experiment | `scripts/dna/smoke_gpu.sbatch` (jobs 49264546, 49264548, 49266706): Table 1 + oracle, 20 steps each of base DFM / dMFM / 4-step, C0 regressor, Tables 13, 17, Fig 5, Table 18 | all run |

## Raw data: re-downloading `yeast_sequences.txt`

The pipeline starts from the loop-seq intrinsic-cyclizability table of

> Basu, A., Bobrovnikov, D.G., Qureshi, Z., Kayikcioglu, T., Ngo, T.T.M., Ranjan, A., Eustermann, S.,
> Cieza, B., Morgan, M.T., Hejna, M., Rube, H.T., Hopfner, K.-P., Wolberger, C., Song, J.S., Ha, T.
> *Measuring DNA mechanics on the genome scale.* Nature 589, 462-467 (2021).

The file `data/yeast_sequences.txt` (used as
`data/dna/raw/yeast_sequences.txt` here) is gone and is not redistributed. To rebuild the dataset:

1. Get the loop-seq **Tiling library** table of that paper (published with the paper as
   supplementary data; the same data were used to train DNAcycP, Li et al., NAR 2022). It has
   82,368 sequences: 576 *S. cerevisiae* genes, a 1,044-bp region around each +1 nucleosome dyad,
   50-bp fragments every 7 bp. What `prepare_yeast_parent_splits` expects (and what the runs
   used): a tab-separated file with a header containing at least the columns `Sequence` (the
   100-nt library sequence; the central 50 nt, `Sequence[25:75]`, are the fragment) and `C0`
   (intrinsic cyclizability), rows in tiling order (143 consecutive windows per gene). The script
   checks all of this (576 parents x 143 windows, overlaps consistent) and fails otherwise. The
   exact supplementary-file number / URL was not recorded with the runs and could not be
   verified here (the journal site needs a login); the sha256 below is the arbiter.
2. Put it at `data/dna/raw/yeast_sequences.txt` and check it:

   ```bash
   sha256sum data/dna/raw/yeast_sequences.txt
   # must print aabfb6c51cb19e99a492f1a86d93fd658360181b927ba31489f432f810aadbc2
   # (the same value is recorded as "source_sha256" in
   #  data/dna/yeast_parent_disjoint/parent_split_seed0.json)
   ```

3. Rebuild the windows and splits (deterministic, seed 0, stride 7):

   ```bash
   sbatch scripts/dna/prepare_data.sbatch      # or run the module directly on a CPU node
   ```

   It writes `yeast_parent_L{50,100,200,400}.pt` (+ `c0` labels for L=50),
   `yeast_parent_L*_split_seed0.pt` and `parent_split_seed0.json`. Compare the JSON against the
   shipped one: all index counts and parent lists must be identical (Table 10).

If the hash does not match, the file is a different release of the table: do **not** retrain on it
silently; the shipped derived files under `data/dna/yeast_parent_disjoint/` are the ones every paper
number is based on.

## Reproducing, end to end

```bash
source scripts/env.sh
python scripts/manifest.py verify --group dna --fetch-missing   # artifacts
python -m pytest tests/dna -q                                   # CPU checks (~2-4 min; also: python tests/dna/test_dmfm_port.py)
PARTS="sample train reward posterior nfe" sbatch scripts/dna/smoke_gpu.sbatch   # GPU smoke (~4 min)
PARTS=gradmc sbatch scripts/dna/smoke_gpu.sbatch                # gradient check vs stored runs (~4 min)

# full-size paper runs (each is one Slurm job; see the header of each file)
LENGTH=50 sbatch scripts/dna/train_base_dfm.sbatch
LENGTH=50 sbatch scripts/dna/train_dmfm.sbatch
LENGTH=50 sbatch scripts/dna/train_dmfm_4step.sbatch
sbatch scripts/dna/train_c0_regressors.sbatch
TARGET_C0=1.0 sbatch scripts/dna/table1_c0_guidance.sbatch      # and TARGET_C0=-1.0
bash  scripts/dna/tables4_5_c0_diversity.sh                    # CPU, ~3 min; defaults = results/dna n=1000 sets
LENGTH=50 REWARD=motif sbatch scripts/dna/table13_reward_attainment.sbatch
REWARD=gc sbatch scripts/dna/fig5_tables14_15_gradient_mc.sbatch ; bash scripts/dna/fig5_tables14_15_rescore.sh ...
sbatch scripts/dna/table17_posterior_diversity.sbatch
sbatch scripts/dna/table18_dmfm4_gradient_mc.sbatch             # 12-task array
sbatch scripts/dna/fig4_panelB.sbatch                           # Fig 4 regeneration (separate work)
MODE=c0 sbatch scripts/dna/score_evo2.sbatch                    # separate evo2 env
```
