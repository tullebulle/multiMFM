# Reproducibility plan: consolidating the dMFM / multiMFM paper into MultiMFM

Status as of 2026-09-29. Internal working document for the authors. It records candid
mismatches between the paper and the surviving run records, so don't commit it to a
public or anonymized mirror as-is.

**Goal:** every experiment in the ICLR 2027 submission ("Scalable inference-time steering
in molecular design with multimodal Meta Flow Maps", `dMFM (40).pdf`) is reproducible
from this repo alone: code, checkpoints, the results behind each table, and one notebook
per paper section.

## 1. Purge situation and backup

The netscratch purge has already deleted much of the source material. Source repos are
under `/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/`:

| Repo | What is left |
|---|---|
| `norway` (toy, Fig 1) | Nothing. 0 files, only empty directories. |
| `tabasco` (QM9 origin) | Code and checkpoints gone. Run logs/JSONs in `codex_scratch/`, `docs/`, two driver scripts survive. |
| `dirichlet-flow-matching` (DNA) | Everything with mtime before 2026-07-25 gone, incl. the core library. Paper checkpoints and results (Jul 25–27) survive. |
| `geom-drug-tabasco` (GEOM) | Intact apart from the vendored `mfm/` (recoverable with `git checkout`). |
| `mfm` | Only unrelated ImageNet tilt-matching checkpoints. |

The surviving DNA and GEOM artifacts are the same age as files that were already
deleted. **Everything paper-relevant that survives was copied (not moved) on 2026-09-29 to**
`/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/` (~7.4 GB,
`SHA256SUMS.txt` inside; holylabs is not purged). Use it as the source of truth if a
netscratch copy disappears.

The purged DNA library is recoverable, and **a complete clone of `tullebulle/DNA-MFM` at commit
`187fe7b` is on non-purged holylabs** at
`/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/DNA-MFM_187fe7b`. Use it as
the source of originals: no GitHub access is needed, and a credentialed clone was denied by the
permission system (2026-09-29), so nobody should attempt one. `187fe7b` is the commit recorded in
the dMFM training args, and it matches the 2026-09-25 reconstruction in
`/n/netscratch/kozinsky_lab/Lab/uunneberg/scratch_dfm_recon_20260925/` file for file. Verbatim
originals copied out of the clone live in `results/dna/original_scripts/`, with sha256s in
`_sources.json` re-verified against the clone. Note that the clone holds
`scripts/score_yeast_c0_evo2.py` **as source**, so the `.pyc` claim below is obsolete.
Two July files are newer than that commit and must be taken from the surviving repo:
`utils/mfm_diag_distill.py`, `utils/parsing.py`. `dinko/c0_regressors.py` is not in the clone
at all.

## 2. Experiment inventory

| Paper item | Code | Checkpoints | Results behind the numbers |
|---|---|---|---|
| Fig 1 toy (Norway / smiley), Table 12 | Lost; partly recoverable (§3.4) | Lost | Lost |
| Table 1, Fig 3 (DNA C0 guidance) | Yes, but the published run is **GLASS on the base DFM, not dMFM** (§7.8). dMFM version: `dmfm.experiments.sample_c0_guidance_dmfm` | Yes | Yes (GLASS n=32 and n=1000); dMFM n=1000 pending |
| Tables 4–5 (DNA diversity) | Yes (analyzer lives in an output dir) | Yes | Yes (n=1000) |
| Table 6 (Evo2 NLL) | **Recovered**: the scorer is source in the holylabs DNA-MFM clone (the local repo has only the `.pyc`); the re-derived module is AST-identical to it on the whole scoring path. Evo2 env rebuilt, 7B weights re-fetched | Evo2 is public | Lost; rerunning |
| Tables 9–11 (DNA training, split, C0 regressors) | Yes | Yes | Yes |
| Table 13 (exact motif rewards) | Yes | Yes | Yes |
| Fig 4 left (value error vs N) | **Recovered** (DNA-MFM@187fe7b -> `results/dna/original_scripts/benchmark_yeast_split_value_mc.py`; ported to `dmfm.benchmarks.value_error`) | Older `split65k` models, purged | Lost |
| Fig 4 right (NFE comparison) | Plot script survives; baselines are a hardcoded CSV, confirmed hardcoded in the upstream commit too; real implementations now in `dmfm.benchmarks.search` | Older `split65k` models, purged | Lost |
| Fig 5, Tables 14–15 (scale-free MC) | Yes, but it is **GLASS vs GLASS**: estimator and reference are both the GLASS posterior on the base DFM, no dMFM loaded (§7.8). dMFM rerun in `dmfm.scalefree` | Yes | Yes (GLASS); dMFM pending |
| Table 17 (posterior diversity) | Yes | Yes | Yes |
| Table 18 (dMFM vs GLASS-128) | Yes | Yes (4-step students) | Yes |
| Fig 6, Table 16 (base marginals) | None | Yes | Lost |
| Fig 7 (tilt histograms), Fig 8 (guided sample) | Fig 7 script recoverable; Fig 8 producer unknown | Yes | Lost |
| Table 2, 21, 22 (QM9) | In repo for the SS row; non-SS row configs only in tabasco logs | In repo | Yes (tabasco `codex_scratch`) |
| Table 3, Table 21 GEOM row | Yes (`geom-drug-tabasco` + uncommitted TFG-Flow code) | Yes | Yes |
| Tables 19–20 (molecular training) | Stored in checkpoint args | Yes | n/a |

## 3. Source map

### 3.1 QM9 (already mostly here)
- `src/multimfm/steer_search.py` reproduces the **multiMFM-SS** row; `scripts/run_steer_search.sh` has the exact per-property SS configs (default `NUM=100`; paper used 1000).
- The **non-SS multiMFM** row came from `tabasco/codex_scratch/check/search/` (`run_prop.sh` → `qm9_length_conditioned_guidance_experiment.py`, which is `src/multimfm/property_targets.py` here). Per-property (mu, reward_scale): cv 2.80/1.5, mu 0.78/8.638, alpha 3.40/0.3, gap 0.17/12000, homo 0.50/41500, lumo 0.44/12000; fixed seed 8, target-seed 2026, n=1000, 128 steps, 32 value samples, 4 inner steps, `--mfm-diagonal`.
- SS run records: `tabasco/codex_scratch/memory/search/k_{prop}_{a,b}` (`bol1k.manifest`).
- The base model trainer is `qm9_data.py`'s `main()` with the args stored in the checkpoint, not `train_base_flow.py` as the README says.
- The MFM student cannot be retrained end to end: its init checkpoint (`mfm_diag_rollout8_forced_20260611_long_spr4/.../student_step_3000.pt`) was purged.
- Guide/oracle regressors are the upstream TFG-Flow published weights.
- Uncommitted QM9 baseline sweep (`src/multimfm/baselines.py`, `scripts/pilot_baselines.sbatch`, `scripts/sweep_nfe_benchmark.sbatch`, results in gitignored `outputs/`). `outputs/pilot/figures/make_figs.py` hardcodes pilot numbers and includes power-law-extrapolated best-of-N points that were not measured.

### 3.2 DNA (`dirichlet-flow-matching`)
- Library: `tullebulle/DNA-MFM@187fe7b` plus the July `utils/mfm_diag_distill.py`, `utils/parsing.py`, `dinko/c0_regressors.py`.
- Scripts (in `scripts/` unless noted):
  - Base DFM: `train_dna_dfm_dit.py` (`slurm/train_yeast_parent_L*_dfm.sbatch`)
  - dMFM: `train_dna.py` + `utils/mfm_diag_distill.py` (`slurm/train_yeast_parent_dmfm_diagonal.sbatch`)
  - Split: `data/prepare_yeast_parent_splits.py`
  - C0 regressors: `train_parent_c0_regressor.py` (`slurm/train_yeast_parent_c0_regressors.sbatch`)
  - Table 1 / Fig 7: `sample_yeast_c0_guidance_hist.py` (recon dir only) + `score_yeast_c0_oracle.py`
  - Tables 4–5: `workdir/rebuttal_parent_c0_diversity_n1000_20260925/analyze_parent_c0_diversity.py` (+ `vendor/`)
  - Table 13: `sample_parent_reward_attainment.py`
  - Fig 5, Tables 14–15: `ablate_glass_gradient_mc.py` + `rescore_gradient_pairs_scale_free.py`, `summarize_gradient_accuracy_metrics.py`, `plot_gradient_accuracy_required_mc.py`
  - Table 17: `eval_parent_posterior_diversity.py`
  - Table 18: `ablate_dmfm_one_step_gradient_mc.py`
  - Fig 4 right: `benchmark_yeast_split_dmfm_steering.py` (hardcoded baselines, lines 28–48)
  - Fig 4 left: `benchmark_yeast_split_value_mc.py` (recovered verbatim from the holylabs
    DNA-MFM@187fe7b clone, 2026-09-29; the published panel's reward is a synthetic smooth
    sequence reward, not `f_cyc`, and its reference is teacher-SDE, not GLASS)
  - Evo2: `score_parent_*_evo2.py`; the C0-set scorer survives only as `scripts/__pycache__/score_yeast_c0_evo2.cpython-39.pyc`
- Selected checkpoints (`workdir/`):
  - Base: `yeast_parent_a_L{50,100,200,400}_dfm_dit_h192_b4/best.pt` (35 MB each)
  - dMFM: `yeast_parent_a_L50_dmfm_diagonal_h192_b4/epoch=95-step=49000.ckpt`, `…L100…/epoch=80-step=39000.ckpt`, `…L200…/epoch=51-step=45000.ckpt`, and **`yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/epoch=33-step=45750-v1.ckpt`** (different run dir); 40 MB each
  - 4-step students (Table 18): `yeast_parent_a_L*_dmfm_4step_esd_gap_upto025_corrected_20260727/`
  - C0: `yeast_parent_c0_{guide,oracle}_2026-07-26/best_state.pt`
  - Each needs its `args.json`.
- Data: `data/yeast_parent_disjoint/` (368 MB). The raw loop-seq TSV `data/yeast_sequences.txt` is gone; its sha256 is in `parent_split_seed0.json` for verifying a re-download from Basu et al. 2021.
- Docs worth carrying over: `docs/` (15 files, esp. `rebuttal_data_management_scaling.md`, `what_rebuttal_posterior_map_accuracy.md`, `rebuttal_mc_scaling_response.md`).
- Environment: `seq_h100` (Python 3.9, torch 2.1); reported partly gutted, needs checking. Evo2 needs its own env (Python 3.12, torch 2.7, `evo2`, 7B weights).

### 3.3 GEOM-Drugs (`geom-drug-tabasco` + `TFG-Flow`)
- Base teacher: `validation_teacher/big_curve/ckpts/epoch=38-step=89310.ckpt` (297 MB). Hydra run config: `outputs/2026-07-26/01-03-35/.hydra/` (batch 512 on 1 GPU; the committed `configs/experiment/geom_dfm.yaml` now says 256).
- MFM student: `outputs/mfm_student_geom_stage2_ddp/stage2_20260727-222154/student_step_10000.pt` (121 MB). Lineage: stage1a → 1b → 2a → 2b, all backed up.
- Fingerprint guide head: `TFG-Flow/storage/ckpts/geom_structure_head_ddp_cont2/best.pth` (4.3 MB). The authors trained it using TFG-Flow's EGNN architecture; it is not an upstream release.
- Code:
  - Steering: `glass/{mfm_fingerprint_steering,glass_fingerprint_steering,glass_guided_sampling_experiment}.py`
  - MFM trainer: `metaflow/{mfm_student_train,geom_teacher}.py`
  - Base training: `src/train.py` + `configs/`
  - PB validity: `validation_teacher/check_pb_validity.py`
  - Uncommitted TFG-Flow code: `train_structure_head.py`, `dataloader/datasets/{geom_fingerprint,build_geom_fingerprints}.py`, `scripts/geom_drug/`, and the `num_atom_types` patch to `diffusion/predictor.py`.
- Table 3 command: job 35670937, `glass/run_mfm_capgap_steering.sbatch` (n=64). 4,096 × 3 rerun: `glass/run_capgap_4096_3seed.sbatch`.
- Data: `processed_geom_{train,val,test}.pt` are public (HF `carlosinator/tabasco-geom-drugs`); the LMDB rebuilds from them. Fingerprint LMDB `TFG-Flow/storage/datasets/geom_fp/` (val 173 MB needed for steering targets; train 1.3 GB only for retraining the head).
- Known bug: targets come from the `geom_fp` val split (up to 82 atoms) but loading uses `max_len=71`; seed 21 of the 4,096 rerun crashed on a 72-atom target.
- `src/tabasco` here equals geom-drug-tabasco HEAD minus two small fixes (`callbacks/posebusters.py`, `data/components/lmdb_base.py`); port those. The uncommitted Sep-2 changes there are explicit-H work, not paper-related.

### 3.4 Toy example (Fig 1)
- All code, checkpoints and the notebook `mfm_guidance_norway_smiley.ipynb` are purged.
- Recoverable fragments:
  - `~/.codex/sessions/2026/06/03/rollout-2026-06-03T13-08-02-019e8e74-d7f5-7962-9f77-c063607ddbdd.jsonl`: `mfm_model.py` (jsonl line ~2287), `train_mfm_diagonal.py` (~2288), `probe_mfm_diagonal.py` (~2289), `glass.py` (~2290), parts of `train_mfm_consistency.py` (~15669–15738).
  - `~/.cursor/projects/n-netscratch-kozinsky-lab-Lab-uunneberg-repos-norway/terminals/1.txt`: exact training commands and final steering settings.
- Not recoverable: `data.py`, `model.py`, `train.py`, `reward.py`, `value_fn.py`, `mfm_value_fn.py`, experiment scripts, notebook.
- Design notes: `tabasco/docs/mfm_lab_notebook.md`; `src/tabasco/models/components/flow_map_transformer.py` ports the norway inductive biases.
- The Fig 1 PDFs were last edited on a Mac, so a laptop copy may exist.

## 4. Target layout

```
src/tabasco/        as now (+ 2 fixes from geom-drug-tabasco HEAD)
src/multimfm/       molecules: QM9 as now; new geom/ subpackage (fingerprint steering,
                    GEOM teacher loader, multi-GPU MFM trainer)
src/dmfm/           DNA: DNA-MFM@187fe7b library + July scripts, packaged (no flat
                    script-to-script imports)
src/toy/            multimodal flag/smiley, rebuilt
third_party/tfg_flow/  + GEOM fingerprint dataset, structure-head trainer, predictor patch
mfm/                vendored upstream MFM library (as now)
scripts/            parameterized sbatch/CLI for full-size runs (no absolute paths)
notebooks/          one per paper section (below)
results/            small CSV/JSON behind every table (tracked in git)
checkpoints/, data/ downloaded via manifest (sha256)
docs/               provenance notes carried over from the source repos
```

One main environment (`multimfm`: Python 3.11, torch 2.5.1). DNA code must be tested
under it. Optional separate `evo2` environment for Table 6.

## 5. Notebooks

Each notebook opens with `RERUN = False`: rebuild the table/figure from `results/` on CPU in
seconds. `RERUN = True`: load checkpoints and rerun at small n on one GPU. Full-size runs go
through `scripts/`. Guided sampling is not bit-reproducible (nondeterministic CUDA
backward), so reruns match within confidence intervals, not exactly.

| Notebook | Covers |
|---|---|
| `00_setup` | Download checkpoints/data, verify sha256 |
| `01_toy_multimodal` | Fig 1, Table 12 |
| `02_dna_base_model` | Tables 9–11, 16; Fig 6 |
| `03_dna_cyclizability` | Table 1, Fig 3, Figs 7–8, Tables 4–5 |
| `04_dna_exact_rewards` | Table 13 |
| `05_dna_mc_scaling` | Fig 5, Tables 14, 15, 17, 18 |
| `06_dna_benchmarks` | Fig 4 (both panels) |
| `07_qm9_property_guidance` | Tables 2, 19–22 (+ QM9 baseline sweep if it goes in the paper) |
| `08_geom_fingerprint` | Table 3, GEOM rows of Tables 19–21 |
| `09_evo2_plausibility` | Table 6 (separate env) |

## 6. Checkpoint and data hosting

| Item | Size |
|---|---|
| QM9 models + regressors | ~47 MB (already in Git LFS) |
| DNA models (base, dMFM, 4-step, C0) | ~0.47 GB |
| DNA data (`yeast_parent_disjoint`) | 368 MB |
| GEOM teacher + student + head | ~0.42 GB |
| GEOM fingerprint val split | 173 MB |
| GEOM molecules | 2.2 GB, public on HF |

Recommendation: host models and data on Hugging Face Hub with a sha256 manifest and a
download cell, and keep git for code and small results. Git LFS quotas are likely to be hit.
The README names both authors while the paper is under double-blind review, so a
submission link needs an anonymized mirror.

## 7. Decisions

Provenance of these decisions, so the record is not misread later: items 1-6 were *proposed* by the
agents doing the 2026-09-29 reproducibility work, which then began acting on them before any author
had seen them. They were put to Ulrik Unneberg late on 2026-09-29, in the Claude Code session that
produced this round, and endorsed all six as written. They have not yet been through co-author
review. The evidence and the outcome so far are recorded under each item.
Items 7-9 are findings from the round-2 reruns; 9 is still open.

1. **Table 3 (GEOM), most serious.** 0.227 Tanimoto / 87.5% PB came from n=64, one seed, the
   best of a 4-point grid (job 35670937). **Decided: rerun exactly as the paper describes it and
   report what comes out.** At 4,096 x 3 seeds, with (lambda, mu) selected on a held-out target
   pool, guidance was negative at every setting tried in the first sweep (0.2287 -> 0.2241).
   **SETTLED 2026-09-29 (job 49274410, complete 24/24): gain -0.00031 +- 0.00065, i.e. 0.5 sigma
   from zero; head gain -0.00034; PB 87.75 -> 87.14.** The earlier +0.00066 was a replicate-2
   artifact of the partial set. The only defensible statement is **"no demonstrated gain at
   scale"** -- do not assert a negative effect either. The base generator (0.2287 +- 0.0005)
   already exceeds TFG-Flow's guided 0.208; PB validity falls 88.0 -> 85.0. The published n=64
   number is reproducible as sampling noise: 64-molecule draws from the 4,096 population are
   positive 33.6% of the time, 4.2% as best-of-four. Also established: TFG-Flow released no
   fingerprint predictor, so "off-the-shelf" was false, and its reference values are measured on
   molecules capped at 30 heavy atoms against our 71.
   **The guidance term is NOT defective -- that escape route is closed** (`docs/geom_guidance_diagnosis.md`).
   Five checks, each with error bars: finite-difference of dr/dcoords ratio 1.0003; of dr/datomics
   1.0000-1.0026; of grad_x V_t through the real 10-step flow map (TF32 forced off, noise floor
   4.8e-7) 1.0058 joint / 0.9845 coords; a model-free analytic chain test exact to 7 digits 12/12;
   and a pre-registered sign test where `-mu` is *worse* at all four matched magnitudes. Method
   note: a first GPU attempt with production TF32 produced garbage (FD slopes proportional to 1/h)
   and was nearly misread -- force `highest` for any FD work. The `grad-sign` step scan is
   **underpowered as a sign test** and must not be quoted either way (its old "plumbing correct"
   string came from `max(positive deltas) > 0`, not a sign test; removed from the code).
   **Why it cannot work, measured rather than argued:** FP2 is a *graph* fingerprint -- exactly
   invariant to coordinate perturbations below 0.006 A and randomised by 0.2 A (self-Tanimoto 0.258
   vs 0.228 baseline). There is no displacement scale at which it responds smoothly, so there is no
   operating window for coordinate guidance. Consistently, the head's own accuracy collapses with
   dose (0.757 unguided -> 0.140 at lambda=1), which explains the falling head-Tanimoto without any
   sign error. The grid reports 16/17 points and **none clears the pre-registered bar**; lambda=1 --
   the paper's literal reward, never previously run -- is catastrophic (gain -0.088/-0.118/-0.152 at
   mu 0.25/0.5/1.0; PB 88.3% -> 44.9/31.3/**14.5**%), and the paper's `(1-t)/t` schedule is the
   mildest but still negative. Two further real defects that do not rescue the table: the published
   runs used lambda=0.01-0.10 where the paper states lambda=1, and the atom-type channel is inert
   (0 of 420 argmax flips at 10% of state RMS; `mean_atoms` identical in 20 of 24 shards).
   **Consequence for the paper:** Table 3's guidance claim does not survive. The salvageable
   contribution is the negative result plus its mechanism -- that 3D coordinate guidance cannot
   steer a graph-topology fingerprint -- which is a genuine finding about the task, not a failure of
   multiMFM. Authors' decision needed on whether to reframe Table 3 that way or drop it.
2. **Table 1** matched the n=32 pilot while the text said 1,000 pairs. **Decided: switch to
   n=1000** -- then superseded by item 8: the n=1000 run is also GLASS, so Table 1 becomes a dMFM
   rerun.
3. **Fig 4** used the purged `split65k` models, and its four baselines were hardcoded numbers.
   **Decided: rerun both panels on the parent-disjoint models with real baselines.** The left
   panel's producer was then recovered from DNA-MFM@187fe7b, so it is a port rather than a
   reconstruction; its protocol differs from the paper's description (a synthetic smooth reward,
   not f_cyc; a teacher-SDE reference, not GLASS). The baselines are implemented in
   `dmfm.benchmarks.search`, each tuned on a separate pilot at matched NFE. The published dMFM
   curve also fails to match its own surviving rerun (8-36% higher errors).
4. **Table 2 SS row mixed replicates** under four different rules. **Decided: rerun all six
   properties with 3 seeds and report mean +- SE**, for the multiMFM and multiMFM-SS rows alike.
5. **Text fixes.** All applied to `iclr2027_conferencemod.tex`; see `docs/paper_changes.md`.
   - Table 7 diag/consistency weights 0.35/10 -> 1.0/1.0.
   - App. F.6 t=0.3 -> t=0.5.
   - Table 12 covered the base stage only; now split into base, diagonal and ESD stages.
   - The GEOM MFM was distilled on 150k of 1.17M molecules (and QM9 on 40k of 50k).
   - Table 20's omitted GEOM settings added.
   - Algorithm 3 says steer-search stores the MFM look-aheads while the code harvested the base
     model's `endpoint_prediction`. **Decided: change the code to match Algorithm 3** and rerun.
6. **Fig 1** predates the toy MFM, so it may show plain GLASS guidance. **Decided: regenerate
   from the rebuilt toy code**, in both a GLASS-guided and a multiMFM-guided version, so the
   published figure can be identified.
7. **Tables 4-6 and App B.1** were computed on the purged `split65k` models, not the
   parent-disjoint models B.1 describes. **Decided: regenerate on parent-disjoint and report the
   weaker result** -- negative-target clustering moves from 0.9507x to 0.9872x of held-out.
8. **The DNA experiments labelled dMFM were run with GLASS on the base DFM (new, serious).**
   From the stored run arguments:
   - Table 1, Fig 3 and Fig 7: `sample_yeast_c0_guidance_hist.py` loads only the base DFM and the
     C0 guide and has no dMFM code path; the value gradient is 4 Euler steps of the GLASS
     posterior velocity. So SS 3.1's "We evaluate dMFM for DNA design" and App F.4's "tilting the
     dMFM" do not describe these runs. **Decided: rerun Table 1 with dMFM**, n=1000, paired.
     **Update (2026-09-29, supersedes the earlier "markedly worse" reading):** that reading came
     from a mis-wired configuration and is withdrawn. Two defects: the sampler composed the
     *diagonal* one-step student four times instead of loading the 4-step ESD student
     (`checkpoints/dna/dmfm_4step/L50`); and `guided_sample`'s `grad_clip` only clips *downward*,
     which interacts badly with the dMFM posterior. Its value gradient is heavy-tailed along the
     trajectory (mean norm 8-65, clip binds on 30-66% of guided steps) where GLASS's is not (1.1-7,
     binds on 2-17%), so the dMFM trajectory is dominated by a minority of saturated steps. Note
     this is *not* simple over-driving: under the stock rule, turning dMFM down makes it
     monotonically worse (guidance_frac 8->4->2->1 gives 0.729->0.792->0.898->1.109 at y*=+1).
     Making the step scale-free (rescale to exactly `grad_clip`) removes the tail, and only then
     does more strength help. With the correct student and the scale-free rule on *both* arms:
     matched `guidance_frac 8`, n=64 -- dMFM 0.189/0.153 vs GLASS 0.171/0.128 (paired
     +0.017+-0.018, +0.025+-0.019, ~1 SEM). Best-of-grid vs best-of-grid, n=32 -- dMFM
     (gf=16) 0.144+-0.020 / 0.165+-0.025 vs GLASS 0.150+-0.023 (gf=4) / 0.104+-0.014 (gf=8).
     At n=32 this looked like a tie at y*=+1. **FINAL, both targets complete at n=1000
     (`results/dna/c0_guidance/table1_dmfm_n1000_summary.json`, `_status: complete`): dMFM is not at
     parity -- it trails matched GLASS by 0.04-0.05 at both targets, 6-9 SEM.**

     | y* | dMFM unguided | dMFM guided | matched GLASS-4 guided | paired diff (dMFM - GLASS) | frac improved |
     |---|---|---|---|---|---|
     | +1 | 1.174 [1.149, 1.200] | **0.193 [0.184, 0.204]** | 0.153 [0.145, 0.160] | **+0.041 +- 0.006** | 0.975 |
     | -1 | 0.833 [0.806, 0.857] | **0.172 [0.164, 0.180]** | 0.124 [0.117, 0.130] | **+0.049 +- 0.006** | 0.954 |

     The n=64 "tie" (+0.017 +- 0.018) and the n=800 y*=+1-only figure were small-sample versions of
     the same effect; at n=1000 the sign is unambiguous and the two targets agree closely.
     **What this supports:** dMFM guidance reaches the target region -- oracle MAE 1.17 -> 0.19 and
     0.83 -> 0.17 against a parent-disjoint oracle, 97.5% / 95.4% of pairs improved, versus the
     mis-wired config's 0.610 / 0.408. **What it does not support:** parity with, or superiority to,
     GLASS. Do not claim either. For scale, published stock-rule GLASS n=1000 gave 0.148 / 0.119, so
     the dMFM row is modestly worse than the published numbers, not dramatically so.
     **Table 1's cells move:** published 0.157 / 0.153 become 0.193 / 0.172, and the in-text sentence
     becomes "reduces oracle MAE from 1.17 to 0.19 for y*=+1 and from 0.83 to 0.17 for y*=-1".
     Controls at full n: every arm's unguided column is bit-identical to the stored GLASS n=1000 run
     (1000/1000 sequences at both targets, `unguided_identical_across_arms` true), so the entire
     difference is attributable to the guidance term. All shards came from the h100 arrays
     49352242/49352244; the kempner_requeue hedges never started. The superseded
     `table1_dmfm_PRELIMINARY_partial.json` was deleted so nothing stale can be cited. The differing `guidance_frac` (16 vs 4-8) is mechanical -- normalising lowers
     the step wherever the raw gradient exceeded `grad_clip`, so a larger frac restores the same
     total drive -- and the matched-frac comparison is reported alongside so nothing rests on it.
     Two caveats: job 49353655, extending GLASS's scale-free grid to gf=16/32 for a symmetric
     top-of-range, was still pending, and if GLASS improves there the y*=+1 tie may move in its
     favour; and the published `guidance_frac 8` is not GLASS's optimum either (gf=4 is better at
     y*=+1). Evidence: `docs/dmfm_glass_parity.md`, jobs 49349568
     and 49350990. Slurm 49277977 is the mis-wired n=1000 run and **must not be printed as Table 1**;
     the rerun with `checkpoints/dna/dmfm_4step/L50` is outstanding.
   - Fig 5 and Tables 14-15: estimator *and* reference are GLASS on the base DFM. The paper
     contradicts itself, F.6 conceding this while F.2 and the Fig 5 caption claim dMFM.
     **Decided: rerun with dMFM finite-N against the GLASS-2048 reference.**
   - Not affected: Tables 13, 17 and 18 did load the dMFM students.
   - **Resolution chosen by the authors (2026-09-29): do not relabel the paper to say GLASS --
     rerun every DNA demo with dMFM, so the paper's dMFM claims are backed by dMFM runs.**
     Slightly different numbers are expected and acceptable. Parity (above) is what makes this
     viable. To keep the reruns mutually comparable and reportable, every dMFM demo uses the same
     two conventions, and **both must be stated in the paper**:
     1. **Student:** the artifact distilled for the number of composed jumps actually used --
        `paths.dmfm_ckpt_for_steps(L, nfe_value)`, i.e. the 4-step ESD student at 4 steps. Never
        the diagonal one-step student composed 4x (that is defect 1).
     2. **Gradient scale:** the scale-free rule (rescale the value gradient to exactly
        `grad_clip`), applied to *both* arms wherever a GLASS comparison is shown.
     3. **The (steps, MC) split, not just the NFE (added 2026-09-30).** The value gradient costs
        `steps x MC` network calls, so a budget can be spent many ways and the split is part of the
        method, not an implementation detail. Pinning both arms to 4 steps x MC 8 -- GLASS's
        estimator -- is what produced the apparent dMFM deficit: a one-step flow map affords MC 32
        at the same NFE, which is the whole point of having a flow map. At matched NFE 32 (n=600
        paired) dMFM 1x32 gives **0.146 / 0.120** against GLASS's own best 2x16 at 0.143 / 0.117,
        i.e. **+0.004 +- 0.006 and +0.003 +- 0.006, a tie at both targets**, and slightly ahead of
        GLASS 4x8 (0.149 / 0.129). MC 8->32 moves dMFM by 0.031/0.035 while MC 8->16 moves GLASS by
        -0.001/0.020, so dMFM carries the MC-noisier gradient and 4x8 spends its budget GLASS's way.
        **Table 1's dMFM row should be regenerated at 1 step x MC 32: 0.193/0.172 -> ~0.146/0.120.**
        Note that at 1 composed step, convention 1 selects the *diagonal* student -- the artifact the
        rest of the paper already uses. Evidence: `docs/dmfm_deficit_rescue.md`.
     Two corrections to `docs/dmfm_glass_parity.md` that follow: its §6 "best-of-grid tie at gf=16"
     does **not** reproduce (0.213 [0.189, 0.239] at n=200 against a claimed 0.144 +- 0.020), so the
     per-arm `guidance_frac 16` it motivated is not dMFM's optimum; and its student table cites an
     anneal setting that did not bind, since `_sample_s_u` applies it only for `gap_mode=random`
     while that run used `gap_mode=long`. Also note the deficit is 2.4-2.8x larger on the guide than
     on the oracle, so Table 1's oracle column understates it. The stock
        clip-down rule is not usable across posteriors: it makes the dMFM trajectory a minority of
        saturated steps, and it silently re-tunes strength when the posterior changes.
     Per-demo status: Table 1/Fig 3 (dna-core, n=1000); Fig 5/Tables 14-15 (`dna_sf_dmfm*`, dMFM
     finite-N vs GLASS-2048 reference, as the paper's caption claims); Fig 7, Fig 8 and the
     generated-sequence sets behind Tables 4-6 were regenerated from dMFM samples. Fig 6/Table 16
     are base-model marginals and unaffected.
     **Correction (2026-09-30):** the published Tables 4-6 sets were already **dMFM**, not GLASS
     (`DMFMSteeringRunner` differentiates through composed student maps), so Table 5's caption was
     right and `docs/dmfm_demo_reruns.md` is wrong on this point.
     **Withdrawn (2026-09-30):** the former carry-over caveat, that the guidance signal rides on the
     4-step posterior's discretisation error, is **refuted** -- see §7.9. Report the operating point
     as "N=8, ESS 1-3" (the reward temperature, not discretisation, is what limits the averaging).
9. **WITHDRAWN 2026-09-30: "a well-integrated posterior does not guide at all."** This was our
   error. The claim rested on one arm, `glass_euler32`, with two independent defects: it integrates
   32 Euler steps to the **singular data time t=1**, where the endpoint is a simplex vertex (argmax
   margin exactly 1.00000), V is piecewise constant and the true fp32 gradient norm is 1.2e-5 -- the
   gradient was numerically dead rather than small -- and it was the **only** arm never given the
   item-8 scale-free rule, so its step was thousands of times smaller than every other arm's.
   Fixing both (n=8, y*=-1, identical noise and MC pool, unguided 0.8651, job 49434328): published
   arm **0.8651, 0/8 improved**; `end_time` alone 0.6014; rule alone 0.7530 (4/8, a coin flip);
   **both fixed 0.1939, 8/8 improved**, against 0.1548 for the Table 1 4-step estimator (paired
   +0.039 +- 0.064, 0.6 SEM) on a **25x smaller gradient**. A pre-registered sign test (job 49442813)
   shows the claim's own reference integrator, GLASS RK4-25 at t=0.999 (gradient norm 0.202),
   delivers **89%** of the coarse estimator's per-step benefit, with -grad negative and both random
   controls at zero. **Correct statement:** guidance constants are not transportable between
   posterior samplers without renormalisation -- which is why the scale-free rule exists. App. F.4's
   paragraph and the former §7.9 headline must not ship as written. Note the QM9 DPS sweep is a
   *different* limit (it refines the outer guided ODE, dt-invariant; this refines the inner value
   estimator at fixed `nfe_traj`), so the two do not corroborate each other; the matched DNA arm is
   job 49440315, pending. Evidence: `docs/dna_findings_audit.md`.
10. **Still open for the authors.**
   - The abstract's "In both cases, our steering algorithms improve target attainment" now covers
     GEOM, where it does not, pending the `grad-sign` diagnostic.
   - SS 3.2's "steer-search integrates the same guided ODE as multiMFM" is false: the SS runs used
     mu = 11.0 against 3.4 for the non-SS row.
   - Whether the QM9 baseline sweep enters the paper, and in the main text or an appendix.
   - Table 7 lists about 10 h training against about 0.7 h recorded, plus a "GLASS NFE 8" entry
     matching no record; Table 9's L=400 column mixes two runs; Table 12's `Total loss` row
     matches no stage.
   - Pre-existing build blockers, in no agent's remit: `iclr2027_conference.sty` exists nowhere,
     two Fig 5 PDFs are missing, and 14 cited keys are absent from `bibliography.bib`.

## 8. Task list

### Phase 0: safety
- [x] Back up surviving artifacts to holylabs (2026-09-29)
- [x] Move MultiMFM from kozinsky_lab to albergo_lab netscratch (2026-09-29)
- [ ] Look for the toy notebook / Fig 1 sources on the Mac
- [x] `tullebulle/DNA-MFM` is reachable from the cluster; files recovered from 187fe7b are
      recorded with sha256 in `results/dna/original_scripts/_sources.json`

### Phase 1: decisions (§7)
- [x] All of §7.1-§7.7 decided by the authors, 2026-09-29
- [x] Apply the text-only fixes (§7.5)
- [ ] Land the reruns the decisions require (§7.1-§7.4, §7.6-§7.8)
- [ ] §7.9: the abstract's cross-dataset claim, the steer-search ODE sentence, the QM9 baseline
      sweep's place in the paper, and the Table 7/9/12 bookkeeping

### Phase 2: port code
- [ ] `src/dmfm/`: DNA-MFM@187fe7b + July files; replace flat script imports; drop dead defaults (`dinko/C0free_torch.pt`, `split65k` paths)
- [ ] Promote `analyze_parent_c0_diversity.py` into the package
- [ ] `src/multimfm/geom/`: fingerprint steering, GEOM teacher loader, multi-GPU MFM trainer; fix the 71 vs 82 atom target bug
- [ ] `third_party/tfg_flow`: GEOM fingerprint dataset, structure-head trainer, predictor patch
- [ ] Port the 2 geom-drug-tabasco fixes into `src/tabasco`
- [ ] Add a script for the non-SS QM9 row (§3.1 configs)
- [ ] Fix stale defaults: `property_targets.py` checkpoint + hist path, `train_mfm_student.py` hist path, PB config paths, GEOM leftovers in `glass_endpoints.py` / `glass_posterior.py`
- [ ] Install `mfm/` properly (`environment.yml` doesn't; `tabasco/sample/glass.py` relies on a path hack)
- [ ] Fix README training instructions
- [ ] Verify DNA code runs in the `multimfm` env; write the `evo2` env file

### Phase 3: artifacts
- [ ] Upload checkpoints + data to HF Hub; write the manifest
- [ ] Populate `results/` with the CSV/JSON behind each table

### Phase 4: notebooks for surviving results
- [ ] 02, 03, 04, 05, 07, 08. Check each against the paper's numbers.

### Phase 5: regenerate what was lost (GPU)
- [ ] Toy example: rebuild from fragments, retrain, regenerate Fig 1 (notebook 01)
- [ ] Fig 4: value-error benchmark + DNA baselines, on parent-disjoint models (06)
- [ ] Fig 6, Table 16, Fig 7, Fig 8 (02, 03)
- [ ] Evo2 scoring for Table 6 (09)

### Phase 6: release
- [ ] Anonymized mirror for submission
- [ ] `THIRD_PARTY.md`: add DNA-MFM, GEOM data, TFG-Flow additions, Evo2
- [ ] Zenodo DOI at camera-ready
