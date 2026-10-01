# Provenance of `results/dna/`

Collected 2026-09-29. Every file below was copied verbatim (unless marked *derived*) from the read-only backup

`B = /n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching`

which is a 2026-09-29 copy of `/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dirichlet-flow-matching`
(paths below are relative to that repo; `workdir/` = `B/workdir/`). Slurm logs are in `B/slurm/logs/`
(`slurm-<name>-<jobid>.out`). Scripts are in `B/scripts/` (the purged library modules are reconstructed in
`scratch_dfm_recon_20260925/`, identical to GitHub `tullebulle/DNA-MFM@187fe7b`). All GPU jobs ran in the
`seq_h100` env (Python 3.9, torch 2.1) unless stated otherwise. `_collect_manifest.json` lists source path and
sha256 of every copied file. Total size ≈ 4 MB.

Exception: `june_split65k_recovered/` (Tables 4–6) holds records whose files were purged from both netscratch and the
backup and which survive only as tool output inside a Codex transcript; see the last section.

The notebooks that consume these files are `notebooks/02`–`05` (and `06`, `09`, owned by other agents) (`_dna_helpers.py` replicates the original
bootstrap code). `paper_values.json` holds the numbers transcribed from `dMFM (40).pdf`.

## Table 1 — `c0_guidance/`

**Authors' decision 2026-09-29: Table 1 now reports the n = 1000 run** (`n1000/`); the submitted PDF printed the
n = 32 pilot (`pilot_n32/`), which notebook 03 still reproduces exactly and shows alongside.
`table1_n1000_summary.json` is *derived* (by `notebooks/_dna_helpers.py::build_table1_summary`, executed in
`notebooks/03_dna_cyclizability.ipynb`) from `n1000/*/oracle/oracle_per_sample.csv`, `*/summary.json` and
`c0_regressors/*`: every revised Table 1 cell, LaTeX rows, the §3.1 in-text numbers, the n = 32 values, and the
Fig 3 caption / Table 11 regressor metrics. CIs use the method of the submitted table (percentile bootstrap of the
mean, 2,000 resamples, `default_rng(0)` re-seeded per statistic, float32; `scripts/score_yeast_c0_oracle.py`); the
recomputation equals the `oracle_summary.csv` written by the original scoring jobs to ≤ 2.2e-16.


| File(s) | Source | Job / command |
|---|---|---|
| `pilot_n32/target_{1p0,m1p0}/{args.json,summary.json,sample_scores.csv}` | `workdir/yeast_parent_glass_pilot_target{1p0,m1p0}_g8.0_seed0/` (2026-07-26 08:30) | jobs 35242279 (+1) / 35242278 (−1), `slurm/sample_yeast_parent_c0_glass_pilot.sbatch` (copied here) → `scripts/sample_yeast_c0_guidance_hist.py` (script purged; reconstruction in `scratch_dfm_recon_20260925/scripts/`), flags in `args.json`: n_samples 32, batch 8, seed 0, mc 8, nfe_traj 64, nfe_value 4, guidance_frac 8, reward_sigma 0.15, reward_scale 0.5, t ∈ [0.5, 0.95], base `yeast_parent_a_L50_dfm_dit_h192_b4/best.pt`, guide `yeast_parent_c0_guide_2026-07-26/best_state.pt` |
| `pilot_n32/target_*/oracle/oracle_{per_sample,summary}.csv` | same dirs, `oracle/` | `scripts/score_yeast_c0_oracle.py --oracle_model_type park_cnn --bootstrap 2000 --seed 0 --reverse_complement_average`, oracle `yeast_parent_c0_oracle_2026-07-26/best_state.pt`. **Caveat:** the in-job oracle step failed and the standalone oracle jobs 35242533/35242534 were killed; these CSVs come from a later, unlogged run. A CPU re-score (2026-09-25) matches them to < 4e-4 per sample (see `diversity/n1000/README.md`). |
| `n1000/target_{1p0,m1p0}/…` (same layout) | `workdir/yeast_parent_glass_n1000_target{1p0,m1p0}_g8.0_seed0/` (2026-09-25) | jobs 48651189 (+1) / 48651190 (−1), `diversity/n1000/generate_n1000.sbatch` (copied), reconstructed sampler, identical flags except `--n_samples 1000`; oracle scored in the same job (logs `scratch_dfm_recon_20260925/n1000_logs/dfm_n1000_{p1,m1}-4865119?.out`) |

The posterior samples behind the guidance gradient come from **GLASS** on the base flow (`nfe_value` = 4 GLASS steps,
mc = 8); `args.json` has no dMFM checkpoint and the DNA repo's `docs/rebuttal_cyc.md` describes the run as "GLASS
posterior guidance (MC=8)". Paper §3.1 presents Table 1 as dMFM guidance (discrepancy, not resolved here).

## Tables 4–5, parent-disjoint recomputation (not the printed numbers) — `diversity/n1000/`

Source `workdir/rebuttal_parent_c0_diversity_n1000_20260925/` (CPU, 2026-09-25, multimfm env), command in its
`README.md` (copied): `analyze_parent_c0_diversity.py --run_dirs <the two n1000 dirs>` with defaults
`--baseline_n 1000 --seed 0 --max_pairwise_n 3000 --k_nearest_train 5 --size_matched_draws 1000`, using the
vendored DNA-MFM@187fe7b metric functions. `diversity/analyze_parent_c0_diversity.py` is the analyzer (from
`workdir/rebuttal_parent_c0_diversity_20260925/`). `diversity_per_sample.csv` is *derived*: the original with the
absolute `run_dir` column dropped (rows are identified by `target_c0` + `set`).
These parent-disjoint numbers are **not** the ones printed in the paper: paper Tables 4–5 come from the June 2026
`split65k` analysis, recovered in `june_split65k_recovered/` (below). Only the paper's `train_sample` row coincides
(= `full_sample` here: same 82,368 windows, same seed-0 draw).

## Tables 7–10 — `training/`

| File | Source | Job |
|---|---|---|
| `base_L{50,100,200,400}_args.json` | `workdir/yeast_parent_a_L*_dfm_dit_h192_b4/args.json` | 35140730 / 35140732 / 35140729 / 35140734 (2026-07-25), `slurm/train_yeast_parent_L*_dfm.sbatch` → `scripts/train_dna_dfm_dit.py` |
| `dmfm_L{50,100,200}_args.json` | `workdir/yeast_parent_a_L*_dmfm_diagonal_h192_b4/args.json` | 35247690 / 35247691 / 35247693 (2026-07-26), `slurm/train_yeast_parent_dmfm_diagonal.sbatch` → `scripts/train_dna.py` |
| `dmfm_L400_parent_esd_run_args.json` | `workdir/yeast_parent_a_L400_dmfm_diagonal_h192_b4/args.json` | 35247701 |
| `dmfm_L400_args.json`, `dmfm_L400_finetune_log.log` | `workdir/yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/{args.json,log.log}` | no Slurm log found (fine-tune from step 45,000 of the run above, `--batch_size 8`, no consistency loss) |
| `dmfm_4step_L*_args.json` | `workdir/yeast_parent_a_L*_dmfm_4step_esd_gap_upto025_corrected_20260727/args.json` | 35458568 / 35458569 / 35458567 / 35458570 (2026-07-27), `slurm/train_yeast_parent_dmfm_4step_continue.sbatch` |
| `checkpoint_summary.json` | *derived*: `torch.load` of `best.pt`/`last.pt` of the base runs and of the selected dMFM / 4-step checkpoints (steps, parameter counts per module, checkpoint-callback monitor) | — |
| `parent_split_seed0.json` | `data/yeast_parent_disjoint/parent_split_seed0.json` | 35241776 (2026-07-26), `slurm/prepare_yeast_parent_disjoint_data.sbatch` → `data/prepare_yeast_parent_splits.py` |
| `sbatch/*.sbatch` | `slurm/` | the training sbatch files named above |

Selected dMFM checkpoints: L50 `epoch=95-step=49000.ckpt`, L100 `epoch=80-step=39000.ckpt`, L200
`epoch=51-step=45000.ckpt`, L400 `yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/epoch=33-step=45750-v1.ckpt`.

## Table 11, Fig 3 caption — `c0_regressors/{guide,oracle}/`

`workdir/yeast_parent_c0_{guide,oracle}_2026-07-26/{metadata.json,test_predictions.csv,history.csv}`; job 35241778
(2026-07-26), `slurm/train_yeast_parent_c0_regressors.sbatch` → `scripts/train_parent_c0_regressor.py`
(`park_cnn`, RC augmentation + averaging; guide a_train/a_val seed 0, oracle b_train/b_val seed 1).

## Table 13, App. F.1 — `exact_rewards/{motif,conjunction}/L{50,100,200,400}/`

`workdir/rebuttal_reward_attainment_{motif,conjunction}_20260731/L*/{reward_attainment_samples.csv,
reward_attainment_summary.csv,metadata.json,evo2/evo2_summary.csv,evo2/evo2_per_sequence.csv}`.
Generation: jobs 35383601 / 35383624 / 35383648 / 35383653 (motif L50/100/200/400) and 35383663 / 35383669 /
35383675 / 35383680 (conjunction), `slurm/sample_parent_reward_attainment.sbatch` (copied) →
`scripts/sample_parent_reward_attainment.py` (seed 20260731, 100 pairs, bootstrap 2000; full args in each
`metadata.json`). Evo 2: jobs 35387510–35387523, `slurm/score_parent_reward_attainment_evo2.sbatch` (copied) →
`scripts/score_parent_reward_attainment_evo2.py`, model `evo2_7b_base`. (The directory suffix `_20260731` is a
label; files date from 2026-07-26.)

## Fig 5, Tables 14–15 — `scale_free_mc/`

| File | Source | Job |
|---|---|---|
| `gc/…` | `workdir/rebuttal_gradient_accuracy_gc_pairmetrics/` | 35338216 (2026-07-26) |
| `motif/…` | `workdir/rebuttal_gradient_accuracy_motif/` | 35258314 |
| `conjunction/…` | `workdir/rebuttal_gradient_accuracy_conjunction/` | 35258315 |

Each: `slurm/ablate_yeast_glass_gradient_mc.sbatch` (copied) → `scripts/ablate_glass_gradient_mc.py`
(32 test states, 8 pools, N = 1…256, reference GLASS-2048, t_eval 0.5, seed 20260726; `run_metadata.json`), then
rescored 2026-08-02 (no Slurm log) with `scripts/rescore_gradient_pairs_scale_free.py --input-dir <run>`
(defaults `--bootstrap 10000 --seed 20260802`), which wrote `gradient_scale_free_summary_all_lengths.csv` and
`L*/gradient_scale_free_per_state.csv` (copied). `scale_free_per_state_sums.csv` is *derived* here from the
runs' `L*/gradient_pairs.pt` (not copied, 1.9–15 MB each; they stay in the backup): per (reward, length, state, N)
the ℓ1 error summed over the 8 pools, the reference ℓ1 norm and the pool-mean cosine, float64.
`older_figures_dir/` = `workdir/rebuttal_gradient_accuracy_figures/*.csv` (an older "required MC" summary; not
used for the paper figure).

## Table 17 — `posterior_diversity/`

`workdir/parent_posterior_diversity_mapselected_flowmap1_glass100_20260726/{posterior_condition_metrics.csv,
posterior_diversity_summary.csv,run_config.csv}`; job 35378081 (2026-07-26),
`slurm/eval_parent_posterior_diversity.sbatch` (copied) → `scripts/eval_parent_posterior_diversity.py`
(seed 20260730, 32 sources × 16 futures, dMFM 1 map vs GLASS 100 steps). `posterior_futures.csv` (11 MB) was not copied.

## Table 18 — `dmfm_vs_glass/{gc,motif,conjunction}/`

`workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/<reward>/{gradient_mae_summary_all_lengths.csv,
run_metadata.json,L*/gradient_mae_per_condition.csv,L*/metadata.json}`; job array 35642548 (tasks 0–11,
2026-07-27), `slurm/evaluate_dmfm4_glass_mae_precision.sbatch` (copied) → `scripts/ablate_dmfm_one_step_gradient_mc.py`
(4-step students, 32 states, 8 nested pools, GLASS-128 RK4-50 reference, **t_eval 0.5**, seed 20260802), MAE
files written by `scripts/rescore_gradient_pairs_mae.py` (2026-07-27 23:33, no Slurm log).

## Fig 4 (right) — `benchmarks/`

`fig4_right_hardcoded_baselines.csv`: verbatim copy of the `OLD_NON_DMFM_BASELINES` string in
`scripts/benchmark_yeast_split_dmfm_steering.py` (lines 28–48). No code producing these baseline numbers exists in
the repo. `benchmark_script_defaults.json`: the same script's argparse defaults, transcribed by hand (only record of
the dMFM configuration; the run dir `workdir/yeast_split_dmfm_steering_benchmark_2026-05-13_12-51-45` is empty).

`recovered_2026-05-13/` — a dMFM rerun of the Fig 4 (right) benchmark, **not** the plotted run (see
`figure_digitized/`): `nfe_benchmark_agg.csv` and `args.json` are verbatim tool output (`sed -n` of
`workdir/yeast_split_dmfm_steering_benchmark_2026-05-13_12-51-45/{nfe_benchmark_agg.csv,args.json}`) from
`~/.codex/sessions/2026/06/03/rollout-2026-06-03T13-57-06-019e8ea1-c496-7e71-9b42-7598ccca1ab7.jsonl` lines 105/106;
`dmfm_guidance_repeat_rows.column_output.txt` is `column -t` output of that run's per-repeat CSV
(`~/.codex/sessions/2026/07/26/rollout-2026-07-26T14-54-34-019f9fc7-6b22-7a43-83ce-30cbdbdebf12.jsonl` line 193),
`dmfm_guidance_repeat_rows.csv` is *derived* from it. The run's args are the script defaults above (split65k models,
target 0.3); its dMFM errors are 8–36% above the plotted ones. Line numbers and sha256: `_recovery_manifest_may2026.json`.

## Fig 6 / Table 16 inputs — `base_model/` (figures now owned by notebook 09)

Fig 6 / Table 16 and Figs 7–8 moved to `notebooks/09_dna_regenerated.ipynb` (dna-regen agent, results in
`results/dna/regen/`); notebooks 02/03 only point there. The first-round analog figures stay as files only:
`notebooks/figures/dna_fig6_analog_parent_disjoint.png`, `notebooks/figures/dna_fig7_analog_n1000.png`.

`real_base_freqs_L50.csv` is *derived* from `data/yeast_parent_disjoint/yeast_parent_L50.pt` and
`yeast_parent_L50_split_seed0.pt`: per-position A/C/G/T frequencies (tokens 0–3 = A, C, G, T) for all windows,
`a_train` and `test`. Used for an analog of Fig 6 on the parent-disjoint model; the paper's Fig 6 data are lost.

## Tables 4–6 (paper numbers) — `june_split65k_recovered/`

The run directories `workdir/yeast_split_dmfm_c0_samples_target_{p1,m1}_2026-06-02/`,
`workdir/yeast_split_dmfm_c0_diversity_2026-06-0{2,9}/`, `workdir/yeast_split_dmfm_c0_evo2_2026-06-10/` and the tracked
copies `docs/results/yeast_split_dmfm_c0_{diversity_summary_2026-06-09,evo2_summary_2026-06-10}.csv` and
`docs/clean_split_sequence_validation.md` are empty/gone in netscratch **and** in the backup, as are the `split65k`
models and `data/yeast_mid50_split_seed0_train65000_val8000.pt`. Their content was printed by shell commands in the
Codex session `~/.codex/sessions/2026/06/03/rollout-2026-06-03T13-57-06-019e8ea1-c496-7e71-9b42-7598ccca1ab7.jsonl`
(started 2026-06-03, resumed through 2026-07-25; the 2026-07-25/26 rollouts in the same directory tree are forks
with copies of the same lines). The files here are those tool outputs, extracted programmatically (not retyped) from
the JSON line given below; `_recovery_manifest.json` lists line numbers and sha256.

| File | Transcript line | Command whose stdout it is |
|---|---|---|
| `yeast_split_dmfm_c0_evo2_summary_2026-06-10.csv` (Table 6) | 2229 (2026-06-10 20:18) | `sed -n '1,20p' docs/results/yeast_split_dmfm_c0_evo2_summary_2026-06-10.csv` — complete file (header + 8 rows); printed identically again at line 2341 |
| `yeast_split_dmfm_c0_diversity_summary_2026-06-09.head5.csv` (Table 5) | 1055 (2026-06-09 22:31) | `head -5 docs/results/yeast_split_dmfm_c0_diversity_summary_2026-06-09.csv` — header + the 4 generated rows, full precision |
| `yeast_split_dmfm_c0_diversity_summary_2026-06-09.selected_columns.txt` (Table 4) | 1003 (2026-06-09 22:29) | pandas `to_string` of 8 columns of `workdir/yeast_split_dmfm_c0_diversity_2026-06-09/diversity_summary.csv`, all 8 rows (micromamba lock warnings dropped) |
| `yeast_split_dmfm_c0_diversity_2026-06-02.diversity_summary.csv` | 81 (2026-06-03 18:00) | `sed -n '1,20p' workdir/yeast_split_dmfm_c0_diversity_2026-06-02/diversity_summary.csv` — complete earlier summary of the same sample sets (4 generated rows + `train_sample`) |
| `samples_target_{m1,p1}_2026-06-02.summary.json` | 82, 83 | `sed -n '1,220p' workdir/yeast_split_dmfm_c0_samples_target_{m1,p1}_2026-06-02/summary.json` |
| `samples_target_{p1,m1}_2026-06-02.args.json`, `teacher_split65k_args.json` | 2730 (2026-07-25) | `sed -n '1,240p'` of the two sample runs' `args.json` and of `workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/args.json` |
| `clean_split_sequence_validation.md` | 2569 (2026-07-25) | `sed -n '1,220p' docs/clean_split_sequence_validation.md` (whole report, written by `scripts/write_yeast_split_diversity_markdown.py` from the two summaries; its tables are the paper's Tables 4–6; the 2026-06-10 version at line 2278 differs only in printing `49.0000` for the scored bases) |

What the records say about these experiments (all at L = 50, n = 1,000 per set):
- Sampling: `scripts/sample_yeast_split_dmfm_c0.py` via `slurm/sample_yeast_split_dmfm_c0.sbatch` (2026-06-02), teacher
  `yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt`, dMFM
  `yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35/last.ckpt`, guide `dinko/C0free_torch.pt`,
  mc 16, guidance_frac 8, reward σ 0.15, reward_scale 1.0, guidance t ∈ [0.01, 0.95], nfe_sample 96, seeds 0 (+1) / 1 (−1).
- Diversity: `scripts/analyze_yeast_c0_diversity.py --train_pt data/yeast_mid50.pt --split_pt
  data/yeast_mid50_split_seed0_train65000_val8000.pt --targets -1 1 --train_baseline_n 1000`, out dir
  `workdir/yeast_split_dmfm_c0_diversity_2026-06-09` (command in the recovered report).
- Evo 2: `scripts/score_yeast_c0_evo2.py … --baseline_n 1000 --batch_size 1 --bootstrap 2000 --model_name evo2_7b_base`
  in `.envs/evo2` (command in the recovered report). Per-sequence Evo 2 scores were never printed and are lost, so the
  Table 6 CIs cannot be recomputed. The scorer source (`.py` purged, `.pyc` survives) was written in the same session
  (`apply_patch` at transcript line 1485, 2026-06-10; later edits may exist).
- These are the `split65k` models (random window split 65,000 / 8,000 / 9,368), **not** the parent-disjoint models
  that App. B.1 names.

## Figs 4 and 6: plotted values — `figure_digitized/`

The figure PDFs `MC_value.pdf`, `nfe_benchmark.pdf` (Fig 4) and `base.pdf`, `gen_vs_real.pdf`, `marginals.pdf` (Fig 6) in
`/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dMFM/figures/` (backup: `paper_backup_dmfm_20260929/dMFM/figures/`) are
matplotlib vector output and identical to the drawings in `dMFM (40).pdf` (checked 2026-09-29: Fig 4 line vertices agree
after an affine page transform to 2e-5 pt; Fig 6 bar heights agree up to one scale factor; all 1,002 Fig 6 heatmap cell
colours agree). `digitize_paper_figures.py` (needs PyMuPDF; run as
`python digitize_paper_figures.py --fig-dir <dMFM>/figures`, 2026-09-29) maps them back to data coordinates using the
axis ticks in the same PDF: `fig4_right_nfe_benchmark_digitized.csv` (all 5 series, points and error bars),
`fig4_left_MC_value_digitized.csv` (3 series × N = 1…512), `fig6_global_marginals_digitized.csv` (the 12 bar heights),
`fig6_heatmaps_digitized.csv` (every heatmap cell; colormap-LUT inversion, viridis 0–0.35 and coolwarm, value = bin
centre, `bin_half_width` given). These are the paper's plotted values, not an independent record. Checks: the digitized
Fig 4 baselines equal `benchmarks/fig4_right_hardcoded_baselines.csv` to 1e-6; PDF creation dates are 2026-05-07
(`nfe_benchmark.pdf`), 2026-05-04 (`MC_value.pdf`) and 2026-05-03 (Fig 6), i.e. before the `split65k` models
(2026-05-12) and the parent-disjoint models. Table 16 recomputed from the Fig 6 values does **not** match the printed
table (global uniform-vs-real r = −0.852 vs 0.1813; global generated-vs-real 0.9988 vs 0.9991; position-wise 0.9835 vs
0.9863 and 0.065 vs 0.0264).

## Fig 7 — `c0_guidance/may2026_target0p3_recovered/`

The Fig 7 runs (`workdir/yeast_c0_guidance_hist_2026-05-02_*`, target C0 = 0.3, base flow
`yeast_mid50_dfm_dit_h192_b4_lr5e-5_2026-05-02_13-06-17/last.pt`, guide `dinko/C0free_torch.pt`) are empty in netscratch
and the backup. `summary_json_extracts.txt` is verbatim tool output (a loop printing the lines of every run's
`summary.json` that match `target|mae|mean|guided|unguided|distance`) from
`~/.codex/sessions/2026/07/26/rollout-2026-07-26T14-54-34-019f9fc7-6b22-7a43-83ce-30cbdbdebf12.jsonl` line 193;
`summary_json_extracts.csv` is *derived* from it (one row per run). Per-sample scores and each run's guidance strength
were never printed; the launcher defaults at the time (Cursor transcript
`~/.cursor/projects/n-netscratch-kozinsky-lab-Lab-uunneberg-repos-dirichlet-flow-matching/agent-transcripts/1fa0bd60-82fd-4805-9df1-bc7e58f80f74/…jsonl`
line 3) were TARGET_C0 0.30, REWARD_SIGMA 0.15, REWARD_SCALE 0.5, MC 128, NFE_TRAJ 128, NFE_VALUE 12. The paper's three
panels are raster images (`histogram4.png`, `histogram2.png`, `histogram.png`), so they cannot be digitized exactly.

## Original producer scripts — `original_scripts/`

Verbatim copies (2026-09-29) of four scripts believed lost, recovered from a clone of
`github.com/tullebulle/DNA-MFM@187fe7b` that a sibling agent had left in a `/tmp` scratch directory; `/tmp` is not
durable, so they were copied here. These are the **originals**, not reconstructions. Hashes in `_sources.json`; the
matching `slurm/*.sbatch` launchers are included. (`make_yeast_split_sample_figures.py` is also at
`scripts/dna/regen/original/`, placed there by the dna-regen agent.)

Three things they settle:

1. **`benchmark_yeast_split_dmfm_steering.py` — Fig 4 (right) baselines.** Its `OLD_NON_DMFM_BASELINES` string parses to
   a frame **identical** to `benchmarks/fig4_right_hardcoded_baselines.csv` (20 × 11, exact). That file was transcribed
   in round 1 from a transcript; it is now verified against the true source. The baselines remain hardcoded numbers with
   no producing code anywhere, and `mae_sem` is 0 in every row.
2. **`sample_yeast_c0_guidance_hist.py` — the Table 1 sampler.** The reconstruction in
   `scratch_dfm_recon_20260925/scripts/`, which generated the n = 1000 rerun, differs from this original in **17 lines,
   all of them the guide-model class only**: the original imports `C0Torch` (`dinko.torch_model`) and wraps inputs with
   `dna_probs_to_park_input`, the reconstruction calls `ParkC0Regressor` directly (it applies the same permute
   internally). The GLASS velocity, the two-pass MC estimator, the guidance schedule and the RNG stream are unchanged,
   so the n = 1000 rerun is faithful to the original algorithm. (The July 2026 parent-disjoint runs passed a `park_cnn`
   checkpoint, so they must have used a later revision of this script than commit 187fe7b.)
3. **`benchmark_yeast_split_value_mc.py` — Fig 4 (left), whose protocol was unknown.** It shows the panel does **not**
   use the cyclizability predictor: the reward is a synthetic `SmoothDNAReward` (random linear + pairwise terms plus a
   fixed 8-base motif, `--reward_gain 4.0`, `--seed 12345`). The reference \(V_t\) is **teacher SDE sampling with
   `--sde_ref_n 512` samples and `--sde_steps 1024`**, not a GLASS reference; \(V=\operatorname{logsumexp}(r)-\log N\);
   the plotted quantity is the mean of \(|V_{\text{est}}-V_{\text{ref}}|\) over `--n_conditions 8` held-out `test`
   conditions with outer time drawn in `[--tcond_low 0.05, --tcond_high 0.35]`; `--dmfm_nfe 1`, `--fmap_nfe 256`. The
   script also emits an "SDE MC floor" series that the published panel does not show. Its default `--n_list` stops at
   256 while the figure plots N = 1…512, so the plotted run passed a custom `--n_list`.


## Notebook RERUN=True verification (GPU)

Notebooks 02-05 were executed with `RERUN = True` on a GPU to show that the ported package
reproduces the stored full-size results at small n. The repo copies stay in `RERUN = False`
state; executed copies live under `outputs/dna/notebook_rerun/`.

| Job | Partition | Elapsed | What it showed |
|---|---|---|---|
| 49281310 | gpu_test (A100 MIG 3g.20gb) | 4:16 | All four notebooks, zero cell errors. Table 11 reproduced from the checkpoints (guide MAE 0.158756 vs 0.158772 stored; oracle 0.161984 vs 0.161985); 8/8 unguided sequences identical to the stored Table 1 run; motif target score exact; Table 17 55/56 conditions bit-identical; Table 18 16/16 per-condition MAEs (median 0.72% relative). |
| 49285867 | gpu_test | 3:47 | Re-confirmation after three threshold fixes: Table 11 12/12, Table 18 same-subset check 4/4, and all six Table 1 cells containing the n=1000 value in the 8-pair bootstrap CI. Executed copies in `outputs/dna/notebook_rerun/executed_49285867/`. |

Summarise either run with `python scripts/dna/summarize_rerun.py <executed dir>`.

Three small-n flags remain, each explained in the cell that raises it and none of them a port bug:
the cosine-versus-N Spearman check at 2 states x 2 pools (notebook 05), Table 18's gc L=50 MC=8
cell against a 2-state band (05), and one guided row on the same 8 ids falling outside its CI (03).
These are small-sample artifacts: guided sampling is not bit-reproducible, and the stored runs show
comparable spread on the same subsets.

## Table 1 with dMFM guidance — `c0_guidance/dmfm_n1000/`, `c0_guidance/glass_matched_n1000/`, `c0_guidance/table1_dmfm_n1000_summary.json` (dna-core, 2026-09-29)

Authors' decision 2026-09-29: Table 1 reports **dMFM** guidance. The GLASS numbers above (`n1000/`,
`pilot_n32/`, `table1_n1000_summary.json`) are kept as provenance only.

Producer for both arms: `python -m dmfm.experiments.sample_c0_guidance_dmfm --paired` via
`scripts/dna/table1_c0_guidance_dmfm.sbatch`, identical to the GLASS Table 1 protocol (reward_sigma 0.15,
reward_scale 0.5, mc 8 with the frozen per-sample pool `seed + 1e6 + i`, nfe_traj 64, t_max 0.95,
window [0.5, 0.95], guidance_frac 8, coeff_cap 10, grad_clip 10, seed 0, batch 8, per-sample initial
noise `seed + i` shared by each guided/unguided pair) except for two deliberate changes, applied to
**both** arms:

1. the dMFM posterior is `checkpoints/dna/dmfm_4step/L50` composed in 4 steps — the artifact distilled
   on gaps [0, 0.25] and selected on `val_mfm_probe_esd_gap_0.25`, i.e. Table 18's 4-step pairing
   (`paths.dmfm_ckpt_for_steps(50, 4)`; the sampler refuses a mismatched pairing without
   `--allow_gap_mismatch`);
2. `GRAD_NORMALIZE=1` (`--grad_normalize`): the value gradient is rescaled to exactly `grad_clip`
   instead of only clipped down. Table 1's constants were tuned to the GLASS-4 gradient norm (~1-3,
   clip binding on 1-5% of guided steps) while a dMFM posterior's is 5-125 and binds on 6-52%, so under
   the stock rule swapping the posterior multiplies the effective drive 5-12x and the comparison
   measures drive rather than posterior quality. The rule is applied identically to GLASS; nothing is
   tuned per arm. See `docs/dmfm_glass_parity.md`.

| Arm | Slurm (one tree, two routes each) | Shard tree | Small results |
|---|---|---|---|
| dMFM, 4-step student, 4 steps, scale-free | 49352242 (kempner_h100), 49357778 (kempner_requeue) | `outputs/dna/table1_dmfm4_norm_n1000/` | `c0_guidance/dmfm_n1000/` |
| matched GLASS-4, scale-free | 49352244, 49357782 | `outputs/dna/table1_glass_norm_n1000/` | `c0_guidance/glass_matched_n1000/` |

Both conventions are the wrapper's defaults, so a resubmit cannot silently produce the superseded
clip-down configuration: omitting `GRAD_NORMALIZE` gives the scale-free rule with the 4-step student,
`GRAD_NORMALIZE=0` warns and names plan §7 item 8, and `LEGACY_DIAG_STUDENT=1` reproduces the withdrawn
job in full (asserted by `tests/dna/test_script_args.py::test_table1_dmfm_guidance_rule_defaults_safe`).

**Final numbers** (2026-09-30, all 20 shards complete, n = 1000 per target per arm), in
`c0_guidance/table1_dmfm_n1000_summary.json` with `_status: complete`. Oracle-scored, park_cnn,
reverse-complement averaged, percentile bootstrap of the mean with 2,000 resamples and
`default_rng(0)` re-seeded per statistic:

| arm | y* | unguided MAE | guided MAE | paired improvement |
|---|---|---|---|---|
| dMFM, 4-step student, scale-free (Table 1) | +1 | 1.174 [1.149, 1.200] | **0.193 [0.184, 0.204]** | 0.980 [0.953, 1.009] |
| dMFM, 4-step student, scale-free (Table 1) | -1 | 0.833 [0.806, 0.857] | **0.172 [0.164, 0.180]** | 0.660 [0.633, 0.687] |
| matched GLASS-4, scale-free (`matched_glass_arm`) | +1 | 1.174 | 0.153 [0.145, 0.160] | 1.021 [0.994, 1.048] |
| matched GLASS-4, scale-free (`matched_glass_arm`) | -1 | 0.833 | 0.124 [0.117, 0.130] | 0.709 [0.682, 0.735] |

Paired per-sample difference (`dmfm_vs_glass_matched`), dMFM minus GLASS on the same 1,000 ids:
**+0.041 +- 0.006** at y*=+1 and **+0.049 +- 0.006** at y*=-1, so dMFM trails the matched GLASS arm by
about 0.04-0.05 at both targets rather than matching it. Every arm's unguided column is bit-identical
to the stored GLASS n=1000 run (1000/1000 sequences at both targets), which is what attributes the
difference to the guidance term alone. A `table1_dmfm_PRELIMINARY_partial.json` existed while the
shards were queued and was deleted once the full run landed.

**The reward is not eq (17):** every DNA C0 run, including the submitted pilot, uses
`reward_scale * -0.5((f - y*)/reward_sigma)^2 = -11.111 (f - y*)^2`. Inside `log mean_j exp(r_j)` that
constant is an inverse temperature, so it changes the gradient direction and cannot be absorbed into the
tuned strength. Recorded as `_reward` in the summaries; a disclosure gap, not a defect.

Collected by `bash scripts/dna/collect_table1_dmfm.sh 49352242 49352244`
(`dmfm.experiments.collect_table1_c0`): merge shards, oracle-score with `score_c0_oracle
--oracle_model_type park_cnn --reverse_complement_average --bootstrap 2000 --seed 0`, copy the small
results, and write the summary. Its `n1000` holds the dMFM cells (the Table 1 numbers),
`matched_glass_arm` the GLASS baseline under the same rule, `dmfm_vs_glass_matched` the per-sample
paired difference with SEM, `_execution` the device/account/partition per shard,
`_unguided_vs_stored_glass_n1000_run` the count of unguided sequences identical to the stored GLASS
n=1000 run, and `_caveat_posterior_accuracy` the finding that a *more accurate* posterior guides less
(GLASS-32 does not guide at all), so the guidance constants cannot be transported between samplers.

**Withdrawn: Slurm 49277977**, `outputs/dna/table1_dmfm_n1000/` (complete, kept on disk, recorded in the
summary's `_withdrawn_runs`). It composed 4 steps of the *diagonal* student `checkpoints/dna/dmfm/L50`,
which the rest of the paper uses as a one-step map, and used the stock clip-down rule. Oracle-scored at
n=1000 it gives guided MAE 0.610 (+1) / 0.408 (-1) against the GLASS run's 0.148 / 0.119. Do not print it
as Table 1 (`docs/REPRODUCIBILITY_PLAN.md` §7 item 8). Its 1000/1000 unguided sequences per target are
identical to the stored GLASS n=1000 run, which is what isolates the difference to the guidance term.
Cancelled while pending, same configuration: 49293510 (CPU), 49298862 and 49308261 (gpu_requeue).
