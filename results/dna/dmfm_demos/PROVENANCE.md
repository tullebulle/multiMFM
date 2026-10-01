# results/dna/dmfm_demos -- the DNA demos rerun with the dMFM posterior

Written 2026-09-29/30 by the "dmfm demos" agent of the reproducibility round.
Companion write-up: `docs/dmfm_demo_reruns.md`. Decision being implemented:
`docs/REPRODUCIBILITY_PLAN.md` S7 item 8, "Resolution chosen by the authors".

**What this directory is.** Fig 7, Fig 8 and the generated-sequence sets behind Tables 5 and 6
were produced with GLASS guidance on the base DFM while the paper presents them as dMFM
demonstrations. The authors chose to rerun them with dMFM rather than relabel the paper. This
directory holds the dMFM reruns and their matched-GLASS controls.

**Nothing here overwrites the GLASS-era results.** Those stay where they are and remain
citable:

| GLASS-era artifact | stays at |
| --- | --- |
| published (purged `split65k`) Tables 4-6 | `results/dna/paper_values.json`, `results/dna/june_split65k_recovered/` |
| parent-disjoint GLASS recompute of Tables 4-5 | `results/dna/diversity/regen/` (and round-1 `results/dna/diversity/n1000/`) |
| GLASS n=1000 C0 sets those were computed from | `results/dna/c0_guidance/n1000/` |
| GLASS regeneration scripts for Fig 7 / Fig 8 | `scripts/dna/regen/fig7_guidance_sweep.sbatch`, `scripts/dna/regen/fig8_guidance_trace.sbatch` (both still runnable) |

## The two mandatory conventions (S7 item 8)

Every run recorded here uses both, and both are recorded in each run's `args.json` /
`metadata.json`:

1. **Student = the artifact distilled for the number of composed jumps used.** 4 composed
   flow-map steps => `paths.dmfm_ckpt_for_steps(50, 4)` =
   `checkpoints/dna/dmfm_4step/L50/epoch=29-step=15000.ckpt` (the 4-step ESD student). Never
   the diagonal one-step student composed 4x.
2. **Scale-free gradient rule** (`--grad_normalize`): the value gradient is rescaled to exactly
   `grad_clip = 10` instead of only clipped down, applied to **both** arms wherever a GLASS
   comparison is shown.

Carry-over caveat that no rerun fixes (S7 item 9): this is an operating point, N = 8 at 4
posterior steps, not an asymptotically accurate value gradient. See `docs/dmfm_glass_parity.md`.

## Producers

| script | what it runs |
| --- | --- |
| `scripts/dna/dmfm_demos/fig7_tilt_ladder.sbatch` | Fig 7 guidance-strength ladder, both arms |
| `scripts/dna/dmfm_demos/fig8_trace_dmfm.sbatch` | Fig 8 single-sample trajectory trace |
| `scripts/dna/dmfm_demos/tables4_5_diversity_dmfm.sh` | Tables 4-5 diversity statistics (CPU) |
| `scripts/dna/dmfm_demos/table6_evo2_dmfm.sbatch` | Table 6 Evo 2 plausibility (evo2 env) |
| `scripts/dna/dmfm_demos/collect.py` | oracle-scores Fig 7 runs and copies the small files here |

Code paths added (old behaviour is the default in both cases, so no stored result changes):

* `src/dmfm/regen/guidance_trace.py`: `--posterior {glass,dmfm}`, `--student_ckpt`,
  `--allow_gap_mismatch`, `--grad_normalize`, plus a `grad_norm_applied` column in `trace.csv`.
  Without them the module behaves exactly as before (`--posterior glass`, clip-down).
* Fig 7 needed no new code: `dmfm.experiments.sample_c0_guidance_dmfm --paired` already had
  `--posterior` and `--grad_normalize`, and its GLASS mode is numerically identical to the
  published `sample_c0_guidance` sampler
  (`tests/dna/test_api.py::test_paired_glass_equals_table1_sampler`).

## Runs

### Fig 7 -- `fig7/`

Slurm **49425774** (array 0-13, throttle raised from 4 to 6 mid-run to clear a stuck
`JobArrayTaskLimit`), `kempner_requeue` / `kempner_albergo_lab`, 1 GPU per task, 40 min walltime,
2026-09-30. Tasks 0-6 = dMFM arm, 7-13 = matched GLASS arm, at
`guidance_frac in {0, 1, 2, 4, 8, 16, 32}`. Measured 4.9-5.6 min per task at n = 1000, i.e.
**~0.6 GPU-h per 7-point arm** (the GLASS-launcher-estimator figure for Fig 7 was ~46 GPU-h; see
`docs/dmfm_demo_reruns.md`). `kempner_requeue` is preemptible and several tasks were requeued;
a task whose `summary.json` exists exits immediately, so a requeue costs at most one task.

Task 9 (`glass gf = 2`) failed on its first attempt with
`RuntimeError: CUDA error: no kernel image is available for execution on the device`: it landed on
an `NVIDIA RTX PRO 6000 Blackwell Server Edition` node of the `kempner_requeue` pool, compute
capability `sm_120`, which the pinned torch 2.5.1+cu124 does not build for. Resubmitted with
`--constraint="a100|h100|h200"` as Slurm **49436287**. Any GPU job on `kempner_requeue` needs that
constraint. Task 5 (`dmfm gf = 16`) was still `PENDING (Priority)` at the end of the session; the
ladder in `fig7/ladder_summary.csv` therefore holds 12 of 14 points, and
`python scripts/dna/dmfm_demos/collect.py --what fig7` completes it once the last two land.

Settings, both arms: target C0 = 0.30, n = 1000 paired samples, seed 0, batch 8, MC = 8
(frozen per-sample pool `seed + 1e6 + i`), initial noise `seed + i`, `nfe_value 4`,
`nfe_sample 64`, `t_max 0.95`, guidance window [0.50, 0.95], `coeff_cap 10`, `grad_clip 10`,
`--grad_normalize`, `reward_sigma 0.15`, `reward_scale 0.5`. dMFM arm: posterior `dmfm`,
4 composed flow-map steps of the 4-step ESD student. GLASS arm: posterior `glass`, 4 Euler
steps on the base DFM -- i.e. the Table 1 estimator, *not* the published Fig 7 launcher's
MC=128 / nfe_value=12 estimator (see `docs/dmfm_demo_reruns.md` for why).

Strength ladder and the three-panel assignment were fixed before any output was inspected:
panels (a)/(b)/(c) = `guidance_frac` 0 / 8 / 32.

Run dirs: `outputs/dna/dmfm_demos/fig7/<arm>/gf<g>_mc8_n1000_seed0/`; one line per array task
in `outputs/dna/dmfm_demos/fig7/jobs.tsv`.

| file | content |
| --- | --- |
| `fig7/ladder_summary.csv` | one row per (arm, `guidance_frac`): guide- and oracle-scored means, MAE to target, fractions within 0.05/0.10, paired improvement + bootstrap CI |
| `fig7/index.csv` | the same rows under the column names `notebooks/03_dna_cyclizability.ipynb` already reads from the GLASS regeneration's `fig7/index.csv`, so those cells need only a path change; `scores_csv` is relative to `results/dna/dmfm_demos` |
| `fig7/<arm>/<tag>/scores.csv` | per-sample guide and oracle scores, no sequences (~30 kB); enough to redraw the exact histograms, and what `dmfm.regen.plots.fig7` reads |
| `fig7/histograms.csv` | 50-bin guide-score histograms on the fixed range [-2, 2] for every (arm, strength, guided/unguided) -- what the figure plots |
| `fig7/<arm>/<tag>/args.json`, `summary.json`, `run_env.json`, `histogram.png`, `cdf.png`, `oracle/oracle_summary.csv` | per-run record |

Oracle column: `dmfm.experiments.score_c0_oracle --oracle_model_type park_cnn
--reverse_complement_average --bootstrap 2000 --seed 0`, oracle
`checkpoints/dna/c0/oracle/best_state.pt`, run on CPU by `collect.py` -- the Table 1 protocol.

### Fig 8 -- `fig8/`

`dmfm.regen.guidance_trace`, run on **CPU** on the login node, 2026-09-30, ~3 min per arm
(`logs/fig8_{dmfm,glass}_cpu.log`). CPU is deliberate: the initial noise and MC pool are drawn
by seeded CPU generators, so CPU and GPU runs share identical randomness, and a CPU run is
additionally deterministic (the GPU value-gradient backward is not). Fig 8 illustrates one
trajectory and no published number depends on it.

Settings, both arms: target C0 = 0.30, seed 0, `sample_ids 0 1 2 3`, MC = 8, `nfe_value 4`,
`nfe_traj 64`, `t_max 0.95`, guidance window [0.50, 0.95], `guidance_frac 8`, `coeff_cap 10`,
`grad_clip 10`, `--grad_normalize`, `reward_sigma 0.15`, `reward_scale 0.5`.

**Sample selection rule, fixed before any output was looked at:** all four of `sample_ids
0 1 2 3` are recorded, and the figure shows `sample_idx 0`. Nothing is selected on appearance.

| file | content |
| --- | --- |
| `fig8/dmfm/trace.csv`, `metadata.json` | dMFM arm: one row per (step, arm, sample) with `grad_norm`, `grad_norm_applied`, `value`, `endpoint_hard_score`, `guidance_coeff`, `guidance_applied` |
| `fig8/glass/trace.csv`, `metadata.json` | matched GLASS control, same settings |

### Tables 4-5 -- `diversity/`

`dmfm.experiments.analyze_c0_diversity` via `scripts/dna/dmfm_demos/tables4_5_diversity_dmfm.sh`,
CPU. Input sets are the Table 1 dMFM n=1000 sets produced by the `dna-core` agent
(Slurm **49352242**, `outputs/dna/table1_dmfm4_norm_n1000/target_{1p0,m1p0}/merged/`) and the
matched GLASS control (Slurm **49352244**,
`outputs/dna/table1_glass_norm_n1000/target_{1p0,m1p0}/merged/`). Those sets use exactly the
two conventions above, so Tables 4-6 are consistent with Table 1 / Fig 3 by construction.

Analyser defaults are the published ones: `--baseline_n 1000 --seed 0 --max_pairwise_n 3000
--k_nearest_train 5 --pair_chunk 128 --size_matched_draws 1000`, data
`data/dna/yeast_parent_disjoint/yeast_parent_L50{,_split_seed0}.pt`, splits `a_train` /
`a_val` / `test`.

File names follow the analyser, which predates the paper's renumbering:
`table6_baselines.csv` is the paper's **Table 4**, `table7_generated.csv` is the paper's
**Table 5**, `table7_additions.csv` the size-matched-null additions.

Table 4 is real-data baselines only and does not depend on the generator; it is recomputed here
so Table 5's ratios divide by the same denominators, and it must match
`results/dna/diversity/regen/table4_real_baselines.csv` exactly.

### Table 6 -- `table6/`

`dmfm.experiments.score_yeast_c0_evo2` via `scripts/dna/dmfm_demos/table6_evo2_dmfm.sbatch`,
Evo 2 7B base in the separate `evo2` micromamba env
(`scripts/dna/regen/evo2_environment.yml`; weights cached under `cache/xdg/huggingface`).
Identical to `scripts/dna/regen/score_evo2.sbatch MODE=c0` in every argument except
`--run_glob` and `--out_dir`: `--targets -1 1 --baseline_n 1000 --batch_size 1
--bootstrap 2000 --model_name evo2_7b_base`, RC-averaged, seed 0.

`table6/<arm>/` holds `evo2_summary.csv` (the table), `run_environment.json` and
`evo2_per_sequence_nll.csv` -- the per-sequence NLLs with the `seq`, `run_dir`, perplexity,
orientation and scored-positions columns dropped (the sequences are already tracked in
`results/dna/c0_guidance/{dmfm_n1000,glass_matched_n1000}/`), so the bootstrap CIs can be
recomputed without duplicating the sets.

Slurm **49426422** (dMFM sets, `table6/dmfm4_norm/`) and **49426423** (matched GLASS sets,
`table6/glass_norm/`), both `kempner_requeue` / `kempner_albergo_lab`, 2026-09-30. The card each
ran on is in `table6/<arm>/run_environment.json`.

The GLASS Evo 2 job for the *stock-rule* parent-disjoint sets (**49298971**, `MODE=c0` on
`results/dna/c0_guidance/n1000/`) was left queued on the starved route and has not run. It is not
needed for the comparison: `table6/glass_norm/` is a better control, because it differs from the
dMFM arm in the posterior sampler alone.
