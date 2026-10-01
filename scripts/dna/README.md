# DNA (dMFM) run wrappers

Each file reproduces one full-size paper run with the arguments of the original
Slurm job (recorded in `docs/provenance/dna/original_runs.md`). They take no
absolute paths, activate the `multimfm` env through `scripts/env.sh`, run
`python -m dmfm.experiments.<name>`, and write logs to `logs/` (gitignored).
Submit from the repo root. Every parameter is an environment variable with the
original value as its default.

| File | Paper item | Key variables |
|---|---|---|
| `prepare_data.sbatch` | Table 10 (parent-disjoint split) | `INPUT`, `OUT_DIR` |
| `train_base_dfm.sbatch` | Table 9, base generators | `LENGTH` (required), `STEPS`, `BATCH_SIZE` (512/256/128/64 by length) |
| `train_dmfm.sbatch` | Table 9, dMFM (diagonal + ESD) | `LENGTH` (required), `MAX_STEPS`, `LR` |
| `train_dmfm_l400_finetune.sbatch` | the shipped L=400 dMFM | `CKPT` (required: L400 diagonal step-45000) |
| `train_dmfm_4step.sbatch` | 4-step students (Table 18) | `LENGTH` (required), `INIT_CKPT` |
| `train_c0_regressors.sbatch` | Table 11 | `RUN_TAG`, `EPOCHS` |
| `table1_c0_guidance.sbatch` | Table 1 **as submitted** (GLASS on the base DFM; provenance only), Fig 7 histograms | `TARGET_C0`, `N_SAMPLES` (1000; use 32 for the printed pilot), `GUIDANCE_FRAC`, `SEED` |
| `table1_c0_guidance_dmfm.sbatch` | Table 1 (dMFM guidance; what the paper reports now), 10-task array | `N_SAMPLES`, `N_SHARDS`, `POSTERIOR` (`glass` = control), `OUT_ROOT` |
| `collect_table1_dmfm.sh` | merge shards + oracle + `results/dna/c0_guidance/table1_dmfm_n1000_summary.json` (CPU) | job ids (recorded) |
| `tables4_5_c0_diversity.sh` | Tables 4-5 (CPU) | positional: two run dirs, out dir |
| `table13_reward_attainment.sbatch` | Table 13 (diagonal dMFM students) | `LENGTH`, `REWARD` (both required), `DMFM_CKPT` |
| `fig5_tables14_15_gradient_mc.sbatch` | Fig 5, Tables 14-15 | `REWARD`, `OUTPUT_DIR`, `GLASS_END_TIME` (1.0 = what the runs used) |
| `fig5_tables14_15_rescore.sh` | the Aug-2 scale-free rescore behind Tables 14-15 (CPU) | positional: run dirs |
| `table17_posterior_diversity.sbatch` | Table 17 | `DMFM_STEPS`, `GLASS_STEPS` |
| `table18_dmfm4_gradient_mc.sbatch` | Table 18 (12-task array) | `--array`, or `REWARD` + `LENGTH` |
| `fig4_panel{A,B}.sbatch`, `fig4_*` | Fig 4 regeneration on the parent-disjoint models (separate work; not the published panels) | see their headers |
| `score_evo2.sbatch` | Table 6 / Evo2 columns; needs the separate env | `MODE`, `EVO2_ENV` |
| `evo2_environment.yml` | the Evo2 environment (Python 3.12, torch 2.7.1, evo2 0.5.5) | - |
| `notebook_rerun.sbatch` | executes notebooks 02-05 with `RERUN = True` on one GPU (dna-rerun); job-unique scratch via `DNA_RERUN_OUT` | `NOTEBOOKS`, `OUT` |
| `summarize_rerun.py` | summary of the executed RERUN notebooks (errors, mismatches) | positional dir |
| `smoke_gpu.sbatch` | GPU smoke test + numerical checks against stored runs (not a paper run) | `PARTS` (sample train gradmc reward posterior nfe), `OUT`, `REFERENCE` |
| `_header.sh` | shared preamble (repo root, env, logging) | `PY` |
| `_dryrun.py` | parse-only stand-in for `python`: `PY="python scripts/dna/_dryrun.py" bash scripts/dna/<wrapper>` parses every call with the module's own parser and runs nothing (used by `tests/dna/test_script_args.py`) | `DMFM_DRYRUN_OUT` |

Slurm runs a copy of each file from its spool directory, so the wrappers locate the repo
from `SLURM_SUBMIT_DIR` (submit from the repo root) or `MULTIMFM_ROOT`. They can also be run
directly with `bash scripts/dna/<file>` on a GPU node. `tests/dna/test_dmfm_port.py`
dry-runs the training wrappers and checks that they parse to exactly the `args.json` stored
with each shipped model.

Partitions default to `kempner_requeue` with `--account=kempner_albergo_lab`
(the originals mostly used `kempner_h100`, the base models `seas_gpu`);
override with `sbatch --partition=...`.
