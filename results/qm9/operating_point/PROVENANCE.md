# Provenance: results/qm9/operating_point

Everything here answers one question, asked in `docs/qm9_operating_point.md`: is Table 2's QM9
guidance run at the right split of the value-gradient budget between posterior steps
(`--value-glass-steps`) and Monte-Carlo draws (`--value-samples`)?  The two knobs multiply to give
the network evaluations per guided step, so at fixed NFE they trade off.

All runs are on one card generation, **NVIDIA A100-SXM4-40GB** (`--partition=kempner`, account
`kempner_albergo_lab`), which is also the card the reference 4 x 32 runs in
`outputs/qm9/table2_3seed/` used.  Guided sampling is not bit-reproducible
(nondeterministic CUDA backward), so reruns agree within CIs, not exactly.

## Producers

| file | producer |
|---|---|
| `scripts/qm9/operating_point/run_split.sh` | one run at an explicit (steps, MC) split; every other argument held at the published Table 2 value, copied from `scripts/qm9/run_multimfm.sh` and `scripts/run_steer_search.sh`, which are **not** modified |
| `scripts/qm9/operating_point/grid_alpha.sbatch` | the selection grid (seed 8) and any other single-property grid; `PROP`, `N`, `SEED`, `SPLITS` are environment overrides |
| `scripts/qm9/operating_point/grid_eval_seeds.sbatch` | the same splits on the evaluation seeds 1-3 at n = 100, for the baseline ladder |
| `scripts/qm9/operating_point/table2_split.sbatch` | Table 2 at one chosen split, 3 seeds x 6 properties x n = 1000 (only needed if the chosen split differs from 4 x 32) |
| `scripts/qm9/operating_point/collect.py` | `<grid>/summary.csv`, `<grid>/paired.csv`, `<grid>/unguided_control.csv` |
| `scripts/qm9/operating_point/baselines_matched.py` | `baselines_alpha_matched.csv` |

## Runs

| results dir | run dir | Slurm | what |
|---|---|---|---|
| `grid_alpha_n100_s8/` | `outputs/qm9/operating_point/grid_alpha_n100_s8/` | 49464797 (multiMFM row), 49464822 (SS row) | the pre-registered selection grid: alpha, seed 8 / target seed 2026, n = 100, splits 1x128 / 2x64 / 4x32 / 8x16 plus the two non-diagonal diagnostics |
| `grid_homo_n100_s8/` | `outputs/qm9/operating_point/grid_homo_n100_s8/` | 49465288 | the same grid on a second property, as a confirmation that the shape of the split curve is not specific to alpha.  Not part of the selection. |
| `evalseeds_alpha_n100/` | `outputs/qm9/operating_point/evalseeds_alpha_n100/` | 49464799 | the four splits on the held-out evaluation seeds 1-3, n = 100, alpha -- the seed/target/n set of `results/qm9/baselines/baseline_sweep_v2.csv`, so the split axis is paired with every baseline |
| `baselines_alpha_matched.csv` | `outputs/qm9/baselines_v2/seed{1,2,3}/` (existing, not produced here) + the above | 49429598 and the earlier v2 sweep | every steering method on alpha on one NFE axis, each with its own budget knob already swept: best-of-N over N in 4..512, DPS over mu, guide_every and sample_steps, Feynman-Kac over K and lambda, steer-search over MC at 4 steps, and the (steps x MC) splits |

`outputs/` is gitignored; the CSVs here are the small summaries.

## Reference (not produced here, used for the paired comparison)

`outputs/qm9/table2_3seed/{multimfm,ss}/<prop>_s<seed>/per_sample.csv` -- the published 4 x 32
operating point at 3 seeds x 6 properties x n = 1000, collected in
`results/qm9/rerun_3seed/rerun_3seed_summary.csv`.  `collect.py` reproduces that file's
`oracle_mae_mean` to 0.0 for the multiMFM row and to 2.3e-5 for the SS row (the stored SS numbers
are rounded), which is the check that these tables are on Table 2's scale -- including the
Hartree-to-meV factor 27211.386245988 that gap / homo / lumo need.

## Known nuisances, recorded

* Four tasks died at the first `.to(device)` with "CUDA error: CUDA-capable device(s) is/are busy
  or unavailable" -- jobs 49464797_1, 49464799_1, 49465288_0, 49474230_1 -- and **all four were on
  one node, `holygpu8a19102`**, each time while another task of mine already held a card there.
  It is node-specific and not co-location in general: two of my tasks co-located on
  `holygpu8a19202` ran fine, and the environment echo the scripts now print shows Slurm setting
  `CUDA_VISIBLE_DEVICES=0` with the device cgroup constraining and `nvidia-smi -L` reporting a
  different card UUID in each task.  Remedy used: resubmit with `--exclude=holygpu8a19102`.  Every
  script skips a configuration whose `summary.json` exists, so a resubmission resumes rather than
  repeating work and no result here comes from a partial run.
* `unguided_control.csv` is the pairing control: the unguided arm must not depend on the split,
  because `glass_guidance.guided_sample` captures and restores the torch RNG around the value
  gradient.  Measured maximum per-id spread across all six alpha configurations is 2.4e-3 on a
  scale of 8.30 (3e-5 relative), i.e. float nondeterminism, not a different trajectory.
