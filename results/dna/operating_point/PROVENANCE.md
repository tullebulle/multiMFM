# `results/dna/operating_point/` -- provenance

The (posterior steps, MC) operating-point grids at matched value-gradient NFE, one per DNA paper
item. Findings and the verdict table: `docs/dna_operating_point.md`.

The operating point itself was **not** chosen from anything in this directory. It was chosen on
the Table 1 C0 task in `results/dna/deficit_rescue/nfe_L50_n200/` (see
`docs/dmfm_deficit_rescue.md` section 3) as dMFM = 1 composed step x MC 32 and GLASS = 2 Euler
steps x MC 16, both NFE 32, and applied unchanged here. The grids below are reported in full so
that where an item's own optimum differs it can be seen, not so that it can be adopted.

| directory | paper item | producer | command | Slurm | device |
|---|---|---|---|---|---|
| `fig7_grid_nfe32_n64/` | **Fig 7 tilt ladder, GPU, n = 64 -- the primary Fig 7 run** | `dmfm.experiments.operating_point_c0` | `TARGETS=0.30 N_SAMPLES=64 OUT=results/dna/operating_point/fig7_grid_nfe32_n64 sbatch --partition=kempner_h200 scripts/dna/operating_point/c0_grid.sbatch` | **49465152** (H200 `holygpu8a06301`, TF32, 2:49) | GPU |
| `table13_grid/motif_nfe8/` | **Table 13 at NFE 8, GPU, the published n = 100 -- the primary Table 13 run** | `dmfm.experiments.operating_point_reward_grid` | `sbatch --partition=kempner_h200 --array=0-1%1 scripts/dna/operating_point/table13_grid.sbatch` | **49465154_0** (H200, TF32, 3:19) | GPU |
| `table1_anchor/` | where the operating point was chosen | re-analysis of `results/dna/deficit_rescue/nfe_L50_n200/` | see that directory's `README.md` | none (49431420 produced the source data) | -- |
| `fig7_grid_cpu_nfe32_n32/` | Fig 7 tilt ladder (target C0 0.30, `guidance_frac` 8), n = 32 | `dmfm.experiments.operating_point_c0` | `WHAT=c0 sbatch --partition=kozinsky --account=kozinsky_lab scripts/dna/operating_point/cpu_hedge.sbatch` | **49466900** | CPU (fp32) |
| `table18_grid_cpu/gc/` | Table 18 gradient MAE, 8 probes x 2 repeats | `dmfm.experiments.operating_point_gradient_grid` | `WHAT=t18 REWARD=gc sbatch --partition=kozinsky ...` | **49466901** | CPU (fp32) |
| `table13_grid_cpu/motif_nfe8/` | Table 13 exact motif reward at NFE 8, n = 32 | `dmfm.experiments.operating_point_reward_grid` | `WHAT=t13 NFE=8 sbatch --partition=kozinsky ...` | **49467013** | CPU (fp32) |
| `table13_grid_cpu/motif_nfe32/` | the same at NFE 32 | same | `WHAT=t13 NFE=32 sbatch --partition=kozinsky ...` | **49467117** | CPU (fp32) |
| `table13_grid_cpu/motif_nfe8_diagstudent/` | Table 13 at NFE 8 with the diagonal student forced at every step count | same | `WHAT=t13 NFE=8 STUDENT=dmfm OUT=... sbatch --partition=kozinsky ...` | **49469755** | CPU (fp32) |
| `fig4_grid_cpu/` | Fig 4 (right), 8 outputs, convention-1 student | `scripts/dna/operating_point/fig4_grid.py` via `dmfm.benchmarks.run_steering.run_job` | `WHAT=fig4 NFE="16 8" sbatch --partition=kozinsky ...` | **49467015** (after 49466420 was preempted) | CPU (fp32) |
| `fig4_grid_cpu_diag/` | the same with the diagonal student at every step count | same | `WHAT=fig4 NFE="16 8" STUDENT=diagonal OUT=... sbatch ...` | **49466933** | CPU (fp32) |
| `fig4_grid_cpu_n32/` | the same at 32 outputs | same | `WHAT=fig4 NFE="16 8" N_OUTPUTS=32 OUT=... sbatch --partition=kozinsky ...` | **49467657** | CPU (fp32) |
| `table18_grid_cpu/motif/` | Table 18 gradient MAE, `motif` reward, 8 probes x 2 repeats | `dmfm.experiments.operating_point_gradient_grid` | `WHAT=t18 REWARD=motif sbatch --job-name=dna_op_t18b --partition=kozinsky ...` | **49471660** (complete) | CPU (fp32) |
| `fig8_grid_nfe32_ids4to7/`, `fig8_grid_nfe32_ids8to11/` | Fig 8 extended to ids 4-11 for dMFM 1x32 / 4x8 and (ids 4-7) GLASS 2x16 | `dmfm.regen.guidance_trace` | the same command with `--sample_ids 4 5 6 7` / `8 9 10 11` and `--out_dir` under those roots | none (login node, foreground) | CPU (fp32, deterministic) |
| `fig8_grid_nfe32/` | Fig 8 single-trajectory trace, all 8 splits | `dmfm.regen.guidance_trace` | `bash scripts/dna/operating_point/fig8_grid_cpu.sh` (login node, `logs/op_fig8_grid.log`) | none | CPU (fp32, deterministic) |

### GPU jobs: two ran late, three still queued

The account's shared 16-GPU QOS cap was consumed by the same user's other projects for most of the
session. **49465152** (Fig 7, n = 64) and **49465154_0** (Table 13, n = 100) started about 90 minutes
after submission and are the primary runs for those items, listed above. Still queued, writing to
directories the CPU hedges do not touch:

| job | directory it would write | script |
|---|---|---|
| **49464001** (`kempner_h100`, array 0-2%1) | `table18_grid/{gc,motif,conjunction}/` (32 probes x 8 repeats) | `sbatch scripts/dna/operating_point/table18_grid.sbatch` |
| **49465154_1** (`kempner_h200`) | `table13_grid/motif_nfe32/` (n = 100) | the same array, task 1 |
| **49464821** (`kempner_h100`) | `fig4_grid/` (32 outputs x 3 repeats) | `sbatch scripts/dna/operating_point/fig4_grid.sbatch` |

**Which run is primary per item:** Fig 7 -> `fig7_grid_nfe32_n64/` (GPU); Table 13 at NFE 8 ->
`table13_grid/motif_nfe8/` (GPU); everything else -> the `*_cpu*` directories, until the queued GPU
jobs land. The CPU runs are kept as corroboration and their orderings agree with the GPU runs'.

Earlier submissions that were cancelled or preempted and produced nothing: 49463361 and 49464820
(resubmitted to `kempner_h200` as 49465152 / 49465154); 49465255, 49465267, 49465659, 49465663,
49465665 (`serial_requeue`, 8,260 jobs ahead, cancelled before starting); 49466290, 49466301,
49466303, 49466420 (`kozinsky_requeue`, `CANCELLED ... DUE TO PREEMPTION` within six minutes,
resubmitted to the non-preemptible `kozinsky`). `scripts/dna/operating_point/fig7_at_operating_point.sbatch`
is the full-n (1,000-pair) rerun of Fig 7's `gf = 8` point at each arm's operating point, written here
to be run only after a grid had shown a real move. I did not submit it; it was submitted from
elsewhere in the same session as **49476650** (`kempner_h200`, array 0-1) and was still pending. It
writes `fig7_operating_point_n1000/{dmfm_s1mc32,glass_s2mc16}/`.

Each run directory carries its own `run_metadata.json` (or per-shard JSON, for Fig 4) with the
full argument vector, the checkpoints, the host, the GPU name and the TF32 flag. Summaries and
paired statistics are written by `scripts/dna/operating_point/collect.py`:

    python scripts/dna/operating_point/collect.py c0       results/dna/operating_point/fig7_grid_nfe32_n64
    python scripts/dna/operating_point/collect.py gradient results/dna/operating_point/table18_grid/gc
    python scripts/dna/operating_point/collect.py reward   results/dna/operating_point/table13_grid/motif_nfe8
    python scripts/dna/operating_point/collect.py fig4     results/dna/operating_point/fig4_grid
    python scripts/dna/operating_point/collect.py fig8     results/dna/operating_point/fig8_grid_nfe32

## Cross-device warning

CPU is true fp32 and CUDA runs here are TF32, and the CPU and CUDA RNG streams differ, so a CPU
hedge's *absolute* numbers are not comparable with a GPU run's. Every comparison drawn from a run
is between configurations **inside** that run, which is what a matched-NFE split comparison needs;
within a run all splits share the initial noise (and, for the gradient grids, the probe states,
the reference gradient and the candidate MC pools), so the split is the only thing that differs.

Fig 8's CPU route is additionally deterministic: its unguided endpoints reproduce the recorded
Fig 8 dMFM run (`results/dna/dmfm_demos/fig8/dmfm/trace.csv`) to every printed digit, which is the
control that the two runs share one trajectory and differ only in the guidance term.

## Controls checked per run

* **C0 grids**: the unguided oracle/guide score must be identical across all splits at every id
  (the harness shares the initial noise); `summary.csv`'s `unguided_oracle_mae` column is constant
  within a target when this holds.
* **Gradient grids**: all splits share the probe states, the GLASS-128 reference gradient and the
  nested candidate MC pools, so `reference_l2` is constant per probe across splits.
* **Table 13 grids**: the unguided arm is drawn before the MC pool from the same seeded generator,
  so `score_unguided` is identical across splits at every `sample_id`.
