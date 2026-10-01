# Figure 4 -- provenance of the rerun

**The models and the results are gone; the producers are not.** Both panels were made in
May 2026 from the `split65k` models
(`yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35`, dMFM
`yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35`) and the
Keras-derived C0 guide `dinko/C0free_torch.pt`; all three were purged, and both run
directories (`workdir/yeast_split_value_mc_2026-05-13_12-48-10`,
`workdir/yeast_split_dmfm_steering_benchmark_2026-05-13_12-51-45`) survive empty.

Both **producer scripts** were recovered on 2026-09-29 from `tullebulle/DNA-MFM@187fe7b`
and are kept verbatim, with their Slurm launchers and sha256, in
`results/dna/original_scripts/` (this corrects the earlier note in
`docs/REPRODUCIBILITY_PLAN.md` that the left panel had no producer):

* `benchmark_yeast_split_value_mc.py` (left panel) settles its protocol -- see
  "Panel A settings" below. Its reward is **not** the cyclizability guide but a synthetic
  `SmoothDNAReward`, and its reference is teacher SDE sampling, not GLASS.
* `benchmark_yeast_split_dmfm_steering.py` (right panel) ran only the dMFM half. Its four
  baseline curves are a hard-coded CSV inside it (lines 28-48; byte-identical to
  `results/dna/benchmarks/fig4_right_hardcoded_baselines.csv`). **No code that produced
  those numbers has ever existed in the repository**, and their `mae_sem` is 0.000000 in
  every row although `n_repeats` is 3.

Authors' decision 3 (2026-09-29): rerun both panels on the current parent-disjoint models,
with real implementations of the baselines. The numbers here are that rerun: panel A is a
faithful port of the original producer, panel B keeps the original's dMFM configuration and
replaces the hard-coded baselines with implementations of best-of-N, Feynman-Kac steering,
beam search and MCTS following Didi et al. (2026). **They are not comparable to the
submitted figure**: different base model, different dMFM, different C0 guide, and baselines
that are now actually measured.

## What produced these files

| file | producer |
|---|---|
| `panelA_value_error.csv`, `panelA_per_condition.csv.gz` | `python -m dmfm.benchmarks.value_error` |
| `panelA_value_error_c0_reward.csv` | the same with `--reward c0 --reference glass` (robustness variant) |
| `panelB_curves.csv`, `panelB_runs.csv` | `python -m dmfm.benchmarks.run_steering --plan sweep` |
| `panelB_pilot.csv`, `pilot_choice.json` | the same with `--plan pilot`: how each baseline's hyperparameters were chosen |
| `calibration.json` | `--mode calibrate`: unguided C0 distribution vs the sampler stochasticity level `eta` |
| `summary.json` | `python -m dmfm.benchmarks.collect` |
| `paper_sentences.md` | hand-written: the canonical Figure 4 caption and Sec. 3.3 sentences, with the placeholders each collected file fills. Source of truth for the wording the tex agent applies. |
| all of the above, on a GPU | `sbatch --array=0-2 scripts/dna/fig4_all.sbatch` (work-stealing workers) |
| figure | `notebooks/06_dna_benchmarks.ipynb` (`RERUN=False` rebuilds it from these files) |

## Which device produced which numbers

Two shard trees, deliberately separate:

| tree | hardware | arithmetic / RNG | submitted by |
|---|---|---|---|
| `outputs/dna/fig4/` | GPU: `kempner_requeue` (H100, account `kempner_albergo_lab`) **and** `gpu_requeue` (mixed A100/H100/H200/RTX Pro, account `albergo_lab` — `kempner_albergo_lab` is not a valid association there) | TF32 matmuls, CUDA RNG | `scripts/dna/fig4_all.sbatch` |
| `outputs/dna/fig4_cpu/` | CPU, `serial_requeue`, account `albergo_lab` | true fp32, CPU RNG | `scripts/dna/fig4_cpu.sbatch` |

The two GPU routes share one tree on purpose: they are the same arithmetic (CUDA, TF32, the
same seeded Philox stream), so skip-if-exists and the locks let whichever queue frees up
first do the work. Each shard records `device_name`, so the exact GPU model behind any
number is recoverable; `gpu_requeue` is a mixed pool while `kempner_requeue` is H100 only.

**A plotted panel is produced entirely on one device type, never mixed.** CUDA and CPU do
not draw the same samples and do not round identically, and there is no A/B run showing
that difference sits below the reported error bars, so mixing would introduce a confound
that could not be ruled out. `dmfm.benchmarks.collect` warns if a shard directory contains
more than one device type; every shard records `device_type`, `tf32_matmul`,
`slurm_account` and `slurm_partition` in its own metadata.

The **pilot** is the one part that may run on either: it only *ranks* baseline
hyperparameter families and no pilot shard is ever plotted. `plan.pilot_scores` still takes
it from a single tree (the first complete one), so the ranking itself is never half CPU and
half GPU, and `plan.frozen_from_pilot` pins the winners to
`outputs/dna/fig4/frozen_hyperparameters.json` the first time a complete pilot is seen, so a
running sweep cannot switch configurations if a second pilot tree finishes later.

**So expect a GPU sweep whose baseline hyperparameters were ranked by a CPU pilot** — the
CPU queue was the one that freed up first. That is deliberate, not an oversight. A pilot
chooses *which* configuration of each baseline to plot; it contributes no plotted number. The
chosen configuration is then measured on the same device as every other point in the panel,
so the comparison the figure makes is unaffected. `pilot_choice.json` records the tree the
ranking came from and the full grid, so the choice can be re-derived on either device.

Models (all `checkpoints/dna/`, group `dna` of `checkpoints/MANIFEST.json`):
base `base/L50/best.pt`, dMFM `dmfm/L50/epoch=95-step=49000.ckpt`,
guide `c0/guide/best_state.pt`, oracle `c0/oracle/best_state.pt`.

## Slurm jobs
Submissions (from `outputs/dna/fig4_jobs.txt`):

    2026-09-29 14:38:51 pilot=49273015 panelA=49273016 panelA_data=49273017 sweep=49273020
    2026-09-29 14:50:29 note: 49274689 (fig4_all --array=0-2) submitted by another session, not dna-fig4
    2026-09-29 15:02:00 fig4_all=49274689 (--array=0-2, work-stealing workers; same dna-fig4 work, submitted alongside the arrays above -- shards are name- and lock-compatible, so the two sets skip each other's work)
    2026-09-29 16:53:23 fig4_cpu=49296674 (--array=0-7, serial_requeue, account=albergo_lab, STAGES=pilot,panelA, out-root outputs/dna/fig4_cpu, --device cpu)
    2026-09-29 17:04:29 fig4_gpu_requeue=49298117 (--array=0-3, gpu_requeue, account=albergo_lab -- kempner_albergo_lab is not a valid association on gpu_requeue; same out-root outputs/dna/fig4 as 49274689, so shards skip/lock across both routes)
    2026-09-29 18:10:09 cancelled fig4_cpu=49296674 by dna-fig4: one-route-per-item rule; 49298117 (gpu_requeue, all 162 units) is the single route, so every panel is GPU-internal and no panel is half CPU

Job IDs recorded inside the collected shards:

- **panelA**: 8 shards, job IDs 49298117, 49397748, 49397749, 49397750
- **panelB**: 96 shards, job IDs 49298117, 49397748, 49397749, 49397750
- **pilot**: 49 shards, job IDs 49298117, 49397748, 49397749

## Panel A settings

All values are the defaults of the recovered producer `results/dna/original_scripts/benchmark_yeast_split_value_mc.py`.

- reward: synthetic SmoothDNAReward (seed 1234, gain 4.0) -- the original producer's reward, NOT the cyclizability guide
- reference V_t: teacher SDE, 512 samples x 1024 Euler-Maruyama steps. It is itself a Monte Carlo estimate; the `sde_mc` series measures its own finite-N error and the published panel omits that series.
- conditioning states: 32 held-out `test` sequences noised to t ~ U(0.05, 0.35). The original used 8; that number is the shared default of the script and its launcher, not a recorded fact, and the extra conditions here only add error bars (conditions 0-7 are the original set).
- dMFM at NFE = 1; FMAP = teacher ODE from t to 1 in 256 steps; DPS = the student's own denoiser at the conditioning state. DPS and FMAP are computed once per condition, so they are exactly flat in N.
- N = 1..512; the script's default stops at 256, but the launcher takes N_LIST from the environment and the published panel has exactly these points. Each budget is an independently seeded draw, not a nested prefix.
- reference's own MC spread (half vs half of its samples): 0.3238 nats; mean V_ref 6.730 (sd 0.484 across conditions)

## Panel B settings

- 96 base-flow steps, target C0 = 0.3, metric |f_cyc(hard sequence) - y*| under the *guide*; the independent oracle (reverse-complement averaged) is in the `oracle_*` columns
- NFE is measured, not predicted, and counts one forward pass per sampler step; no parallel execution is assumed. `nfe_backward_per_output` is the extra backward passes the dMFM needs and the baselines do not
- at 187 NFE (dMFM MC=1): dMFM 0.133 vs 
- at 278 NFE (dMFM MC=2): dMFM 0.121 vs best_of_n 0.317, fk 0.207, mcts 0.294
- at 460 NFE (dMFM MC=4): dMFM 0.095 vs best_of_n 0.215, fk 0.115, mcts 0.192
- at 824 NFE (dMFM MC=8): dMFM 0.074 vs beam 0.089, best_of_n 0.139, fk 0.058, mcts 0.104
- at 1552 NFE (dMFM MC=16): dMFM 0.061 vs beam 0.050, best_of_n 0.072, fk 0.027, mcts 0.046
