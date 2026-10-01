# `results/dna/findings_audit/` — provenance

Written 2026-09-30 by the DNA findings-audit agent. Everything here belongs to
`docs/dna_findings_audit.md`, which is the narrative; this file maps each artifact to the command,
job and inputs that produced it. Scripts are in `scripts/dna/findings_audit/`.

All DNA artifacts are L = 50 on the parent-disjoint models:
`checkpoints/dna/base/L50/best.pt` (base DFM), `checkpoints/dna/c0/guide/best_state.pt` (guidance),
`checkpoints/dna/c0/oracle/best_state.pt` (scoring), `checkpoints/dna/dmfm/L50` (one-step diagonal
student), `checkpoints/dna/dmfm_4step/L50` (4-step ESD student),
`data/dna/yeast_parent_disjoint/yeast_parent_L50.pt`.

| file / dir | item | produced by | job | notes |
| --- | --- | --- | --- | --- |
| `table16_uniform_r.json` | (b) | `python scripts/dna/findings_audit/table16_uniform_r.py` | none (login node, CPU, seconds) | Inputs: `results/dna/paper_values.json` (`table16_base_marginals`), `results/dna/figure_digitized/fig6_global_marginals_digitized.csv`, `results/dna/regen/fig6_table16/correlations.csv`. The null distribution is 20,000 multinomial draws per set size with `numpy.default_rng(0)`; the Pearson p-value uses a self-contained incomplete-beta implementation so the script runs without scipy's compiled extensions (they need a newer libstdc++ than the login image ships — `source scripts/env.sh` or `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` fixes it). |
| `tables56_attribution.json` | (a) | `python scripts/dna/findings_audit/tables56_attribution.py` | none (login node, CPU, seconds) | Inputs: `results/dna/paper_values.json`, `results/dna/june_split65k_recovered/{samples_target_*.args.json,yeast_split_dmfm_c0_evo2_summary_2026-06-10.csv}`, `results/dna/diversity/regen/{table4_real_baselines,table5_generated}.csv`, `results/dna/dmfm_demos/diversity/{dmfm4_norm,glass_norm}/table7_generated.csv`, `results/dna/dmfm_demos/table6/{dmfm4_norm,glass_norm}/evo2_summary.csv`. No new sampling. |
| `reward_temperature/` | (d) | `python scripts/dna/findings_audit/reward_temperature.py --n_probes 8 --probe_times 0.5 0.7 0.9 --mc 8 64 --n_calib 256 --device cpu` | none (login node, CPU, ~10 min) | 8 probe states from the base flow at each outer time (`paired_initial_noise(seed 0)`, `nfe_traj 64`, `t_max 0.95`), the Table 1 GLASS-4 posterior, the frozen MC pool `paired_eps_pool(seed 0)`; calibration = 256 unguided samples at `t_max 0.999`. `reward_temperature_ess.csv` holds ESS / max-weight / value per (reward, coefficient, target, t, MC). |
| `reward_temperature/grad_norm_vs_coefficient.json` | (d) | the inline script kept verbatim at the end of this file | none (login node, CPU, ~7 min) | Mean / max value-gradient norm and the fraction above `grad_clip = 10` for the Table 1 GLASS-4 posterior at `reward_scale` 0.045 / 0.5 / 1.0, 4 probe states, t in {0.5, 0.7}, both targets, MC 8, identical pool. |
| `direction_usefulness/direction_usefulness_quick_n4.csv` | (c) | `scripts/dna/findings_audit/direction_usefulness.py --n_probes 4 --probe_times 0.5 --deltas 1.484 --n_random 1 --configs glass_euler4 glass_euler32 glass_euler32_e999` | none (login node, CPU, ~8 min) | Pre-registered one-step sign test; the reading is in that script's docstring, written before the run. The `glass_rk4_25_e999` arm was attempted twice on the login node and both attempts were reaped before printing a row, so it is covered only by job 49442813, which runs the full grid on `serial_requeue`. |
| `value_smoothness/` | (c) | `scripts/dna/findings_audit/value_smoothness.py` via `value_smoothness.sbatch` | 49438552 (`serial_requeue`, CPU, COMPLETED 8 min) | Central finite differences of the finite-N value along random unit directions, 4 probe states, MC 8, true fp32, no autodiff. A first login-node attempt was abandoned (one core at load ~50) and its partial log kept as `logs/fa_value_smoothness_long_partial.log`; the rows it printed are the ones quoted in the doc and are reproduced by the Slurm run. |
| `direction_usefulness/` (full grid) | (c) | same script via `direction_usefulness.sbatch` | 49442813 (`serial_requeue`, CPU, COMPLETED 2:35) | 8 probes, t in {0.5, 0.7}, both targets, three displacements, two random directions, all four posteriors including `rk4-25` to 0.999. |
| `sample_posterior_accuracy_cpu_n8/` | (c) | `posterior_accuracy_cpu_hedge.sbatch` | 49434328 (`serial_requeue`, CPU, COMPLETED 5:06) | **This is the run that settled item (c).** End-to-end test at n = 8, y* = -1 only, without RK4-25; run because the GPU pool was capped at the account level. |
| `sample_outer_nfe{64,128,256,512}/` | (c), cross-domain | `outer_refinement.sbatch` | 49440315 | The DNA analogue of the QM9 DPS `sample_steps` sweep: refine the *outer* guided trajectory at a fixed GLASS-4 inner posterior, stock and scale-free rules, n = 32, both targets. |
| `reward_temperature_gpu/` | (d) | same script, `--device` default | `reward_temperature.sbatch`, queued behind the three jobs below | GPU repeat of the CPU run as a device-independence check. |
| `grad_posterior_accuracy/` | (c) | `scripts/dna/findings_audit/posterior_accuracy_probe.py --mode grad ...` via `posterior_accuracy_probe.sbatch` | 49431928 | Reference posterior = GLASS **RK4, 25 steps, end_time 0.999** (Table 18's integrator), 4 probe states, t ∈ {0.5, 0.7}, MC 8, identical MC pool for every configuration. Extends `results/dna/parity/grad_L50_rk4ref/` with the whole Euler ladder (4/8/16/32/100 steps to 1.0 and to 0.999) and RK4 8/25 to 0.999, which `results/dna/parity/` never measured. |
| `sample_posterior_accuracy/` | (c) | same script `--mode sample --n_samples 64` | 49431928 | Same protocol as `results/dna/parity/sample_L50_n64/` (same ids 0-63, same initial noise `seed + i`, same frozen MC pool `seed + 1e6 + i`, `guidance_frac 8`, `coeff_cap 10`, `grad_clip 10`, `nfe_traj 64`, `t_max 0.95`, window [0.5, 0.95], `reward_sigma 0.15`, `reward_scale 0.5`, batch 8, park_cnn oracle RC-averaged), so `glass_euler4` here is directly comparable to that table's row. Configurations: `glass_euler4`, `glass_euler32`, `glass_euler32@norm`, `glass_euler32_e999`, `glass_euler32_e999@norm`, `glass_rk4_25_e999`, `glass_rk4_25_e999@norm`. |
| `sample_reward_scale_{0p045,0p5,1p0}/` | (d) | same script `--mode sample --reward_scale {0.045,0.5,1.0}` | `reward_temperature.sbatch` (queued) | `reward_scale 0.045 = 2 x 0.15^2` makes the log-reward coefficient exactly 1, i.e. the paper's Eq (17)/(48) as written; 0.5 is Table 1 as run; 1.0 is the published Tables 4-6 sets as run. Two configurations each: `glass_euler4` (stock clip-down) and `glass_euler4@norm` (scale-free). n = 32, both targets. |
| `junedose/` | (a) | `dmfm.experiments.analyze_c0_diversity` on the runs in `outputs/dna/findings_audit/` | 49431930 (sampling) + CPU (analysis) | The sampling runs reproduce the **published June protocol** (every default of `dmfm.experiments.sample_c0_guidance_dmfm`, which is the port of `DNA-MFM@187fe7b:scripts/sample_yeast_split_dmfm_c0.py`) on the parent-disjoint model and guide: `mc 16`, `nfe_value 1`, `nfe_sample 96` to `t_max 1.0`, window [0.01, 0.95], `reward_scale 1.0`, `guidance_frac 8`, `coeff_cap 10`, `grad_clip 10`, batch 10, seed 0 (y*=+1) / 1 (y*=-1), n = 1000. `ARM=ablate` adds `--guide_t_start 0.50` and `--reward_scale 0.5` at y*=-1, one knob at a time. The analyser is run with the published defaults (`--baseline_n 1000 --seed 0 --max_pairwise_n 3000 --k_nearest_train 5 --pair_chunk 128 --size_matched_draws 1000`), so its real-data baselines must equal `results/dna/diversity/regen/table4_real_baselines.csv` to the last digit. |

Raw sampling trees live in the gitignored `outputs/dna/findings_audit/`; only the summaries are
copied here.

Queue note, so no id below looks like duplicated compute. Nothing that was cancelled had started.

* 49431452 / 49431495 / 49431496 -- the first submission of the three GPU jobs to `kempner_requeue`,
  where 1,184 of 1,231 pending jobs outranked them. Cancelled, resubmitted to `kempner_h100`
  (37-43 ahead) as 49431928 / 49431930 / 49431932.
* 49431928 -- cancelled while trying to move the item-(c) job to `gpu_requeue`, which turned out to
  reject `kempner_albergo_lab` outright and to give `albergo_lab` a start estimate of 2026-10-06.
  Resubmitted to `kempner_h100` as **49434948**, the authoritative item-(c) job.
* 49431932 (`ARM=ablate`, the one-knob-back controls for item (a)) -- cancelled to free a slot for
  **49440315**, the outer-trajectory refinement arm the round's QM9 result made necessary. The
  knob decomposition is secondary; item (a)'s verdict does not depend on it.
* All GPU jobs carry `--constraint="a100|a100-mig|h100|h200"`: the requeue partitions include 24 RTX
  PRO 6000 (sm_120) nodes, and the pinned torch 2.5.1+cu124 has no sm_120 kernels.
* Every `--mode grad` / `--mode sample` run of `posterior_accuracy_probe.py` forces TF32 **off**
  (`torch.set_float32_matmul_precision("highest")`), unlike the stored `results/dna/parity/` runs,
  which were produced with the parity module's TF32 default. `run_metadata.json` records this as
  `run_env.tf32_forced_off_by_findings_audit`. Set `FA_TF32=1` to reproduce the parity module's
  behaviour instead.

## Verbatim: the inline script behind `reward_temperature/grad_norm_vs_coefficient.json`

```python
import sys, json, torch
sys.path.insert(0, "src")
from dmfm import api
dev = torch.device("cpu")
base, cfg = api.load_base(50, dev); guide = api.load_c0("guide", dev)
L, K = int(cfg.seq_len), int(cfg.alphabet_size)
ids = list(range(4)); x0 = api.paired_initial_noise(ids, L, seed=0, device=dev)
pool = api.paired_eps_pool(ids, L, mc=8, seed=0, device=dev)
post = api.glass_posterior_fn(base, n_steps=4, end_time=1.0)
rows = []
for t in (0.5, 0.7):
    nfe = max(1, int(round(t / (0.95 / 64)))); x_t = api.sample_unguided(base, cfg, x0, nfe=nfe, t_max=t)
    for target in (1.0, -1.0):
        for rs in (0.045, 0.5, 1.0):
            lr = api.c0_log_reward_fn(guide, target, reward_sigma=0.15, reward_scale=rs)
            v, g = api.value_and_grad(post, lr, x_t, t, pool, mc_chunk=8)
            gn = g.flatten(1).norm(dim=1)
            rows.append(dict(t=t, target=target, reward_scale=rs, coefficient=0.5 * rs / 0.15 ** 2,
                             grad_norm_mean=float(gn.mean()), grad_norm_max=float(gn.max()),
                             frac_above_clip10=float((gn > 10).float().mean()), value_mean=float(v.mean())))
json.dump(rows, open("results/dna/findings_audit/reward_temperature/grad_norm_vs_coefficient.json", "w"), indent=1)
```
