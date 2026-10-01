# results/dna/parity -- provenance

Matched-setting dMFM-vs-GLASS comparison for the DNA C0-guidance experiment of Table 1.
Write-up and verdict: `docs/dmfm_glass_parity.md`. Producer:
`dmfm.experiments.parity_dmfm_glass_c0`, wrapper `scripts/dna/parity/dmfm_glass_parity.sbatch`.

Every configuration in both modes shares the Table 1 protocol exactly -- L=50 parent-disjoint
base DFM `checkpoints/dna/base/L50/best.pt`, C0 guide `checkpoints/dna/c0/guide/best_state.pt`,
C0 oracle `checkpoints/dna/c0/oracle/best_state.pt` (reverse-complement averaged), seed 0,
per-sample initial noise `seeded_randn((1, 50, 4), seed + i)`, frozen per-sample MC pool
`seeded_randn((8, 50, 4), seed + 1e6 + i)`, MC=8 with `mc_chunk` 8, reward
`0.5 * -0.5 ((f_cyc - y*) / 0.15)^2`, both targets y* = +-1. Only the posterior sampler differs
between rows; a row labelled `@norm` or `@gf=<x>` additionally changes the one named knob and
says so.

| directory | mode | Slurm | device | command |
|---|---|---|---|---|
| `grad_L50_rk4ref/` | `--mode grad`, **the quotable one**: 4 probe states at t in {0.5, 0.7, 0.9}, reference = GLASS with 25 **RK4** steps to **end_time 0.999** (Table 18's reference integrator) on the identical MC pool | 49350990 | A100-SXM4-40GB, `gpu_test`, `kempner_albergo_lab`, TF32 on | `MODE=grad OUT_DIR=results/dna/parity/grad_L50_rk4ref N_PROBES=4 REFERENCE_STEPS=25 sbatch scripts/dna/parity/dmfm_glass_parity.sbatch` |
| `grad_L50/` | the same with 8 probes and reference = GLASS with 100 **Euler** steps to **end_time 1.0**. **Superseded**: a fine Euler grid taken all the way to t=1 crosses the singular linear-DFM data-time velocity, so the endpoint lands on a hard corner and the reference gradient collapses; kept only so the reference choice is on the record | 49345753 | A100-SXM4-40GB, `gpu_test`, `kempner_albergo_lab`, TF32 on | `MODE=grad REFERENCE_STEPS=100 --reference_solver euler --reference_end_time 1.0` |
| `sample_L50_n64/` | `--mode sample`, 12 configurations x 64 paired unguided/guided samples per target, `nfe_traj 64`, `t_max 0.95`, window [0.5, 0.95], `guidance_frac 8`, `coeff_cap 10`, `grad_clip 10`; 10 min wall | 49349568 | A100-SXM4-40GB, `gpu_test`, `kempner_albergo_lab`, TF32 on | `MODE=sample N_SAMPLES=64 OUT_DIR=results/dna/parity/sample_L50_n64 CONFIGS="..." sbatch scripts/dna/parity/dmfm_glass_parity.sbatch` |
| `sample_L50_n32_knobs/` | the `guidance_frac` (1, 2, 4, 8, 16) and `grad_clip` (2, 4, 10) sweep, both arms, with and without the scale-free rule, 16 configurations x n=32; 5 min wall | 49351325 | A100, `gpu_test`, `kempner_albergo_lab`, TF32 on | `MODE=sample N_SAMPLES=32 OUT_DIR=results/dna/parity/sample_L50_n32_knobs CONFIGS="..." sbatch scripts/dna/parity/dmfm_glass_parity.sbatch` |
| `sample_L50_n32_knobs_hi/` | extends the scale-free `guidance_frac` grid to 16 and 32 for **GLASS** as well, so best-of-grid is symmetric at the top of the range. **Authoritative job 49355914** (`kempner_h200`); earlier ids 49353655 / 49355650 / 49355964 / 49356326 / 49361841 were the same run on partitions that could not schedule it and were all cancelled, so no duplicate burn. Absent directory = it never got capacity; the item closes on the n=800 rerun alone (see `docs/dmfm_glass_parity.md` verdict) | 49355914 | `kempner_h200`, `kempner_albergo_lab` | `MODE=sample N_SAMPLES=32 OUT_DIR=results/dna/parity/sample_L50_n32_knobs_hi CONFIGS="glass_euler4@norm@gf=16 glass_euler4@norm@gf=32 dmfm4_fm4@norm@gf=32 diag_fm4@norm@gf=16 dmfm4_fm4@norm@gf=16 glass_euler4" sbatch scripts/dna/parity/dmfm_glass_parity.sbatch` |

Each directory's `run_metadata.json` records the full argument namespace, the host/device/TF32
state, and the checkpoint path behind every configuration label.

Files:

* `grad_L50*/grad_agreement.csv` -- one row per (target, t, configuration): cosine similarity,
  relative L2 (mean / median / pooled), per-coordinate MAE, gradient norms, the L1 and token
  agreement of the posterior endpoint against the GLASS-100 endpoint drawn from the same eps, and
  endpoint shape diagnostics (simplex mass, argmax margin, soft and hard guide score).
* `grad_L50*/grad_agreement_summary.csv` -- the same averaged over targets and times.
  **Use the cosine and `rel_l2_pooled`**: the per-sample relative L2 divides by a reference
  gradient norm that collapses to ~1e-4 near t=0.9, so its mean is meaningless.
* `sample_L50_n64/sample_scores.csv` -- one row per (target, configuration, sample id) with the
  guide- and oracle-scored unguided and guided C0, both hard sequences, the mean gradient norm
  seen along that trajectory and the fraction of guided steps whose gradient exceeded `grad_clip`.
* `sample_L50_n64/sample_summary.csv` -- per (target, configuration): unguided and guided oracle
  MAE, guided oracle mean, guided guide MAE, paired improvement, fraction improved, and the
  per-sample paired difference in guided oracle error against `glass_euler4` with its SEM.

**Read the verdict in `docs/dmfm_glass_parity.md` before quoting any number here.** The parity
reading of `sample_L50_n64/` is superseded: the n=800 Table 1 dMFM rerun, under the identical
scale-free rule, measures dMFM 0.040 +- 0.007 behind GLASS at y* = +1 (0.193 vs 0.152). The n=32-64
numbers in this directory are statistically compatible with that but much less precise; their
lasting value is the *knob-by-knob ranking* (posterior artifact, step count, guidance rule), not
any claim of parity.

Guided sampling is not bit-reproducible on CUDA (TF32 matmuls, nondeterministic backward), so a
rerun agrees within the paired SEM, not exactly. The unguided halves are shared by construction
(same `x0`, no gradient), so every row's unguided column is identical across configurations --
a built-in check that the pairing is really matched.

Not produced here, but the relevant reference already on disk: the *defective* dMFM n=1000 arrays
of Slurm 49277977 in `outputs/dna/table1_dmfm_n1000/`, and the GLASS n=1000 summary
`results/dna/c0_guidance/table1_n1000_summary.json`.
