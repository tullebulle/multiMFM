# results/dna/deficit_rescue/ -- provenance

Owner: the deficit-rescue agent of the 2026-09-30 round. Question: is the measured
dMFM-vs-GLASS deficit on the DNA C0-guidance experiment (Table 1) our error?
Narrative and conclusions: `docs/dmfm_deficit_rescue.md`.

Every file here comes from `dmfm.experiments.rescue_dmfm_glass_c0`, submitted through
`scripts/dna/rescue/deficit_rescue.sbatch`, and is summarised by
`scripts/dna/rescue/collect_deficit_rescue.py`. All runs are L=50, the parent-disjoint base DFM,
the parent-disjoint C0 guide and the parent-disjoint C0 oracle (reverse-complement averaged,
scored on CPU) -- the Table 1 setting. Within a run every configuration sees the same initial
noise (`seed + i`), the same frozen MC pool (`seed + 1e6 + i` at that configuration's `mc`), the
same guide and the same trajectory settings (`nfe_traj 64`, `t_max 0.95`, window [0.5, 0.95],
`reward_sigma 0.15`, `reward_scale 0.5`, `coeff_cap 10`, `grad_clip 10`, batches of 8, seed 0);
only the posterior sampler and the knobs named in the configuration label differ, and every knob
is offered to both arms.

| directory | Slurm | what |
|---|---|---|
| `nfe_L50_n200/target_{1p0,m1p0}/` | 49431420 (array, `kempner_h200`, H200) | posterior step count and MC budget under the one scale rule both arms share (`@norm`, `guidance_frac 8`): GLASS at 2/4/8 Euler steps, the 4-step ESD student at 1/2/4/8 composed steps, the diagonal student at 1/4, plus three rows that spend a cheaper posterior's saved NFE on more MC draws at the same total NFE as `glass_euler4@norm` |
| `matched_L50_n600/target_{1p0,m1p0}/` | 49438707 (array, `kempner_h200`, H200; y*=+1 on `holygpu8a03504`, y*=-1 on `holygpu8a03303`) | the headline pair at larger n: four configurations all at value-gradient NFE 32 -- GLASS 4 steps x MC 8 (the n=1000 GLASS arm), GLASS 2 steps x MC 16, the 4-step ESD student at 1 step x MC 32, and the diagonal student at 1 step x MC 32 |
| `rule_L50_n200/target_{1p0,m1p0}/` | 49431419 (array, `kempner_h100`, H100; y*=+1 on `holygpu8a15102`, y*=-1 on `holygpu8a17104`) | guidance rule x guidance strength, symmetric: `guidance_frac` in {8, 16, 32} under `@norm` for BOTH arms, `guidance_frac` in {8, 16} under the magnitude-preserving `@relnorm` rule for BOTH arms, the published clip-down GLASS row for reference, and the diagonal student under `@norm` |

Per-directory files: `sample_scores.csv` (one row per configuration x sample id: guide and oracle
scores for the guided and the paired unguided sample, both sequences, and the per-configuration
gradient-norm diagnostics), `run_metadata.json` (every argument, the GPU, the Slurm ids, the
`@relnorm` constants and the resolved checkpoint of every configuration). Per-sweep files, written
by the collector at the sweep root: `summary.csv` (per configuration and target: guided oracle MAE
with a 2,000-resample percentile bootstrap CI, guided guide MAE, fraction improved, gradient norms
before and after the scale rule) and `paired.csv` (per-sample-id paired difference in guided oracle
error against every GLASS configuration in the same sweep, with SEM and bootstrap CI).

Reference numbers these are compared against, both from
`results/dna/c0_guidance/table1_dmfm_n1000_summary.json` (n=1000, `@norm`, `guidance_frac 8`):
dMFM `dmfm4_fm4` 0.193 / 0.172 and matched GLASS-4 0.153 / 0.124 guided oracle MAE at
y* = +1 / -1.

Device record, because a 0.02 effect is the size hardware differences have faked in this repo:
`nfe_L50_n200` and `matched_L50_n600` are NVIDIA H200, `rule_L50_n200` is NVIDIA H100, all with
TF32 on. Absolute numbers must not be compared across those two sweeps; the n=1000 reference numbers are H100 (jobs 49352242
/ 49352244), also TF32 on. Within each target every configuration ran in one process on one card, so
all paired comparisons are same-card; the cross-generation check is that the n=200 sweep reproduces
the n=1000 deficit at the published operating point (+0.050 / +0.041 against +0.041 / +0.049).

Guided sampling is not bit-reproducible (nondeterministic CUDA backward), and the same
configuration at n=32 has been seen to move by ~0.04 between an A100 and an H200
(`docs/dmfm_glass_parity.md` `dmfm4_fm4@norm@gf=16` at y*=+1: 0.144 on job 49351325 vs 0.185 on
job 49355914). Treat any difference below ~2 SEM at these sample sizes as unresolved.
