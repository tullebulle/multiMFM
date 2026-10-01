# `scripts/dna/scalefree` - Fig 5 / Tables 14-15 with dMFM finite-N gradients

The published Fig 5 and Tables 14-15 measured GLASS Monte Carlo error on the base DFM on
*both* sides of the comparison, although the caption and App F.2 describe finite-N **dMFM**
estimates against a GLASS-2048 reference. These scripts rerun the identical protocol with
the dMFM posterior sampler on the finite-N side. Full background, the exact protocol table
and the paper-text fixes: `docs/provenance/dna/scalefree.md`.

| file | what it does |
|---|---|
| `fig5_dmfm_gradient_mc.sbatch` | **GPU route**: job array, one task per (reward, length); `kempner_requeue`, `--requeue`, throttled to 4 GPUs, resumable (a task whose `L{L}/metadata.json` exists exits at once) |
| `fig5_dmfm_gradient_mc_cpu.sbatch` | **CPU route** for when `kempner_requeue` is backed up: `serial_requeue`, `--account=albergo_lab`, 96 tasks (reward x length x 8 probe ranges of 4 states) so no task runs longer than ~1 h. Computes only the dMFM estimates and reuses the published GLASS-2048 reference and reward calibration |
| `smoke.sbatch` | timing/sanity smoke: the full per-probe protocol on 2 conditioning states at the worst-case length, both students |
| `collect.sh` | CPU aggregation: merges probe-range parts, rescores, publishes into `results/dna/scale_free_mc/dmfm/`; re-runnable at any time |

Which route to prefer: the GPU one, because it recomputes the GLASS-2048 reference on the
same device and in the same precision (TF32) as the published run. The CPU route exists
because on CPU that reference costs ~7 h at L=50 and ~43 h at L=400 (79 % of the work),
while the dMFM estimates cost ~1.2 h and ~7.3 h — so it reuses the published reference and
computes the estimates in fp32. The precision consequence is measured, not assumed:
`dmfm.scalefree.precision_bound` bounds how far a CPU-produced Table 14/15 cell can sit from
a GPU-produced one. See `results/dna/scale_free_mc/dmfm/PROVENANCE.md`.

```bash
# smoke first (gpu_test; add --partition/--account overrides if its 2-job QOS limit is full)
sbatch scripts/dna/scalefree/smoke.sbatch

# the real thing: 12 shards for the diagonal students of Tables 13/17
sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch

# the 4-step students of Table 18, reusing the primary run's GLASS-2048 reference
STUDENT=dmfm4 REFERENCE_FROM=outputs/dna/scale_free_mc_dmfm/diag1step \
  sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch

# CPU route (serial_requeue), when the GPU partition is backed up
PUBLISHED_ROOT=/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir \
  sbatch --array=0-95%64 scripts/dna/scalefree/fig5_dmfm_gradient_mc_cpu.sbatch

# aggregate; rerun whenever more shards land
bash scripts/dna/scalefree/collect.sh                                     # GPU route
OUTPUT_ROOT=outputs/dna/scale_free_mc_dmfm_cpu bash scripts/dna/scalefree/collect.sh   # CPU route

# a single failed/preempted shard (task = reward_index * 4 + length_index,
# rewards gc,motif,conjunction and lengths 400,200,100,50)
sbatch --array=5 scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch
```

Everything is also runnable directly:
`python -m dmfm.scalefree.run_shard --length 50 --reward motif --student dmfm`,
`python -m dmfm.scalefree.collect`, `python -m dmfm.scalefree.verify_reference --help`.
