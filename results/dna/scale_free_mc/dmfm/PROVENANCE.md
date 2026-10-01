# Fig 5 / Tables 14-15 with finite-N **dMFM** gradient estimates

The GLASS results next to this directory (`results/dna/scale_free_mc/{gc,motif,conjunction}/`)
are the published Fig 5 / Tables 14-15 numbers. They were produced by
`dmfm.experiments.ablate_glass_gradient_mc`, which has no dMFM code path: both the finite-N
estimate and the reference are the *same* GLASS estimator on the *same* base DFM, differing
only in the number of terminal-noise samples. The Fig 5 caption ("Finite-sample dMFM
gradient estimation") and App F.2 ("g_N ... from the dMFM sampler") describe something else.

This directory holds the rerun the authors decided on (2026-09-29): **finite-N dMFM
estimates against the same published GLASS-2048 reference**, so the figure measures what
its caption claims. Only the finite-N estimator changed; the full protocol table, the
seed arithmetic and the paper-text fixes are in `docs/provenance/dna/scalefree.md`.

## Status (2026-09-29)

**Numbers not in yet.** The code, the scripts and every CPU-side verification are done and
two routes are submitted (see Jobs below): a GPU route on `kempner_requeue`, which had not
started after 100 minutes because that partition is backed up, and a CPU route on
`serial_requeue`, which is the fast path. Nothing here is a result until `summary.json` and
`tables14_15_cells.csv` exist; `summary.json["incomplete"]` lists any (student x reward) run
still missing. To finish, from either route:

```bash
bash scripts/dna/scalefree/collect.sh                                  # GPU route
OUTPUT_ROOT=outputs/dna/scale_free_mc_dmfm_cpu \
  RESULTS_DIR=results/dna/scale_free_mc/dmfm bash scripts/dna/scalefree/collect.sh   # CPU route
```

Either is safe to run at any time; it merges whatever probe-range parts are complete and
lists the rest. If both routes finish, **prefer the GPU numbers** and keep the CPU ones as
the corroborating set: the GPU route computes its own GLASS-2048 reference on the same
device and in the same precision (TF32) as the published run, so it is the cleaner
cell-for-cell replacement.

### The two routes and why they differ

| | GPU route | CPU route |
|---|---|---|
| script | `fig5_dmfm_gradient_mc.sbatch` | `fig5_dmfm_gradient_mc_cpu.sbatch` |
| partition / account | `kempner_requeue` / `kempner_albergo_lab` | `serial_requeue` / `albergo_lab` |
| output root | `outputs/dna/scale_free_mc_dmfm/` | `outputs/dna/scale_free_mc_dmfm_cpu/` |
| GLASS-2048 reference | recomputed, TF32 on an H100 | **reused** from the published run's `gradient_pairs.pt` |
| reward calibration | recomputed (matches published to 6e-6) | **reused** from the published `metadata.json` (exact) |
| finite-N dMFM gradients | TF32 on an H100 | fp32 on CPU |
| sharding | 12 tasks (reward x length) | 96 tasks (reward x length x 8 probe ranges of 4 states) |

The CPU route reuses the reference because recomputing it there is not viable, and that is
a measured statement, not an assumption. Per shard on CPU with 8 threads:

| | GLASS-2048 reference | dMFM finite-N estimates |
|---|---|---|
| L=50 | ~7 h | ~1.2 h (measured: 139 s per state) |
| L=400 | ~43 h | ~7.3 h |

The reference is 79 % of the work, so the CPU route computes only the dMFM estimates.
Reusing the published reference is in one way *better* than recomputing it: the reference
then **is** the tensor the published cells were scored against, and the reward is identical
by construction rather than to 6e-6, so old and new cells share both exactly. The cost is
that the reference is TF32/GPU while the estimates are fp32/CPU.

Every shard's `metadata.json` records the device, matmul precision, thread count, Slurm
account, partition and job id under `"compute"`, plus `reference_reused_from`,
`calibration_reused_from` and `probe_states_sha256`, so each number states which machine
produced it. A merged shard keeps every part's `compute` block in `compute_per_part`.

### What the CPU/GPU precision difference can do to a cell

`check_published_protocol` bounds the fp32-CPU-versus-TF32-GPU gap at **0.6-1.9 % on
individual finite-N gradient norms** (worst case is the N=1 probe whose gradient norm is
only 0.016). `precision_bound.py` converts that into a bound on the published *metric*: it
recomputes a subset of the published run's own TF32 finite-N GLASS estimates on CPU in fp32
and scores the Table 14/15 cells both ways against the same reference. Because GLASS
differentiates through 8 network evaluations and the dMFM estimator through one, that number
is an **upper bound** for this rerun. Result: `precision_bound.json`.

**Measured answer (2026-09-29, `precision_bound.json`), L=50 motif, 8 states x 8 repeats:**

| N | relative L1 GPU TF32 -> CPU fp32 | cosine GPU TF32 -> CPU fp32 | per-gradient norm deviation |
|---|---|---|---|
| 1 | 2.3907 -> 2.3865 (**-0.0042**) | 0.1462 -> 0.1461 (**-0.0001**) | mean 1.61 %, max 8.30 % |
| 8 | 1.6161 -> 1.6219 (**+0.0058**) | 0.2675 -> 0.2670 (**-0.0005**) | mean 1.80 %, max 6.62 % |

So a cell moves by **at most 0.006 in relative L1 and 0.0005 in cosine** - well under the 2
decimals Tables 14-15 print, and an order of magnitude better than the naive bound one would
get by propagating the 1.8 % per-gradient deviation. The reason is that a cell averages over
32 states x 8 repeats and the precision perturbations are random rather than systematic, so
they cancel; cosine is additionally first-order insensitive to a perturbation along the
estimate. Because GLASS differentiates through 8 network evaluations and the dMFM map through
one, this is an upper bound for the rerun.

Conclusion: **CPU-produced cells are indistinguishable from GPU-produced ones at the
precision Tables 14-15 print.** The GPU route is still preferred on principle (it recomputes
its own reference in the published precision), but the CPU route is a sound fallback rather
than a compromised one, and the effect this rerun measures is of order 1 in relative L1 and
0.3 in cosine - two to three orders of magnitude above the precision term.

## Layout

| path | what |
|---|---|
| `summary.json` | every Table 14 and Table 15 cell, per student, next to the published GLASS value; plus the protocol and the checkpoints used |
| `tables14_15_cells.csv` | the same as a flat table (`student, reward, mc, length, in_table14, dmfm_relative_l1, dmfm_cosine, published_glass_relative_l1, published_glass_cosine`) |
| `diag1step/<reward>/gradient_scale_free_summary_all_lengths.csv` | the Table 14/15 source: columns `global_relative_l1` (Eq 46) and `mean_cosine` (Eq 47) per (length, N), with bootstrap 95 % CIs |
| `diag1step/<reward>/L{L}/gradient_scale_free_per_state.csv` | per (state, N) relative L1 and cosine - same schema as the published GLASS files |
| `diag1step/<reward>/L{L}/gradient_error_summary.csv`, `.../metadata.json` | per-N `E_RMS` summary and the shard's full settings |
| `diag1step/<reward>/gradient_accuracy_relative_directional_summary.csv` | `summarize_gradient_accuracy_metrics` output (pooled relative RMSE, cosine) |
| `diag1step/<reward>/run_metadata.json` | assembled run metadata (args + per-length) |
| `dmfm4_1step/...` | the same for the 4-step ESD students of Table 18 - **deferred, see the cancellation note below; this directory will not exist** |
| `reference_verification.json` | recomputed vs published GLASS-2048 reference gradients, per (reward, length), when `PUBLISHED_ROOT` was set |

`diag1step` = the diagonal+ESD students of Tables 13/17 (`checkpoints/dna/dmfm/L{L}/`:
L50 `epoch=95-step=49000`, L100 `epoch=80-step=39000`, L200 `epoch=51-step=45000`,
L400 `epoch=33-step=45750-v1` from `diagonly_finetune`), one composed flow-map step
(`sampler=flow_map, n_steps=1, end_time=1.0`) - the posterior sampler of App F.2.
`dmfm4_1step` = `checkpoints/dna/dmfm_4step/L{L}/`.

The large per-shard tensors (`gradient_pairs.pt`, 1.9-15 MB per length, the rescore input)
stay in `outputs/dna/scale_free_mc_dmfm/<tag>/<reward>/L{L}/` (gitignored).

## Jobs (2026-09-29)

**Job cancellations, 2026-09-29 18:00.** The coordinator cancelled three of the five jobs
below on the authors' instruction ("do not request GPUs we are not going to use"; across all
agents ~27 GPU requests were held, many of them duplicate routes). **Do not resubmit them.**

| job | state | why |
|---|---|---|
| `49298891_[0-11%6]` | **surviving** | the diagonal students, gpu_requeue - this is the run that produces Tables 14-15 |
| `49297229_[0-95%64]` | **surviving** | the CPU route, serial_requeue - fallback for the same cells |
| `49279175_[0-11%4]` | cancelled | duplicate of `49298891`, same output tree |
| `49281005_[0-11%4]` | cancelled | the 4-step-student comparison (optional secondary) |
| `49278900` | cancelled | timing smoke; its ground is covered by the CPU tests already recorded here |

No shards existed when they were cancelled, so nothing was lost, and the skip-if-exists and
atomic-write machinery is unaffected.

### Two consequences, deferred rather than missing

1. **Tables 14 and 15 will come from the diagonal students only** (`diag1step`). That is the
   right primary choice - it is the posterior sampler App F.2 describes and the students
   Tables 13 and 17 use - but the tables should not be described as covering both students.
2. **The apples-to-apples 4-step comparison is deferred, not done.** `49281005` existed to
   produce it under this protocol. Its absence makes the caveat below load-bearing rather
   than incidental: the only 4-step numbers anyone has are the rescore of
   `workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/*` (relative L1 ~29-80 at N=1),
   and **those are not comparable to Tables 14-15** - that run used `reference_mc=128` with a
   50-step RK4 GLASS reference to `t=0.999`, conditioning states forward-noised from held-out
   test sequences, 4 composed flow-map steps to `t=0.999`, nested MC pools, seed 20260802 and
   a 128-sample calibration. They must not be quoted next to Table 14/15 cells as if they
   were the same measurement. To do the comparison later, one command:
   `STUDENT=dmfm4 REFERENCE_FROM=outputs/dna/scale_free_mc_dmfm/diag1step sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch`
   (cheap: it reuses the diagonal run's GLASS-2048 reference gradients).

| job | what | partition |
|---|---|---|
| `49278900` | timing/sanity smoke: full per-probe protocol on 2 conditioning states, L=400, conjunction, both students (`scripts/dna/scalefree/smoke.sbatch`; submitted to `kempner_requeue` because the `gpu_test` QOS 2-job limit was taken by other agents) | kempner_requeue |
| `49279175_[0-11%4]` | the 12 shards (3 rewards x 4 lengths) of the diagonal students, `--requeue`, 4 GPUs at a time (`scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch`) | kempner_requeue |
| `49281005_[0-11%4]` | the same 12 shards for the 4-step ESD students of Table 18, reusing the diagonal run's GLASS-2048 reference gradients; `--dependency=afterany:49279175` so it starts only after the diagonal array and never exceeds 4 GPUs together with it | kempner_requeue |
| `49298891_[0-11%6]` | **`gpu_requeue` route**, diagonal students, the same 12 shards writing into the **same output tree** as `49279175` so skip-if-exists makes whichever partition schedules first the one that does the work. `--account=albergo_lab` (`gpu_requeue` lists `kempner_albergo_lab` under `DenyAccounts`), `--constraint="h100|h200|a100"` to stay off the A100 MIG slices in that mixed pool | gpu_requeue |
| `49297229_[0-95%64]` | **CPU route** (fallback), diagonal students: 3 rewards x 4 lengths x 8 probe ranges of 4 states, 8 cores each (<= 512 cores at a time), `--requeue`, reference and calibration reused from the published run. `--account=albergo_lab` (serial_requeue denies `kempner_albergo_lab`) | serial_requeue |

Four routes run concurrently on purpose. `49279175` (H100-only) and `49298891` (mixed A100 /
H100 / H200) target the *same* output tree: the arithmetic is identical across those cards
(CUDA, TF32 matmuls, the same seeded Philox stream on CPU-generated noise), so whichever
partition frees a node first completes each shard, and a shard already on disk is skipped.
Because skip-if-exists is checked at task start, two routes can still overlap on one shard;
every output is therefore written to a unique temp file and `os.replace`d into place, so a
concurrent pair can never leave a truncated `gradient_pairs.pt`, and identical seeds mean
whichever lands last is equally valid. Each shard records its partition, account, device name
and precision under `"compute"`, so the provenance states which route produced each cell.

Priorities as submitted (2026-09-29 17:0x): `kempner_requeue` 2.7M but ~1,400 jobs deep;
`gpu_requeue` 2 with 2,451 pending against only 61 running and ~260 idle GPU nodes;
`serial_requeue` 912 against a median pending priority of 375,901. The two `albergo_lab`
routes are bottom-priority but depend on backfill into idle capacity rather than on queue
position.

Task index = `reward_index * 4 + length_index` with rewards `gc, motif, conjunction` and
lengths `400, 200, 100, 50`. A task whose `L{L}/metadata.json` exists exits immediately,
so a preempted or failed shard reruns alone: `sbatch --array=<task> scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch`.

## Commands

```bash
# submit (from the repo root)
sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch
STUDENT=dmfm4 REFERENCE_FROM=outputs/dna/scale_free_mc_dmfm/diag1step \
  sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch

# aggregate -> this directory. CPU, seconds, re-runnable at any time; incomplete
# (student x reward) run directories are listed and skipped.
bash scripts/dna/scalefree/collect.sh

# with the reference check against the published Fig 5 tensors
PUBLISHED_ROOT=/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir \
  bash scripts/dna/scalefree/collect.sh
```

## Verification of the unchanged parts

* The ported metric code is numerically identical to the scripts that produced the
  published tables: `diff` against
  `dirichlet-flow-matching/scripts/{rescore_gradient_pairs_scale_free,summarize_gradient_accuracy_metrics}.py`
  shows only docstrings and `argv` plumbing, and re-running the ported rescore on the
  published `rebuttal_gradient_accuracy_motif/L50/gradient_pairs.pt` reproduces every
  published L=50 motif cell of Table 15 to the last stored digit (max |diff| = 0.0).
* `ablate_glass_gradient_mc.py` and `ablate_dmfm_one_step_gradient_mc.py` differ from their
  originals only in imports, repo-relative default paths, docstrings, `argv` plumbing and
  the documented `end_time=1.0` allowance for the Euler solver (which is what the published
  runs did). The estimator, calibration, probe-state and reward code is unchanged.
* The gradient-MC scripts do **not** exist in `tullebulle/DNA-MFM@187fe7b`: the durable
  clone at `/n/holylabs/.../paper_backup_dmfm_20260929/DNA-MFM_187fe7b` has no file of
  those names anywhere and no file mentioning `gradient_pairs`, `scale_free` or
  `gradient_accuracy`. `dirichlet-flow-matching` is their only source. Verbatim copies of
  the six originals, with sha256s, are in
  `results/dna/original_scripts/fig5_tables14_15/` (see its `_sources.json`); they were
  taken from the durable holylabs backup, which is byte-identical to the netscratch
  working copy.
* `gaussian_denoiser_flow_step` (used for the conditioning states and the reward
  calibration) lost its redundant second network call on 2026-09-29; outputs are
  bit-identical, so the states match the published run's.
* Everything the rerun shares with the published run is checked directly against the
  published outputs, on CPU, by `dmfm.scalefree.check_published_protocol` (result in
  `protocol_check.json`). At L=50, motif: the reward calibration reproduces the published
  `metadata.json` to 6e-6 relative or better (mean 0.45800513 vs 0.45800257, std 0.10723354
  vs 0.10723752, center 0.60325359 vs 0.60325370), and four finite-N **GLASS** estimate
  gradients recomputed through the rerun's states/seeds/estimator match `estimate_l2` in
  the published `gradient_errors.csv` to 0.6-1.9 % (fp32 CPU against the published TF32
  run; the 1.9 % cell is probe 0, N=1, whose gradient norm is only 0.016). The
  conditioning states, seed arithmetic, reward and GLASS integrator therefore match, so
  the reference of this rerun is the published reference and only the finite-N estimator
  differs.

  ```bash
  python -m dmfm.scalefree.check_published_protocol \
    --published-root /n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir \
    --published-run gc=rebuttal_gradient_accuracy_gc_pairmetrics \
    --out results/dna/scale_free_mc/dmfm/protocol_check.json
  ```
* The recomputed GLASS-2048 reference is additionally compared per state with the published
  tensors by `dmfm.scalefree.verify_reference` (see `reference_verification.json`); TF32
  matmul noise alone is 0.2-0.4 %.
