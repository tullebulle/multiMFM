# Fig 5 / Tables 14-15: the scale-free Monte Carlo gradient analysis

## What the paper claims and what was actually run

Fig 5's caption reads "Finite-sample **dMFM** gradient estimation from 50 to 400 bp ...
Each estimate is compared with an independent GLASS-2048 reference", and App F.2 defines
`g_N` as "an estimate using N independent terminal samples from the **dMFM** sampler".

The published numbers do not come from a dMFM sampler. They come from
`scripts/ablate_glass_gradient_mc.py` (ported to
`dmfm.experiments.ablate_glass_gradient_mc`), run as
`workdir/rebuttal_gradient_accuracy_{gc_pairmetrics,motif,conjunction}` and rescored on
2026-08-02 by `rescore_gradient_pairs_scale_free.py` +
`summarize_gradient_accuracy_metrics.py`. That script builds only a `DiTSequenceModel` and
has no dMFM code path: each length's `metadata.json` records a single checkpoint, the base
DFM `yeast_parent_a_L{L}_dfm_dit_h192_b4/best.pt`. Both sides of the comparison are
therefore the *same* GLASS estimator on the *same* base DFM, differing only in the number
of terminal-noise samples (N vs 2,048). The paper concedes this at l.1727, where Table 15
is described as comparing "GLASS Monte Carlo error across lengths, without measuring dMFM
approximation error" - which contradicts the Fig 5 caption and App F.2.

Established by the dna-core agent from the stored run arguments and metadata; the stored
`results/dna/scale_free_mc/{gc,motif,conjunction}/gradient_scale_free_summary_all_lengths.csv`
reproduce Tables 14 and 15 cell for cell (columns `global_relative_l1` and `mean_cosine`).

## The rerun (authors' decision, 2026-09-29)

`dmfm.scalefree` reruns the identical protocol with **finite-N dMFM estimates** against the
**same GLASS-2048 reference**, so Fig 5 measures what its caption claims.

Only the finite-N estimator changes. Everything else is taken from the published run:

| ingredient | value | where it comes from |
|---|---|---|
| rewards | quadratic GC target (`z_target=1`), smooth TTTTTC occurrence, smooth conjunction of TTTTTC and AAAATT (`motif_tau=conjunction_tau=0.10`, `target_percentile=0.90`) | `ablate_glass_gradient_mc.target_reward`, unchanged |
| lengths | 50, 100, 200, 400 | published |
| reward calibration | 1,024 base-DFM samples, 64 NFE to `t=0.999`, per length | `calibrate_score`, unchanged |
| conditioning states | 32 per length = base flow from Gaussian noise to `t_eval=0.5` with 32 NFE, `seed + 1_000_000 + L` | `make_probe_states`, unchanged |
| noise pools | 8 per state per N, `seed + 20_000_000 + 1_000_000 L + 10_000 probe + 100 N + repeat` | published seed arithmetic |
| N grid | 1, 2, 4, 8, 16, 32, 64, 128, 256 (Fig 5 plots 1, 4, 16, 64, 256; Table 14 reports 8 and 128) | published |
| reference | GLASS posterior on the base DFM, **8 Euler steps to `t=1.0`**, 2,048 terminal-noise samples, `seed + 10_000_000 + 10_000 L + probe` | published (`--nfe_value 8`; the runs predate `--glass_end_time`, so they integrated to 1.0 - see `fig5_tables14_15_gradient_mc.sbatch`) |
| seed | 20260726 | published |
| metrics | pooled relative L1 (Eq 46) and mean cosine (Eq 47) | `rescore_gradient_pairs_scale_free`, unchanged, `--bootstrap 10000 --seed 20260802` as in the published rescore |
| **finite-N estimator** | **dMFM posterior sampler, one composed flow-map step (`sampler=flow_map, n_steps=1, end_time=1.0`)** | **the change**; App F.2's posterior sampler |

Both value gradients are exact autodiff through the sampler with the published two-pass
chunked estimator (`mc_chunk=16`), i.e. `estimate_value_gradient` from
`ablate_glass_gradient_mc` for the reference and from
`ablate_dmfm_one_step_gradient_mc` for the dMFM side. Neither function was modified.

### Students

Primary (`--student dmfm`, output tag `diag1step`): the diagonal+ESD students that Tables
13 and 17 use, i.e. `dmfm.paths.dmfm_ckpt(L)`:

| L | checkpoint |
|---|---|
| 50 | `checkpoints/dna/dmfm/L50/epoch=95-step=49000.ckpt` |
| 100 | `checkpoints/dna/dmfm/L100/epoch=80-step=39000.ckpt` |
| 200 | `checkpoints/dna/dmfm/L200/epoch=51-step=45000.ckpt` |
| 400 | `checkpoints/dna/dmfm/L400/epoch=33-step=45750-v1.ckpt` (`diagonly_finetune`) |

Secondary (`--student dmfm4`, tag `dmfm4_1step`): the 4-step ESD students of Table 18
(`dmfm.paths.dmfm4_ckpt(L)`), evaluated with the *same* one-step map, purely for
comparison. Its GLASS-2048 reference gradients are reused from the primary run
(`--reference_from`), because the reference does not depend on the student.

Caveat on the existing 4-step numbers: the rescore of
`workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/*` (relative L1 ~29-80 at N=1)
is **not** comparable to Tables 14-15. That run used `reference_mc=128` with a 50-step RK4
GLASS reference to `t=0.999`, conditioning states forward-noised from held-out test
sequences, 4 composed flow-map steps to `t=0.999`, nested MC pools, seed 20260802 and a
128-sample calibration. The `dmfm4_1step` run here is the apples-to-apples version.

## Deviation from the paper's own description

App F.2 and the Fig 5 caption say the 32 conditioning states are "held-out". They are not:
the published run integrates the base flow from Gaussian noise to `t=0.5`
(`make_probe_states`), so the states are model samples, not noised held-out data. This
rerun keeps that construction so the cells stay comparable; the text should say
"32 conditioning states obtained by integrating the base flow to t = 0.5" instead.
(`ablate_dmfm_one_step_gradient_mc.make_heldout_probe_states` is the held-out variant,
used by the Table 18 runs only.)

## Jobs (2026-09-29)

**Surviving, one route per item** after the 18:00 cancellations (see below):
`49298891_[0-11%6]` diagonal students on `gpu_requeue` (`--account=albergo_lab`,
`--constraint="h100|h200|a100"`) and `49297229_[0-95%64]` the CPU route on `serial_requeue`.

**Cancelled 2026-09-29 18:00 on the authors' instruction — do not resubmit:**
`49279175_[0-11%4]` (kempner_requeue diagonal students, a duplicate of `49298891` writing to
the same tree), `49281005_[0-11%4]` (the 4-step-student comparison) and `49278900` (timing
smoke). No shards existed, so nothing was lost.

Two consequences, **deferred rather than missing**: Tables 14-15 will come from the diagonal
students only, and the apples-to-apples 4-step comparison is not done. The latter makes the
caveat in "Which dMFM" below load-bearing — the only 4-step numbers in existence come from a
run with a different reference, different states and a different sampler, and must not be
quoted beside Table 14/15 cells. Full detail and the one command that would produce the
deferred comparison: `results/dna/scale_free_mc/dmfm/PROVENANCE.md`.

## Two routes: GPU and CPU

Prefer the GPU route. It recomputes the GLASS-2048 reference on the same device and in the
same precision (TF32) as the published run, so it is the cleaner cell-for-cell replacement.

The CPU route exists only because the GPU partition was backed up. Measured CPU cost per
shard (8 threads): the GLASS-2048 reference is ~7 h at L=50 and ~43 h at L=400, i.e. 79 % of
the work and not viable; the dMFM estimates are ~1.2 h and ~7.3 h (139 s per state at L=50),
which is viable when split into probe ranges. So the CPU route computes only the dMFM
estimates and reuses the published reference and reward calibration
(`--reference_from`, `--calibration_from`). That reuse is in one respect *better* than
recomputation: the reference then is the tensor the published cells were scored against, and
the reward is identical by construction rather than to 6e-6.

The cost is a fp32-CPU vs TF32-GPU difference on the estimate side.
`dmfm.scalefree.precision_bound` measures its effect **on the published metric**, not just
on gradient norms: it recomputes a subset of the published run's own TF32 finite-N GLASS
estimates on CPU in fp32 and scores the Table 14/15 cells both ways against the same
reference. GLASS differentiates through 8 network evaluations and the dMFM map through one,
so that is an upper bound here. Result in
`results/dna/scale_free_mc/dmfm/precision_bound.json`.

Probe-range sharding: `--probe_start/--probe_end` write `L{L}/parts/probes_A_B.{pt,csv,json}`
and `dmfm.scalefree.collect` merges a set that tiles `[0, 32)`. Every part stores a sha256 of
the 32 conditioning states and the merge refuses to combine parts that disagree, so parts
computed in different processes can never be mixed at different states. Each part also
records the device, matmul precision, thread count, Slurm account, partition and job id under
`"compute"`; a merged shard keeps them all in `compute_per_part`.

## How to run it

```bash
# 12 shards (reward x length), one GPU each, throttled to 4, resumable
sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch
# the 4-step students, reusing the GLASS reference (cheap)
STUDENT=dmfm4 REFERENCE_FROM=outputs/dna/scale_free_mc_dmfm/diag1step \
  sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch
# aggregate -> results/dna/scale_free_mc/dmfm/ (CPU, re-runnable)
bash scripts/dna/scalefree/collect.sh
```

A shard whose `L{L}/metadata.json` exists exits immediately (`--skip_existing`), so a
preempted array task or a single failed shard reruns alone with
`sbatch --array=<task> ...`. Timing smoke: `scripts/dna/scalefree/smoke.sbatch`
(2 states, full N grid and reference, worst-case length; multiply per-probe time by 32).

`scripts/dna/scalefree/collect.sh` optionally verifies that the recomputed GLASS-2048
reference matches the published Fig 5 tensors:

```bash
PUBLISHED_ROOT=/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/workdir \
  bash scripts/dna/scalefree/collect.sh
```

## Outputs

`outputs/dna/scale_free_mc_dmfm/<tag>/<reward>/L{L}/` (gitignored) holds
`gradient_pairs.pt` (the rescore input), `gradient_errors.csv`,
`gradient_error_summary.csv`, `score_calibration_samples.npy` and `metadata.json`.

`results/dna/scale_free_mc/dmfm/` (tracked) holds, per `<tag>/<reward>/`, the same small
files as the published GLASS directories beside it
(`results/dna/scale_free_mc/<reward>/`): `gradient_scale_free_summary_all_lengths.csv`,
`L{L}/gradient_scale_free_per_state.csv`, `L{L}/gradient_scale_free_summary.csv`,
`gradient_accuracy_relative_directional_summary.csv`, `run_metadata.json`, plus
`summary.json` (every Table 14 and 15 cell, new next to the published GLASS value),
`tables14_15_cells.csv` (the same as a flat table) and `PROVENANCE.md` (job IDs).

## Original producer scripts

Verbatim copies, with sha256s and the port diff, are in
`results/dna/original_scripts/fig5_tables14_15/` (`_sources.json` there records
everything). They were taken from the durable backup
`/n/holylabs/.../paper_backup_dmfm_20260929/dirichlet-flow-matching/{scripts,slurm}`, whose
copies are byte-identical to the purge-prone netscratch working copy.

These experiments are **not** in `tullebulle/DNA-MFM@187fe7b`: that clone's `scripts/` and
`slurm/` contain no `ablate_glass_gradient_mc.py`,
`rescore_gradient_pairs_scale_free.py` or `summarize_gradient_accuracy_metrics.py`, and no
file mentioning `gradient_pairs`, `scale_free` or `gradient_accuracy`. The rebuttal runs are
from 2026-07-26 / 08-02 and live only in `dirichlet-flow-matching`, which is therefore the
authoritative original for Fig 5 and Tables 14-15. The ported modules in `src/dmfm` differ
from those originals only in imports, repo-relative default paths, docstrings, `argv`
plumbing and the documented `end_time=1.0` allowance for Euler; nothing numerical changed,
and the ported rescore reproduces the published L=50 motif Table 15 column bitwise.

## Code

- `src/dmfm/scalefree/run_shard.py` - one (length, reward, student) shard.
- `src/dmfm/scalefree/collect.py` - assembles `run_metadata.json`, calls the two published
  rescore scripts unchanged, publishes to `results/`, writes `summary.json`.
- `src/dmfm/scalefree/verify_reference.py` - recomputed vs published GLASS-2048 reference.
- `src/dmfm/scalefree/check_published_protocol.py` - CPU check that everything the rerun
  shares with the published run (calibration, conditioning states, seed arithmetic, GLASS
  estimator) reproduces the published outputs. Verified 2026-09-29 at L=50/motif:
  calibration to 6e-6 relative, four finite-N GLASS estimate gradient norms to 0.6-1.9 %
  of the published `gradient_errors.csv` (fp32 CPU vs the published TF32 run). Result in
  `results/dna/scale_free_mc/dmfm/protocol_check.json`.
- `scripts/dna/scalefree/{fig5_dmfm_gradient_mc.sbatch,smoke.sbatch,collect.sh}`.
