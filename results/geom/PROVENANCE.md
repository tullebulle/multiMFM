# GEOM-Drugs results provenance

Every file in `results/geom/` and where its numbers come from. All source runs lived in
`/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/geom-drug-tabasco` (`SRC` below) and are
backed up under `/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/`.
Nothing here was recomputed or rounded: the CSV/JSON values are read straight out of the
recorded run JSONs and the wandb run file.

Artifacts used by every steering run (see `docs/provenance/geom/checkpoints.md`):

| role | repo path |
|---|---|
| base model (teacher, EMA weights) | `checkpoints/geom/teacher/epoch=38-step=89310.ckpt` |
| multiMFM student (Stage-2b, step 10,000) | `checkpoints/geom/mfm_student/student_step_10000.pt` |
| fingerprint guide head `E` | `checkpoints/geom/structure_head/best.pth` |
| target fingerprints | `data/geom/geom_fp/val.lmdb` (val split) |
| PoseBusters config | `configs/geom/posebusters_geom.yaml` |

## Table 3 (multiMFM row)

| file | run | command |
|---|---|---|
| `table3_n64_grid.csv` | Slurm **35670937**, 2026-07-28, 1x H200, seed 0, n=64 | `SRC/glass/run_mfm_capgap_steering.sbatch` -> `glass/mfm_fingerprint_steering.py`, four (lambda, mu) points |
| `table3_n4096_shards.csv`, `table3_n4096_replicates.csv` | Slurm **47833756**, 2026-09-22, 4x H200, 3 replicates x 4 shards of 1,024 | `SRC/glass/run_capgap_4096_3seed.sbatch`, same configuration as 35670937 except `--num-mols 1024` and the seeds |
| `table3_repro_smoke_n64.csv` | Slurm **47833754**, 2026-09-22 | `SRC/glass/run_capgap_smoke_n64.sbatch`: the exact Table 3 point re-run post-purge under the `multimfm` env |
| `table3_per_molecule.csv` | jobs 35670937 and 47833756 | per-molecule FP2 Tanimoto and FP2-validity of both arms (`tanimoto_real_per_mol`, `valid_per_mol` in the run JSONs), rounded to 6 decimals; lets the notebook recompute the paired gains and SEMs |

Source JSONs: `SRC/outputs/mfm_capgap_steering/`, `SRC/outputs/capgap_4096_3seed/`,
`SRC/outputs/capgap_smoke/`. Logs: `SRC/logs/slurm/mfm_capgap-35670937.out`,
`capgap4096-47833756.out`, `capgap_smoke-47833754.out`.

The paper's Table 3 value (0.227) and its 87.5% PB validity are the `lambda=0.02, mu=0.5`
row of `table3_n64_grid.csv` (unguided 0.2072 -> guided 0.2272, PB 0.875 -> 0.875). The
other three grid points give gains of +0.0103, +0.0018 and -0.0087, so 0.227 is the best of
four at n=64 with one seed.

`table3_n4096_replicates.csv` is the same configuration at n=4,096 x 3 seeds: pooled paired
gain **-0.0046 +- 0.0008** (unguided 0.2287 -> guided 0.2241), i.e. 5.7 SEM *below* zero, and
PB validity 88.0% -> 85.0%. Replicate 2 has 3 shards instead of 4: seed 21 crashed on a
72-atom target (the bug fixed by `multimfm.geom.fingerprint.sample_targets`, see
`docs/provenance/geom/data.md`). The reproduction smoke run gives gain +0.0156 vs +0.0199
recorded, consistent with n=64 noise and non-deterministic CUDA backward.

## Table 3 rerun exactly as described (authors' decision 1, 2026-09-29)

The authors decided to rerun the row as Sec. 3.2 / App. F.8 / App. F.9 describe it and report
whatever comes out. Code: `multimfm.geom.mfm_steering` (unchanged algorithm); scripts:
`scripts/geom/tune_lambda_mu.sbatch` then `scripts/geom/table3_asdescribed.sbatch`; single
collection command `bash scripts/geom/collect_table3_asdescribed.sh`.

| file | run | what |
|---|---|---|
| `table3_asdescribed_tuning.csv`, `table3_asdescribed_selection.json` | Slurm array **49269354** (2026-09-29; recorded guidance schedule: every 2nd outer step, t in [0.1,0.9]). Array 49268477 was cancelled after Blackwell nodes failed (see below), and array **49271101** -- the literal schedule of Algorithm 2 (every outer step, t in [0,0.95], 0.95 being the student's training horizon from Table 20) -- was cancelled while still queued, to stop it competing with the main sweep for slots on a congested cluster; the command that reruns it is in the header of `scripts/geom/tune_lambda_mu.sbatch` (`GRID_OVERRIDE=... GUIDE_EVERY=1 GUIDE_MIN_T=0.0 GUIDE_MAX_T=0.95`) | (lambda, mu) grid on a **held-out** pool of 512 val targets (seed 20260929), 256 targets per point. The array was a 15-point grid (lambda in {0.01,0.02,0.05,0.1,1.0} x mu in {0.25,0.5,1.0}); **5 points landed** (lambda in {0.01,0.02} x the lower mu values) before its remaining 10 tasks were cancelled under the one-route rule, so lambda in {0.05,0.1,1.0} -- including the paper's literal lambda=1 -- is unmeasured on the held-out pool. See "What was actually measured" below |
| `table3_asdescribed_shards.csv`, `table3_asdescribed_seeds.csv`, `table3_asdescribed.json` | Slurm array **49274410** (2026-09-29, `kempner_requeue`, 24 tasks) | 3 seeds x 8 shards x 512 = 4,096 molecules per seed at the selected (lambda, mu) (lambda=0.01, mu=0.25, guiding every 2nd outer step in t in [0.1,0.9]; selected from the 5 grid points that had finished when the evaluation was launched -- a congested cluster left the other 10 queued, and `table3_asdescribed.json` records whether the final best point still agrees), 100-step outer ODE, 10 flow-map inner steps, 4 posterior samples; the unguided arm shares the noise and targets, so it is the base generator on the same molecules |
| `tfgflow_reference.json` | n/a (literature check) | what TFG-Flow reports and releases (see below) |
| `table3_n64_fragility.json` | n/a (bootstrap of existing data) | how often a 64-molecule estimate drawn from the 4,096x3 population reaches the submitted +0.0199 gain (1.06% per draw; 4.2% for the best of four grid points), with the caveat that the n=64 run's targets were an easier draw (unguided 0.2072 vs 0.2287) |

Why the schedule matters: Algorithm 2 computes the value gradient at **every** outer step with
no t window, while the implementation that produced every recorded number guides every 2nd step
inside t in [0.1, 0.9]; the paper states neither. The selection and the evaluation both use the
recorded schedule; the every-step variant is a pending robustness check (see the note on array
49271101 above).

Why a selection step at all: the paper states the reward *shape* `exp(-||E(G)-c||^2)` but no
numeric lambda or mu, and the submitted number came from the best of a four-point grid at
n=64 with no held-out data. Selection now happens on targets the evaluation never sees
(`multimfm.geom.fingerprint.holdout_target_indices`; every result JSON records
`holdout_n`, `holdout_seed`, `tuned_on_holdout` and how many drawn targets were excluded).
The rule, fixed before looking at the sweep: largest paired Tanimoto gain among points whose
guided PB validity is at most 2 points below their own unguided arm.

Infrastructure note: `kempner_requeue` contains Blackwell (`rtx6000pro`) nodes on which
torch 2.5.1+cu124 raises "no kernel image is available for execution on the device"; both
arrays therefore carry `--constraint="cc8.0|cc9.0"` (A100/H100/H200). Array tasks write
per-shard JSONs and exit immediately if their output exists, so preempted tasks rerun alone.

## CPU runs of the decision-critical probes (authorized 2026-09-29)

`kempner_requeue` had ~1,400 jobs ahead of ours while `serial_requeue` had ~4,300 idle cores, so
the authors approved the `albergo_lab` account on `serial_requeue` for queued work that runs
acceptably on CPU. What moved there, and what did not:

**One route per item** (authors' instruction, 2026-09-29, after ~27 GPU requests had accumulated
across the agents, many of them redundant second and third routes for work that had already moved
to CPU). The GPU copies of the diagnostic and of the sweep were cancelled by the coordinator and
must not be resubmitted without asking.

| work | device | why |
|---|---|---|
| grad-sign diagnostic (4 molecules, 8 probe look-aheads, 6 flow-map steps) | CPU, `serial_requeue`, `albergo_lab` (**49293660**) -- the only route | an 8-molecule directional test; fp32 on CPU cannot change its verdict, so a GPU copy buys nothing |
| constant-mu arm, lambda = 0.01, mu in {0.05, 0.10, 0.25, 0.5, 1.0}, n = 256 | CPU (**49293829**, plus **49293743** for mu = 0.25) -- the only route | decision-critical, not published-facing |
| paper-coefficient arm, lambda = 0.01 x mu-scale {0.05, 0.15, 0.5} and lambda = 0.02 x 0.15, n = 256 | CPU (**49293831**) -- the only route | same |
| the 4,096 x 3 as-described evaluation | **stays on GPU** (**49274410**) | sampling 4,096 molecules per seed through the base model is the least CPU-friendly work here, and it is the deliverable |

Outputs go to a separate directory, `results/geom/runs/asdescribed_tuning_cpu/`, and every shard
now records `device`, `float32_matmul_precision`, `slurm_account`, `slurm_partition`,
`slurm_job_id` and `host` in its result JSON; the pooled table
`results/geom/table3_asdescribed_tuning.csv` carries those columns, and
`scripts/geom/collect_asdescribed.py tune --dir <gpu-dir> <cpu-dir>` pools the two.

**Precision.** CPU shards run in true fp32; the CUDA shards run with TF32 matmuls
(`torch.set_float32_matmul_precision("high")`). CPU is therefore the *more* precise of the two,
and for the question grad-sign asks -- does a step along the estimated gradient raise the guide
reward at all, a change of order 1e-1 in a reward of order 1e2 -- fp32-vs-TF32 cannot plausibly
change the verdict. It does mean shards are not bitwise comparable across devices, which is why
**the constant-mu versus paper-coefficient comparison is run entirely on CPU**: the constant-mu
points at lambda = 0.01 are being re-measured there (mu = 0.05, 0.10, 0.25, 0.5, 1.0) so that
both arms of the comparison share a device. The five original constant-mu points remain the
GPU record.

**What was actually measured (corrected 2026-09-29).** The 15-point constant-mu grid was *not*
completed: 5 of its points ran on GPU (lambda = 0.01 x mu in {0.25, 0.5, 1.0}, lambda = 0.02 x
mu in {0.25, 0.5}) and the remaining 10 were cancelled with the one-route rule. The sweep is
therefore:

| arm | points | device |
|---|---|---|
| constant mu | lambda = 0.01 x mu in {0.25, 0.5, 1.0}; lambda = 0.02 x mu in {0.25, 0.5} | GPU (A100, TF32) -- landed |
| constant mu | lambda = 0.01 x mu in {0.05, 0.10, 0.25, 0.5, 1.0} | CPU (fp32) -- queued |
| paper coefficient | lambda = 0.01 x mu-scale in {0.05, 0.15, 0.5}; lambda = 0.02 x mu-scale 0.15 | CPU (fp32) -- queued |

The comparison the authors will act on is the CPU-vs-CPU one: 5 constant points against 4 paper
points at the same n = 256, the same held-out pool, the same metrics and the same pre-registered
selection rule, overlapping exactly at lambda = 0.01 with mu in {0.05, 0.5}. The paper arm is not
tuned harder than the constant arm -- it has fewer points -- and its mu values are lower only
because `(1-t)/t` multiplies them by up to 9x inside `t in [0.1, 0.9]` (at `t = 0.5` the factor is
1, so the grids are directly comparable there).

**Deliberately not measured, not silently missing**: lambda in {0.05, 0.1, 1.0} on the held-out
pool, including **lambda = 1, the literal reading of the paper's reward** `exp(-||E(G)-c||^2)`.
Those were in the cancelled 10 GPU points. Recovering them is a deliberate resubmission:
`sbatch scripts/geom/tune_lambda_mu.sbatch` reruns the 15-point grid and skips the 5 that exist.
The only evidence on lambda = 1 today is indirect: the older GLASS grid at n = 64 degraded sharply
at lambda = 0.5 (-0.042) and lambda = 1.0 (-0.085), which is consistent with the log-mean-exp
saturating into a single-sample gradient when `lambda*|r| ~ 100`.

## The guidance coefficient: the paper's sigma_t^2/2 vs the code's constant mu

The paper's guided ODE carries `sigma_t^2/2 = (1-t)/t` in front of `grad V`; the implementation
that produced every reported number (GEOM and QM9) adds `mu * grad V * dt` with a constant
`mu`. Inside the recorded window `t in [0.1, 0.9]` that is 0.06x the paper's coefficient at
t = 0.1 (18x under-guided, where composition is still being decided) and 4.5x at t = 0.9
(over-guided, where perturbing a nearly-final geometry is what breaks PB validity) -- a
coherent mechanism for "both similarities fall and PB validity falls". **The published GEOM
runs therefore did not implement the sampler the paper states**, which stands as a finding
independently of how the test below turns out.

`multimfm.geom.mu_schedule` adds `--mu-schedule {constant,paper}` to both steering drivers.
`constant` is the default, so nothing already run changes meaning; `paper` pre-multiplies the
gradient by `(1-t)/max(t, eps_t)` (eps_t = 0.05, which caps the coefficient at 19 and never
binds inside `t in [0.1, 0.9]`) and lets `mu` act as an overall scale. Pre-scaling the gradient
is algebraically identical to scaling the coefficient because the step is linear in the
gradient; verified numerically (ratio paper/constant = 9.000 at t = 0.1, matching (1-t)/t), and
the installer is idempotent and reversible.

Held-out sweep with the paper coefficient: originally GPU array 49281926, cancelled and moved to
CPU as **49293831** (the only route): 4 points, lambda = 0.01 x mu-scale in {0.05, 0.15, 0.5} and
lambda = 0.02 x mu-scale 0.15, n = 256 on the same held-out pool with the same metrics. Its
constant-mu counterpart runs on the same device (**49293829**, plus **49293743**) so the comparison
carries no device confound. Outputs carry a `_muspaper` tag and the `mu_schedule` column in
`results/geom/table3_asdescribed_tuning.csv`, which also records `device`, `slurm_account`,
`slurm_partition` and `slurm_job_id` per point.

## Diagnostic: does the value gradient ascend the guide's own reward?

Every held-out (lambda, mu) point measured so far *lowers* both the counted FP2 similarity and
the guide's head-predicted similarity, so the first question is whether the guidance term is
correctly signed and scaled. `python -m multimfm.geom.diagnostics grad-sign` estimates the
value gradient with the driver's own MFM look-ahead sampler and measures the guide reward of
fresh look-aheads at `x` and at `x +- eps * grad`. Slurm **49278980** (2026-09-29; resubmitted leaner than the first attempt 49278467, which I
cancelled myself so it could backfill) writes `results/geom/grad_sign.json`; the login node
cannot run it (SIGKILLed twice at load ~35 -- the teacher and student checkpoints each carry
a 1.14M-entry `all_smiles` list in `data_stats`).

The diagnostic now has a proper launcher, `scripts/geom/grad_sign.sbatch`, with a documented
CPU fallback: the GPU job (**49278980**) sits behind this account's GPU-heavy arrays, so the
same measurement at a reduced size (4 molecules, 8 probe look-aheads, 6 flow-map steps, a
20-step outer trajectory) was also queued on the uncontended `serial_requeue` CPU partition
as **49283540**, writing `results/geom/grad_sign_cpu.json`. Whichever lands first answers the
question; the GPU job's numbers are the ones to quote if both do.

A complementary probe of the same question -- whether the direction is right but the scale
wrong -- sweeps mu below every value tried so far (mu = 0.05, 0.10 at lambda = 0.01, n = 256
on the held-out pool): originally GPU array 49279787, cancelled and folded into the CPU
constant-mu arm **49293829**. If a small enough mu gives a positive gain,
the direction is useful and the recorded runs were simply over-guided; if every mu harms,
the estimated direction itself is not uphill. The hand audit of the sign convention is in
`docs/provenance/geom/sign_audit.md` (no inversion exists in the code path) and the
measured direction of the guide's own objective is recorded in
`results/geom/head_tanimoto_direction.json`.

## What TFG-Flow actually reports and releases

Checked 2026-09-29 against the TFG-Flow paper (arXiv 2501.14216, ICLR 2025) and its public
repository; recorded in `results/geom/tfgflow_reference.json`.

* The cited 0.170 +- 0.001 and 0.208 +- 0.002 are TFG-Flow's **GEOM-Drug** rows of its
  Table 3, averaged over 3 random seeds, so the dataset matches ours. Its "Upper bound" is
  its own *unconditional* generator.
* TFG-Flow's structure evaluation uses "all the structures in the unseen half of training
  set ... averaged over all the structures" (its App. D.1) and its GEOM-Drug data pipeline
  caps molecules at 30 atoms, against 71 here.
* Its guide for the structure task is "a 6-layer discriminative EGNN by adding linear head
  on the average pooling of output node feature" (App. D.2), trained with BCE on the
  fingerprint -- the architecture our head copies.
* **No structural-fingerprint predictor is released**: `storage/` contains only
  `flow_model.pth` and the QM9 property `guide_clf_ckpt.zip` / `oracle_clf_ckpt.zip`. The
  head used for our Table 3 was trained by the authors (four legs, see
  `docs/provenance/geom/checkpoints.md`), so "off-the-shelf from TFG-Flow" is incorrect.

## Context runs (not in the paper)

| file | run | what |
|---|---|---|
| `glass_steering_grid_n64.csv` | `SRC/outputs/glass_fingerprint_steering/`, Slurm 35385104 and siblings | teacher-look-ahead (GLASS) steering, 12 points on the "big" teacher + 8 on the xlarge teacher, n=64 |
| `mfm_steering_grid_n64_older_student.csv` | `SRC/outputs/mfm_fingerprint_steering/` | multiMFM grid with the earlier Stage-2a student (`stage2_20260727-163603/student_step_20000.pt`), n=64 |
| `n4096_first_run_shards.csv` | Slurm **35640178**, `SRC/outputs/beat_0208_4096/` | first 4,096-molecule evaluation: GLASS +0.0064 mean, multiMFM (older student) -0.0035 mean; predates the PB code, so `pb_valid` is empty |

## Base model / Table 21

| file | source |
|---|---|
| `teacher_pb_curve.csv` | `SRC/validation_teacher/*/results/*.json` (`validation_teacher/check_pb_validity.py`, N=200, 200 steps, `posebusters_no_strain.yaml`). 12 checkpoints of the released "big" run; **no entry for the selected step 89310** |
| `teacher_training_pb_validity.csv` | `val/pb_*` in the teacher run's wandb file `SRC/outputs/2026-07-26/01-03-35/wandb/run-20260726_010353-70pueao1/run-70pueao1.wandb`, extracted with `wandb.sdk.internal.datastore`; 39 epochs, N=100, 100 steps, `posebusters_no_strain.yaml` |
| `unguided_baseline.json` | unguided arms of the two Table 3 runs above |
| `table21_geom_row.json` | the three sources above, next to the paper's row |
| `pb_teacher_n100_job49256841.json` | Slurm **49256841**, 2026-09-29: `python -m multimfm.geom.check_pb_validity --ema --num-samples 100 --num-steps 200` on the released checkpoint with the **GEOM** PB config -- the first such measurement (0.880) |
| `port_verification.json` | Slurm **49256831**, 2026-09-29: the ported `multimfm.geom.mfm_steering` re-run of the exact Table 3 point (n=64, seed 0) on an A100, next to the two earlier runs of the same configuration |

**Where 92.0% does and does not appear.** No record measures the base model's PB validity on
4,096 molecules. `pb_intersection = 0.92` occurs in the teacher's training-time validation
(N=100, no-strain config) at epochs 26, 28 and 37; the selected epoch-38 checkpoint logged
0.98 there. The standalone `check_pb_validity.py` curve peaks at 0.905 (epoch 8) and was
never run on the selected checkpoint. The unsteered arms of the runs that produced the
multiMFM number give 87.5% (n=64) and 88.0% (n=4,096x3) under the GEOM PB configuration.

## Hyperparameter tables

| file | source |
|---|---|
| `table19_geom_hparams.json` | `checkpoints/geom/teacher/hydra/config.yaml` (resolved Hydra config of the run) + `data_stats` in the teacher checkpoint |
| `table20_geom_hparams.json` | `args` / `student_cfg` stored in `checkpoints/geom/mfm_student/student_step_10000.pt`, mirrored in `..._args.json` |
| `paper_values.json` | transcribed from the PDF (Tables 3, 19, 20, 21 and App. F.8/F.9 text) |

## Reproducing

CPU, from `results/`: `notebooks/08_geom_fingerprint.ipynb` with `RERUN = False`.
One GPU: `RERUN = True` reruns the paper configuration at n=16, or
`scripts/geom/steer_table3.sbatch` for the n=64 grid / the 4,096x3 evaluation.
