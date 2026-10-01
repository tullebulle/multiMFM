# GEOM-Drugs experiment: provenance

Paper items covered: **Table 3** (structural-fingerprint guidance), the GEOM-Drugs
column of **Tables 19 and 20**, the GEOM-Drugs row of **Table 21**, and **Appendix F.8**
(inference settings: 10 flow-map inner steps, 4 posterior samples, 100-step outer ODE).

* Results and their run provenance: [`results/geom/PROVENANCE.md`](../../../results/geom/PROVENANCE.md)
* Checkpoints, including the student lineage: [`checkpoints.md`](checkpoints.md)
* Data (what ships, what to rebuild, the 71-vs-82 target bug): [`data.md`](data.md)
* Notebook: `notebooks/08_geom_fingerprint.ipynb`

## Where the code came from

Source repositories (read-only, backed up to
`/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/`):
`SRC = /n/netscratch/kozinsky_lab/Lab/uunneberg/repos/geom-drug-tabasco` and
`TFG = /n/netscratch/kozinsky_lab/Lab/uunneberg/repos/TFG-Flow`.

| repo path | origin |
|---|---|
| `src/multimfm/geom/steering.py`, `mfm_steering.py`, `glass_steering.py` | `SRC/glass/{mfm,glass}_fingerprint_steering.py` (one shared driver instead of two copies) |
| `src/multimfm/geom/fingerprint.py` | the head/target/FP2/scoring helpers of `SRC/glass/glass_fingerprint_steering.py` |
| `src/multimfm/geom/reward.py` | the fingerprint-reward hunks that `SRC/glass/glass_guided_sampling_experiment.py` carried on top of the QM9 driver |
| `src/multimfm/geom/teacher.py` | `SRC/metaflow/geom_teacher.py` (+ the duplicate `load_teacher` in the steering script) |
| `src/multimfm/geom/train_student.py` | `SRC/metaflow/mfm_student_train.py` |
| `src/multimfm/geom/check_pb_validity.py` | `SRC/validation_teacher/check_pb_validity.py` |
| `src/multimfm/geom/train_base.py`, `configs/geom/` | `SRC/src/train.py`, `SRC/configs/` at commit `e227cbf` (HEAD) |
| `configs/geom/posebusters_geom.yaml` | `SRC/pb_config.yaml` (the GEOM experimental PB configuration) |
| `third_party/tfg_flow/dataloader/datasets/geom_fingerprint.py`, `build_geom_fingerprints.py` | uncommitted files in `TFG` |
| `third_party/tfg_flow/train_structure_head.py` | uncommitted file in `TFG` |
| `third_party/tfg_flow/diffusion/predictor.py` | `TFG`'s uncommitted `num_atom_types` patch (defaults to the QM9 vocabulary, so the QM9 regressors still load) |
| `scripts/geom/*.sbatch` | `SRC/glass/run_*.sbatch`, `SRC/metaflow/mfm_student_stage*_ddp.sbatch`, `SRC/validation_teacher/eval_pb_validity.sbatch`, `TFG/scripts/geom_drug/*` |

`src/tabasco/` in this repo now equals `SRC` at HEAD exactly: the two fixes the plan
lists were applied on 2026-09-29 (`callbacks/posebusters.py` logs `val/pb_valid` in
addition to the back-compatible `val/pb_intersection`; `data/components/lmdb_base.py`
reopens the LMDB handle after a fork, so DataLoader workers no longer segfault). Both are
additive and the QM9 code paths are unaffected. The uncommitted September changes in
`SRC/src/tabasco` (explicit-hydrogen work: `time_coupling`, `beta_shift`,
`CenteredMetricInterpolant`, ...) are deliberately **not** ported: they postdate the paper
runs, and `configs/geom/` is the HEAD config tree that matches this `src/tabasco`.

## The as-described Table 3 rerun (authors' decision 1)

Added 2026-09-29, after the authors decided to rerun Table 3 exactly as the paper describes
it. Nothing about the steering algorithm changed; what was added is the machinery the
"as described" reading needs:

* `multimfm.geom.fingerprint.holdout_target_indices` and the `exclude=` / `indices=`
  arguments of `sample_targets`, plus the `--holdout-n`, `--holdout-seed` and
  `--tune-on-holdout` flags of both steering CLIs. They make the (lambda, mu) selection set
  disjoint from the evaluation targets by construction; every result JSON records which
  mode it ran in and how many drawn targets the exclusion skipped.
* `scripts/geom/tune_lambda_mu.sbatch` (held-out grid, 15 points) and
  `scripts/geom/table3_asdescribed.sbatch` (3 seeds x 4 shards x 1,024 molecules), both
  resumable job arrays constrained to `cc8.0|cc9.0` GPUs, with
  `scripts/geom/collect_asdescribed.py` + `scripts/geom/collect_table3_asdescribed.sh` as
  the single collection command (it also re-executes notebook 08).
* `results/geom/tfgflow_reference.json`: what TFG-Flow reports and releases, checked against
  its paper and repository, because Table 3 cites it for two reference values and for the
  guide predictor.
* `multimfm.geom.diagnostics grad-sign`: a directional check that the estimated value
  gradient really ascends the guide's own reward, which is the first thing to run when a
  steering run *lowers* the similarity. It needs a compute node (the login node OOM-kills
  it), and has not been run yet.
* `results/geom/table3_n64_fragility.json`: a bootstrap of the existing 4,096x3 per-molecule
  gains showing how often a 64-molecule estimate reaches the submitted +0.0199.

The evaluation deliberately keeps everything the paper does state: 4,096 molecules per seed,
10 flow-map inner steps, 4 posterior samples, a 100-step outer ODE, targets from the val
split of the prepared fingerprint LMDB, Tanimoto of the directly counted FP2 against the
target, and PoseBusters validity of both the guided and the unguided arm (which share noise
and targets, so the unguided arm is Table 21's "Unsteered" column on the same molecules).

Two things the paper does not state and therefore could not be reproduced literally:
the guidance strengths (selected on held-out targets; lambda=1, the literal reading of
`R(G)=exp(-||E(G)-c||^2)`, is one grid point) and the guidance coefficient's time
dependence -- the paper's ODE carries `sigma_t^2/2 = (1-t)/t` in front of the value
gradient, while the implementation that produced every recorded number multiplies it by a
constant `mu`. The code was left as it is and the discrepancy reported.

## Differences from the source code

1. **One steering driver.** The GLASS and multiMFM drivers were near-identical copies;
   they are now `multimfm.geom.steering` plus two thin CLIs that keep each script's own
   defaults. The GLASS driver gained the (post-hoc) PoseBusters option, off by default so
   it still matches its recorded runs.
2. **The fingerprint reward is installed into the shared engine** rather than duplicated:
   `multimfm.geom.reward.install_fingerprint_reward()` adds the `"fingerprint"` reward and
   the 2-D target handling to `multimfm.glass_guidance` (which is the QM9 port of the same
   file and has no fingerprint support). It is idempotent and becomes a no-op if those
   hunks are ever merged into the shared module. See the note in `reward.py`.
3. **The 71-vs-82 target bug is fixed** (`selection="filter"`, old behaviour under
   `--target-selection legacy`). Details and the effect on target sampling: [`data.md`](data.md).
4. **Repo-relative defaults** everywhere (`multimfm.geom.paths`); no `sys.path`
   bootstrapping except the one documented hook that puts the vendored TFG-Flow tree on
   `sys.path` (`paths.ensure_tfg_importable`, the same mechanism `multimfm.qm9_data` uses).
5. **Hydra config drift.** `configs/geom/experiment/geom_dfm.yaml` sets `batch_size: 512`,
   the value the released run used; the file committed in `SRC` said 256. The validation
   callbacks are set to the released run's 100 samples / 100 steps rather than the later
   500 / 200. Both are documented in the file itself.
6. **No personal identifiers**: the wandb entity is gone (offline by default) and
   checkpoints are written to `outputs/geom/...` instead of `checkpoints/<SLURM_JOB_ID>/`
   so `checkpoints/geom/` only ever holds released artifacts.

## Environment

Everything runs in the single `multimfm` environment (`source scripts/env.sh`); the GEOM
work originally used a separate `tabasco` env, which the netscratch purge destroyed. No
package had to be added: `lmdb`, `openbabel` (FP2 fingerprints), `hydra-core`, `omegaconf`,
`tensordict`, `posebusters`, `einops` and `wandb` are all present. The 4,096x3 rerun of
2026-09-22 already ran under `multimfm`.
