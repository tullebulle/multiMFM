# GEOM-Drugs checkpoints

Sizes, sha256 and the recorded `source` for every file are in the tracked
`checkpoints/MANIFEST.json` (group `geom`); verify or re-fetch with

```bash
python scripts/manifest.py verify --group geom [--fetch-missing]
```

The recorded sources are in the holylabs backup
`/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/`, which is not
purge-prone. `SRC` below is the source repo
`/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/geom-drug-tabasco`.

| repo path | size | original location |
|---|---|---|
| `checkpoints/geom/teacher/epoch=38-step=89310.ckpt` | 297 MB | `SRC/validation_teacher/big_curve/ckpts/` |
| `checkpoints/geom/teacher/hydra/{config,overrides,hydra}.yaml` | 12 KB | `SRC/outputs/2026-07-26/01-03-35/.hydra/` |
| `checkpoints/geom/mfm_student/student_step_10000.pt` | 121 MB | `SRC/outputs/mfm_student_geom_stage2_ddp/stage2_20260727-222154/` |
| `checkpoints/geom/mfm_student/student_step_10000_args.json` | 5 KB | extracted from the checkpoint |
| `checkpoints/geom/structure_head/{best.pth,best.txt}` | 4.3 MB | `TFG-Flow/storage/ckpts/geom_structure_head_ddp_cont2/` |
| `checkpoints/geom/structure_head/train_args.json` | 3 KB | reconstructed from the run logs + sbatch |

## Base model (teacher)

Lightning checkpoint: `state_dict` under `model.*`, `data_stats`, and the EMA shadow
weights as a flat list in `optimizer_states[i]["ema"]` (in `model.parameters()` order).
`multimfm.geom.teacher.load_geom_teacher` rebuilds the `FlowMatchingModel`, loads the
weights and swaps in EMA by default -- the weights every paper run used.

Hydra run: `SRC/outputs/2026-07-26/01-03-35`, wandb run `70pueao1`, project `geom-drugs`,
Slurm job 35203410, started 2026-07-26 01:03, checkpoint written 17:48 the same day.
Overrides: `experiment=geom_dfm trainer=gpu datamodule.lmdb_dir=/tmp/geom_dfm_35203410/lmdb_geom`.
The resolved config shipped next to the checkpoint is the authority for the GEOM column of
Table 19 (`results/geom/table19_geom_hparams.json` is that config, read out field by field).
Architecture inferred from the weights: hidden 256, 16 layers, 8 heads, 14,795,529
parameters; `data_stats`: `atom_dim` 9, `max_num_atoms` 71, `coordinate_normalizer` 2.0,
`atom_names` backfilled from `tabasco.chem.constants.ATOM_NAMES`.

Its PoseBusters validity during training is in `results/geom/teacher_training_pb_validity.csv`
(39 epochs, 100 molecules each); the standalone evaluations of other checkpoints of the same
run are in `results/geom/teacher_pb_curve.csv`. **Neither contains a 4,096-molecule number,
so Table 21's 92.0% has no matching record** -- see `results/geom/PROVENANCE.md`.

Rebuild: `sbatch scripts/geom/train_base.sbatch` (1 GPU, ~17 h on an H200).

## multiMFM student: lineage

Four legs, each warm-starting the next; every leg's arguments are stored inside its own
checkpoint (the table is read out of them). All four legs distil from the same teacher
above, on `limit_data=150000` of the 1,141,716 processed molecules -- i.e. the GEOM MFM was
distilled on **150k of the ~1.17M** molecules, which the paper does not say.

| leg | run dir (`SRC/outputs/...`) | stage | GPUs x batch | lr | steps run / horizon | key settings |
|---|---|---|---|---|---|---|
| 1a | `mfm_student_geom_ddp/stage1_20260727-132650` | 1 | 3 x 256 = 768 | 2e-4 | 20,000 / 40,000 | from teacher (`--init-from-teacher`), endpoint-space loss, EMA 0.999, `t_cond_power` 1.0 |
| 1b | `mfm_student_geom_ddp/stage1_20260727-144928` | 1 | 3 x 256 = 768 | 1e-3 | 40,000 / 40,000 | resumes 1a, same objective |
| 2a | `mfm_student_geom_stage2_ddp/stage2_20260727-163603` | 2 | 3 x 128 = 384 | 1e-4 | 20,000 / 30,000 | resumes 1b; +ESD, `max_jump` 0.25, `anneal_frac` 1.0, `diag_weight` 2, `esd_target_clip` 10, EMA 0.995, `t_cond_power` 2.0, `t_cond_zero_rate` 0.1 |
| **2b (released)** | `mfm_student_geom_stage2_ddp/stage2_20260727-222154` | 2 | 4 x 96 = 384 | 1e-4 | **10,000** / 30,000 | resumes 2a; `max_jump` **0.15**, `anneal_frac` **0.2**, `esd_delta_clip` **8**, `esd_target_clip` 10, `diag_weight` 2, EMA 0.995. Slurm job 35633958, wandb `mfm-geom-stage2-ddp4-big-resume58-maxjump0.15-dclip8-ckpt2k` |

Table 20's GEOM column describes leg 2b only ("both students were initialized from
previously trained MFM checkpoints"). It omits `max_jump` 0.15, `anneal_frac` 0.2, the two
ESD clips (8 / 10) and `diag_weight` 2; all of them are in
`results/geom/table20_geom_hparams.json` under `omitted_from_table20`.

**Intermediate checkpoints are not copied into the repo** (121 MB each, 363 MB total);
they are only needed to restart mid-chain, and a full rebuild reruns the chain from the
teacher. They live in the backup at:

| leg | backup path (under `paper_backup_dmfm_20260929/geom-drug-tabasco/outputs/`) | size |
|---|---|---|
| 1a | `mfm_student_geom_ddp/stage1_20260727-132650/student_step_20000.pt` | 120,754,596 B |
| 1b | `mfm_student_geom_ddp/stage1_20260727-144928/student_step_40000.pt` | 120,754,724 B |
| 2a | `mfm_student_geom_stage2_ddp/stage2_20260727-163603/student_step_20000.pt` | 120,754,724 B |

The step-20,000 checkpoint of leg 2a is also the student used by the *earlier* 4,096-molecule
evaluation (`results/geom/n4096_first_run_shards.csv`) and by the earlier n=64 grid
(`results/geom/mfm_steering_grid_n64_older_student.csv`); the leg-2a run diverged after
step 20,000, which is why leg 2b exists.

Rebuild: `LEG=1a sbatch --gres=gpu:3 scripts/geom/train_student.sbatch`, then 1b, 2a, 2b
with `INIT_CKPT` pointing at the previous leg's checkpoint.

## Fingerprint guide head

`checkpoints/geom/structure_head/best.pth` is a bare `state_dict` of TFG-Flow's
`diffusion.predictor.Predictor` with the GEOM atom vocabulary (h=128, 6 EGNN layers,
cls=64, e=128, 1024 outputs, `num_atom_types=9`), trained with BCE against the OpenBabel
FP2 label and selected on val Tanimoto. Best epoch 34, val Tanimoto 0.8523, val BCE 0.0686
(`best.txt`). Full four-leg record in `train_args.json`; the released leg is Slurm job
35548578 (3 GPUs x 256, lr 1e-3 cosine, warmup 500, 40 epochs, warm start from
`geom_structure_head_ddp_cont/best.pth`).

**Paper mismatch.** Section 3.2 says "Our E(G) is off-the-shelf from TFG-Flow". The
architecture and the FP2 label definition are TFG-Flow's, but these weights were trained by
the authors: TFG-Flow ships no GEOM-Drugs structure head (only the QM9 property
regressors in `storage/{guide,oracle}_clf_ckpt.zip`, which this repo uses for QM9).

Rebuild: `sbatch scripts/geom/train_structure_head.sbatch` -- needs the *train* split of
the fingerprint LMDB, which is not shipped (see [`data.md`](data.md)).
