# GEOM-Drugs data

## What ships with the repo

| repo path | size | what | needed for |
|---|---|---|---|
| `data/geom/geom_fp/val.lmdb` | 173 MB | prepared FP2 fingerprint LMDB, val split, 146,401 molecules | Table 3 steering targets (all runs draw targets from `val`) |
| `data/geom/geom_fp/val_meta.json` | <1 KB | its metadata (`max_num_atoms` 82, atom vocabulary) | same |

Both are tracked in `checkpoints/MANIFEST.json` (group `geom`); the recorded source is the
holylabs backup of `TFG-Flow/storage/datasets/geom_fp/`.

Each LMDB record is a pickled dict `{coords (n,3) float32 mean-centred Angstrom,
atom_types (n,) uint8 in [C,N,O,F,S,Cl,Br,I,*], fp_words (32,) uint32}`, read by
`third_party/tfg_flow/dataloader/datasets/geom_fingerprint.py:GEOMFingerprintDataset`,
which divides coordinates by 2.0 (the base model's normalizer) and unpacks the 1024-bit FP2.

## What does not ship, and how to get it

Only needed to *retrain* something; every result in the paper can be recomputed from
what ships plus the checkpoints.

### Processed GEOM-Drugs molecules (base model and MFM student training)

`processed_geom_{train,val,test}.pt` (~2.2 GB in total; lists of heavy-atom RDKit
molecules with one conformer each) are public on Hugging Face, dataset
[`carlosinator/tabasco-geom-drugs`](https://huggingface.co/datasets/carlosinator/tabasco-geom-drugs),
the release of the TABASCO paper. Place them in `data/geom/`:

```bash
huggingface-cli download carlosinator/tabasco-geom-drugs --repo-type dataset \
    --local-dir data/geom
# expected: data/geom/processed_geom_train.pt, processed_geom_val.pt, processed_geom_test.pt
```

(`scripts/env.sh` sets `HF_HUB_OFFLINE=1` for the bundled QM9 data; unset it, or run
the download outside that environment.) The teacher's `data_stats` list 1,141,716 training
SMILES; the fingerprint train LMDB built from the same split has 1,172,182 records (the
paper's "~1.17M conformers").

* **Base model** (`scripts/geom/train_base.sbatch`): the Lightning datamodule builds its own
  LMDB from the `.pt` files into `data/geom/lmdb_geom/` on first use.
* **MFM student** (`scripts/geom/train_student.sbatch`): reads
  `data/geom/processed_geom_train.pt` directly and keeps the first `--limit-data 150000`
  molecules (the paper's student was distilled on those 150k).

### Fingerprint LMDB train/test splits (guide-head training)

`train.lmdb` (1.3 GB, 1,172,182 molecules) and `test.lmdb` (174 MB, 146,791 molecules) are
rebuilt from the tabasco LMDB the base-model datamodule creates (`data/geom/lmdb_geom/`,
so run one base-training job, or a smoke run of it, first). OpenBabel perceives bonds from
the 3-D coordinates and computes FP2; this is CPU-bound and parallel:

```bash
source scripts/env.sh
cd third_party/tfg_flow
for split in val test train; do
  python -m dataloader.datasets.build_geom_fingerprints \
      --source ../../data/geom/lmdb_geom --out ../../data/geom/geom_fp \
      --split "$split" --num-workers 32
done
```

(`val` is only needed if you want to rebuild the shipped file; the original build log is
`TFG-Flow/storage/logs/build_fp.log`, 2026-07-26, zero failures in every split.)
Note that the build draws keys with `imap_unordered`, so record *order* -- and therefore
which target a given seed selects -- can differ from the shipped `val.lmdb`; use the
shipped file to reproduce the paper's targets.

## The 71-vs-82 target bug

The base model supports at most 71 heavy atoms (`data_stats["max_num_atoms"]`; the train
split's maximum is 71), but the val and test fingerprint splits contain molecules of up to
82 atoms. The original `sample_targets` opened the val dataset with `max_len=71`, so any
seed whose random permutation put an over-long molecule among the first `n` indices raised

```
RuntimeError: The expanded size of the tensor (71) must match the existing size (72)
```

That is what killed seed 21 of the 4,096x3 rerun (Slurm 47833756), leaving replicate 2
with 3,072 instead of 4,096 molecules.

**Size of the effect.** 30 of the 146,401 val molecules (0.020%) have more than 71 heavy
atoms (mean 25.1, maximum 82). A 1,024-molecule shard contains one with probability ~19%;
a 4,096-molecule evaluation contains ~0.8 on average.

**Fix** (`multimfm.geom.fingerprint.sample_targets`, default `selection="filter"`): the
dataset is opened at its own padded width and the *same* seeded permutation is walked in
order, skipping molecules with more than 71 heavy atoms until `n` targets are collected.
Consequences:

* any draw that did not hit an over-long molecule -- including the paper's n=64 seed-0 run
  and all 11 surviving shards of the 4,096x3 rerun -- gets bit-identical targets, so the
  recorded numbers are unaffected;
* a draw that used to crash now replaces each over-long target with the next eligible one
  in the permutation (seed 21 at n=1,024: one molecule skipped);
* the target distribution becomes "val molecules of at most 71 heavy atoms", which is
  what the paper describes ("molecules up to 71 heavy atoms") and what the generator can
  produce.

The number skipped is written to every result JSON (`targets_skipped`). The original
behaviour, crash included, stays available as `--target-selection legacy`.
