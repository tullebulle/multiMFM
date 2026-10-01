"""Prepare a GEOM-Drugs structure-fingerprint dataset for the TFG structure head.

Streams molecules from the tabasco GEOM-Drugs LMDB (heavy-atom RDKit mols),
computes for each molecule:
    * mean-centered Angstrom coordinates,
    * atom-type indices into ``GEOM_ATOM_NAMES``,
    * the 1024-bit OpenBabel FP2 fingerprint (packed as 32 uint32 words),
and writes them to a compact prepared LMDB under TFG's ``storage/datasets``.
The prepared LMDB (plus a ``{split}_meta.json``) is the TFG-native, "copied
over" form of the dataset; the large raw tabasco data is not duplicated.

Example
-------
    python -m dataloader.datasets.build_geom_fingerprints \
        --source ../geom-drug-tabasco/data/lmdb_geom \
        --out storage/datasets/geom_fp \
        --split val --num-workers 16

The fingerprint computation is the bottleneck (OpenBabel per molecule); it
parallelises across ``--num-workers`` processes. Use ``--limit`` for smoke tests.
"""

import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from dataloader.datasets.geom_fingerprint import (
    GEOM_ATOM_NAMES,
    GEOM_INDEX_TO_ATOMIC_NUM,
    compute_fp_words,
    mol_to_arrays,
)

# per-worker globals
_SRC_PATH = None
_DB = None


def _init_worker(src_path):
    global _SRC_PATH, _DB
    _SRC_PATH = src_path
    _DB = None


def _get_db():
    global _DB
    if _DB is None:
        import lmdb
        _DB = lmdb.open(
            _SRC_PATH, subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False, map_size=50 * (1024 ** 3),
        )
    return _DB


def _process_key(key):
    """Read one source molecule and return a compact prepared record (or None)."""
    try:
        db = _get_db()
        mol = pickle.loads(db.begin().get(key))["molecule"]

        from rdkit import Chem
        try:
            mol = Chem.RemoveAllHs(mol)  # data is already heavy-atom; belt & braces
        except Exception:
            pass

        arrays = mol_to_arrays(mol)
        if arrays is None:
            return None
        coords, atom_types = arrays
        if coords.shape[0] == 0:
            return None

        # mean-center (unweighted) to match the flow model's coordinate convention
        coords = coords - coords.mean(axis=0, keepdims=True)

        atomic_nums = [GEOM_INDEX_TO_ATOMIC_NUM[int(t)] for t in atom_types]
        fp_words = compute_fp_words(coords, atomic_nums)

        rec = {
            "coords": coords.astype(np.float32),
            "atom_types": atom_types.astype(np.uint8),
            "fp_words": np.asarray(fp_words, dtype="<u4"),
        }
        return pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="../geom-drug-tabasco/data/lmdb_geom",
                    help="dir containing tabasco {split}.lmdb files")
    ap.add_argument("--out", default="storage/datasets/geom_fp",
                    help="output dir for prepared {split}.lmdb + meta")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N source molecules (smoke test)")
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    import lmdb
    from multiprocessing import Pool

    src_path = os.path.join(args.source, f"{args.split}.lmdb")
    assert os.path.exists(src_path), f"source lmdb not found: {src_path}"

    # enumerate source keys
    src = lmdb.open(src_path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False, map_size=50 * (1024 ** 3))
    with src.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    src.close()
    if args.limit is not None:
        keys = keys[: args.limit]
    print(f"[{args.split}] source molecules: {len(keys)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.split}.lmdb")
    if os.path.exists(out_path):
        os.remove(out_path)
        lock = out_path + "-lock"
        if os.path.exists(lock):
            os.remove(lock)

    out = lmdb.open(out_path, subdir=False, map_size=100 * (1024 ** 3),
                    create=True, readonly=False)

    written = 0
    failed = 0
    max_num_atoms = 0
    txn = out.begin(write=True)

    with Pool(args.num_workers, initializer=_init_worker, initargs=(src_path,)) as pool:
        for payload in tqdm(
            pool.imap_unordered(_process_key, keys, chunksize=args.chunksize),
            total=len(keys), desc=f"fp[{args.split}]",
        ):
            if payload is None:
                failed += 1
                continue
            rec = pickle.loads(payload)
            max_num_atoms = max(max_num_atoms, int(rec["atom_types"].shape[0]))
            txn.put(str(written).encode(), payload)
            written += 1
            if written % 50000 == 0:
                txn.commit()
                txn = out.begin(write=True)

    txn.commit()
    out.sync()
    out.close()

    meta = {
        "split": args.split,
        "num_molecules": written,
        "num_failed": failed,
        "max_num_atoms": max_num_atoms,
        "atom_names": GEOM_ATOM_NAMES,
        "fp_nbits": 1024,
        "source": src_path,
        "coords_space": "mean-centered Angstrom (divide by coord_normalizer at load)",
    }
    with open(os.path.join(args.out, f"{args.split}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{args.split}] wrote {written} records "
          f"({failed} failed), max_num_atoms={max_num_atoms}", flush=True)
    print(f"[{args.split}] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
