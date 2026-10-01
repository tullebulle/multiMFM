"""GEOM-Drugs structure-fingerprint dataset for training the TFG structure head.

This module provides everything needed to train the small EGNN head
(`diffusion.predictor.Predictor`) to predict a molecule's 1024-bit structure
fingerprint from `(coords, atom_types, mask)` on GEOM-Drugs data.

Design decisions (see also the top-level answer in the build notes):

* **Atom vocabulary** follows the tabasco GEOM-Drugs teacher exactly:
  ``[C, N, O, F, S, Cl, Br, I, *]`` (heavy-atom only, ``*`` is the dummy/pad
  class at the last index). This is what the GEOM flow model emits, so the head
  lines up with it for downstream guidance. Note this differs from TFG's QM9
  vocabulary (`dataloader.constants.ATOMNAMES`), which lacks Br/I and carries H.

* **No hydrogens.** GEOM-Drugs here is heavy-atom only, matching both the flow
  model and TFG's QM9 heavy-atom setup.

* **Fingerprint label** is TFG's exact definition: OpenBabel FP2 (1024-bit,
  path-based), with bonds re-perceived from the 3-D coordinates
  (`ConnectTheDots` + `PerceiveBondOrders`). It is computed from *real-Angstrom*
  coordinates. We reproduce it with the OpenBabel v3 API because TFG's original
  `dataloader/fingerprints.py` imports the (uninstalled) legacy ``pybel`` and
  ``ase`` packages.

* **Coordinate space.** The prepared dataset stores mean-centered Angstrom
  coordinates. The dataset divides them by ``coord_normalizer`` (default 2.0,
  matching tabasco's ``dataset_normalizer``) at read time so the head is trained
  in the same coordinate space the flow model operates in.
"""

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Atom vocabulary (must match the tabasco GEOM teacher: src/tabasco/chem/constants.py)
# ---------------------------------------------------------------------------
GEOM_ATOM_NAMES = ["C", "N", "O", "F", "S", "Cl", "Br", "I", "*"]
GEOM_ATOM_TO_INDEX = {name: i for i, name in enumerate(GEOM_ATOM_NAMES)}
GEOM_DUMMY_INDEX = GEOM_ATOM_TO_INDEX["*"]  # 8; also used for padding
NUM_GEOM_ATOM_TYPES = len(GEOM_ATOM_NAMES)  # 9

# atom-type index -> atomic number (for OpenBabel). '*' falls back to carbon.
GEOM_INDEX_TO_ATOMIC_NUM = {
    0: 6, 1: 7, 2: 8, 3: 9, 4: 16, 5: 17, 6: 35, 7: 53, 8: 6,
}
# atomic number -> mass, used only for the (fingerprint-invariant) COM centering
# that mirrors TFG's original procedure. Values from ase.data.atomic_masses.
_ATOMIC_MASS = {
    6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    16: 32.06, 17: 35.45, 35: 79.904, 53: 126.904,
}

FP_NBITS = 1024
FP_NWORDS = FP_NBITS // 32  # 32 uint32 words


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------
def compute_fp_words(coords, atomic_nums):
    """Return the OpenBabel FP2 fingerprint as a length-32 uint32 array.

    Faithful reproduction of TFG's ``compute_fingerprint`` (COM-centering and
    distance sorting are kept, though they do not change the graph-based FP2
    bits), using the OpenBabel v3 python API.

    Args:
        coords: ``(n, 3)`` array of Angstrom coordinates.
        atomic_nums: length-``n`` iterable of atomic numbers.

    Returns:
        ``np.ndarray`` of dtype ``<u4`` and shape ``(32,)``.
    """
    from openbabel import openbabel as ob
    from openbabel import pybel

    coords = np.asarray(coords, dtype=np.float64)
    atomic_nums = np.asarray(list(atomic_nums), dtype=np.int64)

    # center of mass + sort by distance to COM (mirrors TFG; FP2 is invariant)
    masses = np.array([_ATOMIC_MASS.get(int(z), 2.0 * int(z)) for z in atomic_nums])
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    coords = coords - com[None, :]
    order = np.argsort(np.sqrt((coords ** 2).sum(1)))
    coords = coords[order]
    atomic_nums = atomic_nums[order]

    obmol = ob.OBMol()
    obmol.BeginModify()
    for (x, y, z), n in zip(coords, atomic_nums):
        a = obmol.NewAtom()
        a.SetAtomicNum(int(n))
        a.SetVector(float(x), float(y), float(z))
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    obmol.EndModify()

    fp = pybel.Molecule(obmol).calcfp()  # default FP2, 1024-bit
    return np.array(fp.fp, dtype="<u4")


def words_to_bits(words):
    """Unpack a length-32 uint32 fingerprint into a length-1024 float tensor."""
    words = np.asarray(words, dtype="<u4")
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")
    return torch.from_numpy(bits.astype(np.float32))


def tanimoto_similarity(A, B):
    """Row-wise Tanimoto similarity between two binary fingerprint batches.

    Identical to TFG's ``dataloader.fingerprints.tanimoto_similarity``.
    """
    intersection = torch.sum(A * B, dim=1)
    sum_A = torch.sum(A, dim=1)
    sum_B = torch.sum(B, dim=1)
    return intersection / (sum_A + sum_B - intersection + 1e-12)


def mol_to_arrays(mol):
    """Extract ``(coords_A, atom_type_idx)`` from a heavy-atom RDKit mol.

    Returns ``None`` if the molecule has no conformer or contains an element
    outside the GEOM vocabulary that is not mappable to the dummy class.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    idx = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        idx.append(GEOM_ATOM_TO_INDEX.get(sym, GEOM_DUMMY_INDEX))
    return coords, np.asarray(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GEOMFingerprintDataset(Dataset):
    """Reads a prepared fingerprint LMDB produced by ``build_geom_fingerprints.py``.

    Each LMDB record is a pickled dict with:
        * ``coords``:     ``(n, 3)`` float32, mean-centered Angstrom
        * ``atom_types``: ``(n,)`` uint8, indices into ``GEOM_ATOM_NAMES``
        * ``fp_words``:   ``(32,)`` uint32, packed FP2 fingerprint

    ``__getitem__`` returns the TFG feature layout expected by ``Predictor``:
        ``coors`` (max_len, 3), ``atom_types`` (max_len,), ``mask`` (max_len,),
        ``num_atoms`` (), ``structure`` (1024,).
    """

    def __init__(self, lmdb_dir, split="train", max_len=None, coord_normalizer=2.0,
                 limit=None):
        self.lmdb_dir = lmdb_dir
        self.split = split
        self.coord_normalizer = coord_normalizer
        self.lmdb_path = os.path.join(lmdb_dir, f"{split}.lmdb")

        meta = self._load_meta()
        self.max_len = int(max_len) if max_len is not None else int(meta["max_num_atoms"])
        assert meta["atom_names"] == GEOM_ATOM_NAMES, (
            f"prepared dataset vocab {meta['atom_names']} != {GEOM_ATOM_NAMES}"
        )

        self.db = None
        self._db_pid = None
        self.keys = None
        self.limit = limit
        # populate keys once (cheap: just the key list)
        self._connect()
        self._maybe_limit()

    # -- lmdb plumbing (fork-safe, mirrors tabasco's BaseLMDBDataset) --------
    def _load_meta(self):
        with open(os.path.join(self.lmdb_dir, f"{self.split}_meta.json")) as f:
            return json.load(f)

    def _connect(self):
        import lmdb
        self.db = lmdb.open(
            self.lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=False, meminit=False, map_size=50 * (1024 ** 3),
        )
        self._db_pid = os.getpid()
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _maybe_limit(self):
        if self.limit is not None:
            self.keys = self.keys[: self.limit]

    def _ensure(self):
        if self.db is None or self._db_pid != os.getpid():
            self.db = None
            self._connect()
            self._maybe_limit()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        self._ensure()
        rec = pickle.loads(self.db.begin().get(self.keys[i]))
        coords = torch.from_numpy(np.asarray(rec["coords"], dtype=np.float32))
        atom_types = torch.from_numpy(np.asarray(rec["atom_types"], dtype=np.int64))
        structure = words_to_bits(rec["fp_words"])

        n = coords.shape[0]
        L = self.max_len

        coors = torch.zeros(L, 3, dtype=torch.float32)
        coors[:n] = coords / self.coord_normalizer

        types = torch.full((L,), GEOM_DUMMY_INDEX, dtype=torch.long)
        types[:n] = atom_types

        mask = torch.zeros(L, dtype=torch.long)
        mask[:n] = 1

        return {
            "coors": coors,
            "atom_types": types,
            "mask": mask,
            "num_atoms": torch.tensor(n, dtype=torch.long),
            "structure": structure,
        }
