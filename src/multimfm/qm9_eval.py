"""Per-molecule PoseBusters validity and SMILES for generated QM9 molecules.

Shared by ``steer_search`` (look-ahead selection and evaluation), ``property_targets``
(per-sample PB flags) and ``baselines``. Validity is the criterion used for every
QM9 PB number in the paper: a molecule is valid when RDKit conversion succeeds and
it passes every check of ``posebusters_no_strain.yaml`` (``~results.isin([False])
.any(axis=1)``); conversion failures count as invalid. Uniqueness follows
``train_base_flow.compute_pb_summary``: canonical SMILES of the converted molecules,
``U = n_unique / n_SMILES``, ``U_all = n_unique / N``.

``PBPool`` runs PoseBusters in forked worker processes, each of which builds its
PoseBusters object once (the original ``steer_search`` rebuilt it per chunk).
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from tensordict import TensorDict

from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com

_WORKER_PB = None


def _init_worker(config_path: str) -> None:
    global _WORKER_PB
    import yaml
    from posebusters import PoseBusters

    from multimfm.posebusters_compat import install_posebusters_compat

    install_posebusters_compat()
    with open(config_path, encoding="utf-8") as fh:
        _WORKER_PB = PoseBusters(config=yaml.safe_load(fh))


def _bust(mols: list) -> list[bool]:
    """Validity of each entry (RDKit mol or None). Runs in a worker (or inline)."""
    from rdkit import Chem

    keep, pb_mols = [], []
    for j, m in enumerate(mols):
        if m is None:
            continue
        try:
            mm = Chem.Mol(m)
        except Exception:
            continue
        if mm is None or mm.GetNumAtoms() == 0:
            continue
        pb_mols.append(mm)
        keep.append(j)
    out = [False] * len(mols)
    if pb_mols:
        try:
            res = _WORKER_PB.bust(mol_pred=pb_mols)
            passes = ~res.isin([False]).any(axis=1)
            for j, ok in zip(keep, passes.tolist()):
                out[j] = bool(ok)
        except RuntimeError:
            pass
    return out


class PBPool:
    """PoseBusters validity over a list of molecules, spread over worker processes."""

    def __init__(self, config_path: Path | str, workers: int = 8) -> None:
        self.config_path = str(config_path)
        self.workers = max(1, int(workers))
        self.n_checked = 0
        self._ex = None
        if self.workers > 1:
            try:
                self._ex = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=mp.get_context("fork"),
                    initializer=_init_worker,
                    initargs=(self.config_path,),
                )
            except Exception as exc:  # pragma: no cover - fallback
                print(f"[pb] could not start worker pool ({exc}); running inline", flush=True)
                self._ex = None
        if self._ex is None:
            _init_worker(self.config_path)

    def valid(self, mols: list) -> list[bool]:
        self.n_checked += len(mols)
        if not mols:
            return []
        if self._ex is None or len(mols) < 2 * self.workers:
            if self._ex is None:
                return _bust(mols)
            return list(self._ex.submit(_bust, mols).result())
        W = self.workers
        chunks = [mols[k::W] for k in range(W)]
        results = list(self._ex.map(_bust, chunks))
        out = [False] * len(mols)
        for k, res in enumerate(results):
            for j, ok in zip(range(k, len(mols), W), res):
                out[j] = ok
        return out

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_state(coords: torch.Tensor, atomics: torch.Tensor, mask: torch.Tensor) -> TensorDict:
    return TensorDict(
        {"coords": mask_and_zero_com(coords, mask),
         "atomics": apply_mask(atomics, mask),
         "padding_mask": mask},
        batch_size=coords.shape[0],
    ).to(coords.device)


def decode(state: TensorDict, converter, *, sanitize: bool = True) -> list:
    """RDKit molecules (None where conversion fails) for every row of ``state``."""
    return list(converter.from_batch(state.detach().cpu(), sanitize=sanitize, rescale_coords=True))


def smiles_of(mols: list) -> list[str | None]:
    from rdkit import Chem

    out = []
    for m in mols:
        if m is None:
            out.append(None)
            continue
        try:
            out.append(Chem.MolToSmiles(m))
        except Exception:
            out.append(None)
    return out


def molecule_records(state: TensorDict, converter, pool: PBPool, *, sanitize: bool = True) -> dict:
    """Per-molecule PB flags + SMILES and the aggregate summary (``pb_summary_for_state`` keys)."""
    mols = decode(state, converter, sanitize=sanitize)
    valid = pool.valid(mols)
    smiles = smiles_of(mols)
    n = len(mols)
    got = [s for s in smiles if s is not None]
    n_conv = sum(m is not None for m in mols)
    summary = {
        "num_generated": n,
        "num_converted": n_conv,
        "conversion_rate": n_conv / max(1, n),
        "pb_valid": sum(valid) / max(1, n),
        "num_smiles": len(got),
        "num_unique_smiles": len(set(got)),
        "unique_smiles_rate": len(set(got)) / max(1, len(got)),
        "unique_smiles_rate_total": len(set(got)) / max(1, n),
    }
    return {"valid": valid, "smiles": smiles, "summary": summary}
