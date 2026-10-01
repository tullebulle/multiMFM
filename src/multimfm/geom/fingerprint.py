"""Structure-fingerprint helpers shared by the two GEOM steering drivers.

Contains everything that is not the steering loop itself: the TFG-Flow structure
head (the guide predictor ``E``), target-fingerprint sampling from the prepared
``geom_fp`` LMDB, OpenBabel FP2 fingerprints of generated molecules, and the
Tanimoto scoring reported in Table 3.

Ported from ``geom-drug-tabasco/glass/glass_fingerprint_steering.py``; the only
behavioural change is the target-selection fix documented on
:func:`sample_targets`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from multimfm.geom.paths import DEFAULT_GEOM_FP_DIR, DEFAULT_STRUCTURE_HEAD, ensure_tfg_importable

ensure_tfg_importable()

from dataloader.datasets.geom_fingerprint import (  # noqa: E402
    GEOM_ATOM_NAMES,
    GEOMFingerprintDataset,
    compute_fp_words,
    tanimoto_similarity,
    words_to_bits,
)
from diffusion.predictor import Predictor  # noqa: E402

__all__ = [
    "GEOM_ATOM_NAMES",
    "holdout_target_indices",
    "FP_NBITS",
    "TARGET_SELECTIONS",
    "load_structure_head",
    "sample_targets",
    "state_to_binary_fps",
    "head_binary_fps",
    "score",
    "tanimoto_similarity",
]

FP_NBITS = 1024

#: GEOM heavy-atom vocab -> atomic number (index 8 is the "*" dummy, dropped).
_ELEMENT_Z = {"C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17, "Br": 35, "I": 53}
_DUMMY_IDX = GEOM_ATOM_NAMES.index("*")

#: How :func:`sample_targets` handles targets larger than the generator supports.
TARGET_SELECTIONS = ("filter", "legacy")


def load_structure_head(
    path: str | Path = DEFAULT_STRUCTURE_HEAD,
    device: str | torch.device = "cpu",
    temperature: float = 0.25,
    *,
    max_len: int = 71,
):
    """Load the trained TFG-Flow fingerprint head used as the guide predictor.

    Architecture (h=128, 6 layers, cls=64, e=128, 1024 outputs, GEOM 9-atom
    vocabulary) matches ``third_party/tfg_flow/train_structure_head.py``'s
    defaults, i.e. the run that produced ``best.pth``.
    """
    head = Predictor(
        max_len=max_len,
        device=str(device),
        h_embed_size=128,
        num_layers=6,
        cls_embed_size=64,
        e_embed_size=128,
        class_num=FP_NBITS,
        num_atom_types=len(GEOM_ATOM_NAMES),
    ).to(device)
    state = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    head.load_state_dict(state)
    head.soft_atom_temperature = float(temperature)
    head.eval()
    for prm in head.parameters():
        prm.requires_grad_(False)
    return head


def sample_targets(
    geom_fp_dir: str | Path = DEFAULT_GEOM_FP_DIR,
    split: str = "val",
    n: int = 64,
    max_atoms: int = 71,
    seed: int = 0,
    *,
    selection: str = "filter",
    exclude: Iterable[int] | None = None,
    indices: Sequence[int] | None = None,
    return_info: bool = False,
):
    """Draw ``n`` target fingerprints from the prepared ``geom_fp`` LMDB.

    **The 71-vs-82 target bug.** The generator is trained on GEOM-Drugs
    molecules of at most 71 heavy atoms (``data_stats["max_num_atoms"]``, and the
    paper's "up to 71 heavy atoms"), but the ``geom_fp`` *val* and *test* splits
    contain molecules of up to 82 atoms. The original code opened the dataset
    with ``max_len=71`` and therefore raised ``RuntimeError: The expanded size of
    the tensor (71) must match the existing size (72)`` as soon as a drawn index
    pointed at an over-long molecule -- which is what killed seed 21 of the
    4,096 x 3 rerun.

    ``selection="filter"`` (default) fixes this by *skipping* targets with more
    than ``max_atoms`` heavy atoms: the same seeded permutation is walked in the
    same order, and over-long entries are passed over until ``n`` targets are
    collected. Draws that contained no over-long molecule -- which includes every
    run reported in the paper -- therefore return exactly the same targets as
    before; draws that previously crashed now continue with the next eligible
    molecule. The steered distribution of targets is thereby restricted to the
    molecule sizes the generator can actually produce: 30 of the 146,401 val
    molecules (0.020%) exceed 71 heavy atoms, so a 4,096-molecule evaluation
    skips ~0.8 targets on average and the target distribution is otherwise
    unchanged (see ``docs/provenance/geom/data.md``).

    ``selection="legacy"`` restores the original behaviour, crash included, for
    bit-level comparison with the recorded runs.

    Args:
        geom_fp_dir: Directory with ``{split}.lmdb`` and ``{split}_meta.json``.
        split: Which prepared split to draw targets from (paper: ``val``).
        n: Number of targets.
        max_atoms: Generator capacity, i.e. ``data_stats["max_num_atoms"]``.
        seed: Seed of the permutation (the paper's runs use the run seed).
        selection: ``"filter"`` or ``"legacy"``.
        exclude: Dataset indices that must not be drawn -- used to keep an
            evaluation draw disjoint from the held-out pool that hyperparameters
            were selected on (see :func:`holdout_target_indices`).
        indices: Use exactly these dataset indices instead of drawing (used to
            evaluate on the held-out pool itself). Ignores ``seed`` and ``n``.
        return_info: Also return a dict describing what was drawn/skipped.

    Returns:
        ``[n, 1024]`` float tensor of binary target fingerprints, or
        ``(targets, info)`` when ``return_info``.
    """
    if selection not in TARGET_SELECTIONS:
        raise ValueError(f"selection must be one of {TARGET_SELECTIONS}, got {selection!r}")
    geom_fp_dir = Path(geom_fp_dir)

    if indices is not None:
        ds = GEOMFingerprintDataset(str(geom_fp_dir), split=split, max_len=None)
        idxs = [int(i) for i in indices]
        targets = torch.stack([ds[i]["structure"].float() for i in idxs])
        info = {"selection": "explicit", "num_considered": len(idxs), "num_skipped": 0,
                "indices": idxs, "split": split, "max_atoms": max_atoms}
        return (targets, info) if return_info else targets

    if selection == "legacy":
        ds = GEOMFingerprintDataset(str(geom_fp_dir), split=split, max_len=max_atoms)
        g = torch.Generator().manual_seed(seed)
        idxs = torch.randperm(len(ds), generator=g)[:n].tolist()
        targets = torch.stack([ds[i]["structure"].float() for i in idxs])
        info = {"selection": "legacy", "num_considered": len(idxs), "num_skipped": 0,
                "indices": idxs, "split": split, "max_atoms": max_atoms}
        return (targets, info) if return_info else targets

    # "filter": pad to the split's own maximum so indexing never overflows, then
    # walk the identical permutation and skip molecules the generator cannot make.
    ds = GEOMFingerprintDataset(str(geom_fp_dir), split=split, max_len=None)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    excluded = frozenset(int(i) for i in exclude) if exclude is not None else frozenset()
    picked, skipped, considered, n_excluded = [], [], 0, 0
    fps = []
    for i in order:
        if i in excluded:
            n_excluded += 1
            continue
        considered += 1
        rec = ds[i]
        if int(rec["num_atoms"]) > max_atoms:
            skipped.append(i)
            continue
        fps.append(rec["structure"].float())
        picked.append(i)
        if len(picked) == n:
            break
    if len(picked) < n:
        raise SystemExit(
            f"only {len(picked)} of the requested {n} targets in split {split!r} have "
            f"<= {max_atoms} heavy atoms."
        )
    targets = torch.stack(fps)
    info = {
        "selection": "filter",
        "num_considered": considered,
        "num_skipped": len(skipped),
        "num_excluded_holdout": n_excluded,
        "skipped_indices": skipped,
        "indices": picked,
        "split": split,
        "max_atoms": max_atoms,
        "dataset_max_atoms": int(ds.max_len),
    }
    return (targets, info) if return_info else targets


def holdout_target_indices(
    geom_fp_dir: str | Path = DEFAULT_GEOM_FP_DIR,
    split: str = "val",
    n: int = 512,
    max_atoms: int = 71,
    seed: int = 20260929,
) -> list[int]:
    """Dataset indices of a held-out target pool, for selecting hyperparameters.

    The pool is a deterministic function of ``(split, n, max_atoms, seed)``. Pass
    it to :func:`sample_targets` as ``exclude=`` for an evaluation draw, and as
    ``indices=`` (a prefix of it) to tune on, so the two target sets are disjoint
    by construction. Used for the (lambda, mu) selection of the as-described
    Table 3 rerun: the paper states no numeric guidance strengths.
    """
    if n <= 0:
        return []
    _, info = sample_targets(
        geom_fp_dir, split, n, max_atoms, seed, selection="filter", return_info=True
    )
    return list(info["indices"])


@torch.no_grad()
def state_to_binary_fps(state: TensorDict, normalizer: float):
    """OpenBabel FP2 (binary 1024) for each generated molecule; drop padding + dummy."""
    coords = state["coords"].detach().cpu()
    atomics = state["atomics"].detach().cpu()
    pad = state["padding_mask"].detach().cpu()
    fps, valid, n_atoms = [], [], []
    for b in range(coords.shape[0]):
        real = ~pad[b]
        idx = atomics[b].argmax(-1)
        keep = real & (idx != _DUMMY_IDX)
        c = coords[b][keep].numpy().astype(np.float64) * normalizer
        types = idx[keep].tolist()
        n_atoms.append(len(types))
        if len(types) < 2:
            fps.append(torch.zeros(FP_NBITS))
            valid.append(False)
            continue
        try:
            zs = [_ELEMENT_Z[GEOM_ATOM_NAMES[t]] for t in types]
            bits = words_to_bits(compute_fp_words(c, zs))
            ok = bool(bits.sum() > 0)
            fps.append(bits)
            valid.append(ok)
        except Exception:
            fps.append(torch.zeros(FP_NBITS))
            valid.append(False)
    return torch.stack(fps), torch.tensor(valid), torch.tensor(n_atoms)


@torch.no_grad()
def head_binary_fps(state: TensorDict, head, device):
    """What the head *believes* the FP is (sigmoid>0.5), hard atom-type path."""
    coords = state["coords"].to(device).float()
    mask = (~state["padding_mask"].to(device)).long()
    atom_types = state["atomics"].to(device).argmax(-1).long()
    logits = head(coords, atom_types, mask)
    return (torch.sigmoid(logits) > 0.5).float().cpu()


def score(state: TensorDict, targets: torch.Tensor, head, normalizer: float, device) -> dict:
    """Tanimoto similarity of generated molecules to their targets (Table 3).

    ``guide_accuracy_mean`` was added on 2026-09-29 and is purely additive: it is
    the Tanimoto similarity between what the guide head *believes* the molecule's
    fingerprint is and the molecule's *actual* OpenBabel FP2. It is not a
    similarity to the target at all -- it measures whether ``E(G)``, the quantity
    guidance ascends, tracks ``FP2(G)``, the quantity Table 3 reports, on
    *generated* molecules. The head scores 0.852 on real GEOM val molecules
    (``checkpoints/geom/structure_head/best.txt``); if it scores far less here,
    ascending ``E`` cannot be expected to raise ``FP2``, however correct the
    gradient is. See ``docs/geom_guidance_diagnosis.md``.
    """
    gen_fps, valid, n_atoms = state_to_binary_fps(state, normalizer)
    tani = tanimoto_similarity(gen_fps, targets.cpu())  # [B]
    head_fps = head_binary_fps(state, head, device)
    head_tani = tanimoto_similarity(head_fps, targets.cpu())
    guide_acc = tanimoto_similarity(head_fps, gen_fps)  # head vs the molecule's own FP2
    vmask = valid
    return {
        "tanimoto_real_mean": float(tani[vmask].mean()) if vmask.any() else 0.0,
        "tanimoto_real_mean_all": float(tani.mean()),
        "tanimoto_head_mean": float(head_tani.mean()),
        "guide_accuracy_mean": (
            float(guide_acc[vmask].mean()) if vmask.any() else 0.0),
        "conversion_rate": float(vmask.float().mean()),
        "mean_atoms": float(n_atoms.float().mean()),
        "n_valid": int(vmask.sum()),
        # per-molecule arrays for exact cross-shard aggregation (multi-GPU 4096-mol runs)
        "tanimoto_real_per_mol": [float(x) for x in tani.tolist()],
        # The guide's own view of each molecule, which is the only signal a method is
        # allowed to select on: picking by ``tanimoto_real_per_mol`` would be selecting on
        # the reported metric, which no guided arm can do.
        "tanimoto_head_per_mol": [float(x) for x in head_tani.tolist()],
        "valid_per_mol": [bool(x) for x in valid.tolist()],
    }
