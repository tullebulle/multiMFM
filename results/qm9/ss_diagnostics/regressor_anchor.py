"""Anchor the QM9 guide and oracle regressors on held-out ground-truth molecules.

Diagnostic for docs/ss_guide_oracle_gap.md (the multiMFM-SS guide/oracle gap).
Run on CPU from the repo root:

    ~/micromamba/envs/multimfm/bin/python results/qm9/ss_diagnostics/regressor_anchor.py \
        --limit 2000 --partition test --out results/qm9/ss_diagnostics/regressor_anchor.json

What it measures, on *real* QM9 molecules in exactly the representation
``multimfm.glass_guidance.tfg_property_prediction`` feeds the regressors
(heavy atoms only, centred coordinates, argmax atom types, TFG atom order,
``padding_mask``):

1. per-kind (guide / oracle) per-property MAE / RMSE / R^2 against the dataset value;
2. the full 6x6 cross-property table (checkpoint for property p scored against
   property q) -- this is the test that the ``REGRESSOR_LOSSES`` loss-string ->
   property mapping in ``multimfm.qm9_data`` picks the right weight file;
3. guide-vs-oracle agreement: correlation of the two predictions and of their
   signed errors, plus the MAE of one regressor treated as the other's target
   ("cross MAE"), which is the quantity a best-of-N selection on the guide can
   drive to zero;
4. a byte-level check that both regressors receive *identical* input tensors
   (forward hooks record a hash of (coords, atom_types, mask)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from multimfm import paths  # noqa: E402
from multimfm.glass_guidance import tfg_property_prediction  # noqa: E402
from multimfm.qm9_data import (  # noqa: E402
    TFG_QM9_ATOM_NAMES,
    datapoint_to_tensor,
    load_dataset,
    load_tfg_regressor,
    tfg_partition_indices,
)
from multimfm.qm9_results import HARTREE_TO_MEV, UNITS  # noqa: E402

PROPS = ("cv", "mu", "alpha", "gap", "homo", "lumo")
ENERGY = {"gap", "homo", "lumo"}


def paper_scale(prop: str) -> float:
    return HARTREE_TO_MEV if prop in ENERGY else 1.0


def build_batch(partition: str, limit: int, max_len: int = 9):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ds = load_dataset("yairschiff/qm9", split="train",
                      cache_dir=os.environ["HF_DATASETS_CACHE"])
    idx = tfg_partition_indices(len(ds), partition, limit=0)
    coords, atomics, pmask, props = [], [], [], {p: [] for p in PROPS}
    for i in idx:
        dp = ds[int(i)]
        t = datapoint_to_tensor(dp, max_len=max_len, include_hydrogens=False)
        if t is None:
            continue
        coords.append(t["coords"])
        atomics.append(t["atomics"])
        pmask.append(t["padding_mask"])
        for p in PROPS:
            props[p].append(float(dp[p]))
        if limit > 0 and len(coords) >= limit:
            break
    state = {
        "coords": torch.stack(coords),
        "atomics": torch.stack(atomics),
        "padding_mask": torch.stack(pmask),
    }
    truth = {p: torch.tensor(v, dtype=torch.float32) for p, v in props.items()}
    return state, truth


def predict(reg, state, batch_size=256):
    out, seen = [], []
    n = state["coords"].shape[0]

    def hook(_m, inputs):
        coords, atom_types, mask = inputs
        h = hashlib.sha256()
        for t in (coords, atom_types, mask):
            h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
        seen.append(h.hexdigest())

    handle = reg.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, n, batch_size):
                sub = {k: v[s:s + batch_size] for k, v in state.items()}
                out.append(tfg_property_prediction(sub, reg, soft=False))
    finally:
        handle.remove()
    return torch.cat(out), seen


def metrics(pred: torch.Tensor, truth: torch.Tensor, scale: float) -> dict:
    err = (pred - truth) * scale
    ss = ((truth - truth.mean()) * scale).square().sum()
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(err.square().mean().sqrt()),
        "mse_raw": float((pred - truth).square().mean()),
        "bias": float(err.mean()),
        "r2": float(1.0 - err.square().sum() / ss) if float(ss) > 0 else None,
        "truth_mad": float((truth - truth.mean()).abs().mean() * scale),
        "truth_std": float(truth.std() * scale),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", default="test",
                    choices=["train_flow", "train_classifier", "validation", "test"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--cross", action="store_true",
                    help="also compute the 6x6 cross-property table")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/qm9/ss_diagnostics/regressor_anchor.json")
    args = ap.parse_args(argv)

    device = torch.device("cpu")
    state, truth = build_batch(args.partition, args.limit)
    n = state["coords"].shape[0]
    print(f"{n} held-out molecules from partition {args.partition}")

    res = {
        "partition": args.partition,
        "num_molecules": n,
        "atom_names": list(TFG_QM9_ATOM_NAMES),
        "hartree_to_mev": HARTREE_TO_MEV,
        "units": {p: UNITS[p] for p in PROPS},
        "per_property": {},
        "cross_property": {},
        "input_identical": {},
    }
    preds = {}
    for prop in PROPS:
        scale = paper_scale(prop)
        entry = {"unit": UNITS[prop], "scale_from_raw": scale}
        hashes = {}
        for kind in ("guide", "oracle"):
            reg = load_tfg_regressor(kind, prop, device)
            reg.soft_atom_temperature = 0.25
            p, seen = predict(reg, state)
            preds[(kind, prop)] = p
            hashes[kind] = seen
            entry[kind] = metrics(p, truth[prop], scale)
            del reg
        res["input_identical"][prop] = bool(hashes["guide"] == hashes["oracle"])
        g, o = preds[("guide", prop)], preds[("oracle", prop)]
        t = truth[prop]
        eg, eo = (g - t) * scale, (o - t) * scale
        entry["agreement"] = {
            "pred_corr": float(np.corrcoef(g.numpy(), o.numpy())[0, 1]),
            "err_corr": float(np.corrcoef(eg.numpy(), eo.numpy())[0, 1]),
            "cross_mae_guide_vs_oracle": float((g - o).abs().mean() * scale),
            "oracle_mae_over_guide_mae": float(eo.abs().mean() / eg.abs().mean()),
        }
        res["per_property"][prop] = entry
        print(prop, json.dumps({k: entry[k] for k in ("guide", "oracle", "agreement")},
                               indent=None)[:400])

    if args.cross:
        for kind in ("guide", "oracle"):
            res["cross_property"][kind] = {}
            for ckpt_prop in PROPS:
                reg = load_tfg_regressor(kind, ckpt_prop, device)
                p, _ = predict(reg, state)
                row = {}
                for target_prop in PROPS:
                    row[target_prop] = metrics(p, truth[target_prop],
                                               paper_scale(target_prop))["mae"]
                res["cross_property"][kind][ckpt_prop] = row
                print("cross", kind, ckpt_prop, {k: round(v, 4) for k, v in row.items()})
                del reg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
