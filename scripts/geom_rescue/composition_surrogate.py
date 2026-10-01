#!/usr/bin/env python
"""Is the guide head a usable surrogate for the *counted* FP2 Tanimoto?

The guidance ascends `r(X) = -||sigmoid(head(X)) - c||^2`. Table 3 scores
`Tanimoto(FP2(X), c)`. Those are different functions, and the diagnosis measured
only that the head *predicts a molecule's own fingerprint* well (0.852 on val,
0.757 on generated molecules). That is necessary but not sufficient: guidance
needs `r` to *order* nearby candidates the way the counted metric orders them.

This script measures exactly that ordering, in the composition channel -- the one
channel whose moves FP2 can actually see (a single element substitution changes
tens of path-based bits coherently, whereas a coordinate move either changes
nothing, below 0.006 A, or randomises, above 0.2 A).

For each molecule it enumerates **every** single-atom element substitution, and
for each candidate computes
  * the TRUE counted FP2 Tanimoto to the target,
  * the head's L2 reward (the paper's `r`),
  * the head's differentiable-Tanimoto reward (metric-aligned alternative),
  * the first-order soft-path gradient score (what the sampler's atom-type
    gradient actually points at).
It then reports the rank correlations, and the true gain obtained by following
each surrogate's own top-1 choice against the oracle ceiling and a random flip.

If the surrogates are uncorrelated with the counted metric, no amount of
sampler-side repair helps. If they are correlated, the inert atom-type channel is
a defect worth fixing.

CPU only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from multimfm.geom.paths import ensure_tfg_importable  # noqa: E402
from multimfm.geom.fingerprint import load_structure_head, _ELEMENT_Z  # noqa: E402

ensure_tfg_importable()
from dataloader.datasets.geom_fingerprint import (  # noqa: E402
    GEOM_ATOM_NAMES, compute_fp_words, words_to_bits,
)

NORM = 2.0
N_REAL = 8
DUMMY = GEOM_ATOM_NAMES.index("*")
OUT = REPO / "results" / "geom" / "rescue"


def tani(a, b):
    i = float((a * b).sum())
    return i / (float(a.sum()) + float(b.sum()) - i + 1e-12)


def fp_of(coords_A, types):
    types = np.asarray(types)
    keep = types != DUMMY
    c = np.asarray(coords_A, dtype=np.float64)[keep]
    zs = [_ELEMENT_Z[GEOM_ATOM_NAMES[int(t)]] for t in types[keep]]
    if len(zs) < 2:
        return torch.zeros(1024)
    try:
        return words_to_bits(compute_fp_words(c, zs))
    except Exception:
        return torch.zeros(1024)


def load_mols(lmdb_dir, split, idxs):
    import lmdb
    db = lmdb.open(str(lmdb_dir / f"{split}.lmdb"), subdir=False, readonly=True,
                   lock=False, readahead=False, meminit=False, map_size=50 * (1024 ** 3))
    out = []
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        for i in idxs:
            rec = pickle.loads(txn.get(keys[i]))
            out.append({"coords": np.asarray(rec["coords"], dtype=np.float64),
                        "types": np.asarray(rec["atom_types"], dtype=np.int64),
                        "fp": words_to_bits(np.asarray(rec["fp_words"], dtype="<u4"))})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-mols", type=int, default=24)
    p.add_argument("--max-atoms", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--greedy-rounds", type=int, default=4)
    p.add_argument("--flip-set", default="all", choices=["all", "cnof"],
                   help="'cnof' restricts substitutions to C/N/O/F, whose covalent radii "
                        "are close enough that retyping an atom at fixed geometry does not "
                        "by itself break a PoseBusters bond-length check.")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    from scipy.stats import spearmanr
    rng = np.random.default_rng(a.seed)
    lmdb_dir = REPO / "data" / "geom" / "geom_fp"
    import lmdb
    db = lmdb.open(str(lmdb_dir / "val.lmdb"), subdir=False, readonly=True, lock=False,
                   readahead=False, meminit=False, map_size=50 * (1024 ** 3))
    with db.begin() as txn:
        n_total = txn.stat()["entries"]
    order = rng.permutation(n_total)
    pool = load_mols(lmdb_dir, "val", order[: a.num_mols * 4])
    mols = [m for m in pool if 10 <= len(m["types"]) <= a.max_atoms][: a.num_mols]
    tg = load_mols(lmdb_dir, "val", order[a.num_mols * 4: a.num_mols * 4 + a.num_mols])
    targets = [m["fp"] for m in tg]
    head = load_structure_head(device="cpu", max_len=a.max_atoms, temperature=a.temperature)
    allowed = [0, 1, 2, 3] if a.flip_set == "cnof" else list(range(N_REAL))
    print(f"{len(mols)} molecules, flip_set={a.flip_set} -> "
          f"{[GEOM_ATOM_NAMES[i] for i in allowed]}", flush=True)

    rows = []
    for mi, (m, c) in enumerate(zip(mols, targets)):
        coords = m["coords"]
        cn = torch.as_tensor(coords / NORM, dtype=torch.float32).unsqueeze(0)
        t0 = m["types"].copy()
        n = len(t0)
        mask = torch.ones(1, n, dtype=torch.long)
        base_true = tani(fp_of(coords, t0), c)

        # --- first-order soft-path gradient (both objectives) on the base state
        lin = {}
        for kind in ("l2", "tanimoto"):
            at = torch.zeros(n, 9)
            at[torch.arange(n), torch.as_tensor(t0)] = 1.0
            at.requires_grad_(True)
            probs = torch.softmax(at.unsqueeze(0) / a.temperature, -1)
            h = probs @ head.atom_emb.weight
            h = h * mask.unsqueeze(-1)
            h, _ = head.gnn.forward(h, cn, mask.bool())
            h = h * mask.unsqueeze(-1)
            h = h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            pr = torch.sigmoid(head.predictor(h))[0]
            val = (-((pr - c) ** 2).sum() if kind == "l2"
                   else (pr * c).sum() / (pr.sum() + c.sum() - (pr * c).sum() + 1e-9))
            g, = torch.autograd.grad(val, [at])
            lin[kind] = (g - g.gather(-1, torch.as_tensor(t0).unsqueeze(-1))).numpy()

        # --- enumerate every single substitution
        cand, true_t, r_l2, r_tan, s_l2, s_tan = [], [], [], [], [], []
        with torch.no_grad():
            for ai in range(n):
                for ti in allowed:
                    if ti == t0[ai]:
                        continue
                    t2 = t0.copy(); t2[ai] = ti
                    cand.append((ai, ti))
                    true_t.append(tani(fp_of(coords, t2), c))
                    pr = torch.sigmoid(head(cn, torch.as_tensor(t2).long().unsqueeze(0), mask))[0]
                    r_l2.append(float(-((pr - c) ** 2).sum()))
                    inter = float((pr * c).sum())
                    r_tan.append(inter / (float(pr.sum()) + float(c.sum()) - inter + 1e-9))
                    s_l2.append(float(lin["l2"][ai, ti]))
                    s_tan.append(float(lin["tanimoto"][ai, ti]))
        true_t = np.asarray(true_t)
        gains = true_t - base_true
        row = {
            "n_atoms": n, "base_true": base_true, "n_candidates": len(cand),
            "frac_candidates_positive": float((gains > 0).mean()),
            "oracle_gain": float(gains.max()),
            "mean_gain": float(gains.mean()),
            "random_gain": float(gains[rng.integers(len(gains))]),
            "rho_head_l2": float(spearmanr(r_l2, true_t).statistic),
            "rho_head_tanimoto": float(spearmanr(r_tan, true_t).statistic),
            "rho_lin_l2": float(spearmanr(s_l2, true_t).statistic),
            "rho_lin_tanimoto": float(spearmanr(s_tan, true_t).statistic),
            "gain_head_l2_top1": float(gains[int(np.argmax(r_l2))]),
            "gain_head_tanimoto_top1": float(gains[int(np.argmax(r_tan))]),
            "gain_lin_l2_top1": float(gains[int(np.argmax(s_l2))]),
            "gain_lin_tanimoto_top1": float(gains[int(np.argmax(s_tan))]),
            # which substitution each surrogate picks, so "did it just move toward a
            # generic composition?" can be checked rather than assumed
            "pick_head_tanimoto": [GEOM_ATOM_NAMES[int(t0[cand[int(np.argmax(r_tan))][0]])],
                                   GEOM_ATOM_NAMES[cand[int(np.argmax(r_tan))][1]]],
            "pick_head_l2": [GEOM_ATOM_NAMES[int(t0[cand[int(np.argmax(r_l2))][0]])],
                             GEOM_ATOM_NAMES[cand[int(np.argmax(r_l2))][1]]],
            "pick_oracle": [GEOM_ATOM_NAMES[int(t0[cand[int(np.argmax(true_t))][0]])],
                            GEOM_ATOM_NAMES[cand[int(np.argmax(true_t))][1]]],
            "base_type_fractions": {GEOM_ATOM_NAMES[i]: round(float((t0 == i).mean()), 4)
                                    for i in range(9) if (t0 == i).any()},
        }
        rows.append(row)
        print(f"  {mi+1}/{len(mols)} n={n} base={base_true:.4f} oracle={row['oracle_gain']:+.4f} "
              f"headL2={row['gain_head_l2_top1']:+.4f} headTani={row['gain_head_tanimoto_top1']:+.4f} "
              f"rho(L2)={row['rho_head_l2']:+.3f} rho(Tani)={row['rho_head_tanimoto']:+.3f}",
              flush=True)

    def st(key):
        v = np.asarray([r[key] for r in rows], float)
        return {"mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}

    summary = {k: st(k) for k in rows[0]
               if k not in ("n_atoms", "n_candidates", "pick_head_tanimoto",
                            "pick_head_l2", "pick_oracle", "base_type_fractions")}
    from collections import Counter
    picks = {k: dict(Counter("->".join(r[k]) for r in rows).most_common())
             for k in ("pick_head_tanimoto", "pick_head_l2", "pick_oracle")}
    res = {"_what": __doc__.split("\n")[0], "n_mols": len(rows), "seed": a.seed,
           "flip_set": a.flip_set, "allowed_types": [GEOM_ATOM_NAMES[i] for i in allowed],
           "temperature": a.temperature, "summary": summary,
           "substitution_histogram": picks, "per_mol": rows}
    print("\nsubstitutions chosen (from->to, count over molecules)")
    for k, v in picks.items():
        print(f"  {k:<22s} {v}")
    print("\nsummary (mean +- SEM over molecules)")
    for k, v in summary.items():
        print(f"  {k:<28s} {v['mean']:+.4f} +- {v['sem']:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"composition_surrogate{('_' + a.tag) if a.tag else ''}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
