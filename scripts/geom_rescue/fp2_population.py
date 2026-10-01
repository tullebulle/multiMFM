#!/usr/bin/env python
"""R1: what FP2 Tanimoto *means* on this benchmark.

Measures, on the prepared geom_fp val split (146,401 real GEOM-Drugs molecules,
the pool Table 3's targets are drawn from):

1. the random-pair FP2 Tanimoto as a function of the two molecules' heavy-atom
   counts -- so the 0.2278 unguided baseline and TFG-Flow's 0.170 "upper bound"
   can be compared on a common footing (TFG-Flow caps molecules at 30 heavy
   atoms, our generator at 71);
2. the *headroom*: for a random target, the distribution of Tanimoto over the
   whole 146k population, and what best-of-N selection would reach. If the best
   attainable match is barely above the mean, no guidance method can move the
   number and the negative result is about the benchmark, not the method.

CPU only, no allocation, a couple of minutes, < 1 GB.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results" / "geom" / "rescue" / "fp2_population.json"
DEFAULT_LMDB = REPO / "data" / "geom" / "geom_fp"


def load_population(lmdb_dir: Path, split: str = "val"):
    """Return (bits [N,1024] uint8, n_atoms [N] int32) for the whole split."""
    import lmdb

    db = lmdb.open(
        str(lmdb_dir / f"{split}.lmdb"), subdir=False, readonly=True, lock=False,
        readahead=False, meminit=False, map_size=50 * (1024 ** 3),
    )
    words, n_atoms = [], []
    with db.begin() as txn:
        for _, v in txn.cursor():
            rec = pickle.loads(v)
            words.append(np.asarray(rec["fp_words"], dtype="<u4"))
            n_atoms.append(int(np.asarray(rec["coords"]).shape[0]))
    w = np.stack(words)                                    # [N, 32] uint32
    bits = np.unpackbits(w.view(np.uint8).reshape(len(w), -1), axis=1, bitorder="little")
    return bits.astype(np.uint8), np.asarray(n_atoms, dtype=np.int32)


def tanimoto_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Tanimoto for two equal-length uint8 bit matrices."""
    A = A.astype(np.int32)
    B = B.astype(np.int32)
    inter = (A * B).sum(1)
    return inter / (A.sum(1) + B.sum(1) - inter + 1e-12)


def tanimoto_one_vs_all(a: np.ndarray, B: np.ndarray, popcount_B: np.ndarray) -> np.ndarray:
    inter = B.astype(np.float32) @ a.astype(np.float32)
    sa = float(a.sum())
    return inter / (sa + popcount_B - inter + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb-dir", type=Path, default=DEFAULT_LMDB)
    p.add_argument("--split", default="val")
    p.add_argument("--pairs", type=int, default=200_000)
    p.add_argument("--targets", type=int, default=200, help="targets for the headroom scan")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()

    bits, n_atoms = load_population(a.lmdb_dir, a.split)
    N = len(bits)
    pop = bits.sum(1)
    rng = np.random.default_rng(a.seed)
    out: dict = {
        "_what": "FP2 Tanimoto population statistics on the geom_fp val split; "
                 "random-pair baselines by molecule size and the attainable headroom.",
        "n_molecules": int(N),
        "n_atoms": {
            "mean": float(n_atoms.mean()), "median": float(np.median(n_atoms)),
            "min": int(n_atoms.min()), "max": int(n_atoms.max()),
            "frac_le_30": float((n_atoms <= 30).mean()),
            "frac_le_71": float((n_atoms <= 71).mean()),
        },
        "bits_set": {"mean": float(pop.mean()), "median": float(np.median(pop))},
    }

    # 1. random-pair Tanimoto, whole population and size-restricted subsets.
    def random_pairs(mask_a, mask_b, label):
        ia = np.flatnonzero(mask_a)
        ib = np.flatnonzero(mask_b)
        if len(ia) < 2 or len(ib) < 2:
            return None
        i = rng.choice(ia, size=a.pairs)
        j = rng.choice(ib, size=a.pairs)
        keep = i != j
        t = tanimoto_rows(bits[i[keep]], bits[j[keep]])
        return {
            "label": label, "n_pairs": int(keep.sum()),
            "mean": float(t.mean()), "sem": float(t.std(ddof=1) / np.sqrt(keep.sum())),
            "median": float(np.median(t)),
            "p90": float(np.quantile(t, 0.90)), "p99": float(np.quantile(t, 0.99)),
            "max": float(t.max()),
            "mean_n_atoms_a": float(n_atoms[i[keep]].mean()),
            "mean_n_atoms_b": float(n_atoms[j[keep]].mean()),
        }

    all_m = np.ones(N, bool)
    out["random_pairs"] = [
        random_pairs(all_m, all_m, "all x all"),
        random_pairs(n_atoms <= 30, n_atoms <= 30, "<=30 x <=30  (TFG-Flow's regime)"),
        random_pairs(n_atoms <= 71, n_atoms <= 71, "<=71 x <=71  (our target filter)"),
        random_pairs((n_atoms >= 20) & (n_atoms <= 30), n_atoms <= 71,
                     "gen-like 20-30 atoms x targets <=71  (our Table 3 pairing)"),
        random_pairs((n_atoms >= 20) & (n_atoms <= 30), (n_atoms >= 20) & (n_atoms <= 30),
                     "20-30 x 20-30"),
    ]

    # 1b. mean Tanimoto as a joint function of both sizes (coarse bins).
    edges = [0, 15, 20, 25, 30, 40, 55, 100]
    grid = []
    for lo_a, hi_a in zip(edges[:-1], edges[1:]):
        ma = (n_atoms > lo_a) & (n_atoms <= hi_a)
        row = []
        for lo_b, hi_b in zip(edges[:-1], edges[1:]):
            mb = (n_atoms > lo_b) & (n_atoms <= hi_b)
            ia, ib = np.flatnonzero(ma), np.flatnonzero(mb)
            if len(ia) < 50 or len(ib) < 50:
                row.append(None)
                continue
            i = rng.choice(ia, size=20_000)
            j = rng.choice(ib, size=20_000)
            k = i != j
            row.append(round(float(tanimoto_rows(bits[i[k]], bits[j[k]]).mean()), 5))
        grid.append({"size_a": f"({lo_a},{hi_a}]", "row": row})
    out["size_grid"] = {"edges": edges, "rows": grid}

    # 2. headroom: for each of `targets` random targets, the whole-population
    #    Tanimoto distribution and best-of-N order statistics.
    tgt_idx = rng.choice(np.flatnonzero(n_atoms <= 71), size=a.targets, replace=False)
    popf = pop.astype(np.float32)
    stats = {k: [] for k in ("mean", "p50", "p90", "p99", "p999", "max",
                             "best_of_64", "best_of_4096")}
    for ti in tgt_idx:
        t = tanimoto_one_vs_all(bits[ti], bits, popf)
        t[ti] = -1.0
        srt = np.sort(t)[::-1]
        stats["mean"].append(float(t.mean()))
        stats["p50"].append(float(np.quantile(t, 0.50)))
        stats["p90"].append(float(np.quantile(t, 0.90)))
        stats["p99"].append(float(np.quantile(t, 0.99)))
        stats["p999"].append(float(np.quantile(t, 0.999)))
        stats["max"].append(float(srt[0]))
        # expected best-of-N from N uniform draws: 1 - (1-q)^N crossing -> use quantile
        stats["best_of_64"].append(float(np.quantile(t, 1 - 1 / 64)))
        stats["best_of_4096"].append(float(np.quantile(t, 1 - 1 / 4096)))
    out["headroom"] = {
        "n_targets": int(a.targets),
        "note": "for each target, Tanimoto against all 146,401 val molecules; "
                "best_of_N is the (1-1/N) quantile, i.e. the median best-of-N draw.",
        **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
           for k, v in stats.items()},
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_atoms", "bits_set")}, indent=2))
    print("\nrandom-pair baselines")
    for r in out["random_pairs"]:
        if r:
            print(f"  {r['label']:<46s} {r['mean']:.5f} +- {r['sem']:.5f}  "
                  f"(n_a={r['mean_n_atoms_a']:.1f} n_b={r['mean_n_atoms_b']:.1f})")
    print("\nheadroom (mean over targets)")
    for k in ("mean", "p50", "p90", "p99", "p999", "max", "best_of_64", "best_of_4096"):
        print(f"  {k:<14s} {out['headroom'][k]['mean']:.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
