#!/usr/bin/env python
"""R2: is there ANY operating point at which ascending the guide head raises the
*counted* FP2 Tanimoto?

`docs/geom_guidance_diagnosis.md` §4 concludes there is none, from FP2's response
to *isotropic* coordinate noise (invariant below 0.006 A, randomised by 0.2 A).
A gradient step is not isotropic noise, so that argument is attacked here
directly, in the most favourable possible setting for the method:

  * real GEOM val molecules (the distribution the head was trained on, so the
    head is at its most accurate -- 0.852 val Tanimoto);
  * no flow model, no Monte-Carlo look-ahead: exact gradients of the exact
    reward the paper writes down, r = -||sigmoid(head(X)) - c||^2;
  * unlimited step-size freedom, swept over 3.5 decades;
  * a *paired isotropic control* at exactly the same displacement RMS, so
    "structured vs random" is measured rather than argued.

If a coordinate ascent with full control cannot beat its own isotropic control on
the counted metric, no sampler schedule can, and the §4 conclusion stands with a
much stronger warrant. If it can, the grid missed an operating point.

The same script measures the atom-type channel, which the diagnosis found inert:
  * `--mode types` enumerates every single-atom element substitution and scores
    it with the *true* counted FP2 (the oracle headroom in composition), then
    asks whether the guide head's own ranking of those substitutions finds them.

CPU only, no allocation.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results" / "geom" / "rescue"
DEFAULT_LMDB = REPO / "data" / "geom" / "geom_fp"

import sys
sys.path.insert(0, str(REPO / "src"))
from multimfm.geom.paths import ensure_tfg_importable  # noqa: E402
from multimfm.geom.fingerprint import load_structure_head, _ELEMENT_Z  # noqa: E402

ensure_tfg_importable()
from dataloader.datasets.geom_fingerprint import (  # noqa: E402
    GEOM_ATOM_NAMES, compute_fp_words, words_to_bits,
)

NORMALIZER = 2.0
N_REAL_TYPES = 8  # C N O F S Cl Br I ; index 8 is the dummy


def tanimoto(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = float((a * b).sum())
    return inter / (float(a.sum()) + float(b.sum()) - inter + 1e-12)


def fp_of(coords_A: np.ndarray, types: np.ndarray) -> torch.Tensor:
    """Counted OpenBabel FP2 of a heavy-atom molecule given Angstrom coords."""
    keep = types != GEOM_ATOM_NAMES.index("*")
    c = np.asarray(coords_A, dtype=np.float64)[keep]
    zs = [_ELEMENT_Z[GEOM_ATOM_NAMES[int(t)]] for t in types[keep]]
    if len(zs) < 2:
        return torch.zeros(1024)
    try:
        return words_to_bits(compute_fp_words(c, zs))
    except Exception:
        return torch.zeros(1024)


def load_molecules(lmdb_dir: Path, split: str, idxs):
    import lmdb
    db = lmdb.open(str(lmdb_dir / f"{split}.lmdb"), subdir=False, readonly=True,
                   lock=False, readahead=False, meminit=False, map_size=50 * (1024 ** 3))
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        out = []
        for i in idxs:
            rec = pickle.loads(txn.get(keys[i]))
            out.append({
                "coords": np.asarray(rec["coords"], dtype=np.float64),   # Angstrom, centred
                "types": np.asarray(rec["atom_types"], dtype=np.int64),
                "fp": words_to_bits(np.asarray(rec["fp_words"], dtype="<u4")),
            })
    return out, len(keys)


def head_reward_and_grad(head, coords_A, types, target_fp, *, want_grad=True,
                         soft_types: torch.Tensor | None = None, temperature=0.25):
    """r = -||sigmoid(head(X)) - c||^2 and d r / d coords (Angstrom units)."""
    n = len(types)
    coords = torch.as_tensor(coords_A / NORMALIZER, dtype=torch.float32).unsqueeze(0)
    coords.requires_grad_(want_grad)
    mask = torch.ones(1, n, dtype=torch.long)
    if soft_types is None:
        logits = head(coords, torch.as_tensor(types).long().unsqueeze(0), mask)
    else:
        probs = torch.softmax(soft_types.unsqueeze(0) / temperature, dim=-1)
        h = probs @ head.atom_emb.weight
        h = h * mask.unsqueeze(-1)
        h, _ = head.gnn.forward(h, coords, mask.bool())
        h = h * mask.unsqueeze(-1)
        h = h.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        logits = head.predictor(h)
    pred = torch.sigmoid(logits)[0]
    r = -((pred - target_fp) ** 2).sum()
    if not want_grad:
        return float(r), None, pred.detach()
    g, = torch.autograd.grad(r, [coords])
    # d r / d (Angstrom) = (d r / d normalized) / NORMALIZER
    return float(r), (g[0] / NORMALIZER).detach().numpy().astype(np.float64), pred.detach()


# --------------------------------------------------------------------------- #
# coordinate ascent vs isotropic control
# --------------------------------------------------------------------------- #
def run_coords(head, mols, targets, steps_A, n_inner, rng, verbose=True):
    rows = []
    for si, s in enumerate(steps_A):
        rec = {"displacement_A": s, "n_inner": n_inner,
               "grad_gain": [], "rand_gain": [], "base": [],
               "grad_selfsim": [], "rand_selfsim": [],
               "grad_reward_gain": []}
        for mi, (m, tfp) in enumerate(zip(mols, targets)):
            c0, types = m["coords"], m["types"]
            base_fp = fp_of(c0, types)
            base_t = tanimoto(base_fp, tfp)
            r0, _, _ = head_reward_and_grad(head, c0, types, tfp, want_grad=False)

            # --- gradient ascent, n_inner steps, total per-atom RMS = s Angstrom
            c = c0.copy()
            per = s / max(n_inner, 1) if n_inner == 1 else s / np.sqrt(n_inner)
            # coherent accumulation is the point: use s/n_inner so the *total*
            # displacement is s if the direction is stable, and report the actual
            # realised displacement instead of assuming it.
            per = s / max(n_inner, 1)
            for _ in range(max(n_inner, 1)):
                _, g, _ = head_reward_and_grad(head, c, types, tfp)
                g = g - g.mean(0, keepdims=True)              # zero-COM, as the sampler does
                rms = np.sqrt((g ** 2).mean())
                if rms < 1e-20:
                    break
                # g/rms has unit RMS per *component*, so its per-atom displacement
                # distance has RMS sqrt(3); divide it out so `per` is the per-atom
                # RMS displacement in Angstrom. (The run recorded in
                # results/geom/rescue/ascent_oracle_coords.json multiplied by
                # sqrt(3) instead, so its `displacement_A` labels are 1/3 of the
                # realised displacement; read `realised_disp_A_mean` there.)
                c = c + (per / np.sqrt(3.0)) * g / rms
            grad_disp = float(np.sqrt(((c - c0) ** 2).sum(1).mean()))
            g_fp = fp_of(c, types)
            rg, _, _ = head_reward_and_grad(head, c, types, tfp, want_grad=False)

            # --- isotropic control at the *realised* displacement
            noise = rng.standard_normal(c0.shape)
            noise -= noise.mean(0, keepdims=True)
            noise *= grad_disp / np.sqrt((noise ** 2).sum(1).mean())
            cr = c0 + noise
            r_fp = fp_of(cr, types)

            rec["base"].append(base_t)
            rec["grad_gain"].append(tanimoto(g_fp, tfp) - base_t)
            rec["rand_gain"].append(tanimoto(r_fp, tfp) - base_t)
            rec["grad_selfsim"].append(tanimoto(g_fp, base_fp))
            rec["rand_selfsim"].append(tanimoto(r_fp, base_fp))
            rec["grad_reward_gain"].append(rg - r0)
            rec.setdefault("realised_disp_A", []).append(grad_disp)
        rows.append(summarize(rec))
        if verbose:
            r = rows[-1]
            print(f"  s={s:<7.4g} A  n_inner={n_inner:<3d} "
                  f"grad {r['grad_gain_mean']:+.5f}+-{r['grad_gain_sem']:.5f}  "
                  f"rand {r['rand_gain_mean']:+.5f}+-{r['rand_gain_sem']:.5f}  "
                  f"diff {r['diff_mean']:+.5f}+-{r['diff_sem']:.5f}  "
                  f"selfsim g/r {r['grad_selfsim_mean']:.3f}/{r['rand_selfsim_mean']:.3f}  "
                  f"dReward {r['grad_reward_gain_mean']:+.3f}")
    return rows


def summarize(rec):
    out = {k: v for k, v in rec.items() if not isinstance(v, list)}
    for k, v in rec.items():
        if isinstance(v, list):
            arr = np.asarray(v, dtype=np.float64)
            out[f"{k}_mean"] = float(arr.mean())
            out[f"{k}_sem"] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    d = np.asarray(rec["grad_gain"]) - np.asarray(rec["rand_gain"])
    out["diff_mean"] = float(d.mean())
    out["diff_sem"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    out["n"] = len(rec["grad_gain"])
    out["frac_grad_positive"] = float((np.asarray(rec["grad_gain"]) > 0).mean())
    return out


# --------------------------------------------------------------------------- #
# atom-type channel: oracle headroom and whether the head finds it
# --------------------------------------------------------------------------- #
def run_types(head, mols, targets, rounds, rng, verbose=True):
    per_mol = []
    for m, tfp in zip(mols, targets):
        types0 = m["types"].copy()
        c = m["coords"]
        base_t = tanimoto(fp_of(c, types0), tfp)
        cur_oracle = types0.copy()
        cur_head = types0.copy()
        cur_rand = types0.copy()
        traj = {"oracle": [base_t], "head": [base_t], "random": [base_t]}
        rank_corr = []
        for _ in range(rounds):
            # ---- enumerate all single-atom substitutions of the ORACLE state
            cand, true_gain = [], []
            for ai in range(len(types0)):
                for ti in range(N_REAL_TYPES):
                    if ti == cur_oracle[ai]:
                        continue
                    t2 = cur_oracle.copy(); t2[ai] = ti
                    cand.append((ai, ti))
                    true_gain.append(tanimoto(fp_of(c, t2), tfp) - traj["oracle"][-1])
            if not cand:
                break
            k = int(np.argmax(true_gain))
            cur_oracle[cand[k][0]] = cand[k][1]
            traj["oracle"].append(traj["oracle"][-1] + true_gain[k])

            # ---- the head's own ranking of substitutions of the HEAD state
            hcand, hscore = [], []
            base_r, _, _ = head_reward_and_grad(head, c, cur_head, tfp, want_grad=False)
            for ai in range(len(types0)):
                for ti in range(N_REAL_TYPES):
                    if ti == cur_head[ai]:
                        continue
                    t2 = cur_head.copy(); t2[ai] = ti
                    r2, _, _ = head_reward_and_grad(head, c, t2, tfp, want_grad=False)
                    hcand.append((ai, ti)); hscore.append(r2 - base_r)
            kh = int(np.argmax(hscore))
            t2 = cur_head.copy(); t2[kh // 1] if False else None
            t2[hcand[kh][0]] = hcand[kh][1]
            traj["head"].append(tanimoto(fp_of(c, t2), tfp))
            # rank correlation of head-predicted vs true gain, on the same state
            if np.array_equal(cur_head, cur_oracle):
                from scipy.stats import spearmanr
                rank_corr.append(float(spearmanr(hscore, true_gain).statistic))
            cur_head = t2

            # ---- random single substitution
            ai = rng.integers(len(types0)); ti = rng.integers(N_REAL_TYPES)
            t2 = cur_rand.copy(); t2[ai] = ti
            cur_rand = t2
            traj["random"].append(tanimoto(fp_of(c, t2), tfp))
        per_mol.append({"base": base_t, "traj": traj, "rank_corr": rank_corr,
                        "n_atoms": int(len(types0))})
        if verbose:
            print(f"  mol n={len(types0):<3d} base {base_t:.4f} -> oracle "
                  f"{traj['oracle'][-1]:.4f}  head {traj['head'][-1]:.4f}  "
                  f"random {traj['random'][-1]:.4f}  rho={rank_corr[:1]}")
    return per_mol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", nargs="+", default=["coords"], choices=["coords", "types"])
    p.add_argument("--num-mols", type=int, default=32)
    p.add_argument("--max-atoms", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps-A", type=float, nargs="+",
                   default=[0.002, 0.006, 0.02, 0.05, 0.1, 0.2, 0.5])
    p.add_argument("--inner", type=int, nargs="+", default=[1, 20])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    rng = np.random.default_rng(a.seed)

    # draw molecules and (disjoint) targets from the val split
    import lmdb
    db = lmdb.open(str(DEFAULT_LMDB / "val.lmdb"), subdir=False, readonly=True,
                   lock=False, readahead=False, meminit=False, map_size=50 * (1024 ** 3))
    with db.begin() as txn:
        n_total = txn.stat()["entries"]
    order = rng.permutation(n_total)
    mols, _ = load_molecules(DEFAULT_LMDB, "val", order[: a.num_mols * 4])
    mols = [m for m in mols if 8 <= len(m["types"]) <= a.max_atoms][: a.num_mols]
    tmols, _ = load_molecules(DEFAULT_LMDB, "val",
                              order[a.num_mols * 4: a.num_mols * 4 + a.num_mols])
    targets = [m["fp"] for m in tmols]
    print(f"{len(mols)} molecules "
          f"(atoms {min(len(m['types']) for m in mols)}-{max(len(m['types']) for m in mols)}), "
          f"{len(targets)} targets")

    head = load_structure_head(device="cpu", max_len=max(len(m["types"]) for m in mols))
    out = {"_what": __doc__.split("\n")[0], "n_mols": len(mols), "seed": a.seed,
           "max_atoms": a.max_atoms}

    if "coords" in a.mode:
        out["coords"] = {}
        for n_inner in a.inner:
            print(f"\ncoordinate ascent, {n_inner} inner step(s):")
            out["coords"][f"inner{n_inner}"] = run_coords(
                head, mols, targets, a.steps_A, n_inner, rng)
    if "types" in a.mode:
        print(f"\natom-type substitutions, {a.rounds} greedy round(s):")
        pm = run_types(head, mols, targets, a.rounds, rng)
        out["types"] = {
            "per_mol": pm,
            "summary": {
                "base": float(np.mean([x["base"] for x in pm])),
                **{f"{k}_final": float(np.mean([x["traj"][k][-1] for x in pm]))
                   for k in ("oracle", "head", "random")},
                **{f"{k}_gain_round1": float(np.mean(
                    [x["traj"][k][1] - x["traj"][k][0] for x in pm]))
                   for k in ("oracle", "head", "random")},
                "head_gain_round1_sem": float(np.std(
                    [x["traj"]["head"][1] - x["traj"]["head"][0] for x in pm], ddof=1)
                    / np.sqrt(len(pm))),
                "rank_corr_mean": float(np.mean(
                    [x["rank_corr"][0] for x in pm if x["rank_corr"]])),
            },
        }
        print(json.dumps(out["types"]["summary"], indent=2))

    path = a.out or (DEFAULT_OUT / f"ascent_oracle{('_' + a.tag) if a.tag else ''}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
