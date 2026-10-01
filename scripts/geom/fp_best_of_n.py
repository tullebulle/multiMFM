#!/usr/bin/env python
"""Best-of-N on the GEOM-Drugs fingerprint task, on the same backbone as multiMFM.

    python scripts/geom/fp_best_of_n.py --num-mols 256 --n-candidates 4 --seed 201 --pb

Why this baseline. multiMFM's claim on this task is that a Monte-Carlo value estimate steers
better per network call than the alternatives. The published comparison is to TFG-Flow, which
fixes the external scale but shares no code with us, so it cannot isolate the estimator.
Best-of-N shares everything -- base generator, targets, guide head, metric, validity
treatment -- and differs only in spending its budget on independent draws rather than on
guidance. It is the comparison that says whether guidance is doing anything that sampling
harder would not.

Selection is on the GUIDE head, never on the reported metric. Picking by the directly
counted FP2 Tanimoto would let the baseline select on the test quantity, which no guided
method can do, and would turn this into an oracle rather than a baseline.

With --pb, a candidate must additionally be PoseBusters-valid to be eligible, and the
best-scoring valid candidate is returned; if none is valid the guide's favourite is returned
and the molecule counts as invalid. That mirrors how steer_search treats its look-aheads, so
the two selection rules are read the same way.

Cost. N candidates cost N full trajectories, i.e. N x sample_steps generative forward passes
per returned molecule and no backward passes, which is what puts it on the NFE axis against
the guided arms.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-mols", type=int, default=256, help="targets, i.e. molecules returned")
    p.add_argument("--n-candidates", type=int, default=4, help="draws per target")
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--max-atoms", type=int, default=None,
                   help="cap heavy atoms (30 reproduces TFG-Flow's protocol)")
    p.add_argument("--target-split", default="val")
    p.add_argument("--target-selection", default="filter")
    p.add_argument("--batch", type=int, default=64, help="molecules per forward batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pb", action="store_true", help="require PoseBusters validity to be eligible")
    p.add_argument("--pb-config", default=None)
    p.add_argument("--soft-atom-temperature", type=float, default=0.25,
                   help="matches the steering drivers' default")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="results/geom/runs/fp_best_of_n")
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    from multimfm.geom import fingerprint as fp
    from multimfm.geom.paths import DEFAULT_GEOM_FP_DIR, DEFAULT_PB_CONFIG, DEFAULT_STRUCTURE_HEAD, DEFAULT_TEACHER
    from multimfm.geom.teacher import load_geom_teacher
    from multimfm.geom.reward import install_fingerprint_reward
    from multimfm.glass_guidance import build_posebusters, guided_sample, pb_summary_for_state
    from multimfm.nfe import NFECounter, attach_standard
    from multimfm.train_base_flow import choose_device, set_seed

    install_fingerprint_reward()
    set_seed(a.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(a.device)
    model, stats, _ = load_geom_teacher(DEFAULT_TEACHER, device, use_ema=True)
    head = fp.load_structure_head(DEFAULT_STRUCTURE_HEAD, device, a.soft_atom_temperature)
    normalizer = float(stats.get("coordinate_normalizer", 2.0))
    max_atoms = int(a.max_atoms or stats.get("max_num_atoms", 71))
    nfe = attach_standard(NFECounter(), model=model, mfm_student=None)

    targets, info = fp.sample_targets(DEFAULT_GEOM_FP_DIR, a.target_split, a.num_mols,
                                      max_atoms, a.seed, selection=a.target_selection,
                                      return_info=True)
    targets = targets.to(device)
    K = a.n_candidates
    print(f"[{a.tag or 'bon'}] n={a.num_mols} K={K} max_atoms={max_atoms} "
          f"targets skipped {info['num_skipped']}")

    pb = build_posebusters(Path(a.pb_config or DEFAULT_PB_CONFIG)) if a.pb else None
    pb_stats = dict(stats)
    pb_stats["atom_names"] = list(fp.GEOM_ATOM_NAMES)
    pb_stats["coordinate_normalizer"] = normalizer

    args_ns = SimpleNamespace(sample_steps=a.sample_steps)
    rows = []
    for lo in range(0, a.num_mols, a.batch):
        hi = min(lo + a.batch, a.num_mols)
        tgt = targets[lo:hi]
        best = None  # (head_score, valid, state_slice) per target, as tensors
        for k in range(K):
            set_seed(a.seed * 100003 + lo * 97 + k)
            init = model._sample_noise_like_batch(batch_size=hi - lo).to(device)
            # mu=0 and value_samples=0 make this a plain unguided draw; the guidance
            # arguments are required by the signature but never reached.
            state, _ = guided_sample(
                model, init, value_sampler="glass", mfm_student=None,
                num_steps=a.sample_steps, mu=0.0, value_samples=0,
                reward_name="fingerprint_tanimoto", target_x=0.0,
                property_name="structure", property_target=tgt, property_regressor=head,
                reward_scale=1.0, value_glass_steps=0, guide_atomics=False,
                guide_min_t=0.1, guide_max_t=0.9, guide_every=1,
                guidance_max_coord_rms=0.0, guidance_max_atom_rms=0.0, eps=1e-6)
            s = fp.score(state, tgt, head, normalizer, device)
            # selection signal: the guide's own view, never the reported metric
            head_tani = torch.as_tensor(s["tanimoto_head_per_mol"], dtype=torch.float)
            valid = torch.as_tensor(s["valid_per_mol"], dtype=torch.bool)
            if pb is not None:
                pbs = pb_summary_for_state(state, data_stats=pb_stats, posebusters=pb, sanitize=True)
                per = pbs.get("pb_valid_per_mol")
                if per is not None:
                    valid = valid & torch.as_tensor(per, dtype=torch.bool)
            true_tani = torch.as_tensor(s["tanimoto_real_per_mol"], dtype=torch.float)
            cand = (head_tani.float(), valid, true_tani)
            if best is None:
                best = cand
            else:
                bh, bv, bt = best
                ch, cv, ct = cand
                # prefer a valid candidate; among equals prefer the higher guide score
                take = (cv & ~bv) | (cv.eq(bv) & (ch > bh))
                best = (torch.where(take, ch, bh), bv | (take & cv), torch.where(take, ct, bt))
        bh, bv, bt = best
        for j in range(hi - lo):
            rows.append({"target_idx": lo + j, "tanimoto_real": float(bt[j]),
                         "head_score": float(bh[j]), "valid": bool(bv[j])})
        print(f"  {hi}/{a.num_mols}  running Tanimoto "
              f"{np.mean([r['tanimoto_real'] for r in rows]):.4f}", flush=True)

    tan = np.array([r["tanimoto_real"] for r in rows])
    summ = {
        "method": "best_of_n", "n_candidates": K, "num_mols": len(rows),
        "max_atoms": max_atoms, "seed": a.seed, "sample_steps": a.sample_steps,
        "tanimoto_real_mean": float(tan.mean()),
        "tanimoto_real_sem": float(tan.std(ddof=1) / math.sqrt(len(tan))) if len(tan) > 1 else 0.0,
        "pb_valid": float(np.mean([r["valid"] for r in rows])) if a.pb else None,
        "nfe": nfe.summary(len(rows)) if hasattr(nfe, "summary") else {},
        "targets_skipped": info["num_skipped"], "args": vars(a),
    }
    tag = a.tag or f"bon_k{K}_n{len(rows)}_seed{a.seed}"
    d = REPO / a.out_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.json").write_text(json.dumps({"summary": summ, "per_molecule": rows}, indent=2) + "\n")
    print(f"\nbest-of-{K}: Tanimoto {summ['tanimoto_real_mean']:.4f} "
          f"+- {summ['tanimoto_real_sem']:.4f} over {len(rows)} molecules"
          + (f", PB {summ['pb_valid']:.3f}" if a.pb else ""))
    out = d / f"{tag}.json"
    try:
        out = out.relative_to(REPO)
    except ValueError:
        pass
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
