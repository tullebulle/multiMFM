#!/usr/bin/env python
"""GEOM-Drugs steering to a radius-of-gyration target, against the same baselines as QM9.

    python scripts/geom/rg_benchmark.py --method multimfm --n 128 --value-samples 4

Why this reward. The fingerprint benchmark cannot separate methods: random molecule pairs
score 0.2164 on it against a best-reported 0.23, and the only lever that moves it leaves
85% of molecules PoseBusters-invalid. R_g is computed exactly from the coordinates and is
exactly differentiable, so the reward *is* the metric -- there is no learned guide between
the claim and the evidence, and no guide-oracle gap to saturate. It is also a coordinate
target, which is the channel the interpolant-annihilation measurement showed survives.

Methods, matching the QM9 comparison so the two are read the same way:

``base``        unguided sampling, the reference point.
``best_of_n``   draw N, return the one closest to the target. The real competitor for a
                cheap scalar reward, which is why targets sit in the tail.
``dps``         steer one trajectory by the gradient of the reward at the one-step
                denoiser endpoint, r(E[x1 | xt]). The like-for-like comparison for multiMFM.
``multimfm``    steer one trajectory by the Monte-Carlo value gradient over flow-map
                posterior samples.

Targets are drawn per molecule from the **size-conditional** R_g distribution (R_g scales
with atom count, so one global target would select on size rather than shape), at a
declared number of standard deviations into the tail.

Every method reports PoseBusters validity alongside the R_g error. Stretching a molecule is
the obvious way to raise R_g and it is exactly what PB penalises, so a target hit at the
cost of validity is not a result -- that is the trap the composition rescue fell into.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def build_targets(rg_stats: dict, n_atoms: torch.Tensor, k_sd: float,
                  rng: np.random.Generator) -> torch.Tensor:
    """Per-molecule target: its own size's mean R_g, pushed k standard deviations out."""
    by = {int(k): v for k, v in rg_stats["by_n_atoms"].items()}
    sizes = np.array(sorted(by))
    out = []
    for n in n_atoms.tolist():
        near = int(sizes[np.argmin(np.abs(sizes - int(n)))])
        s = by[near]
        out.append(s["mean"] + k_sd * s["std"])
    return torch.tensor(out, dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["base", "best_of_n", "dps", "multimfm"])
    p.add_argument("--n", type=int, default=128, help="molecules returned")
    p.add_argument("--n-candidates", type=int, default=1, help="best_of_n: candidates per target")
    p.add_argument("--value-samples", type=int, default=4, help="multimfm: posterior samples")
    p.add_argument("--mfm-steps", type=int, default=10)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--mu", type=float, default=0.5, help="guidance strength")
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--guide-every", type=int, default=1)
    p.add_argument("--guide-min-t", type=float, default=0.1)
    p.add_argument("--guide-max-t", type=float, default=0.9)
    p.add_argument("--k-sd", type=float, default=3.0, help="target, in sd into the tail")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rg-stats", default="results/geom/rg_preflight.json")
    p.add_argument("--tag", default=None)
    p.add_argument("--pb", action="store_true",
                   help="also score PoseBusters validity; required before any number is "
                        "reported, since raising R_g by stretching a molecule is exactly "
                        "what PB penalises")
    p.add_argument("--pb-config", default=None)
    p.add_argument("--out-dir", default="results/geom/rg_benchmark")
    a = p.parse_args()

    from multimfm.geom.paths import DEFAULT_STUDENT, DEFAULT_TEACHER
    from multimfm.geom.teacher import load_geom_teacher
    from multimfm.glass_guidance import (load_mfm_student, production_step_with_optional_guidance,
                                         radius_gyration, value_gradient)
    from multimfm.glass_posterior import endpoint_prediction
    from multimfm.nfe import NFECounter, attach_standard
    if a.pb:
        from multimfm.geom import fingerprint as fp
        from multimfm.geom.paths import DEFAULT_PB_CONFIG
        from multimfm.glass_guidance import build_posebusters, pb_summary_for_state

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, model_args = load_geom_teacher(DEFAULT_TEACHER, dev)
    student = None
    if a.method == "multimfm":
        student = load_mfm_student(DEFAULT_STUDENT, data_stats=stats, teacher_args=model_args,
                                   device=dev, key="student")
    rg_stats = json.loads((REPO / a.rg_stats).read_text())
    nfe = attach_standard(NFECounter(), model=model, mfm_student=student)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed + 7919)
    K = a.n_candidates if a.method == "best_of_n" else 1

    rows = []
    done = 0
    while done < a.n:
        b = min(a.batch, a.n - done)
        init = model._sample_noise_like_batch(batch_size=b).to(dev)
        n_atoms = (~init["padding_mask"]).sum(dim=1).detach().cpu()
        tgt = build_targets(rg_stats, n_atoms, a.k_sd, rng).to(dev)

        if K > 1:                                  # best-of-N: K candidates per target
            idx = torch.arange(b, device=dev).repeat_interleave(K)
            init = model._sample_noise_like_batch(batch_size=b * K).to(dev)
            tgt_rep = tgt[idx]
        else:
            tgt_rep = tgt

        sched = model._get_sample_schedule(a.sample_steps).to(dev)
        st = init
        B = st["coords"].shape[0]
        for i in range(1, len(sched)):
            t_val = sched[i - 1]
            t = t_val.expand(B)
            dt = (sched[i] - sched[i - 1]).expand(B)
            gc = ga = None
            guided = (a.method in ("dps", "multimfm")
                      and a.guide_min_t <= float(t_val) <= a.guide_max_t
                      and (i - 1) % a.guide_every == 0)
            if guided:
                if a.method == "multimfm":
                    gc, ga, _ = value_gradient(
                        model, st, t, reward_name="target_property",
                        property_name="radius_gyration", property_target=tgt_rep,
                        property_regressor=None, target_x=None,
                        reward_scale=a.reward_scale, posterior_samples=a.value_samples,
                        n_steps=a.mfm_steps, value_sampler="mfm", mfm_student=student,
                        mfm_diagonal=True, guide_atomics=False, eps=1e-6, prior_noise=None)
                else:                               # DPS: reward at the one-step endpoint
                    coords = st["coords"].detach().clone().requires_grad_(True)
                    from tensordict import TensorDict
                    cond = TensorDict({"coords": coords, "atomics": st["atomics"],
                                       "padding_mask": st["padding_mask"]}, batch_size=B).to(dev)
                    with torch.enable_grad():
                        x1 = endpoint_prediction(model, cond, t)
                        rg = radius_gyration(x1["coords"], x1["padding_mask"])
                        v = -a.reward_scale * (rg - tgt_rep).square()
                        gc = torch.autograd.grad(v.sum(), coords)[0].detach()
                    # NFECounter tallies the forward itself through its module hooks; the
                    # backward is one per guided step and is accounted for on the axis, not here.
            with torch.no_grad():
                st = production_step_with_optional_guidance(
                    model, st, t, dt, mu=(a.mu if guided else 0.0),
                    grad_coords=gc, grad_atomics=ga)

        with torch.no_grad():
            rg = radius_gyration(st["coords"], st["padding_mask"]).detach().float().cpu()
        err = (rg - tgt_rep.detach().float().cpu()).abs()
        if K > 1:                                   # keep the closest candidate per target
            err = err.view(b, K)
            pick = err.argmin(dim=1)
            flat = torch.arange(b) * K + pick
            keep = flat
            err = err.gather(1, pick[:, None]).squeeze(1)
            rg = rg[keep]
            st = {k: st[k][keep.to(dev)] for k in ("coords", "atomics", "padding_mask")}
        if a.pb:
            pb_stats = dict(stats)
            pb_stats["atom_names"] = list(fp.GEOM_ATOM_NAMES)
            pb_stats["coordinate_normalizer"] = float(stats.get("coordinate_normalizer", 1.0))
            summ_pb = pb_summary_for_state(
                st if isinstance(st, dict) else st,
                data_stats=pb_stats,
                posebusters=build_posebusters(Path(a.pb_config or DEFAULT_PB_CONFIG)),
                sanitize=True)
            pb_batch = float(summ_pb.get("pb_valid", 0.0))
        else:
            pb_batch = None
        for j in range(b):
            rows.append({"n_atoms": int(n_atoms[j]), "target": float(tgt[j]),
                         "rg": float(rg[j]), "abs_err": float(err[j]),
                         "pb_valid_batch": pb_batch})
        done += b
        print(f"  {done}/{a.n}  running MAE {np.mean([r['abs_err'] for r in rows]):.4f}", flush=True)

    errs = np.array([r["abs_err"] for r in rows])
    summ = {
        "method": a.method, "n": len(rows), "k_sd": a.k_sd, "mu": a.mu,
        "value_samples": a.value_samples if a.method == "multimfm" else None,
        "n_candidates": K if a.method == "best_of_n" else None,
        "rg_mae": float(errs.mean()),
        "rg_mae_sem": float(errs.std(ddof=1) / math.sqrt(len(errs))) if len(errs) > 1 else 0.0,
        "pb_valid": (float(np.mean([r["pb_valid_batch"] for r in rows]))
                     if a.pb and rows and rows[0]["pb_valid_batch"] is not None else None),
        "nfe": nfe.summary(len(rows)) if hasattr(nfe, "summary") else {},
        "args": vars(a),
    }
    tag = a.tag or f"{a.method}_k{a.k_sd}_n{len(rows)}_seed{a.seed}"
    d = REPO / a.out_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    (d / "per_molecule.json").write_text(json.dumps(rows) + "\n")
    pbtxt = f"  PB valid {summ['pb_valid']:.3f}" if summ.get("pb_valid") is not None else ""
    print(f"\n{a.method}: R_g MAE {summ['rg_mae']:.4f} +- {summ['rg_mae_sem']:.4f} "
          f"over {len(rows)} molecules (target +{a.k_sd} sd){pbtxt}")
    print(f"wrote {(d / 'summary.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
