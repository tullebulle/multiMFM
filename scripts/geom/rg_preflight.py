#!/usr/bin/env python
"""Is radius of gyration a viable GEOM steering target? Measure before committing.

    python scripts/geom/rg_preflight.py --n 512

The fingerprint task failed for reasons that were knowable in advance, so this checks the
replacement first. It samples unguided molecules from the GEOM teacher and reports:

* the distribution of the radius of gyration R_g -- its mean and spread set where a target
  is "ordinary" and where it is rare, which is the whole question for a steering benchmark;
* PoseBusters validity of those same molecules, the baseline any steered number is read
  against;
* the headroom best-of-N can reach, i.e. how close independent sampling gets to targets at
  +1, +2 and +3 standard deviations. Steering has to beat that to be worth anything, and
  where best-of-N already lands on target there is nothing to win.

R_g is exactly computable and exactly differentiable from the coordinates, so unlike the
fingerprint task there is no learned guide between the claim and the evidence, and no
guide-oracle gap to saturate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/geom/rg_preflight.json")
    a = p.parse_args()

    from multimfm.geom.paths import DEFAULT_TEACHER
    from multimfm.geom.teacher import load_geom_teacher
    from multimfm.glass_guidance import production_step_with_optional_guidance, radius_gyration

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats, _ = load_geom_teacher(DEFAULT_TEACHER, dev)
    torch.manual_seed(a.seed)

    rgs, mols = [], []
    done = 0
    while done < a.n:
        b = min(a.batch, a.n - done)
        st = model._sample_noise_like_batch(batch_size=b).to(dev)
        sched = model._get_sample_schedule(a.sample_steps).to(dev)
        with torch.no_grad():
            for i in range(1, len(sched)):
                t = sched[i - 1].expand(b)
                dt = (sched[i] - sched[i - 1]).expand(b)
                st = production_step_with_optional_guidance(
                    model, st, t, dt, mu=0.0, grad_coords=None, grad_atomics=None)
            rg = radius_gyration(st["coords"], st["padding_mask"])
        rgs.append(rg.detach().float().cpu())
        mols.append((~st["padding_mask"]).sum(dim=1).detach().cpu())
        done += b
        print(f"  sampled {done}/{a.n}", flush=True)

    rg = torch.cat(rgs).numpy()
    natoms = torch.cat(mols).numpy()
    mu, sd = float(rg.mean()), float(rg.std())
    out = {
        "n": int(rg.size), "sample_steps": a.sample_steps,
        "rg_mean": mu, "rg_std": sd,
        "rg_quantiles": {q: float(np.quantile(rg, float(q)))
                         for q in ("0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99")},
    }

    # How close does plain sampling get to a tail target? This is the bar steering must clear.
    bar = {}
    for k in (1, 2, 3):
        tgt = mu + k * sd
        err = np.abs(rg - tgt)
        bar[f"+{k}sd"] = {
            "target": tgt,
            "best_of_1_mean_abs_err": float(err.mean()),
            **{f"best_of_{N}": float(np.mean([err[i:i + N].min()
                                              for i in range(0, len(err) - N + 1, N)]))
               for N in (4, 16, 64) if len(err) >= N},
        }
    out["tail_targets"] = bar

    # R_g scales with molecule size, so a single global target is a size filter rather than a
    # property target. Record the size-conditional statistics the benchmark draws targets from.
    by_size = {}
    for n in sorted(set(int(x) for x in natoms)):
        v = rg[natoms == n]
        if v.size >= 8:
            by_size[int(n)] = {"n_molecules": int(v.size), "mean": float(v.mean()),
                               "std": float(v.std())}
    out["by_n_atoms"] = by_size
    out["n_atoms_range"] = [int(natoms.min()), int(natoms.max())]
    out["pooled_vs_conditional"] = {
        "pooled_std": sd,
        "mean_within_size_std": float(np.mean([b["std"] for b in by_size.values()])) if by_size else None,
        "note": "if the within-size spread is much smaller than the pooled one, most of the "
                "pooled spread is molecule size and a global target would mostly select on size",
    }

    f = REPO / a.out
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nR_g over {rg.size} unguided molecules: mean {mu:.3f}, sd {sd:.3f}")
    print(f"  quantiles: {out['rg_quantiles']}")
    for k, v in bar.items():
        row = "  ".join(f"{kk}={vv:.3f}" for kk, vv in v.items() if kk != "target")
        print(f"  target {k} ({v['target']:.3f}): {row}")
    pv = out["pooled_vs_conditional"]
    if pv["mean_within_size_std"]:
        print(f"\n  pooled sd {pv['pooled_std']:.3f} vs mean within-size sd "
              f"{pv['mean_within_size_std']:.3f} over {len(by_size)} sizes "
              f"(atoms {out['n_atoms_range'][0]}-{out['n_atoms_range'][1]})")
    print(f"wrote {f.relative_to(REPO)}")


if __name__ == "__main__":
    main()
