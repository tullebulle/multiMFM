#!/usr/bin/env python
"""Is the well-integrated posterior's value function locally smooth, or piecewise constant?

Item (c) of `docs/dna_findings_audit.md`. The published claim is that an accurate posterior "does not
guide at all". Two competing mechanisms:

* *artifact*: the one arm that was run (`glass_euler32`, 32 Euler steps to the singular data time
  t = 1) has a saturated endpoint and was given a guidance strength 5,000x below every other arm,
  so it measured nothing; and
* *real*: a well-integrated DFM posterior at t in [0.5, 0.95] maps x_t to a nearly one-hot endpoint,
  so V(x_t) = log E exp(r(endpoint)) is nearly piecewise constant and its gradient carries no
  direction, however it is rescaled.

This distinguishes them without autodiff and without a GPU: central finite differences of V along
random unit directions at a ladder of step sizes h. A smooth V gives a *plateau* -- |dV/dh| tends to
a constant |grad V . u| as h -> 0. A piecewise-constant V gives a ratio that *falls to zero* with h
(the probability of crossing a decision boundary is proportional to h).

    python scripts/dna/findings_audit/value_smoothness.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from dmfm import api  # noqa: E402

POSTERIORS = {
    "glass_euler4": dict(n_steps=4, end_time=1.0, solver="euler"),
    "glass_euler32": dict(n_steps=32, end_time=1.0, solver="euler"),
    "glass_euler32_e999": dict(n_steps=32, end_time=0.999, solver="euler"),
    "glass_rk4_25_e999": dict(n_steps=25, end_time=0.999, solver="rk4"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--n_probes", type=int, default=4)
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--probe_times", type=float, nargs="+", default=[0.5, 0.7])
    p.add_argument("--targets", type=float, nargs="+", default=[1.0, -1.0])
    p.add_argument("--n_dirs", type=int, default=3)
    p.add_argument("--rel_h", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scale", type=float, default=0.5)
    p.add_argument("--nfe_traj", type=int, default=64)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=str(REPO / "results" / "dna" / "findings_audit" / "value_smoothness"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base, cfg = api.load_base(args.length, dev)
    guide = api.load_c0("guide", dev)
    L, K = int(cfg.seq_len), int(cfg.alphabet_size)
    ids = list(range(args.n_probes))
    x0 = api.paired_initial_noise(ids, L, seed=args.seed, device=dev)
    pool = api.paired_eps_pool(ids, L, mc=args.mc, seed=args.seed, device=dev)

    gen = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    dirs = []
    for _ in range(args.n_dirs):
        u = torch.randn(args.n_probes, L, K, generator=gen).to(dev)
        u = u / u.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None]
        dirs.append(u)

    rows = []
    for t in args.probe_times:
        nfe = max(1, int(round(t / (args.t_max / args.nfe_traj))))
        x_t = api.sample_unguided(base, cfg, x0, nfe=nfe, t_max=float(t))
        xnorm = x_t.flatten(1).norm(dim=1)
        for target in args.targets:
            log_reward = api.c0_log_reward_fn(guide, target, reward_sigma=args.reward_sigma,
                                              reward_scale=args.reward_scale)

            def value(x, post):
                with torch.no_grad():
                    eps = pool.reshape(args.n_probes * args.mc, L, K)
                    xr = x.repeat_interleave(args.mc, 0)
                    tr = torch.full((args.n_probes * args.mc,), float(t), device=dev)
                    lr = log_reward(post(eps, xr, tr)).view(args.n_probes, args.mc)
                    return torch.logsumexp(lr, 1) - math.log(args.mc)

            for name, kw in POSTERIORS.items():
                post = api.glass_posterior_fn(base, **kw)
                v0 = value(x_t, post)
                for rel_h in args.rel_h:
                    step = (rel_h * xnorm)[:, None, None]
                    slopes, jumps = [], []
                    for u in dirs:
                        vp = value(x_t + step * u, post)
                        vm = value(x_t - step * u, post)
                        h = (rel_h * xnorm)
                        slopes.append(((vp - vm) / (2 * h)).abs())
                        jumps.append((vp - v0).abs() + (vm - v0).abs())
                    sl = torch.stack(slopes)
                    jp = torch.stack(jumps)
                    rows.append({
                        "t": t, "target": target, "config": name,
                        "n_steps": kw["n_steps"], "end_time": kw["end_time"], "solver": kw["solver"],
                        "rel_h": rel_h,
                        "value_mean": float(v0.mean()),
                        "central_slope_mean": float(sl.mean()),
                        "central_slope_median": float(sl.median()),
                        "abs_change_mean": float(jp.mean()),
                        "frac_dirs_value_unchanged": float((jp < 1e-6).float().mean()),
                    })
                    print(f"  t={t} y*={target:+.0f} {name:20s} h/|x|={rel_h:<7g} "
                          f"|dV/dh|={rows[-1]['central_slope_mean']:.4g} "
                          f"|dV|={rows[-1]['abs_change_mean']:.3g} "
                          f"flat_frac={rows[-1]['frac_dirs_value_unchanged']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "value_smoothness.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "device": str(dev)}, indent=2))
    piv = df.pivot_table(index=["config", "t", "target"], columns="rel_h", values="central_slope_mean")
    print("\n== |central difference of V| per unit displacement, by relative step size ==")
    print(piv.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
