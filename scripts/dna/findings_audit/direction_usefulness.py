#!/usr/bin/env python
"""Pre-registered one-step test: is a posterior's value-gradient *direction* useful for steering?

Item (c) of `docs/dna_findings_audit.md`. Full guided sampling with an accurate posterior needs a
GPU; this is the cheap, single-step version of the same question, and it is registered here before
the numbers exist so the reading cannot be chosen afterwards.

For a probe state `x_t` on the base flow and a posterior `P`:

1. `g = grad_x V_P(x_t)` at the Table 1 reward and MC = 8, `u = g / ||g||`;
2. displace `x_t` by `delta * u`, by `-delta * u` (sign control) and by `delta * r` for random unit
   `r` (magnitude control), at several `delta`;
3. integrate the **base** flow (no further guidance) from `t` to `t_max = 0.95`, decode `argmax`, and
   score the hard sequence with the guide;
4. report the improvement in `|f - y*|` against the undisplaced continuation.

`delta = 1.484` is the displacement one guided step actually takes under the scale-free rule at
t = 0.5 (`dt * min(guidance_frac sigma_t^2, coeff_cap) * grad_clip`
`= (0.95/64) * 10 * 10`), so it is the operating point; the others bracket it.

**Registered readings.** A posterior's direction is useful iff `+u` improves `|f - y*|` more than
`-u` *and* more than a random direction, at the operating `delta`. If that holds for the accurate
posteriors (`euler32` to 0.999, `rk4-25` to 0.999) as well as for the Table 1 estimator, then an
accurate posterior does carry a usable steering direction and the S7-item-9 claim is wrong as
stated. If it holds only for the coarse estimator, the claim survives this test and the mechanism is
about direction, not about the strength constant.

    python scripts/dna/findings_audit/direction_usefulness.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from dmfm import api  # noqa: E402
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step  # noqa: E402

POSTERIORS = {
    "glass_euler4": dict(n_steps=4, end_time=1.0, solver="euler"),
    "glass_euler32": dict(n_steps=32, end_time=1.0, solver="euler"),
    "glass_euler32_e999": dict(n_steps=32, end_time=0.999, solver="euler"),
    "glass_rk4_25_e999": dict(n_steps=25, end_time=0.999, solver="rk4"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--n_probes", type=int, default=8)
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--probe_times", type=float, nargs="+", default=[0.5])
    p.add_argument("--targets", type=float, nargs="+", default=[1.0, -1.0])
    p.add_argument("--deltas", type=float, nargs="+", default=[0.1484, 1.484, 14.84])
    p.add_argument("--n_random", type=int, default=2)
    p.add_argument("--configs", nargs="+", default=list(POSTERIORS))
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scale", type=float, default=0.5)
    p.add_argument("--nfe_traj", type=int, default=64)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=str(REPO / "results" / "dna" / "findings_audit" / "direction_usefulness"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base, cfg = api.load_base(args.length, dev)
    guide = api.load_c0("guide", dev)
    L, K = int(cfg.seq_len), int(cfg.alphabet_size)
    ids = list(range(args.n_probes))
    x0 = api.paired_initial_noise(ids, L, seed=args.seed, device=dev)
    pool = api.paired_eps_pool(ids, L, mc=args.mc, seed=args.seed, device=dev)
    grid = torch.linspace(0, float(args.t_max), int(args.nfe_traj) + 1, device=dev)

    @torch.no_grad()
    def continue_and_score(x, t_from):
        """Integrate the base flow from t_from to t_max, decode argmax, score with the guide."""
        st = x.clone()
        b = st.shape[0]
        for s0, s1 in zip(grid[:-1], grid[1:]):
            if float(s0) < float(t_from) - 1e-9:
                continue
            st, _, _ = gaussian_denoiser_flow_step(cfg, base, st, s0.expand(b), s1.expand(b))
        hard = torch.nn.functional.one_hot(st.argmax(-1), K).float()
        return guide(hard).detach()

    gen = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    rand_dirs = []
    for _ in range(args.n_random):
        r = torch.randn(args.n_probes, L, K, generator=gen).to(dev)
        rand_dirs.append(r / r.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None])

    rows = []
    for t in args.probe_times:
        nfe = max(1, int(round(t / (args.t_max / args.nfe_traj))))
        x_t = api.sample_unguided(base, cfg, x0, nfe=nfe, t_max=float(t))
        f0 = continue_and_score(x_t, t)
        for target in args.targets:
            log_reward = api.c0_log_reward_fn(guide, target, reward_sigma=args.reward_sigma,
                                              reward_scale=args.reward_scale)
            d0 = (f0 - float(target)).abs()
            for name in args.configs:
                post = api.glass_posterior_fn(base, **POSTERIORS[name])
                _, g = api.value_and_grad(post, log_reward, x_t, float(t), pool, mc_chunk=args.mc)
                gn = g.flatten(1).norm(dim=1)
                u = g / gn.clamp_min(1e-12)[:, None, None]
                for delta in args.deltas:
                    variants = {"+grad": u, "-grad": -u}
                    for i, r in enumerate(rand_dirs):
                        variants[f"random{i}"] = r
                    for label, d in variants.items():
                        f = continue_and_score(x_t + float(delta) * d, t)
                        dd = (f - float(target)).abs()
                        rows.append({
                            "t": t, "target": target, "config": name, "delta": delta,
                            "direction": label,
                            "grad_norm_mean": float(gn.mean()),
                            "abs_err_base": float(d0.mean()),
                            "abs_err": float(dd.mean()),
                            "improvement": float((d0 - dd).mean()),
                            "frac_improved": float(((d0 - dd) > 0).float().mean()),
                            "n_probes": args.n_probes,
                        })
                        print(f"  t={t} y*={target:+.0f} {name:20s} delta={delta:<7g} {label:8s} "
                              f"err {float(d0.mean()):.4f} -> {float(dd.mean()):.4f} "
                              f"(improve {rows[-1]['improvement']:+.4f}, "
                              f"{rows[-1]['frac_improved']:.2f} of probes)", flush=True)
                    pd.DataFrame(rows).to_csv(out / "direction_usefulness.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "direction_usefulness.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "device": str(dev)}, indent=2))
    print("\n== mean improvement in |f - y*| by (config, delta, direction) ==")
    print(df.pivot_table(index=["config", "delta"], columns="direction", values="improvement").to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
