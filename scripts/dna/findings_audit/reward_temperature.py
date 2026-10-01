#!/usr/bin/env python
"""Audit of the C0 reward temperature (plan S7, "disclosure gap").

Eq (17)/(48) of the paper state ``r(x) = -(f_cyc(x) - y*)^2``. Every DNA C0 run instead
exponentiates ``reward_scale * -0.5 ((f - y*)/reward_sigma)^2``, i.e. ``-11.111 (f - y*)^2`` at
``reward_sigma 0.15, reward_scale 0.5`` (Table 1, Fig 3, Fig 7, Fig 8, the parity runs) and
``-22.222 (f - y*)^2`` at ``reward_scale 1.0`` (the June Tables 4-6 sets, Fig 4 panel A).

The question is whether the constant is only undisclosed or whether it carries a conclusion. It
enters in two independent ways:

1. **Scale.** ``grad log r = -2 c (f - y*) grad f``, so the guidance step under the stock
   clip-down rule is proportional to ``c`` whenever the gradient is below ``grad_clip`` -- which is
   the GLASS-4 regime (mean norm ~3 against clip 10, clipped at ~5% of steps). The literal Eq (17)
   would therefore take steps ~11x smaller. That is measured by ``--mode sample`` of
   ``posterior_accuracy_probe.py`` at ``--reward_scale 0.045`` (= coefficient exactly 1).
2. **Shape.** ``c`` is the inverse temperature of the finite-N value
   ``V = log (1/N) sum_j exp(c * -(f_j - y*)^2)``, so it sets how concentrated the N softmax
   weights are. A sharp reward makes the estimator depend on one or two draws, which is what the
   MC-convergence analysis (Fig 5, Tables 14-15, 18) is about -- but that analysis uses a
   *different* reward family: ``-reward_beta (z - z_target)^2`` on a standardised gc / motif /
   conjunction score with ``reward_beta = 1, z_target = 1``.

This script measures (2) directly: the effective sample size ``ESS = 1 / sum_j w_j^2`` of the MC
weights at the paper's operating point N = 8, for the C0 reward at each candidate coefficient and
for the gc reward of the MC-convergence tables, on the same probe states and the same posterior.

    python scripts/dna/findings_audit/reward_temperature.py --n_probes 8 --mc 8 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from dmfm import api  # noqa: E402
from dmfm.experiments.ablate_glass_gradient_mc import reward_score  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--n_probes", type=int, default=8)
    p.add_argument("--probe_times", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--mc", type=int, nargs="+", default=[8, 64])
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scales", type=float, nargs="+", default=[0.045, 0.5, 1.0])
    p.add_argument("--targets", type=float, nargs="+", default=[1.0, -1.0])
    p.add_argument("--posterior_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nfe_traj", type=int, default=64)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--n_calib", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=str(REPO / "results" / "dna" / "findings_audit" / "reward_temperature"))
    return p.parse_args(argv)


def ess_stats(log_r: torch.Tensor) -> dict:
    """Softmax-weight concentration of a [B, MC] log-reward matrix."""
    w = torch.softmax(log_r, dim=1)
    mc = log_r.shape[1]
    ess = 1.0 / (w * w).sum(dim=1)
    return {
        "ess_mean": float(ess.mean()), "ess_min": float(ess.min()), "ess_median": float(ess.median()),
        "ess_frac_of_mc": float(ess.mean()) / mc,
        "max_weight_mean": float(w.max(dim=1).values.mean()),
        "log_r_spread_mean": float((log_r.max(dim=1).values - log_r.min(dim=1).values).mean()),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base, cfg = api.load_base(args.length, device)
    guide = api.load_c0("guide", device)
    L = int(cfg.seq_len)
    post = api.glass_posterior_fn(base, n_steps=args.posterior_steps, end_time=1.0)

    ids = list(range(args.n_probes))
    x0 = api.paired_initial_noise(ids, L, seed=args.seed, device=device)
    max_mc = max(args.mc)
    pool = api.paired_eps_pool(ids, L, mc=max_mc, seed=args.seed, device=device)

    # Unconditional C0 statistics and the gc-score statistics of the MC-convergence tables, so the
    # two reward families can be put on one axis (sharpness in standard deviations of the score).
    with torch.no_grad():
        cal_ids = list(range(1000, 1000 + args.n_calib))
        xc = api.sample_unguided(base, cfg, api.paired_initial_noise(cal_ids, L, seed=args.seed, device=device),
                                 nfe=args.nfe_traj, t_max=0.999)
        hard = torch.nn.functional.one_hot(xc.argmax(-1), int(cfg.alphabet_size)).float()
        c0 = guide(hard)
        gc = reward_score(hard, reward="gc", motif=torch.tensor([0], device=device),
                          motif2=torch.tensor([0], device=device), motif_tau=0.1, conjunction_tau=0.1)
    calib = {"n_calib": args.n_calib,
             "c0_mean": float(c0.mean()), "c0_std": float(c0.std()),
             "gc_mean": float(gc.mean()), "gc_std": float(gc.std())}

    rows = []
    for t in args.probe_times:
        nfe = max(1, int(round(t / (args.t_max / args.nfe_traj))))
        x_t = api.sample_unguided(base, cfg, x0, nfe=nfe, t_max=float(t))
        t_vec = torch.full((x_t.shape[0],), float(t), device=device)
        # One posterior endpoint per (probe, MC draw); shared by every reward below.
        with torch.no_grad():
            eps = pool.reshape(len(ids) * max_mc, L, int(cfg.alphabet_size))
            xr = x_t.repeat_interleave(max_mc, 0)
            tr = t_vec.repeat_interleave(max_mc)
            ep = post(eps, xr, tr)
            f = guide(ep).view(len(ids), max_mc)
            gc_s = reward_score(ep, reward="gc", motif=torch.tensor([0], device=device),
                                motif2=torch.tensor([0], device=device), motif_tau=0.1,
                                conjunction_tau=0.1).view(len(ids), max_mc)
        for mc in args.mc:
            fm = f[:, :mc]
            for target in args.targets:
                for rs in args.reward_scales:
                    coeff = 0.5 * float(rs) / float(args.reward_sigma) ** 2
                    log_r = -coeff * (fm - float(target)) ** 2
                    rows.append({
                        "reward": "c0", "t": t, "mc": mc, "target": target,
                        "reward_scale": rs, "coefficient": coeff,
                        "sharpness_in_score_sd": coeff * calib["c0_std"] ** 2,
                        "target_in_score_sd": (float(target) - calib["c0_mean"]) / calib["c0_std"],
                        "value_mean": float((torch.logsumexp(log_r, 1) - math.log(mc)).mean()),
                        **ess_stats(log_r),
                    })
            # The MC-convergence tables' reward, on the same states and the same draws.
            z = (gc_s[:, :mc] - calib["gc_mean"]) / calib["gc_std"]
            log_r = -1.0 * (z - 1.0) ** 2
            rows.append({
                "reward": "gc(z_target=1, beta=1)", "t": t, "mc": mc, "target": 1.0,
                "reward_scale": 1.0, "coefficient": 1.0, "sharpness_in_score_sd": 1.0,
                "target_in_score_sd": 1.0,
                "value_mean": float((torch.logsumexp(log_r, 1) - math.log(mc)).mean()),
                **ess_stats(log_r),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "reward_temperature_ess.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(
        {"args": vars(args), "calibration": calib,
         "device": str(device),
         "_what": "ESS of the finite-N value's softmax weights vs the reward coefficient"},
        indent=2))
    piv = (df[df.mc == min(args.mc)]
           .groupby(["reward", "coefficient", "target"])[["ess_mean", "max_weight_mean", "value_mean"]]
           .mean().reset_index())
    print(json.dumps(calib, indent=1))
    print(piv.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
