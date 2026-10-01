"""Does the multiMFM posterior sampler match GLASS if you give it more steps?

The Fig. 1 panels the paper shipped used GLASS with 8 Euler steps per posterior
sample (``...glass8...``, see docs/provenance/toy/README.md).  The recovered MFM
notebook used **one** composed flow-map step (``n_mfm_steps=1``), and the authors
of the original session concluded from that single-step run that "MFM guidance is
slightly weaker or noisier than the original GLASS guidance" and compensated by
raising the colour guidance ratio from 2 to 6.  The molecular and DNA sides of the
paper use 4-step students, so the natural question is whether the toy gap is a
step-count artefact rather than a property of the method.

This module runs the *same* guidance configuration with different posterior
samplers and reports the metrics side by side:

    GLASS with n_glass_steps in {8, 32, ...}     (the reference)
    MFM   with n_mfm_steps  in {1, 2, 4, 8, ...}

It also reports **per-target-region** colour accuracy for Norway.  Overall match
hides the effect: on the shipped panels the 1-step MFM beats GLASS on the two
large regions (blue 0.790 vs 0.729, red 0.798 vs 0.752) and loses badly on the
thin white border (0.672 vs 0.759) -- which is exactly the "blurrier cross" the
eye sees.  White accuracy is therefore the statistic to watch.

A second knob matters here.  The distilled student was trained with
``s_max = t_cond_max = 0.95`` (both stages), while the figure's GLASS run guided
out to t = 0.98 and the posterior map is always integrated to u = 1.0.  Both are
outside the training window, so ``--guidance_ends 0.95 0.98`` separates
"not enough steps" from "asked outside the trained range"; a student retrained
with ``--t_cond_max 0.99 --s_max 0.99`` removes the second.

    python -m toy.posterior_sweep --task norway --n 4000 --out results/toy/posterior_sweep_norway.json
    python -m toy.posterior_sweep --task smiley --n 2000 --out results/toy/posterior_sweep_smiley.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import guidance as G
from .data import NUM_CLASSES
from .mfm_value_fn import load_mfm
from .paths import BASE_CKPT, MFM_CONSISTENCY_CKPT, RESULTS
from .reward import norway_target
from .reward_smiley import SMILEY_SHAPE, smiley_target
from .sample import load_model
from .value_fn import freeze_model

REGION_NAMES = ("blue", "white", "red")  # model colour order (B, W, R)


def per_region(x: torch.Tensor, target: torch.Tensor) -> dict:
    """Colour accuracy split by which region the sample landed in."""
    pred = x[:, :NUM_CLASSES].argmax(dim=-1).cpu()
    target = target.cpu()
    out = {}
    for c, name in enumerate(REGION_NAMES):
        m = target == c
        out[name] = {"n": int(m.sum()),
                     "accuracy": float((pred[m] == target[m]).float().mean()) if m.any() else float("nan")}
    return out


def norway_rows(base, mfm, args) -> list:
    cfg = dict(G.NORWAY_GLASS)  # the configuration behind the published panel
    reward = G.norway_reward_fn(beta=cfg["beta"], k=cfg["k"])
    rows = []
    for gend in args.guidance_ends:
        for backend, steps in ([("glass", s) for s in args.glass_steps]
                               + [("mfm", s) for s in args.mfm_steps]):
            label = f"{backend}-{steps}  window[0.8,{gend}]"
            print(f"\n=== {label} ===", flush=True)
            t0 = time.time()
            common = dict(n_samples=args.n, n_steps=cfg["n_steps"], reward_fn=reward, device=args.device,
                          seed=cfg["seed"], color_base_freeze_until=cfg["color_base_freeze_until"],
                          guidance_ratio=args.ratio if args.ratio is not None else cfg["guidance_ratio"],
                          n_value_samples=cfg["n_value_samples"], guidance_start=cfg["guidance_start"],
                          guidance_end=gend, guidance_batch_size=cfg["guidance_batch_size"],
                          base_batch_size=cfg["base_batch_size"], verbose=False, k=cfg["k"])
            if backend == "glass":
                x, _ = G.delayed_color_sample_glass(base, n_glass_steps=steps, **common)
            else:
                x, _ = G.delayed_color_sample_mfm(
                    base, mfm, n_mfm_steps=steps, mfm_final_u=cfg.get("mfm_final_u", 1.0), **common)
            dt = time.time() - t0
            m = G.summarize_norway(label, x, k=cfg["k"], verbose=False)
            m["per_region"] = per_region(x, norway_target(x[:, NUM_CLASSES:].cpu(), k=cfg["k"]))
            m.update(backend=backend, posterior_steps=steps, guidance_end=gend,
                     guidance_ratio=common["guidance_ratio"], runtime_s=dt)
            rows.append(m)
            pr = m["per_region"]
            print(f"  reward={m['mean_reward']:.4f} match={m['match']:.4f} "
                  f"B={pr['blue']['accuracy']:.4f} W={pr['white']['accuracy']:.4f} "
                  f"R={pr['red']['accuracy']:.4f}  {dt:.0f}s", flush=True)
    return rows


def smiley_rows(base, mfm, args) -> list:
    cfg = dict(G.SMILEY_GLASS)
    shape = dict(SMILEY_SHAPE)
    beta = args.smiley_beta if args.smiley_beta is not None else cfg["beta"]
    reward = G.smiley_reward_fn(beta=beta, shape=shape)
    rows = []
    for backend, steps in ([("glass", s) for s in args.glass_steps]
                           + [("mfm", s) for s in args.mfm_steps]):
        label = f"{backend}-{steps}  beta={beta}"
        print(f"\n=== smiley {label} ===", flush=True)
        t0 = time.time()
        common = dict(n_samples=args.n, n_steps=cfg["n_steps"], reward_fn=reward, device=args.device,
                      seed=cfg["seed"], pos_lambda=cfg["pos_lambda"], pos_start=cfg["pos_start"],
                      pos_end=cfg["pos_end"], pos_lambda_base=cfg["pos_lambda_base"],
                      color_base_freeze_until=cfg["color_base_freeze_until"],
                      color_guidance_ratio=cfg["color_guidance_ratio"],
                      color_guidance_start=cfg["color_guidance_start"],
                      color_guidance_end=cfg["color_guidance_end"],
                      n_value_samples=cfg["n_value_samples"], guidance_batch_size=cfg["guidance_batch_size"],
                      base_batch_size=cfg["base_batch_size"], verbose=False, shape=shape)
        if backend == "glass":
            x, _ = G.sample_hybrid_glass(base, n_glass_steps=steps, **common)
        else:
            x, _ = G.sample_hybrid_mfm(base, mfm, n_mfm_steps=steps, mfm_final_u=1.0, **common)
        dt = time.time() - t0
        m = G.summarize_smiley(label, x, shape, verbose=False)
        tgt, _ = smiley_target(x[:, NUM_CLASSES:].cpu(), **shape)
        m["per_region"] = per_region(x, tgt)
        m.update(backend=backend, posterior_steps=steps, beta=beta, runtime_s=dt)
        rows.append(m)
        pr = m["per_region"]
        print(f"  reward={m['mean_reward']:.4f} match={m['match']:.4f} cov={m['feature_coverage']:.4f} "
              f"B={pr['blue']['accuracy']:.4f} W={pr['white']['accuracy']:.4f} "
              f"R={pr['red']['accuracy']:.4f}  {dt:.0f}s", flush=True)
    return rows


def run(args) -> dict:
    base = freeze_model(load_model(args.base_ckpt, device=args.device, use_ema=True))
    mfm = load_mfm(args.mfm_ckpt, device=args.device, use_ema=True)
    ck = torch.load(args.mfm_ckpt, map_location="cpu", weights_only=False)
    record = {"mfm_ckpt": str(args.mfm_ckpt), "base_ckpt": str(args.base_ckpt),
              "mfm_training_config": ck.get("training_config", {}),
              "n": args.n, "glass_steps": args.glass_steps, "mfm_steps": args.mfm_steps,
              "device": args.device, "rows": {}}
    if args.task in ("norway", "both"):
        record["rows"]["norway"] = norway_rows(base, mfm, args)
    if args.task in ("smiley", "both"):
        record["rows"]["smiley"] = smiley_rows(base, mfm, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"\nsaved {out}")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="norway", choices=["norway", "smiley", "both"])
    p.add_argument("--base_ckpt", default=str(BASE_CKPT))
    p.add_argument("--mfm_ckpt", default=str(MFM_CONSISTENCY_CKPT))
    p.add_argument("--glass_steps", type=int, nargs="*", default=[8, 32])
    p.add_argument("--mfm_steps", type=int, nargs="*", default=[1, 2, 4, 8])
    p.add_argument("--guidance_ends", type=float, nargs="+", default=[0.95, 0.98],
                   help="Norway only: the end of the colour-guidance window")
    p.add_argument("--ratio", type=float, default=None, help="override the colour guidance ratio")
    p.add_argument("--smiley_beta", type=float, default=None)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--out", default=str(RESULTS / "posterior_sweep.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__ == "__main__":
    main()
