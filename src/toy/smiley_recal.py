"""Calibrate the smiley position-guidance strength against the published Fig. 1.

Why this exists.  The smiley panel of the submitted figure comes from
``uniform_smiley_delayed_color_bigger_smile_thick_mouth_N10000_endpoints.pt``
(see docs/provenance/toy/README.md).  Its geometry is recorded, but its reward
scale beta and colour-guidance ratio are not; only the *predecessor* run at
N = 1000 has them in its file name (beta 16, pos_lambda 0.2).  Running the
rebuilt sampler at beta = 16 gives eye/mouth blobs that are visibly tighter than
the published ones, and a higher reward (0.587 vs the logged 0.555), i.e. the
final run used weaker position guidance.

Beta enters the position guidance linearly (the colour guidance is
ratio-normalised, so beta only rotates it), and under the exact tilt the
stationary spread around a Gaussian feature of width sigma is sigma / sqrt(beta).
So beta is the one knob that sets the blob width, and the published figure is
direct evidence for it.

This module measures the blob widths the same way on both figures -- by
rendering the panel with ``toy.figure.four_panel`` at the published geometry and
classifying pixels by colour, exactly as was done on ``figures/norway.pdf`` --
sweeps beta with the (fast) MFM value function, picks the beta whose eye width
matches, then re-runs the two N = 10,000 smiley panels with it and merges them
into an existing ``results/toy/``.

    python -m toy.smiley_recal --out_dir results/toy                 # sweep + rerun + merge
    python -m toy.smiley_recal --out_dir results/toy --sweep_only    # sweep only

Published targets (measured off figures/norway.pdf rendered at 400 dpi, panel 4
mapped to [-1.08, 1.08]^2):

    red eye   centre (-0.284, +0.328)   pixel std (0.127, 0.100)
    blue eye  centre (+0.312, +0.315)   pixel std (0.120, 0.137)
    mouth     centre (-0.040, -0.427)   pixel std (0.472, 0.213)
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import guidance as G
from .figure import four_panel
from .mfm_value_fn import load_mfm
from .paths import BASE_CKPT, MFM_CONSISTENCY_CKPT, RESULTS
from .reward_smiley import SMILEY_SHAPE
from .sample import load_model
from .value_fn import freeze_model

# panel 4 of the published render, in the 1442 x 469 px reference frame
PANEL_BOX = (1131, 140, 1400, 409)
LIM = 1.08
# Regions the three features are measured in, so that stray points of the same
# colour elsewhere in the panel do not inflate the widths.
WINDOWS = {
    "red": (-0.70, 0.10, 0.00, 0.70),      # (x_lo, x_hi, y_lo, y_hi)
    "blue": (-0.10, 0.70, 0.00, 0.70),
    "white": (-1.08, 1.08, -1.08, 0.00),
}


def _measure(im: np.ndarray) -> dict:
    """Per-feature pixel centroid / std inside WINDOWS, from a 1442 x 469-frame render."""
    H, W, _ = im.shape
    sx, sy = W / 1442, H / 469
    p = im[int(PANEL_BOX[1] * sy):int(PANEL_BOX[3] * sy), int(PANEL_BOX[0] * sx):int(PANEL_BOX[2] * sx)]
    h, w, _ = p.shape
    X = -LIM + 2 * LIM * (np.arange(w) + 0.5) / w
    Y = LIM - 2 * LIM * (np.arange(h) + 0.5) / h
    XX, YY = np.meshgrid(X, Y)
    R, Gc, B = p[..., 0], p[..., 1], p[..., 2]
    masks = {
        "red": (R > 120) & (R > Gc + 40) & (R > B + 40),
        "blue": (B > 100) & (B > R + 40) & (B > Gc + 20),
        "white": (R > 170) & (Gc > 170) & (B > 170),
    }
    out = {}
    for name, m in masks.items():
        xlo, xhi, ylo, yhi = WINDOWS[name]
        m = m & (XX >= xlo) & (XX <= xhi) & (YY >= ylo) & (YY <= yhi)
        if m.sum() < 10:
            out[name] = None
            continue
        out[name] = {"centre": [float(XX[m].mean()), float(YY[m].mean())],
                     "std": [float(XX[m].std()), float(YY[m].std())],
                     "pixels": int(m.sum())}
    return out


def measure_image(path) -> dict:
    """Measure a rendered four-panel figure (e.g. the published figures/norway.pdf)."""
    from PIL import Image

    return _measure(np.asarray(Image.open(path).convert("RGB")).astype(float))


def blob_stats(x, shape=None) -> dict:
    """Render ``x`` as the smiley panel and measure it the same way as the published figure."""
    import matplotlib.pyplot as plt
    from PIL import Image

    dummy = np.zeros((1, 5), dtype=np.float32)
    dummy[0, 0] = 1.0
    fig = four_panel({"prior": dummy, "base": dummy, "norway": dummy, "smiley": np.asarray(x)},
                     smiley_score_kwargs=dict(SMILEY_SHAPE if shape is None else shape))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)  # -> 1442 x 469, the reference frame
    plt.close(fig)
    buf.seek(0)
    im = np.asarray(Image.open(buf).convert("RGB")).astype(float)
    return _measure(im)


# Measured with ``measure_image`` on figures/norway.pdf rendered at 400 dpi.
PUBLISHED_BLOBS = {
    "red": {"centre": [-0.286, 0.332], "std": [0.121, 0.088]},
    "blue": {"centre": [0.312, 0.334], "std": [0.104, 0.084]},
    "white": {"centre": [-0.010, -0.455], "std": [0.444, 0.126]},
}


def eye_width(stats: dict) -> float:
    """Mean of the two eyes' horizontal pixel std -- the quantity beta controls."""
    vals = [stats[c]["std"][0] for c in ("red", "blue") if stats.get(c)]
    return float(np.mean(vals)) if vals else float("nan")


PUBLISHED_EYE_WIDTH = eye_width(PUBLISHED_BLOBS)


def run(args):
    device = args.device
    out_dir = Path(args.out_dir)
    base = freeze_model(load_model(args.base_ckpt, device=device, use_ema=True))
    mfm = load_mfm(args.mfm_ckpt, device=device, use_ema=True)
    shape = dict(SMILEY_SHAPE)

    def sample_mfm(beta, n, n_steps=None, mc=None):
        cfg = dict(G.SMILEY_MFM)
        if mc is not None:
            cfg["n_value_samples"] = mc
        return G.sample_hybrid_mfm(
            base, mfm, n_samples=n, n_steps=n_steps or cfg["n_steps"],
            reward_fn=G.smiley_reward_fn(beta=beta, shape=shape), device=device, seed=cfg["seed"],
            pos_lambda=cfg["pos_lambda"], pos_start=cfg["pos_start"], pos_end=cfg["pos_end"],
            pos_lambda_base=cfg["pos_lambda_base"], color_base_freeze_until=cfg["color_base_freeze_until"],
            color_guidance_ratio=cfg["color_guidance_ratio"], color_guidance_start=cfg["color_guidance_start"],
            color_guidance_end=cfg["color_guidance_end"], n_value_samples=cfg["n_value_samples"],
            n_mfm_steps=cfg["n_mfm_steps"], mfm_final_u=cfg["mfm_final_u"],
            guidance_batch_size=cfg["guidance_batch_size"], base_batch_size=cfg["base_batch_size"],
            verbose=False, shape=shape)

    sweep = []
    for beta in args.betas:
        t0 = time.time()
        x, _ = sample_mfm(beta, args.sweep_n, n_steps=args.sweep_steps, mc=args.sweep_mc)
        st = blob_stats(x.cpu(), shape)
        m = G.summarize_smiley(f"smiley MFM beta={beta} (N={args.sweep_n})", x, shape, verbose=False)
        row = {"beta": beta, "n": args.sweep_n, "n_steps": args.sweep_steps or G.SMILEY_MFM["n_steps"],
               "n_value_samples": args.sweep_mc or G.SMILEY_MFM["n_value_samples"], "eye_width": eye_width(st),
               "eye_width_target": PUBLISHED_EYE_WIDTH, "blobs": st,
               "mean_reward": m["mean_reward"], "match": m["match"],
               "feature_coverage": m["feature_coverage"], "runtime_s": time.time() - t0}
        sweep.append(row)
        print(f"  beta={beta:<6} eye_width={row['eye_width']:.4f} (target {PUBLISHED_EYE_WIDTH:.4f})  "
              f"reward={row['mean_reward']:.4f} match={row['match']:.4f}  {row['runtime_s']:.0f}s", flush=True)

    best = min(sweep, key=lambda r: abs(r["eye_width"] - PUBLISHED_EYE_WIDTH))
    print(f"\nclosest to the published eye width: beta={best['beta']}")

    record = {"published_blobs": PUBLISHED_BLOBS, "published_eye_width": PUBLISHED_EYE_WIDTH,
              "sweep": sweep, "selected_beta": best["beta"],
              "panels_use_beta": G.SMILEY_GLASS["beta"] if args.sweep_only else best["beta"],
              "shape": shape, "sweep_config": {k: v for k, v in G.SMILEY_MFM.items() if k != "beta"},
              "note": ("beta of the final N=10000 GLASS smiley was never recorded; 16 comes from its "
                       "N=1000 predecessor's file name.  Selected by matching the published figure's "
                       "eye width, measured the same way on both rasters.")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smiley_beta_calibration.json").write_text(json.dumps(record, indent=1))
    print(f"saved {out_dir}/smiley_beta_calibration.json")
    if args.sweep_only:
        return record

    # Re-run the two full-size smiley panels at the selected beta and merge them in.
    beta = best["beta"]
    panels = dict(np.load(out_dir / "fig1_panels.npz"))
    metrics = json.loads((out_dir / "steering_metrics.json").read_text())

    t0 = time.time()
    x, _ = sample_mfm(beta, args.panel_n)
    panels["smiley_mfm_panel"] = x.cpu().numpy().astype(np.float32)
    metrics["metrics"]["smiley_mfm_panel"] = G.summarize_smiley(
        f"Smiley MFM-guided (N={args.panel_n}, beta={beta})", x, shape)
    metrics["runtimes_s"]["smiley_mfm_panel"] = time.time() - t0

    cfg = dict(G.SMILEY_GLASS)
    t0 = time.time()
    x, _ = G.sample_hybrid_glass(
        base, n_samples=args.panel_n, n_steps=cfg["n_steps"],
        reward_fn=G.smiley_reward_fn(beta=beta, shape=shape), device=device, seed=cfg["seed"],
        pos_lambda=cfg["pos_lambda"], pos_start=cfg["pos_start"], pos_end=cfg["pos_end"],
        pos_lambda_base=cfg["pos_lambda_base"], color_base_freeze_until=cfg["color_base_freeze_until"],
        color_guidance_ratio=cfg["color_guidance_ratio"], color_guidance_start=cfg["color_guidance_start"],
        color_guidance_end=cfg["color_guidance_end"], n_value_samples=cfg["n_value_samples"],
        n_glass_steps=cfg["n_glass_steps"], guidance_batch_size=cfg["guidance_batch_size"],
        base_batch_size=cfg["base_batch_size"], verbose=True, shape=shape)
    panels["smiley_glass"] = x.cpu().numpy().astype(np.float32)
    metrics["metrics"]["smiley_glass"] = G.summarize_smiley(
        f"Smiley GLASS-guided (N={args.panel_n}, beta={beta})", x, shape)
    metrics["runtimes_s"]["smiley_glass"] = time.time() - t0

    metrics["configs"]["smiley_mfm"]["beta"] = beta
    metrics["configs"]["smiley_glass"]["beta"] = beta
    metrics["configs"]["smiley_beta_calibrated"] = True
    np.savez_compressed(out_dir / "fig1_panels.npz", **panels)
    (out_dir / "steering_metrics.json").write_text(json.dumps(metrics, indent=1))

    from .fig1 import draw
    draw(out_dir, panels, shape, ("glass", "mfm"))
    print(f"merged the recalibrated smiley panels into {out_dir}")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_ckpt", default=str(BASE_CKPT))
    p.add_argument("--mfm_ckpt", default=str(MFM_CONSISTENCY_CKPT))
    p.add_argument("--betas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0])
    p.add_argument("--sweep_n", type=int, default=2000)
    p.add_argument("--sweep_steps", type=int, default=None, help="outer ODE steps during the sweep")
    p.add_argument("--sweep_mc", type=int, default=None, help="posterior MC samples during the sweep")
    p.add_argument("--panel_n", type=int, default=10_000)
    p.add_argument("--sweep_only", action="store_true")
    p.add_argument("--out_dir", default=str(RESULTS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__ == "__main__":
    main()
