"""Produce the Fig. 1 panels (GLASS- and MFM-guided) and the numbers behind them.

    python -m toy.fig1 --out_dir results/toy                 # full size (GPU)
    python -m toy.fig1 --quick --out_dir outputs/toy/quick   # small n, for notebook reruns

Writes
    <out_dir>/fig1_panels.npz        endpoints of every panel (float32, (N, 5): c one-hot, y)
    <out_dir>/steering_metrics.json  configs, summaries, runtimes, checkpoint sha256
    <out_dir>/fig1_glass.png         four panels with GLASS-guided Norway / smiley
    <out_dir>/fig1_mfm.png           four panels with multiMFM-guided Norway / smiley

Settings: see ``toy.guidance`` (NORWAY_MFM / SMILEY_MFM are the recorded final
MFM settings; NORWAY_GLASS / SMILEY_GLASS are only partly recorded).
"""

from __future__ import annotations

import argparse
import hashlib
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
from .sample import load_model, sample
from .value_fn import freeze_model


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(args) -> dict:
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = freeze_model(load_model(args.base_ckpt, device=device, use_ema=True))
    mfm = load_mfm(args.mfm_ckpt, device=device, use_ema=True)

    nor_mfm = dict(G.NORWAY_MFM)
    nor_glass = dict(G.NORWAY_GLASS)
    smi_mfm = dict(G.SMILEY_MFM)
    smi_glass = dict(G.SMILEY_GLASS)
    n_fig = args.n_panel
    if args.quick:
        # CPU-sized rerun for the notebook: fewer particles, a coarser outer ODE,
        # fewer Monte-Carlo posterior samples and fewer GLASS inner steps.  The
        # panels stay recognisable; the metrics are noisier and biased low.
        n_fig = min(n_fig, 400)
        nor_mfm.update(n_samples=n_fig, n_steps=50, n_value_samples=4)
        nor_glass.update(n_samples=n_fig, n_steps=50, n_value_samples=4, n_glass_steps=4)
        smi_mfm.update(n_samples=min(n_fig, 300), n_steps=60, n_value_samples=4)
        smi_glass.update(n_samples=min(n_fig, 300), n_steps=60, n_value_samples=4, n_glass_steps=4)
    panels, metrics, runtimes = {}, {}, {}

    # p0 and p1 panels
    g = torch.Generator(device=device).manual_seed(0)
    panels["prior"] = torch.randn(n_fig, 5, generator=g, device=device).cpu()
    t0 = time.time()
    panels["base"] = sample(base, n_fig, n_steps=100, device=device, seed=0).cpu()
    runtimes["base"] = time.time() - t0
    metrics["base"] = G.summarize_norway("base (plain sampler)", panels["base"], verbose=True)

    nor_reward = G.norway_reward_fn(beta=nor_mfm["beta"], k=nor_mfm["k"])
    common_nor = dict(n_steps=nor_mfm["n_steps"], reward_fn=nor_reward, device=device,
                      color_base_freeze_until=nor_mfm["color_base_freeze_until"],
                      base_batch_size=nor_mfm["base_batch_size"], k=nor_mfm["k"])

    def timed(name, fn, **kw):
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        x, diag = fn(**kw)
        runtimes[name] = time.time() - t
        print(f"finished in {runtimes[name]:.1f}s", flush=True)
        return x.cpu(), diag

    diags = {}
    x, diags["norway_unguided"] = timed(
        "norway_unguided", G.delayed_color_sample_mfm, base_model=base, mfm=mfm,
        n_samples=nor_mfm["n_samples"], seed=nor_mfm["seed"], guidance_ratio=0.0, **common_nor)
    panels["norway_unguided"] = x
    metrics["norway_unguided"] = G.summarize_norway("Norway unguided (delayed colour)", x, k=nor_mfm["k"])

    if "mfm" in args.backends:
        x, diags["norway_mfm"] = timed(
            "norway_mfm", G.delayed_color_sample_mfm, base_model=base, mfm=mfm,
            n_samples=nor_mfm["n_samples"], seed=nor_mfm["seed"],
            guidance_ratio=nor_mfm["guidance_ratio"], n_value_samples=nor_mfm["n_value_samples"],
            n_mfm_steps=nor_mfm["n_mfm_steps"], mfm_final_u=nor_mfm["mfm_final_u"],
            guidance_start=nor_mfm["guidance_start"], guidance_end=nor_mfm["guidance_end"],
            guidance_batch_size=nor_mfm["guidance_batch_size"], **common_nor)
        panels["norway_mfm"] = x
        metrics["norway_mfm"] = G.summarize_norway("Norway MFM-guided", x, k=nor_mfm["k"])
    if "glass" in args.backends:
        x, diags["norway_glass"] = timed(
            "norway_glass", G.delayed_color_sample_glass, base_model=base,
            n_samples=nor_glass["n_samples"], seed=nor_glass["seed"],
            guidance_ratio=nor_glass["guidance_ratio"], n_value_samples=nor_glass["n_value_samples"],
            n_glass_steps=nor_glass["n_glass_steps"], guidance_start=nor_glass["guidance_start"],
            guidance_end=nor_glass["guidance_end"], guidance_batch_size=nor_glass["guidance_batch_size"],
            **common_nor)
        panels["norway_glass"] = x
        metrics["norway_glass"] = G.summarize_norway("Norway GLASS-guided", x, k=nor_glass["k"])

    shape = dict(SMILEY_SHAPE)
    smi_reward = G.smiley_reward_fn(beta=smi_mfm["beta"], shape=shape)

    def smiley_kwargs(cfg):
        return dict(n_steps=cfg["n_steps"], reward_fn=smi_reward, device=device, seed=cfg["seed"],
                    pos_start=cfg["pos_start"], pos_end=cfg["pos_end"], pos_lambda_base=cfg["pos_lambda_base"],
                    color_base_freeze_until=cfg["color_base_freeze_until"],
                    color_guidance_start=cfg["color_guidance_start"], color_guidance_end=cfg["color_guidance_end"],
                    n_value_samples=cfg["n_value_samples"], guidance_batch_size=cfg["guidance_batch_size"],
                    base_batch_size=cfg["base_batch_size"], shape=shape)

    x, diags["smiley_unguided"] = timed(
        "smiley_unguided", G.sample_hybrid_mfm, base_model=base, mfm=mfm, n_samples=smi_mfm["n_samples"],
        pos_lambda=0.0, color_guidance_ratio=0.0, **smiley_kwargs(smi_mfm))
    panels["smiley_unguided"] = x
    metrics["smiley_unguided"] = G.summarize_smiley("Smiley unguided", x, shape)
    if "mfm" in args.backends:
        x, diags["smiley_mfm"] = timed(
            "smiley_mfm", G.sample_hybrid_mfm, base_model=base, mfm=mfm, n_samples=smi_mfm["n_samples"],
            pos_lambda=smi_mfm["pos_lambda"], color_guidance_ratio=smi_mfm["color_guidance_ratio"],
            n_mfm_steps=smi_mfm["n_mfm_steps"], mfm_final_u=smi_mfm["mfm_final_u"], **smiley_kwargs(smi_mfm))
        panels["smiley_mfm"] = x
        metrics["smiley_mfm"] = G.summarize_smiley(f"Smiley MFM-guided (N={smi_mfm['n_samples']})", x, shape)
        if not args.quick and args.n_panel != smi_mfm["n_samples"]:
            x, _ = timed(
                "smiley_mfm_panel", G.sample_hybrid_mfm, base_model=base, mfm=mfm, n_samples=args.n_panel,
                pos_lambda=smi_mfm["pos_lambda"], color_guidance_ratio=smi_mfm["color_guidance_ratio"],
                n_mfm_steps=smi_mfm["n_mfm_steps"], mfm_final_u=smi_mfm["mfm_final_u"], **smiley_kwargs(smi_mfm))
            panels["smiley_mfm_panel"] = x
            metrics["smiley_mfm_panel"] = G.summarize_smiley(f"Smiley MFM-guided (N={args.n_panel})", x, shape)
    if "glass" in args.backends:
        x, diags["smiley_glass"] = timed(
            "smiley_glass", G.sample_hybrid_glass, base_model=base, n_samples=smi_glass["n_samples"],
            pos_lambda=smi_glass["pos_lambda"], color_guidance_ratio=smi_glass["color_guidance_ratio"],
            n_glass_steps=smi_glass["n_glass_steps"], **smiley_kwargs(smi_glass))
        panels["smiley_glass"] = x
        metrics["smiley_glass"] = G.summarize_smiley(f"Smiley GLASS-guided (N={smi_glass['n_samples']})", x, shape)

    np.savez_compressed(out_dir / "fig1_panels.npz", **{k: v.numpy().astype(np.float32) for k, v in panels.items()})
    diag_out = {k: {kk: vv for kk, vv in d.items() if kk != "prior"} for k, d in diags.items()}
    record = {
        "configs": {"norway_mfm": nor_mfm, "norway_glass": nor_glass, "smiley_mfm": smi_mfm,
                    "smiley_glass": smi_glass, "smiley_shape": shape, "n_panel": n_fig, "quick": args.quick},
        "checkpoints": {"base": str(args.base_ckpt), "base_sha256": sha256(args.base_ckpt),
                        "mfm": str(args.mfm_ckpt), "mfm_sha256": sha256(args.mfm_ckpt)},
        "metrics": metrics,
        "runtimes_s": runtimes,
        "diagnostics": diag_out,
        "device": device,
        "torch": torch.__version__,
    }
    (out_dir / "steering_metrics.json").write_text(json.dumps(record, indent=1))
    draw(out_dir, panels, shape, args.backends)
    print(f"saved {out_dir}/fig1_panels.npz, steering_metrics.json, fig1_*.png/.pdf")
    return record


def draw(out_dir, panels, shape, backends=("glass", "mfm")):
    """Render fig1_<backend>.png/.pdf from the panel endpoints (no sampling)."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    for backend in backends:
        smiley_key = "smiley_mfm_panel" if (backend == "mfm" and "smiley_mfm_panel" in panels) else f"smiley_{backend}"
        fig = four_panel({"prior": panels["prior"], "base": panels["base"],
                          "norway": panels[f"norway_{backend}"], "smiley": panels[smiley_key]},
                         smiley_score_kwargs=shape, out_path=out_dir / f"fig1_{backend}.png")
        # the paper includes the figure as a PDF (figures/norway.pdf)
        fig.savefig(out_dir / f"fig1_{backend}.pdf")
        plt.close(fig)


def redraw(out_dir, backends=("glass", "mfm")):
    """Re-render the figures from an existing fig1_panels.npz + steering_metrics.json."""
    out_dir = Path(out_dir)
    panels = dict(np.load(out_dir / "fig1_panels.npz"))
    shape = json.loads((out_dir / "steering_metrics.json").read_text())["configs"]["smiley_shape"]
    draw(out_dir, panels, shape, backends)
    print(f"redrew {out_dir}/fig1_*.png/.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_ckpt", default=str(BASE_CKPT))
    p.add_argument("--mfm_ckpt", default=str(MFM_CONSISTENCY_CKPT))
    p.add_argument("--backends", nargs="+", default=["glass", "mfm"], choices=["glass", "mfm"])
    p.add_argument("--n_panel", type=int, default=10_000, help="points in the p0 / p1 / smiley-MFM panels")
    p.add_argument("--quick", action="store_true", help="small n for notebook reruns")
    p.add_argument("--redraw", action="store_true", help="only re-render the figures from an existing --out_dir")
    p.add_argument("--out_dir", default=str(RESULTS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.redraw:
        redraw(args.out_dir, args.backends)
    else:
        run(args)


if __name__ == "__main__":
    main()
