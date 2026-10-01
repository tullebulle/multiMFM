"""Checkpoint loading and the plain (unguided) Euler sampler of the base model.

RECONSTRUCTED (original ``norway/sample.py`` was purged).  The MFM notebook
imports ``load_model(ckpt, device=..., use_ema=True)`` and
``resolve_figure_path(name)`` from it.  Sampling follows App. E.2.1 / Table 12:
(y0, a0) ~ N(0, I_d) x N(0, I_K), explicit Euler with the endpoint-form
velocity on [0, 1 - eps] (100 steps, eps = 1e-3), continuous output read from
y_hat_1 and the class sampled from Categorical(softmax z_hat) at t = 1 - eps.
The grid ``linspace(0, 1 - eps, n_steps + 1)`` reproduces the times printed in
the recovered notebook log (t = 0.100, 0.599, 0.799, 0.899, 0.989 at steps
11, 61, 81, 91, 100 of 100).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import NUM_CLASSES
from .model import MultimodalDiT
from .paths import BASE_CKPT, FIGURES


def load_model(ckpt_path, device: str = "cpu", use_ema: bool = True) -> MultimodalDiT:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultimodalDiT(**ckpt["config"]).to(device)
    state_key = "ema" if (use_ema and "ema" in ckpt) else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model


def resolve_figure_path(out: str | Path) -> Path:
    path = Path(out)
    if path.parent == Path("."):
        path = FIGURES / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def time_grid(n_steps: int, eps: float = 1e-3, device="cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.linspace(0.0, 1.0 - eps, n_steps + 1, device=device, dtype=dtype)


def base_endpoint(model, t: torch.Tensor, x: torch.Tensor, batch_size: int | None = None):
    """(y_pred, z_logits) evaluated in chunks (the attention kernel fails on 100k batches)."""
    if batch_size is None or x.shape[0] <= batch_size:
        return model(t, x[:, :NUM_CLASSES], x[:, NUM_CLASSES:])
    ys, zs = [], []
    for i in range(0, x.shape[0], batch_size):
        y, z = model(t[i : i + batch_size], x[i : i + batch_size, :NUM_CLASSES], x[i : i + batch_size, NUM_CLASSES:])
        ys.append(y)
        zs.append(z)
    return torch.cat(ys), torch.cat(zs)


def readout(y_pred: torch.Tensor, z_logits: torch.Tensor, generator=None, mode: str = "sample") -> torch.Tensor:
    """Final state concat(onehot(class), y_pred)."""
    if mode == "sample":
        cls = torch.multinomial(F.softmax(z_logits, dim=-1), 1, generator=generator).squeeze(-1)
    elif mode == "argmax":
        cls = z_logits.argmax(dim=-1)
    else:
        raise ValueError(mode)
    return torch.cat([F.one_hot(cls, NUM_CLASSES).to(y_pred.dtype), y_pred], dim=-1)


def prior_sample(n: int, dim: int, generator=None, device="cpu") -> torch.Tensor:
    return torch.randn(n, dim, generator=generator, device=device)


@torch.no_grad()
def sample(
    model,
    n_samples: int,
    n_steps: int = 100,
    eps: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    batch_size: int | None = 10_000,
    readout_mode: str = "sample",
    return_prior: bool = False,
):
    g = torch.Generator(device=device).manual_seed(seed)
    D = model.num_classes + model.y_dim
    x = prior_sample(n_samples, D, generator=g, device=device)
    x0 = x.clone()
    ts = time_grid(n_steps, eps, device=device)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        t = torch.full((n_samples,), float(t0), device=device)
        y_pred, z_logits = base_endpoint(model, t, x, batch_size)
        x1_hat = torch.cat([F.softmax(z_logits, dim=-1), y_pred], dim=-1)
        x = x + (t1 - t0) * (x1_hat - x) / (1.0 - t0)
    t = torch.full((n_samples,), float(ts[-1]), device=device)
    y_pred, z_logits = base_endpoint(model, t, x, batch_size)
    out = readout(y_pred, z_logits, generator=g, mode=readout_mode)
    return (out, x0) if return_prior else out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unguided samples from the base model.")
    p.add_argument("--ckpt", default=str(BASE_CKPT))
    p.add_argument("--n_samples", type=int, default=10_000)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="base_samples.pt")
    args = p.parse_args()
    model = load_model(args.ckpt, args.device)
    x = sample(model, args.n_samples, args.n_steps, device=args.device, seed=args.seed)
    c = x[:, :NUM_CLASSES].argmax(-1)
    y = x[:, NUM_CLASSES:]
    inside = (y.abs() <= 1).all(-1).float().mean().item()
    print(f"colour fractions B/W/R = {[round((c == k).float().mean().item(), 3) for k in range(3)]}")
    print(f"fraction inside square = {inside:.3f}")
    out = resolve_figure_path(args.out)
    torch.save({"samples": x.cpu(), "ckpt": args.ckpt}, out)
    print(f"saved {out}")
