"""MFM posterior sampler and value function.

RECONSTRUCTED (original ``norway/mfm_value_fn.py`` was purged).  The design is
recorded in the terminal log (plan accepted by the user, then implemented):

    load_mfm(ckpt, device, use_ema=True)
    mfm_posterior_sample(mfm, t_cond, x_cond, n_samples, n_mfm_steps, final_u=1.0, seed=...)
    mfm_value_function(mfm, t, x, reward_fn, n_samples, n_mfm_steps, seed=...)

    I = randn(B * N, D)
    for s0, s1 in posterior_grid:
        I = mfm.X(s=s0, u=s1, x=I, t_cond=t, x_cond=x_t_repeated)
    X1_samples = I;  V_t(x_t) = logmeanexp(r(X1_samples))

and "returns the expected (B, MC, D) samples".  The original CLI default was
``--ckpt checkpoints/mfm_consistency_10000.pt`` (rg hit, mfm_value_fn.py:129).
"""

from __future__ import annotations

import argparse

import torch

from .mfm_model import MultimodalMFM
from .paths import MFM_CONSISTENCY_CKPT
from .value_fn import freeze_model, logmeanexp


def load_mfm(ckpt_path, device: str = "cpu", use_ema: bool = True) -> MultimodalMFM:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultimodalMFM(**ckpt["config"]).to(device)
    state_key = "ema" if (use_ema and "ema" in ckpt) else "model"
    model.load_state_dict(ckpt[state_key])
    return freeze_model(model)


def mfm_posterior_sample(
    mfm: MultimodalMFM,
    t_cond: torch.Tensor,
    x_cond: torch.Tensor,
    n_samples: int,
    n_mfm_steps: int = 1,
    final_u: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """x1 ~ p(x1 | x_{t_cond} = x_cond) with ``n_mfm_steps`` flow-map jumps. Returns (B, N, D)."""
    B, D = x_cond.shape
    xc = x_cond.repeat_interleave(n_samples, dim=0)
    tc = t_cond.repeat_interleave(n_samples, dim=0)
    I = torch.randn(xc.shape, generator=generator, device=xc.device, dtype=xc.dtype)
    grid = torch.linspace(0.0, final_u, n_mfm_steps + 1, device=xc.device, dtype=xc.dtype)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        s = torch.full((I.shape[0],), float(s0), device=I.device, dtype=I.dtype)
        u = torch.full_like(s, float(s1))
        I = mfm.X(s, u, I, t_cond=tc, x_cond=xc)
    return I.view(B, n_samples, D)


def mfm_value_function(
    mfm: MultimodalMFM,
    t: torch.Tensor,
    x: torch.Tensor,
    reward_fn,
    n_samples: int = 8,
    n_mfm_steps: int = 1,
    final_u: float = 1.0,
    generator: torch.Generator | None = None,
    return_samples: bool = False,
):
    """V_t(x) = log mean_j exp(reward_fn(X1_j)), X1_j from the MFM posterior sampler."""
    X1 = mfm_posterior_sample(mfm, t, x, n_samples, n_mfm_steps, final_u, generator=generator)
    B, N, D = X1.shape
    r = reward_fn(X1.reshape(B * N, D)).view(B, N)
    V = logmeanexp(r, dim=1)
    return (V, X1) if return_samples else V


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-test of the MFM posterior sampler.")
    parser.add_argument("--ckpt", default=str(MFM_CONSISTENCY_CKPT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()
    mfm = load_mfm(args.ckpt, args.device)
    D = mfm.num_classes + mfm.y_dim
    x = torch.randn(4, D, device=args.device)
    t = torch.full((4,), 0.9, device=args.device)
    for steps in (1, 2, 4):
        X1 = mfm_posterior_sample(mfm, t, x, args.n_samples, steps)
        print(f"n_mfm_steps={steps}: samples {tuple(X1.shape)}")
