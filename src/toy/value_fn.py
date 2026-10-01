"""Teacher velocity, GLASS posterior sampling and the GLASS value function.

RECONSTRUCTED (original ``norway/value_fn.py`` was purged).  The interface is
fixed by the recovered code and transcripts:

* ``freeze_model(model)`` and ``make_joint_teacher_v(teacher, NUM_CLASSES)``
  are imported by ``train_mfm_diagonal.py`` / ``probe_mfm_diagonal.py``; the
  returned ``teacher_v(t, x)`` is called with a (B,) time and the joint state
  x = concat(c, y) and must return the *unconditional* endpoint-form velocity
  (glass.py docstring).
* the terminal log describes the old GLASS value estimator as
  ``X1 = glass_integrate(base_model, t_cond=t, x_cond=x_t, n_samples=N,
  n_steps=n_glass_steps); V_t(x_t) = logmeanexp(r(X1))`` and says
  ``mfm_value_fn.mfm_value_function`` "mirrors value_fn.value_function".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NUM_CLASSES
from .glass import glass_posterior_velocity


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def endpoint_from_base(model: nn.Module, t: torch.Tensor, x: torch.Tensor, num_classes: int = NUM_CLASSES):
    """Base endpoint prediction concat(softmax(z_logits), y_pred) at (t, x)."""
    y_pred, z_logits = model(t, x[:, :num_classes], x[:, num_classes:])
    return torch.cat([F.softmax(z_logits, dim=-1), y_pred], dim=-1)


def make_joint_teacher_v(model: nn.Module, num_classes: int = NUM_CLASSES, eps: float = 1e-6):
    """teacher_v(t, x) = (x1_hat(t, x) - x) / (1 - t), same form as ``MultimodalMFM.v``."""

    def teacher_v(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1_hat = endpoint_from_base(model, t, x, num_classes)
        return (x1_hat - x) / (1.0 - t).clamp_min(eps).unsqueeze(-1)

    return teacher_v


def logmeanexp(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])


def glass_integrate(
    base_model: nn.Module,
    t_cond: torch.Tensor,
    x_cond: torch.Tensor,
    n_samples: int,
    n_steps: int,
    generator: torch.Generator | None = None,
    teacher_v=None,
) -> torch.Tensor:
    """Sample x1 ~ p(x1 | x_{t_cond} = x_cond) by Euler-integrating the GLASS ODE in s.

    t_cond: (B,), x_cond: (B, D).  Returns (B, n_samples, D).  Differentiable
    w.r.t. ``x_cond`` (the teacher is queried at x_star(x_cond, I_s)).
    """
    if teacher_v is None:
        teacher_v = make_joint_teacher_v(base_model, NUM_CLASSES)
    B, D = x_cond.shape
    xc = x_cond.repeat_interleave(n_samples, dim=0)
    tc = t_cond.repeat_interleave(n_samples, dim=0)
    I = torch.randn(xc.shape, generator=generator, device=xc.device, dtype=xc.dtype)
    s_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=xc.device, dtype=xc.dtype)
    for s0, s1 in zip(s_grid[:-1], s_grid[1:]):
        s = torch.full((I.shape[0],), float(s0), device=I.device, dtype=I.dtype)
        I = I + (s1 - s0) * glass_posterior_velocity(s, I, xc, tc, teacher_v, eps=1e-6)
    return I.view(B, n_samples, D)


def value_function(
    base_model: nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    reward_fn,
    n_samples: int = 8,
    n_glass_steps: int = 8,
    generator: torch.Generator | None = None,
    teacher_v=None,
    return_samples: bool = False,
):
    """V_t(x) = log mean_j exp(reward_fn(X1_j)),  X1_j ~ GLASS posterior at (t, x).

    ``reward_fn`` maps (M, D) endpoints to (M,) *already scaled* log-rewards
    (e.g. ``beta * log r``).  Returns (B,) values (and the (B, N, D) samples).
    """
    X1 = glass_integrate(
        base_model, t, x, n_samples, n_glass_steps, generator=generator, teacher_v=teacher_v
    )
    B, N, D = X1.shape
    r = reward_fn(X1.reshape(B * N, D)).view(B, N)
    V = logmeanexp(r, dim=1)
    return (V, X1) if return_samples else V
