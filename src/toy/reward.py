"""Norway-flag reward (paper App. F.7.2).

RECONSTRUCTED (original ``norway/reward.py`` was purged).  Function names and
keyword arguments are the ones used by the recovered notebooks:

* ``norwegian_region_scores(y1, y2, order="model")``            (Fig. 1 notebook)
* ``norwegian_color_reward_for_model_state(X1, k=40.0,
  color_score="log_softmax", detach_regions=True)``               (MFM notebook)
* ``norwegian_color_reward_raw(x, k=40.0, color_score=..., detach_regions=True)``

Formulas follow App. F.7.2:  smooth boxes sigma_k(z; a, b), soft union
a (+) b = 1 - (1-a)(1-b), white / blue Nordic crosses W(y), B(y), region scores
s_R = 1 - W, s_B = B, s_W = W (1 - B), reordered to the model colour order
(B, W, R).

Geometry constants are NOT recorded anywhere that survived.  They were fixed
from two pieces of evidence (never from guided results):

* the logged unguided region fractions B/W/R = 0.30/0.21/0.49 of the uniform
  square (terminal log, 10k and 100k samples) pin the half-widths:
  white 0.30, blue 0.165 (0.16-0.167 all round to the logged values);
* the cross centre x0 = -0.35 was measured from the Fig. 1 raster
  (vertical blue bar centred at y1 = -0.35 +- 0.02).

Square gate: the logged unguided mean reward is 0.324 although colours and
positions are independent, which is impossible for <softmax(c), s(y)> (it
equals 1/3 for any geometry because sum_k s_k >= 1).  Multiplying by the
smiley's square gate S(y) = sigma_k(y1; -1, 1) sigma_k(y2; -1, 1) (App. F.7.3)
reproduces the logged rewards (unguided 0.322 vs 0.324; see provenance), so the
gate is ON by default for the *reported* raw reward.  App. F.7.2 does not
mention it; ``square_gate=False`` gives the formula exactly as printed.  For
colour-only guidance with ``detach_regions=True`` the gate does not change the
direction of the colour gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import NUM_CLASSES

NORWAY_X0 = -0.35
NORWAY_WHITE_HALF = 0.30
NORWAY_BLUE_HALF = 0.165
NORWAY_K = 40.0
LOG_EPS = 1e-6


def smooth_box(z: torch.Tensor, a: float, b: float, k: float) -> torch.Tensor:
    """sigma_k(z; a, b) = sigmoid(k (z - a)) sigmoid(k (b - z))."""
    return torch.sigmoid(k * (z - a)) * torch.sigmoid(k * (b - z))


def soft_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - a) * (1.0 - b)


def square_support_gate(y1: torch.Tensor, y2: torch.Tensor, k: float) -> torch.Tensor:
    """S(y) = sigma_k(y1; -1, 1) sigma_k(y2; -1, 1)."""
    return smooth_box(y1, -1.0, 1.0, k) * smooth_box(y2, -1.0, 1.0, k)


def norwegian_region_scores(
    y1: torch.Tensor,
    y2: torch.Tensor,
    order: str = "model",
    k: float = NORWAY_K,
    x0: float = NORWAY_X0,
    white_half: float = NORWAY_WHITE_HALF,
    blue_half: float = NORWAY_BLUE_HALF,
) -> torch.Tensor:
    """Soft region scores, shape (N, 3).

    order="flag":  (R, B, W) as defined in App. F.7.2.
    order="model": (B, W, R), the model colour convention.
    """
    W = soft_union(
        smooth_box(y1, x0 - white_half, x0 + white_half, k),
        smooth_box(y2, -white_half, white_half, k),
    )
    B = soft_union(
        smooth_box(y1, x0 - blue_half, x0 + blue_half, k),
        smooth_box(y2, -blue_half, blue_half, k),
    )
    s_R = 1.0 - W
    s_B = B
    s_W = W * (1.0 - B)
    if order == "flag":
        return torch.stack([s_R, s_B, s_W], dim=-1)
    if order == "model":
        return torch.stack([s_B, s_W, s_R], dim=-1)
    raise ValueError(f"unknown order {order!r}")


def _scores_for_state(
    x: torch.Tensor, k: float, detach_regions: bool, gate: bool, **geom
) -> torch.Tensor:
    y = x[:, NUM_CLASSES:]
    s = norwegian_region_scores(y[:, 0], y[:, 1], order="model", k=k, **geom)
    if gate:
        s = s * square_support_gate(y[:, 0], y[:, 1], k).unsqueeze(-1)
    return s.detach() if detach_regions else s


def norwegian_color_reward_raw(
    x: torch.Tensor,
    k: float = NORWAY_K,
    color_score: str = "log_softmax",
    detach_regions: bool = True,
    square_gate: bool = True,
    **geom,
) -> torch.Tensor:
    """Reported reward r = S(y) <softmax(c), s(y)> in [0, 1] (the "mean reward" of the logs).

    The model-state colour coordinates c are treated as logits, so a one-hot
    endpoint gives probabilities (0.576, 0.212, 0.212), i.e. the logged max/min.
    ``color_score`` only affects the log-reward; it is accepted for API
    compatibility with the recovered notebook.
    """
    del color_score
    s = _scores_for_state(x, k, detach_regions, square_gate, **geom)
    p = F.softmax(x[:, :NUM_CLASSES], dim=-1)
    return (p * s).sum(dim=-1)


def norwegian_color_reward_for_model_state(
    x: torch.Tensor,
    k: float = NORWAY_K,
    color_score: str = "log_softmax",
    detach_regions: bool = True,
    square_gate: bool = True,
    eps: float = LOG_EPS,
    **geom,
) -> torch.Tensor:
    """Log-reward used inside the value function (multiply by beta outside).

    color_score="log_softmax":  <s(y), log softmax(c)>            (c as logits)
    color_score="prob":         log max(<softmax(c), s(y)>, eps)
    """
    s = _scores_for_state(x, k, detach_regions, square_gate, **geom)
    c = x[:, :NUM_CLASSES]
    if color_score == "log_softmax":
        return (s * F.log_softmax(c, dim=-1)).sum(dim=-1)
    if color_score == "prob":
        return torch.log((F.softmax(c, dim=-1) * s).sum(dim=-1).clamp_min(eps))
    raise ValueError(f"unknown color_score {color_score!r}")


def norway_target(y: torch.Tensor, k: float = NORWAY_K, **geom) -> torch.Tensor:
    """Target colour index (model order) = argmax of the region scores."""
    return norwegian_region_scores(y[:, 0], y[:, 1], order="model", k=k, **geom).argmax(dim=-1)
