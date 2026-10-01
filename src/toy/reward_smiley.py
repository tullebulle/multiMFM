"""Smiley reward (paper App. F.7.3).

RECONSTRUCTED (original ``norway/dMFM_experiments/reward_smiley.py`` was
purged).  Names / keyword arguments are the ones used by the recovered
notebooks: ``smiley_scores(y1, y2, **shape)``,
``smiley_log_reward_for_model_state(X1, detach_regions=False, **shape)``,
``smiley_reward_raw``.

Formulas follow App. F.7.3: Gaussian eyes E_R at (-x_e, y_e), E_B at (x_e, y_e)
with shared scale sigma_e; a Gaussian tube around y2 = y0 + kappa y1^2 gated to
[-m_x, m_x] x [m_y^-, m_y^+]; a square gate S(y); scores in model order
s = S(y) (E_B, M, E_R); r = sum_k softmax(c/T)_k s_k;
log r = logsumexp_k(log max(s_k, eps) + log softmax(c/T)_k).

Defaults are the "bigger smile / thick mouth" geometry used by the good GLASS
smiley (the Fig. 1 panel) and the final MFM smiley, recorded in the MFM
notebook: eye_sigma=0.28, mouth_sigma=0.23, mouth_x_half=0.84,
mouth_y_low=-0.88, mouth_y_high=0.18, mouth_y0=-0.55, mouth_curve=0.65, k=18,
color_temp=1.  The eye centre (x_e, y_e) is not recorded; (0.30, 0.35) was
measured from the Fig. 1 raster (guided eye centroids at (+-0.30, 0.335)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import NUM_CLASSES
from .reward import smooth_box, square_support_gate

SMILEY_SHAPE = {
    "eye_sigma": 0.28,
    "mouth_sigma": 0.23,
    "mouth_x_half": 0.84,
    "mouth_y_low": -0.88,
    "mouth_y_high": 0.18,
    "mouth_y0": -0.55,
    "mouth_curve": 0.65,
    "k": 18.0,
    "color_temp": 1.0,
}
EYE_X = 0.30
EYE_Y = 0.35
LOG_EPS = 1e-6


def smiley_scores(
    y1: torch.Tensor,
    y2: torch.Tensor,
    eye_sigma: float = SMILEY_SHAPE["eye_sigma"],
    mouth_sigma: float = SMILEY_SHAPE["mouth_sigma"],
    mouth_x_half: float = SMILEY_SHAPE["mouth_x_half"],
    mouth_y_low: float = SMILEY_SHAPE["mouth_y_low"],
    mouth_y_high: float = SMILEY_SHAPE["mouth_y_high"],
    mouth_y0: float = SMILEY_SHAPE["mouth_y0"],
    mouth_curve: float = SMILEY_SHAPE["mouth_curve"],
    k: float = SMILEY_SHAPE["k"],
    eye_x: float = EYE_X,
    eye_y: float = EYE_Y,
) -> torch.Tensor:
    """Feature scores in model colour order (B, W, R), shape (N, 3)."""
    two_s2 = 2.0 * eye_sigma**2
    e_red = torch.exp(-((y1 + eye_x) ** 2 + (y2 - eye_y) ** 2) / two_s2)
    e_blue = torch.exp(-((y1 - eye_x) ** 2 + (y2 - eye_y) ** 2) / two_s2)
    tube = torch.exp(-((y2 - mouth_y0 - mouth_curve * y1**2) ** 2) / (2.0 * mouth_sigma**2))
    mouth = (
        tube
        * smooth_box(y1, -mouth_x_half, mouth_x_half, k)
        * smooth_box(y2, mouth_y_low, mouth_y_high, k)
    )
    gate = square_support_gate(y1, y2, k)
    return gate.unsqueeze(-1) * torch.stack([e_blue, mouth, e_red], dim=-1)


def _split_shape(shape: dict) -> tuple[float, dict]:
    shape = dict(shape)
    temp = float(shape.pop("color_temp", 1.0))
    return temp, shape


def smiley_reward_raw(x: torch.Tensor, detach_regions: bool = False, **shape) -> torch.Tensor:
    """r = sum_k softmax(c / T)_k s_k(y)."""
    temp, geom = _split_shape(shape)
    y = x[:, NUM_CLASSES:]
    s = smiley_scores(y[:, 0], y[:, 1], **geom)
    if detach_regions:
        s = s.detach()
    return (F.softmax(x[:, :NUM_CLASSES] / temp, dim=-1) * s).sum(dim=-1)


def smiley_log_reward_for_model_state(
    x: torch.Tensor, detach_regions: bool = False, eps: float = LOG_EPS, **shape
) -> torch.Tensor:
    """log r = logsumexp_k(log max(s_k, eps) + log softmax(c / T)_k)."""
    temp, geom = _split_shape(shape)
    y = x[:, NUM_CLASSES:]
    s = smiley_scores(y[:, 0], y[:, 1], **geom)
    if detach_regions:
        s = s.detach()
    logp = F.log_softmax(x[:, :NUM_CLASSES] / temp, dim=-1)
    return torch.logsumexp(torch.log(s.clamp_min(eps)) + logp, dim=-1)


def smiley_target(y: torch.Tensor, **shape) -> tuple[torch.Tensor, torch.Tensor]:
    """(target colour index, max feature score) for each position."""
    _, geom = _split_shape(shape)
    s = smiley_scores(y[:, 0], y[:, 1], **geom)
    best = s.max(dim=-1)
    return best.indices, best.values
