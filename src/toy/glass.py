"""GLASS posterior velocity for the linear interpolant.

Local port of `mfm.losses.losses.extract_posterior_velocity`, restricted to
the unconditional branch (no class labels, no CFG). The teacher is queried
only on the diagonal s == t with t_cond=0, x_cond=0, so we just need a
function `teacher_v(t, x) -> velocity at (t, x)`.

The auxiliary ODE variable is s in [0, 1]. At s = 0 the integrator starts
from random Gaussian noise; at s = 1 the integrated trajectory is a sample
from p(x_1 | x_{t_cond} = xt_cond).
"""

import torch


def _bcast(coef: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return coef.view(-1, *((1,) * (ref.ndim - 1)))


def glass_posterior_velocity(
    s: torch.Tensor,
    Is: torch.Tensor,
    xt_cond: torch.Tensor,
    t_cond: torch.Tensor,
    teacher_v,
    eps: float = 1e-6,
):
    """Posterior-flow velocity in s for x_1 ~ p(x_1 | x_{t_cond}).

    Args:
        s:        (B,) float in [0, 1] — the auxiliary integration time.
        Is:       (B, ...) float — current state along the s-trajectory.
        xt_cond:  (B, ...) float — observation at t_cond.
        t_cond:   (B,) float in (0, 1) — observation time.
        teacher_v: callable (t, x) -> velocity at (t, x), broadcasts over
                   batch. This is the *unconditional* flow velocity field.

    Returns:
        dIs/ds: (B, ...) — the GLASS posterior velocity.
    """
    s_b = _bcast(s, Is)
    t_b = _bcast(t_cond, Is)
    one_minus_s = 1.0 - s_b
    one_minus_t = 1.0 - t_b

    denom = (t_b ** 2 * one_minus_s ** 2 + one_minus_t ** 2 * s_b ** 2).clamp_min(eps)
    P_norm = one_minus_t ** 2 / denom
    sqrt_P_norm = torch.sqrt(P_norm)

    t_star = 1.0 / (1.0 + one_minus_s * sqrt_P_norm)

    coeff_cond = t_star * one_minus_s ** 2 * t_b / denom
    coeff_Is = t_star * s_b * P_norm
    x_star = coeff_cond * xt_cond + coeff_Is * Is

    # teacher_v is queried on the diagonal at time t_star.
    v_star = teacher_v(t_star.view(s.shape[0]), x_star)

    term2 = t_star * sqrt_P_norm * v_star
    diff_div_x = (one_minus_t ** 2 * (1.0 + s_b) - t_b ** 2 * one_minus_s) / denom
    B_minus_1_div_x = (diff_div_x - (P_norm + sqrt_P_norm)) / (1.0 + one_minus_s * sqrt_P_norm)
    A_div_x = t_star * one_minus_s * t_b / denom
    term1 = A_div_x * xt_cond + B_minus_1_div_x * Is

    return term1 + term2
