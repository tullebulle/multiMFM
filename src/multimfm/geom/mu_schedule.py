"""The guidance coefficient: the paper's ``sigma_t^2/2`` versus the code's constant ``mu``.

**What the paper says.** The guided samplers of Sec. 2 carry the coefficient ``sigma_t^2/2``
in front of the value gradient,

    dX_t = (b_t(X_t) + (sigma_t^2/2) grad V_t(X_t)) dt ,

and for the linear interpolant the matching schedule is ``sigma_t^2 = 2(1-t)/t``, i.e.

    sigma_t^2/2 = (1 - t) / t .

**What the code does.** ``multimfm.glass_guidance.production_step_with_optional_guidance``
adds ``mu * grad * dt`` with a *constant* ``mu`` supplied on the command line. Every GEOM
number in the submission -- and every QM9 number -- was produced that way. Inside the
recorded guidance window ``t in [0.1, 0.9]`` the substitution is wrong by a large,
*t*-dependent factor: with ``mu = 0.5`` it is 0.06x the paper's coefficient at ``t = 0.1``
(18x under-guided, where composition and scaffold are still being decided) and 4.5x at
``t = 0.9`` (over-guided, where perturbing an almost-final geometry is what breaks
PoseBusters validity). See ``docs/provenance/geom/sign_audit.md``.

**What this module adds.** ``install_mu_schedule("paper")`` wraps the engine's production
step so the gradient is pre-multiplied by ``(1-t)/max(t, eps_t)`` before the constant ``mu``
is applied; ``mu`` then plays the role of an overall scale on the paper's coefficient rather
than of the coefficient itself. Pre-scaling the gradient is algebraically identical to
scaling the coefficient, because the step is linear in the gradient:

    x + mu * [(1-t)/t * grad] * dt   ==   x + [(1-t)/t] * mu * grad * dt .

``"constant"`` is the default and leaves the engine untouched, so nothing already run
changes meaning.

**The t -> 0 guard.** ``(1-t)/t`` diverges as ``t -> 0``, so the coefficient is evaluated at
``max(t, eps_t)`` with ``eps_t = 0.05`` by default, capping it at 19. The paper itself notes
that the exact-law statement concerns the ideal dynamics and that "learned posterior maps,
finite Monte Carlo budgets, numerical integration, and practical guidance modifications
introduce approximation error", so some such guard is unavoidable in a discrete sampler.
Inside the recorded window ``t in [0.1, 0.9]`` the clamp never binds (it would only bind for
a run that guides below ``t = 0.05``, e.g. the literal every-step reading of Algorithm 2).

The unguided arm is untouched either way: the wrapper short-circuits on ``mu == 0.0``, which
is what the driver passes for the baseline, so the unguided trajectory is bit-identical with
and without ``--mu-schedule paper``. The paired guided-vs-unguided comparison therefore
measures only the change in the guided arm.

Interaction with the RMS clip: ``guided_sample`` clips the gradient RMS *before* the step,
so with ``"paper"`` the clip applies to the unscaled gradient. The clip is disabled by
default (``--guidance-max-coord-rms 0.0``) and was disabled in every recorded run, so the
two do not interact in practice; combining them is documented but not recommended.
"""

from __future__ import annotations

import torch

from multimfm import glass_guidance as engine
from multimfm.glass_guidance import bcast

__all__ = ["MU_SCHEDULES", "DEFAULT_EPS_T", "paper_coefficient", "install_mu_schedule",
           "active_mu_schedule"]

#: Available coefficient schedules. ``constant`` is the behaviour of every recorded run.
MU_SCHEDULES = ("constant", "paper")

#: ``t`` is clamped from below at this value when evaluating ``(1-t)/t``.
DEFAULT_EPS_T = 0.05


def paper_coefficient(t: torch.Tensor, eps_t: float = DEFAULT_EPS_T) -> torch.Tensor:
    """``sigma_t^2/2 = (1-t)/t`` evaluated at ``max(t, eps_t)``, elementwise."""
    t_safe = t.clamp_min(eps_t)
    return (1.0 - t_safe) / t_safe


def active_mu_schedule() -> tuple[str, float]:
    """The schedule currently installed on the engine (``("constant", eps)`` if none)."""
    return (
        getattr(engine, "_MU_SCHEDULE", "constant"),
        getattr(engine, "_MU_SCHEDULE_EPS_T", DEFAULT_EPS_T),
    )


def install_mu_schedule(schedule: str = "constant", eps_t: float = DEFAULT_EPS_T) -> None:
    """Install ``schedule`` as the guidance coefficient of the shared steering engine.

    Idempotent, and reversible by installing ``"constant"`` again: the original engine
    function is kept on the module so repeated calls never stack wrappers.
    """
    if schedule not in MU_SCHEDULES:
        raise ValueError(f"mu_schedule must be one of {MU_SCHEDULES}, got {schedule!r}")

    original = getattr(engine, "_PRODUCTION_STEP_CONSTANT_MU", None)
    if original is None:
        original = engine.production_step_with_optional_guidance
        engine._PRODUCTION_STEP_CONSTANT_MU = original

    if schedule == "constant":
        engine.production_step_with_optional_guidance = original
        engine._MU_SCHEDULE = "constant"
        engine._MU_SCHEDULE_EPS_T = eps_t
        return

    def production_step_with_paper_coefficient(
        model, state, t, dt, *, mu, grad_coords, grad_atomics
    ):
        if mu != 0.0 and (grad_coords is not None or grad_atomics is not None):
            coeff = paper_coefficient(t, eps_t)
            if grad_coords is not None:
                grad_coords = grad_coords * bcast(coeff, grad_coords)
            if grad_atomics is not None:
                grad_atomics = grad_atomics * bcast(coeff, grad_atomics)
        return original(
            model, state, t, dt, mu=mu,
            grad_coords=grad_coords, grad_atomics=grad_atomics,
        )

    production_step_with_paper_coefficient.__doc__ = (
        "production step whose guidance term is mu * (1-t)/max(t, eps_t) * grad V * dt, "
        "i.e. the paper's sigma_t^2/2 coefficient with mu as an overall scale"
    )
    engine.production_step_with_optional_guidance = production_step_with_paper_coefficient
    engine._MU_SCHEDULE = "paper"
    engine._MU_SCHEDULE_EPS_T = float(eps_t)
