"""Scale-free guidance steps for the GEOM sampler.

`multimfm.glass_guidance.clip_guidance_rms` clips the per-molecule gradient RMS
*down* to ``max_rms`` and leaves smaller gradients alone. The displacement of a
guided step is therefore ``mu * |grad| * dt``, which mixes two things the grid
needs to separate:

* the **direction** of the tilt, which is what ``lambda`` changes (the
  log-mean-exp weighting over the look-ahead samples), and
* the **magnitude** of the displacement, which ``lambda`` also changes, because
  ``grad log mean exp(lambda r)`` scales roughly with ``lambda`` while the tilt
  stays soft, and degenerates to the single best sample once ``lambda`` is large.

That coupling is why the recorded grid never swept displacement cleanly: at
``lambda = 0.01, mu = 0.25`` the coherent coordinate displacement is 0.53 A, and
at ``lambda = 1`` it is ~100x larger, which is why `lambda = 1` destroyed the
molecules (PoseBusters 88% -> 45%). The same defect -- a clip that only acts
downward, so the step is not scale-free -- was found and fixed on the DNA side of
this paper (see `docs/REPRODUCIBILITY_PLAN.md` §7 item 8), where it had made dMFM
look three times worse than it is.

``install_scalefree_guidance()`` replaces the clip with a *rescale*: the
per-molecule gradient is normalised to exactly ``max_rms``, so the displacement
per guided step is ``mu * max_rms * dt`` regardless of ``lambda``, the molecule,
or the point on the trajectory. ``lambda`` then controls only the direction.

The patch is a no-op when ``max_rms <= 0``, which is the default in every
recorded GEOM run, so an un-patched run and a patched run with
``--guidance-max-coord-rms 0`` are identical.
"""

from __future__ import annotations

import torch

from multimfm import glass_guidance as engine

__all__ = ["install_scalefree_guidance", "uninstall_scalefree_guidance",
           "scalefree_guidance_rms"]

_ORIGINAL = None


def scalefree_guidance_rms(grad: torch.Tensor, padding_mask: torch.Tensor,
                           max_rms: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Rescale each molecule's gradient to exactly ``max_rms`` (0 disables)."""
    if max_rms <= 0.0:
        rms = torch.sqrt(grad.square().mean().clamp_min(0.0))
        v = float(rms.detach().cpu())
        return grad, {"rms_before": v, "rms_after": v, "scale_mean": 1.0,
                      "scale_min": 1.0, "rule": "off"}
    mask = (~padding_mask).unsqueeze(-1).to(dtype=grad.dtype)
    denom = (mask.sum(dim=1, keepdim=True) * grad.shape[-1]).clamp_min(1.0)
    per_sample_rms = torch.sqrt(
        ((grad.square() * mask).sum(dim=(1, 2), keepdim=True) / denom).clamp_min(0.0)
    )
    scale = max_rms / per_sample_rms.clamp_min(1e-12)      # no max=1.0 clamp: scale-free
    scaled = grad * scale
    return scaled, {
        "rms_before": float(torch.sqrt(grad.square().mean().clamp_min(0.0)).detach().cpu()),
        "rms_after": float(torch.sqrt(scaled.square().mean().clamp_min(0.0)).detach().cpu()),
        "scale_mean": float(scale.detach().mean().cpu()),
        "scale_min": float(scale.detach().min().cpu()),
        "rule": "scalefree",
    }


def install_scalefree_guidance() -> None:
    """Make guided steps scale-free. Idempotent."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        return
    _ORIGINAL = engine.clip_guidance_rms
    engine.clip_guidance_rms = scalefree_guidance_rms


def uninstall_scalefree_guidance() -> None:
    """Restore the recorded (downward-only clip) behaviour."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        engine.clip_guidance_rms = _ORIGINAL
        _ORIGINAL = None
