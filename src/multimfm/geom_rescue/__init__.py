"""Additive patches used by the GEOM rescue attempt (docs/geom_rescue_attempt.md).

Nothing in this package changes the behaviour of any recorded run: every patch is
opt-in through an ``install_*`` call, and the shared modules under
``multimfm.geom`` / ``multimfm.glass_guidance`` are left untouched on disk.
"""

from multimfm.geom_rescue.scalefree import (
    install_scalefree_guidance,
    uninstall_scalefree_guidance,
)

__all__ = ["install_scalefree_guidance", "uninstall_scalefree_guidance"]
