"""GLASS inference-time steering of the GEOM-Drugs base model toward a target
structure fingerprint (the teacher-look-ahead ablation of Table 3's multiMFM row).

Identical to :mod:`multimfm.geom.mfm_steering` except that the look-ahead clean
endpoints are obtained by integrating the teacher's GLASS posterior field for
``--value-glass-steps`` steps instead of evaluating the distilled meta flow map,
which is far more expensive per guided step.

Example::

    python -m multimfm.geom.glass_steering --num-mols 64 --sample-steps 50 \\
        --value-glass-steps 6 --value-samples 4 --mu 0.5 --reward-lambda 0.05 \\
        --device cuda --tag big_lam0.05_mu0.5

Ported from ``geom-drug-tabasco/glass/glass_fingerprint_steering.py``.
"""

from __future__ import annotations

from multimfm.geom.steering import main as _main


def main() -> None:
    _main("glass", __doc__)


if __name__ == "__main__":
    main()
