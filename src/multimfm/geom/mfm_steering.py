"""multiMFM inference-time steering of the GEOM-Drugs base model (paper Table 3).

At each sampling step the look-ahead clean endpoints ``x1`` are produced by the
distilled **meta flow map** (a ``TabascoFlowMap`` student) integrated in
``--mfm-steps`` steps, their fingerprints are scored by the TFG-Flow structure
head, and ``mu * grad_x log(mean exp(lambda * r))`` is added to the base
velocity. Reports the OpenBabel-FP2 Tanimoto similarity between generated
molecules and their sampled target fingerprints (guided vs unguided) plus
PoseBusters validity.

The paper's Table 3 number is the ``lambda=0.02, mu=0.5`` point of a four-point
grid at ``--num-mols 64`` (Slurm job 35670937); the same configuration at
``--num-mols 4096`` x 3 seeds gives a negative gain. See
``results/geom/PROVENANCE.md`` and ``notebooks/08_geom_fingerprint.ipynb``.

Example (paper configuration, one shard)::

    python -m multimfm.geom.mfm_steering --num-mols 64 --sample-steps 100 \\
        --mfm-steps 10 --value-samples 4 --value-batch-size 8 \\
        --guide-every 2 --guide-min-t 0.1 --guide-max-t 0.9 \\
        --mu 0.5 --reward-lambda 0.02 --seed 0 --device cuda --tag capgap_lam0.02_mu0.5

Ported from ``geom-drug-tabasco/glass/mfm_fingerprint_steering.py``.
"""

from __future__ import annotations

from multimfm.geom.steering import main as _main


def main() -> None:
    _main("mfm", __doc__)


if __name__ == "__main__":
    main()
