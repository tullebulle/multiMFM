"""GEOM-Drugs fingerprint steering by Diffusion Posterior Sampling.

    python -m multimfm.geom.dps_steering --num-mols 256 --reward-kind fingerprint_tanimoto \
        --reward-lambda 20 --mu 2.0 --num-mols 256 --pb

This is the like-for-like baseline for multiMFM on this task. Both steer a single
trajectory by a value gradient, use the same base generator, the same guide head, the same
targets and the same evaluation; the only difference is how the value is estimated. DPS
takes the reward at the model's one-step deterministic endpoint, $r(\\mathbb E[X_1 \\mid X_t])$,
while multiMFM takes a log-mean-exp over flow-map posterior samples. DPS therefore has no
posterior budget to spend, which is the whole of the comparison.

Neither a flow-map student nor GLASS inner steps are needed, so ``--mfm-checkpoint`` and
``--value-glass-steps`` are accepted and ignored.
"""
from multimfm.geom.steering import main as _main


def main():
    _main("denoiser", __doc__)


if __name__ == "__main__":
    main()
