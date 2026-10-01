"""Train the GEOM-Drugs fingerprint guide head (the ``E`` of paper Section 3.2).

Thin wrapper around the vendored TFG-Flow trainer
``third_party/tfg_flow/train_structure_head.py``: it puts the vendored tree on
``sys.path``, points the data/checkpoint defaults at the repo's ``data/geom`` and
``checkpoints/geom`` directories, and delegates. The head is TFG-Flow's EGNN
``Predictor`` (h=128, 6 layers, 1024 outputs) trained with BCE against the
OpenBabel FP2 label; accuracy is reported as Tanimoto similarity.

The head shipped with the paper (``checkpoints/geom/structure_head/best.pth``,
val Tanimoto 0.852) was trained by the authors -- not taken off the shelf -- in
four legs (1 GPU, then a 3-GPU DDP run and two warm-started 3-GPU continuations,
the last with lr 1e-3 cosine over 40 epochs); see
``checkpoints/geom/structure_head/train_args.json``.

Single GPU::

    python -m multimfm.geom.train_structure_head --epochs 40 --batch_size 64 --lr 1e-4

3-GPU DDP (as used for the released head)::

    torchrun --standalone --nproc_per_node=3 -m multimfm.geom.train_structure_head \\
        --batch_size 256 --lr 3e-4 --epochs 10

Requires the *train* split of the prepared fingerprint LMDB, which is not
shipped with the repo; see ``docs/provenance/geom/data.md`` for how to rebuild it.
"""

from __future__ import annotations

import sys

from multimfm.geom.paths import CHECKPOINTS, DEFAULT_GEOM_FP_DIR, ensure_tfg_importable


def main() -> None:
    ensure_tfg_importable()
    import train_structure_head as upstream  # noqa: PLC0415 (vendored TFG-Flow script)

    # Repo-relative defaults, overridable on the command line as usual.
    if not any(a.startswith("--data_dir") for a in sys.argv[1:]):
        sys.argv += ["--data_dir", str(DEFAULT_GEOM_FP_DIR)]
    if not any(a.startswith("--log_dir") for a in sys.argv[1:]):
        sys.argv += ["--log_dir", str(CHECKPOINTS / "structure_head_rerun")]
    upstream.main()


if __name__ == "__main__":
    main()
