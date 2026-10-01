"""Repo-relative default locations for the GEOM-Drugs artifacts.

Nothing in ``multimfm.geom`` contains an absolute path: every default below is
derived from this file's location, so the package works from any working
directory and after the repository is moved or cloned elsewhere.

The artifacts themselves are not in git; they are copied in (and checksummed)
with ``scripts/manifest.py`` -- see ``docs/provenance/geom/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from multimfm.paths import REPO_ROOT  # repo root, shared with the QM9 drivers

CHECKPOINTS = REPO_ROOT / "checkpoints" / "geom"
DATA = REPO_ROOT / "data" / "geom"
CONFIGS = REPO_ROOT / "configs" / "geom"
RESULTS = REPO_ROOT / "results" / "geom"
TFG_ROOT = REPO_ROOT / "third_party" / "tfg_flow"

#: Base (teacher) flow model used for Table 3 and the GEOM row of Table 19.
DEFAULT_TEACHER = CHECKPOINTS / "teacher" / "epoch=38-step=89310.ckpt"
#: Selected multiMFM student (Stage-2b step 10,000); GEOM row of Table 20.
DEFAULT_STUDENT = CHECKPOINTS / "mfm_student" / "student_step_10000.pt"
#: Fingerprint guide head (TFG-Flow EGNN ``Predictor``, trained by the authors).
DEFAULT_STRUCTURE_HEAD = CHECKPOINTS / "structure_head" / "best.pth"
#: Prepared FP2 fingerprint LMDB (only the val split ships; see provenance).
DEFAULT_GEOM_FP_DIR = DATA / "geom_fp"
#: Processed GEOM-Drugs molecules (public on HF; needed only for (re)training).
DEFAULT_GEOM_TRAIN_PT = DATA / "processed_geom_train.pt"
#: PoseBusters configuration used for the GEOM experiments (Table 3 / Table 21).
DEFAULT_PB_CONFIG = CONFIGS / "posebusters_geom.yaml"


def ensure_tfg_importable() -> Path:
    """Put the vendored TFG-Flow tree on ``sys.path`` and return its root.

    TFG-Flow is vendored as upstream ships it (top-level ``dataloader``,
    ``diffusion``, ``networks`` packages), so it is imported by adding its root
    to ``sys.path`` rather than by an editable install. Mirrors what
    ``multimfm.qm9_data.load_tfg_regressor`` does for the QM9 regressors.
    """
    root = str(TFG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return TFG_ROOT
