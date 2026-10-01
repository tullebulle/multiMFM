"""Repo-relative default locations for the toy example (no absolute paths)."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parents[1]
CHECKPOINTS = REPO / "checkpoints" / "toy"
RESULTS = REPO / "results" / "toy"
# Probe / diagnostic figures are run outputs, not paper artifacts: keep them out of git.
FIGURES = REPO / "outputs" / "toy" / "figures"

# Checkpoint names used by the original ``norway`` repo (terminal log, notebook).
BASE_CKPT = CHECKPOINTS / "ckpt_uniform_color_square.pt"
MFM_DIAGONAL_CKPT = CHECKPOINTS / "mfm_diagonal_10000.pt"
MFM_CONSISTENCY_CKPT = CHECKPOINTS / "mfm_consistency_10000.pt"
