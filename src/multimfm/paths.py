"""Repo-relative default locations for checkpoints, caches and outputs.

Every default in the ``multimfm`` command-line drivers resolves through here, so
the drivers work from any working directory (the package is installed editable
from ``src/``). Nothing in this module imports torch or tabasco.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS = REPO_ROOT / "checkpoints"
BASE_FLOW_CKPT = CHECKPOINTS / "base_flow" / "model_step_100000.pt"
MFM_STUDENT_CKPT = CHECKPOINTS / "mfm_student" / "student_step_1000.pt"
REGRESSOR_DIR = CHECKPOINTS / "regressors"

HIST_DIR = REPO_ROOT / "cache" / "histograms"
OUTPUTS = REPO_ROOT / "outputs"
RESULTS = REPO_ROOT / "results"


def tabasco_utils_file(name: str) -> Path:
    """Path of a data file shipped in ``tabasco/utils`` (without importing tabasco)."""
    spec = importlib.util.find_spec("tabasco")
    if spec is None or not spec.submodule_search_locations:
        return REPO_ROOT / "src" / "tabasco" / "utils" / name
    return Path(list(spec.submodule_search_locations)[0]) / "utils" / name


def posebusters_config_path() -> Path:
    """PoseBusters config used for every QM9 validity number (no strain-energy check)."""
    return tabasco_utils_file("posebusters_no_strain.yaml")


def default_hist_cache(property_name: str, partition: str = "train_flow", bins: int = 100) -> Path:
    """Length-conditioned QM9 property-target histogram used to draw guidance targets.

    The files in ``cache/histograms`` are byte-identical to the histograms used for
    the paper runs (``hist_<prop>_train_flow_100.json`` in the original repo; the
    non-SS runs read the same content under ``cache/qm9_property_histograms/``).
    """
    return HIST_DIR / f"hist_{property_name}_{partition}_{bins}.json"
