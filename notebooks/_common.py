"""Shared helpers for the paper notebooks.

Every notebook starts with

    RERUN = False
    from _common import *

With ``RERUN = False`` a notebook rebuilds its tables and figures from the small
files under ``results/`` on CPU. With ``RERUN = True`` it loads checkpoints and
reruns at small n on one GPU; full-size runs go through ``scripts/``.

``compare`` puts the value printed in the paper next to the value recomputed
from ``results/`` so any disagreement is visible rather than silently absorbed.
"""

from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CHECKPOINTS = REPO / "checkpoints"
DATA = REPO / "data"
FIGURES = REPO / "notebooks" / "figures"

for p in (REPO / "src", REPO / "third_party" / "tfg_flow"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MPLCONFIGDIR", str(REPO / "cache" / "matplotlib"))


def load_json(path: str | Path) -> dict:
    return json.loads((RESULTS / path).read_text())


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(RESULTS / path, **kwargs)


def compare(rows: list[dict], rtol: float = 0.0, atol: float | None = None) -> pd.DataFrame:
    """Tabulate paper vs recomputed values.

    Each row: ``{"item": str, "paper": float | str, "ours": float, "note": str (optional),
    "decimals": int (optional)}``.

    A row matches if the two agree to the paper's printed precision, or within
    ``atol`` / ``rtol`` when given. The returned frame carries the tolerance actually
    applied, so a row can never claim "match" under a tolerance nobody chose.

    **Give ``paper`` as the string exactly as the paper prints it** (``"0.40"``,
    ``"1.0"``) whenever it has trailing zeros. Printed precision cannot be recovered
    from a float: ``repr(0.40) == "0.4"``, which previously yielded tolerance 0.05
    instead of 0.005 and reported disagreements as matches. A float ``paper`` still
    works, but its precision is *inferred*; the ``tol_src`` column says so, and
    ``decimals`` overrides it.
    """
    out = []
    for r in rows:
        paper_raw, ours = r["paper"], r["ours"]
        paper = float(paper_raw) if paper_raw is not None else None
        if paper is None or ours is None or (isinstance(ours, float) and np.isnan(ours)):
            status, tol, tol_src = "n/a", None, ""
        else:
            if atol is not None:
                tol, tol_src = atol, "atol"
            elif r.get("decimals") is not None:
                tol, tol_src = 0.5 * 10 ** (-int(r["decimals"])) + 1e-12, "decimals"
            else:
                # Half a unit in the last printed place. Decimal reads that off the
                # literal uniformly for integers ("600" -> 0.5), trailing zeros
                # ("0.40" -> 0.005) and scientific notation ("1e-06" -> 5e-07);
                # counting characters after "." gets all three wrong.
                if isinstance(paper_raw, str):
                    lit, tol_src = paper_raw.strip(), "printed"
                else:
                    # Precision is inferred: a float literal has already lost any
                    # trailing zero, so "0.40" is indistinguishable from "0.4".
                    lit, tol_src = repr(paper_raw), "inferred"
                tol = 0.5 * float(10 ** Decimal(lit).as_tuple().exponent) + 1e-12
            if rtol:
                tol = max(tol, rtol * abs(paper))
            status = "match" if abs(ours - paper) <= tol else "MISMATCH"
        out.append({"item": r["item"], "paper": paper_raw, "ours": ours, "status": status,
                    "tol": tol, "tol_src": tol_src, "note": r.get("note", "")})
    return pd.DataFrame(out)


def savefig(fig, name: str, paper: bool = False) -> Path:
    """Save a notebook figure.

    The top level of ``notebooks/figures/`` is the **vetted paper set**, built by
    ``notebooks/figures/build.py`` from ``results/`` and named by provenance
    (``published_`` digitized from the submission, ``rerun_`` our re-measurement,
    ``new_`` no published counterpart). Notebook output therefore defaults to the
    ``diagnostics/`` subdirectory, so re-executing a notebook cannot drop an
    unvetted image next to the files destined for the paper. Pass ``paper=True``
    only for a figure that genuinely belongs in the paper set, and prefer adding
    it to ``build.py`` instead so it regenerates from ``results/`` alone.
    """
    out = FIGURES if paper else FIGURES / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight")
    return path


__all__ = [
    "REPO", "RESULTS", "CHECKPOINTS", "DATA", "FIGURES",
    "load_json", "load_csv", "compare", "savefig", "np", "pd", "json", "os", "sys", "Path",
]
