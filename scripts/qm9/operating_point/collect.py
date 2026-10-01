#!/usr/bin/env python
"""Collect the QM9 (posterior steps x MC) operating-point runs into results/qm9/operating_point.

    python scripts/qm9/operating_point/collect.py <out_name> <run_root> [<run_root> ...]

Each run root is scanned for ``*/summary.json`` at any depth; the run's own ``args`` give
the split (``value_glass_steps`` x ``value_samples``), the property, the seed and whether
``--mfm-diagonal`` was set, so directory names carry no information and the reference
runs in ``outputs/qm9/table2_3seed/`` can be mixed in unchanged.

Writes into ``results/qm9/operating_point/<out_name>/``:

* ``summary.csv`` one row per (row, variant, property, seed, split): guided oracle MAE
  with a bootstrap 95% CI, PB-valid-only MAE, PB validity, unique rate, guide MAE, the
  measured generative and regressor NFE per molecule, the look-ahead candidate count and
  (multiMFM row only, from ``guided_step_logs.csv``) the mean guidance gradient RMS --
  the dose diagnostic the QM9 setting needs because it has no gradient-normalisation
  rule -- together with the posterior's own mean reward and the mean value estimate,
  which say how good the molecules the posterior drew were.
* ``paired.csv`` per-sample paired differences of guided oracle absolute error against
  the published 4 x 32 split, within the same (row, variant, property, seed) and matched
  on ``sample_idx``, with a SEM and a bootstrap CI over the pairs, plus the fraction of
  pairs improved.  Positive means the non-published split is worse.
* ``unguided_control.csv`` the unguided arm of the multiMFM row across splits: it must be
  identical at every split, because ``guided_sample`` captures and restores the torch RNG
  around the value gradient, so the production trajectory's noise does not depend on how
  many draws the estimator makes.  A non-zero spread here would invalidate the pairing.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "qm9" / "operating_point"

# run -> variant label, for both drivers.  Mirrors multimfm.qm9_results.RERUN_VARIANTS.
VARIANTS = {
    ("multimfm", "guided"): "multimfm",
    ("multimfm", "unguided"): "unsteered",
    ("ss", "best_lookahead_pbvalid"): "ss",            # Algorithm 3, the reported row
    ("ss", "best_lookahead_raw"): "ss_unfiltered",
    ("ss", "base_endpoint_pbvalid"): "ss_base_endpoint",
    ("ss", "base_endpoint_raw"): "ss_base_endpoint_unfiltered",
    ("ss", "mfm_valuescore_pbvalid"): "ss_mfm_valuescore",
    ("ss", "mfm_window_pbvalid"): "ss_mfm_window",
    ("ss", "final_endpoint"): "ss_trajectory_endpoint",
}
PROPS = ["cv", "mu", "alpha", "gap", "homo", "lumo"]
UNITS = {"cv": "cal/(mol K)", "mu": "D", "alpha": "Bohr^3",
         "gap": "meV", "homo": "meV", "lumo": "meV"}
# gap/homo/lumo are stored in Hartree per sample and reported in meV (the factor
# multimfm.qm9_results uses, so these tables are on Table 2's scale).
HARTREE_TO_MEV = 27211.386245988
ENERGY = ("gap", "homo", "lumo")


def scale(prop: str) -> float:
    return HARTREE_TO_MEV if prop in ENERGY else 1.0


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def row_kind(args: dict) -> str:
    """property_targets has --value-sampler / --guide-atomics; steer_search has
    --lookahead-source.  Distinguish on a key only one of them stores."""
    return "ss" if "lookahead_source" in args else "multimfm"


def read_per_sample(path: Path) -> dict[str, list[dict]]:
    by_run: dict[str, list[dict]] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            by_run.setdefault(r["run"], []).append(r)
    return by_run


STEP_LOG_KEYS = ("grad_coords_rms", "grad_atomics_rms", "posterior_reward_mean",
                 "posterior_reward_std", "value_mean")


def step_log_means(run_dir: Path) -> dict[str, float | str]:
    """Means over guided steps of the quantities guided_step_logs.csv records.

    `grad_*_rms` is the dose diagnostic (QM9 applies no gradient-normalisation rule, so a
    split that improves the MAE with a larger gradient is confounded with a larger dose).
    `posterior_reward_mean` and `value_mean` are the estimator diagnostic: they say how
    good the molecules the posterior actually drew were, independently of the guidance.
    Only the multiMFM driver writes this file; steer_search does not.
    """
    f = run_dir / "guided_step_logs.csv"
    out: dict[str, float | str] = {f"{k}_mean": "" for k in STEP_LOG_KEYS}
    if not f.exists():
        return out
    acc: dict[str, list[float]] = {k: [] for k in STEP_LOG_KEYS}
    with f.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("guided", "").lower() != "true":
                continue
            for k in STEP_LOG_KEYS:
                v = r.get(k, "")
                if v not in ("", None):
                    acc[k].append(float(v))
    for k, vals in acc.items():
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
    return out


def scan(roots: list[Path]) -> list[dict]:
    recs: list[dict] = []
    for root in roots:
        for sj in sorted(root.rglob("summary.json")):
            d = json.loads(sj.read_text())
            args = d.get("args") or {}
            if "value_glass_steps" not in args:
                continue
            kind = row_kind(args)
            ps = sj.parent / "per_sample.csv"
            if not ps.exists():
                print(f"  no per_sample.csv beside {sj}", file=sys.stderr)
                continue
            by_run = read_per_sample(ps)
            nfe = ((d.get("nfe") or {}).get("by_phase") or {})
            gen = sum(v.get("nfe_generative", 0.0) for v in nfe.values())
            reg = sum(v.get("nfe_regressor", 0.0) for v in nfe.values())
            cand = (d.get("num_candidates") or {}).get("mfm")
            slog = step_log_means(sj.parent)
            steps = int(args["value_glass_steps"])
            mc = int(args["value_samples"])
            for run, rows in by_run.items():
                variant = VARIANTS.get((kind, run))
                if variant is None:
                    continue
                k = scale(args["property_name"])
                err = np.array([float(r["oracle_abs_error"]) for r in rows]) * k
                gerr = np.array([float(r["guide_abs_error"]) for r in rows]) * k
                pb = np.array([r.get("pb_valid", "") for r in rows])
                pbv = np.array([x.lower() in ("true", "1") for x in pb]) if pb[0] != "" else None
                smi = [r.get("smiles", "") for r in rows]
                nz = [s for s in smi if s]
                lo, hi = boot_ci(err)
                rec = {
                    "row": kind, "variant": variant, "run": run,
                    "property": args["property_name"], "unit": UNITS[args["property_name"]],
                    "steps": steps, "mc": mc, "nfe_value_per_guided_step": steps * mc,
                    "diagonal": bool(args.get("mfm_diagonal")),
                    "split": f"{steps}x{mc}" + ("" if args.get("mfm_diagonal") else "-fm"),
                    "n": len(err), "seed": args["seed"], "target_seed": args["target_seed"],
                    "mu": args["mu"], "reward_scale": args["reward_scale"],
                    "oracle_mae": float(err.mean()),
                    "oracle_mae_ci_lo": lo, "oracle_mae_ci_hi": hi,
                    "oracle_mae_sem": float(err.std(ddof=1) / math.sqrt(len(err))),
                    "oracle_median_ae": float(np.median(err)),
                    "guide_mae": float(gerr.mean()),
                    "pb_valid": float(pbv.mean()) if pbv is not None else "",
                    "oracle_mae_pb_valid_only": (float(err[pbv].mean()) if pbv is not None
                                                 and pbv.any() else ""),
                    "unique_rate_total": (len(set(nz)) / len(err)) if nz else "",
                    "nfe_generative_per_mol": gen, "nfe_regressor_per_mol": reg,
                    "candidates_per_mol": cand if cand is not None else "",
                    **{f"{k}_mean": slog[f"{k}_mean"] for k in STEP_LOG_KEYS},
                    "wall_s_total": (d.get("timing_s") or {}).get("total", ""),
                    "wall_s_selection": (d.get("timing_s") or {}).get("selection_and_eval", ""),
                    "device": d.get("device", ""), "gpu_mem_gb": d.get("cuda_max_memory_gb", ""),
                    "run_dir": str(sj.parent.relative_to(REPO)),
                }
                recs.append(rec)
                rec["_err"] = err
                rec["_idx"] = np.array([int(r["sample_idx"]) for r in rows])
    return recs


def paired(recs: list[dict]) -> list[dict]:
    """Every split against the published 4 x 32 diagonal split, matched on sample_idx."""
    out = []
    key = lambda r: (r["row"], r["variant"], r["property"], r["seed"])
    refs = {key(r): r for r in recs
            if r["steps"] == 4 and r["mc"] == 32 and r["diagonal"]}
    for r in recs:
        ref = refs.get(key(r))
        if ref is None or ref is r:
            continue
        common = np.intersect1d(r["_idx"], ref["_idx"])
        if len(common) == 0:
            continue
        # sample_idx is unique within a run, so plain index maps are enough
        ma = {i: e for i, e in zip(r["_idx"], r["_err"])}
        mb = {i: e for i, e in zip(ref["_idx"], ref["_err"])}
        diff = np.array([ma[i] - mb[i] for i in common])
        lo, hi = boot_ci(diff)
        out.append({
            "row": r["row"], "variant": r["variant"], "property": r["property"],
            "unit": r["unit"], "seed": r["seed"], "n_pairs": len(common),
            "split": r["split"], "vs": ref["split"],
            "mae": r["oracle_mae"], "mae_ref": ref["oracle_mae"],
            "paired_diff": float(diff.mean()),
            "paired_sem": float(diff.std(ddof=1) / math.sqrt(len(diff))),
            "paired_ci_lo": lo, "paired_ci_hi": hi,
            "frac_improved": float((diff < 0).mean()),
        })
    return out


def unguided_control(recs: list[dict]) -> list[dict]:
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        if r["variant"] != "unsteered":
            continue
        groups.setdefault((r["property"], r["seed"]), []).append(r)
    for (prop, seed), rs in sorted(groups.items()):
        idx = rs[0]["_idx"]
        stack = np.stack([r["_err"] for r in rs])
        out.append({"property": prop, "seed": seed, "n_splits": len(rs),
                    "splits": "|".join(sorted(r["split"] for r in rs)),
                    "unguided_oracle_mae": float(stack[0].mean()),
                    "max_per_id_spread": float(np.abs(stack - stack[0]).max()),
                    "n": len(idx)})
    return out


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  nothing to write to {path}")
        return
    cols = [c for c in rows[0] if not c.startswith("_")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  {path.relative_to(REPO)}  ({len(rows)} rows)")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    name, roots = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    recs = scan([p if p.is_absolute() else REPO / p for p in roots])
    print(f"scanned {len(recs)} (run, variant) records")
    order = {p: i for i, p in enumerate(PROPS)}
    recs.sort(key=lambda r: (r["row"], r["variant"], order.get(r["property"], 99),
                             r["seed"], -r["mc"], not r["diagonal"]))
    out = RESULTS / name
    write(out / "summary.csv", recs)
    write(out / "paired.csv", paired(recs))
    write(out / "unguided_control.csv", unguided_control(recs))


if __name__ == "__main__":
    main()
