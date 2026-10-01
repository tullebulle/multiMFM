#!/usr/bin/env python
"""Aggregate the GEOM guidance-diagnosis grid into results/geom/diagnostics/.

Reads the per-point JSONs written by ``multimfm.geom.mfm_steering`` (the same
directory ``scripts/geom/tune_lambda_mu.sbatch`` writes to) and emits one CSV row
per (lambda, mu, mu_schedule) point with the *paired* Tanimoto gain and its paired
SEM, the head-predicted gain, PoseBusters validity and the guide's accuracy on the
molecules it generated.

Deliberately separate from ``scripts/geom/collect_asdescribed.py tune``: that
command rewrites ``results/geom/table3_asdescribed_{tuning.csv,selection.json}``,
which record the (lambda, mu) selection the completed 4,096 x 3 evaluation
(Slurm 49274410) was run under. Adding sign-test and lambda = 1 points to that
selection after the fact would silently change what the evaluation is documented
as having selected. This writes to ``results/geom/diagnostics/`` instead and
selects nothing -- see the pre-registered rule in
``docs/geom_guidance_diagnosis.md`` §0.

    python scripts/geom/collect_guidance_grid.py
    python scripts/geom/collect_guidance_grid.py --dir results/geom/runs/asdescribed_tuning
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO / "results" / "geom" / "runs" / "asdescribed_tuning"
DEFAULT_OUT = REPO / "results" / "geom" / "diagnostics" / "guidance_grid.csv"

FIELDS = [
    "reward_lambda", "mu", "mu_schedule", "sign", "num_mols", "seed",
    "unguided_tanimoto", "guided_tanimoto", "paired_gain", "paired_gain_sem",
    "gain_over_sem", "unguided_head_tanimoto", "guided_head_tanimoto", "head_gain",
    "unguided_pb_valid", "guided_pb_valid", "pb_drop",
    "unguided_guide_accuracy", "guided_guide_accuracy",
    "unguided_mean_atoms", "guided_mean_atoms", "mean_atoms_changed",
    "guide_every", "guide_min_t", "guide_max_t", "mfm_steps", "value_samples",
    "sample_steps", "tuned_on_holdout", "holdout_n", "device", "slurm_job_id", "file",
]


def paired_gain(rec: dict) -> tuple[float, float, int]:
    """Molecule-level paired gain and its SEM over molecules valid in both arms."""
    u, g = rec["unguided"], rec["guided"]
    tu, tg = u["tanimoto_real_per_mol"], g["tanimoto_real_per_mol"]
    vu, vg = u["valid_per_mol"], g["valid_per_mol"]
    d = [b - a for a, b, x, y in zip(tu, tg, vu, vg) if x and y]
    if not d:
        return float("nan"), float("nan"), 0
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return m, float("nan"), n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return m, math.sqrt(var / n), n


def row_for(path: Path) -> dict:
    rec = json.loads(path.read_text())
    u, g = rec["unguided"], rec["guided"]
    gain, sem, _ = paired_gain(rec)
    mu = float(rec["mu"])
    return {
        "reward_lambda": rec["reward_lambda"],
        "mu": mu,
        "mu_schedule": rec.get("mu_schedule", "constant"),
        "sign": "negative" if mu < 0 else ("zero" if mu == 0 else "positive"),
        "num_mols": rec["num_mols"], "seed": rec["seed"],
        "unguided_tanimoto": u["tanimoto_real_mean"],
        "guided_tanimoto": g["tanimoto_real_mean"],
        "paired_gain": gain, "paired_gain_sem": sem,
        "gain_over_sem": (gain / sem) if sem and not math.isnan(sem) else float("nan"),
        "unguided_head_tanimoto": u["tanimoto_head_mean"],
        "guided_head_tanimoto": g["tanimoto_head_mean"],
        "head_gain": g["tanimoto_head_mean"] - u["tanimoto_head_mean"],
        "unguided_pb_valid": u.get("pb_valid"), "guided_pb_valid": g.get("pb_valid"),
        "pb_drop": (
            None if u.get("pb_valid") is None or g.get("pb_valid") is None
            else u["pb_valid"] - g["pb_valid"]),
        # guide_accuracy_mean was added 2026-09-29; older JSONs do not have it.
        "unguided_guide_accuracy": u.get("guide_accuracy_mean"),
        "guided_guide_accuracy": g.get("guide_accuracy_mean"),
        "unguided_mean_atoms": u["mean_atoms"], "guided_mean_atoms": g["mean_atoms"],
        "mean_atoms_changed": u["mean_atoms"] != g["mean_atoms"],
        "guide_every": rec["guide_every"], "guide_min_t": rec["guide_min_t"],
        "guide_max_t": rec["guide_max_t"], "mfm_steps": rec.get("mfm_steps"),
        "value_samples": rec["value_samples"], "sample_steps": rec["sample_steps"],
        "tuned_on_holdout": rec.get("tuned_on_holdout"),
        "holdout_n": rec.get("holdout_n"), "device": rec.get("device"),
        "slurm_job_id": rec.get("slurm_job_id"),
        "file": str(path.relative_to(REPO)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, nargs="+", default=[DEFAULT_DIR])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    paths: list[Path] = []
    for d in args.dir:
        paths.extend(sorted(d.glob("*fingerprint_steering_*.json")))
    rows = []
    for p in paths:
        try:
            rows.append(row_for(p))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  skipped {p.name}: {exc}")
    rows.sort(key=lambda r: (r["mu_schedule"], float(r["reward_lambda"]), float(r["mu"])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    hdr = (f"{'sched':>9}{'lambda':>8}{'mu':>7}{'unguided':>10}{'guided':>10}"
           f"{'gain':>11}{'SEM':>9}{'g/SEM':>7}{'head gain':>11}{'PB drop':>9}"
           f"{'guide acc':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        acc = r["guided_guide_accuracy"]
        pbd = r["pb_drop"]
        print(f"{r['mu_schedule']:>9}{float(r['reward_lambda']):>8.3g}"
              f"{float(r['mu']):>7.3g}{r['unguided_tanimoto']:>10.5f}"
              f"{r['guided_tanimoto']:>10.5f}{r['paired_gain']:>+11.5f}"
              f"{r['paired_gain_sem']:>9.5f}{r['gain_over_sem']:>+7.2f}"
              f"{r['head_gain']:>+11.5f}"
              f"{(f'{pbd:+.4f}' if pbd is not None else 'n/a'):>9}"
              f"{(f'{acc:.4f}' if acc is not None else 'n/a'):>11}")

    # The pre-registered bar, applied mechanically (see docs/geom_guidance_diagnosis.md).
    passing = [r for r in rows
               if not math.isnan(r["gain_over_sem"]) and r["gain_over_sem"] > 2.0
               and r["head_gain"] > 0]
    print(f"\n{len(rows)} points; {len(passing)} clear the pre-registered bar "
          f"(paired gain > 2 x SEM AND head gain > 0)")
    for r in passing:
        print(f"  {r['mu_schedule']} lambda={r['reward_lambda']} mu={r['mu']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
