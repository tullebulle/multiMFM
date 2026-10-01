#!/usr/bin/env python
"""Aggregate every rescue grid point into one CSV and apply the pre-registered bar.

Reads results/geom/rescue/{regime,dose}/*.json (the steering-driver records) and
writes results/geom/rescue/rescue_grid.csv. Every point that exists is listed; no
point is selected. The bar is the same as the diagnosis's: paired counted-FP2 gain
> 2 x paired SEM AND head-predicted gain > 0.

The paired SEM is not stored by the driver, so it is recomputed from the
per-molecule Tanimoto arrays of the two arms, which are paired by construction
(same initial noise, same targets, same sampler draws).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESCUE = REPO / "results" / "geom" / "rescue"
OUT = RESCUE / "rescue_grid.csv"

FIELDS = ["group", "tag", "reward_lambda", "mu", "mu_schedule", "guide_min_t", "guide_max_t",
          "guide_every", "value_samples", "mfm_steps", "guide_atomics", "guidance_rule",
          "guidance_max_coord_rms", "realised_disp_A", "realised_frac_retyped",
          "n_guided_steps", "num_mols", "unguided", "guided", "gain", "gain_sem", "g_over_sem",
          "head_gain", "pb_unguided", "pb_guided", "guide_acc_guided", "clears_bar",
          "device", "slurm_job_id", "file"]


def paired(a, b):
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return m, math.sqrt(var / n)


def main() -> None:
    rows = []
    for group in ("regime", "dose"):
        for f in sorted((RESCUE / group).glob("*.json")):
            r = json.loads(f.read_text())
            u, g = r["unguided"], r["guided"]
            gain, sem = paired(u["tanimoto_real_per_mol"], g["tanimoto_real_per_mol"])
            head_gain = g["tanimoto_head_mean"] - u["tanimoto_head_mean"]
            rows.append({
                "group": group, "tag": r.get("tag"),
                "reward_lambda": r.get("reward_lambda"), "mu": r.get("mu"),
                "mu_schedule": r.get("mu_schedule"), "guide_min_t": r.get("guide_min_t"),
                "guide_max_t": r.get("guide_max_t"), "guide_every": r.get("guide_every"),
                "value_samples": r.get("value_samples"), "mfm_steps": r.get("mfm_steps"),
                "guide_atomics": r.get("guide_atomics"),
                "guidance_rule": r.get("guidance_rule", "stock"),
                "guidance_max_coord_rms": r.get("guidance_max_coord_rms", 0.0),
                "realised_disp_A": r.get("realised_coord_rms_displacement_angstrom"),
                "realised_frac_retyped": r.get("realised_frac_atoms_retyped"),
                "n_guided_steps": r.get("n_guided_steps"),
                "num_mols": r.get("num_mols"),
                "unguided": round(u["tanimoto_real_mean"], 6),
                "guided": round(g["tanimoto_real_mean"], 6),
                "gain": round(gain, 6), "gain_sem": round(sem, 6),
                "g_over_sem": round(gain / sem, 2) if sem else None,
                "head_gain": round(head_gain, 6),
                "pb_unguided": u.get("pb_valid"), "pb_guided": g.get("pb_valid"),
                "guide_acc_guided": g.get("guide_accuracy_mean"),
                "clears_bar": bool(gain > 2 * sem and head_gain > 0) if sem else False,
                "device": r.get("device"), "slurm_job_id": r.get("slurm_job_id"),
                "file": str(f.relative_to(REPO)) if REPO in f.parents else str(f),
            })
    rows.sort(key=lambda r: (r["group"], r.get("tag")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    hdr = (f"{'tag':<44s} {'lam':>5s} {'mu':>7s} {'win':>11s} {'ev':>3s} {'vs':>3s} "
           f"{'dispA':>7s} {'gain':>9s} {'sem':>8s} {'g/sem':>6s} {'head':>9s} {'pb':>7s} bar")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{str(r['tag'])[:44]:<44s} {(r['reward_lambda'] or 0):>5g} {(r['mu'] or 0):>7.4g} "
              f"{str(r['guide_min_t'])+'-'+str(r['guide_max_t']):>11s} "
              f"{(r['guide_every'] or 0):>3d} {(r['value_samples'] or 0):>3d} "
              f"{(r['realised_disp_A'] if r['realised_disp_A'] is not None else float('nan')):>7.4f} "
              f"{r['gain']:>+9.5f} {r['gain_sem']:>8.5f} {str(r['g_over_sem']):>6s} "
              f"{r['head_gain']:>+9.5f} "
              f"{(r['pb_guided'] if r['pb_guided'] is not None else float('nan')):>7.4f} "
              f"{'YES' if r['clears_bar'] else 'no'}")
    n_bar = sum(r["clears_bar"] for r in rows)
    print(f"\n{len(rows)} points; {n_bar} clear the pre-registered bar "
          f"(paired gain > 2 x SEM AND head gain > 0)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
