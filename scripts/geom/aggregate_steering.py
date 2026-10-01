#!/usr/bin/env python
"""Aggregate GEOM fingerprint-steering result JSONs into one CSV + a printed table.

    python scripts/geom/aggregate_steering.py <run-dir> [--out summary.csv]

Reads every ``*fingerprint_steering_*.json`` written by
``multimfm.geom.{mfm,glass}_steering`` in ``<run-dir>`` and reports, per (lambda, mu):
the unguided and guided Tanimoto, the **paired** per-molecule gain with its SEM over
molecules whose FP2 is valid in both arms (the statistic that matters -- the unpaired
difference of means at n=64 is about one SEM), and PoseBusters validity. Shards of the
same replicate (tags ``..._r<R>_s<SEED>``) are pooled exactly, as in the 4,096x3 run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re


def paired_gain(records: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Per-molecule (guided - unguided) over molecules valid in both arms."""
    deltas, ug, g = [], [], []
    for d in records:
        u, gu = d["unguided"], d["guided"]
        tu, tg = u.get("tanimoto_real_per_mol"), gu.get("tanimoto_real_per_mol")
        if tu is None or tg is None:
            continue
        vu, vg = u.get("valid_per_mol"), gu.get("valid_per_mol")
        ug += tu
        g += tg
        for j in range(len(tu)):
            ok = True if (vu is None or vg is None) else bool(vu[j]) and bool(vg[j])
            if ok:
                deltas.append(tg[j] - tu[j])
    return deltas, ug, g


def mean_sem(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def group_key(d: dict) -> tuple:
    """(lambda, mu, replicate) -- shards of one replicate pool together."""
    m = re.search(r"_r(\d+)_s\d+$", d["tag"])
    rep = int(m.group(1)) if m else None
    return (float(d["reward_lambda"]), float(d["mu"]), rep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="CSV path (default: <run-dir>/summary.csv)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "*fingerprint_steering_*.json")))
    if not files:
        raise SystemExit(f"no steering result JSONs in {args.run_dir}")
    groups: dict[tuple, list[dict]] = {}
    for f in files:
        d = json.load(open(f))
        groups.setdefault(group_key(d), []).append(d)

    rows = []
    for (lam, mu, rep), ds in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or -1)):
        deltas, ug, g = paired_gain(ds)
        gain, sem = mean_sem(deltas)
        upb = [d["unguided"].get("pb_valid") for d in ds if d["unguided"].get("pb_valid") is not None]
        gpb = [d["guided"].get("pb_valid") for d in ds if d["guided"].get("pb_valid") is not None]
        rows.append({
            "reward_lambda": lam, "mu": mu, "replicate": "" if rep is None else rep,
            "shards": len(ds), "num_mols": sum(d["num_mols"] for d in ds),
            "n_paired_valid": len(deltas),
            "unguided_tanimoto": sum(ug) / len(ug) if ug else float("nan"),
            "guided_tanimoto": sum(g) / len(g) if g else float("nan"),
            "paired_gain": gain, "paired_gain_sem": sem,
            "unguided_pb_valid": sum(upb) / len(upb) if upb else "",
            "guided_pb_valid": sum(gpb) / len(gpb) if gpb else "",
            "value_sampler": ds[0].get("value_sampler", ""),
            "mfm_steps": ds[0].get("mfm_steps", ""),
            "sample_steps": ds[0]["sample_steps"],
            "target_selection": ds[0].get("target_selection", ""),
            "targets_skipped": sum(d.get("targets_skipped", 0) for d in ds),
        })

    out = args.out or os.path.join(args.run_dir, "summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = f"{'lam':>6}{'mu':>6}{'rep':>5}{'n':>7}{'un_tani':>9}{'g_tani':>9}{'gain':>9}{'SEM':>8}{'un_pb':>7}{'g_pb':>7}"
    print(hdr)
    for r in rows:
        upb = f"{r['unguided_pb_valid']:.3f}" if r["unguided_pb_valid"] != "" else "n/a"
        gpb = f"{r['guided_pb_valid']:.3f}" if r["guided_pb_valid"] != "" else "n/a"
        print(f"{r['reward_lambda']:>6}{r['mu']:>6}{str(r['replicate']):>5}{r['num_mols']:>7}"
              f"{r['unguided_tanimoto']:>9.4f}{r['guided_tanimoto']:>9.4f}"
              f"{r['paired_gain']:>+9.4f}{r['paired_gain_sem']:>8.4f}{upb:>7}{gpb:>7}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
