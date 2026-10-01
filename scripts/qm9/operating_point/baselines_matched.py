#!/usr/bin/env python
"""The alpha budget ladder: every steering method, every budget knob, one NFE axis.

    python scripts/qm9/operating_point/baselines_matched.py

Reads the already-swept baselines in ``outputs/qm9/baselines_v2/seed{1,2,3}/`` (best-of-N
over N, DPS over mu / guide_every / sample_steps, Feynman-Kac over K and lambda, and the
existing steer-search MC ladder at 4 posterior steps) together with the (steps x MC)
split grid of ``outputs/qm9/operating_point/evalseeds_alpha_n100/``, all of which are the
same property, the same n = 100, the same seeds 1-3 and the same target seeds 2027-2029,
so every arm is paired on ``sample_idx``.

Writes ``results/qm9/operating_point/baselines_alpha_matched.csv``: one row per
(method, config, variant) with the mean over seeds, the SEM over seeds, the measured
generative and regressor NFE per molecule, and -- where the sample ids allow it -- the
per-sample paired difference against the published multiMFM-SS 4 x 32 arm.

Why both NFE columns: the generative axis (base flow model + MFM student forwards) is the
one the paper matches on, and the splits of a fixed steps x MC product are exactly matched
on it.  The regressor axis is not matched: it scales with MC alone, so an MC-heavy split
costs more guide-network calls at the same generative budget, and best-of-N costs almost
none.  Reporting one without the other would flatter whichever method the axis suits.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "qm9" / "operating_point" / "baselines_alpha_matched.csv"
BASE = REPO / "outputs" / "qm9" / "baselines_v2"
SPLITS = REPO / "outputs" / "qm9" / "operating_point" / "evalseeds_alpha_n100"
PROP = "alpha"
SEEDS = (1, 2, 3)


def per_sample(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            by[r["run"]].append((int(r["sample_idx"]), float(r["oracle_abs_error"]),
                                 r.get("pb_valid", "")))
    out = {}
    for run, rows in by.items():
        rows.sort()
        out[run] = (np.array([r[0] for r in rows]),
                    np.array([r[1] for r in rows]),
                    np.array([str(r[2]).lower() in ("true", "1") for r in rows]))
    return out


def nfe_of(js: dict) -> tuple[float, float]:
    n = js.get("nfe") or {}
    if "by_phase" in n:
        ph = n["by_phase"]
        return (sum(v.get("nfe_generative", 0.0) for v in ph.values()),
                sum(v.get("nfe_regressor", 0.0) for v in ph.values()))
    return (float(n.get("nfe_generative", n.get("nfe_base_per_mol", 0.0)) or 0.0),
            float(n.get("nfe_regressor", n.get("nfe_regressor_per_mol", 0.0)) or 0.0))


def collect() -> dict[tuple[str, str, str], dict[int, tuple]]:
    """(method, config, run) -> seed -> (idx, err, pb, nfe_gen, nfe_reg)."""
    arms: dict[tuple[str, str, str], dict[int, tuple]] = defaultdict(dict)
    for seed in SEEDS:
        root = BASE / f"seed{seed}"
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if not d.is_dir():
                    continue
                js = next(iter(sorted(d.glob(f"*_{PROP}_seed{seed}.json"))), None)
                cs = next(iter(sorted(d.glob("*_per_sample.csv"))), None)
                if js is not None and cs is not None:      # baselines.py output
                    meta = json.loads(js.read_text())
                    gen, reg = nfe_of(meta)
                    method = js.name.split(f"_{PROP}_")[0]
                    for run, (idx, err, pb) in per_sample(cs).items():
                        arms[(method, d.name, run)][seed] = (idx, err, pb, gen, reg)
                    continue
                sj, ps = d / "summary.json", d / "per_sample.csv"
                if sj.exists() and ps.exists():            # steer_search output
                    meta = json.loads(sj.read_text())
                    if int(meta.get("args", {}).get("seed", -1)) != seed:
                        continue
                    gen, reg = nfe_of(meta)
                    a = meta["args"]
                    cfg = (f"{d.name} ({a['value_glass_steps']}x{a['value_samples']})")
                    for run, (idx, err, pb) in per_sample(ps).items():
                        arms[("steer_search", cfg, run)][seed] = (idx, err, pb, gen, reg)
    if SPLITS.is_dir():
        for sj in sorted(SPLITS.rglob("summary.json")):
            meta = json.loads(sj.read_text())
            a = meta.get("args") or {}
            if "value_glass_steps" not in a:
                continue
            seed = int(a["seed"])
            gen, reg = nfe_of(meta)
            kind = "ss_split" if "lookahead_source" in a else "multimfm_split"
            cfg = f"{a['value_glass_steps']}x{a['value_samples']}"
            ps = sj.parent / "per_sample.csv"
            if not ps.exists():
                continue
            for run, (idx, err, pb) in per_sample(ps).items():
                arms[(kind, cfg, run)][seed] = (idx, err, pb, gen, reg)
    return arms


def main() -> None:
    arms = collect()
    ref_key = ("ss_split", "4x32", "best_lookahead_pbvalid")
    ref = arms.get(ref_key)
    rows = []
    for (method, cfg, run), per in sorted(arms.items()):
        seeds = sorted(per)
        maes = [per[s][1].mean() for s in seeds]
        pbs = [per[s][2].mean() for s in seeds]
        rec = {
            "method": method, "config": cfg, "run": run, "n_seeds": len(seeds),
            "seeds": "|".join(map(str, seeds)),
            "oracle_mae_mean": float(np.mean(maes)),
            "oracle_mae_sem_over_seeds": (float(np.std(maes, ddof=1) / math.sqrt(len(maes)))
                                          if len(maes) > 1 else ""),
            "pb_valid_mean": float(np.mean(pbs)),
            "nfe_generative_per_mol": per[seeds[0]][3],
            "nfe_regressor_per_mol": per[seeds[0]][4],
        }
        if ref is not None and (method, cfg, run) != ref_key:
            shared = [s for s in seeds if s in ref]
            diffs = []
            for s in shared:
                i1, e1 = per[s][0], per[s][1]
                i2, e2 = ref[s][0], ref[s][1]
                m2 = dict(zip(i2.tolist(), e2.tolist()))
                common = [i for i in i1.tolist() if i in m2]
                m1 = dict(zip(i1.tolist(), e1.tolist()))
                diffs.extend(m1[i] - m2[i] for i in common)
            if diffs:
                d = np.array(diffs)
                rec["paired_vs_ss_4x32"] = float(d.mean())
                rec["paired_sem"] = float(d.std(ddof=1) / math.sqrt(len(d)))
                rec["n_pairs"] = len(d)
        rows.append(rec)
    rows.sort(key=lambda r: (r["nfe_generative_per_mol"], r["method"], r["config"]))
    cols = ["method", "config", "run", "n_seeds", "seeds", "oracle_mae_mean",
            "oracle_mae_sem_over_seeds", "pb_valid_mean", "nfe_generative_per_mol",
            "nfe_regressor_per_mol", "paired_vs_ss_4x32", "paired_sem", "n_pairs"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{OUT.relative_to(REPO)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
