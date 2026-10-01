#!/usr/bin/env python
"""Summarise the operating-point grids: per-split means, and paired differences on shared ids.

    python scripts/dna/operating_point/collect.py c0       results/dna/operating_point/fig7_grid_nfe32_n64
    python scripts/dna/operating_point/collect.py gradient results/dna/operating_point/table18_grid/gc
    python scripts/dna/operating_point/collect.py reward   results/dna/operating_point/table13_grid/motif_nfe8
    python scripts/dna/operating_point/collect.py fig4     results/dna/operating_point/fig4_grid
    python scripts/dna/operating_point/collect.py fig8     results/dna/operating_point/fig8_grid_nfe32

Every CI is a 2,000-resample percentile bootstrap with ``default_rng(0)`` -- the method behind
Table 1 and behind ``scripts/dna/rescue/collect_deficit_rescue.py`` -- so the numbers here are
directly comparable with the deficit-rescue tables. Paired statistics are computed on the ids
(or (probe, repeat) cells) that two splits actually share.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BOOT, SEED = 2000, 0


def boot_ci(v: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(v)
    idx = rng.integers(0, n, size=(BOOT, n))
    m = v[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Mean of a - b on the shared index, its SEM and bootstrap CI."""
    shared = a.index.intersection(b.index)
    d = (a.loc[shared] - b.loc[shared]).to_numpy()
    lo, hi = boot_ci(d)
    return {"n_shared": len(shared), "mean_diff": float(d.mean()),
            "sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "frac_first_better": float((d < 0).mean())}


# --------------------------------------------------------------------------- C0 demos
def collect_c0(root: Path) -> None:
    frames = [pd.read_csv(p) for p in sorted(root.glob("target_*/sample_scores.csv"))]
    if not frames:
        raise SystemExit(f"no target_*/sample_scores.csv under {root}")
    df = pd.concat(frames, ignore_index=True)
    recs = []
    for (target, config), g in df.groupby(["target", "config"]):
        err = (g["oracle_guided"] - target).abs().to_numpy()
        gerr = (g["guide_guided"] - target).abs().to_numpy()
        lo, hi = boot_ci(err)
        glo, ghi = boot_ci(gerr)
        recs.append({
            "target": target, "config": config, "n": len(g), "kind": g["kind"].iloc[0],
            "steps": int(g["n_steps"].iloc[0]), "mc": int(g["mc"].iloc[0]),
            "nfe_value": int(g["nfe_value"].iloc[0]), "scale_rule": g["scale_rule"].iloc[0],
            "guidance_frac": float(g["guidance_frac"].iloc[0]),
            "artifact": g["artifact"].iloc[0],
            "unguided_oracle_mae": float((g["oracle_unguided"] - target).abs().mean()),
            "guided_oracle_mae": float(err.mean()),
            "guided_oracle_mae_ci_lo": lo, "guided_oracle_mae_ci_hi": hi,
            "guided_guide_mae": float(gerr.mean()),
            "guided_guide_mae_ci_lo": glo, "guided_guide_mae_ci_hi": ghi,
            "guide_within_0.10": float((gerr < 0.10).mean()),
            "guide_mean": float(g["guide_guided"].mean()),
            "frac_improved": float((((g["oracle_unguided"] - target).abs() - err) > 0).mean()),
            "grad_norm_raw_mean": float(g["grad_norm_raw_mean"].mean()),
            "grad_norm_post_mean": float(g["grad_norm_post_mean"].mean()),
        })
    summary = pd.DataFrame(recs).sort_values(["target", "guided_oracle_mae"])
    summary.to_csv(root / "summary.csv", index=False)
    print(summary.drop(columns=["artifact"]).to_string(index=False))

    rows = []
    for target, gt in df.groupby("target"):
        for metric, col in (("oracle", "oracle_guided"), ("guide", "guide_guided")):
            series = {c: (g.set_index("sample_idx")[col] - target).abs()
                      for c, g in gt.groupby("config")}
            for a, b in combinations(sorted(series), 2):
                rows.append({"target": target, "metric": metric, "first": a, "second": b,
                             **paired(series[a], series[b])})
    pd.DataFrame(rows).to_csv(root / "paired.csv", index=False)
    print(f"\nwrote {root/'summary.csv'} and {root/'paired.csv'}")


# --------------------------------------------------------------------------- gradient grids
def collect_gradient(root: Path) -> None:
    out = []
    for length_dir in sorted(root.glob("L*")):
        f = length_dir / "gradient_grid.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        recs = []
        for (config, mc), g in df.groupby(["config", "mc"]):
            per_probe = g.groupby("probe")["mae"].mean().to_numpy()
            lo, hi = boot_ci(per_probe)
            cos = g.groupby("probe")["cosine_similarity"].mean().to_numpy()
            rel = g.groupby("probe")["relative_l2_error"].mean().to_numpy()
            recs.append({
                "length": int(g["length"].iloc[0]), "config": config,
                "kind": g["kind"].iloc[0], "student": g["student"].iloc[0],
                "steps": int(g["steps"].iloc[0]), "mc": int(mc),
                "nfe": int(g["nfe"].iloc[0]), "n_probes": len(per_probe),
                "n_repeats": int(g["repeat"].nunique()),
                "mean_mae": float(per_probe.mean()), "median_state_mae": float(np.median(per_probe)),
                "ci95_low": lo, "ci95_high": hi,
                "mean_cosine": float(cos.mean()), "mean_relative_l2": float(rel.mean()),
            })
        s = pd.DataFrame(recs).sort_values(["nfe", "mean_mae"])
        s.to_csv(length_dir / "grid_summary.csv", index=False)
        out.append(s)
        # Paired on shared (probe, repeat) cells, within each NFE group.
        rows = []
        key = ["probe", "repeat"]
        for nfe, gn in df.groupby("nfe"):
            series = {c: g.set_index(key)["mae"] for c, g in gn.groupby("config")}
            for a, b in combinations(sorted(series), 2):
                rows.append({"length": int(gn["length"].iloc[0]), "nfe": int(nfe),
                             "first": a, "second": b, **paired(series[a], series[b])})
        pd.DataFrame(rows).to_csv(length_dir / "grid_paired.csv", index=False)
        print(f"--- {length_dir.name}")
        print(s.to_string(index=False))
    if out:
        pd.concat(out, ignore_index=True).to_csv(root / "grid_summary_all_lengths.csv", index=False)
        print(f"\nwrote {root/'grid_summary_all_lengths.csv'}")


# --------------------------------------------------------------------------- Table 13 grids
def collect_reward(root: Path) -> None:
    for length_dir in sorted(root.glob("L*")):
        s = length_dir / "summary.csv"
        if s.exists():
            print(f"--- {length_dir}")
            print(pd.read_csv(s).to_string(index=False))
        f = length_dir / "sample_scores.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        rows = []
        target = float(df["target_score"].iloc[0])
        series = {c: (g.set_index("sample_id")["score_guided"] - target).abs()
                  for c, g in df.groupby("config")}
        for a, b in combinations(sorted(series), 2):
            rows.append({"first": a, "second": b, **paired(series[a], series[b])})
        pd.DataFrame(rows).to_csv(length_dir / "paired.csv", index=False)
        print(f"wrote {length_dir/'paired.csv'}")


# --------------------------------------------------------------------------- Fig 4 shards
def collect_fig4(root: Path) -> None:
    recs = []
    for p in sorted(root.glob("*.json")):
        if p.name in ("student_mode.json", "grid_summary.json"):
            continue
        s = json.loads(p.read_text())
        cfg = s["config_params"]
        recs.append({
            "config": s["config"], "steps": int(cfg.get("nfe_value", 1)), "mc": int(cfg["n_mc"]),
            "nfe_value": int(cfg.get("nfe_value", 1)) * int(cfg["n_mc"]),
            "repeat": s["repeat"], "n_outputs": s["n_outputs"],
            "nfe_gen_per_output": s["nfe"]["nfe_gen_per_output"],
            "mae": s["metrics"]["mae"], "frac10": s["metrics"]["frac10"],
            "oracle_mae": s["metrics"]["oracle_mae"],
            "device_name": s["meta"].get("device_name"),
            # NOT the loaded student: run_steering hard-codes this field. See student_mode.json.
            "shard_meta_dmfm_field": s["meta"]["checkpoints"].get("dmfm"),
            "seconds": s["meta"]["elapsed_s"],
        })
    if not recs:
        raise SystemExit(f"no shards under {root}")
    mode = root / "student_mode.json"
    if mode.exists():
        info = json.loads(mode.read_text())
        for r in recs:
            r["student_mode"] = info["student_mode"]
            r["student"] = info["resolved"].get(str(r["steps"]), "")
    df = pd.DataFrame(recs).sort_values(["nfe_gen_per_output", "mae"])
    df.to_csv(root / "grid_summary.csv", index=False)
    cols = [c for c in df.columns if c not in ("shard_meta_dmfm_field", "device_name")]
    print(df[cols].to_string(index=False))
    # paired on shared sample index within an NFE group
    rows = []
    per = {}
    for p in sorted(root.glob("*.json")):
        if p.name in ("student_mode.json", "grid_summary.json"):
            continue
        s = json.loads(p.read_text())
        cfg = s["config_params"]
        nfe = int(cfg.get("nfe_value", 1)) * int(cfg["n_mc"])
        per.setdefault(nfe, {})[s["config"]] = pd.Series(
            np.abs(np.asarray(s["per_sample"]["guide_score"]) - s["meta"]["target_c0"]))
    for nfe, series in per.items():
        for a, b in combinations(sorted(series), 2):
            rows.append({"nfe_value": nfe, "first": a, "second": b, **paired(series[a], series[b])})
    pd.DataFrame(rows).to_csv(root / "grid_paired.csv", index=False)
    print(f"\nwrote {root/'grid_summary.csv'} and {root/'grid_paired.csv'}")


# --------------------------------------------------------------------------- Fig 8 traces
def collect_fig8(root: Path, target: float = 0.30) -> None:
    """One row per (arm, split): the final hard-decoded C0 of each traced sample.

    Fig 8 shows a single trajectory, so the per-sample columns matter as much as the mean: a
    split can improve the mean over the four traced ids and still make the *displayed* id worse.
    """
    recs = []
    for d in sorted(root.glob("*_s*mc*")):
        f = d / "trace.csv"
        if not f.exists():
            continue
        tr = pd.read_csv(f)
        arm, split = d.name.rsplit("_s", 1)
        steps, mc = split.split("mc")
        last = tr.sort_values("step").groupby(["sample_idx", "arm"]).tail(1)
        g = last[last.arm == "guided"].set_index("sample_idx")["endpoint_hard_score"]
        u = last[last.arm == "unguided"].set_index("sample_idx")["endpoint_hard_score"]
        row = {"arm": arm, "steps": int(steps), "mc": int(mc),
               "nfe_value": int(steps) * int(mc), "n_samples": len(g),
               "guided_abs_err_mean": float((g - target).abs().mean()),
               "unguided_abs_err_mean": float((u - target).abs().mean()),
               "paired_improvement_mean": float(((u - target).abs() - (g - target).abs()).mean()),
               "grad_norm_mean_guided": float(tr[(tr.arm == "guided") & tr.guidance_applied]
                                              ["grad_norm"].mean()),
               "dir": str(d)}
        for sid in sorted(g.index):
            row[f"c0_guided_id{sid}"] = float(g.loc[sid])
            row[f"abs_err_id{sid}"] = float(abs(g.loc[sid] - target))
        recs.append(row)
    if not recs:
        raise SystemExit(f"no traces under {root}")
    df = pd.DataFrame(recs).sort_values(["arm", "steps"])
    df.to_csv(root / "grid_summary.csv", index=False)
    cols = [c for c in df.columns if c != "dir"]
    print(df[cols].to_string(index=False))
    print(f"\nwrote {root/'grid_summary.csv'}")



# --------------------------------------------------------------------------- Table 18 validation
PUBLISHED_T18 = Path("results/dna/dmfm_vs_glass")


def validate_gradient(root: Path) -> None:
    """Paired check of the grid's steps=4 rows against the published Table 18 cells.

    ``make_heldout_probe_states``, the reference noise pool and the nested candidate pools are all
    drawn from **CPU** generators seeded on (seed, length, probe, repeat), and a smaller draw is a
    row-major prefix of a larger one, so a grid run at the published ``--seed`` and ``--t_eval``
    sees the *same* probe states and the *same* MC pools as the published run for every
    (probe, repeat, mc) cell it covers. The published run is ``dmfm_steps = 4`` with the 4-step ESD
    student, which the grid reproduces as ``dmfm4_fm4``; any difference on shared cells is
    arithmetic (TF32 vs fp32) and CUDA nondeterminism, not protocol.
    """
    reward = root.name
    rows = []
    for length_dir in sorted(root.glob("L*")):
        f = length_dir / "gradient_grid.csv"
        pub = PUBLISHED_T18 / reward / length_dir.name / "gradient_mae_per_condition.csv"
        if not f.exists() or not pub.exists():
            continue
        mine = pd.read_csv(f)
        mine = mine[mine.config == "dmfm4_fm4"].set_index(["probe", "repeat", "mc"])["mae"]
        theirs = pd.read_csv(pub).set_index(["probe", "repeat", "mc"])["mae"]
        shared = mine.index.intersection(theirs.index)
        if not len(shared):
            continue
        a, b = mine.loc[shared], theirs.loc[shared]
        rel = ((a - b).abs() / b.abs().clip(lower=1e-12))
        rows.append({"reward": reward, "length": length_dir.name, "n_cells": len(shared),
                     "mean_mine": float(a.mean()), "mean_published": float(b.mean()),
                     "max_abs_diff": float((a - b).abs().max()),
                     "median_rel_diff": float(rel.median()), "max_rel_diff": float(rel.max())})
    if not rows:
        raise SystemExit(f"no shared cells between {root} and {PUBLISHED_T18/reward}")
    df = pd.DataFrame(rows)
    df.to_csv(root / "validation_vs_published_table18.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {root/'validation_vs_published_table18.csv'}")


WHAT = {"c0": collect_c0, "gradient": collect_gradient, "reward": collect_reward,
        "fig4": collect_fig4, "fig8": collect_fig8,
        "validate": validate_gradient}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in WHAT:
        raise SystemExit(f"usage: collect.py {{{'|'.join(WHAT)}}} <root>")
    WHAT[sys.argv[1]](Path(sys.argv[2]))
