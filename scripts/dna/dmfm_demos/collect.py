#!/usr/bin/env python
"""Collect the dMFM demo reruns (Fig 7, Fig 8, Tables 4-6) into results/dna/dmfm_demos/.

Reads the run trees under ``outputs/dna/dmfm_demos/`` (plus the Table 1 dMFM/GLASS n=1000 sets
that Tables 4-6 are computed from) and writes the small tracked artifacts:

    results/dna/dmfm_demos/fig7/ladder_summary.csv     one row per (arm, guidance_frac)
    results/dna/dmfm_demos/fig7/histograms.csv         50-bin guide-score histograms
    results/dna/dmfm_demos/fig7/<arm>/<tag>/…          args.json, summary.json, oracle/
    results/dna/dmfm_demos/fig8/<arm>/…                trace.csv, metadata.json (written in place)
    results/dna/dmfm_demos/diversity/<arm>/…           Tables 4-5 CSVs + tables.md
    results/dna/dmfm_demos/table6/<arm>/…              Table 6 Evo 2 CSVs/JSON

Oracle scoring of the Fig 7 runs (independent b_train park_cnn oracle, RC-averaged, the Table 1
protocol) is done here on CPU if a run has no ``oracle/`` directory yet.

    python scripts/dna/dmfm_demos/collect.py             # everything that exists
    python scripts/dna/dmfm_demos/collect.py --what fig7
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from dmfm import paths  # noqa: E402

OUT = REPO / "outputs" / "dna" / "dmfm_demos"
RES = REPO / "results" / "dna" / "dmfm_demos"
HIST_BINS = 50
HIST_RANGE = (-2.0, 2.0)  # fixed so every strength's histogram is directly comparable


# ------------------------------------------------------------------------------ helpers

def oracle_score(run_dir: Path) -> Path | None:
    """Score one run dir with the independent oracle (Table 1 protocol), on CPU."""
    out = run_dir / "oracle"
    if (out / "oracle_summary.csv").is_file():
        return out
    cmd = [
        sys.executable, "-m", "dmfm.experiments.score_c0_oracle",
        "--oracle_ckpt", str(paths.c0_oracle()),
        "--oracle_model_type", "park_cnn",
        "--run_glob", str(run_dir / "sample_scores.csv"),
        "--out_dir", str(out),
        "--reverse_complement_average",
        "--bootstrap", "2000", "--seed", "0", "--device", "cpu",
    ]
    env = {"PYTHONPATH": str(REPO / "src"), "MPLCONFIGDIR": str(REPO / "cache" / "matplotlib")}
    import os

    proc = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env}, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  oracle scoring FAILED for {run_dir}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        return None
    return out


def copy_small(src: Path, dst: Path, names) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)


# ------------------------------------------------------------------------------ Fig 7

def collect_fig7() -> None:
    root = OUT / "fig7"
    if not root.is_dir():
        print("no outputs/dna/dmfm_demos/fig7; skipping Fig 7")
        return
    rows, hist_rows = [], []
    for arm_dir in sorted(root.iterdir()):
        if not arm_dir.is_dir():
            continue
        for run_dir in sorted(arm_dir.iterdir()):
            if not (run_dir / "summary.json").is_file():
                continue
            args = json.loads((run_dir / "args.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            df = pd.read_csv(run_dir / "sample_scores.csv")
            target = float(args["target_c0"])
            orc_dir = oracle_score(run_dir)
            orc = {}
            if orc_dir is not None:
                osum = pd.read_csv(orc_dir / "oracle_summary.csv")
                for _, r in osum.iterrows():
                    orc[str(r["set"])] = r
            row = {
                "arm": arm_dir.name,
                "guidance_frac": float(args["guidance_frac"]),
                "n": int(len(df)),
                "target_c0": target,
                "posterior": args["posterior"],
                "grad_normalize": bool(args["grad_normalize"]),
                "student_ckpt": args.get("student_ckpt"),
                "mc": int(args["mc"]),
                "nfe_value": int(args["nfe_value"]),
                "nfe_sample": int(args["nfe_sample"]),
                "seed": int(args["seed"]),
                # guide-scored (the quantity the published Fig 7 histograms plot)
                "guide_unguided_mean": summary["unguided"]["mean"],
                "guide_guided_mean": summary["guided"]["mean"],
                "guide_unguided_std": summary["unguided"]["std"],
                "guide_guided_std": summary["guided"]["std"],
                "guide_unguided_mae": summary["unguided"]["mean_abs_to_target"],
                "guide_guided_mae": summary["guided"]["mean_abs_to_target"],
                "guide_guided_frac_within_0.05": summary["guided"]["frac_within_0.05"],
                "guide_guided_frac_within_0.10": summary["guided"]["frac_within_0.10"],
                "guide_unguided_frac_within_0.10": summary["unguided"]["frac_within_0.10"],
                "guide_guided_median": summary["guided"]["median"],
                "guide_guided_p05": summary["guided"]["p05"],
                "guide_guided_p95": summary["guided"]["p95"],
                "frac_moved_toward_target": summary["delta"]["frac_improved"],
                "identical_to_unguided": bool((df["seq_guided"] == df["seq_unguided"]).all()),
            }
            for label in ("unguided", "guided"):
                if label in orc:
                    row[f"oracle_{label}_mean"] = float(orc[label]["oracle_score_mean"])
                    row[f"oracle_{label}_mae"] = float(orc[label]["oracle_target_mae"])
                    row[f"oracle_{label}_mae_ci_low"] = float(orc[label]["oracle_target_mae_ci95_low"])
                    row[f"oracle_{label}_mae_ci_high"] = float(orc[label]["oracle_target_mae_ci95_high"])
            if "paired_guidance_effect" in orc:
                e = orc["paired_guidance_effect"]
                row["oracle_paired_mae_improvement"] = float(e["oracle_target_mae"])
                row["oracle_paired_mae_improvement_ci_low"] = float(e["oracle_target_mae_ci95_low"])
                row["oracle_paired_mae_improvement_ci_high"] = float(e["oracle_target_mae_ci95_high"])
                row["oracle_frac_improved"] = float(e["fraction_guided_improved"])
            rows.append(row)

            edges = np.linspace(*HIST_RANGE, HIST_BINS + 1)
            for label in ("unguided", "guided"):
                counts, _ = np.histogram(df[label].to_numpy(), bins=edges)
                for lo, hi, c in zip(edges[:-1], edges[1:], counts):
                    hist_rows.append({
                        "arm": arm_dir.name, "guidance_frac": float(args["guidance_frac"]),
                        "set": label, "bin_low": float(lo), "bin_high": float(hi), "count": int(c),
                    })
            dst = RES / "fig7" / arm_dir.name / run_dir.name
            copy_small(run_dir, dst,
                       ["args.json", "summary.json", "run_env.json", "histogram.png", "cdf.png"])
            if orc_dir is not None:
                copy_small(orc_dir, dst / "oracle", ["oracle_summary.csv"])
            # Compact per-sample scores (no sequences): ~30 kB, enough to redraw the exact
            # histograms in a notebook. Column names match what regen_plots.fig7 expects.
            cols = ["sample_idx", "unguided", "guided"]
            if (orc_dir is not None) and (orc_dir / "oracle_per_sample.csv").is_file():
                op = pd.read_csv(orc_dir / "oracle_per_sample.csv")
                small = df[cols].merge(op[["sample_idx", "oracle_unguided", "oracle_guided"]],
                                       on="sample_idx", how="left")
            else:
                small = df[cols]
            small.to_csv(dst / "scores.csv", index=False)
            row["scores_csv"] = str((dst / "scores.csv").relative_to(RES))
            row["n_samples"] = row["n"]
            row["unguided_mean"] = row["guide_unguided_mean"]
            row["guided_mean"] = row["guide_guided_mean"]
            row["unguided_mean_abs_to_target"] = row["guide_unguided_mae"]
            row["guided_mean_abs_to_target"] = row["guide_guided_mae"]
    if not rows:
        print("Fig 7: no completed runs yet")
        return
    (RES / "fig7").mkdir(parents=True, exist_ok=True)
    lad = pd.DataFrame(rows).sort_values(["arm", "guidance_frac"]).reset_index(drop=True)
    lad.to_csv(RES / "fig7" / "ladder_summary.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(RES / "fig7" / "histograms.csv", index=False)
    # index.csv carries the column names notebook 03's Fig 7 cells already read from the GLASS
    # regeneration (`results/dna/regen/fig7/index.csv`), so those cells need only a path change.
    # `scores_csv` is relative to results/dna/dmfm_demos.
    idx_cols = ["arm", "guidance_frac", "n_samples", "target_c0", "unguided_mean", "guided_mean",
                "unguided_mean_abs_to_target", "guided_mean_abs_to_target", "scores_csv"]
    lad[idx_cols].to_csv(RES / "fig7" / "index.csv", index=False)
    print(f"Fig 7: {len(lad)} runs -> {RES / 'fig7' / 'ladder_summary.csv'}")
    cols = ["arm", "guidance_frac", "n", "guide_guided_mean", "guide_guided_mae",
            "oracle_guided_mean", "oracle_guided_mae", "guide_guided_frac_within_0.10"]
    print(lad[[c for c in cols if c in lad]].to_string(index=False))


# ------------------------------------------------------------------------------ Tables 4-6

def collect_diversity() -> None:
    root = OUT / "diversity"
    if not root.is_dir():
        print("no outputs/dna/dmfm_demos/diversity; skipping Tables 4-5")
        return
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        copy_small(arm_dir, RES / "diversity" / arm_dir.name, [
            "table6_baselines.csv", "table7_generated.csv", "table7_additions.csv",
            "diversity_summary_full.csv", "size_matched_test_null_summary.csv",
            "sample_counts.csv", "unguided_identity.json", "run_metadata.json", "tables.md",
            "token_order_permutation_check.csv",
        ])
        print(f"Tables 4-5 [{arm_dir.name}] -> {RES / 'diversity' / arm_dir.name}")


def collect_table6() -> None:
    root = OUT / "table6"
    if not root.is_dir():
        print("no outputs/dna/dmfm_demos/table6; skipping Table 6")
        return
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        names = [p.name for p in arm_dir.iterdir()
                 if p.is_file() and p.suffix in {".csv", ".json", ".md"}
                 and p.name != "evo2_per_sequence.csv" and p.stat().st_size < 4_000_000]
        dst = RES / "table6" / arm_dir.name
        copy_small(arm_dir, dst, names)
        # Per-sequence NLLs are worth tracking (they let the bootstrap CIs be recomputed), but
        # `seq` duplicates results/dna/c0_guidance/{dmfm,glass_matched}_n1000, `run_dir` repeats a
        # long constant per row, and perplexity/orientation are derivable. Keep the rest; rows
        # stay joinable on source/set/target_c0/sample_idx.
        per = arm_dir / "evo2_per_sequence.csv"
        if per.is_file():
            df = pd.read_csv(per)
            drop = ("seq", "run_dir", "evo2_perplexity", "evo2_orientation", "evo2_scored_positions")
            df.drop(columns=[c for c in drop if c in df], inplace=True)
            for c in df.columns:
                if df[c].dtype.kind == "f":
                    df[c] = df[c].round(6)
            df.to_csv(dst / "evo2_per_sequence_nll.csv", index=False)
        print(f"Table 6 [{arm_dir.name}] -> {dst} ({len(names)} files + per-sequence NLLs)")


def collect_fig8() -> None:
    root = RES / "fig8"
    if not root.is_dir():
        print("no results/dna/dmfm_demos/fig8; skipping Fig 8")
        return
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        t = arm_dir / "trace.csv"
        if not t.is_file():
            continue
        df = pd.read_csv(t)
        g = df[(df.arm == "guided") & df.guidance_applied]
        print(f"Fig 8 [{arm_dir.name}]: {len(df)} rows, {df.sample_idx.nunique()} samples, "
              f"guided-window mean |g| = {g.grad_norm.mean():.3g}, "
              f"frac |g| > clip = {(g.grad_norm > 10).mean():.3g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--what", nargs="*", default=["fig7", "fig8", "diversity", "table6"])
    a = p.parse_args()
    for what in a.what:
        {"fig7": collect_fig7, "fig8": collect_fig8,
         "diversity": collect_diversity, "table6": collect_table6}[what]()


if __name__ == "__main__":
    main()
