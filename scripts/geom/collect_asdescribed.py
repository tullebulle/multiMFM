#!/usr/bin/env python
"""Aggregate the as-described Table 3 rerun (authors' decision 1) into ``results/geom``.

Two stages:

``tune``  reads the held-out (lambda, mu) sweep and writes
          ``results/geom/table3_asdescribed_tuning.csv`` plus the selected point in
          ``results/geom/table3_asdescribed_selection.json``. Selection rule (fixed in
          advance): the largest paired Tanimoto gain on the held-out targets among the
          points whose guided PoseBusters validity is not more than ``--pb-drop``
          (default 0.02 = 2 points) below the unguided validity of the same run.

``eval``  reads the 3-seed x 4-shard evaluation and writes
          ``results/geom/table3_asdescribed_shards.csv`` (per shard),
          ``table3_asdescribed_seeds.csv`` (per seed, 4,096 molecules each) and
          ``table3_asdescribed.json`` (the headline mean +- SE over seeds, for Table 3
          and the GEOM row of Table 21). It also records the as-described run inside
          ``table21_geom_row.json`` without touching the submitted-run entries.

    python scripts/geom/collect_asdescribed.py tune
    python scripts/geom/collect_asdescribed.py eval
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "geom"
TUNE_DIR = RESULTS / "runs" / "asdescribed_tuning"
EVAL_DIR = RESULTS / "runs" / "asdescribed_eval"


def load_runs(directory: Path) -> list[dict]:
    files = sorted(glob.glob(str(directory / "*fingerprint_steering_*.json")))
    out = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        d["_file"] = os.path.relpath(f, REPO)
        out.append(d)
    return out


def paired(records: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Per-molecule (guided - unguided) over molecules whose FP2 is valid in both arms."""
    deltas, ug, g = [], [], []
    for d in records:
        u, gu = d["unguided"], d["guided"]
        tu, tg = u["tanimoto_real_per_mol"], gu["tanimoto_real_per_mol"]
        vu, vg = u["valid_per_mol"], gu["valid_per_mol"]
        ug += tu
        g += tg
        deltas += [tg[j] - tu[j] for j in range(len(tu)) if vu[j] and vg[j]]
    return deltas, ug, g


def mean_sem(xs) -> tuple[float, float]:
    xs = list(xs)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, REPO)} ({len(rows)} rows)")


def cmd_tune(args: argparse.Namespace) -> None:
    dirs = [Path(d) for d in args.dir] if args.dir else [TUNE_DIR]
    runs = [r for d in dirs for r in load_runs(d)]
    if not runs:
        raise SystemExit(f"no tuning results in {dirs}")
    rows = []
    for d in runs:
        deltas, ug, g = paired([d])
        gain, sem = mean_sem(deltas)
        u, gu = d["unguided"], d["guided"]
        rows.append({
            "reward_lambda": d["reward_lambda"], "mu": d["mu"],
            "mu_schedule": d.get("mu_schedule", "constant"),
            "guide_every": d.get("guide_every"), "guide_min_t": d.get("guide_min_t"),
            "guide_max_t": d.get("guide_max_t"),
            "schedule": ("recorded" if (d.get("guide_every") == 2
                                        and d.get("guide_min_t") == 0.1
                                        and d.get("guide_max_t") == 0.9)
                         else "every-step" if d.get("guide_every") == 1
                         else "other"),
            "num_mols": d["num_mols"], "seed": d["seed"],
            "unguided_tanimoto": u["tanimoto_real_mean"],
            "guided_tanimoto": gu["tanimoto_real_mean"],
            "paired_gain": gain, "paired_gain_sem": sem,
            # the guide's own objective, to separate "guidance did not act" from
            # "guidance acted but the guide disagrees with the counted fingerprint"
            "unguided_head_tanimoto": u["tanimoto_head_mean"],
            "guided_head_tanimoto": gu["tanimoto_head_mean"],
            "head_gain": gu["tanimoto_head_mean"] - u["tanimoto_head_mean"],
            "unguided_pb_valid": u.get("pb_valid"), "guided_pb_valid": gu.get("pb_valid"),
            "pb_drop": (None if u.get("pb_valid") is None or gu.get("pb_valid") is None
                        else u["pb_valid"] - gu["pb_valid"]),
            "guided_conversion": gu["conversion_rate"],
            "tuned_on_holdout": d.get("tuned_on_holdout"),
            "holdout_n": d.get("holdout_n"), "holdout_seed": d.get("holdout_seed"),
            "device": d.get("device"), "slurm_account": d.get("slurm_account"),
            "slurm_partition": d.get("slurm_partition"), "slurm_job_id": d.get("slurm_job_id"),
            "file": d["_file"],
        })
    rows.sort(key=lambda r: (r["mu_schedule"], r["schedule"], r["reward_lambda"], r["mu"]))
    write_csv(RESULTS / "table3_asdescribed_tuning.csv", rows)

    eligible = [r for r in rows if r["pb_drop"] is not None and r["pb_drop"] <= args.pb_drop]
    pool = eligible or rows
    best = max(pool, key=lambda r: r["paired_gain"])
    selection = {
        "_what": "(lambda, mu) selected for the as-described Table 3 rerun",
        "rule": (f"largest paired Tanimoto gain on the held-out targets among points whose "
                 f"guided PB validity is at most {args.pb_drop:.3f} below the unguided "
                 f"validity of the same run; if no point qualifies, the largest gain overall"),
        "any_point_qualified": bool(eligible),
        "grid_points": len(rows),
        "tuning_molecules_per_point": rows[0]["num_mols"],
        "holdout_n": rows[0]["holdout_n"], "holdout_seed": rows[0]["holdout_seed"],
        "holdout_disjoint_from_evaluation": True,
        "schedules_swept": sorted({r["schedule"] for r in rows}),
        "mu_schedules_swept": sorted({r["mu_schedule"] for r in rows}),
        "devices_swept": sorted({str(r["device"]) for r in rows}),
        "device_note": ("CPU shards are true fp32 and CUDA shards use TF32 matmuls, so points "
                        "are comparable in distribution but not bitwise across devices; the "
                        "constant-mu vs paper-coefficient comparison is only made within a "
                        "device (see results/geom/PROVENANCE.md)"),
        "selected": {"reward_lambda": best["reward_lambda"], "mu": best["mu"],
                     "mu_schedule": best["mu_schedule"],
                     "guide_every": best["guide_every"],
                     "guide_min_t": best["guide_min_t"], "guide_max_t": best["guide_max_t"],
                     "schedule": best["schedule"],
                     "paired_gain": best["paired_gain"], "paired_gain_sem": best["paired_gain_sem"],
                     "unguided_tanimoto": best["unguided_tanimoto"],
                     "guided_tanimoto": best["guided_tanimoto"],
                     "unguided_pb_valid": best["unguided_pb_valid"],
                     "guided_pb_valid": best["guided_pb_valid"]},
        "paper_literal_lambda_1": [
            {k: r[k] for k in ("reward_lambda", "mu", "schedule", "paired_gain",
                               "paired_gain_sem", "guided_tanimoto", "guided_pb_valid")}
            for r in rows if abs(r["reward_lambda"] - 1.0) < 1e-9
        ],
        "best_per_schedule": {
            sched: max((r for r in rows if r["schedule"] == sched),
                       key=lambda r: r["paired_gain"])
            for sched in sorted({r["schedule"] for r in rows})
        },
        "note": ("The paper states the reward shape exp(-||E(G)-c||^2) -- i.e. lambda=1 -- "
                 "but no numeric guidance strength mu, and no selection procedure. lambda=1 "
                 "is included in the grid so the literal reading is measured too."),
    }
    out = RESULTS / "table3_asdescribed_selection.json"
    out.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"wrote {os.path.relpath(out, REPO)}")
    print(f"\nselected lambda={best['reward_lambda']} mu={best['mu']} "
          f"(held-out gain {best['paired_gain']:+.4f} +- {best['paired_gain_sem']:.4f}, "
          f"PB {best['unguided_pb_valid']:.3f} -> {best['guided_pb_valid']:.3f})")
    print("\nrun the evaluation with:")
    sched_env = ""
    if best["mu_schedule"] != "constant":
        sched_env += f"MU_SCHEDULE={best['mu_schedule']} "
    if best["schedule"] != "recorded":
        sched_env = (f"GUIDE_EVERY={best['guide_every']} GUIDE_MIN_T={best['guide_min_t']} "
                     f"GUIDE_MAX_T={best['guide_max_t']} ")
    print(f"  LAMBDA={best['reward_lambda']} MU={best['mu']} {sched_env}"
          f"sbatch scripts/geom/table3_asdescribed.sbatch")


def cmd_eval(args: argparse.Namespace) -> None:
    runs = load_runs(Path(args.dir) if args.dir else EVAL_DIR)
    if not runs:
        raise SystemExit(f"no evaluation results in {args.dir or EVAL_DIR}")
    shard_rows, by_seed_group = [], {}
    for d in runs:
        m = re.search(r"_r(\d+)_s(\d+)_", d["tag"])
        group = int(m.group(1)) if m else 0
        deltas, ug, g = paired([d])
        gain, sem = mean_sem(deltas)
        u, gu = d["unguided"], d["guided"]
        shard_rows.append({
            "seed_group": group, "seed": d["seed"], "num_mols": d["num_mols"],
            "reward_lambda": d["reward_lambda"], "mu": d["mu"],
            "sample_steps": d["sample_steps"], "mfm_steps": d.get("mfm_steps"),
            "value_samples": d["value_samples"], "mu_schedule": d.get("mu_schedule", "constant"),
            "guide_every": d.get("guide_every"),
            "guide_min_t": d.get("guide_min_t"), "guide_max_t": d.get("guide_max_t"),
            "unguided_tanimoto": u["tanimoto_real_mean"],
            "guided_tanimoto": gu["tanimoto_real_mean"],
            "paired_gain": gain, "paired_gain_sem": sem,
            "unguided_head_tanimoto": u["tanimoto_head_mean"],
            "guided_head_tanimoto": gu["tanimoto_head_mean"],
            "head_gain": gu["tanimoto_head_mean"] - u["tanimoto_head_mean"],
            "unguided_pb_valid": u.get("pb_valid"), "guided_pb_valid": gu.get("pb_valid"),
            "unguided_conversion": u["conversion_rate"], "guided_conversion": gu["conversion_rate"],
            "targets_skipped": d.get("targets_skipped"),
            "targets_excluded_holdout": d.get("targets_excluded_holdout"),
            "file": d["_file"],
        })
        by_seed_group.setdefault(group, []).append(d)
    shard_rows.sort(key=lambda r: (r["seed_group"], r["seed"]))
    write_csv(RESULTS / f"{args.out_prefix}_shards.csv", shard_rows)

    seed_rows = []
    for group in sorted(by_seed_group):
        ds = by_seed_group[group]
        deltas, ug, g = paired(ds)
        gain, gsem = mean_sem(deltas)
        upb = [d["unguided"]["pb_valid"] for d in ds if d["unguided"].get("pb_valid") is not None]
        gpb = [d["guided"]["pb_valid"] for d in ds if d["guided"].get("pb_valid") is not None]
        seed_rows.append({
            "seed_group": group, "shards": len(ds), "n_molecules": len(ug),
            "n_paired_valid": len(deltas),
            "unguided_tanimoto": sum(ug) / len(ug), "guided_tanimoto": sum(g) / len(g),
            "paired_gain": gain, "paired_gain_sem": gsem,
            "unguided_pb_valid": sum(upb) / len(upb) if upb else None,
            "guided_pb_valid": sum(gpb) / len(gpb) if gpb else None,
            "seeds": ";".join(str(d["seed"]) for d in sorted(ds, key=lambda x: x["seed"])),
        })
    write_csv(RESULTS / f"{args.out_prefix}_seeds.csv", seed_rows)

    # Headline: mean +- SE ACROSS SEEDS (TFG-Flow's +- convention), n_seeds = len(seed_rows)
    g_mean, g_se = mean_sem(r["guided_tanimoto"] for r in seed_rows)
    u_mean, u_se = mean_sem(r["unguided_tanimoto"] for r in seed_rows)
    gain_mean, gain_se = mean_sem(r["paired_gain"] for r in seed_rows)
    gpb_mean, gpb_se = mean_sem(r["guided_pb_valid"] for r in seed_rows)
    upb_mean, upb_se = mean_sem(r["unguided_pb_valid"] for r in seed_rows)
    sel_path = RESULTS / "table3_asdescribed_selection.json"
    selection = json.loads(sel_path.read_text()) if sel_path.exists() else None
    complete = all(r["shards"] == args.shards for r in seed_rows) and len(seed_rows) == args.seed_groups

    headline = {
        "_what": ("Table 3 / Table 21 GEOM row rerun exactly as the paper describes it "
                  "(authors' decision 1, 2026-09-29)"),
        "complete": complete,
        "expected": {"seed_groups": args.seed_groups, "shards_per_seed": args.shards},
        "config": {
            "molecules_per_seed": sum(r["n_molecules"] for r in seed_rows) // max(len(seed_rows), 1),
            "seeds": [r["seeds"] for r in seed_rows],
            "sample_steps": shard_rows[0]["sample_steps"],
            "mfm_steps": shard_rows[0]["mfm_steps"],
            "value_samples": shard_rows[0]["value_samples"],
            "reward_lambda": shard_rows[0]["reward_lambda"],
            "mu": shard_rows[0]["mu"],
            "mu_schedule": shard_rows[0]["mu_schedule"],
            "guide_every": shard_rows[0]["guide_every"],
            "guide_min_t": shard_rows[0]["guide_min_t"],
            "guide_max_t": shard_rows[0]["guide_max_t"],
            "lambda_mu_selected_on": ("held-out target pool, disjoint from the evaluation "
                                      "targets (results/geom/table3_asdescribed_selection.json)"),
            "target_split": "val",
            "guide_head": "checkpoints/geom/structure_head/best.pth (trained by the authors)",
            "teacher": "checkpoints/geom/teacher/epoch=38-step=89310.ckpt (EMA weights)",
            "student": "checkpoints/geom/mfm_student/student_step_10000.pt",
            "pb_config": "configs/geom/posebusters_geom.yaml",
        },
        "multimfm": {"tanimoto_mean": g_mean, "tanimoto_se": g_se,
                     "pb_valid_mean": gpb_mean, "pb_valid_se": gpb_se},
        "base_generator_unguided": {"tanimoto_mean": u_mean, "tanimoto_se": u_se,
                                    "pb_valid_mean": upb_mean, "pb_valid_se": upb_se},
        "paired_gain": {"mean": gain_mean, "se": gain_se},
        "per_seed": seed_rows,
        "selection": None if selection is None else selection["selected"],
    }
    # The sweep may have grown after the evaluation was launched; say so rather than
    # silently implying the evaluation used the current best point.
    if selection is not None:
        sel = selection["selected"]
        headline["selection_matches_evaluated_config"] = bool(
            abs(float(sel["reward_lambda"]) - float(shard_rows[0]["reward_lambda"])) < 1e-12
            and abs(float(sel["mu"]) - float(shard_rows[0]["mu"])) < 1e-12
            and int(sel.get("guide_every", 2)) == int(shard_rows[0]["guide_every"])
        )
        headline["selection_grid_points_now"] = selection.get("grid_points")
        if not headline["selection_matches_evaluated_config"]:
            headline["selection_note"] = (
                "the evaluation was launched with the point selected from the sweep as it stood "
                "then; the sweep has since grown and its best point now differs -- see "
                "results/geom/table3_asdescribed_tuning.csv"
            )
    out = RESULTS / f"{args.out_prefix}.json"
    out.write_text(json.dumps(headline, indent=2) + "\n")
    print(f"wrote {os.path.relpath(out, REPO)}")

    # Record the run in the Table 21 file without touching the submitted-run entries.
    t21_path = RESULTS / "table21_geom_row.json"
    if t21_path.exists() and args.out_prefix == "table3_asdescribed":
        t21 = json.loads(t21_path.read_text())
        t21.setdefault("records", {})["asdescribed_rerun"] = {
            "slurm_job_array": args.job or "see results/geom/PROVENANCE.md",
            "N_per_seed": headline["config"]["molecules_per_seed"],
            "seeds": len(seed_rows),
            "unsteered_pb_pct": None if upb_mean is None else 100 * upb_mean,
            "unsteered_pb_se_pct": None if upb_se is None else 100 * upb_se,
            "multimfm_pb_pct": None if gpb_mean is None else 100 * gpb_mean,
            "multimfm_pb_se_pct": None if gpb_se is None else 100 * gpb_se,
            "pb_config": "configs/geom/posebusters_geom.yaml",
            "complete": complete,
            "note": ("as-described rerun (authors' decision 1): unguided arm shares the noise "
                     "and targets of the guided arm, so it is the Unsteered column for the "
                     "same molecules"),
        }
        t21_path.write_text(json.dumps(t21, indent=2) + "\n")
        print(f"updated {os.path.relpath(t21_path, REPO)}")

    print(f"\n{'seed':>6}{'n':>7}{'un_tani':>9}{'g_tani':>9}{'gain':>9}{'SEM':>8}{'un_pb':>7}{'g_pb':>7}")
    for r in seed_rows:
        print(f"{r['seed_group']:>6}{r['n_molecules']:>7}{r['unguided_tanimoto']:>9.4f}"
              f"{r['guided_tanimoto']:>9.4f}{r['paired_gain']:>+9.4f}{r['paired_gain_sem']:>8.4f}"
              f"{r['unguided_pb_valid']:>7.3f}{r['guided_pb_valid']:>7.3f}")
    print("-" * 62)
    print(f"multiMFM Tanimoto       {g_mean:.4f} +- {g_se:.4f} (SE over {len(seed_rows)} seeds)")
    print(f"base (unguided)         {u_mean:.4f} +- {u_se:.4f}")
    print(f"paired gain             {gain_mean:+.4f} +- {gain_se:.4f}")
    print(f"PB validity multiMFM    {100 * gpb_mean:.1f}% +- {100 * gpb_se:.1f}")
    print(f"PB validity base        {100 * upb_mean:.1f}% +- {100 * upb_se:.1f}")
    if not complete:
        print("\nINCOMPLETE: some shards are missing; rerun the array (finished shards are skipped).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune", help="aggregate the held-out (lambda, mu) sweep and select")
    t.add_argument("--dir", nargs="+", default=None,
                   help="run directories (default: results/geom/runs/asdescribed_tuning); "
                        "pass several to pool the GPU and CPU sweeps")
    t.add_argument("--pb-drop", type=float, default=0.02,
                   help="maximum tolerated PB-validity drop vs the unguided arm (default 0.02)")
    e = sub.add_parser("eval", help="aggregate the 3-seed evaluation")
    e.add_argument("--dir", default=None, help="run directory (default: results/geom/runs/asdescribed_eval)")
    e.add_argument("--out-prefix", default="table3_asdescribed",
                   help="basename prefix for the written result files")
    e.add_argument("--shards", type=int, default=4)
    e.add_argument("--seed-groups", type=int, default=3)
    e.add_argument("--job", default=None, help="Slurm array job id, recorded in the outputs")
    args = ap.parse_args()
    (cmd_tune if args.cmd == "tune" else cmd_eval)(args)


if __name__ == "__main__":
    main()
