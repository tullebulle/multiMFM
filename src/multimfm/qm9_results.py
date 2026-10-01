"""Collect and tabulate the QM9 property-guidance results (Tables 2, 19-22).

The paper's QM9 numbers come from two sets of run records in the original
``tabasco`` repo (backed up under
``/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/tabasco``):

* multiMFM (non-SS): ``codex_scratch/check/search/r<round>_<prop>_mu<mu>/``, written
  by ``property_targets.py`` (``summary.json`` has ``unguided`` and ``guided`` rows,
  incl. PoseBusters and SMILES-uniqueness counts).
* multiMFM-SS: ``codex_scratch/memory/search/k_<prop>_{a,b}/summary.json``, written
  by ``steer_search.py`` (``final_endpoint``, ``best_lookahead_raw`` and
  ``best_lookahead_pbvalid`` rows; the last is the SS number).

``python -m multimfm.qm9_results collect --source <backup>/tabasco/codex_scratch``
copies the selected runs into ``results/qm9`` and writes compact CSVs of every
search run; ``python -m multimfm.qm9_results tables`` rebuilds the table CSVs from
``results/qm9``; ``python -m multimfm.qm9_results hparams`` extracts Tables 19-20
from the checkpoints; ``python -m multimfm.qm9_results baselines --source outputs``
collects the (unpublished) same-backbone baseline sweep.

Everything here is plain JSON/CSV handling; only ``hparams`` needs torch.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shutil
from pathlib import Path

from multimfm import paths

HARTREE_TO_MEV = 27211.386245988  # value used by the original collect scripts
PROPERTIES = ("cv", "mu", "alpha", "gap", "homo", "lumo")
ENERGY = {"gap", "homo", "lumo"}
UNITS = {
    "cv": "cal/(mol K)",
    "mu": "D",
    "alpha": "Bohr^3",
    "gap": "meV",
    "homo": "meV",
    "lumo": "meV",
}
PAPER_LABEL = {
    "cv": "Cv", "mu": "mu", "alpha": "alpha",
    "gap": "Delta eps", "homo": "eps_HOMO", "lumo": "eps_LUMO",
}

# multiMFM (non-SS): (mu, reward_scale) and the search run behind each paper number.
MULTIMFM_CONFIG = {
    "cv": (2.80, 1.5),
    "mu": (0.78, 8.638),
    "alpha": (3.40, 0.3),
    "gap": (0.17, 12000.0),
    "homo": (0.50, 41500.0),
    "lumo": (0.44, 12000.0),
}
MULTIMFM_RUN = {
    "cv": "r2_cv_mu2p80",
    "mu": "r3_mu_mu0p78",
    "alpha": "r5_alpha_mu3p40",
    "gap": "r5_gap_mu0p17",
    "homo": "r3_homo_mu0p50",
    "lumo": "r2_lumo_mu0p44",
}
# Interactive Slurm allocation each selected non-SS run executed in (round*.manifest;
# every run was an `srun --jobid=<alloc> --overlap` step inside that allocation).
MULTIMFM_ALLOCATION = {
    "cv": 34989466, "mu": 34989311, "alpha": 34985247,
    "gap": 34988184, "homo": 34989391, "lumo": 34989345,
}

# multiMFM-SS: (mu, reward_scale, select_min_t) of the k_<prop>_{a,b} runs.
SS_CONFIG = {
    "cv": (5.0, 1.5, 0.5),
    "mu": (3.0, 8.638, 0.3),
    "alpha": (11.0, 0.3, 0.3),
    "gap": (1.0, 12000.0, 0.3),
    "homo": (1.5, 41500.0, 0.3),
    "lumo": (1.5, 12000.0, 0.3),
}
# SS configs of the 2026-09-29 3-seed rerun (Algorithm-3 selection): the submitted ones
# except lumo mu 1.5 -> 3.0 (seed-8 mu grid, pre-registered rule; ss_mu_decision.csv).
SS_CONFIG_RERUN = {**SS_CONFIG, "lumo": (3.0, 12000.0, 0.3)}
SS_ALLOCATION = {  # bol1k.manifest: JOBID per run
    "k_cv_a": 35049269, "k_cv_b": 35049270, "k_alpha_a": 35049272, "k_alpha_b": 35049288,
    "k_mu_a": 35049262, "k_mu_b": 35049200, "k_lumo_a": 35049197, "k_lumo_b": 35049193,
    "k_gap_a": 35049192, "k_gap_b": 35049152, "k_homo_a": 35049153, "k_homo_b": 35049145,
}
# Which replicate each printed multiMFM-SS number matches (reconstructed on
# 2026-09-29 by comparing the printed values with both replicates; see PROVENANCE).
# "a|b": both replicates round to the printed value.
SS_PUBLISHED_SOURCE = {
    "cv": {"mae": "a", "pb": "a"},
    "mu": {"mae": "a|b", "pb": "b"},
    "alpha": {"mae": "mean", "pb": "mean"},
    "gap": {"mae": "a", "pb": "a"},
    "homo": {"mae": "mean", "pb": "b"},
    "lumo": {"mae": "b", "pb": "b"},  # k_lumo_a was killed before writing a summary
}


def to_paper_units(prop: str, value: float | None) -> float | None:
    if value is None:
        return None
    return value * HARTREE_TO_MEV if prop in ENERGY else value


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _run_rows(summary: dict) -> dict:
    return {row["run"]: row for row in summary["summary"] if "run" in row}


def multimfm_record(summary: dict, *, tag: str = "") -> dict:
    """Flatten a property_targets.py summary.json (unguided + guided) into one row."""
    args = summary["args"]
    prop = args["property_name"]
    rows = _run_rows(summary)
    out = {
        "tag": tag,
        "property": prop,
        "mu": float(args["mu"]),
        "reward_scale": float(args["reward_scale"]),
        "guide_max_t": float(args.get("guide_max_t", 0.95)),
        "num_samples": int(args["num_samples"]),
        "seed": int(args["seed"]),
        "target_seed": int(args["target_seed"]),
        "unit": UNITS[prop],
    }
    for run in ("unguided", "guided"):
        row = rows.get(run, {})
        out[f"{run}_oracle_mae"] = to_paper_units(prop, row.get("oracle_mae"))
        out[f"{run}_guide_mae"] = to_paper_units(prop, row.get("guide_mae"))
        out[f"{run}_oracle_median_ae"] = to_paper_units(prop, row.get("oracle_median_abs_error"))
        out[f"{run}_pb_valid"] = row.get("pb_pb_valid")
        out[f"{run}_n_generated"] = row.get("pb_num_generated")
        out[f"{run}_n_smiles"] = row.get("pb_num_smiles")
        out[f"{run}_n_unique_smiles"] = row.get("pb_num_unique_smiles")
        out[f"{run}_unique_rate"] = row.get("pb_unique_smiles_rate")
        out[f"{run}_unique_rate_total"] = row.get("pb_unique_smiles_rate_total")
    return out


def ss_record(summary: dict, *, tag: str = "") -> dict:
    """Flatten a steer_search.py summary.json into one row."""
    args = summary["args"]
    prop = args["property_name"]
    rows = _run_rows(summary)
    out = {
        "tag": tag,
        "property": prop,
        "mu": float(args["mu"]),
        "reward_scale": float(args["reward_scale"]),
        "select_min_t": float(args["select_min_t"]),
        "guide_min_t": float(args.get("guide_min_t", 0.05)),
        "guide_max_t": float(args.get("guide_max_t", 0.95)),
        "num_samples": int(args["num_samples"]),
        "value_samples": int(args.get("value_samples", 0)),
        "seed": int(args["seed"]),
        "target_seed": int(args["target_seed"]),
        "num_candidates": summary.get("num_candidates"),
        "unit": UNITS.get(prop, ""),
    }
    for run, key in (
        ("final_endpoint", "final"),
        ("best_lookahead_raw", "raw"),
        ("best_lookahead_pbvalid", "ss"),
        ("best_lookahead", "ss_unfiltered_legacy"),
    ):
        row = rows.get(run)
        if row is None:
            continue
        out[f"{key}_oracle_mae"] = to_paper_units(prop, row.get("oracle_mae"))
        out[f"{key}_guide_mae"] = to_paper_units(prop, row.get("guide_mae"))
        out[f"{key}_oracle_median_ae"] = to_paper_units(prop, row.get("oracle_median_abs_error"))
        out[f"{key}_pb_valid"] = row.get("pb_pb_valid")
    ss = rows.get("best_lookahead_pbvalid", {})
    out["ss_frac_with_valid_candidate"] = ss.get("frac_with_valid_candidate")
    out["ss_selected_t_mean"] = ss.get("selected_t_mean")
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------------
# collect: copy the paper runs from the original records into results/qm9
# ----------------------------------------------------------------------------

def collect(source: Path, dest: Path) -> None:
    check = source / "check" / "search"
    memory = source / "memory" / "search"

    # 1) selected non-SS runs (full outputs)
    for prop, tag in MULTIMFM_RUN.items():
        out = dest / "runs" / "multimfm" / prop
        out.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "summary.csv", "per_sample.csv", "guided_step_logs.csv"):
            shutil.copy2(check / tag / name, out / name)
        (out / "SOURCE.txt").write_text(f"codex_scratch/check/search/{tag}\n")

    # 2) SS replicates (summary.json is all steer_search writes)
    for prop in PROPERTIES:
        for rep in ("a", "b"):
            tag = f"k_{prop}_{rep}"
            src = memory / tag / "summary.json"
            if not src.exists():
                continue
            out = dest / "runs" / "multimfm_ss" / tag
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / "summary.json")

    # 3) every non-SS search run, one row each
    grid = []
    for path in sorted(glob.glob(str(check / "r*_*" / "summary.json"))):
        tag = Path(path).parent.name
        grid.append(multimfm_record(load_json(path), tag=tag))
    order = {p: i for i, p in enumerate(PROPERTIES)}
    grid.sort(key=lambda r: (order[r["property"]], r["tag"]))
    write_csv(dest / "multimfm_search_grid.csv", grid)

    # 4) every steer_search (best-of-lookahead) search run, one row each
    ss_grid = []
    for path in sorted(glob.glob(str(memory / "*" / "summary.json"))):
        tag = Path(path).parent.name
        summary = load_json(path)
        if summary.get("args", {}).get("property_name") not in UNITS:
            continue
        ss_grid.append(ss_record(summary, tag=tag))
    ss_grid.sort(key=lambda r: (order[r["property"]], r["tag"]))
    write_csv(dest / "multimfm_ss_search_grid.csv", ss_grid)

    # 5) references for the small-n reruns
    ver = dest / "verification"
    for run in ("gpu_A", "gpu_B"):
        out = ver / f"run_gen_{run}"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "summary.csv", "per_sample.csv"):
            shutil.copy2(source / "check" / run / name, out / name)
    for tag in ("v6_alpha_mu11_s30", "v7_alpha_mu11_a", "v7_alpha_mu11_b", "v8_alpha_mu11_c",
                "v3_cv_mu5", "v4_cv_mu5_a", "v4_cv_mu5_b", "v5_cv_mu5_c", "v5_cv_mu5_d"):
        prop = tag.split("_")[1]
        out = ver / f"ss_{prop}_n100_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(memory / tag / "summary.json", out / "summary.json")


# ----------------------------------------------------------------------------
# tables: rebuild Tables 2, 21, 22 from results/qm9
# ----------------------------------------------------------------------------

def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _pick(source: str, a, b, mean):
    if source == "a":
        return a
    if source in ("b", "a|b"):
        return b
    if source == "mean":
        return mean
    raise ValueError(source)


def multimfm_records(results: Path) -> dict[str, dict]:
    out = {}
    for prop in PROPERTIES:
        path = results / "runs" / "multimfm" / prop / "summary.json"
        if path.exists():
            out[prop] = multimfm_record(load_json(path), tag=MULTIMFM_RUN[prop])
    return out


def ss_records(results: Path) -> dict[str, dict]:
    out = {}
    for prop in PROPERTIES:
        for rep in ("a", "b"):
            tag = f"k_{prop}_{rep}"
            path = results / "runs" / "multimfm_ss" / tag / "summary.json"
            if path.exists():
                out[tag] = ss_record(load_json(path), tag=tag)
    return out


def build_tables(results: Path) -> dict[str, list[dict]]:
    mm = multimfm_records(results)
    ss = ss_records(results)
    table2, table21, table22 = [], [], []
    for prop in PROPERTIES:
        m = mm.get(prop, {})
        a = ss.get(f"k_{prop}_a", {})
        b = ss.get(f"k_{prop}_b", {})
        mae_a, mae_b = a.get("ss_oracle_mae"), b.get("ss_oracle_mae")
        pb_a, pb_b = a.get("ss_pb_valid"), b.get("ss_pb_valid")
        mae_mean, pb_mean = _mean([mae_a, mae_b]), _mean([pb_a, pb_b])
        src = SS_PUBLISHED_SOURCE[prop]
        table2.append({
            "property": prop,
            "unit": UNITS[prop],
            "multimfm_oracle_mae": m.get("guided_oracle_mae"),
            "multimfm_run": m.get("tag"),
            "multimfm_mu": m.get("mu"),
            "multimfm_reward_scale": m.get("reward_scale"),
            "unguided_oracle_mae": m.get("unguided_oracle_mae"),
            "ss_oracle_mae_a": mae_a,
            "ss_oracle_mae_b": mae_b,
            "ss_oracle_mae_mean_ab": mae_mean,
            "ss_n_replicates": sum(x is not None for x in (mae_a, mae_b)),
            "ss_oracle_mae_as_published": _pick(src["mae"], mae_a, mae_b, mae_mean),
            "ss_published_mae_source": src["mae"],
        })
        table21.append({
            "property": prop,
            "n": m.get("num_samples"),
            "unsteered_pb_valid": m.get("unguided_pb_valid"),
            "multimfm_pb_valid": m.get("guided_pb_valid"),
            "ss_pb_valid_a": pb_a,
            "ss_pb_valid_b": pb_b,
            "ss_pb_valid_mean_ab": pb_mean,
            "ss_pb_valid_as_published": _pick(src["pb"], pb_a, pb_b, pb_mean),
            "ss_published_pb_source": src["pb"],
            "ss_final_endpoint_pb_valid_a": a.get("final_pb_valid"),
            "ss_final_endpoint_pb_valid_b": b.get("final_pb_valid"),
        })
        table22.append({
            "property": prop,
            "unsteered_unique_rate": m.get("unguided_unique_rate"),
            "unsteered_n_unique": m.get("unguided_n_unique_smiles"),
            "unsteered_n_smiles": m.get("unguided_n_smiles"),
            "steered_unique_rate": m.get("guided_unique_rate"),
            "steered_n_unique": m.get("guided_n_unique_smiles"),
            "steered_n_smiles": m.get("guided_n_smiles"),
            "steered_unique_rate_total": m.get("guided_unique_rate_total"),
            "n_generated": m.get("guided_n_generated"),
        })
    return {"table2": table2, "table21": table21, "table22": table22}


def write_tables(results: Path) -> None:
    for name, rows in build_tables(results).items():
        write_csv(results / f"{name}.csv", rows)


# ----------------------------------------------------------------------------
# per-sample statistics (non-SS only: steer_search stores no per-sample rows)
# ----------------------------------------------------------------------------

def per_sample_errors(per_sample_csv: Path, run: str = "guided") -> list[float]:
    rows = read_csv(per_sample_csv)
    return [float(r["oracle_abs_error"]) for r in rows if r["run"] == run]


def mae_and_se(errors: list[float]) -> tuple[float, float]:
    n = len(errors)
    mean = sum(errors) / n
    var = sum((e - mean) ** 2 for e in errors) / max(1, n - 1)
    return mean, math.sqrt(var / n)


# ----------------------------------------------------------------------------
# hparams: Tables 19-20 from the checkpoints
# ----------------------------------------------------------------------------

def hparams_from_checkpoints(base_ckpt: Path, mfm_ckpt: Path) -> tuple[dict, dict]:
    import torch

    base = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    args, margs, stats = base["args"], base["model_args"], base["stats"]
    group = {k: v for k, v in base["optimizer_state_dict"]["param_groups"][0].items() if k != "params"}
    table19 = {
        "source": {
            "checkpoint": str(base_ckpt.relative_to(paths.REPO_ROOT)),
            "fields": "args, model_args, stats, optimizer_state_dict.param_groups[0], step",
            "code_constants": "multimfm.train_base_flow.build_model (loss weights, time "
                              "distribution); multimfm.qm9_data.main (AdamW, clip 1.0, no "
                              "scheduler, no EMA)",
        },
        "training_set_molecules": sum(int(v) for v in stats["num_atoms_histogram"].values()),
        "partition": args["partition"],
        "limit_data": args["limit_data"],
        "hydrogens": "included" if args["include_hydrogens"] else "excluded",
        "max_heavy_atoms": stats["max_num_atoms"],
        "atom_vector_dim": stats["atom_dim"],
        "coordinate_normalizer": stats["coordinate_normalizer"],
        "atomics_mode": margs["atomics_mode"],
        "dfm_beta_schedule": margs["dfm_beta_schedule"],
        "sample_schedule": margs["sample_schedule"],
        "time_distribution": "beta(alpha=1.8, beta=1) [code constant]",
        "hidden_dim": margs["hidden_dim"],
        "num_layers": margs["num_layers"],
        "num_heads": margs["num_heads"],
        "cross_attention": not margs["no_cross_attention"],
        "optimizer": "AdamW [code: torch.optim.AdamW(lr=args.lr)]",
        "betas": list(group["betas"]),
        "lr": group["lr"],
        "weight_decay": group["weight_decay"],
        "lr_schedule": "constant [code: no scheduler]",
        "grad_clip_l2": "1.0 [code constant]",
        "ema": None,
        "batch_size": args["batch_size"],
        "random_rotation_copies": margs["num_random_augmentations"],
        "checkpoint_step": base["step"],
        "train_steps": args["train_steps"],
        "coord_loss_weight": "1.0 [code constant]",
        "atom_loss_weight": "0.1 [code constant]",
        "seed": args["seed"],
        "args": args,
        "model_args": margs,
    }

    mfm = torch.load(mfm_ckpt, map_location="cpu", weights_only=False)
    a, cfg = mfm["args"], mfm["student_cfg"]
    stage_objective = {
        1: ("diagonal GLASS distillation"
            + (" on teacher-generated conditioning states"
               if a.get("rollout_condition_source") in ("generated", "guided") else "")
            + " (+ direction and condition-sensitivity terms); no ESD term"),
        2: "ESD consistency + diagonal anchor",
        3: "GLASS transition distillation + diagonal anchor",
        4: "GLASS rollout distillation + diagonal anchor",
    }[int(a["stage"])]
    table20 = {
        "source": {
            "checkpoint": str(mfm_ckpt.relative_to(paths.REPO_ROOT)),
            "fields": "args, student_cfg, step",
            "code_constants": "multimfm.train_mfm_student.main (Adam, clip 1.0, "
                              "linear warmup + cosine, single process)",
        },
        "hidden_dim": cfg["hidden_dim"],
        "num_layers": cfg["num_layers"],
        "num_heads": cfg["num_heads"],
        "block_conditioning": cfg["block_conditioning"],
        "velocity_parametrization": cfg["velocity_parametrization"],
        "batch_per_gpu": a["batch_size"],
        "num_gpus": "1 [single-process trainer]",
        "global_batch": a["batch_size"],
        "selected_updates_in_phase": mfm["step"],
        "cosine_schedule_horizon": a["steps"],
        "optimizer": "Adam [code constant]",
        "peak_lr": a["lr"],
        "warmup_steps": a["warmup"],
        "grad_clip_l2": "1.0 [code constant]",
        "ema": a["ema"],
        "stage": a["stage"],
        "objective": stage_objective,
        "esd_term_active": int(a["stage"]) == 2,
        "coord_loss_weight": a["coord_loss_weight"],
        "atom_loss_weight": a["atom_loss_weight"],
        "diag_direction_weight": a["diag_direction_weight"],
        "diag_condition_jvp_weight": a["diag_condition_jvp_weight"],
        "diag_condition_jvp_direction_weight": a["diag_condition_jvp_direction_weight"],
        "t_max": a["t_max"],
        "conditioning_states": a["rollout_condition_source"],
        "generated_condition_steps": a["generated_condition_steps"],
        "posterior_target_steps": a["transition_teacher_steps"],
        "t_cond_grid": a["t_cond_grid"],
        "t_cond_jitter": a["t_cond_jitter"],
        "t_cond_zero_rate": a["t_cond_zero_rate"],
        "init_ckpt": a["init_ckpt"],
        "seed": a["seed"],
        "args": a,
        "student_cfg": cfg,
    }
    return table19, table20


# ----------------------------------------------------------------------------
# baselines: the uncommitted same-backbone sweep (outputs/pilot, outputs/sweep)
# ----------------------------------------------------------------------------

def _baseline_config(method: str, args: dict) -> str:
    if method == "best_of_n":
        return f"N={args.get('n_candidates')}"
    if method == "dps":
        return f"mu={args.get('mu')},guide_every={args.get('guide_every')}"
    if method == "fk":
        return (f"K={args.get('n_particles')},lambda={args.get('fk_lambda')},"
                f"resample_every={args.get('fk_resample_every')}")
    if method == "steer_search":
        return f"mu={args.get('mu')},MC={args.get('value_samples')},select_min_t={args.get('select_min_t')}"
    return ""


def collect_baselines(source: Path, dest: Path) -> list[dict]:
    rows = []
    for phase in ("pilot", "sweep"):
        for path in sorted(glob.glob(str(source / phase / "**" / "*.json"), recursive=True)):
            rel = Path(path).relative_to(source)
            summary = load_json(path)
            if "summary" not in summary:
                continue
            out = dest / "runs" / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            args = summary.get("args", {})
            method = summary.get("method") or ("steer_search" if "select_min_t" in args else "?")
            nfe = summary.get("nfe") or {}
            for row in summary["summary"]:
                rows.append({
                    "phase": phase,
                    "file": str(rel),
                    "method": method,
                    "run": row.get("run"),
                    "config": _baseline_config(method, args),
                    "property": args.get("property_name"),
                    "num_samples": args.get("num_samples"),
                    "seed": args.get("seed"),
                    "oracle_mae": row.get("oracle_mae"),
                    "guide_mae": row.get("guide_mae"),
                    "oracle_median_ae": row.get("oracle_median_abs_error"),
                    "pb_valid": row.get("pb_pb_valid"),
                    "frac_with_valid_candidate": row.get("frac_with_valid_candidate"),
                    "num_candidates": summary.get("num_candidates"),
                    "nfe_base_per_mol": nfe.get("nfe_base_per_mol"),
                    "nfe_regressor_per_mol": nfe.get("nfe_regressor_per_mol"),
                    "nfe_backward_per_mol": nfe.get("nfe_backward_per_mol"),
                    "pb_checks_per_mol": nfe.get("pb_checks_per_mol"),
                    "nfe_total_per_mol": nfe.get("nfe_total_per_mol"),
                    "nfe_measured": bool(nfe),
                })
    write_csv(dest / "baseline_runs.csv", rows)
    return rows


# ----------------------------------------------------------------------------
# rerun3: the 3-seed n = 1000 reruns (authors' decisions 4 and 5, 2026-09-29)
# ----------------------------------------------------------------------------

RERUN_SEEDS = (1, 2, 3)          # evaluation seeds; target seed = 2026 + seed
# (row, run inside summary.json) -> label used in the long table
RERUN_VARIANTS = {
    ("multimfm", "unguided"): "unsteered",
    ("multimfm", "guided"): "multimfm",
    ("ss", "best_lookahead_pbvalid"): "ss",                  # Algorithm 3 (MFM look-aheads)
    ("ss", "best_lookahead_raw"): "ss_unfiltered",
    ("ss", "base_endpoint_pbvalid"): "ss_base_endpoint",     # code behind the submitted row
    ("ss", "base_endpoint_raw"): "ss_base_endpoint_unfiltered",
    ("ss", "mfm_hardscore_pbvalid"): "ss_mfm_hardscore",
    ("ss", "mfm_valuescore_pbvalid"): "ss_mfm_valuescore",
    ("ss", "mfm_window_pbvalid"): "ss_mfm_window",
    ("ss", "final_endpoint"): "ss_trajectory_endpoint",
}
SS_PER_SAMPLE_RUNS = ("final_endpoint", "best_lookahead_pbvalid", "best_lookahead_raw",
                      "base_endpoint_pbvalid")


def _gzip_copy(src: Path, dst: Path, keep_runs=None) -> None:
    import gzip

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, newline="", encoding="utf-8") as fin, \
            gzip.open(dst, "wt", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if keep_runs is None or row["run"] in keep_runs:
                writer.writerow(row)


def _slim_summary(summary: dict) -> dict:
    out = dict(summary)
    nfe = dict(out.get("nfe") or {})
    nfe.pop("raw", None)
    out["nfe"] = nfe
    return out


def rerun_long_rows(source: Path) -> list[dict]:
    """One row per (row, variant, property, seed) from outputs/qm9/table2_3seed."""
    rows = []
    for row_kind in ("multimfm", "ss"):
        for path in sorted(glob.glob(str(source / row_kind / "*_s*" / "summary.json"))):
            summary = load_json(path)
            args = summary["args"]
            prop = args["property_name"]
            seed = int(args["seed"])
            nfe_rows = summary.get("nfe", {})
            for r in summary["summary"]:
                label = RERUN_VARIANTS.get((row_kind, r.get("run")))
                if label is None:
                    continue
                rec = {
                    "row": row_kind, "variant": label, "run": r["run"], "property": prop,
                    "seed": seed, "target_seed": int(args["target_seed"]),
                    "num_samples": int(args["num_samples"]),
                    "mu": float(args["mu"]), "reward_scale": float(args["reward_scale"]),
                    "unit": UNITS[prop],
                    "oracle_mae": to_paper_units(prop, r.get("oracle_mae")),
                    "oracle_mae_pb_valid_only": to_paper_units(prop, r.get("oracle_mae_pb_valid_only")),
                    "guide_mae": to_paper_units(prop, r.get("guide_mae")),
                    "oracle_median_ae": to_paper_units(prop, r.get("oracle_median_abs_error")),
                    "pb_valid": r.get("pb_pb_valid"),
                    "n_smiles": r.get("pb_num_smiles"),
                    "n_unique_smiles": r.get("pb_num_unique_smiles"),
                    "unique_rate": r.get("pb_unique_smiles_rate"),
                    "unique_rate_total": r.get("pb_unique_smiles_rate_total"),
                    "n_generated": r.get("pb_num_generated"),
                    "nfe_generative_per_mol": r.get("nfe_generative_per_mol"),
                    "nfe_generative_bwd_per_mol": r.get("nfe_generative_bwd_per_mol"),
                    "frac_with_valid_candidate": r.get("frac_with_valid_candidate"),
                    "frac_selected_lookahead": r.get("frac_selected_lookahead"),
                    "selected_t_mean": r.get("selected_t_mean"),
                    "pb_checks_per_mol": r.get("pb_checks_per_mol"),
                    "n_truncated_uncertain": r.get("n_truncated_uncertain"),
                }
                if row_kind == "ss":
                    var = (nfe_rows.get("variants") or {}).get(r.get("lookahead_source") or "", {})
                    rec["extra_generative_vs_multimfm"] = var.get("extra_generative_vs_multimfm")
                rows.append(rec)
    order = {p: i for i, p in enumerate(PROPERTIES)}
    rows.sort(key=lambda r: (r["row"], r["variant"], order[r["property"]], r["seed"]))
    return rows


def _mean_se(values):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n < 2:
        return m, None, n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, sd / math.sqrt(n), n


def rerun_summary(long_rows: list[dict], metrics=("oracle_mae", "oracle_mae_pb_valid_only",
                                                  "pb_valid", "unique_rate", "unique_rate_total",
                                                  "n_unique_smiles", "n_smiles",
                                                  "nfe_generative_per_mol")) -> list[dict]:
    """Mean and SE over seeds for every (variant, property, metric)."""
    out = []
    variants = sorted({r["variant"] for r in long_rows})
    for var in variants:
        for prop in PROPERTIES:
            sel = [r for r in long_rows if r["variant"] == var and r["property"] == prop]
            if not sel:
                continue
            rec = {"variant": var, "property": prop, "unit": UNITS[prop],
                   "seeds": ",".join(str(r["seed"]) for r in sel), "n_seeds": len(sel)}
            for m in metrics:
                mean, se, _ = _mean_se([r.get(m) for r in sel])
                rec[f"{m}_mean"] = mean
                rec[f"{m}_se"] = se
                for r in sel:
                    rec[f"{m}_s{r['seed']}"] = r.get(m)
            out.append(rec)
    return out


def collect_rerun(source: Path, dest: Path) -> dict:
    """Copy the 3-seed runs (slim summary.json + gzipped per_sample) and build the tables."""
    for row_kind in ("multimfm", "ss"):
        for path in sorted(glob.glob(str(source / row_kind / "*_s*" / "summary.json"))):
            run_dir = Path(path).parent
            out = dest / "runs" / row_kind / run_dir.name
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps(_slim_summary(load_json(path)), indent=1) + "\n")
            ps = run_dir / "per_sample.csv"
            if ps.exists():
                keep = None if row_kind == "multimfm" else set(SS_PER_SAMPLE_RUNS)
                _gzip_copy(ps, out / "per_sample.csv.gz", keep_runs=keep)
    long_rows = rerun_long_rows(source)
    summ = rerun_summary(long_rows)
    write_csv(dest / "rerun_3seed_long.csv", long_rows)
    write_csv(dest / "rerun_3seed_summary.csv", summ)
    status = {
        "expected_runs": {k: 6 * len(RERUN_SEEDS) for k in ("multimfm", "ss")},
        "found_runs": {k: len(glob.glob(str(source / k / "*_s*" / "summary.json")))
                       for k in ("multimfm", "ss")},
    }
    (dest / "rerun_3seed_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def collect_pilot_ss(source: Path, dest: Path) -> list[dict]:
    """Seed-8 / target-2026 n=100 pilot of the Algorithm-3 steer-search (all variants)."""
    rows = []
    for path in sorted(glob.glob(str(source / "*" / "summary.json"))):
        name = Path(path).parent.name
        summary = load_json(path)
        out = dest / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(_slim_summary(summary), indent=1) + "\n")
        args = summary["args"]
        prop = args["property_name"]
        for r in summary["summary"]:
            rows.append({
                "dir": name, "property": prop, "unit": UNITS[prop], "run": r.get("run"),
                "lookahead_source": r.get("lookahead_source"),
                "lookahead_score": r.get("lookahead_score"),
                "lookahead_t_min": r.get("lookahead_t_min"),
                "topk": args.get("lookahead_topk"), "mu": args.get("mu"),
                "num_samples": args.get("num_samples"), "seed": args.get("seed"),
                "oracle_mae": to_paper_units(prop, r.get("oracle_mae")),
                "guide_mae": to_paper_units(prop, r.get("guide_mae")),
                "oracle_mae_pb_valid_only": to_paper_units(prop, r.get("oracle_mae_pb_valid_only")),
                "pb_valid": r.get("pb_pb_valid"),
                "unique_rate": r.get("pb_unique_smiles_rate"),
                "selected_t_mean": r.get("selected_t_mean"),
                "frac_selected_lookahead": r.get("frac_selected_lookahead"),
                "pb_checks_per_mol": r.get("pb_checks_per_mol"),
                "n_truncated_uncertain": r.get("n_truncated_uncertain"),
                "nfe_generative_per_mol": r.get("nfe_generative_per_mol"),
                "nfe_generative_bwd_per_mol": r.get("nfe_generative_bwd_per_mol"),
                "timing_total_s": (summary.get("timing_s") or {}).get("total"),
                "device": summary.get("device"),
            })
    write_csv(dest / "pilot_ss_runs.csv", rows)
    return rows


SS_RESELECT_RULE = ("keep the submitted SS mu unless another grid mu gives a lower Algorithm-3 "
                    "oracle MAE by more than 2 SE of the difference (per-molecule errors) with "
                    "PB-valid >= 0.95")


def collect_ss_reselect(source: Path, dest: Path) -> list[dict]:
    """Seed-8 n=300 mu grid for the Algorithm-3 steer-search + the pre-registered decision."""
    rows = []
    for path in sorted(glob.glob(str(source / "*_mu*_n*" / "summary.json"))):
        run_dir = Path(path).parent
        summary = load_json(path)
        out = dest / "runs" / run_dir.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(_slim_summary(summary), indent=1) + "\n")
        args = summary["args"]
        prop = args["property_name"]
        errs = {}
        ps = run_dir / "per_sample.csv"
        if ps.exists():
            for r in read_csv(ps):
                errs.setdefault(r["run"], []).append(float(r["oracle_abs_error"]))
            _gzip_copy(ps, out / "per_sample.csv.gz", keep_runs=set(SS_PER_SAMPLE_RUNS))
        for r in summary["summary"]:
            e = errs.get(r["run"])
            se = mae_and_se(e)[1] if e else None
            rows.append({
                "dir": run_dir.name, "property": prop, "unit": UNITS[prop], "mu": float(args["mu"]),
                "submitted_ss_mu": SS_CONFIG[prop][0], "multimfm_mu": MULTIMFM_CONFIG[prop][0],
                "run": r["run"], "num_samples": int(args["num_samples"]),
                "oracle_mae": to_paper_units(prop, r.get("oracle_mae")),
                "oracle_mae_se": to_paper_units(prop, se),
                "pb_valid": r.get("pb_pb_valid"),
                "n_truncated_uncertain": r.get("n_truncated_uncertain"),
            })
    write_csv(dest / "ss_mu_grid.csv", rows)
    # decision per property
    decisions = []
    for prop in PROPERTIES:
        cand = [r for r in rows if r["property"] == prop and r["run"] == "best_lookahead_pbvalid"]
        ref = [r for r in cand if abs(r["mu"] - SS_CONFIG[prop][0]) < 1e-9]
        if not ref:
            continue
        ref = ref[0]
        best, chosen = ref, ref
        for r in cand:
            if r is ref or (r["pb_valid"] or 0) < 0.95:
                continue
            diff = ref["oracle_mae"] - r["oracle_mae"]
            se = math.sqrt((ref["oracle_mae_se"] or 0) ** 2 + (r["oracle_mae_se"] or 0) ** 2)
            if diff > 2 * se and r["oracle_mae"] < best["oracle_mae"]:
                best = r
        chosen = best
        decisions.append({"property": prop, "submitted_ss_mu": ref["mu"], "chosen_mu": chosen["mu"],
                          "submitted_mu_mae": ref["oracle_mae"], "submitted_mu_se": ref["oracle_mae_se"],
                          "chosen_mae": chosen["oracle_mae"], "changed": chosen is not ref,
                          "rule": SS_RESELECT_RULE})
    write_csv(dest / "ss_mu_decision.csv", decisions)
    return decisions


# ----------------------------------------------------------------------------
# baselines v2: the fixed same-backbone sweep (measured NFE for every method)
# ----------------------------------------------------------------------------

def collect_baselines_v2(source: Path, dest: Path) -> list[dict]:
    """Flatten outputs/qm9/baselines_v2/seed<s>/<config>/*.json into one CSV."""
    rows = []
    for path in sorted(glob.glob(str(source / "seed*" / "*" / "*.json"))):
        summary = load_json(path)
        if "summary" not in summary:
            continue
        rel = Path(path).relative_to(source)
        out = dest / "runs_v2" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_slim_summary(summary), indent=1) + "\n")
        args = summary.get("args", {})
        method = summary.get("method") or "steer_search"
        nfe = summary.get("nfe") or {}
        for row in summary["summary"]:
            rec = {
                "file": str(rel), "method": method, "run": row.get("run"),
                "config": _baseline_config(method, args),
                "property": args.get("property_name"), "num_samples": args.get("num_samples"),
                "seed": args.get("seed"), "target_seed": args.get("target_seed"),
                "mu": args.get("mu"), "value_samples": args.get("value_samples"),
                "n_candidates": args.get("n_candidates"), "n_particles": args.get("n_particles"),
                "fk_lambda": args.get("fk_lambda"), "guide_every": args.get("guide_every"),
                "oracle_mae": row.get("oracle_mae"), "guide_mae": row.get("guide_mae"),
                "oracle_median_ae": row.get("oracle_median_abs_error"),
                "pb_valid": row.get("pb_pb_valid"),
                "unique_rate": row.get("pb_unique_smiles_rate"),
                "frac_with_valid_candidate": row.get("frac_with_valid_candidate"),
                "lookahead_source": row.get("lookahead_source"),
                "nfe_generative": row.get("nfe_generative_per_mol"),
                "nfe_generative_bwd": row.get("nfe_generative_bwd_per_mol"),
            }
            g, b = rec["nfe_generative"], rec["nfe_generative_bwd"]
            rec["nfe_generative_fwd_equiv"] = None if g is None else g + 2 * (b or 0.0)
            if method == "steer_search":
                ph = (nfe.get("by_phase") or {})
                rec["nfe_regressor"] = (ph.get("trajectory") or {}).get("nfe_regressor")
                rec["pb_checks_per_mol"] = row.get("pb_checks_per_mol")
            else:
                rec["nfe_regressor"] = nfe.get("nfe_regressor_per_mol")
                rec["pb_checks_per_mol"] = nfe.get("pb_checks_per_mol")
            rows.append(rec)
    write_csv(dest / "baseline_sweep_v2.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--source", type=Path, required=True,
                   help="the original tabasco/codex_scratch directory (or its backup)")
    p.add_argument("--dest", type=Path, default=paths.RESULTS / "qm9")
    p = sub.add_parser("tables")
    p.add_argument("--results", type=Path, default=paths.RESULTS / "qm9")
    p = sub.add_parser("hparams")
    p.add_argument("--base-checkpoint", type=Path, default=paths.BASE_FLOW_CKPT)
    p.add_argument("--mfm-checkpoint", type=Path, default=paths.MFM_STUDENT_CKPT)
    p.add_argument("--dest", type=Path, default=paths.RESULTS / "qm9")
    p = sub.add_parser("baselines")
    p.add_argument("--source", type=Path, default=paths.OUTPUTS,
                   help="directory holding pilot/ and sweep/")
    p.add_argument("--dest", type=Path, default=paths.RESULTS / "qm9" / "baselines")
    p = sub.add_parser("rerun3", help="collect the 3-seed n=1000 reruns")
    p.add_argument("--source", type=Path, default=paths.OUTPUTS / "qm9" / "table2_3seed")
    p.add_argument("--dest", type=Path, default=paths.RESULTS / "qm9" / "rerun_3seed")
    p = sub.add_parser("baselines-v2", help="collect the fixed baseline sweep")
    p.add_argument("--source", type=Path, default=paths.OUTPUTS / "qm9" / "baselines_v2")
    p.add_argument("--dest", type=Path, default=paths.RESULTS / "qm9" / "baselines")
    p = sub.add_parser("ss-reselect", help="collect the seed-8 SS mu grid")
    p.add_argument("--source", type=Path, default=paths.OUTPUTS / "qm9" / "ss_mfm_reselect")
    p.add_argument("--dest", type=Path, default=paths.RESULTS / "qm9" / "ss_mfm_reselect")
    p = sub.add_parser("pilot-ss", help="collect the seed-8 n=100 SS pilot")
    p.add_argument("--source", type=Path, default=paths.OUTPUTS / "qm9" / "ss_mfm_pilot")
    p.add_argument("--dest", type=Path, default=paths.RESULTS / "qm9" / "ss_mfm_pilot")
    args = parser.parse_args()

    if args.cmd == "collect":
        collect(args.source, args.dest)
        write_tables(args.dest)
    elif args.cmd == "tables":
        write_tables(args.results)
    elif args.cmd == "hparams":
        t19, t20 = hparams_from_checkpoints(args.base_checkpoint, args.mfm_checkpoint)
        args.dest.mkdir(parents=True, exist_ok=True)
        (args.dest / "table19_base_training.json").write_text(json.dumps(t19, indent=2, default=str) + "\n")
        (args.dest / "table20_mfm_training.json").write_text(json.dumps(t20, indent=2, default=str) + "\n")
    elif args.cmd == "baselines":
        collect_baselines(args.source, args.dest)
    elif args.cmd == "rerun3":
        print(collect_rerun(args.source, args.dest))
    elif args.cmd == "baselines-v2":
        print(len(collect_baselines_v2(args.source, args.dest)), "rows")
    elif args.cmd == "ss-reselect":
        for d in collect_ss_reselect(args.source, args.dest):
            print(d)
    elif args.cmd == "pilot-ss":
        collect_pilot_ss(args.source, args.dest)
    print(f"done: {args.cmd}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
