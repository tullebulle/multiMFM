"""Print the exact Table 2 / 21 / 22 cells (submitted -> 3-seed rerun) for the tex agent.

    python scripts/qm9/report_cells.py [--markdown]

Reads only results/qm9 (rerun_3seed_summary.csv, paper_values.json). Cells whose runs are
not finished print "pending (k/3 seeds)". Rounding follows the submitted table: Cv/mu/alpha
2 decimals, gap/homo/lumo integers, percentages 1 decimal.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from multimfm import paths
from multimfm.qm9_results import PROPERTIES, SS_CONFIG, SS_CONFIG_RERUN, UNITS

QM9 = paths.RESULTS / "qm9"
DEC = {"cv": 2, "mu": 2, "alpha": 2, "gap": 0, "homo": 0, "lumo": 0}
TEX = {"cv": r"$C_v$", "mu": r"$\mu$", "alpha": r"$\alpha$", "gap": r"$\Delta\varepsilon$",
       "homo": r"$\varepsilon_{\mathrm{HOMO}}$", "lumo": r"$\varepsilon_{\mathrm{LUMO}}$"}


def load():
    path = QM9 / "rerun_3seed" / "rerun_3seed_summary.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[(r["variant"], r["property"])] = r
    return out


def cell(rows, variant, prop, metric="oracle_mae", scale=1.0, dec=None):
    r = rows.get((variant, prop))
    if r is None or not r.get(f"{metric}_mean"):
        return "pending"
    n = int(r["n_seeds"])
    dec = DEC[prop] if dec is None else dec
    m = float(r[f"{metric}_mean"]) * scale
    se = r.get(f"{metric}_se")
    se = float(se) * scale if se not in ("", None) else None
    s = f"{m:.{dec}f}" + (f" ± {se:.{dec + 1}f}" if se is not None else "")
    return s if n == 3 else f"{s} [pending {n}/3]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    rows = load()
    P = json.loads((QM9 / "paper_values.json").read_text())
    t2, t21, t22 = P["table2"]["rows"], P["table21"], P["table22"]
    mm_pub, ss_pub = t2["multiMFM (Ours)"], t2["multiMFM-SS (Ours)"]

    def table(title, header, body):
        print(f"\n## {title}\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for r in body:
            print("| " + " | ".join(str(x) for x in r) + " |")

    table("Table 2 (QM9 rows) — submitted -> 3-seed mean ± SE, n=1000/seed, seeds 1-3",
          ["property", "unit", "multiMFM submitted", "multiMFM NEW", "SS submitted", "SS NEW (Alg. 3)",
           "SS mu sub->new", "SS NEW, submitted harvesting"],
          [[TEX[p], UNITS[p], mm_pub[p], cell(rows, "multimfm", p), ss_pub[p], cell(rows, "ss", p),
            f"{SS_CONFIG[p][0]:g}->{SS_CONFIG_RERUN[p][0]:g}", cell(rows, "ss_base_endpoint", p)]
           for p in PROPERTIES])

    table("Table 21 (QM9 rows) — PB validity %, submitted -> 3-seed mean ± SE",
          ["property", "N", "Unsteered submitted", "Unsteered NEW", "MultiMFM submitted",
           "MultiMFM NEW", "MultiMFM-SS submitted", "MultiMFM-SS NEW"],
          [[TEX[p], "1,000", t21["unsteered"][p], cell(rows, "unsteered", p, "pb_valid", 100, 1),
            t21["multimfm"][p], cell(rows, "multimfm", p, "pb_valid", 100, 1),
            t21["multimfm_ss"][p], cell(rows, "ss", p, "pb_valid", 100, 1)] for p in PROPERTIES])

    table("Table 22 (QM9) — uniqueness %, submitted -> 3-seed mean ± SE (steered = multiMFM row)",
          ["property", "Unsteered U submitted", "Unsteered U NEW", "Steered U submitted",
           "Steered U NEW", "Steered U_all submitted", "Steered U_all NEW", "SS U NEW"],
          [[TEX[p], t22["unsteered_U"][p], cell(rows, "unsteered", p, "unique_rate", 100, 1),
            t22["steered_U"][p], cell(rows, "multimfm", p, "unique_rate", 100, 1),
            t22["steered_U_all"][p], cell(rows, "multimfm", p, "unique_rate_total", 100, 1),
            cell(rows, "ss", p, "unique_rate", 100, 1)] for p in PROPERTIES])

    print("\n## Text claims (§3.2 / Discussion / App. F.9)\n")
    tx = P["text"]
    long_path = QM9 / "rerun_3seed" / "rerun_3seed_long.csv"
    per_seed = []
    if long_path.exists():
        with open(long_path, newline="", encoding="utf-8") as fh:
            per_seed = list(csv.DictReader(fh))

    def mean_pb(variant):
        vals = [rows.get((variant, p)) for p in PROPERTIES]
        if any(v is None or not v.get("pb_valid_mean") for v in vals):
            got = sum(v is not None and bool(v.get("pb_valid_mean")) for v in vals)
            return f"pending [{got}/6 properties]"
        m = 100 * sum(float(v["pb_valid_mean"]) for v in vals) / 6
        by_seed = {}
        for r in per_seed:
            if r["variant"] == variant and r["pb_valid"]:
                by_seed.setdefault(r["seed"], []).append(float(r["pb_valid"]))
        means = [100 * sum(v) / len(v) for v in by_seed.values() if len(v) == 6]
        if len(means) > 1:
            mu = sum(means) / len(means)
            se = math.sqrt(sum((x - mu) ** 2 for x in means) / (len(means) - 1) / len(means))
            return f"{m:.1f} ± {se:.2f}"
        return f"{m:.1f}"

    for label, variant, paper in (("multiMFM mean PB", "multimfm", tx["mean_pb_multimfm_pct"]),
                                  ("base (unsteered) mean PB", "unsteered", tx["base_pb_pct"]),
                                  ("multiMFM-SS mean PB", "ss", tx["mean_pb_multimfm_ss_pct"])):
        print(f"* {label}: submitted {paper}%  ->  {mean_pb(variant)}%")


if __name__ == "__main__":
    main()
