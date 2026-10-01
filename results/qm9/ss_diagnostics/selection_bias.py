"""Paired guide-vs-oracle analysis on the *same* molecules, from stored per-sample files.

Diagnostic for docs/ss_guide_oracle_gap.md. Zero compute: it only re-reads the
per-molecule ``guide_prediction`` / ``oracle_prediction`` columns that
``multimfm.steer_search`` and ``multimfm.property_targets`` already wrote, so both
regressors are compared on identical molecules by construction.

    ~/micromamba/envs/multimfm/bin/python results/qm9/ss_diagnostics/selection_bias.py

Per (variant, run, property, seed) it reports, in paper units:
  guide_mae, oracle_mae              -- error of each regressor against the target
  disagreement_mae = mean|guide-oracle|  -- how far apart the two regressors are on
                                          these molecules (no ground truth exists for
                                          generated molecules, so this is the honest
                                          measure of their mutual reliability)
  pred_corr, err_corr                -- Pearson r of predictions / of signed errors
  oracle_over_guide                  -- the ratio the audit flagged
Runs are ordered by how much selection pressure was applied to the guide, which is
what the dose-response in the doc is built on.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HARTREE_TO_MEV = 27211.386245988
ENERGY = {"gap", "homo", "lumo"}
UNITS = {"cv": "cal/(mol K)", "mu": "D", "alpha": "Bohr^3",
         "gap": "meV", "homo": "meV", "lumo": "meV"}
# selection pressure applied to the guide, for ordering the dose-response
PRESSURE = {
    "unguided": 0,                  # base model, no guidance, no selection
    "guided": 1,                    # guided ODE (gradient ascent on the guide)
    "final_endpoint": 1,            # SS's own guided endpoint, no selection
    "base_endpoint_pbvalid": 2,     # best of ~89 base endpoint look-aheads, PB-filtered
    "base_endpoint_raw": 3,         # best of ~89, no PB filter
    "mfm_valuescore_pbvalid": 4,    # best of ~3680 scored with the soft-atom guide
    "mfm_window_pbvalid": 5,        # best of the t >= 0.5 MFM look-aheads
    "best_lookahead_pbvalid": 6,    # SS as reported: best of ~3680, PB-filtered
    "best_lookahead_raw": 7,        # best of ~3680, no PB filter
}


def read_rows(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stats(rows, prop):
    scale = HARTREE_TO_MEV if prop in ENERGY else 1.0
    g = np.array([float(r["guide_prediction"]) for r in rows]) * scale
    o = np.array([float(r["oracle_prediction"]) for r in rows]) * scale
    t = np.array([float(r["target"]) for r in rows]) * scale
    pb = np.array([int(r["pb_valid"]) for r in rows], dtype=bool)
    st = np.array([float(r.get("selected_t", 1.0) or 1.0) for r in rows])
    la = np.array([int(r.get("selected_lookahead", 0) or 0) for r in rows], dtype=bool)
    eg, eo = g - t, o - t
    return {
        "n": len(rows),
        "unit": UNITS[prop],
        "guide_mae": float(np.abs(eg).mean()),
        "oracle_mae": float(np.abs(eo).mean()),
        "oracle_over_guide": float(np.abs(eo).mean() / max(np.abs(eg).mean(), 1e-12)),
        "disagreement_mae": float(np.abs(g - o).mean()),
        "disagreement_rms": float(np.sqrt(((g - o) ** 2).mean())),
        "pred_corr": float(np.corrcoef(g, o)[0, 1]),
        "err_corr": float(np.corrcoef(eg, eo)[0, 1]),
        "guide_bias": float(eg.mean()),
        "oracle_bias": float(eo.mean()),
        "pb_valid": float(pb.mean()),
        "selected_t_mean": float(st.mean()),
        "frac_selected_lookahead": float(la.mean()),
        # winner's-curse identity: when the guide is driven to the target,
        # |oracle - target| must equal |oracle - guide| up to the residual guide error
        "oracle_mae_minus_disagreement": float(np.abs(eo).mean() - np.abs(g - o).mean()),
    }


def main() -> None:
    out_rows = []
    for variant, root in (("ss", REPO / "results/qm9/rerun_3seed/runs/ss"),
                          ("multimfm", REPO / "results/qm9/rerun_3seed/runs/multimfm")):
        for d in sorted(root.glob("*_s*")):
            prop, seed = d.name.rsplit("_s", 1)
            path = d / "per_sample.csv.gz"
            if not path.exists():
                path = d / "per_sample.csv"
            if not path.exists():
                continue
            rows = read_rows(path)
            by_run: dict[str, list] = {}
            for r in rows:
                by_run.setdefault(r["run"], []).append(r)
            for run, rr in by_run.items():
                rec = {"variant": variant, "run": run, "property": prop, "seed": int(seed),
                       "pressure": PRESSURE.get(run, -1)}
                rec.update(stats(rr, prop))
                out_rows.append(rec)

    out_rows.sort(key=lambda r: (r["property"], r["pressure"], r["run"], r["seed"]))
    csv_path = REPO / "results/qm9/ss_diagnostics/guide_oracle_paired.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", csv_path, len(out_rows), "rows")

    # seed-averaged dose-response summary
    summary = {}
    for r in out_rows:
        key = (r["property"], r["run"])
        summary.setdefault(key, []).append(r)
    agg = []
    for (prop, run), rs in sorted(summary.items(),
                                  key=lambda kv: (kv[0][0], PRESSURE.get(kv[0][1], -1))):
        pick = lambda k: float(np.mean([x[k] for x in rs]))  # noqa: E731
        agg.append({
            "property": prop, "run": run, "pressure": PRESSURE.get(run, -1),
            "seeds": len(rs), "unit": rs[0]["unit"],
            "guide_mae": pick("guide_mae"), "oracle_mae": pick("oracle_mae"),
            "oracle_over_guide": pick("oracle_over_guide"),
            "disagreement_mae": pick("disagreement_mae"),
            "pred_corr": pick("pred_corr"), "err_corr": pick("err_corr"),
            "pb_valid": pick("pb_valid"), "selected_t_mean": pick("selected_t_mean"),
        })
    agg_path = REPO / "results/qm9/ss_diagnostics/guide_oracle_dose_response.csv"
    with open(agg_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print("wrote", agg_path, len(agg), "rows")
    for a in agg:
        print(f"{a['property']:>5} p{a['pressure']} {a['run']:<24} "
              f"guide {a['guide_mae']:10.3f}  oracle {a['oracle_mae']:10.3f}  "
              f"ratio {a['oracle_over_guide']:7.2f}  |g-o| {a['disagreement_mae']:10.3f}  "
              f"r(pred) {a['pred_corr']:.3f}  r(err) {a['err_corr']:6.3f}  "
              f"t {a['selected_t_mean']:.2f}")
    (REPO / "results/qm9/ss_diagnostics/guide_oracle_dose_response.json").write_text(
        json.dumps(agg, indent=2) + "\n")

    # ---- what the selection actually buys, paired by seed (docs table in section 6)
    def series(prop, run, key="oracle_mae"):
        rs = [r for r in out_rows if r["property"] == prop and r["run"] == run]
        return np.array([r[key] for r in sorted(rs, key=lambda x: x["seed"])])

    dec = []
    for prop in ("cv", "mu", "alpha", "gap", "homo", "lumo"):
        unit = next(r["unit"] for r in out_rows if r["property"] == prop)
        m, e, s = (series(prop, "guided"), series(prop, "final_endpoint"),
                   series(prop, "best_lookahead_pbvalid"))
        g = series(prop, "best_lookahead_pbvalid", "guide_mae")
        ge = series(prop, "final_endpoint", "guide_mae")
        dis = series(prop, "best_lookahead_pbvalid", "disagreement_mae")
        d = s - e                                   # paired: same trajectories
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        dec.append({
            "property": prop, "unit": unit, "n_seeds": len(d),
            "multimfm_endpoint_oracle_mae": float(m.mean()),
            "ss_endpoint_oracle_mae": float(e.mean()),
            "ss_selected_oracle_mae": float(s.mean()),
            "ss_selected_oracle_se": float(s.std(ddof=1) / np.sqrt(len(s))),
            "ss_endpoint_guide_mae": float(ge.mean()),
            "ss_selected_guide_mae": float(g.mean()),
            "paired_gain_vs_ss_endpoint": float(d.mean()),
            "paired_gain_se": se,
            "paired_gain_t": float(d.mean() / se) if se else None,
            "gain_vs_ss_endpoint_pct": float(100 * d.mean() / e.mean()),
            "gain_vs_multimfm_pct": float(100 * (s.mean() - m.mean()) / m.mean()),
            "guide_mae_change_pct": float(100 * (g.mean() - ge.mean()) / ge.mean()),
            "disagreement_selected": float(dis.mean()),
        })
    p = REPO / "results/qm9/ss_diagnostics/ss_advantage_decomposition.json"
    p.write_text(json.dumps(dec, indent=2) + "\n")
    print("wrote", p)
    for r in dec:
        print(f"{r['property']:>5} endpoint {r['ss_endpoint_oracle_mae']:10.4f} -> selected "
              f"{r['ss_selected_oracle_mae']:10.4f}  ({r['gain_vs_ss_endpoint_pct']:6.1f} %, "
              f"paired t={r['paired_gain_t']:6.2f})   guide {r['ss_selected_guide_mae']:10.4f} "
              f"({r['guide_mae_change_pct']:6.1f} %)")


if __name__ == "__main__":
    main()
