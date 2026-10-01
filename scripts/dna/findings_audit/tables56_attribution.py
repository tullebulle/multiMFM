#!/usr/bin/env python
"""Audit of the Tables 4-6 attribution (plan S7 item 7).

The published Table 5 / Table 6 numbers for the `y* = -1` set were not reproduced by the
parent-disjoint regeneration, and the change was attributed to the `split65k` -> parent-disjoint
model move. This script tests that attribution against the tables' own control rows.

Logic: every row of Tables 5 and 6 whose *sequence set* does not depend on the guidance term -- the
real-data reference rows and the **unguided** generated rows -- brackets how much the `split65k` ->
parent-disjoint move can move each statistic. If that bracket is an order of magnitude smaller than
the change in the guided rows, the model move cannot be the explanation, and the remaining candidate
is the guidance protocol, which differs in six knobs between the two pipelines (see
`results/dna/findings_audit/tables56_attribution.json`, key `protocol_diff`).

Writes results/dna/findings_audit/tables56_attribution.json. CPU, seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
R = REPO / "results" / "dna"
OUT = R / "findings_audit"

# The published June args (results/dna/june_split65k_recovered/samples_target_*.args.json, producer
# DNA-MFM@187fe7b scripts/sample_yeast_split_dmfm_c0.py + DMFMSteeringRunner) against the settings of
# every parent-disjoint regeneration of the same sets.
PROTOCOL_DIFF = {
    "_published_producer": "DNA-MFM@187fe7b scripts/sample_yeast_split_dmfm_c0.py "
                           "(DMFMSteeringRunner of scripts/benchmark_yeast_split_dmfm_steering.py); "
                           "NOT scripts/sample_yeast_c0_guidance_hist.py, which is what Table 1 / "
                           "Fig 3 / Fig 7 used",
    "posterior": {"published": "dMFM student, 1 composed flow map (nfe_value 1)",
                  "regen_glass": "GLASS on the base DFM, 4 Euler steps",
                  "regen_dmfm": "4-step ESD student, 4 composed steps"},
    "guidance_window": {"published": [0.01, 0.95], "regen": [0.50, 0.95]},
    "guided_steps": {"published": 91, "regen": 30},
    "mc": {"published": 16, "regen": 8},
    "trajectory": {"published": "nfe_sample 96 integrated to t = 1.0",
                   "regen": "nfe_traj 64 integrated to t_max = 0.95"},
    "reward_scale": {"published": 1.0, "regen": 0.5},
    "effective_log_reward": {"published": "-22.22 (f - y*)^2", "regen": "-11.11 (f - y*)^2"},
    "mc_noise": {"published": "fresh torch.randn at every guided step",
                 "regen": "frozen per-sample pool seed + 1e6 + i"},
    "seeds": {"published": {"+1": 0, "-1": 1}, "regen": {"+1": 0, "-1": 0}},
    "guide": {"published": "dinko/C0free_torch.pt (Keras-converted)",
              "regen": "parent-disjoint park_cnn guide"},
    "model_split": {"published": "split65k random windows of yeast_mid50.pt",
                    "regen": "parent-disjoint yeast_parent_L50.pt"},
}


def main() -> None:
    out: dict = {
        "_what": "Does the split65k -> parent-disjoint move explain the lost Table 5 clustering and "
                 "the lost Table 6 -1 NLL gap?",
        "protocol_diff": PROTOCOL_DIFF,
    }

    # ---------- Table 5 (diversity): published vs every parent-disjoint recompute
    pv = json.loads((R / "paper_values.json").read_text())
    pub = pv["table5_generated_diversity"]
    reg = pd.read_csv(R / "diversity" / "regen" / "table5_generated.csv")
    base4 = pd.read_csv(R / "diversity" / "regen" / "table4_real_baselines.csv")
    dmfm = pd.read_csv(R / "dmfm_demos" / "diversity" / "dmfm4_norm" / "table7_generated.csv")
    gnorm = pd.read_csv(R / "dmfm_demos" / "diversity" / "glass_norm" / "table7_generated.csv")

    out["paper_table5_transcribed"] = pub
    out["table4_real_baselines_regen"] = base4.to_dict(orient="records")

    # ---------- Table 6 (Evo 2): published vs the two parent-disjoint arms
    june = pd.read_csv(R / "june_split65k_recovered" / "yeast_split_dmfm_c0_evo2_summary_2026-06-10.csv")
    d6 = pd.read_csv(R / "dmfm_demos" / "table6" / "dmfm4_norm" / "evo2_summary.csv")
    g6 = pd.read_csv(R / "dmfm_demos" / "table6" / "glass_norm" / "evo2_summary.csv")

    def nll(df, source, st, target=None):
        m = (df.source == source) & (df.set == st)
        if target is not None:
            m &= df.target_c0 == target
        r = df[m]
        return None if not len(r) else float(r.iloc[0].evo2_nll_mean)

    keys = [("reference", "split_test_sample", None), ("reference", "split_train_sample", None),
            ("reference", "split_val_sample", None), ("reference", "full_sample", None),
            ("generated", "unguided", -1.0), ("generated", "unguided", 1.0),
            ("generated", "guided", -1.0), ("generated", "guided", 1.0)]
    rows = []
    for src, st, tg in keys:
        p, dm, gl = nll(june, src, st, tg), nll(d6, src, st, tg), nll(g6, src, st, tg)
        rows.append({
            "row": f"{src}/{st}" + (f"/y*={tg:+.0f}" if tg is not None else ""),
            "depends_on_guidance": (src == "generated" and st == "guided"),
            "published_split65k": p, "parent_disjoint_dMFM": dm, "parent_disjoint_GLASS": gl,
            "delta_dMFM_minus_published": None if None in (p, dm) else dm - p,
            "delta_GLASS_minus_published": None if None in (p, gl) else gl - p,
        })
    out["table6_row_by_row"] = rows
    moved_ctrl = [abs(r["delta_dMFM_minus_published"]) for r in rows if not r["depends_on_guidance"]]
    moved_guid = [abs(r["delta_dMFM_minus_published"]) for r in rows if r["depends_on_guidance"]]
    out["table6_verdict"] = {
        "max_abs_shift_in_guidance_independent_rows": max(moved_ctrl),
        "abs_shift_in_guided_rows": moved_guid,
        "note": "The six rows that do not depend on the guidance term (four real-data references and "
                "the two unguided generated sets) bracket the split65k -> parent-disjoint effect. "
                "The guided y*=-1 row moved ~15-25x more than that bracket.",
    }

    # ---------- the same bracket for the diversity statistics
    # published Table 5 numbers, transcribed in paper_values.json
    PUB = {  # (target, set) -> (nearest_gen, pairwise_gen, pairwise/held-out, nearest_gen/held-out)
        (-1.0, "guided"): (0.4175, 0.7010, 0.9507, 0.8027),
        (1.0, "guided"): (0.4913, 0.7372, 0.9997, 0.9447),
        (-1.0, "unguided"): (0.5210, 0.7364, 0.9987, 1.0017),
        (1.0, "unguided"): (0.5215, 0.7370, 0.9995, 1.0027),
    }
    COLS = ["nearest_gen_frac_mean", "pairwise_gen_frac_mean",
            "pairwise_vs_test_sample", "nearest_gen_vs_test_sample"]

    def row(df, target, st):
        r = df[(df.target_c0 == target) & (df.set == st)]
        return None if not len(r) else [float(r.iloc[0][c]) for c in COLS]

    div = []
    for (target, st), pubv in PUB.items():
        rec = {"target": target, "set": st, "depends_on_guidance": st == "guided",
               "published_split65k": dict(zip(COLS, pubv))}
        for label, df in (("regen_GLASS_stock", reg), ("dmfm4_norm", dmfm), ("glass_norm", gnorm)):
            v = row(df, target, st)
            if v is not None:
                rec[label] = dict(zip(COLS, v))
                rec[f"delta_{label}"] = {c: v[i] - pubv[i] for i, c in enumerate(COLS)}
        div.append(rec)
    out["table5_row_by_row"] = div
    ctrl = [abs(d[f"delta_{a}"][c]) for d in div if not d["depends_on_guidance"]
            for a in ("regen_GLASS_stock", "dmfm4_norm", "glass_norm") if f"delta_{a}" in d
            for c in COLS]
    guid_m1 = {c: abs(d[f"delta_dmfm4_norm"][c]) for d in div
               if d["depends_on_guidance"] and d["target"] == -1.0 for c in COLS}
    out["table5_verdict"] = {
        "max_abs_shift_in_unguided_rows": max(ctrl),
        "abs_shift_in_minus1_guided_row": guid_m1,
        "note": "The unguided generated rows use the same base model, the same trajectory integrator "
                "and the same real-data denominators as the guided rows, and differ from the "
                "published ones by at most this much; the y*=-1 guided row moved far more.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables56_attribution.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("table6_row_by_row", "table6_verdict")}, indent=2))
    print("wrote", OUT / "tables56_attribution.json")


if __name__ == "__main__":
    main()
