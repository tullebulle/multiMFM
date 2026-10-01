#!/usr/bin/env python
"""Collect the findings-audit runs into markdown tables for docs/dna_findings_audit.md.

    python scripts/dna/findings_audit/collect.py [grad|sample|rewardT|junedose|all]

Reads only from results/dna/findings_audit/ and outputs/dna/findings_audit/ and prints; it writes
nothing except when called with --write_provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FA = REPO / "results" / "dna" / "findings_audit"
OUTS = REPO / "outputs" / "dna" / "findings_audit"


def md(df: pd.DataFrame, floatfmt: int = 4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.{floatfmt}g}" if pd.notna(v) else "")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, rule] + rows)


def show_grad() -> None:
    f = FA / "grad_posterior_accuracy" / "grad_agreement.csv"
    if not f.exists():
        print("[grad] not present yet:", f)
        return
    df = pd.read_csv(f)
    print("\n## (c) gradient agreement vs GLASS RK4-25 to 0.999 (same MC pool)\n")
    for t in sorted(df.t.unique()):
        s = df[df.t == t]
        piv = (s.groupby(["config", "n_steps", "end_time"])
                 [["cosine", "grad_norm", "ref_grad_norm", "endpoint_l1_vs_ref",
                   "argmax_margin", "soft_hard_gap", "endpoint_token_agree_vs_ref"]]
                 .mean().reset_index().sort_values("cosine", ascending=False))
        print(f"\n### t = {t}\n")
        print(md(piv))


def show_sample(sub: str = "sample_posterior_accuracy", title: str = "(c) end-to-end guided sampling") -> None:
    f = FA / sub / "sample_summary.csv"
    if not f.exists():
        print(f"[{sub}] not present yet:", f)
        return
    df = pd.read_csv(f)
    cols = [c for c in ["target", "config", "n", "unguided_oracle_mae", "guided_oracle_mae",
                        "guided_oracle_mean", "grad_norm_mean", "frac_steps_grad_above_clip",
                        "paired_improvement", "frac_improved"] if c in df.columns]
    pair = [c for c in df.columns if c.startswith("paired_mae_minus_")]
    print(f"\n## {title} -- {sub}\n")
    for tg in sorted(df.target.unique()):
        print(f"\n### y* = {tg:+.0f}\n")
        print(md(df[df.target == tg][cols + pair].sort_values("guided_oracle_mae")))


def show_reward_t() -> None:
    for sub in ("reward_temperature", "reward_temperature_gpu"):
        f = FA / sub / "reward_temperature_ess.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        print(f"\n## (d) MC-weight concentration -- {sub}\n")
        s = df[df.mc == df.mc.min()]
        piv = s.pivot_table(index=["reward", "coefficient", "target"], columns="t",
                            values=["ess_mean", "max_weight_mean"]).round(3)
        print(piv.to_string())
    for tag in ("0p045", "0p5", "1p0"):
        show_sample(f"sample_reward_scale_{tag}", "(d) guided sampling vs reward temperature")


def show_junedose() -> None:
    rows = []
    for d in sorted(OUTS.glob("*/sample_scores.csv")):
        df = pd.read_csv(d)
        import json
        a = json.loads((d.parent / "args.json").read_text()) if (d.parent / "args.json").exists() else {}
        tg = float(a.get("target_c0", "nan"))
        gcol = "guided" if "guided" in df.columns else None
        ucol = "unguided" if "unguided" in df.columns else None
        rec = {"run": d.parent.name, "n": len(df), "target": tg,
               "guide_t_start": a.get("guide_t_start"), "reward_scale": a.get("reward_scale"),
               "mc": a.get("mc"), "nfe_value": a.get("nfe_value"), "nfe_sample": a.get("nfe_sample"),
               "t_max": a.get("t_max"), "seed": a.get("seed")}
        if gcol:
            rec["guide_mae_guided"] = float((df[gcol] - tg).abs().mean())
            rec["guide_mean_guided"] = float(df[gcol].mean())
        if ucol:
            rec["guide_mae_unguided"] = float((df[ucol] - tg).abs().mean())
        rows.append(rec)
    if rows:
        print("\n## (a) June-dose sampling runs on the parent-disjoint model\n")
        print(md(pd.DataFrame(rows)))
    for d in sorted(OUTS.glob("diversity*/table7_generated.csv")):
        print(f"\n### diversity -- {d.parent.name}\n")
        print(md(pd.read_csv(d)))
    for d in sorted(OUTS.glob("diversity*/table6_baselines.csv")):
        print(f"\n### real baselines -- {d.parent.name}\n")
        print(md(pd.read_csv(d)))


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("grad", "all"):
        show_grad()
    if what in ("sample", "all"):
        show_sample()
    if what in ("rewardT", "all"):
        show_reward_t()
    if what in ("junedose", "all"):
        show_junedose()


if __name__ == "__main__":
    main()
