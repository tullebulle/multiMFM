#!/usr/bin/env python
"""Summarise a deficit-rescue run: per-config guided error and paired dMFM-vs-GLASS differences.

    python scripts/dna/rescue/collect_deficit_rescue.py results/dna/deficit_rescue/rule_L50_n200

Reads ``target_*/sample_scores.csv`` (written by ``dmfm.experiments.rescue_dmfm_glass_c0``) and
writes ``summary.csv`` and ``paired.csv`` next to them. Pairing is per sample id against a named
GLASS configuration; the CI is a 2,000-resample percentile bootstrap of the mean paired difference
with ``default_rng(0)``, the method behind Table 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOOT, SEED = 2000, 0


def boot_ci(v: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(v)
    m = np.array([v[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(root: Path) -> None:
    frames = [pd.read_csv(p) for p in sorted(root.glob("target_*/sample_scores.csv"))]
    if not frames:
        raise SystemExit(f"no target_*/sample_scores.csv under {root}")
    df = pd.concat(frames, ignore_index=True)
    recs = []
    for (target, config), g in df.groupby(["target", "config"]):
        err = (g["oracle_guided"] - target).abs().values
        lo, hi = boot_ci(err)
        recs.append({
            "target": target, "config": config, "n": len(g),
            "kind": g["kind"].iloc[0], "n_steps": int(g["n_steps"].iloc[0]),
            "mc": int(g["mc"].iloc[0]), "nfe_value": int(g["nfe_value"].iloc[0]),
            "scale_rule": g["scale_rule"].iloc[0],
            "guidance_frac": float(g["guidance_frac"].iloc[0]),
            "unguided_oracle_mae": float((g["oracle_unguided"] - target).abs().mean()),
            "guided_oracle_mae": float(err.mean()),
            "guided_oracle_mae_ci_lo": lo, "guided_oracle_mae_ci_hi": hi,
            "guided_guide_mae": float((g["guide_guided"] - target).abs().mean()),
            "guide_within_0.10": float(((g["guide_guided"] - target).abs() < 0.10).mean()),
            "frac_improved": float((((g["oracle_unguided"] - target).abs() - err) > 0).mean()),
            "grad_norm_raw_mean": float(g["grad_norm_raw_mean"].mean()),
            "grad_norm_post_mean": float(g["grad_norm_post_mean"].mean()),
            "frac_steps_grad_above_clip": float(g["frac_steps_grad_above_clip"].mean()),
            "frac_tail_clamped": float(g["frac_tail_clamped"].mean()),
        })
    summary = pd.DataFrame(recs).sort_values(["target", "guided_oracle_mae"])
    summary.to_csv(root / "summary.csv", index=False)

    # Paired differences against every GLASS configuration in the run, on shared sample ids.
    glass = sorted(summary[summary.kind == "glass"].config.unique())
    pairs = []
    for target in sorted(df.target.unique()):
        sub = df[df.target == target]
        for base in glass:
            b = sub[sub.config == base].set_index("sample_idx")
            for config in sorted(sub.config.unique()):
                if config == base:
                    continue
                a = sub[sub.config == config].set_index("sample_idx")
                ids = a.index.intersection(b.index)
                d = ((a.loc[ids, "oracle_guided"] - target).abs()
                     - (b.loc[ids, "oracle_guided"] - target).abs()).values
                lo, hi = boot_ci(d)
                pairs.append({"target": target, "config": config, "baseline": base,
                              "n": len(ids), "mean_diff": float(d.mean()),
                              "sem": float(d.std(ddof=1) / len(d) ** 0.5),
                              "ci_lo": lo, "ci_hi": hi,
                              "frac_config_better": float((d < 0).mean())})
    pd.DataFrame(pairs).to_csv(root / "paired.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["target", "config", "n", "nfe_value", "scale_rule", "guidance_frac",
            "guided_oracle_mae", "guided_oracle_mae_ci_lo", "guided_oracle_mae_ci_hi",
            "guided_guide_mae", "grad_norm_raw_mean", "grad_norm_post_mean"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {root/'summary.csv'} and {root/'paired.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
