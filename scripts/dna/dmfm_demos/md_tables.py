#!/usr/bin/env python
"""Print the markdown comparison tables that go into docs/dmfm_demo_reruns.md.

Pure formatting over the collected results: nothing here recomputes a number. Kept in the repo
so the tables in the write-up can be regenerated from the CSVs rather than retyped.

    python scripts/dna/dmfm_demos/md_tables.py fig7
    python scripts/dna/dmfm_demos/md_tables.py fig8
    python scripts/dna/dmfm_demos/md_tables.py table5
    python scripts/dna/dmfm_demos/md_tables.py table6
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results" / "dna"
DEM = RES / "dmfm_demos"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def fig7() -> None:
    lad = pd.read_csv(DEM / "fig7" / "ladder_summary.csv")
    keep = {
        "arm": "arm", "guidance_frac": "gf", "n": "n",
        "guide_guided_mean": "guide mean", "guide_guided_mae": "guide MAE",
        "guide_guided_frac_within_0.10": "guide frac within 0.10",
        "oracle_guided_mean": "oracle mean", "oracle_guided_mae": "oracle MAE",
        "oracle_paired_mae_improvement": "oracle paired improvement",
        "oracle_paired_mae_improvement_ci_low": "CI low",
        "oracle_paired_mae_improvement_ci_high": "CI high",
    }
    t = lad[[c for c in keep if c in lad]].rename(columns=keep)
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map((lambda v: f"{v:g}") if c == "gf" else (lambda v: f"{v:.4f}"))
    print("Unguided reference (identical in every row, same x0): "
          f"guide mean {lad.guide_unguided_mean.iloc[0]:.4f}, guide MAE "
          f"{lad.guide_unguided_mae.iloc[0]:.4f}, oracle MAE "
          f"{lad.oracle_unguided_mae.iloc[0]:.4f}\n")
    print(md(t))


def fig8() -> None:
    rows = []
    for arm in ("dmfm", "glass"):
        d = pd.read_csv(DEM / "fig8" / arm / "trace.csv")
        m = json.loads((DEM / "fig8" / arm / "metadata.json").read_text())
        ids = list(m["settings"]["sample_ids"])
        tgt = float(m["settings"]["target_c0"])
        for sid in ids:
            s = d[d.sample_idx == sid]
            g, u = s[s.arm == "guided"], s[s.arm == "unguided"]
            gw = g[g.guidance_applied]
            fg = m["endpoints"]["guided"]["final_score"][ids.index(sid)]
            fu = m["endpoints"]["unguided"]["final_score"][ids.index(sid)]
            rows.append({
                "arm": arm, "sample_idx": sid,
                "final C0 guided": f"{fg:+.4f}", "final C0 unguided": f"{fu:+.4f}",
                # Pipes are escaped: these strings become markdown table headers.
                r"\|C0-0.30\| guided": f"{abs(fg - tgt):.4f}",
                r"\|C0-0.30\| unguided": f"{abs(fu - tgt):.4f}",
                r"mean \|g\| in window": f"{gw.grad_norm.mean():.2f}",
                r"frac \|g\| > clip": f"{(gw.grad_norm > 10).mean():.3f}",
                "V final guided": f"{g.value.iloc[-1]:.3f}",
                "V final unguided": f"{u.value.iloc[-1]:.3f}",
            })
    print(md(pd.DataFrame(rows)))


_T5 = ["score mean (guide)", "score mean (oracle)", "nearest generated",
       "pairwise generated", "pairwise / held-out", "nearest gen / held-out"]


def _first(path: Path, *names: str) -> Path:
    """The round-1 GLASS collection named these `table5_*`; the analyser writes `table7_*`
    (it predates the paper's renumbering). Accept either."""
    for n in names:
        if (path / n).is_file():
            return path / n
    raise FileNotFoundError(f"none of {names} under {path}")


def table5() -> None:
    srcs = {
        "GLASS parent-disjoint, stock rule": RES / "diversity" / "regen",
        "GLASS parent-disjoint, scale-free": DEM / "diversity" / "glass_norm",
        "dMFM parent-disjoint, scale-free": DEM / "diversity" / "dmfm4_norm",
    }
    pub = json.loads((RES / "paper_values.json").read_text())["table5_generated_diversity"]
    cols = pub["_columns"]
    rows = []
    for key, label in (("-1_guided", "-1 guided"), ("+1_guided", "+1 guided"),
                       ("-1_unguided", "unguided")):
        v = dict(zip(cols, pub[key]))
        rows.append({"source": "published (split65k, GLASS)", "set": label,
                     "guide score": f"{v['guide_score_mean']:.4f}",
                     "nearest gen": f"{v['nearest_gen']:.4f}",
                     "pairwise gen": f"{v['pairwise_gen']:.4f}",
                     "pairwise / held-out": f"{v['pairwise_over_heldout']:.4f}",
                     "nearest gen / held-out": f"{v['nearest_gen_over_heldout']:.4f}",
                     "oracle MAE": ""})
    for label, path in srcs.items():
        g = pd.read_csv(_first(path, "table7_generated.csv", "table5_generated.csv"))
        a = pd.read_csv(_first(path, "table7_additions.csv", "table5_additions.csv"))
        for tgt, setname, nice in ((-1.0, "guided", "-1 guided"), (1.0, "guided", "+1 guided"),
                                   (-1.0, "unguided", "unguided")):
            r = g[(g.target_c0 == tgt) & (g["set"] == setname)].iloc[0]
            ra = a[(a.target_c0 == tgt) & (a["set"] == setname)].iloc[0]
            rows.append({
                "source": label, "set": nice,
                "guide score": f"{r['guide_score_mean']:.4f}",
                "nearest gen": f"{r['nearest_gen_frac_mean']:.4f}",
                "pairwise gen": f"{r['pairwise_gen_frac_mean']:.4f}",
                "pairwise / held-out": f"{r['pairwise_vs_test_sample']:.4f}",
                "nearest gen / held-out": f"{r['nearest_gen_vs_test_sample']:.4f}",
                "oracle MAE": f"{ra['oracle_target_mae']:.4f}",
            })
    print(md(pd.DataFrame(rows)))


def table6() -> None:
    pub = json.loads((RES / "paper_values.json").read_text())["table6_evo2_nll"]
    cols = pub["_columns"]
    keys = [("generated_guided_-1", "-1 guided"), ("generated_unguided_-1", "-1 unguided"),
            ("generated_guided_+1", "+1 guided"), ("generated_unguided_+1", "+1 unguided"),
            ("reference_split_test_sample", "held-out test")]
    rows = []
    for k, nice in keys:
        v = dict(zip(cols, pub[k]))
        rows.append({"source": "published (split65k, GLASS)", "set": nice,
                     "NLL": f"{v['nll_mean']:.4f}",
                     "CI": f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]",
                     "delta held-out": f"{v['delta_heldout']:+.4f}"})
    for label, arm in (("GLASS parent-disjoint, scale-free", "glass_norm"),
                       ("dMFM parent-disjoint, scale-free", "dmfm4_norm")):
        p = DEM / "table6" / arm / "evo2_summary.csv"
        if not p.is_file():
            print(f"<!-- missing {p} -->")
            continue
        ev = pd.read_csv(p)
        held = ev[(ev.source == "reference") & (ev["set"] == "split_test_sample")]
        for tgt, setname, nice in ((-1.0, "guided", "-1 guided"), (-1.0, "unguided", "-1 unguided"),
                                   (1.0, "guided", "+1 guided"), (1.0, "unguided", "+1 unguided")):
            m = (ev.source == "generated") & (ev["set"] == setname) & (ev.target_c0 == tgt)
            if not m.any():
                continue
            r = ev[m].iloc[0]
            rows.append({"source": label, "set": nice,
                         "NLL": f"{r.evo2_nll_mean:.4f}",
                         "CI": f"[{r.evo2_nll_mean_ci95_low:.4f}, {r.evo2_nll_mean_ci95_high:.4f}]",
                         "delta held-out": f"{r.evo2_nll_delta_vs_split_test:+.4f}"})
        if len(held):
            r = held.iloc[0]
            rows.append({"source": label, "set": "held-out test",
                         "NLL": f"{r.evo2_nll_mean:.4f}",
                         "CI": f"[{r.evo2_nll_mean_ci95_low:.4f}, {r.evo2_nll_mean_ci95_high:.4f}]",
                         "delta held-out": "+0.0000"})
    print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    {"fig7": fig7, "fig8": fig8, "table5": table5, "table6": table6}[sys.argv[1]]()
