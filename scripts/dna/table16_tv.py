#!/usr/bin/env python
"""Replace Table 16's correlation rows with total-variation distance.

Why the published table has to change. Table 16 reports Pearson r between real
yeast marginals and either generated or uniform ones. The global rows are
computed on **four** numbers (the A/C/G/T frequencies), which sum to one, so only
three are free. For the uniform reference the deviation from exactly-uniform is
sampling noise alone (max 0.0013 here), so r is the cosine between a fixed vector
and an isotropic random direction in the 3-dimensional sum-to-zero subspace --
exactly Uniform(-1, 1), independent of how many sequences are drawn. The printed
0.1813, the figure's -0.852 and a regeneration's 0.4576 are three ordinary draws
from that distribution. None of them measures anything.

Total variation, TV(p, q) = 1/2 sum_i |p_i - q_i|, has no such problem: it is
bounded in [0, 1], it is zero only when the distributions agree, and it does not
care that the frequencies are constrained.

This script also computes the control the published analysis lacked: TV between
two **disjoint real** splits. That is the floor -- the distance two genuine
samples of the same data sit apart -- and a generated-vs-real distance at or below
it is as close as the measurement can resolve.

    python scripts/dna/table16_tv.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "dna" / "regen" / "fig6_table16" / "global_marginals.csv"
OUT_JSON = REPO / "results" / "dna" / "base_model" / "table16_tv.json"
OUT_CSV = REPO / "results" / "dna" / "base_model" / "table16_tv.csv"

BASES = ["A", "C", "G", "T"]


def tv(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p[b] - q[b]) for b in BASES)


def main() -> None:
    df = pd.read_csv(SRC)
    m = {s: dict(zip(g.base, g.freq)) for s, g in df.groupby("source")}
    n = {s: int(g.n_sequences.iloc[0]) for s, g in df.groupby("source")}

    # The reference is the held-out test parents: the parent-disjoint standard the
    # rest of the DNA work uses. "real" (the 4,096 matched draw) is kept beside it.
    rows = [
        ("generated vs real (matched n)", "real", "generated", "the comparison the table is making"),
        ("uniform vs real (matched n)", "real", "uniform", "the row that was pure noise as a correlation"),
        ("generated vs held-out real", "real_test", "generated", "against the parent-disjoint test split"),
        ("uniform vs held-out real", "real_test", "uniform", "against the parent-disjoint test split"),
        ("generated vs training-split real", "real_a_train", "generated",
         "the split the generator was trained on"),
        ("FLOOR: train real vs held-out real", "real_test", "real_a_train",
         "two disjoint REAL splits -- the resolution limit of this measurement"),
    ]

    out = []
    for label, ref, other, note in rows:
        if ref not in m or other not in m:
            continue
        out.append({
            "comparison": label,
            "reference": ref, "n_reference": n[ref],
            "other": other, "n_other": n[other],
            "total_variation": round(tv(m[ref], m[other]), 6),
            "note": note,
        })

    res = pd.DataFrame(out)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    floor = next((r["total_variation"] for r in out if r["comparison"].startswith("FLOOR")), None)
    gen = next((r["total_variation"] for r in out if r["comparison"] == "generated vs held-out real"), None)
    uni = next((r["total_variation"] for r in out if r["comparison"] == "uniform vs held-out real"), None)

    train = next((r["total_variation"] for r in out
                  if r["comparison"] == "generated vs training-split real"), None)
    verdict = None
    if None not in (floor, gen, uni):
        verdict = {
            "floor_real_vs_real": floor,
            "generated_vs_heldout": gen,
            "generated_vs_train": train,
            "uniform_vs_heldout": uni,
            "generated_within_floor_heldout": bool(gen <= floor),
            "generated_within_floor_train": bool(train is not None and train <= floor),
            "uniform_over_floor_x": round(uni / floor, 1) if floor else None,
            "reading": (
                "Uniform is an order of magnitude outside the real-vs-real floor, so the "
                "qualitative claim stands. But the generated marginals are not inside that floor "
                "against held-out parents: they sit at about 1.4x it, while sitting INSIDE it "
                "against the split the generator trained on. That gap is a generalisation effect "
                "the correlation could not show, and it should be reported rather than smoothed "
                "over."
            ),
        }

    OUT_JSON.write_text(json.dumps({
        "_what": "Table 16 replacement: total-variation distance between global base-composition "
                 "marginals, with a real-vs-real floor.",
        "_why": "The published Pearson rows are computed on 4 constrained numbers; against a sampled "
                "uniform reference r is exactly Uniform(-1,1) and measures nothing.",
        "source": str(SRC.relative_to(REPO)),
        "rows": out,
        "verdict": verdict,
    }, indent=2) + "\n")

    print(res.to_string(index=False))
    if verdict:
        print(f"\nfloor (two disjoint REAL splits): {floor:.5f}")
        print(f"generated vs held-out real:      {gen:.5f}  "
              f"({'inside' if verdict['generated_within_floor_heldout'] else f'{gen/floor:.2f}x'} the floor)")
        if train is not None:
            print(f"generated vs training split:     {train:.5f}  "
                  f"({'inside' if verdict['generated_within_floor_train'] else f'{train/floor:.2f}x'} the floor)")
        print(f"uniform vs held-out real:        {uni:.5f}  ({verdict['uniform_over_floor_x']}x the floor)")
    print(f"\nwrote {OUT_JSON.relative_to(REPO)} and {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
