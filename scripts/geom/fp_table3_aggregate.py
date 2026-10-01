#!/usr/bin/env python
"""Pool the sharded n=4096 fingerprint runs into the two blocks of Table 3.

    python scripts/geom/fp_table3_aggregate.py

Each shard is 1,024 molecules on its own seed; four shards make one replicate of 4,096, and
three replicates make the reported mean. Pooling is done on the **per-molecule** arrays, not
by averaging shard means, so a shard that lost a molecule to a conversion failure does not
get the same weight as a full one.

The guided and unguided arms of a shard share noise and targets, so the gain is paired and
its error bar comes from the spread across replicates, not across molecules.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "results/geom/runs/fp_table3_n4096"
FLOORS = {"ours_n4096": (0.21951, "<= 71 heavy atoms"),
          "ours_max30_n4096": (0.20991, "<= 30 heavy atoms (TFG-Flow's cap)")}


def sem(xs):
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def main() -> None:
    blocks = defaultdict(lambda: defaultdict(list))   # arm -> replicate -> [record]
    for f in sorted(RUNS.glob("*.json")):
        d = json.loads(f.read_text())
        tag = d.get("tag", f.stem)
        arm = "ours_max30_n4096" if "max30" in tag else "ours_n4096"
        rep = tag.split("_rep")[1].split("_")[0]
        blocks[arm][rep].append(d)

    if not blocks:
        raise SystemExit(f"no runs yet in {RUNS.relative_to(REPO)}")

    out = {}
    for arm, reps in blocks.items():
        floor, label = FLOORS[arm]
        per_rep = []
        for rep, shards in sorted(reps.items()):
            g = [x for d in shards for x in d["guided"]["tanimoto_real_per_mol"]]
            u = [x for d in shards for x in d["unguided"]["tanimoto_real_per_mol"]]
            n = min(len(g), len(u))
            gm, um = st.mean(g), st.mean(u)
            per_rep.append({
                "replicate": rep, "n_shards": len(shards), "n_molecules": n,
                "guided": gm, "unguided": um, "gain": gm - um,
                "relative_pct": 100.0 * (gm / um - 1.0),
                "guided_pb": st.mean([d["guided"]["pb_valid"] for d in shards
                                      if d["guided"].get("pb_valid") is not None] or [float("nan")]),
                "unguided_pb": st.mean([d["unguided"]["pb_valid"] for d in shards
                                        if d["unguided"].get("pb_valid") is not None] or [float("nan")]),
            })
        G = [r["guided"] for r in per_rep]
        U = [r["unguided"] for r in per_rep]
        R = [r["relative_pct"] for r in per_rep]
        PB = [r["guided_pb"] for r in per_rep]
        UPB = [r["unguided_pb"] for r in per_rep]
        tot = sum(r["n_molecules"] for r in per_rep)
        out[arm] = {"label": label, "floor": floor, "replicates": per_rep,
                    "n_total": tot,
                    "guided_mean": st.mean(G), "guided_sem": sem(G),
                    "unguided_mean": st.mean(U), "unguided_sem": sem(U),
                    "relative_pct_mean": st.mean(R), "relative_pct_sem": sem(R),
                    "guided_margin": st.mean(G) - floor,
                    "unguided_margin": st.mean(U) - floor,
                    "guided_pb": st.mean(PB), "unguided_pb": st.mean(UPB)}

        print(f"\n=== {arm}  ({label}), floor {floor:.4f} ===")
        for r in per_rep:
            print(f"  rep {r['replicate']}: {r['n_shards']} shards, n={r['n_molecules']:5d}  "
                  f"guided {r['guided']:.4f}  unguided {r['unguided']:.4f}  "
                  f"rel {r['relative_pct']:+.2f}%  PB {r['guided_pb']:.3f}")
        o = out[arm]
        print(f"  TOTAL n={tot}:  guided {o['guided_mean']:.4f} +- {o['guided_sem']:.4f}  "
              f"(margin {o['guided_margin']:+.4f})   "
              f"unguided {o['unguided_mean']:.4f} +- {o['unguided_sem']:.4f} "
              f"(margin {o['unguided_margin']:+.4f})")
        print(f"  relative {o['relative_pct_mean']:+.2f}% +- {o['relative_pct_sem']:.2f}   "
              f"PB {o['guided_pb']:.3f} guided vs {o['unguided_pb']:.3f} unguided")

    dest = REPO / "results/geom/table3_n4096_confirmed.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
