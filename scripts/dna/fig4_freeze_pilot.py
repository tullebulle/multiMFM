#!/usr/bin/env python
"""Record how each Fig 4 baseline's hyperparameters were chosen.

The pilot (`PLAN=pilot`, n = 32, seed 12345 -- a different seed and n from the
reported sweep) runs a grid per method. `dmfm.benchmarks.plan.frozen_from_pilot`
picks the best family per method *at matched NFE* (FK: all configurations cost the
same 768 NFE, so plain MAE; beam/MCTS: each family's MAE interpolated in log NFE
over the NFE range all families share, then averaged), and the sweep uses those
winners automatically. This script only writes the record,
`results/dna/fig4/pilot_choice.json`.

    python scripts/dna/fig4_freeze_pilot.py
"""

from __future__ import annotations

import json

from dmfm import paths
from dmfm.benchmarks import plan

OUT = paths.RESULTS / "fig4"


def main() -> None:
    pilot_dir = paths.OUTPUTS / "fig4" / "panelB" / "pilot"
    scores = plan.pilot_scores(pilot_dir)
    chosen = plan.frozen_from_pilot(pilot_dir)
    grid = []
    for m, fams in scores.items():
        for fam, v in sorted(fams.items(), key=lambda kv: kv[1]["score"]):
            grid.append(
                {
                    "method": m,
                    **dict(zip(plan.FREEZE_KEYS[m], fam)),
                    "score_matched_nfe_mae": v["score"],
                    "matched_nfe_range": v["matched_nfe_range"],
                    "points_nfe_mae": v["points"],
                    "chosen": dict(zip(plan.FREEZE_KEYS[m], fam)) == chosen[m],
                }
            )
    payload = {
        "_what": "Hyperparameters of the Fig 4 (right) baselines, chosen on the pilot (n = 32, seed 12345) "
        "and then frozen for the reported sweep (n = 100 x 3 repeats, seeds 0/1/2). Families are compared "
        "at matched NFE, so no family wins by spending more. The dMFM was not tuned: it uses the settings "
        "of the original benchmark script.",
        "pilot_dir": paths.rel(pilot_dir),
        "chosen": chosen,
        "used_fallback": [m for m in plan.FALLBACK_FROZEN if m not in scores],
        "grid": grid,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_choice.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
