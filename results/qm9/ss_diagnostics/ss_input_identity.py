"""Prove that the guide and the oracle receive byte-identical inputs inside steer-search.

Diagnostic for docs/ss_guide_oracle_gap.md. Runs a tiny CPU steer-search
(``--lookahead-source both``) with a forward-pre-hook on both TFG regressors that
records a SHA-256 of every ``(coords, atom_types, mask)`` triple each one is given,
tagged by the call site. It then checks that

  * every tensor the *oracle* ever sees was also given to the *guide*
    (``evaluate_state`` scores one state with both), and
  * no oracle call happens inside the selection phase (no evaluator leak),

and dumps the per-candidate statistics of the look-ahead pool: guide and oracle
predictions of *all* stored look-aheads, so the winner's-curse geometry
(argmin over the pool on the guide vs. the oracle's view of the same candidates)
can be measured directly.

    ~/micromamba/envs/multimfm/bin/python results/qm9/ss_diagnostics/ss_input_identity.py \
        --property gap --num-samples 4 --sample-steps 8 --value-samples 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import multimfm.steer_search as ss  # noqa: E402
from multimfm.glass_guidance import tfg_property_prediction  # noqa: E402
from multimfm.qm9_data import load_tfg_regressor as _real_loader  # noqa: E402

CALLS: dict[str, list[dict]] = {"guide": [], "oracle": []}
PHASE = {"name": "init"}


def _hash(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        a = np.ascontiguousarray(t.detach().cpu().numpy())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def patched_loader(kind, prop, device):
    reg = _real_loader(kind, prop, device)

    def pre_hook(_m, inputs):
        coords, atom_types, mask = inputs
        CALLS[kind].append({
            "phase": PHASE["name"],
            "hash": _hash(coords, atom_types, mask),
            "shape": list(coords.shape),
            "coord_com_absmax": float(coords.mean(dim=1).abs().max()),
            "dtype": str(coords.dtype),
            "n_mask_atomtype": int((atom_types == 8).logical_and(mask.bool()).sum()),
        })

    reg.register_forward_pre_hook(pre_hook)
    return reg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--property", default="gap")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=8)
    ap.add_argument("--value-samples", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/qm9/ss_diagnostics/ss_input_identity.json")
    args = ap.parse_args(argv)

    ss.load_tfg_regressor = patched_loader

    # tag the phase so a leak into selection would be visible
    real_eval = ss.evaluate_state
    real_pred = ss.predicted_property

    def tagged_eval(*a, **k):
        PHASE["name"] = "evaluate_state"
        try:
            return real_eval(*a, **k)
        finally:
            PHASE["name"] = "other"

    def tagged_pred(*a, **k):
        if PHASE["name"] != "evaluate_state":
            PHASE["name"] = "selection_or_guidance"
        return real_pred(*a, **k)

    ss.evaluate_state = tagged_eval
    ss.predicted_property = tagged_pred

    out_dir = REPO / "outputs" / "ss_input_identity"
    argv2 = [
        "--device", "cpu", "--property-name", args.property,
        "--num-samples", str(args.num_samples), "--sample-steps", str(args.sample_steps),
        "--value-samples", str(args.value_samples), "--value-glass-steps", "2",
        "--lookahead-source", "both", "--lookahead-topk", "64",
        "--mu", "11.0", "--reward-scale", "12000.0", "--pb-workers", "2",
        "--output-dir", str(out_dir),
    ]
    ss.main(argv2)

    g = CALLS["guide"]
    o = CALLS["oracle"]
    guide_hashes = {c["hash"] for c in g}
    report = {
        "args": {k: str(v) for k, v in vars(args).items()} | {"steer_search_argv": argv2},
        "n_guide_calls": len(g),
        "n_oracle_calls": len(o),
        "oracle_phases": sorted({c["phase"] for c in o}),
        "guide_phases": sorted({c["phase"] for c in g}),
        "every_oracle_input_also_given_to_guide": all(c["hash"] in guide_hashes for c in o),
        "oracle_calls_outside_evaluate_state": [c["phase"] for c in o
                                                if c["phase"] != "evaluate_state"],
        "oracle_call_detail": o,
        "guide_call_detail_evaluate_state": [c for c in g if c["phase"] == "evaluate_state"],
    }
    summary = json.loads((out_dir / "summary.json").read_text())
    report["ss_summary_rows"] = [{k: r.get(k) for k in
                                  ("run", "guide_mae", "oracle_mae", "pb_pb_valid",
                                   "selected_t_mean", "frac_selected_lookahead")}
                                 for r in summary["summary"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("oracle_call_detail",
                                   "guide_call_detail_evaluate_state")}, indent=2))
    print("wrote", args.out)
    if not report["every_oracle_input_also_given_to_guide"]:
        raise SystemExit("MISMATCH: the oracle saw a tensor the guide did not")


if __name__ == "__main__":
    main()
