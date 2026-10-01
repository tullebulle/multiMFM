#!/usr/bin/env python
"""multiMFM GEOM steering with a *scale-free* guidance step, so the displacement
dose is swept directly instead of through lambda and mu together.

Motivation (docs/geom_rescue_attempt.md H4). FP2 is invariant to coordinate
displacements below ~0.006 A and randomised by ~0.2 A, with a transition band in
between (self-Tanimoto 0.978 at 0.02 A, 0.793 at 0.06 A). The recorded grid swept
(lambda, mu), and because the value gradient's magnitude scales with lambda, the
*displacement* was never swept cleanly: the smallest recorded point corresponds to
a coherent coordinate displacement of roughly 0.1 A and the largest to tens of
Angstrom. **The 0.02-0.06 A band, the only band in which FP2 both responds and is
not randomised, was never visited.**

With `multimfm.geom_rescue.install_scalefree_guidance()` and
`--guidance-max-coord-rms R`, each molecule's gradient is renormalised to RMS R,
so the displacement per guided step is exactly `mu * R * dt` and the coherent
total over the guided window is `mu * R * sum(dt) * normalizer` Angstrom. lambda
then affects only the *direction* of the tilt, which also makes the paper's
literal `lambda = 1` testable without the 100x magnitude blow-up that destroyed
the molecules in the recorded lambda = 1 runs.

All arguments are `multimfm.geom.mfm_steering`'s. The record additionally carries
the realised RMS coordinate displacement between the guided and unguided arms
(they share the initial noise and every sampler random draw), which is the dose
that was actually delivered rather than the one that was requested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from multimfm.geom import steering
from multimfm.geom_rescue import install_scalefree_guidance

_STATES: list = []


def main() -> None:
    install_scalefree_guidance()

    base_run_one = steering._run_one

    def _run_one(*a, **k):
        state, logs = base_run_one(*a, **k)
        _STATES.append((state["coords"].detach().clone(),
                        state["atomics"].detach().argmax(-1).clone(),
                        state["padding_mask"].detach().clone(), logs))
        return state, logs

    steering._run_one = _run_one

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    steering.add_common_arguments(p, value_sampler="mfm")
    p.add_argument("--rule-note", default="scalefree",
                   help="recorded in the JSON so the run cannot be mistaken for a stock run")
    args = p.parse_args()
    if args.guidance_max_coord_rms <= 0.0:
        print("WARNING: --guidance-max-coord-rms is 0, so the scale-free rule is a no-op "
              "and this run reproduces the stock configuration.", file=sys.stderr)
    result = steering.run_steering(args, value_sampler="mfm")

    # realised dose: the two arms share the initial noise and all sampler draws
    extra = {"guidance_rule": args.rule_note,
             "guidance_max_coord_rms": args.guidance_max_coord_rms,
             "guidance_max_atom_rms": args.guidance_max_atom_rms}
    if len(_STATES) >= 2:
        (cu, tu, pm, _), (cg, tg, _, logs) = _STATES[0], _STATES[1]
        real = (~pm).unsqueeze(-1).float()
        d = ((cg - cu) ** 2 * real).sum() / real.sum().clamp_min(1)
        extra["realised_coord_rms_displacement_normalized"] = float(d.sqrt())
        extra["realised_coord_rms_displacement_angstrom"] = float(d.sqrt()) * 2.0
        flips = ((tg != tu) & (~pm)).float().sum() / (~pm).float().sum().clamp_min(1)
        extra["realised_frac_atoms_retyped"] = float(flips)
        guided_rows = [r for r in logs if r.get("guided")]
        extra["n_guided_steps"] = len(guided_rows)
        if guided_rows:
            extra["grad_coords_rms_unclipped_mean"] = float(
                sum(r.get("grad_coords_rms_unclipped", r.get("grad_coords_rms", 0.0))
                    for r in guided_rows) / len(guided_rows))
            extra["grad_atomics_rms_mean"] = float(
                sum(r.get("grad_atomics_rms", 0.0) for r in guided_rows) / len(guided_rows))
    out = args.output_dir / f"mfm_fingerprint_steering_{args.tag}.json"
    rec = json.loads(out.read_text())
    rec.update(extra)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print("realised dose: "
          f"{extra.get('realised_coord_rms_displacement_angstrom', float('nan')):.4f} A RMS, "
          f"{extra.get('realised_frac_atoms_retyped', 0.0)*100:.2f}% of atoms retyped, "
          f"{extra.get('n_guided_steps', 0)} guided steps")


if __name__ == "__main__":
    main()
