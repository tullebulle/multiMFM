#!/usr/bin/env python
"""Measure, with no model at all, that the sampler erases guided perturbations at
the final step -- and that for the atom-type channel this is total.

Both production interpolants are contractions toward the network's endpoint
prediction:

  coords  (CenteredMetricInterpolant / SDEMetricInterpolant):
          x_next = x_t + [(x1_pred - x_t)/(1 - t)] * dt
  atomics (GaussianSimplexAtomInterpolant):
          x_next = x_t + [dbeta/(1 - beta_t)] * (psi - x_t)

On any schedule that reaches t = 1, the last step has dt = 1 - t and
dbeta = 1 - beta_t, so both coefficients are exactly 1 and the state is
*overwritten*: x_final = x1_pred, atomics_final = psi. A perturbation injected at
any earlier step therefore survives only through the network's response to it.

This script replaces the network by a fixed stub (so there is no response at all)
and measures the surviving fraction of a perturbation injected at each t. The
result is the *direct* transmission of the guidance term, which is what the
atom-type argmax depends on.

Run: python scripts/geom_rescue/interpolant_annihilation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "geom" / "rescue" / "interpolant_annihilation.json"

import sys
sys.path.insert(0, str(REPO / "src"))
from tabasco.flow.interpolate import (  # noqa: E402
    CenteredMetricInterpolant, GaussianSimplexAtomInterpolant,
)


def log_schedule(n: int) -> torch.Tensor:
    T = 1.0 - torch.logspace(-2, 0, n + 1).flip(0)
    T = T - T.min()
    return T / T.max()


def run(num_steps=100, n_atoms=25, n_types=9, seed=0):
    torch.manual_seed(seed)
    T = log_schedule(num_steps)
    pad = torch.zeros(1, n_atoms, dtype=torch.bool)
    ci = CenteredMetricInterpolant(key="coords", key_pad_mask="padding_mask")
    ai = GaussianSimplexAtomInterpolant(key="atomics", key_pad_mask="padding_mask")

    # a fixed stub "network": constant endpoint prediction, so the only way a
    # perturbation can reach the final state is by direct transmission.
    x1 = torch.randn(1, n_atoms, 3)
    x1 = x1 - x1.mean(1, keepdim=True)
    logits = torch.randn(1, n_atoms, n_types) * 3.0
    pred = TensorDict({"coords": x1, "atomics": logits}, batch_size=1)

    # a fixed random perturbation direction (zero-COM in coords, so it is not
    # removed by the sampler's re-centring, and differential in atomics, so it can
    # in principle move an argmax)
    torch.manual_seed(seed + 77)
    pert_c = torch.randn(1, n_atoms, 3)
    pert_c = pert_c - pert_c.mean(1, keepdim=True)
    pert_a = torch.randn(1, n_atoms, n_types)

    rows = []
    for inject in range(0, num_steps):
        out = {}
        for eps in (0.0, 1.0):
            st = TensorDict({
                "coords": torch.zeros(1, n_atoms, 3),
                "atomics": torch.randn(1, n_atoms, n_types) * 0.0,
                "padding_mask": pad,
            }, batch_size=1)
            torch.manual_seed(1234)
            st["coords"] = torch.randn(1, n_atoms, 3)
            st["coords"] = st["coords"] - st["coords"].mean(1, keepdim=True)
            st["atomics"] = torch.randn(1, n_atoms, n_types)
            for i in range(num_steps):
                t = T[i].expand(1)
                dt = (T[i + 1] - T[i]).expand(1)
                nc = ci.step(st, pred, t, dt)
                na = ai.step(st, pred, t, dt)
                if i == inject:
                    nc = nc + eps * pert_c
                    na = na + eps * pert_a
                st = TensorDict({"coords": nc - nc.mean(1, keepdim=True),
                                 "atomics": na, "padding_mask": pad}, batch_size=1)
            out[eps] = (st["coords"].clone(), st["atomics"].clone())
        dc = (out[1.0][0] - out[0.0][0]).abs().mean().item()
        da = (out[1.0][1] - out[0.0][1]).abs().mean().item()
        rows.append({"inject_step": inject, "t": float(T[inject]),
                     "surviving_coord_perturbation": dc,
                     "surviving_atomics_perturbation": da})
    return rows, float(T[-2])


def main() -> None:
    rows, t_last = run()
    keep = [r for r in rows if r["inject_step"] in
            (0, 5, 10, 20, 30, 40, 47, 50, 60, 75, 90, 95, 98, 99)]
    print(f"log schedule, 100 steps, unit perturbation injected after one step, "
          f"stub network (no response), last step t = {t_last:.5f}")
    print(f"{'step':>5} {'t':>8} {'coords survive':>16} {'atomics survive':>17}")
    for r in keep:
        print(f"{r['inject_step']:>5} {r['t']:>8.5f} "
              f"{r['surviving_coord_perturbation']:>16.3e} "
              f"{r['surviving_atomics_perturbation']:>17.3e}")
    guided = [r for r in rows if 0.1 <= r["t"] <= 0.9]
    res = {
        "_what": __doc__.split("\n")[0],
        "schedule": "log", "num_steps": 100,
        "note": "direct transmission only: the stub network cannot respond to the "
                "perturbation, so these are the guidance term's own survival factors.",
        "rows": rows,
        "recorded_window_t_0.1_0.9": {
            "n_steps": len(guided),
            "max_surviving_coords": max(r["surviving_coord_perturbation"] for r in guided),
            "max_surviving_atomics": max(r["surviving_atomics_perturbation"] for r in guided),
        },
        "final_step": rows[-1],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\ninside the recorded guided window t in [0.1, 0.9] ({len(guided)} steps): "
          f"max surviving coords {res['recorded_window_t_0.1_0.9']['max_surviving_coords']:.2e}, "
          f"max surviving atomics {res['recorded_window_t_0.1_0.9']['max_surviving_atomics']:.2e}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
