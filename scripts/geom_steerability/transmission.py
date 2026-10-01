#!/usr/bin/env python
"""Probe A: direct (network-free) transmission of an injected perturbation.

Extends ``scripts/geom_rescue/interpolant_annihilation.py`` in three ways:

1. it uses the interpolant the released teacher actually samples with,
   ``SDEMetricInterpolant`` (the rescue script used the deterministic
   ``CenteredMetricInterpolant``), so the Langevin score term -- which is a
   *contraction toward the origin* and is active exactly inside the guided
   window t < 0.9 -- is included;
2. it reports the analytic per-step retention factor and its cumulative
   product, so the survival curve is a formula and not only 14 sampled rows;
3. it sweeps the *terminal* time of the schedule, which answers "what would
   survive if the sampler stopped short of t = 1?" -- the question a sampler
   fix has to answer.

Retention of the state (holding the network's endpoint prediction fixed):

  coords, deterministic:  1 - a,              a = dt/(1-t)
  coords, SDE:            1 - a - c,          c = s(t)*dt/(1-t)^2,  s = SampleNoiseSchedule
  atomics (Gaussian simplex, linear beta): 1 - a

With G = dx1_pred/dx_t = I the SDE factor becomes 1 - s(t)*dt/(1-t) instead.

Run: python scripts/geom_steerability/transmission.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tensordict import TensorDict

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "geom" / "steerability" / "transmission.json"

import sys
sys.path.insert(0, str(REPO / "src"))
from tabasco.flow.interpolate import (  # noqa: E402
    CenteredMetricInterpolant, GaussianSimplexAtomInterpolant, SDEMetricInterpolant,
)
from tabasco.sample.noise_schedule import SampleNoiseSchedule  # noqa: E402

NORMALIZER = 2.0  # coords are stored as Angstrom / 2


def log_schedule(n: int) -> torch.Tensor:
    T = 1.0 - torch.logspace(-2, 0, n + 1).flip(0)
    T = T - T.min()
    return T / T.max()


def analytic(T: torch.Tensor, sched: SampleNoiseSchedule):
    """Per-step retention factors and the cumulative survival to the endpoint."""
    n = len(T) - 1
    rows = []
    for i in range(n):
        t = float(T[i]); dt = float(T[i + 1] - T[i])
        a = dt / (1.0 - t) if t < 1.0 else 1.0
        s = float(sched(torch.tensor(t)))
        c = s * dt / (1.0 - t) ** 2 if t < 1.0 else 0.0
        rows.append({
            "step": i, "t": t, "dt": dt, "a": a, "s_of_t": s, "c": c,
            "ret_det": 1.0 - a,
            "ret_sde_stubG0": 1.0 - a - c,
            "ret_sde_GI": 1.0 - s * dt / (1.0 - t) if t < 1.0 else 0.0,
            "ret_atomics": 1.0 - a,
        })
    # cumulative survival of a perturbation injected INTO the state at t_{i}
    # (i.e. after step i-1) and carried to the end of the schedule
    for key in ("ret_det", "ret_sde_stubG0", "ret_sde_GI", "ret_atomics"):
        acc = 1.0
        for r in reversed(rows):
            r["surv_" + key] = acc  # survival for an injection into x_{t_i}
            acc = acc * r[key]
    return rows


def numeric_sde(num_steps=100, n_atoms=25, n_types=9, seed=0, injects=None):
    """The rescue script's measurement, redone with the production SDE interpolant."""
    torch.manual_seed(seed)
    T = log_schedule(num_steps)
    pad = torch.zeros(1, n_atoms, dtype=torch.bool)
    ci = SDEMetricInterpolant(
        key="coords", key_pad_mask="padding_mask",
        langevin_sampling_schedule=SampleNoiseSchedule(cutoff=0.9),
        white_noise_sampling_scale=0.01,
    )
    ai = GaussianSimplexAtomInterpolant(key="atomics", key_pad_mask="padding_mask")

    x1 = torch.randn(1, n_atoms, 3); x1 = x1 - x1.mean(1, keepdim=True)
    logits = torch.randn(1, n_atoms, n_types) * 3.0
    pred = TensorDict({"coords": x1, "atomics": logits}, batch_size=1)

    torch.manual_seed(seed + 77)
    pert_c = torch.randn(1, n_atoms, 3); pert_c = pert_c - pert_c.mean(1, keepdim=True)
    pert_c = pert_c / pert_c.square().mean().sqrt()
    pert_a = torch.randn(1, n_atoms, n_types)
    pert_a = pert_a / pert_a.square().mean().sqrt()

    if injects is None:
        injects = list(range(0, num_steps))
    rows = []
    for inject in injects:
        out = {}
        for eps in (0.0, 1.0):
            torch.manual_seed(1234)
            c0 = torch.randn(1, n_atoms, 3); c0 = c0 - c0.mean(1, keepdim=True)
            st = TensorDict({"coords": c0, "atomics": torch.randn(1, n_atoms, n_types),
                             "padding_mask": pad}, batch_size=1)
            for i in range(num_steps):
                t = T[i].expand(1); dt = (T[i + 1] - T[i]).expand(1)
                torch.manual_seed(90000 + i)          # same white noise in both arms
                nc = ci.step(st, pred, t, dt)
                na = ai.step(st, pred, t, dt)
                if i == inject:
                    nc = nc + eps * pert_c
                    na = na + eps * pert_a
                st = TensorDict({"coords": nc - nc.mean(1, keepdim=True),
                                 "atomics": na, "padding_mask": pad}, batch_size=1)
            out[eps] = (st["coords"].clone(), st["atomics"].clone())
        dc = (out[1.0][0] - out[0.0][0]).square().mean().sqrt().item()
        da = (out[1.0][1] - out[0.0][1]).square().mean().sqrt().item()
        rows.append({"inject_step": inject, "t_injected_into": float(T[inject + 1]),
                     "surviving_coord_rms": dc, "surviving_atomics_rms": da})
    return rows


def truncation_sweep(T: torch.Tensor, sched: SampleNoiseSchedule):
    """Survival to a schedule that STOPS at t_end < 1 (the sampler-fix question)."""
    rows = analytic(T, sched)
    out = []
    for t_end in (0.90, 0.95, 0.99, 0.999, 0.9995, 1.0):
        # cut the schedule at the last step whose t_{i+1} <= t_end
        cut = max(i for i, r in enumerate(rows) if r["t"] + r["dt"] <= t_end + 1e-12)
        sub = rows[: cut + 1]
        for key in ("ret_det", "ret_sde_stubG0", "ret_sde_GI", "ret_atomics"):
            acc = 1.0
            for r in reversed(sub):
                r["trunc_surv_" + key] = acc
                acc = acc * r[key]
        rec = {"t_end": float(sub[-1]["t"] + sub[-1]["dt"]), "n_steps": len(sub)}
        for probe_t in (0.10, 0.30, 0.50, 0.70, 0.85, 0.90):
            j = min(range(len(sub)), key=lambda k: abs(sub[k]["t"] - probe_t))
            rec[f"inject_t={probe_t}"] = {
                "t": sub[j]["t"],
                "det": sub[j]["trunc_surv_ret_det"],
                "sde_stubG0": sub[j]["trunc_surv_ret_sde_stubG0"],
                "sde_GI": sub[j]["trunc_surv_ret_sde_GI"],
                "atomics": sub[j]["trunc_surv_ret_atomics"],
            }
        out.append(rec)
    return out


def main() -> None:
    T = log_schedule(100)
    sched = SampleNoiseSchedule(cutoff=0.9)
    rows = analytic(T, sched)
    injects = [0, 5, 10, 20, 30, 40, 47, 50, 60, 75, 90, 95, 98, 99]
    num = numeric_sde(injects=injects)

    print("== analytic per-step retention, log schedule, 100 steps, production SDE ==")
    print(f"{'step':>4} {'t':>8} {'dt':>9} {'a':>7} {'s(t)':>7} {'c':>8} "
          f"{'ret_det':>8} {'ret_sde':>9} {'ret_sdeGI':>10} {'surv_det':>9} {'surv_sdeGI':>11}")
    for i in (0, 1, 3, 5, 10, 20, 30, 40, 47, 49, 50, 60, 75, 90, 98, 99):
        r = rows[i]
        print(f"{r['step']:>4} {r['t']:>8.5f} {r['dt']:>9.5f} {r['a']:>7.4f} "
              f"{r['s_of_t']:>7.3f} {r['c']:>8.4f} {r['ret_det']:>8.4f} "
              f"{r['ret_sde_stubG0']:>9.4f} {r['ret_sde_GI']:>10.4f} "
              f"{r['surv_ret_det']:>9.2e} {r['surv_ret_sde_GI']:>11.2e}")

    print("\n== numeric, production SDE interpolant, stub network (direct channel only) ==")
    print(f"{'inject after step':>17} {'t injected into':>16} {'coords survive':>15} {'atomics survive':>16}")
    for r in num:
        print(f"{r['inject_step']:>17} {r['t_injected_into']:>16.5f} "
              f"{r['surviving_coord_rms']:>15.3e} {r['surviving_atomics_rms']:>16.3e}")

    trunc = truncation_sweep(T, sched)
    print("\n== if the schedule STOPPED at t_end (survival of the DIRECT channel) ==")
    print(f"{'t_end':>8} {'steps':>6} " + " ".join(f"{'t='+str(p):>12}" for p in
          (0.10, 0.30, 0.50, 0.70, 0.85, 0.90)))
    for rec in trunc:
        cells = []
        for p in (0.10, 0.30, 0.50, 0.70, 0.85, 0.90):
            cells.append(f"{rec[f'inject_t={p}']['sde_GI']:>12.3e}")
        print(f"{rec['t_end']:>8.5f} {rec['n_steps']:>6} " + " ".join(cells))

    res = {
        "_what": __doc__.split("\n")[0],
        "schedule": "log", "num_steps": 100,
        "langevin": {"class": "SampleNoiseSchedule", "cutoff": 0.9,
                     "white_noise_sampling_scale": 0.01},
        "analytic_rows": rows,
        "numeric_sde_stub": num,
        "truncation_sweep": trunc,
        "guided_window_t_0.1_0.9": {
            "n_steps": sum(1 for r in rows if 0.1 <= r["t"] <= 0.9),
            "max_direct_surv_det": max(r["surv_ret_det"] for r in rows if 0.1 <= r["t"] <= 0.9),
            "max_direct_surv_sde_GI": max(r["surv_ret_sde_GI"] for r in rows if 0.1 <= r["t"] <= 0.9),
            "max_numeric_coords": max(r["surviving_coord_rms"] for r in num
                                      if 0.1 <= r["t_injected_into"] <= 0.9),
            "max_numeric_atomics": max(r["surviving_atomics_rms"] for r in num
                                       if 0.1 <= r["t_injected_into"] <= 0.9),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
