#!/usr/bin/env python
"""Where, in the sampler the recorded runs actually used, does guidance act?

`docs/geom_guidance_diagnosis.md` §4 estimates the guidance dose as "guidance on
every 2nd of 100 steps inside t in [0.1, 0.9], so 40 guided steps ... dt = 0.01".
Both of those are properties of a *uniform* schedule. The released GEOM teacher's
config (`checkpoints/geom/teacher/hydra/config.yaml`) sets

    sample_schedule: log
    coords_interpolant: SDEMetricInterpolant
      langevin_sampling_schedule: SampleNoiseSchedule(cutoff=0.9)
      white_noise_sampling_scale: 0.01

so (a) the time grid is logarithmic, with dt varying by an order of magnitude
across the guided window, and (b) the coordinate sampler is a Langevin SDE whose
score term and injected white noise are switched on for t < 0.9 and off for
t >= 0.9 -- i.e. **exactly inside the guided window and nowhere else**.

This script reports the step census, the per-step guidance displacement against
the sampler's own injected noise, and how much of the trajectory is left
unguided, using the measured gradient RMS from
`results/geom/diagnostics/fd_check_t0.5_lam0.01_cuda.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "geom" / "rescue" / "schedule_census.json"


def log_schedule(num_steps: int) -> np.ndarray:
    T = 1.0 - np.logspace(-2, 0, num_steps + 1)[::-1]
    T = T - T.min()
    return T / T.max()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--grad-rms", type=float, default=2.794,
                   help="measured coords value-gradient RMS through the flow map at t=0.5")
    p.add_argument("--mu", type=float, default=0.25)
    p.add_argument("--state-rms", type=float, default=0.2637)
    p.add_argument("--normalizer", type=float, default=2.0)
    p.add_argument("--white-noise-scale", type=float, default=0.01)
    p.add_argument("--cutoff", type=float, default=0.9)
    p.add_argument("--windows", nargs="+", default=["0.1,0.9", "0.7,1.0", "0.85,1.0", "0.1,0.5"])
    p.add_argument("--guide-every", type=int, nargs="+", default=[1, 2])
    a = p.parse_args()

    for name, T in (("log", log_schedule(a.num_steps)),
                    ("linear", np.linspace(0, 1, a.num_steps + 1))):
        if name == "log":
            T_log = T
    T = T_log
    t = T[:-1]
    dt = np.diff(T)
    g = np.where(t < a.cutoff, 1.0 / (t + 1e-2), 0.0)          # SampleNoiseSchedule
    noise_rms = np.sqrt(2 * g * a.white_noise_scale) * dt      # per-step, normalized units
    guide_rms = a.mu * a.grad_rms * dt

    out = {"_what": __doc__.split("\n")[0], "schedule": "log", "num_steps": a.num_steps,
           "mu": a.mu, "grad_rms": a.grad_rms, "state_coords_rms": a.state_rms,
           "t_of_step": {str(i): float(t[i]) for i in
                         (0, 1, 2, 3, 5, 10, 20, 30, 47, 49, 50, 60, 75, 90, 99)},
           "frac_steps_with_t_above_0.9": float((t >= 0.9).mean()),
           "windows": {}}

    for w in a.windows:
        lo, hi = (float(x) for x in w.split(","))
        for every in a.guide_every:
            sel = (t >= lo) & (t <= hi) & (np.arange(len(t)) % every == 0)
            tot_guide = float(guide_rms[sel].sum())
            tot_noise_coh = float(noise_rms[sel].sum())
            tot_noise_rw = float(np.sqrt((noise_rms[sel] ** 2).sum()))
            out["windows"][f"t in [{lo},{hi}] every {every}"] = {
                "n_guided_steps": int(sel.sum()),
                "dt_min": float(dt[sel].min()) if sel.any() else None,
                "dt_max": float(dt[sel].max()) if sel.any() else None,
                "guide_disp_per_step_A": {
                    "min": float(guide_rms[sel].min() * a.normalizer) if sel.any() else None,
                    "max": float(guide_rms[sel].max() * a.normalizer) if sel.any() else None},
                "guide_disp_coherent_A": tot_guide * a.normalizer,
                "guide_disp_randomwalk_A": float(
                    np.sqrt((guide_rms[sel] ** 2).sum()) * a.normalizer),
                "sampler_noise_coherent_A": tot_noise_coh * a.normalizer,
                "sampler_noise_randomwalk_A": tot_noise_rw * a.normalizer,
                "guide_over_noise_per_step_median": (
                    None if not sel.any() or not (noise_rms[sel] > 0).any()
                    else float(np.median(guide_rms[sel][noise_rms[sel] > 0]
                                         / noise_rms[sel][noise_rms[sel] > 0]))),
                "frac_guided_steps_with_injected_noise": float(
                    (noise_rms[sel] > 0).mean()) if sel.any() else None,
                "unguided_dt_after_window": float(dt[t > hi].sum()),
                "unguided_steps_after_window": int((t > hi).sum()),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"log schedule, {a.num_steps} steps: "
          f"{out['frac_steps_with_t_above_0.9']*100:.0f}% of steps have t >= 0.9")
    print("  t at step index:", {k: round(v, 4) for k, v in out["t_of_step"].items()})
    for k, v in out["windows"].items():
        print(f"\n{k}: {v['n_guided_steps']} guided steps, dt {v['dt_min']:.2e}..{v['dt_max']:.2e}")
        print(f"   guidance displacement  coherent {v['guide_disp_coherent_A']:.3f} A, "
              f"random-walk {v['guide_disp_randomwalk_A']:.3f} A "
              f"(per step {v['guide_disp_per_step_A']['min']:.4f}..{v['guide_disp_per_step_A']['max']:.4f} A)")
        gn = v["guide_over_noise_per_step_median"]
        print(f"   sampler's own injected noise  random-walk {v['sampler_noise_randomwalk_A']:.3f} A "
              f"on {v['frac_guided_steps_with_injected_noise']*100:.0f}% of the guided steps; "
              f"guidance/noise per step (median) "
              + ("n/a (deterministic: t >= 0.9 cutoff)" if gn is None else f"{gn:.2f}"))
        print(f"   left unguided after the window: {v['unguided_steps_after_window']} steps, "
              f"dt sum {v['unguided_dt_after_window']:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
