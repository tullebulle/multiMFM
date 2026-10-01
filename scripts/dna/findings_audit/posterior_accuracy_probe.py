#!/usr/bin/env python
"""Audit of the "a well-integrated posterior does not guide at all" finding (plan S7 item 9,
`docs/dmfm_glass_parity.md` section 2).

The published finding rests on one configuration, ``glass_euler32`` = GLASS with **32 Euler steps
to end_time 1.0**, run under the *stock* clip-down guidance rule. Two things about that
configuration are suspicious and are what this script tests:

1. ``end_time = 1.0``. The same document discards its own 100-Euler-to-1.0 reference as
   "numerically degenerate" because a fine Euler grid crosses the singular linear-DFM data-time
   velocity and drives the endpoint onto a simplex corner (its reference gradient norm is 8e-6).
   ``glass_euler32`` is in that family, so it may be a *degenerate* posterior rather than an
   *accurate* one. The accurate integrator used as the Table-18 reference is RK4-25 to 0.999.
2. The stock clip-down rule never rescales a gradient that is *below* ``grad_clip``, so a
   posterior whose gradient is 20-5000x smaller gets a 20-5000x smaller step. The scale-free rule
   (``@norm``) that S7 item 8 mandates for every dMFM/GLASS comparison was **not** applied to
   ``glass_euler32``.

This wraps :mod:`dmfm.experiments.parity_dmfm_glass_c0` without modifying it: extra entries are
injected into its ``CONFIGS`` dict and ``Bank.posterior`` is patched to honour a ``solver`` key, so
every number stays comparable to ``results/dna/parity/``.

    python scripts/dna/findings_audit/posterior_accuracy_probe.py --mode grad --n_probes 4 \
        --probe_times 0.5 0.7 --reference_steps 25 --reference_solver rk4 \
        --reference_end_time 0.999 --out_dir results/dna/findings_audit/grad_posterior_accuracy
    python scripts/dna/findings_audit/posterior_accuracy_probe.py --mode sample --n_samples 32 \
        --configs glass_euler4 glass_euler32 glass_euler32@norm glass_euler32_e999 \
                  glass_euler32_e999@norm glass_rk4_25_e999@norm
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from dmfm import api  # noqa: E402
from dmfm.experiments import parity_dmfm_glass_c0 as P  # noqa: E402

EXTRA = {
    # GLASS Euler ladder to end_time 1.0: how the posterior degenerates as the grid is refined.
    "glass_euler8": dict(kind="glass", student=None, n_steps=8, end_time=1.0, solver="euler",
                         note="GLASS 8 Euler steps to 1.0"),
    "glass_euler16": dict(kind="glass", student=None, n_steps=16, end_time=1.0, solver="euler",
                          note="GLASS 16 Euler steps to 1.0"),
    "glass_euler100": dict(kind="glass", student=None, n_steps=100, end_time=1.0, solver="euler",
                           note="GLASS 100 Euler steps to 1.0 -- the reference the parity doc "
                                "discarded as numerically degenerate"),
    # The same step counts, stopped just short of the singular data time.
    "glass_euler4_e999": dict(kind="glass", student=None, n_steps=4, end_time=0.999, solver="euler",
                              note="Table 1 posterior, end_time 0.999"),
    "glass_euler32_e999": dict(kind="glass", student=None, n_steps=32, end_time=0.999, solver="euler",
                               note="32 Euler steps to 0.999 -- an ACCURATE fine-grid posterior"),
    "glass_euler100_e999": dict(kind="glass", student=None, n_steps=100, end_time=0.999,
                                solver="euler", note="100 Euler steps to 0.999"),
    # The Table-18 reference integrator used as an *estimator*: the most accurate posterior we have.
    "glass_rk4_25_e999": dict(kind="glass", student=None, n_steps=25, end_time=0.999, solver="rk4",
                              note="GLASS RK4 25 steps to 0.999 -- the well-integrated reference "
                                   "posterior, here used to guide"),
    "glass_rk4_8_e999": dict(kind="glass", student=None, n_steps=8, end_time=0.999, solver="rk4",
                             note="GLASS RK4 8 steps to 0.999"),
}


def _posterior(self, name: str):
    spec = P.CONFIGS[name]
    if spec["kind"] == "glass":
        return api.glass_posterior_fn(self.base, n_steps=spec["n_steps"],
                                      end_time=spec["end_time"],
                                      solver=spec.get("solver", "euler"))
    return api.dmfm_posterior_fn(self.student(spec["student"]), sampler="flow_map",
                                n_steps=spec["n_steps"], end_time=spec["end_time"])


def _no_tf32(out) -> None:
    """Force true fp32 for every numerical comparison here.

    ``parity_dmfm_glass_c0.main`` enables TF32 (that is how the stored parity results were
    produced), but TF32 has already produced garbage once in this round -- the GEOM
    finite-difference slopes -- and Table 18's reference gradient norm moves 0.0125 vs 0.0168
    with TF32 on/off. This audit compares gradient norms of ~0.15 against 3-125, so the
    conclusion must not rest on 10-bit mantissas. Set ``FA_TF32=1`` to keep the parity module's
    TF32 setting instead.
    """
    if os.environ.get("FA_TF32", "0") == "1":
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    meta = Path(out) / "run_metadata.json"
    if meta.exists():  # main() wrote run_env before we turned TF32 off; keep the record honest
        d = json.loads(meta.read_text())
        d["run_env"]["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        d["run_env"]["tf32_forced_off_by_findings_audit"] = True
        d["run_env"]["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        meta.write_text(json.dumps(d, indent=2, sort_keys=True))


def main(argv=None) -> None:
    P.CONFIGS.update(EXTRA)
    P.Bank.posterior = _posterior
    for name in ("run_grad", "run_sample"):
        orig = getattr(P, name)

        def wrapped(args, out, device, _orig=orig):
            _no_tf32(out)
            print(f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
                  f"float32_matmul_precision={torch.get_float32_matmul_precision()}", flush=True)
            return _orig(args, out, device)

        setattr(P, name, wrapped)
    P.main(argv)


if __name__ == "__main__":
    main()
