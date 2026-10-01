#!/usr/bin/env python
"""End-to-end sign test of the guidance chain, with no checkpoints and no GPU.

The question `docs/REPRODUCIBILITY_PLAN.md` §7 item 1 asks -- is the guidance term
mis-signed? -- factors into two parts. `multimfm.geom.diagnostics fd-check` checks the
reward gradient and the full Monte-Carlo chain numerically, but it needs the teacher,
the student and the head, so it needs a compute allocation. This script checks the
*sign convention* of the same code path in a couple of seconds on any machine, by
replacing the two learned pieces with stubs whose exact analytic answer is known:

* the look-ahead posterior is stubbed to the identity, ``x_1 = x_t``, so
  ``V_t(x) = log mean_i exp(lambda r(x)) = lambda r(x)`` exactly;
* the reward is ``r(x) = x[0, 0, 0]`` (``reward_name="first_atom_x"``), so
  ``grad_x V_t = lambda * e_{0,0,0}`` exactly;
* the base model is stubbed to zero velocity, so the *only* thing that moves the
  state is the guidance term.

Then ``guided_sample`` must return ``coords[0, 0, 0] = mu * lambda * sum(dt)`` over the
guided steps, with a **positive** sign. Every line that a real run executes between
``torch.autograd.grad`` and the returned molecule is exercised: ``value_gradient``'s
log-mean-exp and ``autograd.grad``, ``clip_guidance_rms``, the mu schedule wrapper, and
``production_step_with_optional_guidance``.

    python scripts/geom/sign_chain_selfcheck.py

Prints a table and exits non-zero if any sign or magnitude is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict

from multimfm import glass_guidance as engine
from multimfm.geom.mu_schedule import install_mu_schedule, paper_coefficient

B, N, A = 2, 3, 9


class ZeroVelocityModel:
    """Base generator with zero velocity: the guidance term is the only motion."""

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.coords_interpolant = self
        self.atomics_interpolant = self

    def _get_sample_schedule(self, num_steps: int) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, num_steps + 1)

    def _call_net(self, state, t):
        return state

    def step(self, batch_t, pred, t, dt):
        # identity: x_{t+dt} = x_t
        key = "coords" if batch_t["coords"].shape[-1] == 3 else "atomics"
        return batch_t[key]


def _identity_posterior(student, cond_state, t_cond, *, posterior_samples, n_steps,
                        diagonal, prior_noise=None):
    """x_1 = x_t, repeated `posterior_samples` times (differentiably)."""
    return engine.repeat_state(cond_state, posterior_samples)


def run(mu: float, lam: float, num_steps: int, guide_every: int,
        mu_schedule: str) -> tuple[float, float]:
    """Return (observed coords[0,0,0], analytic expectation)."""
    install_mu_schedule(mu_schedule)
    real_posterior = engine.differentiable_mfm_posterior
    # `guided_sample` and `value_gradient` look the name up in the module globals at
    # call time, so rebinding here is exactly what a real run would execute.
    engine.differentiable_mfm_posterior = _identity_posterior
    try:
        model = ZeroVelocityModel(num_steps)
        init = TensorDict(
            {"coords": torch.zeros(B, N, 3), "atomics": torch.zeros(B, N, A),
             "padding_mask": torch.zeros(B, N, dtype=torch.bool)},
            batch_size=B).to(torch.device("cpu"))
        final, logs = engine.guided_sample(
            model, init, value_sampler="mfm", mfm_student=object(),
            mfm_diagonal=False, num_steps=num_steps, mu=mu, reward_scale=lam,
            value_samples=4, value_glass_steps=2, reward_name="first_atom_x",
            target_x=0.0, property_name="structure", property_target=0.0,
            property_regressor=None, guide_atomics=True,
            guide_min_t=0.1, guide_max_t=0.9, guide_every=guide_every,
            guidance_max_coord_rms=0.0, guidance_max_atom_rms=0.0, eps=1e-6,
            value_batch_size=0,
        )
    finally:
        engine.differentiable_mfm_posterior = real_posterior
        install_mu_schedule("constant")

    # Analytic expectation: sum over the guided steps of mu * coeff(t) * lambda * dt,
    # minus the zero-COM projection, which removes the per-molecule mean of the
    # gradient over the N atoms: e_{0,0,0} -> (1 - 1/N) on atom 0.
    dt = 1.0 / num_steps
    expect = 0.0
    for row in logs:
        if row["guided"]:
            coeff = (float(paper_coefficient(torch.tensor(row["t"])))
                     if mu_schedule == "paper" else 1.0)
            expect += mu * coeff * lam * dt * (1.0 - 1.0 / N)
    return float(final["coords"][0, 0, 0]), expect


def main() -> int:
    rows = []
    for mu_schedule in ("constant", "paper"):
        for mu in (0.25, 1.0, -0.25):
            for lam in (0.01, 1.0):
                got, want = run(mu, lam, num_steps=100, guide_every=2,
                                mu_schedule=mu_schedule)
                ok = abs(got - want) <= 1e-9 + 1e-5 * abs(want)
                rows.append((mu_schedule, mu, lam, got, want, ok))

    hdr = (f"{'mu sched':>10}{'mu':>7}{'lambda':>8}{'observed x[0,0,0]':>20}"
           f"{'analytic':>16}{'ok':>5}")
    print(hdr)
    print("-" * len(hdr))
    for sched, mu, lam, got, want, ok in rows:
        print(f"{sched:>10}{mu:>7.3g}{lam:>8.3g}{got:>20.10g}{want:>16.10g}"
              f"{'YES' if ok else 'NO':>5}")
    bad = [r for r in rows if not r[5]]
    if bad:
        print(f"\nFAIL: {len(bad)} of {len(rows)} cases disagree with the analytic value.")
        return 1
    print(f"\nPASS: all {len(rows)} cases match the analytic value, including the sign.")
    print("  Positive mu moves the state in the +grad direction, i.e. the guidance term")
    print("  ASCENDS the log-reward. Negative mu descends it, by the same magnitude.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
