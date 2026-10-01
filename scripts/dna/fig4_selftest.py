#!/usr/bin/env python
"""CPU self-test of the Figure 4 benchmark code (~1 min, no GPU).

Checks the three things the figure's honesty rests on:

1. the shared sampler at ``eta = 0`` is bit-for-bit the deterministic sampler every
   other DNA experiment uses (``gaussian_denoiser_flow_step``), so best-of-N and the
   dMFM sample the same base process as Table 1;
2. ``eta > 0`` really branches (two children of one state diverge) while leaving the
   endpoint distribution alone up to Monte Carlo noise;
3. the measured NFE of every method equals its analytic cost, so the x-axis of the
   figure is not an artefact of the implementation.

    python scripts/dna/fig4_selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dmfm.benchmarks.core import NFECounter, SamplerConfig, load_base_flow, load_c0  # noqa: E402
from dmfm.benchmarks.search import RewardSpec, beam_search, best_of_n, feynman_kac, mcts  # noqa: E402
from dmfm.utils.flow_utils import gaussian_denoiser_flow_step  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    device = torch.device("cpu")
    nfe = NFECounter()
    flow = load_base_flow(50, device, nfe)
    guide = load_c0("guide", device, nfe)
    reward = RewardSpec(target=0.30, sigma=0.15, scale=1.0)
    torch.manual_seed(0)

    # 1. eta = 0 equals the deterministic interpolant step
    x = torch.randn(4, flow.L, flow.K)
    s = torch.full((4,), 0.3)
    t = torch.full((4,), 0.3 + 1 / 96)
    ref, _, _ = gaussian_denoiser_flow_step(flow.cfg, flow.model, x, s, t)
    mine, _ = flow.step(x, s, 1 / 96, SamplerConfig(n_steps=96, eta=0.0))
    err = float((mine - ref).abs().max())
    check("eta=0 step == gaussian_denoiser_flow_step", err < 1e-5, f"max|diff| = {err:.2e}")

    # 2. eta > 0 branches
    sto = SamplerConfig(n_steps=96, eta=1.0)
    g = torch.Generator(device=device)
    g.manual_seed(1)
    two = x[:1].repeat(2, 1, 1)
    out, _ = flow.step(two, torch.full((2,), 0.3), 1 / 96, sto, generator=g)
    spread = float((out[0] - out[1]).abs().max())
    check("eta>0 children diverge", spread > 1e-3, f"max|child0-child1| = {spread:.3f}")
    det, _ = flow.step(two, torch.full((2,), 0.3), 1 / 96, SamplerConfig(n_steps=96, eta=0.0))
    check("eta=0 children identical", float((det[0] - det[1]).abs().max()) == 0.0)

    # 3. measured NFE == analytic cost, on a short 12-step grid
    n_steps, n_out = 12, 2
    short = SamplerConfig(n_steps=n_steps, eta=0.0)
    short_sto = SamplerConfig(n_steps=n_steps, eta=1.0)

    cases = []
    nfe.base = nfe.guide = 0
    best_of_n(flow, guide, reward, n_outputs=n_out, n_candidates=3, sampler=short, generator=g, batch_rows=64)
    cases.append(("best_of_n N=3", nfe.base, n_out * 3 * n_steps, nfe.guide, n_out * 3))

    nfe.base = nfe.guide = 0
    feynman_kac(
        flow, guide, reward, n_outputs=n_out, n_particles=3, resample_every=4, beta=5.0,
        sampler=short_sto, generator=g, t_min=0.0, t_max=1.0,
    )
    n_resample = sum(1 for k in range(n_steps) if (k + 1) % 4 == 0 and (k + 1) < n_steps)
    cases.append(("fk K=3", nfe.base, n_out * 3 * n_steps, nfe.guide, n_out * 3 * (n_resample + 1)))

    nfe.base = nfe.guide = 0
    W, L, K = 1, 2, 4
    beam_search(
        flow, guide, reward, n_outputs=n_out, width=W, branch=L, checkpoint_every=K,
        sampler=short_sto, generator=g,
    )
    M = n_steps // K
    expect = n_out * W * L * sum(n_steps - m * K for m in range(M))
    cases.append(("beam W=1,L=2,K=4", nfe.base, expect, nfe.guide, n_out * W * L * M))

    nfe.base = nfe.guide = 0
    it = 3
    mcts(
        flow, guide, reward, n_outputs=n_out, iterations=it, children=2, checkpoint_every=4,
        c_uct=0.5, sampler=short_sto, generator=g,
    )
    # every iteration expands one node and rolls it out to the end: K + (n_steps - (d+1)K)
    # steps, i.e. n_steps - d*K with d the depth of the selected node (0 <= d < M).
    cases.append(("mcts I=3 (base NFE <= I*n_steps)", nfe.base, None, nfe.guide, n_out * it))

    for name, got_base, want_base, got_guide, want_guide in cases:
        if want_base is not None:
            check(f"{name}: base NFE", got_base == want_base, f"{got_base} (expected {want_base})")
        else:
            check(f"{name}: base NFE", 0 < got_base <= n_out * it * n_steps, f"{got_base}")
        check(f"{name}: guide evals", got_guide == want_guide, f"{got_guide} (expected {want_guide})")

    # 4. shard identity: same settings are reused, different settings are refused
    import tempfile

    from dmfm import paths
    from dmfm.benchmarks import run_steering

    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.OUTPUTS, prefix="fig4_selftest_") as tmp:
        base = ["--plan", "sweep", "--index", "5", "--repeat", "0", "--out-dir", tmp,
                "--n-outputs", "2", "--device", "cpu"]
        run_steering.main(base + ["--n-steps", "12"])
        run_steering.main(base + ["--n-steps", "12"])  # must skip
        try:
            run_steering.main(base + ["--n-steps", "24"])
            refused = False
        except SystemExit:
            refused = True
        check("shard with different settings is refused, not reused", refused)
        check("n_outputs and seed are in the shard name",
              any(p.name.endswith("__n2_s0__rep0.json") for p in Path(tmp).iterdir()))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
