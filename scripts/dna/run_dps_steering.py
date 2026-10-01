#!/usr/bin/env python
"""Run the DNA DPS steering baseline and write shards in the benchmark's own schema.

    python scripts/dna/run_dps_steering.py --n-steps 96 --guide-t-start 0.479 \
        --guidance-frac 8 --n-outputs 300 --repeat 0 --out-dir outputs/dna/fig4/dps/b192

DPS has no sampling knob, so its budget axis is the number of integration steps and the
guidance window -- the same two knobs the QM9 sweep gives it. Cost per returned sample is
``n_steps + G + 1.1 G`` with ``G`` guided steps, which places it on the same ladder as the
other methods.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--n-steps", type=int, default=96)
    p.add_argument("--n-outputs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-c0", type=float, default=0.30)
    p.add_argument("--reward-sigma", type=float, default=0.15)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--guidance-frac", type=float, default=8.0)
    p.add_argument("--coeff-cap", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--guide-t-start", type=float, default=0.01)
    p.add_argument("--guide-t-end", type=float, default=0.95)
    p.add_argument("--use-student", action="store_true",
                   help="score the dMFM student's denoiser instead of the base model's")
    p.add_argument("--label", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    from dmfm.benchmarks import core
    from dmfm.benchmarks.dmfm_steering import GuidanceConfig
    from dmfm.benchmarks.dps_steering import sample_dps_guided
    from dmfm.benchmarks.search import RewardSpec

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = a.seed + 1000 * 900 + a.repeat          # own stream, disjoint from the plan's shards
    torch.manual_seed(seed)

    nfe = core.NFECounter()
    flow = core.load_base_flow(a.length, dev, nfe=nfe)
    guide = core.load_c0("guide", dev, nfe)
    oracle = core.load_c0("oracle", dev, nfe, rc_average=True)
    student = core.load_dmfm_student(a.length, dev)[0] if a.use_student else None

    sampler = core.SamplerConfig(n_steps=a.n_steps)
    guidance = GuidanceConfig(t_start=a.guide_t_start, t_end=a.guide_t_end,
                              guidance_frac=a.guidance_frac, coeff_cap=a.coeff_cap,
                              grad_clip=a.grad_clip)
    reward = RewardSpec(target=a.target_c0, sigma=a.reward_sigma, scale=a.reward_scale)

    n_before = dict(base=nfe.base, student=nfe.student, guide=nfe.guide, backward=nfe.backward)
    out = sample_dps_guided(flow, student, guide, reward, n_outputs=a.n_outputs,
                            sampler=sampler, guidance=guidance,
                            batch_size=a.batch_size, nfe=nfe)
    used = {k: getattr(nfe, k) - v for k, v in n_before.items()}

    scores = out["score"].detach().float().cpu()
    with torch.no_grad():
        x = torch.nn.functional.one_hot(out["tokens"].to(dev), flow.cfg.alphabet_size).float()
        orc = oracle.model(x).detach().float().cpu()
    metrics = core.summarize_errors(scores, a.target_c0)
    metrics.update({f"oracle_{k}": v for k, v in core.summarize_errors(orc, a.target_c0).items()})

    gen = (used["base"] + used["student"]) / a.n_outputs
    bwd = used["backward"] / a.n_outputs
    label = a.label or f"steps={a.n_steps},t0={a.guide_t_start},gf={a.guidance_frac}"
    shard = {
        "method": "dps", "config": label, "n_outputs": a.n_outputs,
        "repeat": a.repeat, "seed": seed, "metrics": metrics,
        "nfe": {"nfe_gen_per_output": gen, "nfe_backward_per_output": bwd,
                "nfe_guide_per_output": used["guide"] / a.n_outputs,
                "nfe_fwd_equiv_per_output": gen + 1.1 * bwd},
        "config_params": vars(a),
    }
    d = Path(a.out_dir); d.mkdir(parents=True, exist_ok=True)
    f = d / f"dps__{label.replace(',', '_').replace('=', '')}__n{a.n_outputs}_s{a.seed}__rep{a.repeat}.json"
    f.write_text(json.dumps(shard, indent=1) + "\n")
    print(f"{label}: mae={metrics['mae']:.4f} oracle_mae={metrics['oracle_mae']:.4f} "
          f"gen={gen:.0f} bwd={bwd:.0f} fwd_equiv={gen + 1.1 * bwd:.0f} -> {f}")


if __name__ == "__main__":
    main()
