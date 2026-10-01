#!/usr/bin/env python
"""QM9 value-function estimation error against the number of posterior samples.

    python scripts/qm9/value_error.py --n-conditions 32 --out results/qm9/value_error.csv

The QM9 analogue of the DNA posterior-recovery benchmark. For conditioning states x_t
drawn by integrating the base model to t ~ U(0.05, 0.35), it measures how well each
estimator recovers

    V_t(x) = log E[ exp(r(X_1)) | X_t = x ],

with r the property reward used for steering. Estimators:

* **multiMFM** -- log-mean-exp over N posterior samples from the flow map (one map
  evaluation each), so its cost grows with N;
* **DPS** -- the denoiser approximation r(E[X_1 | X_t = x]), one network call and
  independent of N;
* **FMAP** -- the base ODE integrated deterministically from t to 1, scored at that
  endpoint: a better point estimate than the one-step denoiser, and a best case for any
  deterministic one, but still independent of N;
* **reference** -- the same log-mean-exp over ``--n-ref`` GLASS posterior samples on the
  base model, i.e. the exact posterior sampler at a budget none of the estimators get.

The reference is itself a finite-sample estimate, so no curve can be resolved below its
own Monte-Carlo spread; that spread is reported as ``ref_selfsplit`` (two disjoint halves
of the reference pool scored against each other) and should be drawn with the curves.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mfm-checkpoint", default=None)
    p.add_argument("--property-name", default="alpha")
    p.add_argument("--n-conditions", type=int, default=32)
    p.add_argument("--n-ref", type=int, default=512)
    p.add_argument("--mcs", default="1,2,4,8,16,32,64,128,256")
    p.add_argument("--sample-steps", type=int, default=128)
    p.add_argument("--reward-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-seed", type=int, default=2027)
    p.add_argument("--chunk", type=int, default=64, help="posterior samples per forward")
    p.add_argument("--out", default="results/qm9/value_error.csv")
    # the histogram loader shares these with the samplers; keep the same defaults
    p.add_argument("--partition", default="train_flow")
    p.add_argument("--hist-bins", type=int, default=100)
    p.add_argument("--hist-cache", default=None)
    p.add_argument("--force-rebuild-hist", action="store_true")
    a = p.parse_args()

    from multimfm import paths
    from multimfm.glass_guidance import (differentiable_joint_glass_posterior,
                                         differentiable_mfm_posterior, grouped_rewards,
                                         load_flow_model, load_mfm_student,
                                         production_step_with_optional_guidance)
    from multimfm.glass_posterior import endpoint_prediction
    from multimfm.property_targets import (build_or_load_histograms, resolve_hist_cache,
                                            sample_property_target)
    from multimfm.qm9_data import choose_device, load_tfg_regressor

    dev = choose_device("cuda")
    ck = a.checkpoint or str(paths.BASE_FLOW_CKPT)
    mck = a.mfm_checkpoint or str(paths.MFM_STUDENT_CKPT)
    model, stats, model_args = load_flow_model(ck, dev)
    student = load_mfm_student(Path(mck), data_stats=stats, teacher_args=model_args,
                               device=dev, key="student")
    reg = load_tfg_regressor("guide", a.property_name, dev)
    reg.soft_atom_temperature = 0.25
    reg.eval()
    for prm in reg.parameters():
        prm.requires_grad_(False)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    resolve_hist_cache(a)
    hist = build_or_load_histograms(a, max_len=int(stats["max_num_atoms"]))
    init = model._sample_noise_like_batch(batch_size=a.n_conditions).to(dev)
    lengths = (~init["padding_mask"]).sum(dim=1).detach().cpu()
    trng = np.random.default_rng(a.target_seed)
    targets = torch.tensor([sample_property_target(hist, int(l), trng) for l in lengths],
                           dtype=torch.float32, device=dev)

    # conditioning states: integrate the base model to each state's own t ~ U(0.05, 0.35)
    sched = model._get_sample_schedule(a.sample_steps).to(dev)
    t_cond = torch.rand(a.n_conditions, device=dev) * 0.30 + 0.05
    state = init.clone()
    for i in range(1, len(sched)):
        t_val = sched[i - 1]
        live = (t_cond > float(t_val))
        if not bool(live.any()):
            break
        t = t_val.expand(state["coords"].shape[0])
        dt = (sched[i] - sched[i - 1]).expand(state["coords"].shape[0])
        nxt = production_step_with_optional_guidance(
            model, state, t, dt, mu=0.0, grad_coords=None, grad_atomics=None)
        for k in ("coords", "atomics"):
            state[k] = torch.where(live.view(-1, *([1] * (state[k].dim() - 1))), nxt[k], state[k])

    def value_of(sampler: str, n: int) -> torch.Tensor:
        """log-mean-exp of the scaled reward over n posterior samples, per condition."""
        acc = []
        done = 0
        while done < n:
            m = min(a.chunk, n - done)
            with torch.no_grad():
                if sampler == "mfm":
                    post = differentiable_mfm_posterior(student, state, t_cond,
                                                        posterior_samples=m, n_steps=4,
                                                        diagonal=True, prior_noise=None)
                else:
                    post = differentiable_joint_glass_posterior(model, state, t_cond,
                                                                posterior_samples=m,
                                                                n_steps=4, eps=1e-6,
                                                                prior_noise=None)
                r = grouped_rewards(post, batch_size=a.n_conditions, posterior_samples=m,
                                    reward_name="target_property", target_x=None,
                                    property_name=a.property_name, property_target=targets,
                                    property_regressor=reg)
            acc.append(a.reward_scale * r)
            done += m
        return torch.cat(acc, dim=-1)                       # [n_cond, n]

    lme = lambda s: torch.logsumexp(s, dim=-1) - math.log(s.shape[-1])

    ref_s = value_of("glass", a.n_ref)
    ref = lme(ref_s)
    half = a.n_ref // 2
    selfsplit = (lme(ref_s[:, :half]) - lme(ref_s[:, half:])).abs().mean().item()

    # DPS: the one-call denoiser point estimate, r(E[X_1 | X_t]).
    with torch.no_grad():
        x1 = endpoint_prediction(model, state, t_cond)
        r_dps = grouped_rewards(x1, batch_size=a.n_conditions, posterior_samples=1,
                                reward_name="target_property", target_x=None,
                                property_name=a.property_name, property_target=targets,
                                property_regressor=reg)
    dps = (a.reward_scale * r_dps).reshape(a.n_conditions)
    dps_err = (dps - ref).abs().mean().item()

    # FMAP: integrate the base ODE deterministically from each state's own t to 1 and score
    # that endpoint. A better point estimate than the one-step denoiser and a best case for
    # any deterministic one, but still a point estimate -- like DPS it has no N to spend, so
    # neither can represent the spread of the posterior that V_t integrates over.
    with torch.no_grad():
        fm = state.clone()
        for i in range(1, len(sched)):
            t_val = sched[i - 1]
            live = (t_cond <= float(t_val))
            if not bool(live.any()):
                continue
            t = t_val.expand(fm["coords"].shape[0])
            dt = (sched[i] - sched[i - 1]).expand(fm["coords"].shape[0])
            nxt = production_step_with_optional_guidance(
                model, fm, t, dt, mu=0.0, grad_coords=None, grad_atomics=None)
            for k in ("coords", "atomics"):
                fm[k] = torch.where(live.view(-1, *([1] * (fm[k].dim() - 1))), nxt[k], fm[k])
        r_fm = grouped_rewards(fm, batch_size=a.n_conditions, posterior_samples=1,
                               reward_name="target_property", target_x=None,
                               property_name=a.property_name, property_target=targets,
                               property_regressor=reg)
    fmap = (a.reward_scale * r_fm).reshape(a.n_conditions)
    fmap_err = (fmap - ref).abs().mean().item()

    rows = []
    mfm_s = value_of("mfm", max(int(x) for x in a.mcs.split(",")))
    for n in [int(x) for x in a.mcs.split(",")]:
        err = (lme(mfm_s[:, :n]) - ref).abs().mean().item()
        rows.append({"estimator": "multiMFM", "n": n, "abs_value_error": round(err, 5)})
        rows.append({"estimator": "DPS", "n": n, "abs_value_error": round(dps_err, 5)})
        rows.append({"estimator": "FMAP", "n": n, "abs_value_error": round(fmap_err, 5)})
        rows.append({"estimator": "reference self-split", "n": n,
                     "abs_value_error": round(selfsplit, 5)})

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["estimator", "n", "abs_value_error"])
        w.writeheader(); w.writerows(rows)
    print(f"n_conditions={a.n_conditions} n_ref={a.n_ref} selfsplit={selfsplit:.4f} "
          f"dps={dps_err:.4f} fmap={fmap_err:.4f}")
    for r in rows:
        if r["estimator"] == "multiMFM":
            print(f"  N={r['n']:>4}  multiMFM |V-V*| = {r['abs_value_error']:.4f}")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
