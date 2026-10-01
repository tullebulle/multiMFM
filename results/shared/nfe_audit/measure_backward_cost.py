#!/usr/bin/env python
"""Measure what a guidance backward pass actually costs, and re-time DNA Fig 4 right.

Two measurements, both on one GPU, both needed by `docs/nfe_accounting_audit.md`.

A. **Backward/forward ratio on the real code path.** Both codebases freeze every
   model parameter (``requires_grad_(False)``) and take the gradient w.r.t. the
   input state only, so the backward should cost roughly one forward's worth of
   matmuls, not two. This times the value-gradient call against the *same*
   posterior draw with autograd off, so the ratio is
   ``(forward + backward) / forward`` on the exact tensors the samplers use.
   The forward-equivalent axis of the audit is
   ``forwards + k * (forwards inside the graph)`` with ``k = ratio - 1``.

B. **Batch-matched wall-clock for DNA Figure 4 right.** The stored ``elapsed_s``
   in ``results/dna/fig4/panelB_runs.csv`` is not a fair clock for two reasons:
   (i) ``run_steering`` starts its timer *before* ``load_dmfm_student``, so the
   dMFM rows include a checkpoint load the baselines never pay; (ii) the driver
   defaults put 2,048-3,200 rows in flight for best-of-N / FK but only
   ``--dmfm-batch 25`` for the dMFM. Here every method is timed after all
   checkpoints are resident, and the dMFM is timed at its default width and at a
   width that matches the baselines.

Writes ``backward_cost.json`` and ``dna_wallclock_rerun.csv`` next to this file.

    python results/shared/nfe_audit/measure_backward_cost.py --part both
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# A tag keeps two concurrent Slurm attempts (different GPU models) from racing on
# the same output file; the unsuffixed names are the ones the audit doc cites.
TAG = os.environ.get("AUDIT_TAG", "")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, *, warmup: int = 2, reps: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts = sorted(ts)
    return {"median_s": ts[len(ts) // 2], "min_s": ts[0], "max_s": ts[-1], "reps": reps}


# --------------------------------------------------------------- A: DNA ratio


def dna_ratio(device, batches=(25, 100, 400, 1600)) -> dict:
    import torch.nn.functional as F  # noqa: F401
    from dmfm.benchmarks.core import NFECounter, load_base_flow, load_c0, load_dmfm_student
    from dmfm.benchmarks.dmfm_steering import PosteriorSampler, dmfm_value_and_grad
    from dmfm.benchmarks.search import RewardSpec

    nfe = NFECounter()
    flow = load_base_flow(50, device, nfe)
    guide = load_c0("guide", device, nfe, rc_average=False)
    student = load_dmfm_student(50, device)
    post = PosteriorSampler(flow, student, nfe)
    reward = RewardSpec(target=0.30, sigma=0.15, scale=1.0, hard=True)

    n_frozen = sum(1 for p in student.parameters() if not p.requires_grad)
    n_total = sum(1 for p in student.parameters())

    out = {"frozen_student_params": f"{n_frozen}/{n_total}",
           "frozen_guide_params": f"{sum(1 for p in guide.model.parameters() if not p.requires_grad)}"
                                  f"/{sum(1 for p in guide.model.parameters())}",
           "by_batch": {}}
    for B in batches:
        x = torch.randn(B, flow.L, flow.K, device=device)
        t = torch.full((B,), 0.5, device=device)

        def fwd_only():
            with torch.no_grad():
                x1 = post.dmfm(torch.randn_like(x), x, t, 1)
                guide.model(x1)

        def fwd_bwd():
            dmfm_value_and_grad(post, guide, reward, x, t, n_mc=1, n_steps=1, nfe=nfe)

        def base_fwd():
            with torch.no_grad():
                flow.denoiser(x, t)

        f = timeit(fwd_only)
        fb = timeit(fwd_bwd)
        bf = timeit(base_fwd)
        out["by_batch"][B] = {
            "student_fwd_plus_guide_s": f["median_s"],
            "student_fwd_plus_guide_plus_backward_s": fb["median_s"],
            "ratio_fwdbwd_over_fwd": fb["median_s"] / f["median_s"],
            "implied_k": fb["median_s"] / f["median_s"] - 1.0,
            "base_net_fwd_s": bf["median_s"],
            "raw": {"fwd": f, "fwd_bwd": fb, "base_fwd": bf},
        }
    return out


# --------------------------------------------------------------- A: QM9 ratio


def qm9_ratio(device, batches=(25, 100), mcs=(1, 4, 16)) -> dict:
    from tensordict import TensorDict  # noqa: F401

    from multimfm import paths
    from multimfm.glass_guidance import (
        differentiable_mfm_posterior,
        load_flow_model,
        load_mfm_student,
        value_gradient,
    )
    from multimfm.qm9_data import load_tfg_regressor

    model, data_stats, model_args = load_flow_model(paths.BASE_FLOW_CKPT, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    student = load_mfm_student(paths.MFM_STUDENT_CKPT, data_stats=data_stats,
                              teacher_args=model_args, device=device, key="student")
    reg = load_tfg_regressor("guide", "alpha", device)
    reg.soft_atom_temperature = 0.25
    for p in reg.parameters():
        p.requires_grad_(False)
    reg.eval()

    out = {"frozen_student_params":
           f"{sum(1 for p in student.parameters() if not p.requires_grad)}/"
           f"{sum(1 for p in student.parameters())}",
           "params": {"base_net": sum(p.numel() for p in getattr(model, "net", model).parameters()),
                      "mfm_student": sum(p.numel() for p in student.parameters()),
                      "guide_regressor": sum(p.numel() for p in reg.parameters())},
           "by_batch": {}}
    # How expensive is one regressor call relative to one generative forward? The
    # regressor is off the plotted NFE axis for every method, and the guided methods
    # make far more regressor calls than best-of-N, so this ratio decides whether
    # leaving it off the axis matters.
    from multimfm.glass_guidance import predicted_property
    for B in (100, 400):
        st = model._sample_noise_like_batch(batch_size=B).to(device)
        t = torch.full((B,), 0.5, device=device)
        try:
            r_reg = timeit(lambda st=st: predicted_property(
                st, property_name="alpha", property_regressor=reg, soft=True), reps=5)
            with torch.no_grad():
                r_base = timeit(lambda st=st, t=t: model._call_net(st, t), reps=5)
        except Exception as exc:                            # noqa: BLE001
            out[f"regressor_vs_base_B{B}"] = {"error": repr(exc)[:300]}
            continue
        out[f"regressor_vs_base_B{B}"] = {
            "regressor_fwd_s": r_reg["median_s"], "base_net_fwd_s": r_base["median_s"],
            "regressor_over_base": r_reg["median_s"] / r_base["median_s"]
            if r_base["median_s"] else None}
    for B in batches:
        state = model._sample_noise_like_batch(batch_size=B).to(device)
        t = torch.full((B,), 0.5, device=device)
        for mc in mcs:
            def fwd_only():
                with torch.no_grad():
                    differentiable_mfm_posterior(student, state, t, posterior_samples=mc,
                                                 n_steps=4, diagonal=True)

            def fwd_bwd():
                value_gradient(model, state, t, value_sampler="mfm", mfm_student=student,
                               mfm_diagonal=True, posterior_samples=mc, n_steps=4,
                               reward_name="target_property", reward_scale=0.3, target_x=0.0,
                               property_name="alpha", property_target=70.0,
                               property_regressor=reg, guide_atomics=True, eps=1e-6)

            try:
                f = timeit(fwd_only, reps=3)
                fb = timeit(fwd_bwd, reps=3)
            except RuntimeError as exc:                     # OOM at large B*mc
                out["by_batch"][f"B{B}_MC{mc}"] = {"error": str(exc)[:200]}
                torch.cuda.empty_cache()
                continue
            out["by_batch"][f"B{B}_MC{mc}"] = {
                "posterior_fwd_s": f["median_s"],
                "value_gradient_fwd_plus_bwd_s": fb["median_s"],
                "ratio_fwdbwd_over_fwd": fb["median_s"] / f["median_s"],
                "implied_k": fb["median_s"] / f["median_s"] - 1.0,
                "note": "fwd_only omits the regressor forward that fwd_bwd includes, "
                        "so the ratio is a slight OVER-estimate of the backward's share",
            }
    return out


# ------------------------------------------------------ B: DNA wall-clock rerun


def dna_wallclock(device, reps: int = 3) -> list[dict]:
    from dmfm.benchmarks.core import NFECounter, SamplerConfig, load_base_flow, load_c0, load_dmfm_student
    from dmfm.benchmarks.dmfm_steering import GuidanceConfig, PosteriorSampler, sample_dmfm_guided
    from dmfm.benchmarks.search import RewardSpec, beam_search, best_of_n, feynman_kac

    nfe = NFECounter()
    flow = load_base_flow(50, device, nfe)
    guide = load_c0("guide", device, nfe, rc_average=False)
    student = load_dmfm_student(50, device)          # loaded BEFORE any timer starts
    post = PosteriorSampler(flow, student, nfe)
    reward = RewardSpec(target=0.30, sigma=0.15, scale=1.0, hard=True)
    n_out = 100

    rows = []

    def run(label, method, config, fn, rows_in_flight):
        nfe_before = NFECounter(nfe.base, nfe.student, nfe.guide, nfe.backward)
        fn()                                         # warm-up (also fills the counters)
        gen_nfe = (nfe.gen - nfe_before.gen) / n_out
        bwd = (nfe.backward - nfe_before.backward) / n_out
        sync()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t0)
        ts = sorted(ts)
        rows.append({"method": method, "config": config, "label": label,
                     "nfe_gen_per_output": gen_nfe, "nfe_backward_per_output": bwd,
                     "rows_in_flight": rows_in_flight,
                     "s_median": ts[len(ts) // 2], "s_min": ts[0], "s_max": ts[-1],
                     "s_per_output": ts[len(ts) // 2] / n_out,
                     "n_outputs": n_out, "reps": reps})
        print(f"  {label:<34} NFE/out={gen_nfe:8.1f}  bwd/out={bwd:7.1f}  "
              f"median={ts[len(ts)//2]:7.3f}s", flush=True)

    det = SamplerConfig(n_steps=96, eta=0.0)
    sde = SamplerConfig(n_steps=96, eta=1.0)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    print("dMFM at the shipped width (--dmfm-batch 25) and batch-matched (100):", flush=True)
    for mc in (1, 4, 16):
        for bs in (25, 100):
            g = GuidanceConfig(n_mc=mc, nfe_value=1)
            run(f"dmfm MC={mc} dmfm_batch={bs}", "dmfm", f"MC={mc}",
                lambda mc=mc, bs=bs, g=g: sample_dmfm_guided(
                    flow, post, guide, reward, n_outputs=n_out, sampler=det,
                    guidance=g, batch_size=bs, nfe=nfe),
                25 * mc if bs == 25 else 100 * mc)

    print("search baselines on the same GPU, same n_outputs:", flush=True)
    for n in (2, 5, 9, 32):
        run(f"best_of_n N={n}", "best_of_n", f"N={n}",
            lambda n=n: best_of_n(flow, guide, reward, n_outputs=n_out, sampler=det,
                                  n_candidates=n, batch_rows=2048, generator=gen),
            min(max(1, 2048 // n) * n, n_out * n))
    for k in (2, 5, 9, 32):
        run(f"fk K={k}", "fk", f"K={k}",
            lambda k=k: feynman_kac(flow, guide, reward, n_outputs=n_out, sampler=sde,
                                    n_particles=k, beta=50.0, resample_every=8,
                                    t_min=0.01, t_max=0.95, generator=gen),
            n_out * k)
    for ce in (48, 16, 4):
        run(f"beam W=1,L=4,K={ce}", "beam", f"W=1,L=4,K={ce}",
            lambda ce=ce: beam_search(flow, guide, reward, n_outputs=n_out, sampler=sde,
                                      width=1, branch=4, checkpoint_every=ce, generator=gen),
            n_out * 4)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ratio", "wallclock", "both"], default="both")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    meta = {"device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
    print(json.dumps(meta), flush=True)

    if args.part in ("ratio", "both"):
        res = {"meta": meta}
        print("== DNA backward/forward ratio ==", flush=True)
        res["dna"] = dna_ratio(device)
        for b, v in res["dna"]["by_batch"].items():
            print(f"  B={b:<5} fwd={v['student_fwd_plus_guide_s']*1e3:7.2f} ms  "
                  f"fwd+bwd={v['student_fwd_plus_guide_plus_backward_s']*1e3:7.2f} ms  "
                  f"ratio={v['ratio_fwdbwd_over_fwd']:.2f}  k={v['implied_k']:.2f}", flush=True)
        (HERE / f"backward_cost{TAG}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
        print("== QM9 backward/forward ratio ==", flush=True)
        try:
            res["qm9"] = qm9_ratio(device)
            for b, v in res["qm9"]["by_batch"].items():
                if "error" in v:
                    print(f"  {b}: {v['error']}", flush=True)
                else:
                    print(f"  {b:<12} fwd={v['posterior_fwd_s']*1e3:8.2f} ms  "
                          f"fwd+bwd={v['value_gradient_fwd_plus_bwd_s']*1e3:8.2f} ms  "
                          f"ratio={v['ratio_fwdbwd_over_fwd']:.2f}  k={v['implied_k']:.2f}", flush=True)
        except Exception as exc:                            # noqa: BLE001
            res["qm9"] = {"error": repr(exc)[:400]}
            print(f"  QM9 part failed: {exc!r}", flush=True)
        (HERE / f"backward_cost{TAG}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")

    if args.part in ("wallclock", "both"):
        print("== DNA wall-clock rerun ==", flush=True)
        import csv
        rows = dna_wallclock(device)
        with (HERE / f"dna_wallclock_rerun{TAG}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["device_name"])
            w.writeheader()
            for r in rows:
                w.writerow(r | {"device_name": meta["device_name"]})
        print(f"wrote {HERE / f'dna_wallclock_rerun{TAG}.csv'}", flush=True)


if __name__ == "__main__":
    main()
