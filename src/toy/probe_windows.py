"""Where in (s, t_cond) is the distilled toy MFM actually accurate?

``toy.probe_mfm_diagonal`` (recovered original) reports the diagonal GLASS-velocity
error in four coarse bins over the *training* window.  Fig. 1 guidance uses the
student outside that window in two places, so this module measures the same error
on an explicit grid, including beyond the trained range:

* ``t_cond`` -- the outer ODE time at which the value function is evaluated.  The
  Norway panel guides over t in [0.8, 0.98]; both training stages of the shipped
  student used ``t_cond_max = 0.95``.
* ``s`` -- the inner time of the posterior map.  ``mfm_posterior_sample`` always
  integrates to ``final_u = 1.0``, while both stages used ``s_max``/``time_max``
  of 0.95, so the end of every posterior draw is extrapolation.

It also reports the *endpoint* error of the composed map itself: for
``n_mfm_steps`` in a list, how far the composed map's endpoint sits from a
well-resolved GLASS integration (128 Euler steps) started from the same noise --
the quantity that actually enters the value estimate.

    python -m toy.probe_windows --ckpt checkpoints/toy/mfm_consistency_10000.pt \
        --out results/toy/probe_windows_t095.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import NUM_CLASSES, DATA_SAMPLERS
from .glass import glass_posterior_velocity
from .mfm_value_fn import load_mfm, mfm_posterior_sample
from .paths import BASE_CKPT, MFM_CONSISTENCY_CKPT, RESULTS
from .sample import load_model
from .value_fn import freeze_model, glass_integrate, make_joint_teacher_v


def _batch(sample_data, n, s, t_cond, generator, device):
    """One evaluation batch at fixed (s, t_cond), built exactly like the trainer's."""
    c1, y1 = sample_data(n, generator=generator, device=device)
    x1 = torch.cat([F.one_hot(c1, NUM_CLASSES).to(y1.dtype), y1], dim=-1)
    x0 = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
    s_v = torch.full((n,), float(s), device=device, dtype=x1.dtype)
    I_s = (1.0 - s_v).unsqueeze(-1) * x0 + s_v.unsqueeze(-1) * x1
    noise = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
    t_v = torch.full((n,), float(t_cond), device=device, dtype=x1.dtype)
    x_cond = (1.0 - t_v).unsqueeze(-1) * noise + t_v.unsqueeze(-1) * x1
    return s_v, I_s, t_v, x_cond


@torch.no_grad()
def velocity_grid(mfm, teacher_v, sample_data, s_grid, t_grid, n, device, seed=0) -> list:
    g = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for t_cond in t_grid:
        for s in s_grid:
            s_v, I_s, t_v, x_cond = _batch(sample_data, n, s, t_cond, g, device)
            target = glass_posterior_velocity(s_v, I_s, x_cond, t_v, teacher_v)
            pred = mfm.v(s_v, s_v, I_s, t_cond=t_v, x_cond=x_cond)
            diff = pred - target
            rows.append({
                "s": float(s), "t_cond": float(t_cond),
                "rmse": float(diff.pow(2).sum(-1).mean().sqrt()),
                "relative_l2": float(diff.pow(2).sum(-1).mean().sqrt()
                                     / target.pow(2).sum(-1).mean().sqrt().clamp_min(1e-12)),
                "cosine": float(F.cosine_similarity(pred, target, dim=-1).mean()),
            })
    return rows


@torch.no_grad()
def endpoint_error(mfm, base, sample_data, t_grid, step_counts, n, device,
                   glass_ref_steps=128, seed=0, n_post=128, glass_steps=(8, 32)) -> list:
    """Conditional posterior p(x1 | x_t) from the composed map vs a 128-step GLASS reference.

    This is what the value function sees: for each conditioning state x_t it draws a
    set of endpoints and takes logmeanexp of the reward over that set.

    Every sampler is started from the *same* initial posterior noise (identically
    seeded generators, and both samplers draw it as their first random op), so the
    comparison is paired and free of Monte-Carlo noise:

    * ``rmse``        pointwise ||x1 - x1_ref|| from the same noise -- the map error;
    * ``mean_shift``  ||E[x1 | x_t] - E_ref[x1 | x_t]|| averaged over x_t -- the bias
                      the value gradient inherits;
    * ``spread_ratio``  ||sd(x1 | x_t)|| / ||sd_ref(x1 | x_t)|| averaged over x_t
                      (1.0 = right amount of posterior noise, < 1 = over-confident).
      The norm is taken over coordinates first, because the colour coordinates'
      conditional spread collapses to ~0 at large t_cond.
    """
    teacher_v = make_joint_teacher_v(base, NUM_CLASSES)
    rows = []
    for ti, t_cond in enumerate(t_grid):
        g = torch.Generator(device=device).manual_seed(seed + 1000 * ti)
        c1, y1 = sample_data(n, generator=g, device=device)
        x1 = torch.cat([F.one_hot(c1, NUM_CLASSES).to(y1.dtype), y1], dim=-1)
        noise = torch.randn(x1.shape, generator=g, device=device, dtype=x1.dtype)
        t_v = torch.full((n,), float(t_cond), device=device, dtype=x1.dtype)
        x_cond = (1.0 - t_v).unsqueeze(-1) * noise + t_v.unsqueeze(-1) * x1

        def gen():  # identically seeded -> identical initial posterior noise
            return torch.Generator(device=device).manual_seed(seed + 7 + 1000 * ti)

        ref = glass_integrate(base, t_v, x_cond, n_post, glass_ref_steps, generator=gen(),
                              teacher_v=teacher_v)                       # (n, n_post, D)
        ref_mu, ref_sd = ref.mean(1), ref.std(1).norm(dim=-1)
        row = {"t_cond": float(t_cond), "n_cond": int(n), "n_post": int(n_post),
               "glass_ref_steps": glass_ref_steps}

        def score(xs, tag):
            row[f"{tag}_rmse"] = float((xs - ref).pow(2).sum(-1).mean().sqrt())
            row[f"{tag}_mean_shift"] = float((xs.mean(1) - ref_mu).norm(dim=-1).mean())
            row[f"{tag}_spread_ratio"] = float((xs.std(1).norm(dim=-1) / ref_sd.clamp_min(1e-8)).mean())

        for k in step_counts:
            score(mfm_posterior_sample(mfm, t_v, x_cond, n_post, n_mfm_steps=k, generator=gen()),
                  f"mfm_{k}step")
        for gs in glass_steps:
            score(glass_integrate(base, t_v, x_cond, n_post, gs, generator=gen(), teacher_v=teacher_v),
                  f"glass_{gs}step")
        rows.append(row)
    return rows


def run(args) -> dict:
    device = args.device
    base = freeze_model(load_model(args.base_ckpt, device=device, use_ema=True))
    mfm = load_mfm(args.ckpt, device=device, use_ema=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sample_data = DATA_SAMPLERS[ck.get("dataset", "uniform_color_square")]
    teacher_v = make_joint_teacher_v(base, NUM_CLASSES)

    grid = velocity_grid(mfm, teacher_v, sample_data, args.s_grid, args.t_grid,
                         args.n, device, seed=args.seed)
    ends = endpoint_error(mfm, base, sample_data, args.t_grid, args.mfm_steps,
                          args.n_endpoint, device, seed=args.seed, n_post=args.n_post,
                          glass_steps=tuple(args.glass_steps))
    tc = ck.get("training_config", {})
    record = {"ckpt": str(args.ckpt), "training_config": tc,
              "trained_s_max": tc.get("s_max", tc.get("time_max")),
              "trained_t_cond_max": tc.get("t_cond_max"),
              "n": args.n, "n_endpoint": args.n_endpoint,
              "velocity_grid": grid, "endpoint_error": ends}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))

    print(f"trained window: s <= {record['trained_s_max']}, t_cond <= {record['trained_t_cond_max']}")
    print("\ndiagonal velocity relative L2  (rows = t_cond, cols = s)")
    print("  t_cond \\ s " + "".join(f"{s:>9.2f}" for s in args.s_grid))
    for t in args.t_grid:
        vals = [r for r in grid if r["t_cond"] == t]
        print(f"  {t:>10.2f} " + "".join(f"{r['relative_l2']:>9.4f}" for r in vals))
    print("\nconditional posterior vs GLASS-128, same initial noise "
          "(pointwise rmse / conditional-mean shift / spread ratio)")
    hdr = [f"mfm-{k}" for k in args.mfm_steps] + [f"glass-{k}" for k in args.glass_steps]
    print("  t_cond " + "".join(f"{h:>18}" for h in hdr))
    for r in ends:
        cells = [f"{r[f'mfm_{k}step_rmse']:.3f}/{r[f'mfm_{k}step_mean_shift']:.3f}/{r[f'mfm_{k}step_spread_ratio']:.2f}"
                 for k in args.mfm_steps]
        cells += [f"{r[f'glass_{k}step_rmse']:.3f}/{r[f'glass_{k}step_mean_shift']:.3f}/{r[f'glass_{k}step_spread_ratio']:.2f}"
                  for k in args.glass_steps]
        print(f"  {r['t_cond']:>6.2f} " + "".join(f"{c:>18}" for c in cells))
    print(f"\nsaved {out}")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(MFM_CONSISTENCY_CKPT))
    p.add_argument("--base_ckpt", default=str(BASE_CKPT))
    p.add_argument("--s_grid", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.999])
    p.add_argument("--t_grid", type=float, nargs="+",
                   default=[0.2, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
    p.add_argument("--mfm_steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--n_endpoint", type=int, default=256, help="conditioning states")
    p.add_argument("--n_post", type=int, default=128, help="posterior samples per conditioning state")
    p.add_argument("--glass_steps", type=int, nargs="+", default=[8, 32])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(RESULTS / "probe_windows.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__ == "__main__":
    main()
