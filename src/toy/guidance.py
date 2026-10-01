"""Reward-guided samplers for the Norway and smiley examples (GLASS or MFM value).

RECONSTRUCTED.  The original experiment scripts
(``experiment_uniform_norway_mfm_guidance.py``,
``experiment_uniform_smiley_mfm_guidance.py`` and their GLASS predecessors under
``dMFM_experiments/``) were purged.  What survives and fixes this module:

* the MFM notebook (Cursor history, 2026-05-11/12) with the exact call
  signatures ``delayed_color_sample_mfm(...)`` and ``sample_hybrid_mfm(...)``,
  the final configs (see ``NORWAY_MFM`` / ``SMILEY_MFM`` below) and part of the
  unguided Norway log (colour velocity exactly 0 before t = 0.8, then
  ||v_base,c|| ~ 7.2; unguided reward 0.324, match 0.334);
* the terminal log: guidance is ``dx/dt = base_velocity + guidance`` with
  ``guidance = grad_x V_t(x)``; the MFM only replaces the posterior/value
  estimator; colour guidance uses a *fixed guidance ratio* that rescales the
  colour gradient to ``ratio * ||v_base,c||`` (diagnostics print
  ``||v_base,c||``, ``||grad_c||`` and ``guidance/base``); base and guidance
  passes are chunked (``base_batch_size``, ``guidance_batch_size``); smiley
  position guidance is ``pos_lambda`` "with SDE-sigma scaling"
  (``pos_lambda_base="sde_sigma_sq"``), interpreted as
  sigma_t^2 = 2 (1 - t) / t, the linear-interpolant SDE diffusion used by the
  same author's DNA code (``dmfm.models.dna_models.sde_sigma_sq``).

"Delayed colour": the base colour velocity is set to zero while
t < ``color_base_freeze_until`` (colour stays at its prior draw), then the
colour follows the ordinary endpoint velocity (softmax(z) - c) / (1 - t).
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .data import COLOR_NAMES, NUM_CLASSES
from .mfm_value_fn import mfm_value_function
from .reward import norway_target, norwegian_color_reward_raw, norwegian_region_scores
from .reward_smiley import SMILEY_SHAPE, smiley_reward_raw, smiley_target
from .sample import base_endpoint, readout, resolve_figure_path, time_grid
from .value_fn import make_joint_teacher_v, value_function

# Final settings recorded in the MFM notebook (mfm_guidance_norway_smiley.ipynb).
NORWAY_MFM = {
    "n_samples": 10_000, "n_steps": 100, "n_value_samples": 8, "n_mfm_steps": 1,
    "mfm_final_u": 1.0, "beta": 16.0, "k": 40.0, "color_base_freeze_until": 0.8,
    "guidance_start": 0.8, "guidance_end": 0.95, "guidance_ratio": 6.0,
    "guidance_batch_size": 1000, "base_batch_size": 10_000, "seed": 0,
}
SMILEY_MFM = {
    "n_samples": 1000, "n_steps": 200, "n_value_samples": 16, "n_mfm_steps": 1,
    "mfm_final_u": 1.0, "beta": 64.0, "pos_lambda": 0.2, "pos_start": 0.0,
    "pos_end": 0.999, "pos_lambda_base": "sde_sigma_sq", "color_base_freeze_until": 0.8,
    "color_guidance_ratio": 2.0, "color_guidance_start": 0.8, "color_guidance_end": 0.98,
    "guidance_batch_size": 500, "base_batch_size": 10_000, "seed": 0,
}
# GLASS-guided settings -- these are the ones the *published* Fig. 1 used.
# The four-panel notebook's stored `bundle["sources"]` (recovered from the Cursor
# local history of norway_base_smiley_four_panel.ipynb, saved 2026-05-06 02:30 Z,
# outputs intact) names the two endpoint files behind the guided panels:
#
#   norway: figures/uniform_norway_slow_color_guidance_N10000_freeze_t08_ratio2_endpoints.pt
#   smiley: figures/uniform_smiley_guidance_N1000_lam0p2_beta16_mc16_steps200_glass8_chunk100_endpoints.pt
#   prior : torch.randn with seed=0
#
# and the final save of that notebook (15:20 Z the same day, the version whose
# output is figures/norway.pdf in the paper) swaps the smiley panel for
#   figures/uniform_smiley_delayed_color_bigger_smile_thick_mouth_N10000_endpoints.pt
# i.e. the same GLASS sampler at N = 10000 with the larger "bigger smile / thick
# mouth" geometry (SMILEY_SHAPE).  All of these predate the toy MFM.  See
# docs/provenance/toy/README.md.
#
# Recorded: Norway N=10000, colour frozen until t=0.8, ratio 2, window to 0.98
# ("the Norway guidance window previously went to 0.98"), MC 8; smiley
# pos_lambda 0.2, beta 16, MC 16, 200 outer steps, 8 GLASS inner steps, guidance
# chunk 100.  NOT recorded: the smiley colour ratio for GLASS, and beta for the
# final N=10000 smiley run (the N=1000 predecessor used 16).
NORWAY_GLASS = dict(NORWAY_MFM, guidance_ratio=2.0, guidance_end=0.98, n_glass_steps=8)
for _k in ("n_mfm_steps", "mfm_final_u"):
    NORWAY_GLASS.pop(_k)
# (the recorded guidance chunk was 100; chunking only splits the value-gradient
# pass, so 1000 is the same computation with a different RNG stream -- it makes
# the 10k x 200-step GLASS smiley run tractable.)
SMILEY_GLASS = dict(
    SMILEY_MFM, n_samples=10_000, n_glass_steps=8, beta=16.0, guidance_batch_size=1000
)
for _k in ("n_mfm_steps", "mfm_final_u"):
    SMILEY_GLASS.pop(_k)

# Metrics printed by the ORIGINAL runs (Cursor terminal log, docs/provenance/toy/
# recovered/cursor_terminal_1.txt).  Used as the reference the rebuild is checked
# against -- the paper itself reports no numbers for the toy.
ORIGINAL_RUN_METRICS = {
    "norway_unguided_N10000": {"mean_reward": 0.324, "match": 0.334},
    "norway_glass_ratio2_smoke": {"mean_reward": 0.407, "match": 0.573},
    "norway_mfm_1step_ratio2_N10000": {"mean_reward": 0.405, "match": 0.564},
    "norway_mfm_1step_ratio6_N10000": {"mean_reward": 0.471, "match": 0.768},
    "smiley_glass_bigger_geometry": {"mean_reward": 0.555, "match": 0.959},
    "smiley_mfm_beta64_ratio2_late": {"mean_reward": 0.563, "match": 0.961},
}


def sde_sigma_sq(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """sigma_t^2 = 2 (1/t - 1) for the linear interpolant (alpha = 1 - t, beta = t)."""
    return 2.0 * (1.0 / t.clamp_min(eps) - 1.0)


def _lambda_base(kind: str, t: torch.Tensor) -> torch.Tensor:
    if kind in ("constant", "none", None):
        return torch.ones_like(t)
    if kind == "sde_sigma_sq":
        return sde_sigma_sq(t)
    if kind == "tilt":  # exact tilt coefficient (1 - t) / t = sigma_t^2 / 2
        return 0.5 * sde_sigma_sq(t)
    raise ValueError(f"unknown pos_lambda_base {kind!r}")


def _value_grad(value_fn, t: float, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    grads = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].detach().requires_grad_(True)
        tb = torch.full((xb.shape[0],), t, device=x.device, dtype=x.dtype)
        with torch.enable_grad():
            V = value_fn(tb, xb)
            (g,) = torch.autograd.grad(V.sum(), xb)
        grads.append(g.detach())
    return torch.cat(grads)


def _ratio_guidance(v_base_c: torch.Tensor, grad_c: torch.Tensor, ratio: float, eps: float = 1e-12):
    base_norm = v_base_c.norm(dim=-1, keepdim=True)
    grad_norm = grad_c.norm(dim=-1, keepdim=True)
    direction = torch.where(grad_norm > eps, grad_c / grad_norm.clamp_min(eps), torch.zeros_like(grad_c))
    return ratio * base_norm * direction


def guided_sample(
    base_model,
    value_fn,
    n_samples: int,
    n_steps: int,
    reward_monitor=None,
    device: str = "cpu",
    seed: int = 0,
    color_base_freeze_until: float = 0.8,
    color_guidance_ratio: float = 0.0,
    color_guidance_start: float = 0.8,
    color_guidance_end: float = 0.95,
    pos_lambda: float = 0.0,
    pos_start: float = 0.0,
    pos_end: float = 0.999,
    pos_lambda_base: str = "sde_sigma_sq",
    guidance_batch_size: int = 1000,
    base_batch_size: int = 10_000,
    eps: float = 1e-3,
    readout_mode: str = "sample",
    verbose: bool = True,
    print_every: int | None = None,
):
    """Euler sampler with delayed colour, ratio-normalised colour guidance and
    (optional) sigma^2-scaled position guidance.  Returns (x1, diagnostics)."""
    g = torch.Generator(device=device).manual_seed(seed)
    D = NUM_CLASSES + base_model.y_dim
    x = torch.randn(n_samples, D, generator=g, device=device)
    x_prior = x.clone()
    ts = time_grid(n_steps, eps, device=device)
    print_every = print_every or max(1, n_steps // 10)
    diag = {k: [] for k in ("t", "reward", "v_base_c", "grad_c", "guidance_over_base", "grad_y", "guid_y")}
    for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:]), start=1):
        t_val = float(t0)
        dt = float(t1 - t0)
        t_vec = torch.full((n_samples,), t_val, device=device)
        with torch.no_grad():
            y_pred, z_logits = base_endpoint(base_model, t_vec, x, base_batch_size)
            x1_hat = torch.cat([F.softmax(z_logits, dim=-1), y_pred], dim=-1)
            v = (x1_hat - x) / (1.0 - t_val)
        if t_val < color_base_freeze_until:
            v[:, :NUM_CLASSES] = 0.0
        v_base_c = v[:, :NUM_CLASSES].clone()

        color_on = color_guidance_ratio > 0 and color_guidance_start <= t_val <= color_guidance_end
        pos_on = pos_lambda > 0 and pos_start <= t_val <= pos_end
        gc_norm = gy_norm = guid_y = 0.0
        ratio_obs = 0.0
        if color_on or pos_on:
            grad = _value_grad(value_fn, t_val, x, guidance_batch_size)
            if color_on:
                gc = grad[:, :NUM_CLASSES]
                applied = _ratio_guidance(v_base_c, gc, color_guidance_ratio)
                v[:, :NUM_CLASSES] = v[:, :NUM_CLASSES] + applied
                gc_norm = gc.norm(dim=-1).mean().item()
                ratio_obs = (applied.norm(dim=-1) / v_base_c.norm(dim=-1).clamp_min(1e-12)).mean().item()
            if pos_on:
                gy = grad[:, NUM_CLASSES:]
                lam = pos_lambda * _lambda_base(pos_lambda_base, t_vec).unsqueeze(-1)
                gpos = torch.nan_to_num(lam * gy, nan=0.0, posinf=0.0, neginf=0.0)
                v[:, NUM_CLASSES:] = v[:, NUM_CLASSES:] + gpos
                gy_norm = gy.norm(dim=-1).mean().item()
                guid_y = gpos.norm(dim=-1).mean().item()
        with torch.no_grad():
            r = (
                reward_monitor(torch.cat([x[:, :NUM_CLASSES], y_pred], dim=-1)).mean().item()
                if reward_monitor is not None
                else float("nan")
            )
        diag["t"].append(t_val)
        diag["reward"].append(r)
        diag["v_base_c"].append(v_base_c.norm(dim=-1).mean().item())
        diag["grad_c"].append(gc_norm)
        diag["guidance_over_base"].append(ratio_obs)
        diag["grad_y"].append(gy_norm)
        diag["guid_y"].append(guid_y)
        if verbose and (i == 1 or i % print_every == 1 or i == n_steps):
            print(
                f"  step {i:3d}/{n_steps} t={t_val:.3f} reward={r:.3f} "
                f"||v_base,c||={diag['v_base_c'][-1]:.3f} ||grad_c||={gc_norm:.3f} "
                f"guidance/base={ratio_obs:.3f}"
                + (f" ||grad_y||={gy_norm:.3f} ||guid_y||={guid_y:.3f}" if pos_lambda > 0 else ""),
                flush=True,
            )
        x = (x + dt * v).detach()
    t_vec = torch.full((n_samples,), float(ts[-1]), device=device)
    with torch.no_grad():
        y_pred, z_logits = base_endpoint(base_model, t_vec, x, base_batch_size)
    out = readout(y_pred, z_logits, generator=g, mode=readout_mode)
    diag["prior"] = x_prior.cpu()
    return out, diag


# --------------------------------------------------------------------- Norway
def norway_reward_fn(beta: float = 16.0, k: float = 40.0, **kw):
    """beta * <s(y), log softmax(c)>, regions stop-gradiented (colour-only guidance)."""
    from .reward import norwegian_color_reward_for_model_state

    def fn(X1):
        return beta * norwegian_color_reward_for_model_state(
            X1, k=k, color_score="log_softmax", detach_regions=True, **kw
        )

    return fn


def delayed_color_sample_mfm(
    base_model, mfm, n_samples, n_steps, reward_fn, device, seed=0,
    color_base_freeze_until=0.8, guidance_ratio=0.0, n_value_samples=8, n_mfm_steps=1,
    mfm_final_u=1.0, guidance_start=0.8, guidance_end=0.95, guidance_batch_size=1000,
    base_batch_size=10_000, verbose=True, k=40.0,
):
    gen = torch.Generator(device=device).manual_seed(seed + 12345)

    def value_fn(t, x):
        return mfm_value_function(mfm, t, x, reward_fn, n_value_samples, n_mfm_steps, mfm_final_u, generator=gen)

    return guided_sample(
        base_model, value_fn, n_samples, n_steps,
        reward_monitor=lambda z: norwegian_color_reward_raw(z, k=k), device=device, seed=seed,
        color_base_freeze_until=color_base_freeze_until, color_guidance_ratio=guidance_ratio,
        color_guidance_start=guidance_start, color_guidance_end=guidance_end,
        guidance_batch_size=guidance_batch_size, base_batch_size=base_batch_size, verbose=verbose,
    )


def delayed_color_sample_glass(
    base_model, n_samples, n_steps, reward_fn, device, seed=0,
    color_base_freeze_until=0.8, guidance_ratio=0.0, n_value_samples=8, n_glass_steps=8,
    guidance_start=0.8, guidance_end=0.98, guidance_batch_size=1000,
    base_batch_size=10_000, verbose=True, k=40.0,
):
    gen = torch.Generator(device=device).manual_seed(seed + 12345)
    teacher_v = make_joint_teacher_v(base_model, NUM_CLASSES)

    def value_fn(t, x):
        return value_function(base_model, t, x, reward_fn, n_value_samples, n_glass_steps, generator=gen, teacher_v=teacher_v)

    return guided_sample(
        base_model, value_fn, n_samples, n_steps,
        reward_monitor=lambda z: norwegian_color_reward_raw(z, k=k), device=device, seed=seed,
        color_base_freeze_until=color_base_freeze_until, color_guidance_ratio=guidance_ratio,
        color_guidance_start=guidance_start, color_guidance_end=guidance_end,
        guidance_batch_size=guidance_batch_size, base_batch_size=base_batch_size, verbose=verbose,
    )


def summarize_norway(label: str, x: torch.Tensor, k: float = 40.0, verbose: bool = True) -> dict:
    x = x.detach().cpu()
    reward = norwegian_color_reward_raw(x, k=k)
    y = x[:, NUM_CLASSES:]
    pred = x[:, :NUM_CLASSES].argmax(dim=-1)
    target = norway_target(y, k=k)
    out = {
        "label": label,
        "n": int(x.shape[0]),
        "mean_reward": reward.mean().item(),
        "min_reward": reward.min().item(),
        "max_reward": reward.max().item(),
        "reward_ungated_mean": norwegian_color_reward_raw(x, k=k, square_gate=False).mean().item(),
        "match": (pred == target).float().mean().item(),
        "inside_square": (y.abs() <= 1).all(-1).float().mean().item(),
        "color_fractions_BWR": [(pred == c).float().mean().item() for c in range(NUM_CLASSES)],
        "region_fractions_BWR": [(target == c).float().mean().item() for c in range(NUM_CLASSES)],
        "y_mean": y.mean(0).tolist(),
    }
    if verbose:
        print(f"\n{label}")
        print(f"  mean reward = {out['mean_reward']:.3f} (min {out['min_reward']:.3f}, max {out['max_reward']:.3f})")
        print(f"  target-color match = {out['match']:.3f}")
        print(f"  fraction inside square = {out['inside_square']:.3f}")
        print("  color fractions B/W/R = " + "/".join(f"{v:.2f}" for v in out["color_fractions_BWR"]))
        print("  region fractions B/W/R = " + "/".join(f"{v:.2f}" for v in out["region_fractions_BWR"]))
    return out


# --------------------------------------------------------------------- Smiley
def smiley_reward_fn(beta: float = 64.0, shape: dict | None = None):
    from .reward_smiley import smiley_log_reward_for_model_state

    shape = dict(SMILEY_SHAPE if shape is None else shape)

    def fn(X1):
        return beta * smiley_log_reward_for_model_state(X1, detach_regions=False, **shape)

    return fn


def sample_hybrid_mfm(
    base_model, mfm, n_samples, n_steps, reward_fn, device, seed=0,
    pos_lambda=0.2, pos_start=0.0, pos_end=0.999, pos_lambda_base="sde_sigma_sq",
    color_base_freeze_until=0.8, color_guidance_ratio=2.0, color_guidance_start=0.8,
    color_guidance_end=0.98, n_value_samples=16, n_mfm_steps=1, mfm_final_u=1.0,
    guidance_batch_size=500, base_batch_size=10_000, verbose=True, shape=None,
):
    gen = torch.Generator(device=device).manual_seed(seed + 12345)
    shape = dict(SMILEY_SHAPE if shape is None else shape)

    def value_fn(t, x):
        return mfm_value_function(mfm, t, x, reward_fn, n_value_samples, n_mfm_steps, mfm_final_u, generator=gen)

    return guided_sample(
        base_model, value_fn, n_samples, n_steps,
        reward_monitor=lambda z: smiley_reward_raw(z, **shape), device=device, seed=seed,
        color_base_freeze_until=color_base_freeze_until, color_guidance_ratio=color_guidance_ratio,
        color_guidance_start=color_guidance_start, color_guidance_end=color_guidance_end,
        pos_lambda=pos_lambda, pos_start=pos_start, pos_end=pos_end, pos_lambda_base=pos_lambda_base,
        guidance_batch_size=guidance_batch_size, base_batch_size=base_batch_size, verbose=verbose,
    )


def sample_hybrid_glass(
    base_model, n_samples, n_steps, reward_fn, device, seed=0,
    pos_lambda=0.2, pos_start=0.0, pos_end=0.999, pos_lambda_base="sde_sigma_sq",
    color_base_freeze_until=0.8, color_guidance_ratio=2.0, color_guidance_start=0.8,
    color_guidance_end=0.98, n_value_samples=16, n_glass_steps=8,
    guidance_batch_size=500, base_batch_size=10_000, verbose=True, shape=None,
):
    gen = torch.Generator(device=device).manual_seed(seed + 12345)
    teacher_v = make_joint_teacher_v(base_model, NUM_CLASSES)
    shape = dict(SMILEY_SHAPE if shape is None else shape)

    def value_fn(t, x):
        return value_function(base_model, t, x, reward_fn, n_value_samples, n_glass_steps, generator=gen, teacher_v=teacher_v)

    return guided_sample(
        base_model, value_fn, n_samples, n_steps,
        reward_monitor=lambda z: smiley_reward_raw(z, **shape), device=device, seed=seed,
        color_base_freeze_until=color_base_freeze_until, color_guidance_ratio=color_guidance_ratio,
        color_guidance_start=color_guidance_start, color_guidance_end=color_guidance_end,
        pos_lambda=pos_lambda, pos_start=pos_start, pos_end=pos_end, pos_lambda_base=pos_lambda_base,
        guidance_batch_size=guidance_batch_size, base_batch_size=base_batch_size, verbose=verbose,
    )


def summarize_smiley(label: str, x: torch.Tensor, shape: dict | None = None,
                     coverage_threshold: float = 0.5, verbose: bool = True) -> dict:
    """Coverage = fraction with max_k s_k(y) > threshold (threshold not recorded; 0.5 assumed)."""
    shape = dict(SMILEY_SHAPE if shape is None else shape)
    x = x.detach().cpu()
    reward = smiley_reward_raw(x, **shape)
    y = x[:, NUM_CLASSES:]
    pred = x[:, :NUM_CLASSES].argmax(dim=-1)
    target, strength = smiley_target(y, **shape)
    covered = strength > coverage_threshold
    out = {
        "label": label,
        "n": int(x.shape[0]),
        "mean_reward": reward.mean().item(),
        "feature_coverage": covered.float().mean().item(),
        "match": (pred == target).float().mean().item(),
        "match_on_covered": (pred == target)[covered].float().mean().item() if covered.any() else float("nan"),
        "color_fractions_BWR": [(pred == c).float().mean().item() for c in range(NUM_CLASSES)],
        "coverage_threshold": coverage_threshold,
    }
    if verbose:
        print(f"\n{label}")
        print(f"  mean reward = {out['mean_reward']:.3f}")
        print(f"  feature coverage = {out['feature_coverage']:.3f}")
        print(f"  color-target match = {out['match']:.3f}")
        print("  color fractions B/W/R = " + "/".join(f"{v:.2f}" for v in out["color_fractions_BWR"]))
    return out


# ------------------------------------------------------------------- plotting
PLOT_COLORS = {"blue": "#006DCC", "white": "#F8F8F8", "red": "#FF2D20"}


def panel(ax, x, title, marker_size=5.5, lim=1.08, shuffle=True, seed=0):
    """Scatter one set of endpoints on a dark background (single shuffled draw order)."""
    x = x.detach().cpu()
    pred = x[:, :NUM_CLASSES].argmax(dim=-1)
    y = x[:, NUM_CLASSES:]
    order = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(x.shape[0])
    cols = [PLOT_COLORS[COLOR_NAMES[int(c)]] for c in pred[order]]
    ax.scatter(y[order, 0], y[order, 1], s=marker_size, c=cols, edgecolors="none", rasterized=True)
    ax.set_facecolor("#050505")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def plot_diags(diags: dict, out_name: str):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    keys = [("reward", "reward (monitor)"), ("v_base_c", "||v_base,c||"),
            ("grad_c", "||grad_c V||"), ("guidance_over_base", "guidance / base")]
    for label, d in diags.items():
        for ax, (k, title) in zip(axes, keys):
            ax.plot(d["t"], d[k], label=label)
            ax.set_title(title)
            ax.set_xlabel("t")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    out = resolve_figure_path(out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
