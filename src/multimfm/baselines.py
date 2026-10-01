"""Same-backbone inference-time steering baselines for QM9 property guidance.

Implements the methods that multiMFM / multiMFM-SS are compared against, ALL on the
same base flow model, with the same targets, the same regressors, the same validity
treatment, and the same evaluation code path (``steer_search.evaluate_state``):

  base       unconditional sampling (no guidance)
  best_of_n  N independent unguided samples per target; keep the best by the GUIDE
             regressor. Reported both unfiltered and PB-valid-filtered (the lowest
             guide error among PB-valid candidates, else the unfiltered best),
             mirroring steer_search's raw / pb-valid split.
  dps        Diffusion Posterior Sampling: guide by grad_x r(x1_hat(x_t)) where
             x1_hat is the model's ONE-STEP deterministic endpoint prediction.
             No posterior sampling -- this is the "denoiser" value approximation.
  fk         Feynman-Kac steering: K particles per target, periodically reweighted
             by exp(lambda * delta_r) on the lookahead endpoint and multinomially
             resampled. Sequential-Monte-Carlo steering, no gradients.

NFE accounting (identical for every method, including multiMFM in ``steer_search``
and ``property_targets``): forward hooks on the base network, the MFM student and the
regressors (``multimfm.nfe``) count rows per forward call, split into
generative forward passes (base + MFM), the subset that is back-propagated, and
regressor calls; ``summary.json["nfe"]`` holds them per generated molecule (i.e.
per target). The primary axis is ``nfe_generative``; ``nfe_generative_fwd_equiv``
(= forward + 2 x backward) accounts for gradient-based methods' backward passes.
The legacy manual counters of the 2026-09 pilot/sweep are reproduced as
``nfe_base_per_mol`` etc. for comparison.

Targets are drawn EXACTLY as in steer_search.py (same ``--seed`` / ``--target-seed``
/ ``--num-samples`` give the same molecule lengths and the same property targets),
so every method is steered toward an identical target set.

Changes vs the version behind the 2026-09-23/25 pilot and sweep: NFE measured by
hooks (the manual counts are unchanged for base/best-of-N/FK/DPS); candidate
selection checks PoseBusters lazily in guide-error order (same selection as the
exhaustive check, far fewer PB calls, which makes N = 256-512 affordable);
``--batch-size`` now actually chunks sampling; per-sample CSV; output file name
``<method>_<prop>_seed<seed>.json`` unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from multimfm import paths  # noqa: E402
from multimfm.glass_guidance import (  # noqa: E402
    clip_guidance_rms,
    clone_state,
    load_flow_model,
    predicted_property,
    production_step_with_optional_guidance,
    repeat_state,
    require_linear_dfm_atoms,
    seed_all,
)
from multimfm.glass_posterior import endpoint_prediction  # noqa: E402
from multimfm.nfe import NFECounter, attach_standard  # noqa: E402
from multimfm.property_targets import (  # noqa: E402
    build_or_load_histograms,
    sample_property_target,
)
from multimfm.qm9_data import choose_device, load_tfg_regressor  # noqa: E402
from multimfm.qm9_eval import PBPool, make_state  # noqa: E402
from multimfm.steer_search import evaluate_state, lazy_select  # noqa: E402
from tabasco.chem.convert import MoleculeConverter  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=["base", "best_of_n", "dps", "fk"])
    p.add_argument("--checkpoint", type=Path, default=paths.BASE_FLOW_CKPT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    # --- shared with steer_search: these define the target set. Do not diverge. ---
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--target-seed", type=int, default=2026)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=128)
    p.add_argument("--property-name", default="alpha",
                   choices=["alpha", "cv", "gap", "homo", "lumo", "mu"])
    p.add_argument("--partition", default="train_flow")
    p.add_argument("--hist-bins", type=int, default=100)
    p.add_argument("--soft-atom-temperature", type=float, default=0.25)
    p.add_argument("--guide-min-t", type=float, default=0.05)
    p.add_argument("--guide-max-t", type=float, default=0.95)
    p.add_argument("--guide-every", type=int, default=1)
    # --- method knobs ---
    p.add_argument("--n-candidates", type=int, default=8,
                   help="best_of_n: independent samples per target.")
    p.add_argument("--mu", type=float, default=0.45, help="dps: guidance strength.")
    p.add_argument("--guidance-max-rms", type=float, default=0.0,
                   help="dps: per-sample gradient RMS cap (0 = off).")
    p.add_argument("--n-particles", type=int, default=8, help="fk: particles per target.")
    p.add_argument("--fk-lambda", type=float, default=1.0,
                   help="fk: reward temperature in the resampling weights.")
    p.add_argument("--fk-ess-threshold", type=float, default=0.5,
                   help="fk: resample only when the effective sample size falls below this "
                        "fraction of K (0 = resample unconditionally, the old behaviour).")
    p.add_argument("--fk-resample-every", type=int, default=8,
                   help="fk: steps between resampling events.")
    p.add_argument("--fk-weight", choices=["delta", "absolute"], default="delta",
                   help="fk: weights from the reward INCREMENT (delta, standard FK-IPS) "
                        "or from the raw reward (absolute).")
    # --- eval / io ---
    p.add_argument("--batch-size", type=int, default=8192,
                   help="best_of_n: chunk the N x num_samples unguided batch (0 = one batch).")
    p.add_argument("--pb-workers", type=int, default=16)
    p.add_argument("--no-sanitize", action="store_true")
    p.add_argument("--posebusters-config", type=Path, default=None)
    p.add_argument("--hist-cache", type=Path, default=None)
    p.add_argument("--force-rebuild-hist", action="store_true")
    p.add_argument("--output-dir", type=Path, default=paths.OUTPUTS / "baselines")
    return p.parse_args(argv)


# ---------------------------------------------------------------- samplers


def _cat_states(states):
    return TensorDict(
        {k: torch.cat([s[k] for s in states], dim=0) for k in ("coords", "atomics", "padding_mask")},
        batch_size=sum(s["padding_mask"].shape[0] for s in states),
    ).to(states[0].device)


@torch.no_grad()
def sample_unguided(model, init_state, *, steps: int, batch_size: int = 0) -> TensorDict:
    """Plain production sampling (one base-model call per step), optionally chunked."""
    n = init_state["coords"].shape[0]
    if 0 < batch_size < n:
        parts = []
        for a in range(0, n, batch_size):
            sl = slice(a, min(n, a + batch_size))
            sub = TensorDict({k: init_state[k][sl] for k in ("coords", "atomics", "padding_mask")},
                             batch_size=sl.stop - sl.start).to(init_state.device)
            parts.append(sample_unguided(model, sub, steps=steps, batch_size=0))
        return _cat_states(parts)
    state = clone_state(init_state)
    schedule = model._get_sample_schedule(steps).to(state["coords"].device)
    B = state["coords"].shape[0]
    for i in range(1, len(schedule)):
        t = schedule[i - 1].expand(B)
        dt = (schedule[i] - schedule[i - 1]).expand(B)
        state = production_step_with_optional_guidance(
            model, state, t, dt, mu=0.0, grad_coords=None, grad_atomics=None)
    return state


def dps_gradient(model, state, t, *, targets, guide_reg, property_name):
    """grad_x of the reward at the ONE-STEP endpoint estimate x1_hat(x_t).

    This is the DPS / denoiser value approximation: V(x_t) ~ r(E[x1 | x_t]),
    in contrast to multiMFM's V(x_t) = log E[exp(r(x1)) | x_t] over sampled posteriors.
    """
    coords = state["coords"].detach().clone().requires_grad_(True)
    atomics = state["atomics"].detach().clone().requires_grad_(True)
    cond = TensorDict(
        {"coords": coords, "atomics": atomics, "padding_mask": state["padding_mask"]},
        batch_size=state["padding_mask"].shape[0],
    ).to(state.device)
    with torch.enable_grad():
        x1_hat = endpoint_prediction(model, cond, t)
        pred = predicted_property(x1_hat, property_name=property_name,
                                  property_regressor=guide_reg)
        reward = -(pred - targets).square().sum()
        grads = torch.autograd.grad(reward, [coords, atomics], allow_unused=True)
    gc = torch.zeros_like(coords) if grads[0] is None else grads[0]
    ga = torch.zeros_like(atomics) if grads[1] is None else grads[1]
    return gc.detach(), ga.detach()


def sample_dps(model, init_state, *, args, targets, guide_reg) -> TensorDict:
    state = clone_state(init_state)
    device = state["coords"].device
    schedule = model._get_sample_schedule(args.sample_steps).to(device)
    B = state["coords"].shape[0]
    for i in range(1, len(schedule)):
        t_val = schedule[i - 1]
        t = t_val.expand(B)
        dt = (schedule[i] - schedule[i - 1]).expand(B)
        guide = (args.mu != 0.0 and (i - 1) % args.guide_every == 0
                 and args.guide_min_t <= float(t_val) <= args.guide_max_t)
        gc = ga = None
        if guide:
            gc, ga = dps_gradient(model, state, t, targets=targets,
                                  guide_reg=guide_reg, property_name=args.property_name)
            gc, _ = clip_guidance_rms(gc, state["padding_mask"], args.guidance_max_rms)
            ga, _ = clip_guidance_rms(ga, state["padding_mask"], args.guidance_max_rms)
        with torch.no_grad():
            state = production_step_with_optional_guidance(
                model, state, t, dt, mu=args.mu, grad_coords=gc, grad_atomics=ga)
    return state


@torch.no_grad()
def sample_fk(model, init_state, *, args, targets_rep, guide_reg, n_targets: int, K: int) -> TensorDict:
    """Feynman-Kac steering over K particles per target.

    ``init_state`` holds n_targets*K particles laid out target-major (particle k of
    target j at row j*K+k), so a view of (n_targets, K) groups particles correctly.
    At each resampling event the particles of a target are reweighted by
    exp(lambda * delta_r) on their lookahead endpoints and multinomially resampled.
    """
    state = clone_state(init_state)
    device = state["coords"].device
    schedule = model._get_sample_schedule(args.sample_steps).to(device)
    N = state["coords"].shape[0]
    prev_r = None

    def lookahead_reward(st, t):
        x1 = endpoint_prediction(model, st, t)
        pred = predicted_property(x1, property_name=args.property_name,
                                  property_regressor=guide_reg)
        return -(pred - targets_rep).square()

    for i in range(1, len(schedule)):
        t_val = schedule[i - 1]
        t = t_val.expand(N)
        dt = (schedule[i] - schedule[i - 1]).expand(N)

        in_window = args.guide_min_t <= float(t_val) <= args.guide_max_t
        do_resample = (in_window and args.fk_resample_every > 0
                       and (i - 1) % args.fk_resample_every == 0)
        if do_resample:
            r = lookahead_reward(state, t)
            if args.fk_weight == "delta" and prev_r is None:
                # First event: with no previous reward there is no increment to weight by.
                # Weighting by the ABSOLUTE reward here instead would apply a spread set by
                # the reward's scale rather than its change, which on QM9 (r ~ -100) collapses
                # the particles on the very first event. Record the baseline and skip.
                prev_r = r
                logw = torch.zeros_like(r)
            elif args.fk_weight == "delta":
                logw = args.fk_lambda * (r - prev_r)
                prev_r = r
            else:
                logw = args.fk_lambda * r
                prev_r = r
            logw = logw.view(n_targets, K)
            logw = logw - logw.max(dim=1, keepdim=True).values      # stabilize
            probs = torch.softmax(logw, dim=1)
            # Adaptive resampling. Resampling with replacement loses diversity even when the
            # weights are uniform -- K draws from K particles duplicate some and drop others --
            # so repeating it every few steps drives the population to one ancestor regardless
            # of lambda. Measured: at lambda = 0 this cost FK a factor of 2.5 against
            # best-of-N with the same K, to which it should be identical. Standard SMC
            # practice is to resample only once the effective sample size falls below a
            # fraction of K, which is what this does; ess_threshold = 0 restores the old
            # unconditional behaviour.
            ess = 1.0 / probs.square().sum(dim=1)                   # [n_targets], in [1, K]
            need = ess < float(getattr(args, "fk_ess_threshold", 0.5)) * K
            idx = torch.multinomial(probs, num_samples=K, replacement=True)  # [n_targets,K]
            keep = torch.arange(K, device=idx.device).unsqueeze(0).expand_as(idx)
            idx = torch.where(need.unsqueeze(1), idx, keep)
            flat = (torch.arange(n_targets, device=device).unsqueeze(1) * K + idx).reshape(-1)
            state = TensorDict(
                {"coords": state["coords"][flat],
                 "atomics": state["atomics"][flat],
                 "padding_mask": state["padding_mask"][flat]},
                batch_size=N,
            ).to(device)
            if prev_r is not None:
                prev_r = prev_r[flat]

        state = production_step_with_optional_guidance(
            model, state, t, dt, mu=0.0, grad_coords=None, grad_atomics=None)
    return state


# ---------------------------------------------------------------- selection


def select_best(cand_state, *, n_targets, K, targets, guide_reg, property_name,
                converter, pool, sanitize, mask_single):
    """Pick, per target, the best of K candidates by GUIDE error.

    Returns (raw_state, pbvalid_state, frac_with_valid, pb_checks). ``raw`` ignores
    validity; ``pbvalid`` is the lowest-error PoseBusters-valid candidate, falling
    back to the raw best when a target has none (PB checked lazily in error order,
    which gives the same choice as checking every candidate).
    """
    with torch.no_grad():
        pred = predicted_property(cand_state, property_name=property_name,
                                  property_regressor=guide_reg)
    err = (pred - targets.repeat_interleave(K)).abs().view(n_targets, K).float().cpu()
    coords = cand_state["coords"].view(n_targets, K, *cand_state["coords"].shape[1:]).cpu()
    atomics = cand_state["atomics"].view(n_targets, K, *cand_state["atomics"].shape[1:]).cpu()
    rows = torch.arange(n_targets)
    idx_raw = err.argmin(dim=1)
    raw = make_state(coords[rows, idx_raw], atomics[rows, idx_raw], mask_single.cpu())
    # lazy PB walk over the K candidates; the "endpoint" slot is the raw best with
    # +inf error, i.e. the fallback when nothing is valid
    fallback = make_state(coords[rows, idx_raw], atomics[rows, idx_raw], mask_single.cpu())
    n0 = pool.n_checked
    sel, resolved = lazy_select(
        err_la=err, t_la=torch.zeros_like(err), coords_la=coords, atomics_la=atomics,
        err_end=torch.full((n_targets,), float("inf")), final_state=fallback,
        mask=mask_single, converter=converter, sanitize=sanitize, pool=pool)
    sel = torch.where(sel == K, idx_raw, sel)
    pbv = make_state(coords[rows, sel], atomics[rows, sel], mask_single.cpu())
    dev = cand_state["coords"].device
    return raw.to(dev), pbv.to(dev), float(resolved.float().mean()), pool.n_checked - n0


# ---------------------------------------------------------------- main


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.hist_cache is None:
        args.hist_cache = paths.default_hist_cache(args.property_name, args.partition,
                                                   args.hist_bins)
    if args.posebusters_config is None:
        args.posebusters_config = paths.posebusters_config_path()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    model, data_stats, _ = load_flow_model(args.checkpoint, device)
    require_linear_dfm_atoms(model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    guide_reg = load_tfg_regressor("guide", args.property_name, device)
    oracle_reg = load_tfg_regressor("oracle", args.property_name, device)
    for reg in (guide_reg, oracle_reg):
        reg.soft_atom_temperature = args.soft_atom_temperature
        reg.eval()
        for prm in reg.parameters():
            prm.requires_grad_(False)
    nfe = attach_standard(NFECounter(), model=model, regressors=(guide_reg, oracle_reg))

    hist = build_or_load_histograms(args, max_len=int(data_stats["max_num_atoms"]))

    # --- identical target construction to steer_search.main() ---
    seed_all(args.seed)
    init_state = model._sample_noise_like_batch(batch_size=args.num_samples).to(device)
    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
    trng = np.random.default_rng(args.target_seed)
    target_values = [sample_property_target(hist, int(l), trng) for l in lengths]
    targets = torch.tensor(target_values, dtype=torch.float32, device=device)

    sanitize = not args.no_sanitize
    converter = MoleculeConverter(
        atom_names=list(data_stats["atom_names"]),
        dataset_normalizer=float(data_stats.get("coordinate_normalizer", 1.0)),
    )
    mask_single = init_state["padding_mask"]
    pool = PBPool(args.posebusters_config, args.pb_workers)
    summary, per_rows = [], {}
    ev = lambda st, run: evaluate_state(st, targets=targets, property_name=args.property_name,
                                        guide_reg=guide_reg, oracle_reg=oracle_reg,
                                        converter=converter, pool=pool, sanitize=sanitize, run=run)
    sel_checks = 0

    seed_all(args.seed)
    if args.method in ("base", "dps"):
        with nfe.phase("sampling"):
            if args.method == "base":
                state = sample_unguided(model, init_state, steps=args.sample_steps)
            else:
                state = sample_dps(model, init_state, args=args, targets=targets,
                                   guide_reg=guide_reg)
        with nfe.phase("evaluation"):
            row, per = ev(state, args.method)
        summary.append(row)
        per_rows[args.method] = per

    elif args.method in ("best_of_n", "fk"):
        K = args.n_candidates if args.method == "best_of_n" else args.n_particles
        # K candidates per target, sharing that target's atom count; fresh noise each.
        rep = repeat_state(init_state, K)
        rep = model._sample_noise_like_batch(rep).to(device)
        targets_rep = targets.repeat_interleave(K)
        with nfe.phase("sampling"):
            if args.method == "best_of_n":
                cand = sample_unguided(model, rep, steps=args.sample_steps,
                                       batch_size=args.batch_size)
            else:
                cand = sample_fk(model, rep, args=args, targets_rep=targets_rep,
                                 guide_reg=guide_reg, n_targets=args.num_samples, K=K)
        with nfe.phase("selection"):
            raw, pbv, frac, sel_checks = select_best(
                cand, n_targets=args.num_samples, K=K, targets=targets, guide_reg=guide_reg,
                property_name=args.property_name, converter=converter, pool=pool,
                sanitize=sanitize, mask_single=mask_single)
        for st, tag in ((raw, f"{args.method}_raw"), (pbv, f"{args.method}_pbvalid")):
            with nfe.phase("evaluation"):
                row, per = ev(st, tag)
            if tag.endswith("pbvalid"):
                row["frac_with_valid_candidate"] = frac
            summary.append(row)
            per_rows[tag] = per
    pool.close()

    B = args.num_samples
    steer = nfe.summary(B, phase="sampling")
    sel = nfe.summary(B, phase="selection")
    nfe_meta = {
        **steer,
        "nfe_regressor_selection": sel["nfe_regressor"],
        "pb_checks_per_mol": sel_checks / B,
        # legacy keys (manual counters of the 2026-09 sweep)
        "nfe_base_per_mol": steer["nfe_base"],
        "nfe_regressor_per_mol": steer["nfe_regressor"] + sel["nfe_regressor"],
        "nfe_backward_per_mol": steer["nfe_generative_bwd"],
        "nfe_total_per_mol": steer["nfe_base"] + steer["nfe_regressor"] + sel["nfe_regressor"],
        "by_phase": nfe.by_phase(B),
        "measured": "forward hooks (multimfm.nfe)",
    }
    for row in summary:
        row["nfe_generative_per_mol"] = steer["nfe_generative"]
        row["nfe_generative_bwd_per_mol"] = steer["nfe_generative_bwd"]
    meta = {
        "method": args.method,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary,
        "nfe": nfe_meta,
        "targets": [float(x) for x in target_values],
        "lengths": [int(x) for x in lengths],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_{args.property_name}_seed{args.seed}"
    with open(args.output_dir / f"{tag}_per_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "sample_idx", "target", "guide_prediction", "oracle_prediction",
                    "oracle_abs_error", "pb_valid", "smiles"])
        for run, per in per_rows.items():
            for i in range(B):
                o = per["oracle_prediction"][i]
                w.writerow([run, i, target_values[i], per["guide_prediction"][i], o,
                            abs(o - target_values[i]), int(per["pb_valid"][i]), per["smiles"][i] or ""])
    (args.output_dir / f"{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    for row in summary:
        print({k: row[k] for k in ("run", "oracle_mae", "pb_pb_valid") if k in row})
    print("NFE/molecule:", {k: v for k, v in nfe_meta.items() if k != "by_phase"})


if __name__ == "__main__":
    main()
