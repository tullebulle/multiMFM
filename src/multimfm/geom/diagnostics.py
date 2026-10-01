"""Diagnostics for the GEOM-Drugs fingerprint steering, for interpreting its results.

``grad-sign`` was written to answer the question the as-described rerun raises: when
steering *lowers* the fingerprint similarity, is the guidance term wired up with the wrong
sign, or is it correctly signed and simply unhelpful at this scale? It takes a
mid-trajectory state, estimates ``grad_x log E[exp(lambda r(X_1))]`` with the same MFM
look-ahead sampler the driver uses, then measures the *guide-predicted* reward of fresh
look-ahead endpoints at ``x`` and at ``x +- eps * grad`` for a few step sizes.

**It cannot actually answer that question, and its verdict string was changed on
2026-09-29 to say so.** Two measured reasons (docs/geom_guidance_diagnosis.md §2.2):
it re-draws the look-ahead noise between the gradient and the probes, so it probes a
different function from the one the gradient was taken of and reports no uncertainty;
and at its default steps every probe displaces the state by 6e3 to 6e6 times the value
function's measured linear radius, so which ray comes out ahead is decided by curvature.
The old verdict compared the single best probe against step 0, which is not a sign test at
all. It now reports the *paired* ``+/-`` comparison at matched ``|step|`` and the
displacement in units of the state RMS, and declares itself underpowered.
Use ``fd-check`` for the sign and magnitude, and
``scripts/geom/grad_sign_seeds.sbatch`` for this scan's own error bars.

    python -m multimfm.geom.diagnostics grad-sign --device cuda
    python -m multimfm.geom.diagnostics grad-sign --num-mols 4 --posterior-samples 4 \
        --mfm-steps 6 --device cpu

``fd-check`` is the stricter, *deterministic* companion added on 2026-09-29. ``grad-sign``
re-draws the look-ahead noise between the gradient and each probe, so it measures a noisy
function and cannot detect a wrong magnitude (or a small-but-correct gradient buried in
Monte Carlo noise). ``fd-check`` instead freezes the look-ahead prior noise, which makes
``V(x) = log mean_i exp(lambda r(x_1^i(x)))`` an ordinary deterministic function of ``x``,
and compares the coded gradient with a central finite difference of that same function.
Three stages, each a strict pass/fail:

1. ``reward``  -- ``d r / d coords`` of ``r = -||sigmoid(head(X_1)) - c||^2`` against a
   central FD of the *same* hard-atom-type function the evaluation scores.
2. ``soft``    -- the soft-atom-type relaxation used when ``guide_atomics`` is on: autograd
   vs FD, plus how far the relaxed prediction is from the hard one it stands in for.
3. ``value``   -- the whole chain: ``grad_x V_t(x)`` from :func:`value_gradient` against a
   central FD of ``V`` at frozen prior noise, and the size of the displacement the
   production step actually applies.

    python -m multimfm.geom.diagnostics fd-check --device cpu --num-mols 2

Both print a table and, with ``--out``, write a JSON record.

Note on the login node: the two commands load ~420 MB of checkpoints and peak at ~1.3 GB
RSS, so they run fine there (verified 2026-09-29); an earlier comment claiming the login
node OOM-kills them was wrong.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from multimfm import glass_guidance as engine
from multimfm.geom import fingerprint as fp
from multimfm.geom.paths import DEFAULT_STRUCTURE_HEAD, DEFAULT_STUDENT, DEFAULT_TEACHER
from multimfm.geom.reward import fingerprint_reward, install_fingerprint_reward
from multimfm.geom.teacher import load_geom_teacher
from multimfm.glass_guidance import load_mfm_student, value_gradient
from multimfm.train_base_flow import choose_device, set_seed
from tabasco.sample.flow_map import _prior_like


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grad-sign", help="directional check of the reward gradient")
    g.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    g.add_argument("--mfm-checkpoint", type=Path, default=DEFAULT_STUDENT)
    g.add_argument("--structure-head", type=Path, default=DEFAULT_STRUCTURE_HEAD)
    g.add_argument("--num-mols", type=int, default=8)
    g.add_argument("--t", type=float, default=0.5, help="trajectory time to probe")
    g.add_argument("--outer-steps", type=int, default=100,
                   help="outer ODE steps used to reach --t unguided")
    g.add_argument("--posterior-samples", type=int, default=4)
    g.add_argument("--mfm-steps", type=int, default=10)
    g.add_argument("--reward-lambda", type=float, default=0.02)
    g.add_argument("--steps", type=float, nargs="+", default=[0.1, 0.3, 1.0],
                   help="step sizes probed in +grad and -grad")
    g.add_argument("--probe-samples", type=int, default=16,
                   help="look-ahead endpoints per probe (fixed seed, so probes are paired)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    g.add_argument("--out", type=Path, default=None)

    f = sub.add_parser(
        "fd-check",
        help="deterministic finite-difference check of the guidance gradient",
    )
    f.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    f.add_argument("--mfm-checkpoint", type=Path, default=DEFAULT_STUDENT)
    f.add_argument("--structure-head", type=Path, default=DEFAULT_STRUCTURE_HEAD)
    f.add_argument("--num-mols", type=int, default=2)
    f.add_argument("--value-batch-size", type=int, default=1,
                   help="Chunk size for value_gradient (1 keeps peak memory small).")
    f.add_argument("--t", type=float, default=0.5, help="trajectory time to probe")
    f.add_argument("--outer-steps", type=int, default=100,
                   help="outer ODE steps used to reach --t unguided")
    f.add_argument("--posterior-samples", type=int, default=4)
    f.add_argument("--mfm-steps", type=int, default=10)
    f.add_argument("--reward-lambda", type=float, default=0.01)
    f.add_argument("--soft-atom-temperature", type=float, default=0.25)
    f.add_argument("--rel-steps", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2],
                   help="FD displacement sizes, as a fraction of the state's own RMS. "
                        "The gradient direction is normalised first, so these are the "
                        "*relative* size of the probe step, not a raw step length.")
    f.add_argument("--component-rel-step", type=float, default=1e-4,
                   help="Probe step for the componentwise check, as a fraction of the "
                        "coordinate RMS. Kept separate from --rel-steps because a "
                        "single-component difference hits fp32 cancellation far earlier "
                        "than a whole-gradient one.")
    f.add_argument("--components", type=int, default=12,
                   help="How many individual tensor entries to check componentwise.")
    f.add_argument("--mu", type=float, default=0.25,
                   help="mu the production run used, for the magnitude report.")
    f.add_argument("--guide-every", type=int, default=2)
    f.add_argument("--sample-steps", type=int, default=100)
    f.add_argument("--guide-min-t", type=float, default=0.1)
    f.add_argument("--guide-max-t", type=float, default=0.9)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    f.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = leave the default alone).")
    f.add_argument("--stages", nargs="+",
                   choices=["reward", "soft", "value", "magnitude"],
                   default=["reward", "soft", "value", "magnitude"],
                   help="Which stages to run. Dropping 'value' and 'magnitude' skips\n"
                        "loading the teacher and student entirely, so stages 1-2 "
                        "run in seconds inside the login node's 8 GB cap.")
    f.add_argument("--out", type=Path, default=None)

    q = sub.add_parser(
        "fp2-sensitivity",
        help="how large a coordinate displacement does FP2 even notice?",
    )
    q.add_argument("--num-mols", type=int, default=16)
    q.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                   help="isotropic coordinate noise, RMS per component, in the "
                        "model's normalized frame (multiply by 2.0 for Angstrom).")
    q.add_argument("--repeats", type=int, default=3)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def cmd_grad_sign(args: argparse.Namespace) -> dict:
    install_fingerprint_reward()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    model, stats, _ = load_geom_teacher(args.teacher_checkpoint, device, use_ema=True)
    head = fp.load_structure_head(args.structure_head, device)
    student = load_mfm_student(
        args.mfm_checkpoint, data_stats=stats,
        teacher_args=SimpleNamespace(hidden_dim=256, num_layers=16, num_heads=8),
        device=device, key="student",
    )
    max_atoms = int(stats.get("max_num_atoms", 71))
    B = args.num_mols
    targets = fp.sample_targets(n=B, max_atoms=max_atoms, seed=args.seed).to(device)

    # integrate the base model, unguided, up to --t
    set_seed(args.seed)
    state = model._sample_noise_like_batch(batch_size=B).to(device)
    schedule = model._get_sample_schedule(args.outer_steps).to(device)
    for i in range(1, len(schedule)):
        t0, t1 = float(schedule[i - 1]), float(schedule[i])
        if t0 >= args.t:
            break
        t_vec = torch.full((B,), t0, device=device)
        dt_vec = torch.full((B,), min(t1, args.t) - t0, device=device)
        with torch.no_grad():
            state = engine.production_step_with_optional_guidance(
                model, state, t_vec, dt_vec, mu=0.0, grad_coords=None, grad_atomics=None)

    t_vec = torch.full((B,), args.t, device=device)
    with torch.enable_grad():
        grad_coords, grad_atomics, gstats = value_gradient(
            model, state, t_vec, value_sampler="mfm", mfm_student=student,
            mfm_diagonal=False, posterior_samples=args.posterior_samples,
            n_steps=args.mfm_steps, reward_name="fingerprint",
            reward_scale=args.reward_lambda, target_x=0.0, property_name="structure",
            property_target=targets, property_regressor=head, guide_atomics=True,
            eps=1e-6, value_batch_size=min(B, 8),
        )

    def probe(step: float) -> float:
        """Mean guide-predicted reward of look-aheads from x + step * grad."""
        moved = TensorDict(
            {
                "coords": state["coords"] + step * grad_coords,
                "atomics": state["atomics"] + step * grad_atomics,
                "padding_mask": state["padding_mask"],
            },
            batch_size=B,
        ).to(device)
        set_seed(args.seed + 991)  # same look-ahead noise for every probe
        with torch.no_grad():
            post = engine.differentiable_mfm_posterior(
                student, moved, t_vec, posterior_samples=args.probe_samples,
                n_steps=args.mfm_steps, diagonal=False)
            tgt = targets.repeat_interleave(args.probe_samples, dim=0)
            return float(fingerprint_reward(post, head, tgt).mean())

    base = probe(0.0)
    probes = {0.0: base}
    for s in args.steps:
        probes[float(s)] = probe(float(s))
        probes[-float(s)] = probe(-float(s))

    ordered = sorted(probes)
    print(f"\n# grad rms: coords {gstats['grad_coords_rms']:.4g} "
          f"atomics {gstats.get('grad_atomics_rms', float('nan')):.4g}")
    print(f"# probing at t={args.t}, n={B} molecules, lambda={args.reward_lambda}, "
          f"{args.probe_samples} look-aheads per probe")
    print(f"\n{'step':>8}{'guide reward':>15}{'delta':>12}")
    for s in ordered:
        print(f"{s:>8.3f}{probes[s]:>15.4f}{probes[s] - base:>+12.4f}")
    # The only defensible reading of this scan is the PAIRED +/- comparison at matched
    # |step|: a correctly signed gradient must make delta(+step) - delta(-step) positive.
    # Comparing the best single probe against step 0, as an earlier version of this verdict
    # did, is not a sign test -- the surface is strongly curved, so at large |step| either
    # ray can come out ahead for reasons that have nothing to do with the gradient.
    state_rms = float(torch.sqrt(
        ((state["coords"] * (~state["padding_mask"]).unsqueeze(-1)) ** 2).sum()
        / ((~state["padding_mask"]).sum() * 3).clamp_min(1)))
    grad_rms = gstats["grad_coords_rms"]
    paired = []
    for s in [x for x in ordered if x > 0]:
        if -s not in probes:
            continue
        paired.append({
            "abs_step": s,
            "delta_plus": probes[s] - base,
            "delta_minus": probes[-s] - base,
            "plus_minus_advantage": probes[s] - probes[-s],
            # how far the probe moves, in units of the state's own coordinate RMS
            "displacement_over_state_rms": s * grad_rms / max(state_rms, 1e-12),
        })
    print(f"\n{'|step|':>8}{'delta(+)':>11}{'delta(-)':>11}{'(+)-(-)':>11}"
          f"{'displ/state rms':>17}")
    for r in paired:
        print(f"{r['abs_step']:>8.3f}{r['delta_plus']:>+11.4f}{r['delta_minus']:>+11.4f}"
              f"{r['plus_minus_advantage']:>+11.4f}"
              f"{r['displacement_over_state_rms']:>17.4g}")
    n_fav = sum(1 for r in paired if r["plus_minus_advantage"] > 0)
    verdict = (
        f"UNDERPOWERED AS A SIGN TEST: {n_fav} of {len(paired)} matched +/- pairs favour "
        f"+grad. This scan re-draws the look-ahead noise between the gradient and the "
        f"probes and reports no uncertainty, and every step above displaces the state by "
        f"far more than the value function's measured linear radius (~1e-6 of the state "
        f"RMS), so its ordering cannot establish the sign either way. Use "
        f"`fd-check` for the sign and magnitude, and "
        f"`scripts/geom/grad_sign_seeds.sbatch` for this scan's own error bars. See "
        f"docs/geom_guidance_diagnosis.md."
    )
    print(f"\nverdict: {verdict}")

    record = {
        "_what": "directional check of the fingerprint value gradient",
        "t": args.t, "num_mols": B, "reward_lambda": args.reward_lambda,
        "posterior_samples": args.posterior_samples, "mfm_steps": args.mfm_steps,
        "probe_samples": args.probe_samples, "seed": args.seed,
        "grad_coords_rms": gstats["grad_coords_rms"],
        "grad_atomics_rms": gstats.get("grad_atomics_rms"),
        "guide_reward_by_step": {str(s): probes[s] for s in ordered},
        "paired_plus_minus": paired,
        "state_coords_rms": state_rms,
        "verdict": verdict,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")
    return record


# --------------------------------------------------------------------------- #
# fd-check: deterministic finite-difference verification of the guidance term
# --------------------------------------------------------------------------- #
def _rms(x: torch.Tensor, mask: torch.Tensor) -> float:
    """RMS of ``x`` over the real (non-padded) entries only."""
    m = (~mask).unsqueeze(-1).to(x.dtype)
    denom = (m.sum() * x.shape[-1]).clamp_min(1.0)
    return float(torch.sqrt(((x * m) ** 2).sum() / denom))


def _masked_state(coords, atomics, padding_mask) -> TensorDict:
    return TensorDict(
        {"coords": coords, "atomics": atomics, "padding_mask": padding_mask},
        batch_size=padding_mask.shape[0],
    )


def _dataset_state(B: int, max_atoms: int, seed: int, device):
    """``B`` real GEOM val molecules as a state, plus ``B`` *other* molecules' FP2.

    Head-only evaluation point: one-hot atom types, mean-centred and /normalizer
    coordinates -- exactly the distribution the structure head was trained on. The
    targets are drawn from a disjoint slice of the same permutation, so the reward
    is the real steering objective (match another molecule's fingerprint) rather
    than a self-match with zero gradient.
    """
    fp.ensure_tfg_importable()
    from dataloader.datasets.geom_fingerprint import GEOMFingerprintDataset

    ds = GEOMFingerprintDataset(str(fp.DEFAULT_GEOM_FP_DIR), split="val",
                                max_len=max_atoms)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    picked, targets = [], []
    for i in order:
        rec = ds[i]
        if int(rec["mask"].sum()) > max_atoms:
            continue
        picked.append(rec)
        if len(picked) == 2 * B:
            break
    src, tgt = picked[:B], picked[B:2 * B]
    coords = torch.stack([r["coors"].float() for r in src]).to(device)
    mask = torch.stack([r["mask"].bool() for r in src]).to(device)
    atom_types = torch.stack([r["atom_types"].long() for r in src]).to(device)
    atomics = torch.nn.functional.one_hot(
        atom_types, num_classes=len(fp.GEOM_ATOM_NAMES)).float()
    targets = torch.stack([r["structure"].float() for r in tgt]).to(device)
    return (_masked_state(coords, atomics, ~mask).to(device), targets)


def _unguided_state_at(model, B: int, t: float, outer_steps: int, device, seed: int):
    """Integrate the base model, unguided, up to time ``t`` and return the state."""
    set_seed(seed)
    state = model._sample_noise_like_batch(batch_size=B).to(device)
    schedule = model._get_sample_schedule(outer_steps).to(device)
    for i in range(1, len(schedule)):
        t0, t1 = float(schedule[i - 1]), float(schedule[i])
        if t0 >= t:
            break
        t_vec = torch.full((B,), t0, device=device)
        dt_vec = torch.full((B,), min(t1, t) - t0, device=device)
        with torch.no_grad():
            state = engine.production_step_with_optional_guidance(
                model, state, t_vec, dt_vec, mu=0.0,
                grad_coords=None, grad_atomics=None)
    return state


def _best_ratio(rows: list[dict]) -> dict:
    """The probe step whose FD ratio is closest to 1.

    A central difference has two error sources pulling in opposite directions:
    O(h^2) truncation, which shrinks with h, and cancellation round-off, which grows
    as 1/h (in fp32 the smallest probes below can give an FD of exactly 0). So the
    right question is never "what does the smallest h give" but "is there a step at
    which the difference matches the gradient". A correct gradient has such a step;
    an incorrect one has none at any h.
    """
    return min(rows, key=lambda r: abs(r["ratio"] - 1.0))


def _fd_report(name: str, rows: list[dict]) -> None:
    print(f"\n  {name}")
    print(f"    {'rel step':>10}{'h':>12}{'FD slope':>14}"
          f"{'|grad|^2':>14}{'ratio':>9}{'  above noise':>13}")
    for r in rows:
        flag = ("" if "above_noise_floor" not in r
                else ("  yes" if r["above_noise_floor"] else "  .  (noise)"))
        print(f"    {r['rel_step']:>10.1e}{r['h']:>12.4e}{r['fd_slope']:>14.6e}"
              f"{r['grad_sq']:>14.6e}{r['ratio']:>9.4f}{flag}")


def cmd_fd_check(args: argparse.Namespace) -> dict:
    """Check the coded guidance gradient against a central finite difference.

    Everything is made deterministic first: the look-ahead prior noise is drawn once
    and reused, so ``V_t(x) = log mean_i exp(lambda r(x_1^i(x)))`` is an ordinary
    function of ``x`` and a central difference of it is exact up to O(h^2). A
    correctly wired ascent direction must give
    ``(V(x + h g) - V(x - h g)) / (2 h) = |g|^2 > 0``, i.e. ratio -> 1 as h -> 0.
    A ratio of about -1 is a sign inversion; a ratio far from +-1 that does not
    stabilise as h shrinks is a scale or plumbing defect.
    """
    install_fingerprint_reward()
    if args.threads > 0:
        torch.set_num_threads(int(args.threads))
    set_seed(args.seed)
    # Exact fp32, NOT the production "high" (TF32) setting. A central difference of a
    # function evaluated in TF32 is hopeless here: TF32 carries ~10 mantissa bits, and
    # the 10-step flow-map rollout compounds that, so V is only reproducible to ~1%
    # between calls while the linear FD signal at a probe step small enough to *be*
    # linear is of the same order. Measured on a CUDA run with TF32 on
    # (results/geom/diagnostics/fd_check_t0.5_lam0.01_cuda.json): the stage-3 FD slope
    # scaled as 1/h across three step sizes, i.e. it was pure noise. See
    # docs/geom_guidance_diagnosis.md §3.4.
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = choose_device(args.device)

    need_chain = bool({"value", "magnitude"} & set(args.stages))
    head = fp.load_structure_head(
        args.structure_head, device, args.soft_atom_temperature)
    B = args.num_mols
    P = args.posterior_samples if need_chain else 1
    lam = args.reward_lambda

    record: dict = {
        "_what": "deterministic finite-difference check of the GEOM fingerprint "
                 "guidance gradient",
        "stages": list(args.stages),
        "t": args.t, "num_mols": B, "posterior_samples": P, "reward_lambda": lam,
        "mfm_steps": args.mfm_steps, "outer_steps": args.outer_steps,
        "soft_atom_temperature": args.soft_atom_temperature,
        "seed": args.seed, "device": str(device),
    }

    if need_chain:
        model, stats, _ = load_geom_teacher(
            args.teacher_checkpoint, device, use_ema=True)
        student = load_mfm_student(
            args.mfm_checkpoint, data_stats=stats,
            teacher_args=SimpleNamespace(hidden_dim=256, num_layers=16, num_heads=8),
            device=device, key="student",
        )
        max_atoms = int(stats.get("max_num_atoms", 71))
        normalizer = float(stats.get("coordinate_normalizer", 2.0))
        targets = fp.sample_targets(n=B, max_atoms=max_atoms, seed=args.seed).to(device)
        state = _unguided_state_at(
            model, B, args.t, args.outer_steps, device, args.seed)
        mask = state["padding_mask"]
        record["evaluation_point"] = (
            "MFM look-ahead endpoint from an unguided base-model state at t")

        # Frozen look-ahead noise: makes the whole chain a deterministic map.
        set_seed(args.seed + 4242)
        prior_noise = _prior_like(engine.repeat_state(state, P))

        def posterior_at(coords, atomics):
            """Look-ahead endpoints from (coords, atomics) at the frozen prior noise.

            Always called under ``torch.no_grad()``: the finite differences need only
            the function value, and the login-node memory cap (8 GB) does not allow
            holding the rollout graph as well as the autograd pass below.
            """
            cond = _masked_state(coords, atomics, mask).to(device)
            t_vec = torch.full((B,), args.t, device=device)
            return engine.differentiable_mfm_posterior(
                student, cond, t_vec, posterior_samples=P,
                n_steps=args.mfm_steps, diagonal=False, prior_noise=prior_noise)

        # The endpoint state at which the reward is actually evaluated.
        with torch.no_grad():
            endpoint = posterior_at(state["coords"].clone(), state["atomics"].clone())
        endpoint_targets = targets.repeat_interleave(P, dim=0)
    else:
        # Head-only mode: evaluate the reward gradient on real GEOM molecules, which
        # needs neither the teacher nor the student and so fits the login node.
        model = student = None
        normalizer, max_atoms = 2.0, 71
        endpoint, endpoint_targets = _dataset_state(B, max_atoms, args.seed, device)
        state, mask = endpoint, endpoint["padding_mask"]
        targets = endpoint_targets
        record["evaluation_point"] = (
            "real GEOM-Drugs val molecules (one-hot atom types), head-only mode")
    record["state_coords_rms"] = _rms(state["coords"], mask)
    record["state_atomics_rms"] = _rms(state["atomics"], mask)
    record["n_real_atoms"] = [int(v) for v in (~mask).sum(dim=1).tolist()]
    ep_mask = endpoint["padding_mask"]
    record["endpoint_coords_rms"] = _rms(endpoint["coords"], ep_mask)
    record["endpoint_atomics_rms"] = _rms(endpoint["atomics"], ep_mask)
    record["endpoint_atomics_row_sum_mean"] = float(
        endpoint["atomics"][~ep_mask].sum(dim=-1).mean())
    record["endpoint_atomics_max_mean"] = float(
        endpoint["atomics"][~ep_mask].max(dim=-1).values.mean())

    # ------------------------------------------------------------------ #
    # Stage 1: d r / d coords of the *hard* reward the evaluation scores.
    # ------------------------------------------------------------------ #
    def r_hard(coords) -> torch.Tensor:
        st = _masked_state(coords, endpoint["atomics"], ep_mask)
        return fingerprint_reward(st, head, endpoint_targets, soft=False)

    c0 = endpoint["coords"].detach().clone().requires_grad_(True)
    r0 = r_hard(c0)
    (g1,) = torch.autograd.grad(r0.sum(), [c0])
    g1 = engine.mask_and_zero_com(g1, ep_mask).detach()
    g1_rms = _rms(g1, ep_mask)
    dirn = g1 / max(g1_rms, 1e-30)               # unit-RMS ascent direction
    grad_sq = float((g1 * dirn).sum())           # = |g|^2 / |g|_rms
    scale = record["endpoint_coords_rms"]
    rows1 = []
    for rel in args.rel_steps:
        h = rel * scale
        with torch.no_grad():
            rp = float(r_hard(c0.detach() + h * dirn).sum())
            rm = float(r_hard(c0.detach() - h * dirn).sum())
        fd = (rp - rm) / (2 * h)
        rows1.append({"rel_step": rel, "h": h, "fd_slope": fd,
                      "grad_sq": grad_sq, "ratio": fd / grad_sq if grad_sq else float("nan")})
    # componentwise check on a few real entries
    gen = torch.Generator().manual_seed(args.seed + 7)
    real = (~ep_mask).nonzero(as_tuple=False)
    picks = real[torch.randperm(real.shape[0], generator=gen)[: args.components]]
    # fp32 cancellation: a single-component difference is ~|g| * 2h against an
    # absolute reward of order 10^3, so h must stay well above sqrt(eps) ~ 3e-4
    # relative. Fixed at 1e-4 of the coordinate RMS rather than taken from
    # --rel-steps, which the directional sweep drives as low as 1e-8.
    h_c = args.component_rel_step * scale
    comp = []
    for b, i in picks.tolist():
        d = int(torch.randint(0, 3, (1,), generator=gen))
        e = torch.zeros_like(c0)
        e[b, i, d] = 1.0
        with torch.no_grad():
            fd = (float(r_hard(c0.detach() + h_c * e).sum())
                  - float(r_hard(c0.detach() - h_c * e).sum())) / (2 * h_c)
        comp.append({"b": b, "atom": i, "dim": d,
                     "autograd": float(g1[b, i, d]), "fd": fd})
    med_rel_err = float(torch.median(torch.tensor(
        [abs(c["fd"] - c["autograd"]) / max(abs(c["autograd"]), 1e-12) for c in comp])))
    record["stage1_reward_coords"] = {
        "grad_rms": g1_rms, "reward_mean": float(r0.mean()),
        "directional": rows1, "componentwise": comp,
        "componentwise_median_rel_err": med_rel_err,
        "pass": bool(0.9 < _best_ratio(rows1)["ratio"] < 1.1),
        "best_ratio": _best_ratio(rows1),
    }

    # ------------------------------------------------------------------ #
    # Stage 2: the soft atom-type relaxation used when guide_atomics is on.
    # ------------------------------------------------------------------ #
    from multimfm.geom.reward import fingerprint_prediction
    with torch.no_grad():
        pred_hard = fingerprint_prediction(
            _masked_state(endpoint["coords"], endpoint["atomics"], ep_mask),
            head, soft=False)
    a_soft = endpoint["atomics"].detach().clone().requires_grad_(True)
    pred_soft = fingerprint_prediction(
        _masked_state(endpoint["coords"].detach(), a_soft, ep_mask), head, soft=True)
    r_soft = -((pred_soft - endpoint_targets) ** 2).sum(dim=-1)
    (g2,) = torch.autograd.grad(r_soft.sum(), [a_soft])
    g2 = engine.apply_mask(g2, ep_mask).detach()
    g2_rms = _rms(g2, ep_mask)

    def r_soft_at(atomics) -> torch.Tensor:
        st = _masked_state(endpoint["coords"].detach(), atomics, ep_mask)
        return fingerprint_reward(st, head, endpoint_targets, soft=True)

    dirn2 = g2 / max(g2_rms, 1e-30)
    grad_sq2 = float((g2 * dirn2).sum())
    scale2 = record["endpoint_atomics_rms"]
    rows2, hard_follow = [], []
    for rel in args.rel_steps:
        h = rel * scale2
        with torch.no_grad():
            rp = float(r_soft_at(a_soft.detach() + h * dirn2).sum())
            rm = float(r_soft_at(a_soft.detach() - h * dirn2).sum())
        fd = (rp - rm) / (2 * h)
        rows2.append({"rel_step": rel, "h": h, "fd_slope": fd, "grad_sq": grad_sq2,
                      "ratio": fd / grad_sq2 if grad_sq2 else float("nan")})
    # does a step along the *soft* gradient also raise the *hard* reward?
    for rel in (1e-2, 1e-1, 1.0):
        h = rel * scale2
        with torch.no_grad():
            a_p = engine.apply_mask(a_soft.detach() + h * dirn2, ep_mask)
            a_m = engine.apply_mask(a_soft.detach() - h * dirn2, ep_mask)
            rhp = float(fingerprint_reward(
                _masked_state(endpoint["coords"], a_p, ep_mask),
                head, endpoint_targets, soft=False).sum())
            rhm = float(fingerprint_reward(
                _masked_state(endpoint["coords"], a_m, ep_mask),
                head, endpoint_targets, soft=False).sum())
            r_h0 = float(fingerprint_reward(
                _masked_state(endpoint["coords"], endpoint["atomics"], ep_mask),
                head, endpoint_targets, soft=False).sum())
            changed = int((a_p.argmax(-1) != endpoint["atomics"].argmax(-1))[~ep_mask].sum())
        hard_follow.append({"rel_step": rel, "hard_reward_plus": rhp,
                            "hard_reward_minus": rhm, "hard_reward_base": r_h0,
                            "argmax_changed": changed,
                            "n_real": int((~ep_mask).sum())})
    record["stage2_soft_atomics"] = {
        "grad_rms": g2_rms,
        "reward_soft_mean": float(r_soft.detach().mean()),
        "reward_hard_mean": float(r0.detach().mean()),
        "pred_soft_vs_hard_l2": float((pred_soft.detach() - pred_hard).norm(dim=-1).mean()),
        "pred_soft_mean": float(pred_soft.detach().mean()),
        "pred_hard_mean": float(pred_hard.mean()),
        "directional": rows2, "hard_follow": hard_follow,
        "pass": bool(0.9 < _best_ratio(rows2)["ratio"] < 1.1),
        "best_ratio": _best_ratio(rows2),
    }

    if need_chain:
        # ------------------------------------------------------------------ #
        # Stage 3: the full chain, grad_x V_t(x), at frozen look-ahead noise.
        # ------------------------------------------------------------------ #
        t_vec = torch.full((B,), args.t, device=device)
        with torch.enable_grad():
            gc, ga, gstats = value_gradient(
                model, state, t_vec, value_sampler="mfm", mfm_student=student,
                mfm_diagonal=False, posterior_samples=P, n_steps=args.mfm_steps,
                reward_name="fingerprint", reward_scale=lam, target_x=0.0,
                property_name="structure", property_target=targets,
                property_regressor=head, guide_atomics=True, eps=1e-6,
                prior_noise=prior_noise, value_batch_size=args.value_batch_size,
            )

        tgt_rep = targets.repeat_interleave(P, dim=0)

        def value_at(coords, atomics) -> float:
            """``sum_b V_t(x_b)`` at the frozen prior noise, soft atom-type path.

            ``soft=True`` is forced so this is the identical function that
            :func:`value_gradient` differentiates (there ``atomics.requires_grad`` is
            True, which selects the soft path); under ``no_grad`` the auto-detection
            would silently switch to the hard path and the FD would test a
            *different* function.
            """
            with torch.no_grad():
                post = posterior_at(coords.detach().clone(), atomics.detach().clone())
                rw = fingerprint_reward(post, head, tgt_rep, soft=True).view(B, P)
                v = torch.logsumexp(lam * rw, dim=-1) - math.log(P)
            return float(v.sum())

        # Noise floor: V must be reproducible between calls, or no finite difference of
        # it means anything. Repeated evaluations at the SAME x; the spread bounds the
        # smallest |V(x+hg) - V(x-hg)| that can be believed.
        v_repeats = [value_at(state["coords"], state["atomics"]) for _ in range(3)]
        v_spread = max(v_repeats) - min(v_repeats)
        record["stage3_value_noise_floor"] = {
            "value_repeats": v_repeats, "value_spread": v_spread,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }

        gc_rms, ga_rms = _rms(gc, mask), _rms(ga, mask)
        joint_rms = math.sqrt(0.5 * (gc_rms ** 2 + ga_rms ** 2))
        dc, da = gc / max(joint_rms, 1e-30), ga / max(joint_rms, 1e-30)
        grad_sq3 = float((gc * dc).sum() + (ga * da).sum())
        scale3 = math.sqrt(
            0.5 * (record["state_coords_rms"] ** 2 + record["state_atomics_rms"] ** 2))
        rows3 = []
        for rel in args.rel_steps:
            h = rel * scale3
            vp = value_at(state["coords"] + h * dc, state["atomics"] + h * da)
            vm = value_at(state["coords"] - h * dc, state["atomics"] - h * da)
            fd = (vp - vm) / (2 * h)
            rows3.append({"rel_step": rel, "h": h, "fd_slope": fd, "grad_sq": grad_sq3,
                          "ratio": fd / grad_sq3 if grad_sq3 else float("nan"),
                          "delta_v": vp - vm, "expected_delta_v": 2 * h * grad_sq3,
                          "above_noise_floor": abs(vp - vm) > 3 * v_spread})
        # coords-only variant, to separate the two modalities
        rows3c = []
        dc_only = gc / max(gc_rms, 1e-30)
        grad_sq3c = float((gc * dc_only).sum())
        for rel in args.rel_steps:
            h = rel * record["state_coords_rms"]
            vp = value_at(state["coords"] + h * dc_only, state["atomics"])
            vm = value_at(state["coords"] - h * dc_only, state["atomics"])
            fd = (vp - vm) / (2 * h)
            rows3c.append({"rel_step": rel, "h": h, "fd_slope": fd, "grad_sq": grad_sq3c,
                           "ratio": fd / grad_sq3c if grad_sq3c else float("nan"),
                           "delta_v": vp - vm, "expected_delta_v": 2 * h * grad_sq3c,
                           "above_noise_floor": abs(vp - vm) > 3 * v_spread})
        record["stage3_value_gradient"] = {
            "grad_coords_rms": gc_rms, "grad_atomics_rms": ga_rms,
            "value_mean": gstats["value_mean"],
            "posterior_reward_mean": gstats["posterior_reward_mean"],
            "posterior_reward_std": gstats["posterior_reward_std"],
            "value_at_x": value_at(state["coords"], state["atomics"]),
            "directional_joint": rows3, "directional_coords_only": rows3c,
            "best_ratio_joint": _best_ratio(
                [r for r in rows3 if r["above_noise_floor"]] or rows3),
            "best_ratio_coords_only": _best_ratio(
                [r for r in rows3c if r["above_noise_floor"]] or rows3c),
            "pass": bool(
                any(r["above_noise_floor"] for r in rows3)
                and 0.9 < _best_ratio(
                    [r for r in rows3 if r["above_noise_floor"]])["ratio"] < 1.1
            ),
        }

        # ------------------------------------------------------------------ #
        # Stage 4: how big is the displacement the production step applies?
        # ------------------------------------------------------------------ #
        dt = 1.0 / args.sample_steps
        n_guided = max(
            1,
            int((args.guide_max_t - args.guide_min_t) * args.sample_steps / args.guide_every),
        )
        per_step_c = args.mu * gc_rms * dt
        per_step_a = args.mu * ga_rms * dt
        from multimfm.geom.mu_schedule import paper_coefficient
        coeff = float(paper_coefficient(torch.tensor(args.t)))
        record["stage4_magnitude"] = {
            "mu": args.mu, "dt": dt, "guide_every": args.guide_every,
            "n_guided_steps_in_window": n_guided,
            "paper_coefficient_at_t": coeff,
            "coord_displacement_per_guided_step_rms": per_step_c,
            "coord_displacement_per_guided_step_over_state_rms":
                per_step_c / max(record["state_coords_rms"], 1e-30),
            "coord_displacement_coherent_total_rms": per_step_c * n_guided,
            "coord_displacement_coherent_total_angstrom":
                per_step_c * n_guided * normalizer,
            "atomics_displacement_per_guided_step_rms": per_step_a,
            "atomics_displacement_coherent_total_rms": per_step_a * n_guided,
            "atomics_displacement_total_over_state_rms":
                per_step_a * n_guided / max(record["state_atomics_rms"], 1e-30),
            "paper_schedule_would_scale_by": coeff,
        }

    # ------------------------------------------------------------------ #
    print("\n================ fd-check ================")
    print(f"  t={args.t}  n_mols={B}  posterior={P}  mfm_steps={args.mfm_steps}  "
          f"lambda={lam}  device={device}")
    print(f"  state rms: coords {record['state_coords_rms']:.4f} "
          f"atomics {record['state_atomics_rms']:.4f}")
    print(f"  reward evaluated at: {record['evaluation_point']}")
    print(f"  evaluation-point rms: coords {record['endpoint_coords_rms']:.4f} "
          f"atomics {record['endpoint_atomics_rms']:.4f} "
          f"(row sum {record['endpoint_atomics_row_sum_mean']:.3f}, "
          f"max {record['endpoint_atomics_max_mean']:.3f})")

    s1 = record["stage1_reward_coords"]
    print(f"\n[1] d r / d coords, hard atom types (the reward the metric scores)"
          f"   grad rms {s1['grad_rms']:.4g}  r mean {s1['reward_mean']:.3f}")
    _fd_report("directional (ratio -> 1 means correct sign AND magnitude)",
               s1["directional"])
    print(f"    best ratio {s1['best_ratio']['ratio']:.4f} at rel step "
          f"{s1['best_ratio']['rel_step']:.0e}")
    print(f"    componentwise median relative error: "
          f"{s1['componentwise_median_rel_err']:.3e} over {len(s1['componentwise'])} entries")
    print(f"    => {'PASS' if s1['pass'] else 'FAIL'}")

    s2 = record["stage2_soft_atomics"]
    print(f"\n[2] d r / d atomics, soft relaxation (temperature "
          f"{args.soft_atom_temperature})   grad rms {s2['grad_rms']:.4g}")
    print(f"    r_soft {s2['reward_soft_mean']:.3f} vs r_hard "
          f"{s2['reward_hard_mean']:.3f};  ||p_soft - p_hard||_2 = "
          f"{s2['pred_soft_vs_hard_l2']:.4f}  "
          f"(mean p_soft {s2['pred_soft_mean']:.4f}, p_hard {s2['pred_hard_mean']:.4f})")
    _fd_report("directional on r_soft", s2["directional"])
    print(f"    best ratio {s2['best_ratio']['ratio']:.4f} at rel step "
          f"{s2['best_ratio']['rel_step']:.0e}")
    print(f"    {'rel step':>10}{'hard r(-)':>14}{'hard r(0)':>14}"
          f"{'hard r(+)':>14}{'argmax flips':>14}")
    for r in s2["hard_follow"]:
        print(f"    {r['rel_step']:>10.1e}{r['hard_reward_minus']:>14.3f}"
              f"{r['hard_reward_base']:>14.3f}{r['hard_reward_plus']:>14.3f}"
              f"{r['argmax_changed']:>8d}/{r['n_real']:<5d}")
    print(f"    => {'PASS' if s2['pass'] else 'FAIL'}")

    if not need_chain:
        print("\n[3] [skipped: --stages did not include 'value']")
        print("[4] [skipped: --stages did not include 'magnitude']")
        verdict = (
            "reward gradient VERIFIED (stages 1-2); the full chain was not run"
            if s1["pass"] else
            "reward gradient FAILS the finite-difference check (stage 1)"
        )
        record["verdict"] = verdict
        print(f"\nverdict: {verdict}\n")
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record, indent=2) + "\n")
            print(f"wrote {args.out}")
        return record

    s3 = record["stage3_value_gradient"]
    print(f"\n[3] grad_x V_t(x), full chain at frozen look-ahead noise"
          f"   grad rms coords {s3['grad_coords_rms']:.4g} atomics "
          f"{s3['grad_atomics_rms']:.4g}")
    print(f"    V(x) = {s3['value_at_x']:.6f}  (mean per molecule "
          f"{s3['value_mean']:.6f}), posterior reward "
          f"{s3['posterior_reward_mean']:.3f} +- {s3['posterior_reward_std']:.3f}")
    nf = record["stage3_value_noise_floor"]
    print(f"    noise floor: V at the SAME x over 3 calls spans {nf['value_spread']:.3e} "
          f"(precision {nf['float32_matmul_precision']}); rows whose |delta V| is not at "
          f"least 3x that are marked '.'")
    _fd_report("directional, coords + atomics", s3["directional_joint"])
    _fd_report("directional, coords only", s3["directional_coords_only"])
    print(f"    best ratio: joint {s3['best_ratio_joint']['ratio']:.4f} at rel step "
          f"{s3['best_ratio_joint']['rel_step']:.0e}; coords only "
          f"{s3['best_ratio_coords_only']['ratio']:.4f} at rel step "
          f"{s3['best_ratio_coords_only']['rel_step']:.0e}")
    print(f"    => {'PASS' if s3['pass'] else 'FAIL'}")

    s4 = record["stage4_magnitude"]
    print(f"\n[4] displacement the production step applies (mu={s4['mu']}, "
          f"dt={s4['dt']}, {s4['n_guided_steps_in_window']} guided steps)")
    print(f"    coords: {s4['coord_displacement_per_guided_step_rms']:.4g} per step "
          f"= {100 * s4['coord_displacement_per_guided_step_over_state_rms']:.4f}% of "
          f"the state rms; coherent total "
          f"{s4['coord_displacement_coherent_total_rms']:.4g} "
          f"({s4['coord_displacement_coherent_total_angstrom']:.4g} A)")
    print(f"    atomics: {s4['atomics_displacement_per_guided_step_rms']:.4g} per step; "
          f"coherent total {s4['atomics_displacement_coherent_total_rms']:.4g} "
          f"= {100 * s4['atomics_displacement_total_over_state_rms']:.3f}% of the "
          f"atom-state rms")
    print(f"    the paper's (1-t)/t coefficient at t={args.t} would multiply this by "
          f"{s4['paper_coefficient_at_t']:.3f}")

    lin = [r["rel_step"] for r in s3["directional_joint"] if 0.9 < r["ratio"] < 1.1]
    record["stage3_value_gradient"]["largest_linear_rel_step"] = max(lin) if lin else None
    print(f"    largest probe step still linear (ratio within 10% of 1): "
          f"{record['stage3_value_gradient']['largest_linear_rel_step']}")
    print(f"    the production step displaces "
          f"{s4['coord_displacement_per_guided_step_over_state_rms']:.4g} of the state rms "
          f"per guided step -- compare that to the number above")

    verdict = (
        "gradient VERIFIED correct in sign and magnitude"
        if s1["pass"] and s2["pass"] and s3["pass"]
        else "gradient FAILS the finite-difference check -- see the failing stage"
    )
    record["verdict"] = verdict
    print(f"\nverdict: {verdict}\n")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")
    return record


def cmd_fp2_sensitivity(args: argparse.Namespace) -> dict:
    """Tanimoto of a molecule's FP2 to its own FP2 after a coordinate perturbation.

    FP2 is a *graph* fingerprint: bonds are re-perceived from the coordinates with
    OpenBabel's ``ConnectTheDots`` before any bit is set, so FP2 is piecewise
    constant in the coordinates and completely blind to displacements below the
    bond-perception thresholds. This measures where that floor is, which is what
    decides whether a guidance displacement of a given size can move Table 3's
    metric at all -- in either direction.

    Uses real GEOM val molecules, so it characterises the metric, not the model,
    and needs neither the teacher, the student nor the head.
    """
    set_seed(args.seed)
    state, _ = _dataset_state(args.num_mols, 71, args.seed, torch.device("cpu"))
    base_fps, base_valid, n_atoms = fp.state_to_binary_fps(state, 2.0)
    keep = base_valid
    rows = []
    g = torch.Generator().manual_seed(args.seed + 31)
    for sigma in args.sigmas:
        sims, changed = [], []
        for _ in range(args.repeats):
            noise = torch.randn(state["coords"].shape, generator=g) * sigma
            noise = engine.mask_and_zero_com(noise, state["padding_mask"])
            moved = _masked_state(state["coords"] + noise, state["atomics"],
                                 state["padding_mask"])
            fps, valid, _ = fp.state_to_binary_fps(moved, 2.0)
            both = keep & valid
            sim = fp.tanimoto_similarity(fps, base_fps)
            sims.append(float(sim[both].mean()))
            changed.append(float(((fps != base_fps).any(dim=-1) & both).float().sum()
                                 / max(int(both.sum()), 1)))
        rows.append({
            "sigma_normalized": sigma, "sigma_angstrom": sigma * 2.0,
            "self_tanimoto_mean": sum(sims) / len(sims),
            "frac_molecules_with_any_bit_change": sum(changed) / len(changed),
        })
    record = {
        "_what": "FP2 self-similarity after an isotropic coordinate perturbation "
                 "(how much displacement Table 3's metric can even see)",
        "num_mols": args.num_mols, "repeats": args.repeats, "seed": args.seed,
        "mean_atoms": float(n_atoms.float().mean()),
        "n_valid_base": int(base_valid.sum()),
        "rows": rows,
    }
    print(f"\n  FP2 sensitivity to coordinate noise ({args.num_mols} real GEOM val "
          f"molecules, {args.repeats} draws each)")
    print(f"  {'sigma (norm)':>13}{'sigma (A)':>11}{'self-Tanimoto':>15}"
          f"{'frac changed':>14}")
    for r in rows:
        print(f"  {r['sigma_normalized']:>13.4g}{r['sigma_angstrom']:>11.4g}"
              f"{r['self_tanimoto_mean']:>15.4f}"
              f"{r['frac_molecules_with_any_bit_change']:>14.3f}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"  wrote {args.out}")
    return record


def main() -> None:
    args = parse_args()
    if args.cmd == "fd-check":
        cmd_fd_check(args)
    elif args.cmd == "fp2-sensitivity":
        cmd_fp2_sensitivity(args)
    else:
        cmd_grad_sign(args)


if __name__ == "__main__":
    main()
