"""steer_search (SS): multiMFM steering + PB-valid best-look-ahead selection (Algorithm 3).

multiMFM-SS integrates the guided multiMFM ODE and, along the way, keeps the
look-aheads the steering already produced. At the end it returns, per molecule, the
PoseBusters-valid candidate with the lowest guide-predicted error among the stored
look-aheads and the trajectory endpoint, falling back to the endpoint when no
candidate is PB-valid (paper §3.2, Appendix Algorithm 3). Selection uses the guide
regressor; we report the held-out ORACLE MAE and PoseBusters validity.

Two look-ahead sources (``--lookahead-source``):

``mfm`` (default; the paper's Algorithm 3)
    The look-aheads are the N MFM posterior samples ``X_{0,1}(eps_i; t_k, x_{t_k})``
    that the value estimator draws at every guided step (N = ``--value-samples``); they
    are reused as they are, so steer-search adds **no generative-model evaluations**
    (verified by the NFE counters in ``summary.json``). All guided steps contribute
    (``--mfm-select-min-t 0``). The error of look-ahead i is
    ``err_i = |R(y_1^i, a_1^i) - target|`` with the guide R applied to the look-ahead
    molecule (argmax atom types, as Algorithm 3's err_i and as the submitted code scored
    its candidates): one extra *regressor* forward per look-ahead
    (``--lookahead-score guide_hard``, default). ``--lookahead-score value`` instead
    reuses the guide values of the tilt (soft-atom relaxation), i.e. no extra
    evaluation of any network; on the seed-8 pilot it selected worse molecules in
    5 of 6 properties (results/qm9/ss_mfm_pilot). To keep memory bounded for n = 1000,
    only the ``--lookahead-topk`` lowest-error look-aheads per molecule are kept, on
    CPU; PoseBusters is then run lazily in order of increasing error, which returns
    exactly Algorithm 3's choice unless a molecule has more than K look-aheads and
    all K kept ones are PB-invalid (counted as ``n_truncated_uncertain``; 0 means the
    selection is exact).

``base_endpoint`` (the code behind the submitted multiMFM-SS row)
    At every step with ``t >= --select-min-t`` the BASE model's one-step endpoint
    prediction ``x1_hat = endpoint_prediction(model, x_t, t)`` is computed (one extra
    base forward pass per molecule and harvested step) and scored with the guide on
    argmax atom types. This is the original ``best_of_lookahead.py`` behaviour
    (runs ``k_<prop>_{a,b}``); the lazy PB walk returns the same selection as its
    exhaustive PB check (``--pb-check-all`` reproduces the exhaustive check).

``both`` runs the two selections on the same guided trajectory (the base harvest
does not change the trajectory), for a paired comparison; the NFE of each variant
is reported separately.

Outputs in ``--output-dir``: ``summary.json`` (rows ``final_endpoint``,
``best_lookahead_raw`` / ``best_lookahead_pbvalid`` for the primary source, i.e. the
SS number, and ``base_endpoint_raw`` / ``base_endpoint_pbvalid`` in ``both`` mode;
NFE per molecule by phase) and ``per_sample.csv`` (per molecule and row: target,
guide/oracle predictions, PB flag, SMILES, selected look-ahead time).

The ``final_endpoint`` row is the guided trajectory's last state at the SS guidance
strength. Paper settings: ``scripts/run_steer_search.sh``.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from multimfm import paths  # noqa: E402
from multimfm.glass_guidance import (  # noqa: E402
    capture_torch_rng,
    clip_guidance_rms,
    clone_state,
    load_flow_model,
    load_mfm_student,
    predicted_property,
    production_step_with_optional_guidance,
    require_linear_dfm_atoms,
    restore_torch_rng,
    seed_all,
    value_gradient,
)
from multimfm.glass_posterior import endpoint_prediction  # noqa: E402
from multimfm.nfe import NFECounter, attach_standard  # noqa: E402
from multimfm.property_targets import (  # noqa: E402
    build_or_load_histograms,
    sample_property_target,
)
from multimfm.qm9_data import choose_device, load_tfg_regressor  # noqa: E402
from multimfm.qm9_eval import PBPool, decode, make_state, molecule_records  # noqa: E402
from tabasco.chem.convert import MoleculeConverter  # noqa: E402


# ------------------------------------------------------------------ legacy helpers
# (kept for multimfm.baselines and old notebooks)

def parallel_pb_valid(all_mols, config_path, workers):
    """all_mols: list length C of lists length B (RDKit mol or None) -> valid[C, B]."""
    C = len(all_mols)
    B = len(all_mols[0]) if C else 0
    flat = [all_mols[c][i] for c in range(C) for i in range(B)]
    with PBPool(config_path, workers) as pool:
        ok = pool.valid(flat)
    return torch.tensor(ok, dtype=torch.bool).view(C, B)


def gather_by_idx(cand, idx):  # cand [C,B,...], idx [B] -> [B,...]
    B = idx.shape[0]
    return cand[idx, torch.arange(B, device=idx.device)]


# ------------------------------------------------------------------ CLI

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=paths.BASE_FLOW_CKPT,
                   help="Base QM9 flow-matching checkpoint.")
    p.add_argument("--mfm-checkpoint", type=Path, default=paths.MFM_STUDENT_CKPT,
                   help="Meta flow map (MFM) student checkpoint used as the value sampler.")
    p.add_argument("--mfm-key", default="student")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--target-seed", type=int, default=2026)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=128)
    p.add_argument("--mu", type=float, default=0.45)
    p.add_argument("--reward-scale", type=float, default=0.3)
    p.add_argument("--value-samples", type=int, default=32)
    p.add_argument("--value-batch-size", type=int, default=0)
    p.add_argument("--value-glass-steps", type=int, default=4)
    p.add_argument("--mfm-diagonal", action="store_true")
    p.add_argument("--guide-every", type=int, default=1)
    p.add_argument("--guide-min-t", type=float, default=0.05)
    p.add_argument("--guide-max-t", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--soft-atom-temperature", type=float, default=0.25)
    p.add_argument("--property-name", default="alpha",
                   choices=["alpha", "cv", "gap", "homo", "lumo", "mu"],
                   help="QM9 property to steer toward.")
    p.add_argument("--partition", default="train_flow")
    p.add_argument("--hist-bins", type=int, default=100)
    # steer_search knobs
    p.add_argument("--lookahead-source", choices=["mfm", "base_endpoint", "both"], default="mfm",
                   help="mfm: store the value estimator's MFM look-aheads (Algorithm 3, no extra "
                        "generative evaluations; default). base_endpoint: the original code, "
                        "harvesting the base model's one-step endpoint prediction (one extra base "
                        "forward per harvested step). both: both selections on one trajectory.")
    p.add_argument("--mfm-select-min-t", type=float, default=0.0,
                   help="mfm source: keep look-aheads from guided steps with t >= this (0 = every "
                        "guided step, as in Algorithm 3).")
    p.add_argument("--mfm-select-max-t", type=float, default=1.01)
    p.add_argument("--lookahead-topk", type=int, default=1024,
                   help="mfm source: per molecule, keep the K lowest-error look-aheads on CPU "
                        "(0 = keep all; memory ~ steps x value_samples x n x atoms).")
    p.add_argument("--lookahead-score", choices=["value", "guide_hard"], default="guide_hard",
                   help="mfm source: error used to rank look-aheads and the endpoint. guide_hard "
                        "(default) = the guide on the look-ahead molecule (argmax atom types; one "
                        "regressor forward per look-ahead, no generative ones). value = reuse the "
                        "value estimator's guide values (soft atom types, the R inside the tilt; no "
                        "extra evaluation of any network).")
    p.add_argument("--select-min-t", type=float, default=0.5,
                   help="base_endpoint source: only harvest endpoint predictions with t >= this.")
    p.add_argument("--select-max-t", type=float, default=1.01)
    p.add_argument("--pb-candidates", type=int, default=0,
                   help="base_endpoint source: harvest subgrid size; <=0 means every in-window step.")
    p.add_argument("--diagnostic-variants", action="store_true",
                   help="Also run, on the same trajectory, the MFM selection with the other "
                        "--lookahead-score (mfm_valuescore / mfm_hardscore) and with the base_endpoint time window "
                        "t >= --select-min-t (mfm_window). No extra generative evaluations.")
    p.add_argument("--pb-check-all", action="store_true",
                   help="PB-check every stored candidate instead of the lazy walk (the original "
                        "exhaustive check; slow, for verification).")
    p.add_argument("--pb-workers", type=int, default=16,
                   help="CPU processes to spread the PoseBusters validity check across.")
    p.add_argument("--output-dir", type=Path, default=paths.OUTPUTS / "steer_search")
    p.add_argument("--hist-cache", type=Path, default=None)
    p.add_argument("--force-rebuild-hist", action="store_true")
    p.add_argument("--no-sanitize", action="store_true")
    p.add_argument("--posebusters-config", type=Path, default=None)
    return p.parse_args(argv)


# ------------------------------------------------------------------ look-ahead store

class LookaheadStore:
    """Per-molecule look-ahead candidates on CPU.

    ``topk > 0``: keep only the K lowest-error candidates per molecule (bounded memory).
    ``topk == 0``: keep everything (exact but unbounded).
    """

    def __init__(self, n_mol: int, n_atoms: int, atom_dim: int, topk: int, *,
                 kind: str = "mfm", score: str = "value", t_min: float = 0.0,
                 t_max: float = 1.01) -> None:
        self.B, self.L, self.A, self.K = n_mol, n_atoms, atom_dim, int(topk)
        self.kind, self.score, self.t_min, self.t_max = kind, score, float(t_min), float(t_max)
        self.offered = torch.zeros(n_mol, dtype=torch.long)
        if self.K > 0:
            self.coords = torch.zeros(n_mol, self.K, n_atoms, 3)
            self.atomics = torch.zeros(n_mol, self.K, n_atoms, atom_dim)
            self.err = torch.full((n_mol, self.K), float("inf"))
            self.t = torch.full((n_mol, self.K), float("nan"))
        else:
            self._c, self._a, self._e, self._t = [], [], [], []

    def add(self, coords, atomics, err, t_value: float) -> None:
        """coords [B,M,L,3], atomics [B,M,L,A], err [B,M] (any device)."""
        coords = coords.detach().float().cpu()
        atomics = atomics.detach().float().cpu()
        err = err.detach().float().cpu()
        B, M = err.shape
        self.offered += M
        t = torch.full((B, M), float(t_value))
        if self.K <= 0:
            self._c.append(coords); self._a.append(atomics); self._e.append(err); self._t.append(t)
            return
        # only candidates that can enter the buffer
        worst = self.err.max(dim=1).values                       # [B]
        if not bool((err < worst[:, None]).any()):
            return
        e = torch.cat([self.err, err], dim=1)                    # [B, K+M]
        idx = torch.topk(e, self.K, dim=1, largest=False, sorted=False).indices
        rows = torch.arange(B)[:, None]
        c_all = torch.cat([self.coords, coords], dim=1)
        a_all = torch.cat([self.atomics, atomics], dim=1)
        t_all = torch.cat([self.t, t], dim=1)
        self.coords = c_all[rows, idx]
        self.atomics = a_all[rows, idx]
        self.err = e[rows, idx]
        self.t = t_all[rows, idx]

    def tensors(self):
        if self.K > 0:
            return self.coords, self.atomics, self.err, self.t
        if not self._e:
            z = torch.zeros(self.B, 0)
            return (torch.zeros(self.B, 0, self.L, 3), torch.zeros(self.B, 0, self.L, self.A), z, z)
        return (torch.cat(self._c, 1), torch.cat(self._a, 1), torch.cat(self._e, 1),
                torch.cat(self._t, 1))

    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in self.tensors())


# ------------------------------------------------------------------ selection

def lazy_select(*, err_la, t_la, coords_la, atomics_la, err_end, final_state, mask, converter,
                sanitize, pool, check_all=False, first_chunk=2, max_chunk=64):
    """Algorithm 3's selection, evaluating PoseBusters lazily.

    Candidates of molecule b are its stored look-aheads (err_la[b, :], inf = empty)
    and the endpoint (err_end[b]). They are visited in order of increasing error
    (look-aheads before the endpoint on ties, matching the strict "<" in Algorithm 3)
    and the first PB-valid one is selected; if none is valid the endpoint is returned.
    Returns (sel [B] long, C = endpoint), resolved [B] bool (selected one is PB-valid)).
    """
    B, C = err_la.shape
    all_err = torch.cat([err_la, err_end[:, None].cpu()], dim=1)          # [B, C+1]
    order = torch.sort(all_err, dim=1, stable=True).indices
    n_entries = torch.isfinite(all_err).sum(dim=1)
    sel = torch.full((B,), C, dtype=torch.long)
    resolved = torch.zeros(B, dtype=torch.bool)
    ptr = torch.zeros(B, dtype=torch.long)
    final_cpu = final_state.detach().cpu()
    mask_cpu = mask.detach().cpu()
    chunk = int(all_err.shape[1]) if check_all else first_chunk
    while True:
        todo = (~resolved) & (ptr < n_entries)
        if not bool(todo.any()):
            break
        pairs = []
        for b in todo.nonzero().flatten().tolist():
            hi = min(int(ptr[b]) + chunk, int(n_entries[b]))
            for r in range(int(ptr[b]), hi):
                pairs.append((b, int(order[b, r])))
        bs = torch.tensor([b for b, _ in pairs])
        cs = torch.tensor([c for _, c in pairs])
        is_end = cs == C
        cc = torch.where(is_end[:, None, None], final_cpu["coords"][bs],
                         coords_la[bs, cs.clamp(max=max(C - 1, 0))] if C else final_cpu["coords"][bs])
        aa = torch.where(is_end[:, None, None], final_cpu["atomics"][bs],
                         atomics_la[bs, cs.clamp(max=max(C - 1, 0))] if C else final_cpu["atomics"][bs])
        st = make_state(cc, aa, mask_cpu[bs])
        ok = pool.valid(decode(st, converter, sanitize=sanitize))
        for (b, c), v in zip(pairs, ok):     # pairs are in rank order per molecule
            if v and not resolved[b]:
                resolved[b] = True
                sel[b] = c
        ptr[todo] += chunk
        chunk = min(chunk * 2, max_chunk) if not check_all else chunk
    return sel, resolved


def gather_candidates(sel, *, coords_la, atomics_la, t_la, err_la, err_end, final_state):
    """States/metadata of the selected candidate per molecule (sel == C -> endpoint)."""
    B, C = err_la.shape
    final_cpu = final_state.detach().cpu()
    is_end = sel == C
    safe = sel.clamp(max=max(C - 1, 0))
    rows = torch.arange(B)
    if C:
        coords = torch.where(is_end[:, None, None], final_cpu["coords"], coords_la[rows, safe])
        atomics = torch.where(is_end[:, None, None], final_cpu["atomics"], atomics_la[rows, safe])
        t_sel = torch.where(is_end, torch.ones(B), t_la[rows, safe])
        e_sel = torch.where(is_end, err_end.cpu(), err_la[rows, safe])
    else:
        coords, atomics = final_cpu["coords"], final_cpu["atomics"]
        t_sel, e_sel = torch.ones(B), err_end.cpu()
    return coords, atomics, t_sel, e_sel, is_end


# ------------------------------------------------------------------ guided trajectory

def guided_sample_collect(model, mfm_student, init_state, *, args, targets, guide_reg, nfe,
                          stores):
    """Guided multiMFM trajectory; fills ``stores`` ({"mfm": ..., "base_endpoint": ...}).

    Returns (final_state, info) with the per-source candidate times and step counts.
    """
    state = clone_state(init_state)
    device = state["coords"].device
    schedule = model._get_sample_schedule(args.sample_steps).to(device)
    tgt = targets.to(device)
    B = state["coords"].shape[0]
    mfm_stores = {k: v for k, v in stores.items() if v.kind == "mfm"}
    base_stores = {k: v for k, v in stores.items() if v.kind == "base_endpoint"}

    # old harvest grid (base_endpoint source), unchanged from the original script
    in_window = [i for i in range(1, len(schedule))
                 if args.select_min_t <= float(schedule[i - 1]) <= args.select_max_t]
    if args.pb_candidates and args.pb_candidates > 0 and len(in_window) > args.pb_candidates:
        sel = np.linspace(0, len(in_window) - 1, args.pb_candidates).round().astype(int)
        harvest = sorted({in_window[j] for j in sel})
    else:
        harvest = in_window

    info = {"mfm_steps": [], "base_steps": [], "guided_steps": 0}
    for i in range(1, len(schedule)):
        t_value = schedule[i - 1]
        dt_value = schedule[i] - schedule[i - 1]
        t = t_value.expand(B)
        should_guide = (
            args.mu != 0.0 and args.value_samples > 0 and args.guide_every > 0
            and (i - 1) % args.guide_every == 0
            and float(t_value) >= args.guide_min_t and float(t_value) <= args.guide_max_t
        )
        takers = [st for st in mfm_stores.values()
                  if should_guide and st.t_min <= float(t_value) <= st.t_max]
        keep = bool(takers)
        grad_coords = grad_atomics = None
        if should_guide:
            info["guided_steps"] += 1
            rng = capture_torch_rng(device)
            with nfe.phase("trajectory"), torch.enable_grad():
                grad_coords, grad_atomics, stats = value_gradient(
                    model, state, t, value_sampler="mfm", mfm_student=mfm_student,
                    mfm_diagonal=args.mfm_diagonal, posterior_samples=args.value_samples,
                    n_steps=args.value_glass_steps, reward_name="target_property",
                    reward_scale=args.reward_scale, target_x=0.0,
                    property_name=args.property_name, property_target=tgt,
                    property_regressor=guide_reg, guide_atomics=True, eps=args.eps,
                    value_batch_size=args.value_batch_size, keep_posterior=keep,
                )
            grad_coords, _ = clip_guidance_rms(grad_coords, state["padding_mask"], 0.0)
            grad_atomics, _ = clip_guidance_rms(grad_atomics, state["padding_mask"], 0.0)
            restore_torch_rng(rng, device)
            if keep:
                pc, pa = stats.pop("posterior_coords"), stats.pop("posterior_atomics")
                reward = stats.pop("posterior_reward")                 # [B, N] = -(R - y*)^2
                errs = {"value": (-reward).clamp_min(0.0).sqrt()}
                if any(st.score == "guide_hard" for st in takers):
                    with nfe.phase("selection"), torch.no_grad():
                        flat = make_state(pc.reshape(-1, *pc.shape[2:]),
                                          pa.reshape(-1, *pa.shape[2:]),
                                          state["padding_mask"].repeat_interleave(pc.shape[1], 0))
                        pred = predicted_property(flat, property_name=args.property_name,
                                                  property_regressor=guide_reg, soft=False)
                        errs["guide_hard"] = (pred.view(B, -1) - tgt[:, None]).abs()
                for st in takers:
                    st.add(pc, pa, errs[st.score], float(t_value))
                info["mfm_steps"].append(float(t_value))
                del pc, pa, reward, errs

        if base_stores and i in harvest:
            with nfe.phase("harvest_base_endpoint"), torch.no_grad():
                cand = endpoint_prediction(model, state, t)
            with nfe.phase("selection"), torch.no_grad():
                pred = predicted_property(cand, property_name=args.property_name,
                                          property_regressor=guide_reg)
            err = (pred - tgt).abs()
            for st in base_stores.values():
                st.add(cand["coords"][:, None], cand["atomics"][:, None], err[:, None],
                       float(t_value))
            info["base_steps"].append(float(t_value))

        with nfe.phase("trajectory"), torch.no_grad():
            dt = dt_value.expand(B)
            state = production_step_with_optional_guidance(
                model, state, t, dt, mu=args.mu, grad_coords=grad_coords, grad_atomics=grad_atomics)
    return state, info


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate_state(state, *, targets, property_name, guide_reg, oracle_reg, converter, pool,
                   sanitize, run):
    guide = predicted_property(state, property_name=property_name, property_regressor=guide_reg)
    oracle = predicted_property(state, property_name=property_name, property_regressor=oracle_reg)
    tgt = targets.to(guide.device)
    rec = molecule_records(state, converter, pool, sanitize=sanitize)
    valid = torch.tensor(rec["valid"], dtype=torch.bool, device=guide.device)
    oerr = (oracle - tgt).abs()
    row = {
        "run": run,
        "guide_mae": float((guide - tgt).abs().mean().cpu()),
        "oracle_mae": float(oerr.mean().cpu()),
        "oracle_median_abs_error": float(oerr.median().cpu()),
        "oracle_mae_pb_valid_only": float(oerr[valid].mean().cpu()) if bool(valid.any()) else None,
        "pb_pb_valid": rec["summary"]["pb_valid"],
    }
    row.update({f"pb_{k}": v for k, v in rec["summary"].items() if k != "pb_valid"})
    per = {"guide_prediction": guide.cpu().tolist(), "oracle_prediction": oracle.cpu().tolist(),
           "pb_valid": rec["valid"], "smiles": rec["smiles"]}
    return row, per


# backwards-compatible name used by multimfm.baselines
def evaluate(state, *, targets, property_name, guide_reg, oracle_reg, data_stats,
             posebusters=None, sanitize=True, run, pool=None, pb_workers=8):
    converter = MoleculeConverter(
        atom_names=list(data_stats["atom_names"]),
        dataset_normalizer=float(data_stats.get("coordinate_normalizer", 1.0)),
    )
    own = pool is None
    if own:
        pool = PBPool(paths.posebusters_config_path(), pb_workers)
    try:
        row, _ = evaluate_state(state, targets=targets, property_name=property_name,
                                guide_reg=guide_reg, oracle_reg=oracle_reg, converter=converter,
                                pool=pool, sanitize=sanitize, run=run)
    finally:
        if own:
            pool.close()
    return row


def main(argv=None) -> None:
    args = parse_args(argv)
    t_start = time.time()
    if args.hist_cache is None:
        args.hist_cache = paths.default_hist_cache(args.property_name, args.partition,
                                                   args.hist_bins)
    if args.posebusters_config is None:
        args.posebusters_config = paths.posebusters_config_path()
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    model, data_stats, model_args = load_flow_model(args.checkpoint, device)
    require_linear_dfm_atoms(model)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    mfm_student = load_mfm_student(args.mfm_checkpoint, data_stats=data_stats,
                                   teacher_args=model_args, device=device, key=args.mfm_key)
    guide_reg = load_tfg_regressor("guide", args.property_name, device)
    oracle_reg = load_tfg_regressor("oracle", args.property_name, device)
    for reg in (guide_reg, oracle_reg):
        reg.soft_atom_temperature = args.soft_atom_temperature
        reg.eval()
        for prm in reg.parameters():
            prm.requires_grad_(False)
    nfe = attach_standard(NFECounter(), model=model, mfm_student=mfm_student,
                          regressors=(guide_reg, oracle_reg))

    max_len = int(data_stats["max_num_atoms"])
    hist = build_or_load_histograms(args, max_len=max_len)

    seed_all(args.seed)
    init_state = model._sample_noise_like_batch(batch_size=args.num_samples).to(device)
    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
    trng = np.random.default_rng(args.target_seed)
    target_values = [sample_property_target(hist, int(l), trng) for l in lengths]
    targets = torch.tensor(target_values, dtype=torch.float32, device=device)
    B = args.num_samples
    L = init_state["coords"].shape[1]
    A = init_state["atomics"].shape[-1]

    sources = ["mfm", "base_endpoint"] if args.lookahead_source == "both" else [args.lookahead_source]
    stores = {}
    for src in sources:
        if src == "mfm":
            stores["mfm"] = LookaheadStore(B, L, A, args.lookahead_topk, kind="mfm",
                                           score=args.lookahead_score,
                                           t_min=args.mfm_select_min_t,
                                           t_max=args.mfm_select_max_t)
        else:
            stores["base_endpoint"] = LookaheadStore(B, L, A, 0, kind="base_endpoint",
                                                     score="guide_hard",
                                                     t_min=args.select_min_t,
                                                     t_max=args.select_max_t)
    if args.diagnostic_variants:
        alt = "value" if args.lookahead_score == "guide_hard" else "guide_hard"
        alt_name = "mfm_valuescore" if alt == "value" else "mfm_hardscore"
        stores[alt_name] = LookaheadStore(B, L, A, args.lookahead_topk, kind="mfm", score=alt,
                                          t_min=args.mfm_select_min_t,
                                          t_max=args.mfm_select_max_t)
        stores["mfm_window"] = LookaheadStore(B, L, A, args.lookahead_topk, kind="mfm",
                                              score=args.lookahead_score,
                                              t_min=args.select_min_t, t_max=args.select_max_t)
        sources = sources + [alt_name, "mfm_window"]

    seed_all(args.seed)
    t0 = time.time()
    final_state, info = guided_sample_collect(
        model, mfm_student, init_state, args=args, targets=targets, guide_reg=guide_reg,
        nfe=nfe, stores=stores)
    t_traj = time.time() - t0

    mask = init_state["padding_mask"]
    sanitize = not args.no_sanitize
    converter = MoleculeConverter(
        atom_names=list(data_stats["atom_names"]),
        dataset_normalizer=float(data_stats.get("coordinate_normalizer", 1.0)),
    )
    pool = PBPool(args.posebusters_config, args.pb_workers)

    rows, per_rows, diag = [], {}, {}
    ev = lambda st, run: evaluate_state(st, targets=targets, property_name=args.property_name,
                                        guide_reg=guide_reg, oracle_reg=oracle_reg,
                                        converter=converter, pool=pool, sanitize=sanitize, run=run)
    with nfe.phase("evaluation"):
        row, per = ev(final_state, "final_endpoint")
    rows.append(row)
    per_rows["final_endpoint"] = (per, torch.ones(B), torch.zeros(B, dtype=torch.bool))

    t0 = time.time()
    for src in sources:
        store = stores[src]
        coords_la, atomics_la, err_la, t_la = store.tensors()
        with nfe.phase("selection"), torch.no_grad():
            soft = store.kind == "mfm" and store.score == "value"
            pred_end = predicted_property(final_state, property_name=args.property_name,
                                          property_regressor=guide_reg, soft=bool(soft))
        err_end = (pred_end - targets).abs().float().cpu()
        n_before = pool.n_checked
        sel, resolved = lazy_select(
            err_la=err_la, t_la=t_la, coords_la=coords_la, atomics_la=atomics_la, err_end=err_end,
            final_state=final_state, mask=mask, converter=converter, sanitize=sanitize, pool=pool,
            check_all=args.pb_check_all)
        n_checks = pool.n_checked - n_before
        C = err_la.shape[1]
        # unfiltered: lowest error over look-aheads and endpoint
        all_err = torch.cat([err_la, err_end[:, None]], dim=1)
        raw_idx = torch.sort(all_err, dim=1, stable=True).indices[:, 0]
        prefix = "best_lookahead" if src == sources[0] else src
        for kind, idx in (("raw", raw_idx), ("pbvalid", sel)):
            coords, atomics, t_sel, e_sel, is_end = gather_candidates(
                idx, coords_la=coords_la, atomics_la=atomics_la, t_la=t_la, err_la=err_la,
                err_end=err_end, final_state=final_state)
            st = make_state(coords.to(device), atomics.to(device), mask)
            run = f"{prefix}_{kind}"
            with nfe.phase("evaluation"):
                row, per = ev(st, run)
            row.update({
                "lookahead_source": src,
                "lookahead_score": store.score,
                "lookahead_t_min": store.t_min,
                "selected_t_mean": float(t_sel.mean()),
                "frac_selected_lookahead": float((~is_end).float().mean()),
                "selection_err_mean": float(e_sel.mean()),
            })
            if kind == "pbvalid":
                kth = err_la.max(dim=1).values if C else torch.full((B,), -float("inf"))
                uncertain = (~resolved | (is_end & (err_end > kth))) & (store.offered > C) \
                    if store.K > 0 else torch.zeros(B, dtype=torch.bool)
                row.update({
                    "frac_with_valid_candidate": float(resolved.float().mean()),
                    "pb_checks_per_mol": n_checks / B,
                    "n_truncated_uncertain": int(uncertain.sum()),
                })
            rows.append(row)
            per_rows[run] = (per, t_sel, ~is_end)
        diag[src] = {
            "num_candidates_per_mol": float(store.offered.float().mean()),
            "stored_per_mol": int(C),
            "topk": int(store.K),
            "store_mb": store.nbytes() / 2**20,
            "steps": [t for t in (info["mfm_steps"] if store.kind == "mfm" else info["base_steps"])
                      if store.t_min <= t <= store.t_max],
            "pb_checks": int(n_checks),
        }
    t_sel_total = time.time() - t0
    pool.close()

    # ---- NFE accounting (per generated molecule)
    by_phase = nfe.by_phase(B)
    traj = by_phase.get("trajectory", {})
    harvest = by_phase.get("harvest_base_endpoint", {})
    sel_phase = by_phase.get("selection", {})
    variants = {}
    for src in sources:
        is_base = stores[src].kind == "base_endpoint"
        extra = harvest.get("nfe_generative", 0.0) if is_base else 0.0
        extra_bwd = harvest.get("nfe_generative_bwd", 0.0) if is_base else 0.0
        variants[src] = {
            "nfe_generative": traj.get("nfe_generative", 0.0) + extra,
            "nfe_generative_bwd": traj.get("nfe_generative_bwd", 0.0) + extra_bwd,
            "extra_generative_vs_multimfm": extra,
        }
    nfe_meta = {
        "by_phase": by_phase,
        "variants": variants,
        "selection_generative_rows": sel_phase.get("nfe_generative", 0.0) * B,
        "raw": nfe.raw(),
        "guided_steps": info["guided_steps"],
        "note": "per generated molecule; phases: trajectory = value estimator + ODE steps "
                "(identical to plain multiMFM), harvest_base_endpoint = the original SS harvest, "
                "selection = guide re-scoring, evaluation = final guide/oracle predictions.",
    }
    for row in rows:
        src = row.get("lookahead_source")
        if src is not None:
            row["nfe_generative_per_mol"] = variants[src]["nfe_generative"]
            row["nfe_generative_bwd_per_mol"] = variants[src]["nfe_generative_bwd"]
    rows[0]["nfe_generative_per_mol"] = traj.get("nfe_generative", 0.0)
    rows[0]["nfe_generative_bwd_per_mol"] = traj.get("nfe_generative_bwd", 0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": rows,
        "num_candidates": {s: diag[s]["num_candidates_per_mol"] for s in sources},
        "lookahead": diag,
        "nfe": nfe_meta,
        "timing_s": {"trajectory": t_traj, "selection_and_eval": t_sel_total,
                     "total": time.time() - t_start},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_max_memory_gb": (torch.cuda.max_memory_allocated(device) / 2**30
                               if device.type == "cuda" else None),
        "lengths": [int(x) for x in lengths],
        "targets": [float(x) for x in target_values],
    }
    # candidate_t kept for the old notebooks (base_endpoint harvest times)
    if "base_endpoint" in diag:
        meta["candidate_t"] = diag["base_endpoint"]["steps"] + [1.0]

    with open(args.output_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "sample_idx", "length", "target", "guide_prediction", "guide_abs_error",
                    "oracle_prediction", "oracle_abs_error", "pb_valid", "smiles", "selected_t",
                    "selected_lookahead"])
        for run, (per, t_sel, is_la) in per_rows.items():
            for i in range(B):
                g, o = per["guide_prediction"][i], per["oracle_prediction"][i]
                w.writerow([run, i, int(lengths[i]), target_values[i], g, abs(g - target_values[i]),
                            o, abs(o - target_values[i]), int(per["pb_valid"][i]),
                            per["smiles"][i] or "", float(t_sel[i]), int(is_la[i])])
    (args.output_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    for e in rows:
        print({k: e[k] for k in ("run", "guide_mae", "oracle_mae", "pb_pb_valid") if k in e})
    print("NFE per molecule by variant:", json.dumps(variants))
    print(f"timing: {meta['timing_s']}")


if __name__ == "__main__":
    main()
