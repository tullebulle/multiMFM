#!/usr/bin/env python
"""Terminal composition guidance on *generated* GEOM-Drugs molecules.

Why this exists. `docs/geom_guidance_diagnosis.md` §6.2 found the atom-type half
of Table 3's guidance numerically inert. Two structural reasons make that a
defect rather than a property (derived in `docs/geom_rescue_attempt.md`):

1. `GaussianSimplexAtomInterpolant.step` is a contraction toward
   `psi = softmax(logits)` with coefficient `dbeta/(1 - beta)`, which equals
   **exactly 1 at the final step** of the linear schedule. So `x_final = psi`
   and every additive perturbation of the atom state is erased. (The same holds
   for coordinates, but there the endpoint predictor is nearly the identity at
   t = 0.99, so the coordinate perturbation passes through; `psi`'s argmax is a
   categorical decision and a 0.0019-sized nudge never changes it.)
2. One `mu` is applied to both modalities, and the value that makes the
   coordinate step 2.6% of the coordinate state RMS makes the atom-type step
   0.19% of the atom state RMS -- three orders below what flips an argmax.

At t = 1 the look-ahead is the identity, so `V_1 = lambda * r` exactly and
terminal guidance is plain ascent on the paper's own reward. This script samples
the **unguided** arm (bit-identical to the recorded unguided arm at the same
seed, so the comparison is exactly paired), then applies composition guidance to
the terminal state under several rules and reports the counted FP2 Tanimoto,
PoseBusters validity and guide accuracy of each.

Arms:
  head_exact  greedy substitution maximising the guide head's own reward
              (a real guidance method: head + target only)
  head_lin    greedy substitution from the first-order soft-path gradient, i.e.
              what the sampler's own atom-type gradient points at
  grad_step   `atomics <- atomics + eta * ghat` then argmax -- the sampler's own
              rule at a step size that can flip something
  oracle      greedy substitution chosen by the TRUE counted FP2 (the attainable
              headroom; NOT a guidance method, reported as a ceiling only)
  random      random substitution (control)

Every arm is reported. Nothing is selected.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from multimfm.geom import fingerprint as fp
from multimfm.geom.paths import (
    DEFAULT_GEOM_FP_DIR, DEFAULT_PB_CONFIG, DEFAULT_STRUCTURE_HEAD, DEFAULT_TEACHER, RESULTS,
)
from multimfm.geom.reward import install_fingerprint_reward
from multimfm.geom.teacher import load_geom_teacher
from multimfm.glass_guidance import build_posebusters, guided_sample, pb_summary_for_state
from multimfm.train_base_flow import choose_device, set_seed
from tabasco.sample.glass import require_linear_dfm_atoms

N_REAL = 8
DUMMY = 8


def soft_tanimoto(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    inter = (p * c).sum(-1)
    return inter / (p.sum(-1) + c.sum(-1) - inter + 1e-9)


def reward_of(pred: torch.Tensor, targets: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "l2":
        return -((pred - targets) ** 2).sum(-1)
    return soft_tanimoto(pred, targets)


@torch.no_grad()
def head_reward_for_types(head, coords, mask_long, types, targets, kind, chunk=64):
    """r for a batch of (coords, types); coords in the normalized frame."""
    out = []
    for s in range(0, types.shape[0], chunk):
        e = min(s + chunk, types.shape[0])
        logits = head(coords[s:e], types[s:e], mask_long[s:e])
        out.append(reward_of(torch.sigmoid(logits), targets[s:e], kind))
    return torch.cat(out)


@torch.no_grad()
def greedy_round(head, state, targets, kind, *, mode, rng, allow_dummy, target_fps=None,
                 normalizer=2.0, allowed=None):
    """One greedy substitution per molecule. Returns new atomics (one-hot)."""
    coords = state["coords"]
    pad = state["padding_mask"]
    mask_long = (~pad).long()
    types = state["atomics"].argmax(-1)
    B, L = types.shape
    ntypes = 9 if allow_dummy else N_REAL
    allowed = list(range(ntypes)) if allowed is None else list(allowed)
    real = ~pad
    new_types = types.clone()

    if mode == "random":
        for b in range(B):
            idx = torch.nonzero(real[b]).flatten()
            if len(idx) == 0:
                continue
            a = idx[int(rng.integers(len(idx)))]
            new_types[b, a] = int(allowed[int(rng.integers(len(allowed)))])
        return new_types

    if mode == "oracle":
        for b in range(B):
            idx = torch.nonzero(real[b]).flatten().tolist()
            base = target_fps[b]
            best, bestk = None, -1.0
            for a in idx:
                for t in allowed:
                    if t == int(types[b, a]):
                        continue
                    t2 = types[b].clone(); t2[a] = t
                    st = TensorDict({"coords": coords[b:b+1].cpu(),
                                     "atomics": torch.nn.functional.one_hot(
                                         t2.unsqueeze(0), 9).float().cpu(),
                                     "padding_mask": pad[b:b+1].cpu()}, batch_size=1)
                    g, _, _ = fp.state_to_binary_fps(st, normalizer)
                    k = float(fp.tanimoto_similarity(g, base.unsqueeze(0))[0])
                    if k > bestk:
                        bestk, best = k, (a, t)
            if best is not None:
                new_types[b, best[0]] = best[1]
        return new_types

    # head_exact: build all candidate flips per molecule and score them with the head
    for b in range(B):
        idx = torch.nonzero(real[b]).flatten().tolist()
        cands = [(a, t) for a in idx for t in allowed if t != int(types[b, a])]
        if not cands:
            continue
        cand_types = types[b].unsqueeze(0).repeat(len(cands), 1)
        for k, (a, t) in enumerate(cands):
            cand_types[k, a] = t
        r = head_reward_for_types(
            head,
            coords[b].unsqueeze(0).expand(len(cands), -1, -1).contiguous(),
            mask_long[b].unsqueeze(0).expand(len(cands), -1).contiguous(),
            cand_types.contiguous(),
            targets[b].unsqueeze(0).expand(len(cands), -1).contiguous(),
            kind,
        )
        k = int(r.argmax())
        new_types[b, cands[k][0]] = cands[k][1]
    return new_types


def lin_round(head, state, targets, kind, temperature, allow_dummy, allowed=None):
    """One substitution per molecule from the first-order soft-path gradient."""
    coords = state["coords"]
    pad = state["padding_mask"]
    mask_long = (~pad).long()
    types = state["atomics"].argmax(-1)
    atomics = torch.nn.functional.one_hot(types, 9).float().requires_grad_(True)
    probs = torch.softmax(atomics / temperature, dim=-1)
    h = probs @ head.atom_emb.weight
    h = h * mask_long.unsqueeze(-1)
    h, _ = head.gnn.forward(h, coords, (~pad))
    h = h * mask_long.unsqueeze(-1)
    h = h.sum(1) / mask_long.sum(1, keepdim=True).clamp_min(1)
    pred = torch.sigmoid(head.predictor(h))
    val = reward_of(pred, targets, kind).sum()
    g, = torch.autograd.grad(val, [atomics])
    ntypes = 9 if allow_dummy else N_REAL
    allowed = list(range(ntypes)) if allowed is None else list(allowed)
    cur = g.gather(-1, types.unsqueeze(-1))
    gain = (g - cur)[..., allowed]                    # [B, L, |allowed|]
    same = (types.unsqueeze(-1) == torch.as_tensor(allowed, device=g.device))
    gain = gain.masked_fill(same, -1e9).masked_fill(pad.unsqueeze(-1), -1e9)
    flat = gain.reshape(gain.shape[0], -1).argmax(-1)
    new_types = types.clone()
    for b in range(gain.shape[0]):
        a, j = divmod(int(flat[b]), len(allowed))
        new_types[b, a] = allowed[j]
    return new_types.detach()


def types_to_state(state, types):
    return TensorDict({
        "coords": state["coords"],
        "atomics": torch.nn.functional.one_hot(types, 9).float(),
        "padding_mask": state["padding_mask"],
    }, batch_size=state["padding_mask"].shape[0]).to(state.device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--structure-head", type=Path, default=DEFAULT_STRUCTURE_HEAD)
    p.add_argument("--geom-fp-dir", type=Path, default=DEFAULT_GEOM_FP_DIR)
    p.add_argument("--num-mols", type=int, default=256)
    p.add_argument("--oracle-mols", type=int, default=32)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--objective", nargs="+", default=["l2", "tanimoto"])
    p.add_argument("--eta", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    p.add_argument("--flip-set", default="all", choices=["all", "cnof"],
                   help="'cnof' restricts substitutions to C/N/O/F, whose covalent radii "
                        "(0.70-0.77 A) are close enough that retyping an atom without moving "
                        "it does not by itself break PoseBusters bond-length checks.")
    p.add_argument("--allow-dummy", action="store_true",
                   help="also allow real<->dummy substitutions (changes molecule size)")
    p.add_argument("--soft-atom-temperature", type=float, default=0.25)
    p.add_argument("--holdout-n", type=int, default=512)
    p.add_argument("--holdout-seed", type=int, default=20260929)
    p.add_argument("--tune-on-holdout", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    pbg = p.add_mutually_exclusive_group()
    pbg.add_argument("--pb", dest="pb", action="store_true", default=True)
    pbg.add_argument("--no-pb", dest="pb", action="store_false")
    p.add_argument("--pb-config", type=Path, default=DEFAULT_PB_CONFIG)
    p.add_argument("--out", type=Path, default=RESULTS / "rescue" / "terminal_generated.json")
    a = p.parse_args()

    install_fingerprint_reward()
    set_seed(a.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(a.device)
    model, stats, _ = load_geom_teacher(a.teacher_checkpoint, device, use_ema=True)
    require_linear_dfm_atoms(model)
    head = fp.load_structure_head(a.structure_head, device, a.soft_atom_temperature)
    normalizer = float(stats.get("coordinate_normalizer", 2.0))
    max_atoms = int(stats.get("max_num_atoms", 71))

    holdout = fp.holdout_target_indices(a.geom_fp_dir, "val", a.holdout_n, max_atoms,
                                        a.holdout_seed)
    targets, tinfo = fp.sample_targets(a.geom_fp_dir, "val", a.num_mols, max_atoms, a.seed,
                                       indices=holdout[: a.num_mols], return_info=True)
    targets = targets.to(device)

    # unguided arm: identical to the recorded unguided arm at this seed
    set_seed(a.seed)
    init_state = model._sample_noise_like_batch(batch_size=a.num_mols).to(device)
    set_seed(a.seed)
    t0 = time.time()
    base_state, _ = guided_sample(
        model, init_state, value_sampler="mfm", mfm_student=None, num_steps=a.sample_steps,
        mu=0.0, reward_scale=0.01, value_samples=0, value_glass_steps=10,
        reward_name="fingerprint", target_x=0.0, property_name="structure",
        property_target=targets, property_regressor=head, guide_atomics=True,
        guide_min_t=0.1, guide_max_t=0.9, guide_every=2,
        guidance_max_coord_rms=0.0, guidance_max_atom_rms=0.0, eps=1e-6, value_batch_size=8,
    )
    print(f"sampled {a.num_mols} molecules in {time.time() - t0:.1f}s", flush=True)

    posebusters = build_posebusters(a.pb_config) if a.pb else None
    pb_stats = dict(stats)
    pb_stats["atom_names"] = list(fp.GEOM_ATOM_NAMES)
    pb_stats["coordinate_normalizer"] = normalizer

    def score_state(st, pb=True):
        s = fp.score(st, targets, head, normalizer, device)
        if pb and posebusters is not None:
            s["pb_valid"] = float(pb_summary_for_state(
                st, data_stats=pb_stats, posebusters=posebusters, sanitize=True
            ).get("pb_valid", 0.0))
        return s

    # hard one-hot terminal state: the composition the metric actually sees
    hard0 = types_to_state(base_state, base_state["atomics"].argmax(-1))
    base_score = score_state(hard0)
    base_per = np.asarray(base_score["tanimoto_real_per_mol"])
    base_score["atom_type_fractions"] = {
        fp.GEOM_ATOM_NAMES[i]: round(float(
            (hard0["atomics"].argmax(-1)[~hard0["padding_mask"]] == i).sum())
            / max(int((~hard0["padding_mask"]).sum()), 1), 5) for i in range(9)}
    print(f"base counted FP2 Tanimoto {base_score['tanimoto_real_mean']:.5f} "
          f"pb={base_score['pb_valid']:.4f}", flush=True)

    allowed = [0, 1, 2, 3] if a.flip_set == "cnof" else None
    rng = np.random.default_rng(a.seed)
    out = {
        "_what": __doc__.split("\n")[0],
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": str(device), "num_mols": a.num_mols, "seed": a.seed,
        "sample_steps": a.sample_steps, "rounds": a.rounds,
        "allow_dummy": bool(a.allow_dummy), "flip_set": a.flip_set,
        "target_info_selection": tinfo["selection"],
        "base": base_score, "arms": {},
    }

    base_types = hard0["atomics"].argmax(-1)
    base_pad = hard0["padding_mask"]

    def type_hist(st):
        t = st["atomics"].argmax(-1)[~st["padding_mask"]]
        n = int(t.numel())
        return {fp.GEOM_ATOM_NAMES[i]: round(float((t == i).sum()) / max(n, 1), 5)
                for i in range(9)}

    def record(name, st):
        s = score_state(st)
        s["atom_type_fractions"] = type_hist(st)
        s["frac_atoms_retyped_vs_base"] = float(
            ((st["atomics"].argmax(-1) != base_types) & (~base_pad)).float().sum()
            / (~base_pad).float().sum().clamp_min(1))
        per = np.asarray(s["tanimoto_real_per_mol"])
        d = per - base_per
        s["paired_gain"] = float(d.mean())
        s["paired_gain_sem"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        s["frac_improved"] = float((d > 0).mean())
        s.pop("tanimoto_real_per_mol", None)
        s.pop("valid_per_mol", None)
        out["arms"][name] = s
        # write after every arm, so a failure in a later arm (e.g. the oracle
        # ceiling, which is the slowest and least essential) cannot lose the ones
        # already measured
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))
        print(f"  {name:<34s} T={s['tanimoto_real_mean']:.5f} "
              f"gain={s['paired_gain']:+.5f}+-{s['paired_gain_sem']:.5f} "
              f"pb={s['pb_valid']:.4f} gacc={s['guide_accuracy_mean']:.3f}", flush=True)

    for kind in a.objective:
        # ---- head_exact
        st = hard0
        for r in range(1, a.rounds + 1):
            nt = greedy_round(head, st, targets, kind, mode="head_exact", rng=rng,
                              allow_dummy=a.allow_dummy, allowed=allowed)
            st = types_to_state(st, nt)
            record(f"head_exact/{kind}/round{r}", st)
        # ---- head_lin
        st = hard0
        for r in range(1, a.rounds + 1):
            nt = lin_round(head, st, targets, kind, a.soft_atom_temperature,
                           a.allow_dummy, allowed=allowed)
            st = types_to_state(st, nt)
            record(f"head_lin/{kind}/round{r}", st)
        # ---- grad_step (the sampler's own rule)
        coords = base_state["coords"]
        pad = base_state["padding_mask"]
        mask_long = (~pad).long()
        types = base_state["atomics"].argmax(-1)
        atomics = torch.nn.functional.one_hot(types, 9).float().requires_grad_(True)
        probs = torch.softmax(atomics / a.soft_atom_temperature, dim=-1)
        h = probs @ head.atom_emb.weight
        h = h * mask_long.unsqueeze(-1)
        h, _ = head.gnn.forward(h, coords, (~pad))
        h = h * mask_long.unsqueeze(-1)
        h = h.sum(1) / mask_long.sum(1, keepdim=True).clamp_min(1)
        val = reward_of(torch.sigmoid(head.predictor(h)), targets, kind).sum()
        g, = torch.autograd.grad(val, [atomics])
        g = g * (~pad).unsqueeze(-1)
        gn = g / g.square().mean().clamp_min(1e-30).sqrt()
        for e in a.eta:
            sc = atomics.detach() + e * gn
            keep = allowed if allowed is not None else list(
                range(9 if a.allow_dummy else N_REAL))
            # an atom whose current type is outside the allowed set keeps it
            cur_ok = torch.zeros_like(sc, dtype=torch.bool)
            cur_ok.scatter_(-1, types.unsqueeze(-1), True)
            colmask = torch.zeros(sc.shape[-1], dtype=torch.bool, device=sc.device)
            colmask[torch.as_tensor(keep, device=sc.device)] = True
            sc = sc.masked_fill(~(colmask | cur_ok), -1e9)
            nt = sc.argmax(-1)
            flipped = float(((nt != types) & (~pad)).float().sum() / (~pad).float().sum())
            st = types_to_state(base_state, nt)
            record(f"grad_step/{kind}/eta{e}", st)
            out["arms"][f"grad_step/{kind}/eta{e}"]["frac_atoms_flipped"] = flipped

    # ---- random control
    st = hard0
    for r in range(1, a.rounds + 1):
        nt = greedy_round(head, st, targets, "l2", mode="random", rng=rng,
                          allow_dummy=a.allow_dummy)
        st = types_to_state(st, nt)
        record(f"random/round{r}", st)

    # ---- oracle ceiling on a subset (uses the true counted FP2; not a method)
    if a.oracle_mols > 0:
      try:
        k = min(a.oracle_mols, a.num_mols)
        sub = TensorDict({"coords": hard0["coords"][:k], "atomics": hard0["atomics"][:k],
                          "padding_mask": hard0["padding_mask"][:k]}, batch_size=k).to(device)
        sub_t = targets[:k]
        gen_fps, _, _ = fp.state_to_binary_fps(sub, normalizer)
        cur = sub
        oracle = {"base": float(fp.tanimoto_similarity(gen_fps, sub_t.cpu()).mean()),
                  "rounds": []}
        for r in range(a.rounds):
            nt = greedy_round(head, cur, sub_t, "l2", mode="oracle", rng=rng,
                              allow_dummy=a.allow_dummy, target_fps=sub_t.cpu(),
                              normalizer=normalizer, allowed=allowed)
            cur = types_to_state(cur, nt)
            gf, _, _ = fp.state_to_binary_fps(cur, normalizer)
            oracle["rounds"].append(
                float(fp.tanimoto_similarity(gf, sub_t.cpu()).mean()))
            print(f"  oracle round{r+1} (n={k}) T={oracle['rounds'][-1]:.5f}", flush=True)
        out["oracle_ceiling"] = oracle
      except Exception as exc:          # never lose the measured arms to the ceiling stage
        out["oracle_ceiling_error"] = repr(exc)
        print(f"  oracle stage failed (arms above are unaffected): {exc!r}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
