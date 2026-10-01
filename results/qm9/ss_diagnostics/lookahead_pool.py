"""Decompose the multiMFM-SS gain: look-ahead pool quality vs. guide selection.

Diagnostic for docs/ss_guide_oracle_gap.md. It reruns the *exact* steer-search
trajectory of the 3-seed rerun (arguments read from the stored ``summary.json``,
only ``--num-samples`` reduced) and then, on the *same* candidate pool, measures

  endpoint      the guided trajectory endpoint (t = 1), no selection
  random        a uniformly random look-ahead from the pool
  random_pbv    a uniformly random *PoseBusters-valid* look-ahead
  guide_sel     Algorithm 3: the PB-valid look-ahead with the lowest guide error
  oracle_sel    the same with the ORACLE as scorer -- reported ONLY as the
                diagnostic upper bound on what selection could buy; it is never
                used to produce any number in the paper (selecting on the
                evaluator is forbidden)

scored with both regressors on identical tensors. ``random_pbv -> guide_sel`` is
how much the guide-argmin actually buys in oracle terms; ``endpoint -> random_pbv``
is how much comes from the pool being look-aheads rather than the endpoint.
Also dumps the pool-level joint distribution of (guide error, oracle error).

    ~/micromamba/envs/multimfm/bin/python results/qm9/ss_diagnostics/lookahead_pool.py \
        --property gap --num-samples 64 --pool-subsample 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))


from multimfm.glass_guidance import (  # noqa: E402
    load_flow_model,
    load_mfm_student,
    predicted_property,
    require_linear_dfm_atoms,
    seed_all,
)
from multimfm.nfe import NFECounter, attach_standard  # noqa: E402
from multimfm.property_targets import build_or_load_histograms, sample_property_target  # noqa: E402
from multimfm.qm9_data import choose_device, load_tfg_regressor  # noqa: E402
from multimfm.qm9_eval import PBPool, decode, make_state  # noqa: E402
from multimfm.steer_search import LookaheadStore, guided_sample_collect, lazy_select  # noqa: E402
from multimfm.qm9_results import HARTREE_TO_MEV, UNITS  # noqa: E402
from tabasco.chem.convert import MoleculeConverter  # noqa: E402

ENERGY = {"gap", "homo", "lumo"}


def stored_args(prop: str, seed: int) -> dict:
    p = REPO / f"results/qm9/rerun_3seed/runs/ss/{prop}_s{seed}/summary.json"
    return json.loads(p.read_text())["args"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--property", default="gap")
    ap.add_argument("--run-seed", type=int, default=1, help="which stored rerun to copy args from")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--pool-subsample", type=int, default=24,
                    help="random look-aheads per molecule to score (PB cost ~ n x this)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pb-workers", type=int, default=16)
    ap.add_argument("--sample-steps", type=int, default=0,
                    help="override the stored ODE step count (0 = keep it; only for a "
                         "reduced-scale CPU check, which is labelled in the output)")
    ap.add_argument("--value-samples", type=int, default=0,
                    help="override the stored posterior sample count (0 = keep it)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file name, e.g. _cpu_reduced")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "results/qm9/ss_diagnostics")
    a = ap.parse_args(argv)

    sa = stored_args(a.property, a.run_seed)
    args = SimpleNamespace(**sa)
    args.checkpoint = Path(sa["checkpoint"])
    args.mfm_checkpoint = Path(sa["mfm_checkpoint"])
    args.hist_cache = Path(sa["hist_cache"])
    args.posebusters_config = Path(sa["posebusters_config"])
    args.num_samples = a.num_samples
    args.device = a.device
    args.pb_workers = a.pb_workers
    args.lookahead_source = "mfm"
    args.lookahead_topk = 0            # keep the whole pool
    args.diagnostic_variants = False
    args.force_rebuild_hist = False
    args.reduced = bool(a.sample_steps or a.value_samples)
    if a.sample_steps:
        args.sample_steps = a.sample_steps
    if a.value_samples:
        args.value_samples = a.value_samples

    scale = HARTREE_TO_MEV if a.property in ENERGY else 1.0
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

    hist = build_or_load_histograms(args, max_len=int(data_stats["max_num_atoms"]))
    seed_all(args.seed)
    init_state = model._sample_noise_like_batch(batch_size=args.num_samples).to(device)
    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
    trng = np.random.default_rng(args.target_seed)
    target_values = [sample_property_target(hist, int(l), trng) for l in lengths]
    targets = torch.tensor(target_values, dtype=torch.float32, device=device)
    B = args.num_samples
    L, A = init_state["coords"].shape[1], init_state["atomics"].shape[-1]

    store = LookaheadStore(B, L, A, 0, kind="mfm", score=args.lookahead_score,
                           t_min=args.mfm_select_min_t, t_max=args.mfm_select_max_t)
    seed_all(args.seed)
    final_state, info = guided_sample_collect(
        model, mfm_student, init_state, args=args, targets=targets, guide_reg=guide_reg,
        nfe=nfe, stores={"mfm": store})
    coords_la, atomics_la, err_la, t_la = store.tensors()
    C = err_la.shape[1]
    print(f"pool: {C} look-aheads per molecule, {B} molecules")

    mask = init_state["padding_mask"]
    sanitize = not args.no_sanitize
    converter = MoleculeConverter(atom_names=list(data_stats["atom_names"]),
                                  dataset_normalizer=float(
                                      data_stats.get("coordinate_normalizer", 1.0)))
    pool = PBPool(args.posebusters_config, args.pb_workers)
    tgt_cpu = targets.detach().cpu()

    def score(coords, atomics):
        st = make_state(coords.to(device), atomics.to(device), mask)
        with torch.no_grad():
            g = predicted_property(st, property_name=a.property,
                                   property_regressor=guide_reg, soft=False).cpu()
            o = predicted_property(st, property_name=a.property,
                                   property_regressor=oracle_reg, soft=False).cpu()
        pb = np.array(pool.valid(decode(st, converter, sanitize=sanitize)), dtype=bool)
        return g, o, pb

    rows = {}

    def record(name, g, o, pb, t_sel=None, extra=None):
        eg, eo = (g - tgt_cpu).abs() * scale, (o - tgt_cpu).abs() * scale
        rec = {"guide_mae": float(eg.mean()), "oracle_mae": float(eo.mean()),
               "oracle_median_ae": float(eo.median()),
               "disagreement_mae": float(((g - o).abs() * scale).mean()),
               "pb_valid": float(pb.mean()), "n": int(len(g))}
        if t_sel is not None:
            rec["selected_t_mean"] = float(np.mean(t_sel))
        if extra:
            rec.update(extra)
        rows[name] = rec
        print(f"{name:<12} guide {rec['guide_mae']:10.3f}  oracle {rec['oracle_mae']:10.3f}  "
              f"|g-o| {rec['disagreement_mae']:9.3f}  PB {rec['pb_valid']:.3f}")

    # --- endpoint
    ec, ea = final_state["coords"].detach().cpu(), final_state["atomics"].detach().cpu()
    g, o, pb = score(ec, ea)
    record("endpoint", g, o, pb, t_sel=[1.0] * B)

    # --- random pool subsample: joint (guide err, oracle err) over candidates
    rng = np.random.default_rng(12345)
    S = min(a.pool_subsample, C)
    idx = np.stack([rng.choice(C, size=S, replace=False) for _ in range(B)])  # [B,S]
    pool_g = np.zeros((B, S)); pool_o = np.zeros((B, S)); pool_pb = np.zeros((B, S), bool)
    for s in range(S):
        col = torch.as_tensor(idx[:, s])
        rows_i = torch.arange(B)
        gg, oo, pp = score(coords_la[rows_i, col], atomics_la[rows_i, col])
        pool_g[:, s], pool_o[:, s], pool_pb[:, s] = gg.numpy(), oo.numpy(), pp
    eg_pool = np.abs(pool_g - tgt_cpu.numpy()[:, None]) * scale
    eo_pool = np.abs(pool_o - tgt_cpu.numpy()[:, None]) * scale
    rows["random"] = {
        "guide_mae": float(eg_pool.mean()), "oracle_mae": float(eo_pool.mean()),
        "oracle_median_ae": float(np.median(eo_pool)),
        "disagreement_mae": float((np.abs(pool_g - pool_o) * scale).mean()),
        "pb_valid": float(pool_pb.mean()), "n": int(B * S),
        "selected_t_mean": float(t_la.numpy()[np.arange(B)[:, None], idx].mean()),
    }
    print(f"{'random':<12} guide {rows['random']['guide_mae']:10.3f}  "
          f"oracle {rows['random']['oracle_mae']:10.3f}  "
          f"|g-o| {rows['random']['disagreement_mae']:9.3f}  PB {rows['random']['pb_valid']:.3f}")

    # first PB-valid of the random subsample (order is random -> unbiased)
    sel_r, gr, orr, pbr, tr = [], [], [], [], []
    for b in range(B):
        w = np.nonzero(pool_pb[b])[0]
        j = int(w[0]) if len(w) else None
        if j is None:
            gr.append(float(g[b])); orr.append(float(o[b])); pbr.append(bool(pb[b])); tr.append(1.0)
        else:
            gr.append(pool_g[b, j]); orr.append(pool_o[b, j]); pbr.append(True)
            tr.append(float(t_la[b, int(idx[b, j])]))
        sel_r.append(j)
    record("random_pbv", torch.tensor(gr, dtype=torch.float32),
           torch.tensor(orr, dtype=torch.float32), np.array(pbr), t_sel=tr,
           extra={"frac_no_pb_valid_in_subsample": float(np.mean([s is None for s in sel_r]))})

    # --- Algorithm 3: guide-argmin with the lazy PB walk (the reported SS number)
    with torch.no_grad():
        pred_end = predicted_property(final_state, property_name=a.property,
                                      property_regressor=guide_reg, soft=False)
    err_end = (pred_end - targets).abs().float().cpu()
    sel, resolved = lazy_select(err_la=err_la, t_la=t_la, coords_la=coords_la,
                               atomics_la=atomics_la, err_end=err_end,
                               final_state=final_state, mask=mask, converter=converter,
                               sanitize=sanitize, pool=pool)
    rows_i = torch.arange(B)
    safe = sel.clamp(max=max(C - 1, 0))
    is_end = sel == C
    cc = torch.where(is_end[:, None, None], ec, coords_la[rows_i, safe])
    aa = torch.where(is_end[:, None, None], ea, atomics_la[rows_i, safe])
    g2, o2, pb2 = score(cc, aa)
    t2 = torch.where(is_end, torch.ones(B), t_la[rows_i, safe]).numpy()
    record("guide_sel", g2, o2, pb2, t_sel=t2,
           extra={"frac_selected_lookahead": float((~is_end).float().mean()),
                  "frac_with_valid_candidate": float(resolved.float().mean())})

    # --- diagnostic-only upper bound: argmin of the ORACLE error over the SAME
    #     random PB-valid subsample (never used for any reported number)
    go, oo_, pbo, to = [], [], [], []
    for b in range(B):
        w = np.nonzero(pool_pb[b])[0]
        if not len(w):
            go.append(float(g[b])); oo_.append(float(o[b])); pbo.append(bool(pb[b])); to.append(1.0)
            continue
        j = int(w[np.argmin(eo_pool[b, w])])
        go.append(pool_g[b, j]); oo_.append(pool_o[b, j]); pbo.append(True)
        to.append(float(t_la[b, int(idx[b, j])]))
    record("oracle_sel_subsample", torch.tensor(go, dtype=torch.float32),
           torch.tensor(oo_, dtype=torch.float32), np.array(pbo), t_sel=to,
           extra={"note": "diagnostic upper bound over the random subsample only; "
                          "selecting on the evaluator is never reported"})
    pool.close()

    # --- pool-level winner's-curse geometry
    flat_g, flat_o = eg_pool.reshape(-1), eo_pool.reshape(-1)
    geom = {
        "pool_err_corr": float(np.corrcoef(flat_g, flat_o)[0, 1]),
        "pool_pred_corr": float(np.corrcoef(pool_g.reshape(-1), pool_o.reshape(-1))[0, 1]),
        "pool_guide_mae": float(flat_g.mean()),
        "pool_oracle_mae": float(flat_o.mean()),
        "pool_disagreement_mae": float((np.abs(pool_g - pool_o) * scale).mean()),
        "min_guide_err_in_subsample": float(eg_pool.min(axis=1).mean()),
        "oracle_err_at_argmin_guide_in_subsample": float(
            eo_pool[np.arange(B), eg_pool.argmin(axis=1)].mean()),
        "min_oracle_err_in_subsample": float(eo_pool.min(axis=1).mean()),
    }
    out = {
        "property": a.property, "unit": UNITS[a.property], "scale_from_raw": scale,
        "num_molecules": B, "pool_size_per_molecule": int(C),
        "reduced_scale": bool(args.reduced),
        "pool_subsample": int(S),
        "stored_args_from": f"results/qm9/rerun_3seed/runs/ss/{a.property}_s{a.run_seed}",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rows": rows, "pool_geometry": geom,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    p = a.out_dir / f"lookahead_pool_{a.property}{a.tag}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(geom, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
