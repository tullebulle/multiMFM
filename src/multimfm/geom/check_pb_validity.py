"""PoseBusters validity of a GEOM-Drugs base (teacher) checkpoint.

This is the "Unsteered (%)" column of Table 21 for GEOM-Drugs: generate
molecules with the frozen base model and report the fraction that passes every
PoseBusters check, plus the per-check pass rates. Reuses the same sampling + PB
pipeline as training validation, so the number is comparable to ``val/pb_valid``.

Two PoseBusters configurations are in play. The recorded teacher PB curve
(``results/geom/teacher_pb_curve.csv``) was measured with tabasco's training
config ``src/tabasco/utils/posebusters_no_strain.yaml``; Table 3 / Table 21 use
the GEOM experimental config ``configs/geom/posebusters_geom.yaml`` (same checks
plus non-aromatic ring non-flatness), which is the default here. Pass
``--pb-config src/tabasco/utils/posebusters_no_strain.yaml`` to reproduce the
curve.

Examples::

    # default teacher, 200 molecules at 200 integration steps, EMA weights
    python -m multimfm.geom.check_pb_validity --ema

    # a specific checkpoint, more samples, writing a JSON record
    python -m multimfm.geom.check_pb_validity --checkpoint checkpoints/geom/teacher/... \\
        --num-samples 500 --num-steps 200 --ema --out results/geom/runs/pb_teacher.json

Ported from ``geom-drug-tabasco/validation_teacher/check_pb_validity.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from multimfm.geom.paths import CHECKPOINTS, DEFAULT_PB_CONFIG, DEFAULT_TEACHER
from multimfm.geom.teacher import infer_arch, load_geom_teacher
from multimfm.posebusters_compat import install_posebusters_compat
from multimfm.train_base_flow import compute_pb_summary, sample_molecules


def find_latest_checkpoint(search_dir: Path) -> Path:
    ckpts = [p for p in search_dir.rglob("*.ckpt") if p.name != "last.ckpt"]
    if not ckpts:
        ckpts = list(search_dir.rglob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"No .ckpt found under {search_dir}")
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help=f"Checkpoint path. Default: {DEFAULT_TEACHER.relative_to(DEFAULT_TEACHER.parents[3])}, "
                        "or the most recent *.ckpt under checkpoints/geom/ if that is absent.")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--num-steps", type=int, default=200,
                   help="Euler integration steps for generation (matches validation).")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--num-heads", type=int, default=None,
                   help="Attention heads (not inferable from weights). "
                        "Default: 8 for hidden>=128 else 4.")
    p.add_argument("--pb-config", type=Path, default=DEFAULT_PB_CONFIG,
                   help="PoseBusters config. Default: the GEOM experimental config.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="Optional result JSON path.")
    p.add_argument("--ema", action="store_true",
                   help="Evaluate the EMA weights (from optimizer_states) instead of the "
                        "raw online weights. This matches the weights used during training "
                        "validation and is the recommended teacher metric.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = DEFAULT_TEACHER if DEFAULT_TEACHER.exists() else find_latest_checkpoint(CHECKPOINTS)
    print(f"# checkpoint: {ckpt_path}")

    torch.manual_seed(args.seed)
    model, stats, margs = load_geom_teacher(
        ckpt_path, args.device, use_ema=args.ema, num_heads=args.num_heads
    )
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hidden_dim, num_layers = infer_arch(raw["state_dict"])
    print(f"# epoch={raw.get('epoch')}  global_step={raw.get('global_step')}  "
          f"atom_dim={stats.get('atom_dim')}  max_atoms={stats.get('max_num_atoms')}")
    print(f"# arch (inferred): hidden_dim={hidden_dim} num_layers={num_layers} "
          f"num_heads={margs.num_heads}")
    print(f"# weights: {model.geom_weights}")

    print(f"# generating {args.num_samples} molecules @ {args.num_steps} steps ...")
    mols, mol_indices, total = sample_molecules(
        model, stats=stats, num_samples=args.num_samples,
        batch_size=args.batch_size, num_steps=args.num_steps, sanitize=True,
    )

    install_posebusters_compat()
    from posebusters import PoseBusters  # noqa: PLC0415

    with open(args.pb_config, encoding="utf-8") as fh:
        pb = PoseBusters(config=yaml.safe_load(fh))
    summary = compute_pb_summary(pb, mols, mol_indices, total)

    print("\n===== PoseBusters validity =====")
    print(f"generated              : {summary['num_generated']}")
    print(f"RDKit-converted        : {summary['num_converted']} "
          f"({summary['conversion_rate']:.3f})")
    print(f"\n>>> PB-VALIDITY (pass all checks) = {summary['pb_valid']:.4f} <<<\n")
    print("per-check pass rate (of all generated):")
    for k in sorted(summary):
        if k.startswith("pb_") and k not in ("pb_valid", "pb_intersection"):
            print(f"  {k:<32} {summary[k]:.3f}")

    if args.out is not None:
        record = {
            "checkpoint": str(ckpt_path),
            "weights": model.geom_weights,
            "epoch": raw.get("epoch"),
            "global_step": raw.get("global_step"),
            "num_steps": args.num_steps,
            "num_samples": args.num_samples,
            "hidden_dim": hidden_dim,
            "pb_config": str(args.pb_config),
            "seed": args.seed,
            "pb_valid": float(summary["pb_valid"]),
            "conversion_rate": float(summary["conversion_rate"]),
            **{k: float(summary[k]) for k in summary
               if k.startswith("pb_") and k not in ("pb_valid", "pb_intersection")},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
        print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
