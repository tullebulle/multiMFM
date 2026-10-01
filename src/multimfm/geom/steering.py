"""Shared driver for GEOM-Drugs structural-fingerprint steering (paper Table 3).

Two look-ahead samplers share one driver, exactly as in the source repo:

* ``value_sampler="glass"`` -- integrate the teacher's GLASS posterior field for
  the look-ahead endpoints (``multimfm.geom.glass_steering``);
* ``value_sampler="mfm"``  -- integrate the distilled meta flow map instead
  (``multimfm.geom.mfm_steering``), which is the **multiMFM** row of Table 3.

Both add ``mu * grad_x log E[exp(lambda * r(X_1))]`` to the base velocity, with
``r(X_1) = -||sigmoid(head(X_1)) - c||^2`` on a target fingerprint ``c``
(:mod:`multimfm.geom.reward`), then report the OpenBabel-FP2 Tanimoto similarity
of the generated molecules to their targets, guided vs unguided, plus
PoseBusters validity.

Ported from ``geom-drug-tabasco/glass/{glass,mfm}_fingerprint_steering.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import torch

from multimfm.geom import fingerprint as fp
from multimfm.geom.paths import (
    DEFAULT_GEOM_FP_DIR,
    DEFAULT_PB_CONFIG,
    DEFAULT_STRUCTURE_HEAD,
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    RESULTS,
)
from multimfm.geom.mu_schedule import DEFAULT_EPS_T, MU_SCHEDULES, install_mu_schedule
from multimfm.geom.reward import install_fingerprint_reward
from multimfm.geom.teacher import load_geom_teacher
from multimfm.glass_guidance import (
    build_posebusters,
    guided_sample,
    load_mfm_student,
    pb_summary_for_state,
)
from multimfm.train_base_flow import choose_device, set_seed
from tabasco.sample.glass import require_linear_dfm_atoms

__all__ = ["add_common_arguments", "run_steering", "main"]


def add_common_arguments(p: argparse.ArgumentParser, *, value_sampler: str) -> argparse.ArgumentParser:
    """CLI options shared by the GLASS and MFM drivers.

    Defaults reproduce the recorded runs: the MFM driver's defaults are those of
    ``glass/run_mfm_capgap_steering.sbatch`` (job 35670937, Table 3), the GLASS
    driver's those of ``glass/run_fingerprint_steering.sbatch``.
    """
    is_mfm = value_sampler == "mfm"
    p.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--structure-head", type=Path, default=DEFAULT_STRUCTURE_HEAD)
    p.add_argument("--geom-fp-dir", type=Path, default=DEFAULT_GEOM_FP_DIR)
    p.add_argument("--target-split", default="val")
    p.add_argument("--target-selection", choices=fp.TARGET_SELECTIONS, default="filter",
                   help="'filter' skips targets larger than the generator's "
                        "max_num_atoms (default); 'legacy' reproduces the original "
                        "behaviour, which crashes on such a target.")
    p.add_argument("--holdout-n", type=int, default=0,
                   help="Size of the held-out target pool that hyperparameters are "
                        "selected on. Evaluation draws exclude it; >0 keeps the "
                        "selection set disjoint from the evaluation targets.")
    p.add_argument("--holdout-seed", type=int, default=20260929,
                   help="Seed defining the held-out target pool.")
    p.add_argument("--tune-on-holdout", action="store_true",
                   help="Take the targets FROM the held-out pool (selection runs) "
                        "instead of excluding it (evaluation runs).")
    p.add_argument("--num-mols", type=int, default=64)
    p.add_argument("--sample-steps", type=int, default=100 if is_mfm else 50)
    p.add_argument("--value-samples", type=int, default=4)
    p.add_argument("--value-batch-size", type=int, default=8)
    p.add_argument("--guide-every", type=int, default=2)
    p.add_argument("--guide-min-t", type=float, default=0.1)
    p.add_argument("--guide-max-t", type=float, default=0.9)
    p.add_argument("--mu", type=float, default=0.5, help="Guidance strength.")
    p.add_argument("--mu-schedule", choices=MU_SCHEDULES, default="constant",
                   help="Coefficient in front of the value gradient. 'constant' (default, and "
                        "what every recorded run used) adds mu * grad V * dt. 'paper' adds "
                        "mu * sigma_t^2/2 * grad V * dt with sigma_t^2/2 = (1-t)/t, the "
                        "coefficient the paper's guided ODE states; mu is then an overall "
                        "scale. See multimfm.geom.mu_schedule.")
    p.add_argument("--mu-schedule-eps-t", type=float, default=DEFAULT_EPS_T,
                   help="Lower clamp on t when evaluating (1-t)/t, to guard the t->0 blow-up "
                        "(inert inside the recorded window t in [0.1, 0.9]).")
    p.add_argument("--reward-lambda", type=float, default=0.05 if is_mfm else 0.1,
                   help="lambda in exp(-lambda * L2); passed as GLASS reward_scale.")
    p.add_argument("--max-atoms", type=int, default=None,
                   help="cap heavy atoms for target selection; 30 matches TFG-Flow's GEOM "
                        "pipeline, whose generator is capped there while ours allows 71.")
    p.add_argument("--reward-kind", choices=["fingerprint", "fingerprint_tanimoto"],
                   default="fingerprint",
                   help="'fingerprint' is -||p-c||^2, the objective used so far; "
                        "'fingerprint_tanimoto' is the differentiable Tanimoto, which ranks "
                        "candidates at Spearman 0.708 against the counted metric where the "
                        "L2 form manages 0.375.")
    p.add_argument("--soft-atom-temperature", type=float, default=0.25)
    p.add_argument("--no-guide-atomics", action="store_true",
                   help="Guide coordinates only (default also steers composition).")
    p.add_argument("--guidance-max-coord-rms", type=float, default=0.0)
    p.add_argument("--guidance-max-atom-rms", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--no-ema", action="store_true", help="Use raw (not EMA) teacher weights.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tag", default="mfm_run" if is_mfm else "run")
    p.add_argument("--output-dir", type=Path,
                   default=RESULTS / ("runs/mfm_steering" if is_mfm else "runs/glass_steering"))
    p.add_argument("--pb-config", type=Path, default=DEFAULT_PB_CONFIG,
                   help="PoseBusters config used for pb_valid (GEOM configuration).")
    pb = p.add_mutually_exclusive_group()
    pb.add_argument("--pb", dest="pb", action="store_true", default=is_mfm,
                    help="Compute PoseBusters validity (default: on for multiMFM).")
    pb.add_argument("--no-pb", dest="pb", action="store_false", help="Skip PoseBusters.")
    if is_mfm:
        p.add_argument("--mfm-checkpoint", type=Path, default=DEFAULT_STUDENT)
        p.add_argument("--mfm-key", default="student", help="State-dict key in the MFM ckpt.")
        p.add_argument("--mfm-diagonal", action="store_true",
                       help="Use the diagonal (single-time) flow-map call.")
        p.add_argument("--mfm-steps", type=int, default=10,
                       help="Integration steps for each meta-flow-map endpoint.")
    else:
        p.add_argument("--value-glass-steps", type=int, default=6,
                       help="Integration steps for each GLASS look-ahead endpoint.")
    return p


def _run_one(model, student, head, init_state, targets, args, *, mu, value_samples,
             value_sampler: str):
    """One ``guided_sample`` call (mu=0 / value_samples=0 gives the unguided run)."""
    return guided_sample(
        model,
        init_state,
        value_sampler=value_sampler,
        mfm_student=student,
        mfm_diagonal=bool(getattr(args, "mfm_diagonal", False)),
        num_steps=args.sample_steps,
        mu=mu,
        reward_scale=args.reward_lambda,
        value_samples=value_samples,
        value_glass_steps=(
            args.mfm_steps if value_sampler == "mfm" else args.value_glass_steps
        ),
        reward_name=getattr(args, "reward_kind", "fingerprint"),
        target_x=0.0,
        property_name="structure",
        property_target=targets,
        property_regressor=head,
        guide_atomics=not args.no_guide_atomics,
        guide_min_t=args.guide_min_t,
        guide_max_t=args.guide_max_t,
        guide_every=args.guide_every,
        guidance_max_coord_rms=args.guidance_max_coord_rms,
        guidance_max_atom_rms=args.guidance_max_atom_rms,
        eps=args.eps,
        value_batch_size=args.value_batch_size,
    )


def run_steering(args: argparse.Namespace, *, value_sampler: str) -> dict:
    """Run one unguided + one guided sample and return the result record."""
    install_fingerprint_reward()
    install_mu_schedule(args.mu_schedule, args.mu_schedule_eps_t)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    model, stats, _ = load_geom_teacher(
        args.teacher_checkpoint, device, use_ema=not args.no_ema
    )
    require_linear_dfm_atoms(model)
    head = fp.load_structure_head(args.structure_head, device, args.soft_atom_temperature)
    normalizer = float(stats.get("coordinate_normalizer", 2.0))
    # TFG-Flow's GEOM pipeline caps molecules at 30 heavy atoms while ours allows 71, which
    # is the one protocol difference left between the two sets of numbers once the scoring
    # convention is accounted for (our conversion rate is 1.0, so averaging over valid
    # molecules and over all molecules give the same value). --max-atoms matches their regime.
    max_atoms = int(getattr(args, "max_atoms", None) or stats.get("max_num_atoms", 71))

    student = None
    if value_sampler == "mfm":
        # student_cfg is present in the checkpoint, so teacher_args is only a fallback.
        hidden_dim = model.net.linear_embed.weight.shape[0] if hasattr(model, "net") else 256
        teacher_args = SimpleNamespace(hidden_dim=hidden_dim, num_layers=16, num_heads=8)
        student = load_mfm_student(
            args.mfm_checkpoint,
            data_stats=stats,
            teacher_args=teacher_args,
            device=device,
            key=args.mfm_key,
        )

    # Held-out target pool: evaluation draws exclude it, selection runs draw from it,
    # so the (lambda, mu) selection and the evaluation never share a target.
    holdout = []
    if args.holdout_n > 0:
        holdout = fp.holdout_target_indices(
            args.geom_fp_dir, args.target_split, args.holdout_n, max_atoms, args.holdout_seed
        )
    if args.tune_on_holdout:
        if not holdout:
            raise SystemExit("--tune-on-holdout requires --holdout-n > 0")
        if args.num_mols > len(holdout):
            raise SystemExit(
                f"--num-mols {args.num_mols} exceeds the held-out pool ({len(holdout)})"
            )
        targets, target_info = fp.sample_targets(
            args.geom_fp_dir, args.target_split, args.num_mols, max_atoms, args.seed,
            indices=holdout[: args.num_mols], return_info=True,
        )
    else:
        targets, target_info = fp.sample_targets(
            args.geom_fp_dir,
            args.target_split,
            args.num_mols,
            max_atoms,
            args.seed,
            selection=args.target_selection,
            exclude=holdout or None,
            return_info=True,
        )
    targets = targets.to(device)

    extra = (
        f"mfm={Path(args.mfm_checkpoint).name} mfm_steps={args.mfm_steps} "
        if value_sampler == "mfm"
        else f"glass_steps={args.value_glass_steps} "
    )
    print(
        f"[{args.tag}] teacher={Path(args.teacher_checkpoint).name} "
        f"weights={model.geom_weights} {extra}normalizer={normalizer} "
        f"n_mols={args.num_mols} lambda={args.reward_lambda} mu={args.mu}"
        f"[{args.mu_schedule}] "
        f"guide_atomics={not args.no_guide_atomics} "
        f"targets={target_info['selection']}"
        f"(skipped {target_info['num_skipped']}, "
        f"holdout {'draw-from' if args.tune_on_holdout else 'excluded'} n={args.holdout_n})"
    )

    set_seed(args.seed)
    init_state = model._sample_noise_like_batch(batch_size=args.num_mols).to(device)

    set_seed(args.seed)
    unguided, _ = _run_one(model, student, head, init_state, targets, args,
                           mu=0.0, value_samples=0, value_sampler=value_sampler)
    set_seed(args.seed)
    guided, _ = _run_one(model, student, head, init_state, targets, args,
                         mu=args.mu, value_samples=args.value_samples,
                         value_sampler=value_sampler)

    unguided_score = fp.score(unguided, targets, head, normalizer, device)
    guided_score = fp.score(guided, targets, head, normalizer, device)

    # PoseBusters validity (pass-all-checks). Teacher stats lack atom_names/normalizer,
    # so supply the canonical GEOM vocab + /2.0 frame.
    if args.pb:
        pb_stats = dict(stats)
        pb_stats["atom_names"] = list(fp.GEOM_ATOM_NAMES)
        pb_stats["coordinate_normalizer"] = normalizer
        posebusters = build_posebusters(args.pb_config)
        for state, out in ((unguided, unguided_score), (guided, guided_score)):
            summary = pb_summary_for_state(
                state, data_stats=pb_stats, posebusters=posebusters, sanitize=True
            )
            out["pb_valid"] = float(summary.get("pb_valid", 0.0))
    else:
        unguided_score["pb_valid"] = None
        guided_score["pb_valid"] = None

    result = {
        "tag": args.tag,
        "value_sampler": value_sampler,
        # where this shard ran: CPU runs are true fp32, CUDA runs use TF32 matmuls
        # (torch.set_float32_matmul_precision("high")), so shards are comparable in
        # distribution but not bitwise across devices.
        "device": str(device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "host": socket.gethostname(),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "structure_head": str(args.structure_head),
        "weights": model.geom_weights,
        "reward_lambda": args.reward_lambda,
        "mu": args.mu,
        "mu_schedule": args.mu_schedule,
        "mu_schedule_eps_t": args.mu_schedule_eps_t,
        "guide_min_t": args.guide_min_t,
        "guide_max_t": args.guide_max_t,
        "guide_every": args.guide_every,
        "guide_atomics": not args.no_guide_atomics,
        "sample_steps": args.sample_steps,
        "value_samples": args.value_samples,
        "num_mols": args.num_mols,
        "seed": args.seed,
        "target_split": args.target_split,
        "target_selection": target_info["selection"],
        "targets_skipped": target_info["num_skipped"],
        "targets_excluded_holdout": target_info.get("num_excluded_holdout", 0),
        "holdout_n": args.holdout_n,
        "holdout_seed": args.holdout_seed,
        "tuned_on_holdout": bool(args.tune_on_holdout),
        "target_index_first": target_info["indices"][0],
        "target_index_last": target_info["indices"][-1],
        "unguided": unguided_score,
        "guided": guided_score,
        "tanimoto_gain": guided_score["tanimoto_real_mean"] - unguided_score["tanimoto_real_mean"],
    }
    if value_sampler == "mfm":
        result["mfm_checkpoint"] = str(args.mfm_checkpoint)
        result["mfm_steps"] = args.mfm_steps
    else:
        result["value_glass_steps"] = args.value_glass_steps

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "mfm_fingerprint_steering" if value_sampler == "mfm" else "fingerprint_steering"
    out = args.output_dir / f"{prefix}_{args.tag}.json"
    # Write atomically: the sharded runs are resumed by "output exists and is non-empty",
    # so a task preempted mid-write must not leave a truncated JSON behind.
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(out)

    def _pb(s):
        return f"{s['pb_valid']:.3f}" if s["pb_valid"] is not None else "n/a"

    label = "MFM" if value_sampler == "mfm" else "GLASS"
    print(f"\n===== {label} fingerprint steering result =====")
    print(f"  UNGUIDED  real-FP2 Tanimoto = {unguided_score['tanimoto_real_mean']:.4f} "
          f"(head {unguided_score['tanimoto_head_mean']:.3f}, "
          f"conv {unguided_score['conversion_rate']:.2f}, pb_valid {_pb(unguided_score)})")
    print(f"  GUIDED    real-FP2 Tanimoto = {guided_score['tanimoto_real_mean']:.4f} "
          f"(head {guided_score['tanimoto_head_mean']:.3f}, "
          f"conv {guided_score['conversion_rate']:.2f}, pb_valid {_pb(guided_score)})")
    print(f"  GAIN (guided - unguided)    = {result['tanimoto_gain']:+.4f}")
    print(f"  wrote {out}")
    return result


def main(value_sampler: str, doc: str | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, value_sampler=value_sampler)
    return run_steering(parser.parse_args(), value_sampler=value_sampler)
