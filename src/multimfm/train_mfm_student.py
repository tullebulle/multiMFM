"""Meta-flow-map (MFM) student training for QM9 (multi-stage GLASS distillation).

Trains ``TabascoFlowMap`` against the frozen base model's GLASS posterior
velocity. ``--stage`` selects the objective (1 = diagonal distillation, on real
endpoints or, with ``--rollout-condition-source generated``, on teacher-generated
conditioning states; 2 = ESD consistency + diagonal anchor; 3 = GLASS transition
distillation; 4 = GLASS rollout distillation). Periodically evaluates the
few-step posterior sampler against the many-step GLASS posterior (PoseBusters
validity) and checkpoints the EMA student every ``--eval-every`` steps.

The shipped student (``checkpoints/mfm_student/student_step_1000.pt``) is step
1000 of a stage-1 generated-condition phase initialised from an earlier (purged)
checkpoint; its full argument dict is stored in the checkpoint and the exact
command is in ``scripts/qm9/train_mfm.sbatch``.

Run:
  python -m multimfm.train_mfm_student --device cuda --steps 5000
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from multimfm import paths
from multimfm.glass_guidance import (  # noqa: E402
    build_posebusters,
    clip_guidance_rms,
    is_tfg_property,
    load_flow_model,
    pb_summary_for_state,
    production_step_with_optional_guidance,
    value_gradient,
)
from multimfm.glass_endpoints import (  # noqa: E402
    make_condition_state_from_endpoint,
    sample_glass_posterior,
)
from multimfm.qm9_data import load_qm9_tensors, load_tfg_regressor  # noqa: E402
from multimfm.property_targets import (  # noqa: E402
    build_or_load_histograms,
    sample_property_target,
)
from tabasco.data.utils import TensorDictCollator  # noqa: E402
from tabasco.flow.mfm_losses import (  # noqa: E402
    diagonal_distill_loss,
    esd_consistency_loss,
    glass_posterior_diagonal_distill_loss,
    glass_rollout_distill_loss,
    glass_transition_distill_loss,
)
from tabasco.models.components.flow_map_transformer import TabascoFlowMap  # noqa: E402
from tabasco.sample.flow_map import consistency_posterior_sample  # noqa: E402
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com  # noqa: E402

# Frozen QM9 teacher (base flow model). Overridable with --teacher-checkpoint.
CKPT = paths.BASE_FLOW_CKPT


def build_student(teacher, margs, hidden_dim=None, num_layers=None, num_heads=None,
                  block_conditioning="additive",
                  velocity_parametrization="endpoint",
                  encoder_depth=None) -> TabascoFlowMap:
    return TabascoFlowMap(
        spatial_dim=teacher.data_stats["spatial_dim"],
        atom_dim=teacher.data_stats["atom_dim"],
        num_heads=num_heads or margs.num_heads,
        num_layers=num_layers or margs.num_layers,
        hidden_dim=hidden_dim or margs.hidden_dim,
        cross_attention=not getattr(margs, "no_cross_attention", False),
        add_sinusoid_posenc=True,
        max_num_atoms=teacher.data_stats["max_num_atoms"],
        block_conditioning=block_conditioning,
        velocity_parametrization=velocity_parametrization,
        encoder_depth=encoder_depth,
    )


def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def infinite_batches(tensors, batch_size):
    loader = DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TensorDictCollator(),
        drop_last=True,
    )
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def sample_production_condition_state(
    model,
    *,
    batch_size: int,
    t_cond: torch.Tensor,
    num_steps: int,
) -> TensorDict:
    """Sample an unguided production-trajectory state at a scalar condition time."""
    if t_cond.ndim != 1 or t_cond.numel() == 0:
        raise ValueError("t_cond must be a non-empty 1D tensor")
    t_target = float(t_cond[0].detach().cpu())
    if not torch.allclose(t_cond, torch.full_like(t_cond, t_target), atol=1e-6):
        raise ValueError("generated condition source requires a scalar t_cond per batch")

    state = model._sample_noise_like_batch(batch_size=batch_size).to(t_cond.device)
    schedule = model._get_sample_schedule(num_steps).to(t_cond.device)
    for i in range(1, len(schedule)):
        t0 = float(schedule[i - 1])
        t1 = float(schedule[i])
        if t0 >= t_target:
            break
        dt_val = min(t1, t_target) - t0
        if dt_val <= 0.0:
            continue
        t = torch.full((batch_size,), t0, device=t_cond.device)
        dt = torch.full((batch_size,), dt_val, device=t_cond.device)
        pred = model._call_net(state, t)
        coords = model.coords_interpolant.step(state, pred, t, dt)
        atomics = model.atomics_interpolant.step(state, pred, t, dt)
        state = TensorDict(
            {
                "coords": mask_and_zero_com(coords, state["padding_mask"]),
                "atomics": apply_mask(atomics, state["padding_mask"]),
                "padding_mask": state["padding_mask"],
            },
            batch_size=state["padding_mask"].shape[0],
        ).to(t_cond.device)
    return state


def sample_guided_production_condition_state(
    model,
    init_state: TensorDict,
    *,
    t_cond: torch.Tensor,
    num_steps: int,
    mu: float,
    reward_scale: float,
    value_samples: int,
    value_glass_steps: int,
    property_name: str,
    property_target: torch.Tensor,
    property_regressor,
    guide_atomics: bool,
    guide_min_t: float,
    guide_max_t: float,
    guide_every: int,
    guidance_max_coord_rms: float,
    guidance_max_atom_rms: float,
    eps: float,
) -> TensorDict:
    """Sample a strongly guided production-trajectory state at scalar ``t_cond``."""
    if t_cond.ndim != 1 or t_cond.numel() == 0:
        raise ValueError("t_cond must be a non-empty 1D tensor")
    t_target = float(t_cond[0].detach().cpu())
    if not torch.allclose(t_cond, torch.full_like(t_cond, t_target), atol=1e-6):
        raise ValueError("guided condition source requires a scalar t_cond per batch")

    state = TensorDict(
        {
            "coords": init_state["coords"].clone(),
            "atomics": init_state["atomics"].clone(),
            "padding_mask": init_state["padding_mask"].clone(),
        },
        batch_size=init_state["padding_mask"].shape[0],
    ).to(t_cond.device)
    schedule = model._get_sample_schedule(num_steps).to(t_cond.device)
    for i in range(1, len(schedule)):
        t0 = float(schedule[i - 1])
        t1 = float(schedule[i])
        if t0 >= t_target:
            break
        dt_val = min(t1, t_target) - t0
        if dt_val <= 0.0:
            continue
        t = torch.full((state["coords"].shape[0],), t0, device=t_cond.device)
        dt = torch.full((state["coords"].shape[0],), dt_val, device=t_cond.device)
        should_guide = (
            mu != 0.0
            and value_samples > 0
            and guide_every > 0
            and (i - 1) % guide_every == 0
            and t0 >= guide_min_t
            and t0 <= guide_max_t
        )
        grad_coords = None
        grad_atomics = None
        if should_guide:
            with torch.enable_grad():
                grad_coords, grad_atomics, _ = value_gradient(
                    model,
                    state,
                    t,
                    value_sampler="glass",
                    posterior_samples=value_samples,
                    n_steps=value_glass_steps,
                    reward_name="target_property",
                    reward_scale=reward_scale,
                    target_x=0.0,
                    property_name=property_name,
                    property_target=property_target,
                    property_regressor=property_regressor,
                    guide_atomics=guide_atomics,
                    eps=eps,
                )
            grad_coords, _ = clip_guidance_rms(
                grad_coords,
                state["padding_mask"],
                guidance_max_coord_rms,
            )
            if grad_atomics is not None:
                grad_atomics, _ = clip_guidance_rms(
                    grad_atomics,
                    state["padding_mask"],
                    guidance_max_atom_rms,
                )
        with torch.no_grad():
            state = production_step_with_optional_guidance(
                model,
                state,
                t,
                dt,
                mu=mu,
                grad_coords=grad_coords,
                grad_atomics=grad_atomics if guide_atomics else None,
            )
    return state


@torch.no_grad()
def evaluate(
    student,
    teacher,
    eval_x1,
    data_stats,
    posebusters,
    device,
    t_conds,
    mfm_steps,
    glass_steps,
    *,
    diagonal: bool = False,
):
    rows = []
    for t_cond_val in t_conds:
        endpoint = eval_x1.clone().to(device)
        cond, t_cond = make_condition_state_from_endpoint(
            teacher, endpoint, t_cond=t_cond_val
        )
        # MFM few-step posterior
        mfm = consistency_posterior_sample(
            student, cond, t_cond, n_steps=mfm_steps, diagonal=diagonal
        )
        # GLASS many-step posterior (1 sample per condition)
        glass = sample_glass_posterior(
            teacher, cond, t_cond, samples_per_condition=1, num_steps=glass_steps
        )
        pb_mfm = pb_summary_for_state(
            mfm, data_stats=data_stats, posebusters=posebusters, sanitize=True
        )
        pb_glass = pb_summary_for_state(
            glass, data_stats=data_stats, posebusters=posebusters, sanitize=True
        )
        rows.append(
            {
                "t_cond": t_cond_val,
                "mfm_pb_valid": pb_mfm.get("pb_valid", 0.0),
                "mfm_conv": pb_mfm.get("conversion_rate", 0.0),
                "glass_pb_valid": pb_glass.get("pb_valid", 0.0),
                "glass_conv": pb_glass.get("conversion_rate", 0.0),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--partition", default="train_flow")
    ap.add_argument("--limit-data", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--t-max", type=float, default=0.99, help="clamp s and t_cond below 1")
    ap.add_argument("--endpoint-space-loss", action="store_true",
                    help="weight loss by (1-s)^2 (endpoint-space; use with endpoint param)")
    ap.add_argument("--diag-loss-type", choices=["mse", "adaptive"], default="mse",
                    help="diagonal target loss; adaptive matches the MFM v2 recipe")
    ap.add_argument("--diag-adaptive-p", type=float, default=0.5)
    ap.add_argument("--diag-adaptive-c", type=float, default=0.01)
    ap.add_argument("--diag-direction-weight", type=float, default=0.0,
                    help="extra masked 1-cosine loss on diagonal GLASS velocities")
    ap.add_argument("--diag-condition-jvp-weight", type=float, default=0.0,
                    help="finite-difference loss matching d v_diag / d x_cond")
    ap.add_argument("--diag-condition-jvp-direction-weight", type=float, default=0.0,
                    help="masked 1-cosine loss for finite-difference d v_diag / d x_cond")
    ap.add_argument("--diag-condition-jvp-eps", type=float, default=0.03,
                    help="per-sample RMS condition perturbation for diagonal sensitivity loss")
    ap.add_argument("--diag-condition-jvp-mode",
                    choices=["both", "coords", "atomics"],
                    default="both",
                    help="which conditioning modality to perturb for the diagonal sensitivity loss")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                    help=("1=diagonal distill; 2=ESD/JVP; "
                          "3=direct GLASS transition distill; "
                          "4=GLASS rollout distill"))
    ap.add_argument("--init-ckpt", default=None,
                    help="student checkpoint to initialize from (e.g. a Stage-1 run)")
    ap.add_argument("--skip-init-coord-head", action="store_true",
                    help="when loading --init-ckpt, leave the coordinate output head at initialization")
    ap.add_argument("--init-from-teacher", action="store_true",
                    help="initialize the student from the teacher's endpoint predictor")
    ap.add_argument("--esd-weight", type=float, default=1.0)
    ap.add_argument("--diag-weight", type=float, default=1.0)
    ap.add_argument("--coord-loss-weight", type=float, default=1.0,
                    help="weight for coordinate loss terms in diagonal/ESD/transition objectives")
    ap.add_argument("--atom-loss-weight", type=float, default=1.0,
                    help="weight for atom-state loss terms in diagonal/ESD/transition objectives")
    ap.add_argument("--esd-target-clip", type=float, default=0.0,
                    help="clamp the GLASS velocity target to +/- this (0=off)")
    ap.add_argument("--esd-delta-clip", type=float, default=0.0,
                    help="per-sample L2 cap for the ESD delta (u-s)*jvp before adding vss")
    ap.add_argument("--esd-loss-type", choices=["mse", "adaptive"], default="mse")
    ap.add_argument("--esd-adaptive-p", type=float, default=0.5)
    ap.add_argument("--esd-adaptive-c", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=500,
                    help="LR warmup steps (then cosine decay)")
    ap.add_argument("--anneal-frac", type=float, default=0.5,
                    help="fraction of steps over which the s->u jump grows to full")
    ap.add_argument("--max-jump", type=float, default=0.5,
                    help="cap on the inner-time jump (u - s) for the ESD term")
    ap.add_argument("--t-cond-power", type=float, default=1.0,
                    help="sample t_cond as rand()**power * t_max")
    ap.add_argument("--t-cond-zero-rate", type=float, default=0.0,
                    help="probability of forcing t_cond=0, matching MFM-style sampling")
    ap.add_argument("--t-cond-grid", default="",
                    help="optional comma-separated t_cond values sampled uniformly, e.g. 0.3,0.6,0.9")
    ap.add_argument("--t-cond-jitter", type=float, default=0.0,
                    help="uniform jitter radius for --t-cond-grid samples")
    ap.add_argument("--s-grid", default="",
                    help="optional comma-separated inner-time s values sampled uniformly")
    ap.add_argument("--s-grid-jitter", type=float, default=0.0,
                    help="uniform jitter radius for --s-grid samples")
    ap.add_argument("--s-power", type=float, default=1.0,
                    help="sample unconstrained s as rand()**power * t_max when --s-grid is unset")
    ap.add_argument("--transition-steps", default="1,2,4",
                    help="comma-separated coarse sampler grids for Stage 3, e.g. 1,2,4")
    ap.add_argument("--transition-teacher-steps", type=int, default=64,
                    help="fine GLASS steps used to build Stage-3 transition targets")
    ap.add_argument("--rollout-steps", type=int, default=4,
                    help="student coarse steps for Stage-4 rollout distillation")
    ap.add_argument("--rollout-final-weight", type=float, default=2.0,
                    help="extra weight for the final coarse node in Stage-4 rollout loss")
    ap.add_argument("--rollout-velocity-weight", type=float, default=0.0,
                    help="Stage-4 weight for per-step GLASS coarse-transition velocity targets")
    ap.add_argument("--rollout-teacher-forced-velocity-weight", type=float, default=0.0,
                    help="Stage-4 weight for velocity targets queried on GLASS coarse nodes")
    ap.add_argument("--rollout-spread-weight", type=float, default=0.0,
                    help="Stage-4 weight for coordinate posterior-spread matching")
    ap.add_argument("--rollout-spread-samples", type=int, default=1,
                    help="posterior samples per condition for Stage-4 spread matching")
    ap.add_argument("--rollout-diagonal", action="store_true",
                    help="Stage-4 trains diagonal Euler rollout v(s,s,.) instead of off-diagonal jumps")
    ap.add_argument("--rollout-condition-source",
                    choices=["endpoint", "generated", "guided"],
                    default="endpoint",
                    help=("Stage-4 conditioning source: endpoint uses noised real QM9 "
                          "endpoints; generated uses unguided production states at t_cond; "
                          "guided uses GLASS-guided production states at t_cond"))
    ap.add_argument("--generated-condition-steps", type=int, default=64,
                    help="production steps used when --rollout-condition-source is generated/guided")
    ap.add_argument("--guided-condition-property-name", default="alpha",
                    help="TFG property used to guide generated conditioning states")
    ap.add_argument("--guided-condition-mu", type=float, default=0.45)
    ap.add_argument("--guided-condition-reward-scale", type=float, default=0.3)
    ap.add_argument("--guided-condition-value-samples", type=int, default=16)
    ap.add_argument("--guided-condition-value-glass-steps", type=int, default=4)
    ap.add_argument("--guided-condition-guide-every", type=int, default=1)
    ap.add_argument("--guided-condition-guide-min-t", type=float, default=0.05)
    ap.add_argument("--guided-condition-guide-max-t", type=float, default=0.85)
    ap.add_argument("--guided-condition-guide-atomics", action="store_true")
    ap.add_argument("--guided-condition-max-coord-rms", type=float, default=0.0)
    ap.add_argument("--guided-condition-max-atom-rms", type=float, default=0.0)
    ap.add_argument("--guided-condition-soft-atom-temperature", type=float, default=0.25)
    ap.add_argument("--guided-condition-eps", type=float, default=1e-6)
    ap.add_argument("--target-seed", type=int, default=2026)
    ap.add_argument("--hist-bins", type=int, default=100)
    ap.add_argument("--hist-cache", type=Path, default=None,
                    help="target histogram for --rollout-condition-source guided "
                         "(default: cache/histograms/hist_<prop>_train_flow_100.json)")
    ap.add_argument("--force-rebuild-hist", action="store_true")
    ap.add_argument("--eval-diagonal", action="store_true",
                    help="evaluate checkpoints with diagonal Euler posterior sampling")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--mfm-steps", type=int, default=4)
    ap.add_argument("--glass-steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(paths.OUTPUTS / "mfm_student"))
    ap.add_argument("--teacher-checkpoint", type=Path, default=CKPT,
                    help="frozen base flow model used as the GLASS teacher")
    ap.add_argument("--max-updates", type=int, default=0,
                    help="stop after this many updates without changing the LR schedule "
                         "(which still uses --steps as its horizon); 0 = run all --steps")
    ap.add_argument("--hidden-dim", type=int, default=None, help="override student width")
    ap.add_argument("--num-layers", type=int, default=None, help="override student depth")
    ap.add_argument("--num-heads", type=int, default=None, help="override student heads")
    ap.add_argument("--block-conditioning", choices=["additive", "adaln", "gated", "gated_delta", "split_gated"], default="additive",
                    help="additive=time added to tokens; gated=additive plus zero-init per-block gates; adaln=per-block adaLN-Zero modulation")
    ap.add_argument("--encoder-depth", type=int, default=None,
                    help="encoder depth for split_gated conditioning; defaults to about 2/3 of layers")
    ap.add_argument("--velocity-parametrization",
                    choices=[
                        "endpoint",
                        "direct",
                        "endpoint_residual",
                        "coord_direct_atom_endpoint",
                        "coord_endpoint_residual_atom_endpoint",
                    ],
                    default="endpoint",
                    help="student output parametrization for diagonal velocity")
    ap.add_argument("--train-residual-only", action="store_true",
                    help="freeze non-residual parameters; requires a residual parametrization")
    ap.add_argument("--train-coord-output-only", action="store_true",
                    help="freeze all parameters except out_coord_linear")
    ap.add_argument("--train-conditioning-adapter-only", action="store_true",
                    help="freeze all parameters except zero-init gated/delta conditioning adapters")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir) / time.strftime(f"stage{args.stage}_%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir={out_dir}")

    teacher, data_stats, margs = load_flow_model(Path(args.teacher_checkpoint), device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = build_student(teacher, margs, hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers, num_heads=args.num_heads,
                            block_conditioning=args.block_conditioning,
                            velocity_parametrization=args.velocity_parametrization,
                            encoder_depth=args.encoder_depth).to(device)
    same_size = (args.hidden_dim in (None, margs.hidden_dim)
                 and args.num_layers in (None, margs.num_layers))
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        init_state = init["student"]
        if args.skip_init_coord_head:
            init_state = {
                k: v
                for k, v in init_state.items()
                if not k.startswith("out_coord_linear.")
            }
            print("skipping coordinate output head from init checkpoint")
        incompatible = student.load_state_dict(init_state, strict=False)
        print(f"initialized student from {args.init_ckpt} (step {init.get('step')})")
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "  non-strict load: "
                f"missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
    elif args.init_from_teacher:
        if same_size:
            student.init_from_base(
                teacher.net,
                copy_output_heads=(args.velocity_parametrization != "direct"),
            )
        else:
            print("WARNING: student size != teacher; skipping init_from_base (training from scratch)")
    if args.train_residual_only:
        if args.velocity_parametrization not in (
            "endpoint_residual",
            "coord_endpoint_residual_atom_endpoint",
        ):
            raise ValueError(
                "--train-residual-only requires a residual velocity parametrization"
            )
        for name, param in student.named_parameters():
            param.requires_grad_(name.startswith("residual_"))
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"training residual heads only: trainable_params={trainable:,}")
    if args.train_coord_output_only:
        for name, param in student.named_parameters():
            param.requires_grad_(name.startswith("out_coord_linear."))
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"training coordinate output head only: trainable_params={trainable:,}")
    if args.train_conditioning_adapter_only:
        adapter_prefixes = ("block_cond_gates.", "cond_delta_coord_embed.", "cond_delta_atom_embed.")
        for name, param in student.named_parameters():
            param.requires_grad_(name.startswith(adapter_prefixes))
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"training conditioning adapters only: trainable_params={trainable:,}")
    ema = {k: v.detach().clone() for k, v in student.state_dict().items()}
    print(f"teacher net={sum(p.numel() for p in teacher.net.parameters()):,} "
          f"student={sum(p.numel() for p in student.parameters()):,}")

    data_args = SimpleNamespace(
        partition=args.partition, limit_data=args.limit_data,
        max_len=args.max_len, include_hydrogens=False,
    )
    tensors, _ = load_qm9_tensors(data_args)
    print(f"loaded {len(tensors)} QM9 molecules")
    collate = TensorDictCollator()
    eval_x1 = collate(tensors[: args.n_eval])
    batches = infinite_batches(tensors[args.n_eval :], args.batch_size)

    guided_regressor = None
    guided_hist = None
    guided_target_rng = None
    if args.rollout_condition_source == "guided":
        if not is_tfg_property(args.guided_condition_property_name):
            raise ValueError(
                "--guided-condition-property-name must be a TFG property, got "
                f"{args.guided_condition_property_name!r}"
            )
        guided_regressor = load_tfg_regressor(
            "guide",
            args.guided_condition_property_name,
            device,
        )
        guided_regressor.soft_atom_temperature = args.guided_condition_soft_atom_temperature
        guided_regressor.eval()
        for param in guided_regressor.parameters():
            param.requires_grad_(False)
        hist_args = argparse.Namespace(**vars(args))
        hist_args.property_name = args.guided_condition_property_name
        guided_hist = build_or_load_histograms(
            hist_args,
            max_len=int(data_stats["max_num_atoms"]),
        )
        guided_target_rng = np.random.default_rng(args.target_seed)

    posebusters = build_posebusters(paths.posebusters_config_path())
    opt = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=args.lr)
    t_cond_grid = None
    if args.t_cond_grid.strip():
        values = [float(x.strip()) for x in args.t_cond_grid.split(",") if x.strip()]
        if not values:
            raise ValueError("--t-cond-grid was set but no numeric values were parsed")
        t_cond_grid = torch.tensor(values, dtype=torch.float32, device=device)
        if torch.any(t_cond_grid < 0) or torch.any(t_cond_grid > args.t_max):
            raise ValueError(
                f"--t-cond-grid values must lie in [0, t_max={args.t_max}]"
            )
    s_grid = None
    if args.s_grid.strip():
        values = [float(x.strip()) for x in args.s_grid.split(",") if x.strip()]
        if not values:
            raise ValueError("--s-grid was set but no numeric values were parsed")
        s_grid = torch.tensor(values, dtype=torch.float32, device=device)
        if torch.any(s_grid < 0) or torch.any(s_grid > args.t_max):
            raise ValueError(f"--s-grid values must lie in [0, t_max={args.t_max}]")

    def sample_inner_s(shape):
        if s_grid is not None:
            idx = torch.randint(s_grid.numel(), shape, device=device)
            s_val = s_grid[idx]
            if args.s_grid_jitter > 0:
                jitter = (2.0 * torch.rand_like(s_val) - 1.0) * args.s_grid_jitter
                s_val = (s_val + jitter).clamp(0.0, args.t_max)
            return s_val
        return torch.rand(shape, device=device).pow(args.s_power) * args.t_max
    transition_steps = tuple(
        int(x.strip()) for x in args.transition_steps.split(",") if x.strip()
    )
    if args.stage in (3, 4):
        if not transition_steps:
            raise ValueError("--transition-steps must contain at least one integer")
        for n in transition_steps:
            if args.transition_teacher_steps % n != 0:
                raise ValueError(
                    "--transition-teacher-steps must be divisible by every "
                    f"transition grid; got teacher_steps={args.transition_teacher_steps}, "
                    f"transition_steps={transition_steps}"
                )
    if args.stage == 4 and args.transition_teacher_steps % args.rollout_steps != 0:
        raise ValueError(
            "--transition-teacher-steps must be divisible by --rollout-steps; "
            f"got teacher_steps={args.transition_teacher_steps}, "
            f"rollout_steps={args.rollout_steps}"
        )

    bins = np.linspace(0, 1, 6)
    running = {"loss": [], "coord": [], "atom": []}
    bin_l2 = {i: [] for i in range(len(bins) - 1)}

    for step in range(1, args.steps + 1):
        cur_lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for group in opt.param_groups:
            group["lr"] = cur_lr

        batch = next(batches).to(device)
        B = batch["padding_mask"].shape[0]
        scalar_t_cond_batch = (
            args.rollout_condition_source in ("generated", "guided")
            and args.stage in (1, 4)
        )
        if t_cond_grid is not None:
            shape = (1,) if scalar_t_cond_batch else (B,)
            idx = torch.randint(t_cond_grid.numel(), shape, device=device)
            t_cond = t_cond_grid[idx]
            if args.t_cond_jitter > 0:
                jitter = (2.0 * torch.rand_like(t_cond) - 1.0) * args.t_cond_jitter
                t_cond = (t_cond + jitter).clamp(0.0, args.t_max)
        else:
            shape = (1,) if scalar_t_cond_batch else (B,)
            t_cond_base = torch.rand(shape, device=device).pow(args.t_cond_power)
            t_cond = t_cond_base * args.t_max
        if args.t_cond_zero_rate > 0:
            keep_noisy = torch.rand_like(t_cond) >= args.t_cond_zero_rate
            t_cond = t_cond * keep_noisy.to(t_cond.dtype)
        if scalar_t_cond_batch:
            t_cond = t_cond.expand(B)

        opt.zero_grad()
        if args.stage == 1:
            if args.rollout_condition_source in ("generated", "guided"):
                s = sample_inner_s((1,)).expand(B)
                if args.rollout_condition_source == "generated":
                    cond_state = sample_production_condition_state(
                        teacher,
                        batch_size=B,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                    )
                else:
                    if guided_hist is None or guided_target_rng is None:
                        raise RuntimeError("guided condition source was not initialized")
                    init_state = teacher._sample_noise_like_batch(batch_size=B).to(device)
                    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
                    target_values = [
                        sample_property_target(guided_hist, int(length), guided_target_rng)
                        for length in lengths
                    ]
                    property_targets = torch.tensor(
                        target_values,
                        dtype=torch.float32,
                        device=device,
                    )
                    cond_state = sample_guided_production_condition_state(
                        teacher,
                        init_state,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                        mu=args.guided_condition_mu,
                        reward_scale=args.guided_condition_reward_scale,
                        value_samples=args.guided_condition_value_samples,
                        value_glass_steps=args.guided_condition_value_glass_steps,
                        property_name=args.guided_condition_property_name,
                        property_target=property_targets,
                        property_regressor=guided_regressor,
                        guide_atomics=args.guided_condition_guide_atomics,
                        guide_min_t=args.guided_condition_guide_min_t,
                        guide_max_t=args.guided_condition_guide_max_t,
                        guide_every=args.guided_condition_guide_every,
                        guidance_max_coord_rms=args.guided_condition_max_coord_rms,
                        guidance_max_atom_rms=args.guided_condition_max_atom_rms,
                        eps=args.guided_condition_eps,
                    )
                loss, comp = glass_posterior_diagonal_distill_loss(
                    student,
                    teacher,
                    cond_state,
                    t_cond,
                    s,
                    posterior_steps=args.transition_teacher_steps,
                    endpoint_space_loss=args.endpoint_space_loss,
                    loss_type=args.diag_loss_type,
                    adaptive_p=args.diag_adaptive_p,
                    adaptive_c=args.diag_adaptive_c,
                    coord_weight=args.coord_loss_weight,
                    atom_weight=args.atom_loss_weight,
                    direction_weight=args.diag_direction_weight,
                    condition_jvp_weight=args.diag_condition_jvp_weight,
                    condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                    condition_jvp_eps=args.diag_condition_jvp_eps,
                    condition_jvp_mode=args.diag_condition_jvp_mode,
                )
            else:
                s = sample_inner_s((B,))
                loss, comp = diagonal_distill_loss(
                    student, teacher, batch, s, t_cond,
                    endpoint_space_loss=args.endpoint_space_loss,
                    loss_type=args.diag_loss_type,
                    adaptive_p=args.diag_adaptive_p,
                    adaptive_c=args.diag_adaptive_c,
                    coord_weight=args.coord_loss_weight,
                    atom_weight=args.atom_loss_weight,
                    direction_weight=args.diag_direction_weight,
                    condition_jvp_weight=args.diag_condition_jvp_weight,
                    condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                    condition_jvp_eps=args.diag_condition_jvp_eps,
                    condition_jvp_mode=args.diag_condition_jvp_mode,
                )
        elif args.stage == 2:
            # Stage 2: off-diagonal ESD consistency + diagonal anchor.
            two = torch.rand(B, 2, device=device) * args.t_max
            s_lo, u_hi = two.min(dim=1).values, two.max(dim=1).values
            # anneal the jump: shrink toward the diagonal early in training.
            prog = min(1.0, step / max(1, int(args.anneal_frac * args.steps)))
            mid = 0.5 * (s_lo + u_hi)
            jump = (u_hi - s_lo).clamp(max=args.max_jump)  # cap the jump size
            half = 0.5 * jump * prog
            s, u = mid - half, mid + half
            esd_loss, comp = esd_consistency_loss(
                student, teacher, batch, s, u, t_cond,
                target_clip=args.esd_target_clip,
                delta_clip=args.esd_delta_clip,
                loss_type=args.esd_loss_type,
                adaptive_p=args.esd_adaptive_p,
                adaptive_c=args.esd_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
            )
            s_diag = sample_inner_s((B,))
            diag_loss, dcomp = diagonal_distill_loss(
                student, teacher, batch, s_diag, t_cond,
                endpoint_space_loss=args.endpoint_space_loss,
                loss_type=args.diag_loss_type,
                adaptive_p=args.diag_adaptive_p,
                adaptive_c=args.diag_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                direction_weight=args.diag_direction_weight,
                condition_jvp_weight=args.diag_condition_jvp_weight,
                condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                condition_jvp_eps=args.diag_condition_jvp_eps,
                condition_jvp_mode=args.diag_condition_jvp_mode,
            )
            loss = args.esd_weight * esd_loss + args.diag_weight * diag_loss
            comp = {"coord_l2": dcomp["coord_l2"], "atom_l2": dcomp["atom_l2"],
                    **comp}
        elif args.stage == 3:
            # Stage 3: direct GLASS transition distillation on the exact coarse
            # grids used by one-/two-/four-step posterior samplers.
            n_flow_steps = int(transition_steps[torch.randint(
                len(transition_steps), (), device=device
            ).item()])
            interval_idx = int(torch.randint(n_flow_steps, (), device=device).item())
            trans_loss, comp = glass_transition_distill_loss(
                student,
                teacher,
                batch,
                t_cond,
                n_flow_steps=n_flow_steps,
                interval_idx=interval_idx,
                teacher_steps=args.transition_teacher_steps,
                loss_type=args.esd_loss_type,
                adaptive_p=args.esd_adaptive_p,
                adaptive_c=args.esd_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
            )
            s_diag = sample_inner_s((B,))
            diag_loss, dcomp = diagonal_distill_loss(
                student, teacher, batch, s_diag, t_cond,
                endpoint_space_loss=args.endpoint_space_loss,
                loss_type=args.diag_loss_type,
                adaptive_p=args.diag_adaptive_p,
                adaptive_c=args.diag_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                direction_weight=args.diag_direction_weight,
                condition_jvp_weight=args.diag_condition_jvp_weight,
                condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                condition_jvp_eps=args.diag_condition_jvp_eps,
                condition_jvp_mode=args.diag_condition_jvp_mode,
            )
            loss = args.esd_weight * trans_loss + args.diag_weight * diag_loss
            comp = {"coord_l2": dcomp["coord_l2"], "atom_l2": dcomp["atom_l2"],
                    **comp}
        else:
            # Stage 4: train the realized few-step rollout against GLASS coarse
            # nodes, so gradients see compounding sampler error.
            cond_state = None
            if args.rollout_condition_source in ("generated", "guided"):
                if args.rollout_condition_source == "generated":
                    cond_state = sample_production_condition_state(
                        teacher,
                        batch_size=B,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                    )
                else:
                    if guided_hist is None or guided_target_rng is None:
                        raise RuntimeError("guided condition source was not initialized")
                    init_state = teacher._sample_noise_like_batch(batch_size=B).to(device)
                    lengths = (~init_state["padding_mask"]).sum(dim=1).detach().cpu()
                    target_values = [
                        sample_property_target(guided_hist, int(length), guided_target_rng)
                        for length in lengths
                    ]
                    property_targets = torch.tensor(
                        target_values,
                        dtype=torch.float32,
                        device=device,
                    )
                    cond_state = sample_guided_production_condition_state(
                        teacher,
                        init_state,
                        t_cond=t_cond,
                        num_steps=args.generated_condition_steps,
                        mu=args.guided_condition_mu,
                        reward_scale=args.guided_condition_reward_scale,
                        value_samples=args.guided_condition_value_samples,
                        value_glass_steps=args.guided_condition_value_glass_steps,
                        property_name=args.guided_condition_property_name,
                        property_target=property_targets,
                        property_regressor=guided_regressor,
                        guide_atomics=args.guided_condition_guide_atomics,
                        guide_min_t=args.guided_condition_guide_min_t,
                        guide_max_t=args.guided_condition_guide_max_t,
                        guide_every=args.guided_condition_guide_every,
                        guidance_max_coord_rms=args.guided_condition_max_coord_rms,
                        guidance_max_atom_rms=args.guided_condition_max_atom_rms,
                        eps=args.guided_condition_eps,
                    )
            rollout_loss, comp = glass_rollout_distill_loss(
                student,
                teacher,
                batch,
                t_cond,
                n_flow_steps=args.rollout_steps,
                teacher_steps=args.transition_teacher_steps,
                loss_type=args.esd_loss_type,
                adaptive_p=args.esd_adaptive_p,
                adaptive_c=args.esd_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                final_weight=args.rollout_final_weight,
                velocity_weight=args.rollout_velocity_weight,
                teacher_forced_velocity_weight=args.rollout_teacher_forced_velocity_weight,
                spread_weight=args.rollout_spread_weight,
                spread_samples=args.rollout_spread_samples,
                diagonal=args.rollout_diagonal,
                cond_state=cond_state,
            )
            s_diag = sample_inner_s((B,))
            diag_loss, dcomp = diagonal_distill_loss(
                student, teacher, batch, s_diag, t_cond,
                endpoint_space_loss=args.endpoint_space_loss,
                loss_type=args.diag_loss_type,
                adaptive_p=args.diag_adaptive_p,
                adaptive_c=args.diag_adaptive_c,
                coord_weight=args.coord_loss_weight,
                atom_weight=args.atom_loss_weight,
                direction_weight=args.diag_direction_weight,
                condition_jvp_weight=args.diag_condition_jvp_weight,
                condition_jvp_direction_weight=args.diag_condition_jvp_direction_weight,
                condition_jvp_eps=args.diag_condition_jvp_eps,
                condition_jvp_mode=args.diag_condition_jvp_mode,
            )
            loss = args.esd_weight * rollout_loss + args.diag_weight * diag_loss
            comp = {"coord_l2": dcomp["coord_l2"], "atom_l2": dcomp["atom_l2"],
                    **comp}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in student.state_dict().items():
                if ema[k].dtype.is_floating_point:
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)

        running["loss"].append(float(loss))
        running["coord"].append(comp["coord_l2"])
        running["atom"].append(comp["atom_l2"])
        bidx = min(int(t_cond.mean().item() * (len(bins) - 1)), len(bins) - 2)
        bin_l2[bidx].append(float(loss))

        if step % args.log_every == 0:
            n = args.log_every
            extra = ""
            if args.stage == 1:
                extra = (f"  coordCos {comp.get('coord_cosine', 0):.3f}  "
                         f"atomCos {comp.get('atom_cosine', 0):.3f}  "
                         f"dir {comp.get('direction_loss', 0):.4f}  "
                         f"cjvpC {comp.get('condition_jvp_coord_l2', 0):.4f}  "
                         f"cjvpCos {comp.get('condition_jvp_coord_cosine', 0):.3f}")
            elif args.stage == 2:
                extra = (f"  esd_coord {comp.get('esd_coord_l2', 0):.4f}  "
                         f"esd_atom {comp.get('esd_atom_l2', 0):.4f}  "
                         f"delta_c {comp.get('delta_coord_norm_clipped', 0):.2f}  "
                         f"jvp_c {comp.get('jvp_coord_norm', 0):.2f}  "
                         f"tgt_c {comp.get('target_coord_norm', 0):.2f}")
            elif args.stage == 3:
                extra = (f"  trans_coord {comp.get('transition_coord_l2', 0):.4f}  "
                         f"trans_atom {comp.get('transition_atom_l2', 0):.4f}  "
                         f"grid {int(comp.get('transition_n_flow_steps', 0))}  "
                         f"idx {int(comp.get('transition_interval_idx', 0))}  "
                         f"tgt_c {comp.get('transition_target_coord_norm', 0):.2f}")
            elif args.stage == 4:
                extra = (f"  roll_coord {comp.get('rollout_coord_l2', 0):.4f}  "
                         f"roll_atom {comp.get('rollout_atom_l2', 0):.4f}  "
                         f"final_c {comp.get('rollout_final_coord_l2', 0):.4f}  "
                         f"final_a {comp.get('rollout_final_atom_l2', 0):.4f}  "
                         f"vel_c {comp.get('rollout_velocity_coord_l2', 0):.4f}  "
                         f"tfv_c {comp.get('rollout_teacher_forced_velocity_coord_l2', 0):.4f}  "
                         f"spr {comp.get('rollout_coord_spread_l2', 0):.4f}  "
                         f"sprR {comp.get('rollout_final_coord_spread_ratio', 0):.2f}  "
                         f"diagR {int(comp.get('rollout_diagonal', 0))}  "
                         f"steps {int(comp.get('rollout_steps', 0))}")
            print(f"step {step:5d}  loss {np.mean(running['loss'][-n:]):.4f}  "
                  f"diag_coord {np.mean(running['coord'][-n:]):.4f}  "
                  f"diag_atom {np.mean(running['atom'][-n:]):.4f}{extra}")

        if step % args.eval_every == 0 or step == args.steps:
            # eval with EMA weights
            backup = {k: v.detach().clone() for k, v in student.state_dict().items()}
            student.load_state_dict(ema)
            student.eval()
            rows = evaluate(student, teacher, eval_x1, data_stats, posebusters,
                            device, [0.3, 0.6, 0.9], args.mfm_steps, args.glass_steps,
                            diagonal=args.eval_diagonal)
            student.load_state_dict(backup)
            student.train()
            eval_mode = "diag" if args.eval_diagonal else "jump"
            print(
                f"  [eval @ {step}] (mfm {eval_mode} {args.mfm_steps}-step "
                f"vs glass {args.glass_steps}-step)"
            )
            for r in rows:
                print(f"    t_cond {r['t_cond']:.2f}  "
                      f"MFM pb_valid {r['mfm_pb_valid']:.2f} conv {r['mfm_conv']:.2f}  | "
                      f"GLASS pb_valid {r['glass_pb_valid']:.2f} conv {r['glass_conv']:.2f}")
            torch.save(
                {"student": ema, "margs": vars(margs), "data_stats": data_stats,
                 "step": step, "args": vars(args),
                 "student_cfg": {
                     "hidden_dim": student.hidden_dim,
                     "num_layers": student.num_layers,
                     "num_heads": args.num_heads or margs.num_heads,
                     "block_conditioning": args.block_conditioning,
                     "velocity_parametrization": args.velocity_parametrization,
                     "encoder_depth": student.encoder_depth,
                 }},
                out_dir / f"student_step_{step}.pt",
            )
        if args.max_updates and step >= args.max_updates:
            print(f"stopping after --max-updates={args.max_updates} (schedule horizon {args.steps})")
            break

    print(f"\nDone. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
