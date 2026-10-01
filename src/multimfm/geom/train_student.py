"""Multi-GPU MFM student distillation on GEOM-Drugs (paper Table 20, GEOM column).

Distills a ``TabascoFlowMap`` student from the frozen GEOM-Drugs base model's
GLASS posterior field. Stage 1 matches the teacher on the diagonal (s=u) over
the full ``t_cond`` range; Stage 2 adds the endpoint self-distillation (ESD)
consistency loss over a capped ``u-s`` jump window; Stages 3/4 add transition
and rollout objectives. Periodically evaluates the few-step consistency
posterior sampler against the many-step GLASS posterior (PoseBusters validity)
and checkpoints the student.

Launch with torchrun for DDP (the paper's student used 4 GPUs, batch 96 each)::

    torchrun --standalone --nproc_per_node=4 -m multimfm.geom.train_student \
        --stage 2 --batch-size 96 --steps 30000 --lr 1e-4 --warmup 500 \
        --max-jump 0.15 --anneal-frac 0.2 --esd-delta-clip 8 --esd-target-clip 10 \
        --diag-weight 2 --ema 0.995 --limit-data 150000

``scripts/geom/train_student.sbatch`` wraps the exact stage-by-stage recipe that
produced ``checkpoints/geom/mfm_student/student_step_10000.pt``; the lineage and
every stage's arguments are in ``docs/provenance/geom/checkpoints.md``.

Ported from ``geom-drug-tabasco/metaflow/mfm_student_train.py``: the algorithm is
unchanged, the flat script imports became package imports and the defaults are
repo-relative.
"""

from __future__ import annotations

import argparse
import builtins
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.utils.data import DataLoader

from multimfm.geom.paths import (
    DEFAULT_GEOM_TRAIN_PT,
    DEFAULT_TEACHER as DEFAULT_TEACHER_PATH,
    REPO_ROOT,
)
from multimfm.geom.teacher import load_geom_teacher
from multimfm.paths import default_hist_cache
from multimfm.glass_guidance import (
    build_posebusters,
    clip_guidance_rms,
    is_tfg_property,
    pb_summary_for_state,
    production_step_with_optional_guidance,
    value_gradient,
)
from multimfm.glass_endpoints import (
    make_condition_state_from_endpoint,
    sample_glass_posterior,
)
from multimfm.qm9_data import load_tfg_regressor  # property-guided conditioning only
from multimfm.train_base_flow import load_processed_mols, mols_to_dataset
from tabasco.data.utils import TensorDictCollator
from tabasco.flow.mfm_losses import (
    diagonal_distill_loss,
    esd_consistency_loss,
    glass_posterior_diagonal_distill_loss,
    glass_rollout_distill_loss,
    glass_transition_distill_loss,
)
from tabasco.models.components.flow_map_transformer import TabascoFlowMap
from tabasco.sample.flow_map import consistency_posterior_sample
from tabasco.utils.tensor_ops import apply_mask, mask_and_zero_com

# The QM9 property-target histograms are only needed by the optional
# property-guided rollout path (--rollout-condition-source guided); GEOM
# distillation never touches them.
try:
    from multimfm.property_targets import (
        build_or_load_histograms,
        sample_property_target,
    )

    _HAS_QM9_LENGTH_COND = True
except ImportError:  # pragma: no cover - QM9 driver absent
    build_or_load_histograms = None
    sample_property_target = None
    _HAS_QM9_LENGTH_COND = False

#: PoseBusters configuration used by the in-loop evaluation (training-validation
#: config, as in the recorded runs -- not the GEOM Table-3 config).
DEFAULT_EVAL_PB_CONFIG = REPO_ROOT / "src/tabasco/utils/posebusters_no_strain.yaml"

# Default GEOM-Drugs teacher (the "big" 256/16/8 base model, EMA weights).
DEFAULT_TEACHER = str(DEFAULT_TEACHER_PATH)


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


def flat_lr(step, warmup, base_lr):
    """Linear warmup to base_lr over `warmup` steps, then hold constant (no decay).
    Use for fine-tuning a resumed checkpoint at a small LR."""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr


class AdaptiveWindow:
    """Plateau-gated curriculum for the Stage-2 ESD jump cap (u - s window).

    Grows ``cur`` by ``bump`` toward ``final`` ONLY after the tracked loss has
    plateaued -- relative improvement of its EMA over the last ``patience`` steps
    < ``tol`` -- and at least ``min_dwell`` steps have elapsed since the last bump,
    so each window is consolidated before it widens. Feed it a rank-synchronized
    loss so the schedule is identical across DDP ranks.
    """

    def __init__(self, start, final, bump, patience, tol, min_dwell, ema_decay=0.999):
        self.cur = float(start)
        self.final = float(final)
        self.bump = float(bump)
        self.patience = int(patience)
        self.tol = float(tol)
        self.min_dwell = int(min_dwell)
        self.ema_decay = float(ema_decay)
        self.loss_ema = None
        self.ref_ema = None        # loss EMA at the start of the current plateau window
        self.dwell = 0             # steps since the last bump
        self.since_ref = 0         # steps since ref_ema was set
        self.n_bumps = 0

    def update(self, loss):
        loss = float(loss)
        self.loss_ema = loss if self.loss_ema is None else (
            self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * loss
        )
        self.dwell += 1
        self.since_ref += 1
        if self.ref_ema is None:
            self.ref_ema = self.loss_ema
        if self.since_ref >= self.patience:
            improved = (self.ref_ema - self.loss_ema) / max(abs(self.ref_ema), 1e-8)
            if improved < self.tol and self.dwell >= self.min_dwell and self.cur < self.final:
                self.cur = min(self.final, self.cur + self.bump)
                self.dwell = 0
                self.n_bumps += 1
            self.ref_ema = self.loss_ema
            self.since_ref = 0
        return self.cur


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
    ap.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER,
                    help="GEOM-Drugs teacher Lightning checkpoint to distill from.")
    ap.add_argument("--no-ema", action="store_true",
                    help="Distill from the teacher's raw weights instead of EMA (default EMA).")
    ap.add_argument("--data-path", default=str(DEFAULT_GEOM_TRAIN_PT),
                    help="Processed GEOM-Drugs .pt file (list of RDKit mols) for clean endpoints.")
    ap.add_argument("--partition", default="train_flow",
                    help="(Unused for GEOM; kept for CLI compatibility.)")
    ap.add_argument("--limit-data", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=0,
                    help="0 => use the teacher's max_num_atoms (GEOM).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", choices=["cosine", "flat"], default="cosine",
                    help="'cosine' = warmup then cosine decay to 0 over --steps; "
                         "'flat' = warmup then hold --lr constant (for fine-tuning a resumed "
                         "checkpoint at small LR without a decaying/high-noise schedule).")
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
                    help="cap on the inner-time jump (u - s) for the ESD term (start value)")
    ap.add_argument("--max-jump-final", type=float, default=None,
                    help="if set, SLOWLY ramp the jump cap from --max-jump to this over "
                         "--max-jump-warmup-frac of training, so each gap size is consolidated "
                         "before the window widens (curriculum; default = no growth).")
    ap.add_argument("--max-jump-warmup-frac", type=float, default=1.0,
                    help="fraction of steps over which --max-jump ramps to --max-jump-final")
    # --- adaptive (plateau-gated) window growth: grow the jump cap only once the
    #     current window's loss has plateaued (instead of a fixed-rate ramp). ---
    ap.add_argument("--adaptive-window", action="store_true",
                    help="grow the ESD jump cap adaptively (plateau-gated) from --max-jump "
                         "toward --max-jump-final; overrides the fixed --max-jump-warmup-frac ramp")
    ap.add_argument("--window-bump", type=float, default=0.01,
                    help="increment added to the jump cap at each plateau")
    ap.add_argument("--window-patience", type=int, default=2000,
                    help="steps over which to measure loss-plateau for a bump")
    ap.add_argument("--window-tol", type=float, default=0.02,
                    help="relative loss-EMA improvement below which the window is 'plateaued'")
    ap.add_argument("--window-min-dwell", type=int, default=2000,
                    help="minimum steps to train each window before it may grow")
    ap.add_argument("--window-ema-decay", type=float, default=0.999,
                    help="EMA decay for the plateau-tracked loss")
    ap.add_argument("--uniform-window-sampling", action="store_true",
                    help="sample the jump width uniformly in [0, cur_max_jump] (kills the "
                         "point-mass-at-cap of the two-uniform-spread sampling)")
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
    ap.add_argument("--hist-cache", type=Path, default=default_hist_cache("alpha"),
                    help="QM9 property histogram; only read by "
                         "--rollout-condition-source guided, never by GEOM distillation.")
    ap.add_argument("--force-rebuild-hist", action="store_true")
    ap.add_argument("--eval-diagonal", action="store_true",
                    help="evaluate checkpoints with diagonal Euler posterior sampling")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--mfm-steps", type=int, default=4)
    ap.add_argument("--eval-mfm-steps", type=str, default=None,
                    help="comma list of inner-step counts for the in-loop eval, e.g. '2,4'. "
                         "Defaults to str(--mfm-steps). Once max_jump is wide, few-step (2,4) "
                         "is the meaningful metric; 10-step no longer probes the trained window.")
    ap.add_argument("--glass-steps", type=int, default=64)
    ap.add_argument("--eval-pb-config", type=Path, default=DEFAULT_EVAL_PB_CONFIG,
                    help="PoseBusters config for the in-loop evaluation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs/geom/mfm_student"))
    # Weights & Biases (rank-0 only). Defaults match the GEOM teacher project.
    ap.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    ap.add_argument("--wandb-project", default="geom-drugs")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"),
                    help="W&B entity; defaults to $WANDB_ENTITY (the paper runs used "
                         "the authors' own entity).")
    ap.add_argument("--wandb-name", default=None)
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

    # ---- distributed (torchrun --standalone --nproc_per_node=N) ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    is_main = rank == 0
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    if not is_main:  # silence non-main ranks' stdout
        builtins.print = lambda *a, **k: None
    print(f"[rank {rank}/{world_size}] device={device} distributed={distributed}")

    torch.manual_seed(args.seed)  # identical model init across ranks
    out_dir = Path(args.out_dir) / time.strftime(f"stage{args.stage}_%Y%m%d-%H%M%S")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"out_dir={out_dir}")

    use_wandb = (not args.no_wandb) and is_main
    if use_wandb:
        import wandb  # noqa: PLC0415
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name or f"mfm-geom-stage{args.stage}-ddp{world_size}",
            config={**vars(args), "world_size": world_size,
                    "global_batch_size": args.batch_size * world_size},
        )

    teacher, data_stats, margs = load_geom_teacher(
        args.teacher_checkpoint, device, use_ema=not args.no_ema
    )
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

    # GEOM-Drugs clean endpoints: load processed RDKit mols and convert to padded
    # {coords, atomics, padding_mask} TensorDicts (heavy-atom, /2.0 normalizer),
    # matching the teacher's data_stats (atom_dim, coordinate_normalizer).
    geom_mols = load_processed_mols(Path(args.data_path), args.limit_data)
    geom_ds = mols_to_dataset(geom_mols, repeat=1, include_hydrogens=False)
    tensors = geom_ds.tensors
    if int(geom_ds.stats["atom_dim"]) != int(data_stats["atom_dim"]):
        raise ValueError(
            f"data atom_dim {geom_ds.stats['atom_dim']} != teacher atom_dim "
            f"{data_stats['atom_dim']} (vocab mismatch)."
        )
    if int(geom_ds.stats["max_num_atoms"]) > int(data_stats["max_num_atoms"]):
        raise ValueError(
            f"data max_num_atoms {geom_ds.stats['max_num_atoms']} exceeds teacher "
            f"max_num_atoms {data_stats['max_num_atoms']}."
        )
    print(f"loaded {len(tensors)} GEOM-Drugs molecules "
          f"(max_atoms={geom_ds.stats['max_num_atoms']}, atom_dim={geom_ds.stats['atom_dim']})")
    collate = TensorDictCollator()
    eval_x1 = collate(tensors[: args.n_eval])  # same held-out set on all ranks (rank 0 evals)
    train_tensors = tensors[args.n_eval :]
    if distributed:
        train_tensors = train_tensors[rank::world_size]  # shard training data across ranks
        print(f"[rank {rank}] train shard = {len(train_tensors)} molecules")
    batches = infinite_batches(train_tensors, args.batch_size)
    # De-correlate per-rank sampling (times/noise) AFTER identical model init.
    if distributed:
        torch.manual_seed(args.seed + 100003 * rank)

    guided_regressor = None
    guided_hist = None
    guided_target_rng = None
    if args.rollout_condition_source == "guided":
        if not _HAS_QM9_LENGTH_COND:
            raise SystemExit(
                "--rollout-condition-source guided needs the QM9-specific "
                "qm9_length_conditioned_guidance_experiment module, which is not "
                "present in this repo. Basic GEOM distillation does not require it."
            )
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

    posebusters = build_posebusters(Path(args.eval_pb_config))
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

    window_ctl = None
    if args.stage == 2 and args.adaptive_window:
        window_final = args.max_jump_final if args.max_jump_final is not None else 1.0
        window_ctl = AdaptiveWindow(
            start=args.max_jump, final=window_final, bump=args.window_bump,
            patience=args.window_patience, tol=args.window_tol,
            min_dwell=args.window_min_dwell, ema_decay=args.window_ema_decay,
        )
        print(f"adaptive window: start={args.max_jump} final={window_final} bump={args.window_bump} "
              f"patience={args.window_patience} tol={args.window_tol} min_dwell={args.window_min_dwell} "
              f"uniform_sampling={args.uniform_window_sampling}")

    # inner-step counts for the in-loop eval (few-step is the meaningful metric once
    # the window is wide); defaults to the single --mfm-steps for backward compat.
    eval_step_list = [int(x) for x in (args.eval_mfm_steps or str(args.mfm_steps)).split(",")
                      if x.strip()]
    if is_main:
        print(f"in-loop eval step counts: {eval_step_list}")

    bins = np.linspace(0, 1, 6)
    running = {"loss": [], "coord": [], "atom": []}
    bin_l2 = {i: [] for i in range(len(bins) - 1)}

    for step in range(1, args.steps + 1):
        if args.lr_schedule == "flat":
            cur_lr = flat_lr(step, args.warmup, args.lr)
        else:
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
            # --- current jump cap (u - s window) ---
            if window_ctl is not None:
                cur_max_jump = window_ctl.cur                       # plateau-gated growth
            elif args.max_jump_final is not None:
                mj_prog = min(1.0, step / max(1, int(args.max_jump_warmup_frac * args.steps)))
                cur_max_jump = args.max_jump + (args.max_jump_final - args.max_jump) * mj_prog
            else:
                cur_max_jump = args.max_jump
            # --- sample (s, u) with gap <= cur_max_jump ---
            if args.uniform_window_sampling:
                # width ~ U[0, cur_max_jump]; start ~ U[0, t_max - width]. Uniform width,
                # no point-mass at the cap.
                w = torch.rand(B, device=device) * cur_max_jump
                s = torch.rand(B, device=device) * (args.t_max - w).clamp(min=1e-6)
                u = s + w
            else:
                # legacy: clamped spread of two uniforms, annealed by prog.
                prog = min(1.0, step / max(1, int(args.anneal_frac * args.steps)))
                two = torch.rand(B, 2, device=device) * args.t_max
                s_lo, u_hi = two.min(dim=1).values, two.max(dim=1).values
                mid = 0.5 * (s_lo + u_hi)
                jump = (u_hi - s_lo).clamp(max=cur_max_jump)
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
            comp["cur_max_jump"] = float(cur_max_jump)
            # Advance the adaptive window on a rank-synchronized plateau signal so all
            # DDP ranks keep an identical cur_max_jump.
            if window_ctl is not None:
                sig = float(comp.get("esd_coord_l2", float(esd_loss)))
                if distributed:
                    _t = torch.tensor([sig], device=device)
                    dist.all_reduce(_t)
                    sig = (_t / world_size).item()
                window_ctl.update(sig)
                comp["n_window_bumps"] = window_ctl.n_bumps
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
        if distributed:
            # Manual DDP: average gradients across ranks (zero-fill unused params so
            # every rank participates in each all_reduce in the same order).
            for p in student.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size
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
                         f"tgt_c {comp.get('target_coord_norm', 0):.2f}  "
                         f"win {comp.get('cur_max_jump', 0):.3f}  "
                         f"bumps {int(comp.get('n_window_bumps', 0))}")
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
            if use_wandb:
                wlog = {
                    "train/loss": float(np.mean(running["loss"][-n:])),
                    "train/diag_coord_l2": float(np.mean(running["coord"][-n:])),
                    "train/diag_atom_l2": float(np.mean(running["atom"][-n:])),
                    "train/lr": cur_lr,
                }
                if window_ctl is not None:
                    wlog["train/cur_max_jump"] = float(window_ctl.cur)
                    wlog["train/n_window_bumps"] = int(window_ctl.n_bumps)
                wandb.log(wlog, step=step)

        if step % args.eval_every == 0 or step == args.steps:
            if is_main:
                # eval with EMA weights (rank 0 only)
                backup = {k: v.detach().clone() for k, v in student.state_dict().items()}
                student.load_state_dict(ema)
                student.eval()
                eval_mode = "diag" if args.eval_diagonal else "jump"
                # Sweep one or more inner-step counts. Once max_jump is wide, few-step
                # rollouts (2,4) are the meaningful metric; a 10-step eval no longer
                # probes the trained window. GLASS (velocity-field ref) is identical
                # across step counts, so log it once from the first sweep entry.
                log_d = {}
                glass_logged = False
                mfm_means = {}
                for ns in eval_step_list:
                    rows = evaluate(student, teacher, eval_x1, data_stats, posebusters,
                                    device, [0.3, 0.6, 0.9], ns, args.glass_steps,
                                    diagonal=args.eval_diagonal)
                    mean_ns = float(np.mean([r["mfm_pb_valid"] for r in rows]))
                    mfm_means[ns] = mean_ns
                    print(
                        f"  [eval @ {step}] (mfm {eval_mode} {ns}-step "
                        f"vs glass {args.glass_steps}-step)  MEAN={mean_ns:.3f}"
                    )
                    for r in rows:
                        print(f"    t_cond {r['t_cond']:.2f}  "
                              f"MFM pb_valid {r['mfm_pb_valid']:.2f} conv {r['mfm_conv']:.2f}  | "
                              f"GLASS pb_valid {r['glass_pb_valid']:.2f} conv {r['glass_conv']:.2f}")
                    if use_wandb:
                        for r in rows:
                            tc = f"{r['t_cond']:.1f}"
                            log_d[f"eval/mfm_pb_valid_t{tc}_s{ns}"] = r["mfm_pb_valid"]
                            log_d[f"eval/mfm_conv_t{tc}_s{ns}"] = r["mfm_conv"]
                            if not glass_logged:
                                log_d[f"eval/glass_pb_valid_t{tc}"] = r["glass_pb_valid"]
                        log_d[f"eval/mfm_pb_valid_mean_s{ns}"] = mean_ns
                        glass_logged = True
                student.load_state_dict(backup)
                student.train()
                if use_wandb:
                    # headline metric = the smallest (hardest) step count in the sweep
                    log_d["eval/mfm_pb_valid_mean"] = mfm_means[eval_step_list[0]]
                    wandb.log(log_d, step=step)
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
            if distributed:
                dist.barrier()  # keep ranks in lock-step across rank-0 eval/save

    if use_wandb:
        wandb.finish()
    if distributed:
        dist.destroy_process_group()
    print(f"\nDone. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
