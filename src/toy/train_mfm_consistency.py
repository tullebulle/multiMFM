"""Train off-diagonal MFM consistency with an ESD-teacher loss.

PARTLY RECOVERED.  Original: ``norway/train_mfm_consistency.py`` (purged).
Verbatim from a Codex transcript (2026-06-11, sed/nl of the live file):
  * original lines 80-135: tail of ``sample_s_u``, ``make_consistency_batch``,
    head of ``split_loss``;
  * original lines 240-295: the training-loop body (batch, GLASS diagonal
    target, diagonal loss, ESD JVP target, consistency loss, optimiser step).
Recovered by ``rg`` (single lines, with original line numbers): the docstring
line "`esd_teacher` setup. Gaps start near zero and are annealed to random full"
(l.15), ``consistency_weight: float = 1.0`` (l.179),
``stop_grad_esd_teacher: bool = True`` (l.182), the log keys
``consistency_loss{,_c,_y}`` (l.234-236), the progress print
``diag {..:.6f}  esd {..:.6f}`` (l.302), ``"objective":
"esd_teacher_consistency_plus_diagonal_glass"`` (l.328), CLI
``--ckpt checkpoints/mfm_consistency.pt`` (l.366), ``--consistency_weight 1.0``
(l.386), ``--no_stop_grad_esd_teacher`` (l.389).
Everything else is RECONSTRUCTED to fit those fragments, the recovered
``train_mfm_diagonal.py`` (same helpers / checkpoint layout) and the terminal
log (exact command, gap schedule "take midpoint and distance, shrink the
distance by gap_scale, anneal gap_scale from 0 to 1 over --gap_anneal_steps",
gap_scale = step / 5000 at steps 1, 250, 1000, 5000; flash SDP disabled inside
the JVP; log every 250 steps; total = diag + esd, i.e. both weights 1).

Objective (MFM "esd_teacher" setup): for x_1 ~ data, s <= u,
    I_s   = (1 - s) eps + s x_1,   x_cond = (1 - t_cond) eps' + t_cond x_1
    vss   = GLASS posterior velocity at (s, I_s | t_cond, x_cond)
    L_diag = || v(s, s, I_s) - vss ||^2
    L_ESD  = || v(s, u, I_s) - sg[ vss + (u - s) d/ds v(s, u, I_s) |_(ds=1, dI=vss) ] ||^2
with separate colour (c) and position (y) weights.

The paper-run command (terminal log):
    python -u train_mfm_consistency.py --teacher_ckpt ckpt_uniform_color_square.pt \\
        --init_ckpt checkpoints/mfm_diagonal_10000.pt \\
        --ckpt checkpoints/mfm_consistency_10000.pt --steps 10000 --batch_size 1024 \\
        --lr 1e-4 --warmup 500 --gap_anneal_steps 5000 --ema_decay 0.995 --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import NUM_CLASSES
from .glass import glass_posterior_velocity
from .mfm_model import MultimodalMFM
from .paths import BASE_CKPT, CHECKPOINTS, MFM_DIAGONAL_CKPT
from .train_mfm_diagonal import DATA_SAMPLERS, cosine_lr, ema_update, load_teacher, sample_time
from .value_fn import make_joint_teacher_v


def math_sdp_kernel_context():
    """Force the math SDPA backend: forward-mode AD does not support flash attention."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:  # older torch
        if torch.cuda.is_available():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            )
        return contextlib.nullcontext()


def gap_scale_for_step(step: int, warmup_steps: int, anneal_steps: int) -> float:
    """0 during ``warmup_steps``, then linear to 1 over ``anneal_steps``."""
    if step <= warmup_steps:
        return 0.0
    if anneal_steps <= 0:
        return 1.0
    return min(1.0, (step - warmup_steps) / anneal_steps)


def sample_s_u(
    batch_size: int,
    step: int,
    warmup_steps: int,
    anneal_steps: int,
    t_min: float,
    t_max: float,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    a = sample_time(batch_size, t_min, t_max, generator, device)
    b = sample_time(batch_size, t_min, t_max, generator, device)
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    mid = 0.5 * (lo + hi)
    # ---- recovered verbatim from here (original l.80) ----
    dist = hi - lo
    scale = gap_scale_for_step(step, warmup_steps, anneal_steps)
    s = mid - 0.5 * scale * dist
    u = mid + 0.5 * scale * dist
    return s, u, scale


def make_consistency_batch(
    sample_data,
    batch_size: int,
    step: int,
    generator: torch.Generator,
    device: str,
    time_min: float,
    time_max: float,
    gap_warmup_steps: int,
    gap_anneal_steps: int,
    t_cond_min: float,
    t_cond_max: float,
    t_cond_zero_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    c1, y1 = sample_data(batch_size, generator=generator, device=device)
    c1_onehot = F.one_hot(c1, NUM_CLASSES).to(y1.dtype)
    x1 = torch.cat([c1_onehot, y1], dim=-1)

    s, u, gap_scale = sample_s_u(
        batch_size=batch_size,
        step=step,
        warmup_steps=gap_warmup_steps,
        anneal_steps=gap_anneal_steps,
        t_min=time_min,
        t_max=time_max,
        generator=generator,
        device=device,
    )
    x0 = torch.randn(x1.shape, generator=generator, device=device, dtype=x1.dtype)
    I_s = (1.0 - s).unsqueeze(-1) * x0 + s.unsqueeze(-1) * x1

    noise_cond = torch.randn(
        x1.shape, generator=generator, device=device, dtype=x1.dtype
    )
    t_cond = sample_time(batch_size, t_cond_min, t_cond_max, generator, device)
    if t_cond_zero_prob > 0.0:
        zero_mask = torch.rand(batch_size, generator=generator, device=device)
        t_cond = torch.where(
            zero_mask < t_cond_zero_prob,
            torch.zeros_like(t_cond),
            t_cond,
        )
    x_cond = (1.0 - t_cond).unsqueeze(-1) * noise_cond + t_cond.unsqueeze(-1) * x1
    return s, u, I_s, t_cond, x_cond, gap_scale


def split_loss(diff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_c = diff[:, :NUM_CLASSES].pow(2).mean()
    loss_y = diff[:, NUM_CLASSES:].pow(2).mean()
    # ---- recovered span ends at original l.135 ----
    return loss_c + loss_y, loss_c, loss_y


def load_init_mfm(init_ckpt: str, device: str, use_ema: bool = True) -> tuple[MultimodalMFM, dict]:
    ckpt = torch.load(init_ckpt, map_location=device)
    model = MultimodalMFM(**ckpt["config"]).to(device)
    state_key = "ema" if (use_ema and "ema" in ckpt) else "model"
    model.load_state_dict(ckpt[state_key])
    return model, ckpt


def train(
    teacher_ckpt: str,
    init_ckpt: str | None,
    ckpt_path: str,
    dataset: str = "auto",
    steps: int = 10000,
    batch_size: int = 1024,
    lr: float = 1e-4,
    warmup: int = 500,
    ema_decay: float = 0.995,
    log_every: int = 250,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_teacher_ema: bool = True,
    init_use_ema: bool = True,
    time_min: float = 1e-3,
    time_max: float = 0.95,
    gap_warmup_steps: int = 0,
    gap_anneal_steps: int = 5000,
    t_cond_min: float = 1e-3,
    t_cond_max: float = 0.95,
    t_cond_zero_prob: float = 0.0,
    c_weight: float = 1.0,
    y_weight: float = 1.0,
    diagonal_weight: float = 1.0,
    consistency_weight: float = 1.0,
    grad_clip: float = 1.0,
    target_clip: float = 10.0,
    stop_grad_esd_teacher: bool = True,
):
    torch.manual_seed(seed)
    g_data = torch.Generator(device=device).manual_seed(seed + 1)

    teacher, teacher_bundle, model_cfg = load_teacher(
        teacher_ckpt, device=device, use_ema=use_teacher_ema
    )
    if dataset == "auto":
        dataset = teacher_bundle.get("dataset", "french_flag")
    sample_data = DATA_SAMPLERS[dataset]

    if init_ckpt:
        model, init_bundle = load_init_mfm(init_ckpt, device, use_ema=init_use_ema)
        model_cfg = init_bundle["config"]
    else:
        model = MultimodalMFM(**model_cfg).to(device).init_from_base(teacher)
    model.train()
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    teacher_v = make_joint_teacher_v(teacher, NUM_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(
        f"device={device}  params={sum(p.numel() for p in model.parameters()):,}  "
        f"batch={batch_size}  steps={steps}  dataset={dataset}"
    )
    print(
        f"teacher={teacher_ckpt}  init={init_ckpt}  time=[{time_min}, {time_max}]  "
        f"t_cond=[{t_cond_min}, {t_cond_max}]  gap_anneal_steps={gap_anneal_steps}  "
        f"target_clip={target_clip}"
    )

    log = {
        "step": [],
        "loss": [],
        "diag_loss": [],
        "diag_loss_c": [],
        "diag_loss_y": [],
        "consistency_loss": [],
        "consistency_loss_c": [],
        "consistency_loss_y": [],
        "gap_scale": [],
        "mean_gap": [],
        "max_gap": [],
        "lr": [],
    }
    t0 = time.time()
    # ---- recovered verbatim from here (original l.241) ----
    for step in range(1, steps + 1):
        cur_lr = cosine_lr(step, warmup, steps, lr)
        for group in opt.param_groups:
            group["lr"] = cur_lr

        s, u, I_s, t_cond, x_cond, gap_scale = make_consistency_batch(
            sample_data=sample_data,
            batch_size=batch_size,
            step=step,
            generator=g_data,
            device=device,
            time_min=time_min,
            time_max=time_max,
            gap_warmup_steps=gap_warmup_steps,
            gap_anneal_steps=gap_anneal_steps,
            t_cond_min=t_cond_min,
            t_cond_max=t_cond_max,
            t_cond_zero_prob=t_cond_zero_prob,
        )

        with torch.no_grad():
            vss = glass_posterior_velocity(s, I_s, x_cond, t_cond, teacher_v, eps=1e-6)
            if target_clip > 0.0:
                vss = vss.clamp(-target_clip, target_clip)

        diag_pred = model.v(s, s, I_s, t_cond=t_cond, x_cond=x_cond)
        diag_loss, diag_loss_c, diag_loss_y = split_loss(diag_pred - vss)
        diag_loss = c_weight * diag_loss_c + y_weight * diag_loss_y

        def vsu_fn(s_: torch.Tensor, u_: torch.Tensor, x_: torch.Tensor):
            return model.v(s_, u_, x_, t_cond=t_cond, x_cond=x_cond)

        with math_sdp_kernel_context():
            vsu, jvp = torch.func.jvp(
                vsu_fn,
                (s, u, I_s),
                (torch.ones_like(s), torch.zeros_like(u), vss),
            )
        esd_target = vss + (u - s).unsqueeze(-1) * jvp
        if stop_grad_esd_teacher:
            esd_target = esd_target.detach()
        consistency_loss, consistency_loss_c, consistency_loss_y = split_loss(
            vsu - esd_target
        )
        consistency_loss = c_weight * consistency_loss_c + y_weight * consistency_loss_y

        loss = diagonal_weight * diag_loss + consistency_weight * consistency_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        # ---- recovered span ends at original l.295 ----
        ema_update(ema_model, model, ema_decay)

        if step == 1 or step % log_every == 0 or step == steps:
            gap = (u - s).detach()
            print(
                f"step {step:6d}  loss {loss.item():.6f}  "
                f"diag {diag_loss.item():.6f}  esd {consistency_loss.item():.6f}  "
                f"gap_scale {gap_scale:.3f}  mean_gap {gap.mean().item():.4f}  "
                f"max_gap {gap.max().item():.4f}  lr {cur_lr:.2e}  ({time.time() - t0:.1f}s)"
            )
            log["step"].append(step)
            log["loss"].append(loss.item())
            log["diag_loss"].append(diag_loss.item())
            log["diag_loss_c"].append(diag_loss_c.item())
            log["diag_loss_y"].append(diag_loss_y.item())
            log["consistency_loss"].append(consistency_loss.item())
            log["consistency_loss_c"].append(consistency_loss_c.item())
            log["consistency_loss_y"].append(consistency_loss_y.item())
            log["gap_scale"].append(gap_scale)
            log["mean_gap"].append(gap.mean().item())
            log["max_gap"].append(gap.max().item())
            log["lr"].append(cur_lr)

    ckpt = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "config": model_cfg,
        "dataset": dataset,
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_dataset": teacher_bundle.get("dataset"),
        "init_ckpt": None if init_ckpt is None else str(init_ckpt),
        "objective": "esd_teacher_consistency_plus_diagonal_glass",
        "training_config": {
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "warmup": warmup,
            "ema_decay": ema_decay,
            "seed": seed,
            "use_teacher_ema": use_teacher_ema,
            "init_use_ema": init_use_ema,
            "time_min": time_min,
            "time_max": time_max,
            "gap_warmup_steps": gap_warmup_steps,
            "gap_anneal_steps": gap_anneal_steps,
            "t_cond_min": t_cond_min,
            "t_cond_max": t_cond_max,
            "t_cond_zero_prob": t_cond_zero_prob,
            "c_weight": c_weight,
            "y_weight": y_weight,
            "diagonal_weight": diagonal_weight,
            "consistency_weight": consistency_weight,
            "grad_clip": grad_clip,
            "target_clip": target_clip,
            "stop_grad_esd_teacher": stop_grad_esd_teacher,
        },
        "log": log,
    }
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"saved {ckpt_path}")
    return model, ema_model, log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", default=str(BASE_CKPT))
    p.add_argument("--init_ckpt", default=str(MFM_DIAGONAL_CKPT))
    p.add_argument("--ckpt", default=str(CHECKPOINTS / "mfm_consistency.pt"))
    p.add_argument("--dataset", default="auto", choices=["auto", *sorted(DATA_SAMPLERS)])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_teacher_ema", action="store_true")
    p.add_argument("--init_use_raw", action="store_true", help="init from raw (not EMA) diagonal weights")
    p.add_argument("--time_min", type=float, default=1e-3)
    p.add_argument("--time_max", type=float, default=0.95)
    p.add_argument("--gap_warmup_steps", type=int, default=0)
    p.add_argument("--gap_anneal_steps", type=int, default=5000)
    p.add_argument("--t_cond_min", type=float, default=1e-3)
    p.add_argument("--t_cond_max", type=float, default=0.95)
    p.add_argument("--t_cond_zero_prob", type=float, default=0.0)
    p.add_argument("--c_weight", type=float, default=1.0)
    p.add_argument("--y_weight", type=float, default=1.0)
    p.add_argument("--diagonal_weight", type=float, default=1.0)
    p.add_argument("--consistency_weight", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--target_clip", type=float, default=10.0)
    p.add_argument("--no_stop_grad_esd_teacher", action="store_true")
    args = p.parse_args()
    train(
        teacher_ckpt=args.teacher_ckpt,
        init_ckpt=args.init_ckpt or None,
        ckpt_path=args.ckpt,
        dataset=args.dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=args.warmup,
        ema_decay=args.ema_decay,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        use_teacher_ema=not args.no_teacher_ema,
        init_use_ema=not args.init_use_raw,
        time_min=args.time_min,
        time_max=args.time_max,
        gap_warmup_steps=args.gap_warmup_steps,
        gap_anneal_steps=args.gap_anneal_steps,
        t_cond_min=args.t_cond_min,
        t_cond_max=args.t_cond_max,
        t_cond_zero_prob=args.t_cond_zero_prob,
        c_weight=args.c_weight,
        y_weight=args.y_weight,
        diagonal_weight=args.diagonal_weight,
        consistency_weight=args.consistency_weight,
        grad_clip=args.grad_clip,
        target_clip=args.target_clip,
        stop_grad_esd_teacher=not args.no_stop_grad_esd_teacher,
    )
