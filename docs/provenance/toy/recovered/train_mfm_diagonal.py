"""Train the diagonal of the endpoint-predicting Meta Flow Map.

Objective:
    mfm.v(s, s, I_s, t_cond, x_cond) ~= GLASS posterior velocity

where:
    x_1      ~ data
    I_s      = (1 - s) eps + s x_1
    x_cond   = (1 - t_cond) eps_cond + t_cond x_1

The model remains endpoint-predicting. Its velocity is derived from
concat(softmax(z_logits), y_pred), as implemented in `mfm_model.py`.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import NUM_CLASSES, Y_DIM, sample_flag, sample_uniform_color_square
from glass import glass_posterior_velocity
from mfm_model import MultimodalMFM
from model import MultimodalDiT
from value_fn import freeze_model, make_joint_teacher_v


DATA_SAMPLERS = {
    "french_flag": sample_flag,
    "uniform_color_square": sample_uniform_color_square,
}


def cosine_lr(
    step: int,
    warmup: int,
    total: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.data.copy_(b.data)


def load_teacher(
    ckpt_path: str,
    device: str,
    use_ema: bool = True,
) -> tuple[MultimodalDiT, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = MultimodalDiT(**cfg).to(device)
    state_key = "ema" if (use_ema and "ema" in ckpt) else "model"
    model.load_state_dict(ckpt[state_key])
    return freeze_model(model), ckpt, cfg


def sample_time(
    batch_size: int,
    t_min: float,
    t_max: float,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    r = torch.rand(batch_size, generator=generator, device=device)
    return t_min + (t_max - t_min) * r


def make_diagonal_batch(
    sample_data,
    batch_size: int,
    generator: torch.Generator,
    device: str,
    s_min: float,
    s_max: float,
    t_cond_min: float,
    t_cond_max: float,
    t_cond_zero_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    c1, y1 = sample_data(batch_size, generator=generator, device=device)
    c1_onehot = F.one_hot(c1, NUM_CLASSES).to(y1.dtype)
    x1 = torch.cat([c1_onehot, y1], dim=-1)

    x0 = torch.randn(
        x1.shape, generator=generator, device=device, dtype=x1.dtype
    )
    s = sample_time(batch_size, s_min, s_max, generator, device)
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
    return s, I_s, t_cond, x_cond


def train(
    teacher_ckpt: str,
    ckpt_path: str,
    dataset: str = "auto",
    steps: int = 20000,
    batch_size: int = 2048,
    lr: float = 3e-4,
    warmup: int = 500,
    ema_decay: float = 0.999,
    log_every: int = 100,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_teacher_ema: bool = True,
    init_from_teacher: bool = True,
    s_min: float = 1e-3,
    s_max: float = 0.95,
    t_cond_min: float = 1e-3,
    t_cond_max: float = 0.95,
    t_cond_zero_prob: float = 0.0,
    c_weight: float = 1.0,
    y_weight: float = 1.0,
    grad_clip: float = 1.0,
    target_clip: float = 0.0,
):
    torch.manual_seed(seed)
    g_data = torch.Generator(device=device).manual_seed(seed + 1)

    teacher, teacher_bundle, model_cfg = load_teacher(
        teacher_ckpt, device=device, use_ema=use_teacher_ema
    )
    if dataset == "auto":
        dataset = teacher_bundle.get("dataset", "french_flag")
    if dataset not in DATA_SAMPLERS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATA_SAMPLERS)}"
        )
    sample_data = DATA_SAMPLERS[dataset]

    model = MultimodalMFM(**model_cfg).to(device)
    if init_from_teacher:
        model.init_from_base(teacher)
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
        f"teacher={teacher_ckpt}  init_from_teacher={init_from_teacher}  "
        f"s=[{s_min}, {s_max}]  t_cond=[{t_cond_min}, {t_cond_max}]  "
        f"t_cond_zero_prob={t_cond_zero_prob}"
    )

    log = {
        "step": [],
        "loss": [],
        "loss_c": [],
        "loss_y": [],
        "pred_norm": [],
        "target_norm": [],
        "lr": [],
    }
    t0 = time.time()
    for step in range(1, steps + 1):
        cur_lr = cosine_lr(step, warmup, steps, lr)
        for group in opt.param_groups:
            group["lr"] = cur_lr

        s, I_s, t_cond, x_cond = make_diagonal_batch(
            sample_data=sample_data,
            batch_size=batch_size,
            generator=g_data,
            device=device,
            s_min=s_min,
            s_max=s_max,
            t_cond_min=t_cond_min,
            t_cond_max=t_cond_max,
            t_cond_zero_prob=t_cond_zero_prob,
        )

        with torch.no_grad():
            target = glass_posterior_velocity(
                s, I_s, x_cond, t_cond, teacher_v, eps=1e-6
            )
            if target_clip > 0.0:
                target = target.clamp(-target_clip, target_clip)

        pred = model.v(s, s, I_s, t_cond=t_cond, x_cond=x_cond)
        diff = pred - target
        loss_c = diff[:, :NUM_CLASSES].pow(2).mean()
        loss_y = diff[:, NUM_CLASSES:].pow(2).mean()
        loss = c_weight * loss_c + y_weight * loss_y

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ema_update(ema_model, model, ema_decay)

        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                pred_norm = pred.norm(dim=-1).mean()
                target_norm = target.norm(dim=-1).mean()
            print(
                f"step {step:6d}  loss {loss.item():.5f}  "
                f"L_c {loss_c.item():.5f}  L_y {loss_y.item():.5f}  "
                f"|pred| {pred_norm.item():.3f}  |target| {target_norm.item():.3f}  "
                f"lr {cur_lr:.2e}  ({time.time() - t0:.1f}s)"
            )
            log["step"].append(step)
            log["loss"].append(loss.item())
            log["loss_c"].append(loss_c.item())
            log["loss_y"].append(loss_y.item())
            log["pred_norm"].append(pred_norm.item())
            log["target_norm"].append(target_norm.item())
            log["lr"].append(cur_lr)

    ckpt = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "config": model_cfg,
        "dataset": dataset,
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_dataset": teacher_bundle.get("dataset"),
        "objective": "diagonal_glass_velocity",
        "training_config": {
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "warmup": warmup,
            "ema_decay": ema_decay,
            "seed": seed,
            "use_teacher_ema": use_teacher_ema,
            "init_from_teacher": init_from_teacher,
            "s_min": s_min,
            "s_max": s_max,
            "t_cond_min": t_cond_min,
            "t_cond_max": t_cond_max,
            "t_cond_zero_prob": t_cond_zero_prob,
            "c_weight": c_weight,
            "y_weight": y_weight,
            "grad_clip": grad_clip,
            "target_clip": target_clip,
        },
        "log": log,
    }
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"saved {ckpt_path}")
    return model, ema_model, log


if __name__ == "__main__":
