"""Train the base multimodal flow model (endpoint-predicting DiT).

RECONSTRUCTED (original ``norway/train.py`` was purged).  Evidence used:

* objective, interpolant and architecture: paper App. E.2.1 and Table 12
  (Gaussian source for both modalities, linear interpolant, endpoint losses
  L = MSE(y_hat_1, y1) + CE(z_hat, a1), t ~ U[0, 1]; D=128, 6 blocks, 8 heads,
  MLP ratio 4; AdamW, peak lr 3e-4, cosine decay with 500 warmup steps, EMA
  0.999, batch 4096, 20,000 steps).  D=128/6/8 gives 1,848,837 parameters,
  matching the "~1.85M params" MultimodalDiT quoted in the norway
  ``testing.ipynb`` (Cursor history, 2026-05-07).
* checkpoint layout: ``train_mfm_diagonal.load_teacher`` reads
  ``ckpt["config"]`` (MultimodalDiT kwargs), ``ckpt["ema"]``/``ckpt["model"]``
  and ``ckpt.get("dataset")``; the testing notebook plots ``log["step"],
  log["loss"], log["loss_y"], log["loss_c"]``.
* LR schedule / EMA update / grad clipping: copied from the recovered
  ``train_mfm_diagonal.py`` (same author, same repo).
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import DATA_SAMPLERS, NUM_CLASSES, Y_DIM
from .model import MultimodalDiT
from .paths import BASE_CKPT
from .train_mfm_diagonal import cosine_lr, ema_update


def train(
    ckpt_path: str = str(BASE_CKPT),
    dataset: str = "uniform_color_square",
    steps: int = 20000,
    batch_size: int = 4096,
    lr: float = 3e-4,
    warmup: int = 500,
    ema_decay: float = 0.999,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 6,
    mlp_ratio: int = 4,
    log_every: int = 100,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    grad_clip: float = 1.0,
):
    torch.manual_seed(seed)
    g_data = torch.Generator(device=device).manual_seed(seed + 1)
    sample_data = DATA_SAMPLERS[dataset]

    config = dict(
        num_classes=NUM_CLASSES, y_dim=Y_DIM, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, mlp_ratio=mlp_ratio,
    )
    model = MultimodalDiT(**config).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(
        f"device={device}  params={sum(p.numel() for p in model.parameters()):,}  "
        f"batch={batch_size}  steps={steps}  dataset={dataset}"
    )
    log = {"step": [], "loss": [], "loss_y": [], "loss_c": [], "lr": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        cur_lr = cosine_lr(step, warmup, steps, lr)
        for group in opt.param_groups:
            group["lr"] = cur_lr

        c1, y1 = sample_data(batch_size, generator=g_data, device=device)
        a1 = F.one_hot(c1, NUM_CLASSES).to(y1.dtype)
        y0 = torch.randn(y1.shape, generator=g_data, device=device)
        a0 = torch.randn(a1.shape, generator=g_data, device=device)
        t = torch.rand(batch_size, generator=g_data, device=device)
        tb = t.unsqueeze(-1)
        y_t = (1.0 - tb) * y0 + tb * y1
        a_t = (1.0 - tb) * a0 + tb * a1

        y_pred, z_logits = model(t, a_t, y_t)
        loss_y = F.mse_loss(y_pred, y1)
        loss_c = F.cross_entropy(z_logits, c1)
        loss = loss_y + loss_c

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ema_update(ema_model, model, ema_decay)

        if step == 1 or step % log_every == 0 or step == steps:
            print(
                f"step {step:6d}  loss {loss.item():.5f}  L_y {loss_y.item():.5f}  "
                f"L_c {loss_c.item():.5f}  lr {cur_lr:.2e}  ({time.time() - t0:.1f}s)"
            )
            log["step"].append(step)
            log["loss"].append(loss.item())
            log["loss_y"].append(loss_y.item())
            log["loss_c"].append(loss_c.item())
            log["lr"].append(cur_lr)

    ckpt = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "config": config,
        "dataset": dataset,
        "objective": "endpoint_mse_plus_ce",
        "training_config": {
            "steps": steps, "batch_size": batch_size, "lr": lr, "warmup": warmup,
            "ema_decay": ema_decay, "seed": seed, "grad_clip": grad_clip,
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
    p.add_argument("--ckpt", default=str(BASE_CKPT))
    p.add_argument("--dataset", default="uniform_color_square", choices=sorted(DATA_SAMPLERS))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--mlp_ratio", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    train(
        ckpt_path=a.ckpt, dataset=a.dataset, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
        warmup=a.warmup, ema_decay=a.ema_decay, d_model=a.d_model, n_heads=a.n_heads,
        n_layers=a.n_layers, mlp_ratio=a.mlp_ratio, log_every=a.log_every, seed=a.seed,
        device=a.device, grad_clip=a.grad_clip,
    )
