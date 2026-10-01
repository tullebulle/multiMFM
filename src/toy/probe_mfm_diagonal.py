"""Probe a diagonal MFM checkpoint against the GLASS velocity target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .glass import glass_posterior_velocity
from .mfm_model import MultimodalMFM
from .paths import BASE_CKPT, CHECKPOINTS, FIGURES
from .train_mfm_diagonal import DATA_SAMPLERS, load_teacher, make_diagonal_batch
from .value_fn import make_joint_teacher_v


def resolve_figure_path(out: str | Path) -> Path:
    path = Path(out)
    if path.parent == Path("."):
        path = FIGURES / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_mfm(ckpt_path: str, device: str, use_ema: bool) -> tuple[MultimodalMFM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultimodalMFM(**ckpt["config"]).to(device)
    state_key = "ema" if use_ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, ckpt


def train_style_loss(pred: torch.Tensor, target: torch.Tensor, k: int) -> dict:
    diff = pred - target
    loss_c = diff[:, :k].pow(2).mean()
    loss_y = diff[:, k:].pow(2).mean()
    mse = diff.pow(2).mean()
    vec_rmse = diff.pow(2).sum(dim=-1).mean().sqrt()
    rel_l2 = (
        diff.pow(2).sum(dim=-1).mean().sqrt()
        / target.pow(2).sum(dim=-1).mean().sqrt().clamp_min(1e-12)
    )
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
    return {
        "loss": (loss_c + loss_y).item(),
        "loss_c": loss_c.item(),
        "loss_y": loss_y.item(),
        "mse_per_coord": mse.item(),
        "vector_rmse": vec_rmse.item(),
        "relative_l2": rel_l2.item(),
        "cosine": cosine.item(),
        "pred_norm": pred.norm(dim=-1).mean().item(),
        "target_norm": target.norm(dim=-1).mean().item(),
    }


def bin_metric(
    values: torch.Tensor,
    err_sq: torch.Tensor,
    edges: torch.Tensor,
) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "lo": lo.item(),
                "hi": hi.item(),
                "n": int(mask.sum().item()),
                "rmse": err_sq[mask].sum(dim=-1).mean().sqrt().item(),
            }
        )
    return rows


def plot_loss(log: dict, out: str | Path) -> Path:
    out = resolve_figure_path(out)
    steps = log["step"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if "loss_c" in log:
        ax.plot(steps, log["loss"], label="total")
        ax.plot(steps, log["loss_c"], label=r"$c$")
        ax.plot(steps, log["loss_y"], label=r"$y$")
    else:
        ax.plot(steps, log["loss"], label="total")
        ax.plot(steps, log["diag_loss"], label="diagonal")
        ax.plot(steps, log["consistency_loss"], label="ESD")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("diagonal GLASS distillation loss")
    ax.set_title("MFM diagonal training")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_pred_target_scatter(
    pred: torch.Tensor,
    target: torch.Tensor,
    out: str | Path,
    max_points: int = 4096,
) -> Path:
    out = resolve_figure_path(out)
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    if pred.shape[0] > max_points:
        pred = pred[:max_points]
        target = target[:max_points]

    names = [r"$v_c^B$", r"$v_c^W$", r"$v_c^R$", r"$v_y^1$", r"$v_y^2$"]
    fig, axes = plt.subplots(1, pred.shape[-1], figsize=(3.0 * pred.shape[-1], 3.0))
    if pred.shape[-1] == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.scatter(target[:, j], pred[:, j], s=4, alpha=0.35, rasterized=True)
        lo = min(target[:, j].min().item(), pred[:, j].min().item())
        hi = max(target[:, j].max().item(), pred[:, j].max().item())
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], c="black", lw=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(names[j])
        ax.set_xlabel("GLASS")
        if j == 0:
            ax.set_ylabel("MFM")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


@torch.no_grad()
def probe(args):
    teacher, teacher_bundle, _ = load_teacher(
        args.teacher_ckpt, device=args.device, use_ema=not args.no_teacher_ema
    )
    raw_model, ckpt = load_mfm(args.ckpt, device=args.device, use_ema=False)
    ema_model, _ = load_mfm(args.ckpt, device=args.device, use_ema=True)

    dataset = args.dataset
    if dataset == "auto":
        dataset = ckpt.get("dataset") or teacher_bundle.get("dataset", "french_flag")
    sample_data = DATA_SAMPLERS[dataset]
    k = raw_model.num_classes
    teacher_v = make_joint_teacher_v(teacher, k)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    preds_raw = []
    preds_ema = []
    targets = []
    baselines = []
    ss = []
    t_conds = []
    remaining = args.n_eval
    while remaining > 0:
        b = min(args.batch_size, remaining)
        s, I_s, t_cond, x_cond = make_diagonal_batch(
            sample_data=sample_data,
            batch_size=b,
            generator=generator,
            device=args.device,
            s_min=args.s_min,
            s_max=args.s_max,
            t_cond_min=args.t_cond_min,
            t_cond_max=args.t_cond_max,
            t_cond_zero_prob=args.t_cond_zero_prob,
        )
        target = glass_posterior_velocity(s, I_s, x_cond, t_cond, teacher_v, eps=1e-6)
        pred_raw = raw_model.v(s, s, I_s, t_cond=t_cond, x_cond=x_cond)
        pred_ema = ema_model.v(s, s, I_s, t_cond=t_cond, x_cond=x_cond)
        baseline = teacher_v(s, I_s)

        preds_raw.append(pred_raw.cpu())
        preds_ema.append(pred_ema.cpu())
        targets.append(target.cpu())
        baselines.append(baseline.cpu())
        ss.append(s.cpu())
        t_conds.append(t_cond.cpu())
        remaining -= b

    pred_raw = torch.cat(preds_raw)
    pred_ema = torch.cat(preds_ema)
    target = torch.cat(targets)
    baseline = torch.cat(baselines)
    s_all = torch.cat(ss)
    t_cond_all = torch.cat(t_conds)

    metrics = {
        "checkpoint": args.ckpt,
        "teacher_ckpt": args.teacher_ckpt,
        "dataset": dataset,
        "n_eval": args.n_eval,
        "raw": train_style_loss(pred_raw, target, k),
        "ema": train_style_loss(pred_ema, target, k),
        "teacher_unconditioned_baseline": train_style_loss(baseline, target, k),
    }
    raw_err_sq = (pred_raw - target).pow(2)
    base_err_sq = (baseline - target).pow(2)
    edges = torch.tensor(args.bin_edges)
    metrics["raw_rmse_by_s"] = bin_metric(s_all, raw_err_sq, edges)
    metrics["baseline_rmse_by_s"] = bin_metric(s_all, base_err_sq, edges)
    metrics["raw_rmse_by_t_cond"] = bin_metric(t_cond_all, raw_err_sq, edges)
    metrics["baseline_rmse_by_t_cond"] = bin_metric(t_cond_all, base_err_sq, edges)

    loss_plot = plot_loss(ckpt["log"], args.loss_plot)
    scatter_plot = plot_pred_target_scatter(pred_raw, target, args.scatter_plot)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"saved {loss_plot}")
    print(f"saved {scatter_plot}")
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CHECKPOINTS / "mfm_diagonal_1000.pt"))
    p.add_argument("--teacher_ckpt", default=str(BASE_CKPT))
    p.add_argument("--dataset", default="auto", choices=["auto", *sorted(DATA_SAMPLERS)])
    p.add_argument("--n_eval", type=int, default=16384)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_teacher_ema", action="store_true")
    p.add_argument("--s_min", type=float, default=1e-3)
    p.add_argument("--s_max", type=float, default=0.95)
    p.add_argument("--t_cond_min", type=float, default=1e-3)
    p.add_argument("--t_cond_max", type=float, default=0.95)
    p.add_argument("--t_cond_zero_prob", type=float, default=0.0)
    p.add_argument("--bin_edges", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.95])
    p.add_argument("--loss_plot", default="mfm_diagonal_1000_loss.png")
    p.add_argument("--scatter_plot", default="mfm_diagonal_1000_glass_scatter.png")
    p.add_argument("--metrics_out", default=str(FIGURES / "mfm_diagonal_1000_probe.json"))
    args = p.parse_args()
    probe(args)
