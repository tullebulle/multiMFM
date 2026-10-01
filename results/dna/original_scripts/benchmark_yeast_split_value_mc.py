#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from probe_dna_mfm_glass import load_data_seqs, load_student, load_teacher
from utils.flow_utils import gaussian_beta, gaussian_denoiser_flow_step
from utils.yeast_splits import load_yeast_split_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce MC value-estimator benchmark for clean-split yeast dMFM.")
    p.add_argument(
        "--teacher_ckpt",
        default="workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt",
    )
    p.add_argument("--teacher_args", default=None)
    p.add_argument(
        "--student_ckpt",
        default="workdir/yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35/epoch=622-step=632000.ckpt",
    )
    p.add_argument("--data_pt", default="data/yeast_mid50.pt")
    p.add_argument("--split_pt", default="data/yeast_mid50_split_seed0_train65000_val8000.pt")
    p.add_argument("--data_split", default="test", choices=["train", "val", "test", "full"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fig_dir", default="figures/split_clean")
    p.add_argument("--n_conditions", type=int, default=8)
    p.add_argument("--n_list", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--sde_ref_n", type=int, default=512)
    p.add_argument("--sde_steps", type=int, default=1024)
    p.add_argument("--dmfm_nfe", type=int, default=1)
    p.add_argument("--fmap_nfe", type=int, default=256)
    p.add_argument("--reward_gain", type=float, default=4.0)
    p.add_argument("--tcond_low", type=float, default=0.05)
    p.add_argument("--tcond_high", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


class SmoothDNAReward:
    def __init__(self, length: int, alphabet_size: int, device: torch.device, *, seed: int, gain: float):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.W = torch.randn((length, alphabet_size), device=device, generator=gen) / math.sqrt(length)
        self.P = torch.randn((length - 1, alphabet_size, alphabet_size), device=device, generator=gen) / math.sqrt(length - 1)
        self.gain = float(gain)
        self.motif_pos = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], device=device)
        self.motif_base = torch.tensor([0, 3, 0, 3, 0, 0, 2, 1], device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lin = (x * self.W).sum(dim=(1, 2))
        pair = (x[:, :-1, :, None] * x[:, 1:, None, :] * self.P[None, :, :, :]).sum(dim=(1, 2, 3))
        motif = x[:, self.motif_pos, self.motif_base].mean(dim=1)
        gc = (x[:, :, 1] + x[:, :, 2]).mean(dim=1)
        entropy = -(x.clamp_min(1e-12) * x.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=1)
        r = (
            torch.tanh(lin)
            + 0.7 * torch.tanh(pair)
            + 1.5 * torch.sigmoid(8.0 * (motif - 0.35))
            - 2.0 * (gc - 0.52).pow(2)
            + 0.15 * entropy
        )
        return self.gain * r


def value_stats(r: torch.Tensor) -> dict[str, float]:
    V = torch.logsumexp(r, dim=0) - math.log(r.numel())
    w = torch.softmax(r, dim=0)
    ess = 1.0 / w.pow(2).sum()
    return {
        "V": float(V.detach().cpu()),
        "r_mean": float(r.mean().detach().cpu()),
        "r_std": float(r.std().detach().cpu()),
        "r_min": float(r.min().detach().cpu()),
        "r_max": float(r.max().detach().cpu()),
        "ess": float(ess.detach().cpu()),
        "ess_frac": float((ess / r.numel()).detach().cpu()),
    }


@torch.no_grad()
def make_condition(args, seqs, K, L, device, cond_idx: int, *, t_low: float, t_high: float) -> dict:
    seq_idx = torch.randint(len(seqs), (1,)).item()
    seq = seqs[seq_idx : seq_idx + 1].to(device)
    x1 = F.one_hot(seq, num_classes=K).float()
    t_val = t_low + (t_high - t_low) * torch.rand((), device=device).item()
    t_cond_1 = torch.full((1,), t_val, device=device)
    beta_t = gaussian_beta(args, t_cond_1)
    noise = torch.randn((1, L, K), device=device)
    x_t_1 = beta_t[:, None, None] * x1 + (1.0 - beta_t[:, None, None]) * noise
    return {"cond_idx": cond_idx, "seq_idx": seq_idx, "t_cond_value": t_val, "x_t_1": x_t_1, "t_cond_1": t_cond_1}


@torch.no_grad()
def sample_teacher_sde_N(teacher, condition, n: int, *, sde_steps: int, seed: int, device: torch.device):
    x_t = condition["x_t_1"].expand(n, -1, -1).contiguous()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return teacher.sample_sde(
        x_t.clone(),
        t_start=float(condition["t_cond_value"]),
        t_end=1.0,
        n_steps=sde_steps,
        generator=gen,
    ).detach()


@torch.no_grad()
def sample_dmfm_N(student, condition, n: int, *, K: int, L: int, nfe: int, seed: int, device: torch.device):
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((n, L, K), device=device, generator=gen)
    x_t = condition["x_t_1"].expand(n, -1, -1).contiguous()
    t_cond = condition["t_cond_1"].expand(n).contiguous()
    for k in range(int(nfe)):
        r0 = torch.full((n,), k / int(nfe), device=device, dtype=x.dtype)
        r1 = torch.full((n,), (k + 1) / int(nfe), device=device, dtype=x.dtype)
        x = student(r0, r1, x, t_cond, x_t).detach()
    return x


@torch.no_grad()
def dps_value_estimate(student, condition, reward_fn):
    x_t_1 = condition["x_t_1"]
    t_1 = condition["t_cond_1"]
    D = student.Psi_st(x_t_1, t_1, t_1, t_cond=t_1, x_cond=x_t_1)
    return value_stats(reward_fn(D))


@torch.no_grad()
def fmap_endpoint_teacher(teacher, args, condition, *, nfe: int, device: torch.device):
    x = condition["x_t_1"].clone()
    grid = torch.linspace(float(condition["t_cond_value"]), 1.0, int(nfe) + 1, device=device, dtype=x.dtype)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        b = x.shape[0]
        x, _, _ = gaussian_denoiser_flow_step(args, teacher, x, s0.expand(b), s1.expand(b))
    return x.detach()


def plot_mc_value(plot_df: pd.DataFrame, *, dmfm_nfe: int, fmap_nfe: int, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["DPS denoiser", f"FMAP ODE NFE={fmap_nfe}", f"dMFM NFE={dmfm_nfe}"]
    style = {
        "DPS denoiser": dict(marker="^", linewidth=3.0),
        f"FMAP ODE NFE={fmap_nfe}": dict(marker="D", linewidth=3.0),
        f"dMFM NFE={dmfm_nfe}": dict(marker="o", linewidth=4.0, color="C3"),
    }
    names = {
        "DPS denoiser": "DPS denoiser",
        f"FMAP ODE NFE={fmap_nfe}": "FMAP",
        f"dMFM NFE={dmfm_nfe}": f"dMFM NFE={dmfm_nfe}",
    }
    for label in order:
        sub = plot_df[plot_df["label"] == label].sort_values("n_samples")
        if sub.empty:
            continue
        ax.plot(sub["n_samples"], sub["abs_V_err_mean"], label=names[label], **style[label])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$", fontsize=22)
    ax.set_ylabel(r"$|\hat V_t(x) - V_t(x)|$", fontsize=22)
    ax.set_title(r"Value estimate accuracy vs posterior sample count", fontsize=18)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    torch.manual_seed(cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_args = cli.teacher_args or str(Path(cli.teacher_ckpt).parent / "args.json")
    out_dir = Path(cli.out_dir or f"workdir/yeast_split_value_mc_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    fig_dir = Path(cli.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    args, teacher, _, _ = load_teacher(cli.teacher_ckpt, teacher_args, alphabet_size=4, device=device)
    student, _ = load_student(args, alphabet_size=4, device=device, student_ckpt=cli.student_ckpt)
    teacher.eval()
    student.eval()

    seqs = load_data_seqs(cli.data_pt).long()
    if cli.data_split != "full":
        idx = load_yeast_split_indices(cli.split_pt, cli.data_split)
        seqs = seqs[idx]

    K, L = 4, int(getattr(args, "seq_len", 50))
    reward_fn = SmoothDNAReward(L, K, device, seed=1234, gain=cli.reward_gain)
    rows = []

    print(f"device={device}")
    print(f"teacher={cli.teacher_ckpt}")
    print(f"student={cli.student_ckpt}")
    print(f"out_dir={out_dir}")

    for cond_idx in range(cli.n_conditions):
        condition = make_condition(
            args,
            seqs,
            K,
            L,
            device,
            cond_idx,
            t_low=cli.tcond_low,
            t_high=cli.tcond_high,
        )
        print(f"condition {cond_idx:03d}: seq={condition['seq_idx']} t={condition['t_cond_value']:.3f}", flush=True)

        x_ref = sample_teacher_sde_N(
            teacher,
            condition,
            cli.sde_ref_n,
            sde_steps=cli.sde_steps,
            seed=cli.seed + 10000 * cond_idx + 1,
            device=device,
        )
        ref = value_stats(reward_fn(x_ref))
        dps = dps_value_estimate(student, condition, reward_fn)
        fmap = value_stats(reward_fn(fmap_endpoint_teacher(teacher, args, condition, nfe=cli.fmap_nfe, device=device)))

        for n in cli.n_list:
            x_sde = sample_teacher_sde_N(
                teacher,
                condition,
                n,
                sde_steps=cli.sde_steps,
                seed=cli.seed + 10000 * cond_idx + 100 + n,
                device=device,
            )
            sde_est = value_stats(reward_fn(x_sde))
            rows.append(
                {
                    "cond_idx": cond_idx,
                    "seq_idx": condition["seq_idx"],
                    "t_cond": condition["t_cond_value"],
                    "method": "sde_mc",
                    "n_samples": n,
                    "V_ref": ref["V"],
                    "V_est": sde_est["V"],
                    "V_err": sde_est["V"] - ref["V"],
                    "abs_V_err": abs(sde_est["V"] - ref["V"]),
                    **{f"ref_{k}": v for k, v in ref.items()},
                    **{f"est_{k}": v for k, v in sde_est.items()},
                }
            )

            x_dmfm = sample_dmfm_N(
                student,
                condition,
                n,
                K=K,
                L=L,
                nfe=cli.dmfm_nfe,
                seed=cli.seed + 10000 * cond_idx + 200 + n,
                device=device,
            )
            dmfm_est = value_stats(reward_fn(x_dmfm))
            for method, est in [("dmfm", dmfm_est), ("dps", dps), ("fmap", fmap)]:
                rows.append(
                    {
                        "cond_idx": cond_idx,
                        "seq_idx": condition["seq_idx"],
                        "t_cond": condition["t_cond_value"],
                        "method": method,
                        "n_samples": n,
                        "V_ref": ref["V"],
                        "V_est": est["V"],
                        "V_err": est["V"] - ref["V"],
                        "abs_V_err": abs(est["V"] - ref["V"]),
                        **{f"ref_{k}": v for k, v in ref.items()},
                        **{f"est_{k}": v for k, v in est.items()},
                    }
                )
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["method", "n_samples"], as_index=False)
        .agg(
            V_err_mean=("V_err", "mean"),
            V_err_std=("V_err", "std"),
            abs_V_err_mean=("abs_V_err", "mean"),
            abs_V_err_std=("abs_V_err", "std"),
            ref_r_std_mean=("ref_r_std", "mean"),
            est_r_std_mean=("est_r_std", "mean"),
            ref_ess_frac_mean=("ref_ess_frac", "mean"),
            est_ess_frac_mean=("est_ess_frac", "mean"),
        )
    )
    label_map = {
        "dmfm": f"dMFM NFE={cli.dmfm_nfe}",
        "dps": "DPS denoiser",
        "fmap": f"FMAP ODE NFE={cli.fmap_nfe}",
        "sde_mc": "SDE MC floor",
    }
    plot_df = summary.assign(label=summary["method"].map(label_map))

    df.to_csv(out_dir / "value_mc_rows.csv", index=False)
    summary.to_csv(out_dir / "value_mc_summary.csv", index=False)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(cli), f, indent=2, sort_keys=True)
    plot_mc_value(plot_df, dmfm_nfe=cli.dmfm_nfe, fmap_nfe=cli.fmap_nfe, out_path=fig_dir / "MC_value.pdf")
    plot_mc_value(plot_df, dmfm_nfe=cli.dmfm_nfe, fmap_nfe=cli.fmap_nfe, out_path=out_dir / "MC_value.pdf")
    print(summary.sort_values(["n_samples", "abs_V_err_mean"]).to_string(index=False))
    print(f"wrote {fig_dir / 'MC_value.pdf'}")


if __name__ == "__main__":
    main()
