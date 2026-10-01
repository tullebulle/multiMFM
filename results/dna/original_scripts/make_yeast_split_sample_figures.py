#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dinko.torch_model import C0Torch
from model.dna_models import DiTSequenceModel
from sample_yeast_c0_guidance_hist import cyclizability_reward
from utils.flow_utils import gaussian_denoiser_flow_step
from utils.yeast_splits import load_yeast_split_indices


BASES = np.array(["A", "C", "G", "T"])
DISPLAY_ORDER = np.array([3, 2, 1, 0])  # T, G, C, A, matching the legacy figure.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate yeast sample-quality PDFs for the clean-split DFM teacher.")
    p.add_argument(
        "--ckpt",
        default="workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt",
    )
    p.add_argument("--data_pt", default="data/yeast_mid50.pt")
    p.add_argument("--split_pt", default="data/yeast_mid50_split_seed0_train65000_val8000.pt")
    p.add_argument("--real_split", default="test", choices=["train", "val", "test", "full"])
    p.add_argument("--c0_ckpt", default="dinko/C0free_torch.pt")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fig_dir", default="figures/split_clean")
    p.add_argument("--n_samples", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--nfe", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_flash_attn", action="store_true")
    return p.parse_args()


def load_tokens(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    seqs = payload["seqs"] if isinstance(payload, dict) and "seqs" in payload else payload
    if not isinstance(seqs, torch.Tensor):
        raise TypeError(f"Expected tensor or dict with 'seqs' in {path}, got {type(seqs)}")
    seqs = seqs.long()
    if seqs.ndim != 2:
        raise ValueError(f"Expected token tensor [N,L], got {tuple(seqs.shape)}")
    return seqs


def load_teacher(ckpt_path: str | Path, device: torch.device, use_flash_attn: bool) -> tuple[DiTSequenceModel, SimpleNamespace]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_payload = ckpt["model_cfg"]
    cfg = cfg_payload if isinstance(cfg_payload, SimpleNamespace) else SimpleNamespace(**cfg_payload)
    cfg.use_flash_attn = bool(use_flash_attn or getattr(cfg, "use_flash_attn", False))
    cfg.flow_temp = float(getattr(cfg, "flow_temp", 1.0))

    model = DiTSequenceModel(cfg, alphabet_size=cfg.alphabet_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def generate_tokens(
    model: DiTSequenceModel,
    cfg: SimpleNamespace,
    *,
    n_samples: int,
    batch_size: int,
    nfe: int,
    device: torch.device,
) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    grid = torch.linspace(0, 1, nfe + 1, device=device)
    for start in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - start)
        x = torch.randn(b, cfg.seq_len, cfg.alphabet_size, device=device)
        for s, t in zip(grid[:-1], grid[1:]):
            x, _, _ = gaussian_denoiser_flow_step(cfg, model, x, s.expand(b), t.expand(b))
        outs.append(x.argmax(-1).cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def score_c0(tokens: torch.Tensor, c0_model: C0Torch, *, batch_size: int, device: torch.device) -> np.ndarray:
    scores: list[torch.Tensor] = []
    for start in range(0, len(tokens), batch_size):
        tok = tokens[start : start + batch_size].to(device)
        onehot = F.one_hot(tok, num_classes=4).float()
        scores.append(cyclizability_reward(c0_model, onehot).detach().cpu())
    return torch.cat(scores).reshape(-1).numpy()


def base_freq(tokens: torch.Tensor) -> np.ndarray:
    onehot = F.one_hot(tokens.long(), num_classes=4).float()
    return onehot.mean(0).numpy()


def global_freq(tokens: torch.Tensor) -> np.ndarray:
    return F.one_hot(tokens.long(), num_classes=4).float().mean((0, 1)).numpy()


def seq_strings(tokens: torch.Tensor) -> list[str]:
    arr = tokens.cpu().numpy()
    return ["".join(BASES[row].tolist()) for row in arr]


def plot_base_heatmaps(real_freq, gen_freq, uniform_freq, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 5.2), sharex=True)
    vmax = max(float(real_freq.max()), float(gen_freq.max()), float(uniform_freq.max()))
    for ax, title, mat in zip(
        axes,
        ["generated", "uniform", "real"],
        [gen_freq, uniform_freq, real_freq],
    ):
        im = ax.imshow(mat[:, DISPLAY_ORDER].T, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("base")
        ax.set_yticks(range(4), ["T", "G", "C", "A"])
    axes[-1].set_xlabel("position")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.012)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_diff_heatmaps(real_freq, gen_freq, uniform_freq, out_path: Path) -> None:
    diffs = [gen_freq - real_freq, uniform_freq - real_freq]
    vmax = max(float(np.abs(d).max()) for d in diffs)
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 4.1), sharex=True)
    for ax, title, mat in zip(axes, ["generated - real", "uniform - real"], diffs):
        im = ax.imshow(
            mat[:, DISPLAY_ORDER].T,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_ylabel("base")
        ax.set_yticks(range(4), ["T", "G", "C", "A"])
    axes[-1].set_xlabel("position")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.012)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_marginals(real, gen, uniform, out_path: Path) -> None:
    x = np.arange(4)
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.bar(x - w, real, w, label="real")
    ax.bar(x, gen, w, label="generated")
    ax.bar(x + w, uniform, w, label="uniform")
    ax.set_title("Global Base Marginals")
    ax.set_ylabel("frequency")
    ax.set_xticks(x, BASES)
    ax.set_ylim(0, max(0.32, float(max(real.max(), gen.max(), uniform.max())) * 1.12))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cyclizability(real, gen, uniform, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins = np.linspace(
        min(float(real.min()), float(gen.min()), float(uniform.min())),
        max(float(real.max()), float(gen.max()), float(uniform.max())),
        60,
    )
    for label, values in [("real", real), ("generated", gen), ("uniform", uniform)]:
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2.0, label=label)
    ax.set_title("Cyclizability Score Distributions")
    ax.set_xlabel("cyclizability")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir or f"workdir/yeast_split_sample_figures_{stamp}")
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"out_dir={out_dir}")
    print(f"fig_dir={fig_dir}")

    data = load_tokens(args.data_pt)
    if args.real_split == "full":
        real_pool = data
    else:
        idx = load_yeast_split_indices(args.split_pt, args.real_split)
        real_pool = data[idx]
    if len(real_pool) < args.n_samples:
        real = real_pool[torch.randint(len(real_pool), (args.n_samples,))]
    else:
        real = real_pool[torch.randperm(len(real_pool))[: args.n_samples]]

    model, cfg = load_teacher(args.ckpt, device=device, use_flash_attn=args.use_flash_attn)
    gen = generate_tokens(model, cfg, n_samples=args.n_samples, batch_size=args.batch_size, nfe=args.nfe, device=device)
    uniform = torch.randint(0, cfg.alphabet_size, gen.shape)

    c0 = C0Torch().to(device).eval()
    c0.load_state_dict(torch.load(args.c0_ckpt, map_location=device), strict=True)
    for p in c0.parameters():
        p.requires_grad_(False)

    real_score = score_c0(real, c0, batch_size=args.batch_size, device=device)
    gen_score = score_c0(gen, c0, batch_size=args.batch_size, device=device)
    uniform_score = score_c0(uniform, c0, batch_size=args.batch_size, device=device)

    real_freq = base_freq(real)
    gen_freq = base_freq(gen)
    uniform_freq = base_freq(uniform)

    plot_base_heatmaps(real_freq, gen_freq, uniform_freq, fig_dir / "base.pdf")
    plot_diff_heatmaps(real_freq, gen_freq, uniform_freq, fig_dir / "gen_vs_real.pdf")
    plot_marginals(global_freq(real), global_freq(gen), global_freq(uniform), fig_dir / "marginals.pdf")
    plot_cyclizability(real_score, gen_score, uniform_score, fig_dir / "cycl_dist.pdf")

    torch.save({"real": real, "generated": gen, "uniform": uniform}, out_dir / "tokens.pt")
    pd.DataFrame(
        {
            "source": np.repeat(["real", "generated", "uniform"], [len(real_score), len(gen_score), len(uniform_score)]),
            "c0": np.concatenate([real_score, gen_score, uniform_score]),
        }
    ).to_csv(out_dir / "c0_scores.csv", index=False)
    with open(out_dir / "generated.fa", "w") as f:
        for i, seq in enumerate(seq_strings(gen)):
            f.write(f">generated_{i}\n{seq}\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "ckpt": str(args.ckpt),
                "data_pt": str(args.data_pt),
                "split_pt": str(args.split_pt),
                "real_split": args.real_split,
                "n_samples": args.n_samples,
                "nfe": args.nfe,
                "c0_mean": {
                    "real": float(real_score.mean()),
                    "generated": float(gen_score.mean()),
                    "uniform": float(uniform_score.mean()),
                },
                "c0_std": {
                    "real": float(real_score.std()),
                    "generated": float(gen_score.std()),
                    "uniform": float(uniform_score.std()),
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote PDFs: {sorted(str(p) for p in fig_dir.glob('*.pdf'))}")


if __name__ == "__main__":
    main()
