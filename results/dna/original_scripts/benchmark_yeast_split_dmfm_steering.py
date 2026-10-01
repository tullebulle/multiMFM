#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import math
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
from probe_dna_mfm_glass import load_args_json, load_student
from sample_yeast_c0_guidance_hist import dna_probs_to_park_input
from utils.flow_utils import gaussian_denoiser_flow_step


OLD_NON_DMFM_BASELINES = """method,config,target_mc,gen_nfe_per_output,reward_evals_per_output,mae_mean,mae_sem,frac05_mean,frac10_mean,successes_seen,n_repeats
beam,"L=1,K=48,beta=0.0",1,240.0,3.0,0.521709,0.000000,0.040000,0.050000,13.0,3
best_of_N,"N=2",1,192.0,2.0,0.378953,0.000000,0.060000,0.140000,,3
fk,"L=1,K=48,beta=5.0",1,240.0,3.0,0.310291,0.000000,0.100000,0.100000,10.0,3
mcts,"M=1,K=48,C=0.1,children=4,best_rollout",1,240.0,2.0,0.395437,0.000000,0.090000,0.150000,15.0,3
beam,"L=1,K=32,beta=0.0",2,288.0,4.0,0.500254,0.000000,0.050000,0.070000,19.0,3
best_of_N,"N=3",2,288.0,3.0,0.293011,0.000000,0.100000,0.200000,,3
fk,"L=1,K=32,beta=5.0",2,288.0,4.0,0.210982,0.000000,0.160000,0.270000,23.0,3
mcts,"M=1,K=32,C=0.1,children=4,best_rollout",2,288.0,3.0,0.321696,0.000000,0.130000,0.200000,21.0,3
beam,"L=2,K=48,beta=0.0",4,480.0,5.0,0.332530,0.000000,0.060000,0.130000,24.0,3
best_of_N,"N=5",4,480.0,5.0,0.207317,0.000000,0.140000,0.320000,,3
fk,"L=2,K=48,beta=5.0",4,480.0,5.0,0.223329,0.000000,0.120000,0.360000,26.0,3
mcts,"M=2,K=48,C=0.1,children=4,best_rollout",4,480.0,4.0,0.252176,0.000000,0.200000,0.260000,31.0,3
beam,"L=2,K=16,beta=0.0",8,864.0,13.0,0.153788,0.000000,0.180000,0.280000,102.0,3
best_of_N,"N=9",8,864.0,9.0,0.120121,0.000000,0.260000,0.530000,,3
fk,"L=2,K=16,beta=5.0",8,864.0,13.0,0.124627,0.000000,0.440000,0.560000,158.0,3
mcts,"M=2,K=16,C=0.1,children=4,best_rollout",8,864.0,12.0,0.131805,0.000000,0.240000,0.490000,116.0,3
beam,"L=3,K=12,beta=0.0",16,1584.0,25.0,0.037757,0.000000,0.660000,1.000000,413.0,3
best_of_N,"N=16",16,1536.0,16.0,0.078966,0.000000,0.440000,0.720000,,3
fk,"L=3,K=12,beta=5.0",16,1584.0,25.0,0.121939,0.000000,0.310000,0.570000,340.0,3
mcts,"M=2,K=8,C=0.1,children=4,best_rollout",16,1442.0,24.0,0.085869,0.000000,0.450000,0.700000,234.0,3
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run clean-split dMFM C0 steering sweep and plot matched-budget NFE benchmark.")
    p.add_argument(
        "--teacher_ckpt",
        default="workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt",
    )
    p.add_argument("--teacher_args", default=None)
    p.add_argument(
        "--student_ckpt",
        default="workdir/yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35/epoch=622-step=632000.ckpt",
    )
    p.add_argument("--c0_ckpt", default="dinko/C0free_torch.pt")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fig_dir", default="figures/split_clean")
    p.add_argument("--target_c0", type=float, default=0.30)
    p.add_argument("--n_outputs", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--mc_list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--nfe_sample", type=int, default=96)
    p.add_argument("--nfe_value", type=int, default=1)
    p.add_argument("--guide_t_start", type=float, default=0.01)
    p.add_argument("--guide_t_end", type=float, default=0.95)
    p.add_argument("--guidance_frac", type=float, default=8.0)
    p.add_argument("--coeff_cap", type=float, default=10.0)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--reward_sigma", type=float, default=0.15)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_teacher(ckpt_path: str | Path, device: torch.device) -> tuple[DiTSequenceModel, SimpleNamespace]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_cfg"]
    cfg = cfg if isinstance(cfg, SimpleNamespace) else SimpleNamespace(**cfg)
    cfg.use_flash_attn = bool(getattr(cfg, "use_flash_attn", False))
    cfg.flow_temp = float(getattr(cfg, "flow_temp", 1.0))
    model = DiTSequenceModel(cfg, alphabet_size=cfg.alphabet_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def bcast_like(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while c.ndim < x.ndim:
        c = c[..., None]
    return c.to(device=x.device, dtype=x.dtype)


class DMFMSteeringRunner:
    def __init__(self, cli: argparse.Namespace, device: torch.device):
        self.cli = cli
        self.device = device
        self.teacher, self.cfg = load_teacher(cli.teacher_ckpt, device)
        self.L = int(self.cfg.seq_len)
        self.K = int(self.cfg.alphabet_size)

        teacher_args = cli.teacher_args or str(Path(cli.teacher_ckpt).parent / "args.json")
        dmfm_args = load_args_json(teacher_args)
        self.student, self.student_incompat = load_student(
            dmfm_args,
            alphabet_size=self.K,
            device=device,
            student_ckpt=cli.student_ckpt,
        )
        self.student.eval()
        for p in self.student.parameters():
            p.requires_grad_(False)

        self.c0 = C0Torch().to(device).eval()
        self.c0.load_state_dict(torch.load(cli.c0_ckpt, map_location=device), strict=True)
        for p in self.c0.parameters():
            p.requires_grad_(False)

    def cyclizability_score(self, x_acgt: torch.Tensor) -> torch.Tensor:
        return self.c0(dna_probs_to_park_input(x_acgt))

    def reward_from_score(self, score: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((score - float(self.cli.target_c0)) / float(self.cli.reward_sigma)).pow(2)

    def target_reward(self, x1_acgt: torch.Tensor) -> torch.Tensor:
        return self.reward_from_score(self.cyclizability_score(x1_acgt))

    @torch.no_grad()
    def score_terminal(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        soft = self.cyclizability_score(x)
        tok = x.argmax(-1)
        hard = F.one_hot(tok, self.K).float()
        hard_score = self.cyclizability_score(hard)
        return soft.detach(), hard_score.detach(), tok.detach()

    def dmfm_integrate_diff(self, eps0: torch.Tensor, x_cond: torch.Tensor, t_cond: torch.Tensor, n_steps: int):
        x = eps0
        grid = torch.linspace(0, 1, int(n_steps) + 1, device=x.device, dtype=x.dtype)
        for r0, r1 in zip(grid[:-1], grid[1:]):
            b = x.shape[0]
            x = self.student(r0.expand(b), r1.expand(b), x, t_cond, x_cond)
        return x

    def value_grad_target_dmfm(self, x: torch.Tensor, t: torch.Tensor, *, n_mc: int, n_steps: int):
        B = x.shape[0]
        x_leaf = x.detach().clone().requires_grad_(True)
        eps = torch.randn(B * n_mc, self.L, self.K, device=x.device, dtype=x.dtype)
        x_rep = x_leaf.repeat_interleave(n_mc, 0)
        t_rep = t.detach().to(x.device, x.dtype).repeat_interleave(n_mc)
        x1 = self.dmfm_integrate_diff(eps, x_rep, t_rep, n_steps=n_steps)
        r = float(self.cli.reward_scale) * self.target_reward(x1).reshape(B, n_mc)
        V = torch.logsumexp(r, dim=1) - math.log(n_mc)
        grad = torch.autograd.grad(V.sum(), x_leaf)[0]
        return V.detach(), grad.detach()

    @torch.no_grad()
    def sample_uncond_batch(self, B: int) -> torch.Tensor:
        x = torch.randn(B, self.L, self.K, device=self.device)
        grid = torch.linspace(0, 1, int(self.cli.nfe_sample) + 1, device=self.device)
        for s0, s1 in zip(grid[:-1], grid[1:]):
            x, _, _ = gaussian_denoiser_flow_step(self.cfg, self.teacher, x, s0.expand(B), s1.expand(B))
        return x

    def sample_steered_batch(self, B: int, *, n_mc: int) -> torch.Tensor:
        x = torch.randn(B, self.L, self.K, device=self.device)
        grid = torch.linspace(0, 1, int(self.cli.nfe_sample) + 1, device=self.device)
        for s0, s1 in zip(grid[:-1], grid[1:]):
            b = x.shape[0]
            s = s0.expand(b)
            t_next = s1.expand(b)
            dt = float(s1 - s0)
            with torch.no_grad():
                x_base, _, _ = gaussian_denoiser_flow_step(self.cfg, self.teacher, x, s, t_next)
            if float(s0) >= self.cli.guide_t_start and float(s0) <= self.cli.guide_t_end:
                _, g = self.value_grad_target_dmfm(x, s, n_mc=n_mc, n_steps=self.cli.nfe_value)
                gnorm = g.flatten(1).norm(dim=1).clamp_min(1e-8)
                if self.cli.grad_clip is not None:
                    g = g * (float(self.cli.grad_clip) / gnorm).clamp(max=1.0)[:, None, None]
                coeff = float(self.cli.guidance_frac) * self.teacher.sde_sigma_sq(s)
                coeff = coeff.clamp(max=float(self.cli.coeff_cap))
                x = x_base + dt * coeff[:, None, None] * g
            else:
                x = x_base
        return x.detach()

    def generate_many(self, *, n: int, batch: int, n_mc: int | None) -> dict[str, torch.Tensor]:
        softs, hards, toks = [], [], []
        done = 0
        while done < n:
            b = min(batch, n - done)
            if n_mc is None:
                x = self.sample_uncond_batch(b)
            else:
                x = self.sample_steered_batch(b, n_mc=n_mc)
            soft, hard, tok = self.score_terminal(x)
            softs.append(soft.cpu())
            hards.append(hard.cpu())
            toks.append(tok.cpu())
            done += b
            torch.cuda.empty_cache()
        return {"soft_score": torch.cat(softs).reshape(-1), "hard_score": torch.cat(hards).reshape(-1), "tokens": torch.cat(toks)}

    def guided_step_count(self) -> int:
        grid = torch.linspace(0, 1, int(self.cli.nfe_sample) + 1)
        return sum(1 for s0 in grid[:-1] if self.cli.guide_t_start <= float(s0) <= self.cli.guide_t_end)

    def dmfm_cost(self, mc: int) -> dict[str, float]:
        n_guided = self.guided_step_count()
        return {
            "gen_nfe_per_output": float(self.cli.nfe_sample + n_guided * int(mc) * int(self.cli.nfe_value)),
            "reward_evals_per_output": float(1 + n_guided * int(mc)),
        }


def summarize_repeat(scores: torch.Tensor, target: float) -> dict[str, float]:
    d = (scores - float(target)).abs()
    return {
        "mae": float(d.mean()),
        "frac05": float((d <= 0.05).float().mean()),
        "frac10": float((d <= 0.10).float().mean()),
    }


def plot_nfe_benchmark(agg_df: pd.DataFrame, out_path: Path) -> None:
    style = {
        "dMFM_guided": dict(marker="o", linewidth=2.8),
        "best_of_N": dict(marker="s", linewidth=2.0),
        "beam": dict(marker="^", linewidth=2.0),
        "fk": dict(marker="D", linewidth=2.0),
        "mcts": dict(marker="P", linewidth=2.0),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, g in agg_df.groupby("method"):
        g = g.sort_values("gen_nfe_per_output")
        ax.errorbar(
            g["gen_nfe_per_output"],
            g["mae_mean"],
            yerr=g["mae_sem"],
            capsize=3,
            label=method,
            **style.get(method, dict(marker="o")),
        )
    for _, r in agg_df[agg_df["method"] == "dMFM_guided"].iterrows():
        ax.text(r["gen_nfe_per_output"], r["mae_mean"], f"MC={int(r['target_mc'])}", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Generative model NFE per returned sample")
    ax.set_ylabel(r"$|f_{cyc}(x)-y^*|$")
    ax.set_title("Inference-time Steering Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cli.out_dir or f"workdir/yeast_split_dmfm_steering_benchmark_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    fig_dir = Path(cli.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"teacher={cli.teacher_ckpt}")
    print(f"student={cli.student_ckpt}")
    print(f"out_dir={out_dir}")

    runner = DMFMSteeringRunner(cli, device)
    rows = []
    for mc in cli.mc_list:
        repeat_rows = []
        for rep in range(cli.repeats):
            torch.manual_seed(cli.seed + 1000 * mc + rep)
            out = runner.generate_many(n=cli.n_outputs, batch=cli.batch_size, n_mc=mc)
            rep_summary = summarize_repeat(out["hard_score"], cli.target_c0)
            repeat_rows.append(rep_summary)
            rows.append({"method": "dMFM_guided", "target_mc": mc, "repeat": rep, **rep_summary})
            print(f"mc={mc:02d} repeat={rep} mae={rep_summary['mae']:.5f} frac10={rep_summary['frac10']:.3f}", flush=True)
        rep_df = pd.DataFrame(repeat_rows)
        cost = runner.dmfm_cost(mc)
        rows.append(
            {
                "method": "dMFM_guided_agg",
                "target_mc": mc,
                "repeat": -1,
                "mae": float(rep_df["mae"].mean()),
                "frac05": float(rep_df["frac05"].mean()),
                "frac10": float(rep_df["frac10"].mean()),
                **cost,
            }
        )

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "dmfm_guidance_repeat_rows.csv", index=False)

    agg_rows = []
    for mc, g in raw_df[raw_df["method"] == "dMFM_guided"].groupby("target_mc"):
        cost = runner.dmfm_cost(int(mc))
        agg_rows.append(
            {
                "method": "dMFM_guided",
                "config": f"MC={int(mc)}",
                "target_mc": int(mc),
                **cost,
                "mae_mean": float(g["mae"].mean()),
                "mae_sem": float(g["mae"].std(ddof=1) / math.sqrt(max(len(g), 1))) if len(g) > 1 else 0.0,
                "frac05_mean": float(g["frac05"].mean()),
                "frac10_mean": float(g["frac10"].mean()),
                "successes_seen": np.nan,
                "n_repeats": int(len(g)),
            }
        )
    dmfm_agg = pd.DataFrame(agg_rows)
    baselines = pd.read_csv(io.StringIO(OLD_NON_DMFM_BASELINES))
    agg_df = pd.concat([baselines, dmfm_agg], ignore_index=True)
    agg_df.to_csv(out_dir / "nfe_benchmark_agg.csv", index=False)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(cli), f, indent=2, sort_keys=True)

    plot_nfe_benchmark(agg_df, out_dir / "nfe_benchmark.pdf")
    plot_nfe_benchmark(agg_df, fig_dir / "nfe_benchmark.pdf")
    print(dmfm_agg.to_string(index=False))
    print(f"wrote {fig_dir / 'nfe_benchmark.pdf'}")


if __name__ == "__main__":
    main()
