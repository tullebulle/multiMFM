#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_yeast_split_dmfm_steering import DMFMSteeringRunner, summarize_repeat
from sample_yeast_c0_guidance_hist import hard_token_strings, target_distances_np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate paired unguided and clean-split dMFM-guided yeast C0 samples, "
            "saving sample_scores.csv for diversity/novelty analysis."
        )
    )
    p.add_argument(
        "--teacher_ckpt",
        default="workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt",
    )
    p.add_argument("--teacher_args", default=None)
    p.add_argument("--student_args", default=None)
    p.add_argument(
        "--student_ckpt",
        default="workdir/yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35/last.ckpt",
    )
    p.add_argument("--c0_ckpt", default="dinko/C0free_torch.pt")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--run_name", default="yeast_split_dmfm_c0_samples")
    p.add_argument("--target_c0", type=float, default=1.0)
    p.add_argument("--target_c0_second", type=float, default=None)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--mc", type=int, default=16)
    p.add_argument("--nfe_sample", type=int, default=96)
    p.add_argument("--t_max", type=float, default=1.0)
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


def target_list(args: argparse.Namespace) -> list[float]:
    targets = [float(args.target_c0)]
    if args.target_c0_second is not None:
        targets.append(float(args.target_c0_second))
    return targets


def write_outputs(args: argparse.Namespace, out_dir: Path, runner: DMFMSteeringRunner, unguided: dict, guided: dict) -> None:
    targets = target_list(args)
    ung_scores = unguided["hard_score"].numpy()
    gui_scores = guided["hard_score"].numpy()
    ung_dist = target_distances_np(ung_scores, targets)
    gui_dist = target_distances_np(gui_scores, targets)
    seq_ung = hard_token_strings(
        torch.nn.functional.one_hot(unguided["tokens"], runner.K).float(),
        runner.K,
    )
    seq_gui = hard_token_strings(
        torch.nn.functional.one_hot(guided["tokens"], runner.K).float(),
        runner.K,
    )

    rows = []
    for i, (ung, gui, du, dg, su, sg) in enumerate(zip(ung_scores, gui_scores, ung_dist, gui_dist, seq_ung, seq_gui)):
        rows.append(
            {
                "sample_idx": i,
                "seed": int(args.seed + i),
                "unguided": float(ung),
                "guided": float(gui),
                "delta": float(gui - ung),
                "unguided_abs_to_target": float(du),
                "guided_abs_to_target": float(dg),
                "abs_to_target_improvement": float(du - dg),
                "seq_unguided": su,
                "seq_guided": sg,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_scores.csv", index=False)

    with open(out_dir / "sequences_unguided.fa", "w") as f_ung, open(out_dir / "sequences_guided.fa", "w") as f_gui:
        for _, row in df.iterrows():
            sid = int(row["sample_idx"])
            f_ung.write(f">sample_{sid} seed={int(row['seed'])}\n{row['seq_unguided']}\n")
            f_gui.write(f">sample_{sid} seed={int(row['seed'])}\n{row['seq_guided']}\n")

    summary = {
        "n_samples": int(len(df)),
        "targets": targets,
        "unguided": summarize_repeat(unguided["hard_score"], args.target_c0),
        "guided": summarize_repeat(guided["hard_score"], args.target_c0),
        "delta": {
            "mean": float(df["delta"].mean()),
            "std": float(df["delta"].std(ddof=0)),
            "abs_to_target_improvement_mean": float(df["abs_to_target_improvement"].mean()),
            "frac_improved": float((df["abs_to_target_improvement"] > 0).mean()),
        },
        "dmfm_cost": runner.dmfm_cost(args.mc),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(pd.DataFrame([{"set": "unguided", **summary["unguided"]}, {"set": "guided", **summary["guided"]}]).to_string(index=False))
    print(f"wrote {out_dir / 'sample_scores.csv'}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.out_dir is None:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        target_label = str(args.target_c0).replace("-", "m").replace(".", "p")
        args.out_dir = f"workdir/{args.run_name}_target{target_label}_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"teacher={args.teacher_ckpt}")
    print(f"student={args.student_ckpt}")
    print(f"out_dir={out_dir}")
    print(f"target={target_list(args)}, n={args.n_samples}, mc={args.mc}")

    runner = DMFMSteeringRunner(args, device)

    torch.manual_seed(args.seed)
    unguided = runner.generate_many(n=args.n_samples, batch=args.batch_size, n_mc=None)
    torch.manual_seed(args.seed)
    guided = runner.generate_many(n=args.n_samples, batch=args.batch_size, n_mc=args.mc)

    write_outputs(args, out_dir, runner, unguided, guided)


if __name__ == "__main__":
    main()
