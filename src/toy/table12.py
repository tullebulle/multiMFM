"""Table 12 (toy training hyperparameters): paper values vs what the runs used.

Table 12 in the submitted PDF prints a single column for "the multiMFM toy
example", but the toy is trained in three stages with different budgets:

    0  base       multimodal flow matching on p1 = U(C) x U([-1,1]^2)
    1  diagonal   GLASS distillation of the MFM on the diagonal s == t
    2  ESD        off-diagonal (consistency) stage, initialised from stage 1

Only stage 0 matches the printed batch size / step count / lr / EMA.  The ESD
stage is missing from the table altogether (reproducibility plan sec. 7.5).
This module reads the actual settings back off the three checkpoints and writes
them next to the printed values:

    python -m toy.table12 --out results/toy/table12_actual.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .paths import BASE_CKPT, MFM_CONSISTENCY_CKPT, MFM_DIAGONAL_CKPT, RESULTS

# Exactly as printed in Table 12 of the submitted PDF (dMFM (40).pdf, p. 27).
PAPER_TABLE12 = {
    "Number of classes K": 3,
    "Continuous dimension d": 2,
    "Source distribution": "N(0, I) for both modalities",
    "Probability path": "Linear interpolant",
    "Model width D": 128,
    "Transformer blocks": 6,
    "Attention heads": 8,
    "MLP expansion ratio": 4,
    "Tokens per sample": 2,
    "Conditioning": "adaLN-Zero",
    "Optimizer": "AdamW",
    "Peak learning rate": 3e-4,
    "Learning-rate schedule": "Cosine decay with warmup",
    "Warmup steps": 500,
    "EMA decay": 0.999,
    "Batch size": 4096,
    "Training steps": 20000,
    "Continuous loss Ly": "MSE on y_hat_1",
    "Discrete loss Lc": "Cross-entropy on z_hat",
    "Consistency loss L_ESD": "ESD loss",
    "Total loss": "L = Ly + Lc + L_ESD",
    "Integrator": "Explicit Euler",
    "Sampling steps": 100,
    "Early-stop margin eps": 1e-3,
    "Class readout": "Sample from softmax(z_hat) at t = 1 - eps",
}

# Fixed by the code rather than by a checkpoint field.
SAMPLING = {
    "Integrator": "Explicit Euler",
    "Sampling steps": 100,
    "Early-stop margin eps": 1e-3,
    "Class readout": "Sample from softmax(z_hat) at t = 1 - eps",
    "Source distribution": "N(0, I) for both modalities",
    "Probability path": "Linear interpolant",
    "Conditioning": "adaLN-Zero",
    "Tokens per sample": 2,
    "Optimizer": "AdamW",
    "Learning-rate schedule": "Cosine decay with warmup",
}

STAGE_LOSS = {
    "base": "L = Ly (MSE on y_hat_1) + Lc (cross-entropy on z_hat)",
    "diagonal": "L = y_weight * ||v_y - v_y^GLASS||^2 + c_weight * ||v_c - v_c^GLASS||^2  (both weights 1.0)",
    "esd": "L = diagonal_weight * L_diag + consistency_weight * L_ESD  (both weights 1.0)",
}


def stage_row(name: str, ckpt_path: Path) -> dict:
    """Read one stage's settings straight out of its checkpoint."""
    if not Path(ckpt_path).exists():
        return {"stage": name, "checkpoint": str(ckpt_path), "present": False}
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tc = dict(ck.get("training_config", {}))
    arch = dict(ck.get("config", {}))
    row = {
        "stage": name,
        "checkpoint": str(ckpt_path),
        "present": True,
        "objective": ck.get("objective"),
        "dataset": ck.get("dataset"),
        "Model width D": arch.get("d_model"),
        "Transformer blocks": arch.get("n_layers"),
        "Attention heads": arch.get("n_heads"),
        "MLP expansion ratio": arch.get("mlp_ratio"),
        "Number of classes K": arch.get("num_classes"),
        "Continuous dimension d": arch.get("y_dim"),
        "Training steps": tc.get("steps"),
        "Batch size": tc.get("batch_size"),
        "Peak learning rate": tc.get("lr"),
        "Warmup steps": tc.get("warmup"),
        "EMA decay": tc.get("ema_decay"),
        "Gradient clipping": tc.get("grad_clip"),
        "Seed": tc.get("seed"),
        "Loss": STAGE_LOSS.get(name),
        "other": {
            k: v
            for k, v in tc.items()
            if k not in {"steps", "batch_size", "lr", "warmup", "ema_decay", "grad_clip", "seed"}
        },
    }
    if "teacher_ckpt" in ck:
        row["teacher_ckpt"] = ck["teacher_ckpt"]
    if "init_ckpt" in ck:
        row["init_ckpt"] = ck["init_ckpt"]
    log = ck.get("log", {})
    if log.get("loss"):
        row["final_logged_loss"] = float(log["loss"][-1])
    return row


def build(base=BASE_CKPT, diagonal=MFM_DIAGONAL_CKPT, esd=MFM_CONSISTENCY_CKPT) -> dict:
    stages = {
        "base": stage_row("base", base),
        "diagonal": stage_row("diagonal", diagonal),
        "esd": stage_row("esd", esd),
    }
    # Which printed cell corresponds to which stage.
    comparison = []
    for key in ("Number of classes K", "Continuous dimension d", "Model width D",
                "Transformer blocks", "Attention heads", "MLP expansion ratio",
                "Peak learning rate", "Warmup steps", "EMA decay", "Batch size",
                "Training steps"):
        comparison.append({
            "row": key,
            "paper": PAPER_TABLE12[key],
            "base": stages["base"].get(key),
            "diagonal": stages["diagonal"].get(key),
            "esd": stages["esd"].get(key),
        })
    return {
        "paper_table12": PAPER_TABLE12,
        "sampling_fixed_by_code": SAMPLING,
        "stages": stages,
        "comparison": comparison,
        "note": (
            "Table 12 prints one column. Its Optimization/Training block matches the BASE stage only. "
            "The diagonal stage used batch 2048 (same lr 3e-4, EMA 0.999, 10k steps); the ESD stage used "
            "10k steps, batch 1024, lr 1e-4, EMA 0.995, gap_anneal_steps 5000 -- none of which appear in "
            "the table (reproducibility plan sec. 7.5). The L_ESD row is also listed under the same "
            "'Total loss' as Ly + Lc, but no single stage optimises all three together: the base stage "
            "uses Ly + Lc, the ESD stage uses L_diag + L_ESD."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_ckpt", default=str(BASE_CKPT))
    p.add_argument("--diagonal_ckpt", default=str(MFM_DIAGONAL_CKPT))
    p.add_argument("--esd_ckpt", default=str(MFM_CONSISTENCY_CKPT))
    p.add_argument("--out", default=str(RESULTS / "table12_actual.json"))
    args = p.parse_args()
    record = build(Path(args.base_ckpt), Path(args.diagonal_ckpt), Path(args.esd_ckpt))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    for row in record["comparison"]:
        print(f"  {row['row']:<24} paper={str(row['paper']):<12} base={str(row['base']):<10} "
              f"diag={str(row['diagonal']):<10} esd={row['esd']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
