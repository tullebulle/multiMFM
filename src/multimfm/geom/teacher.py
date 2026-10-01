"""Load a GEOM-Drugs continuous-FM teacher (base model) for steering / distillation.

The GEOM teachers are PyTorch-Lightning checkpoints
(``{state_dict (model.* prefixed), data_stats, optimizer_states[ema], ...}``),
NOT the single-script ``{model_state_dict, stats, model_args}`` format that the
QM9 driver's ``load_flow_model`` understands. This module builds the
``FlowMatchingModel`` from the Lightning checkpoint and, by default, swaps in the
EMA weights -- the weights used for validation during training and the ones the
paper's runs used.

Ported from ``geom-drug-tabasco/metaflow/geom_teacher.py`` (the
``load_geom_teacher`` entry point) and ``glass/glass_fingerprint_steering.py``
(``load_teacher``), which were two copies of the same loader.

    python -m multimfm.geom.teacher                  # inspect the default teacher
    python -m multimfm.geom.teacher --checkpoint ... --no-ema
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from multimfm.geom.paths import DEFAULT_TEACHER
from multimfm.train_base_flow import build_model

__all__ = ["infer_arch", "teacher_model_args", "load_geom_teacher"]


def infer_arch(state_dict: dict) -> tuple[int, int]:
    """Infer ``(hidden_dim, num_layers)`` from the teacher weights."""
    hidden_dim = state_dict["model.net.linear_embed.weight"].shape[0]
    layer_ids = {
        int(k.split("transformer.layers.")[1].split(".")[0])
        for k in state_dict
        if "transformer.layers." in k
    }
    return hidden_dim, (max(layer_ids) + 1 if layer_ids else 0)


def teacher_model_args(hidden_dim: int, num_layers: int, num_heads: int | None = None):
    """The architecture namespace ``build_model`` expects for a GEOM teacher.

    ``num_random_augmentations=7`` and ``sample_schedule="log"`` reproduce the
    Hydra training config (``configs/geom/experiment/geom_dfm.yaml``); they
    affect sampling, not the parameter shapes.
    """
    return SimpleNamespace(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads or (8 if hidden_dim >= 128 else 4),
        implementation="reimplemented",
        no_cross_attention=False,
        atomics_mode="dfm_gaussian",
        dfm_beta_schedule="linear",
        dfm_beta_power=1.0,
        num_random_augmentations=7,
        sample_schedule="log",
    )


def load_geom_teacher(
    checkpoint: str | Path = DEFAULT_TEACHER,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
    *,
    num_heads: int | None = None,
    verbose: bool = True,
):
    """Return ``(model, data_stats, model_args)`` for a Lightning GEOM teacher.

    Args:
        checkpoint: Lightning ``.ckpt`` written by the Hydra base trainer.
        device: Device to move the model to.
        use_ema: Swap in the EMA shadow weights from ``optimizer_states``.
        num_heads: Override the inferred attention-head count.
        verbose: Print a one-line description of what was loaded.
    """
    checkpoint = Path(checkpoint)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    stats = ckpt.get("data_stats")
    if stats is None:
        raise SystemExit(f"teacher checkpoint {checkpoint} has no 'data_stats'.")

    # The Lightning teacher's data_stats omits a couple of keys the QM9 loader used
    # to add and that the eval / MoleculeConverter path needs. Backfill them from the
    # tabasco heavy-atom vocabulary (order must match the teacher's atomics channels).
    if not stats.get("atom_names"):
        from tabasco.chem.constants import ATOM_NAMES  # noqa: PLC0415

        stats["atom_names"] = list(ATOM_NAMES)
    if len(stats["atom_names"]) != int(stats["atom_dim"]):
        raise SystemExit(
            f"atom_names ({len(stats['atom_names'])}) != atom_dim ({stats['atom_dim']})."
        )
    stats.setdefault("coordinate_normalizer", 2.0)  # GEOM teacher trained on coords/2.0

    sd = ckpt["state_dict"]
    hidden_dim, num_layers = infer_arch(sd)
    margs = teacher_model_args(hidden_dim, num_layers, num_heads)
    model = build_model(margs, stats)
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if (missing or unexpected) and verbose:
        print(f"# teacher load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")

    weights = "raw"
    if use_ema:
        # EMA shadow weights live in the EMAOptimizer state as a flat list of tensors
        # in model.parameters() order (see tabasco.callbacks.ema.EMAOptimizer.state_dict).
        opt_states = ckpt.get("optimizer_states") or []
        ema = next((o["ema"] for o in opt_states if isinstance(o, dict) and "ema" in o), None)
        if ema is None:
            print("# WARNING: EMA weights requested but none found; using raw weights.")
        else:
            params = list(model.parameters())
            if len(params) != len(ema):
                raise SystemExit(f"EMA/param count mismatch: {len(ema)} vs {len(params)}.")
            with torch.no_grad():
                for prm, e in zip(params, ema):
                    prm.copy_(e.to(prm.device, prm.dtype))
            weights = "ema"

    model.set_data_stats(stats)
    model = model.to(device).eval()
    if verbose:
        print(
            f"# loaded GEOM teacher: {checkpoint} weights={weights} "
            f"hidden={hidden_dim} layers={num_layers} heads={margs.num_heads} "
            f"atom_dim={stats.get('atom_dim')} max_atoms={stats.get('max_num_atoms')}"
        )
    model.geom_weights = weights  # provenance for the result JSONs
    return model, stats, margs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-ema", action="store_true", help="Use raw (not EMA) weights.")
    args = p.parse_args()
    model, stats, margs = load_geom_teacher(
        args.checkpoint, args.device, use_ema=not args.no_ema
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters      : {n_params:,}")
    print(f"weights         : {model.geom_weights}")
    print(f"max_num_atoms   : {stats['max_num_atoms']}")
    print(f"atom_names      : {stats['atom_names']}")
    print(f"normalizer      : {stats['coordinate_normalizer']}")
    print(f"architecture    : {vars(margs)}")


if __name__ == "__main__":
    main()
