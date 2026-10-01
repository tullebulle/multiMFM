"""Load a trained TABASCO flow-matching model from a ``model_step_*.pt`` checkpoint.

Extracted from the original ``pb_failure_diagnostics.py``. Two checkpoint
formats exist:

* ``{model, data_stats, args}`` -- saved by ``train_base_flow.py`` (the original
  ``train_pb_valid_curve.py``, e.g. GEOM-Drugs PB-curve runs);
* ``{model_state_dict, stats, model_args}`` -- saved by ``qm9_data.py``; this is
  the format of the QM9 base model in ``checkpoints/base_flow/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from multimfm.train_base_flow import build_model


def load_lightweight_model(checkpoint_path: Path, device: torch.device):
    """Rebuild a ``FlowMatchingModel`` from a checkpoint and move it to ``device``.

    Returns ``(model, data_stats, model_args)``.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint and "data_stats" in checkpoint:
        model_args = argparse.Namespace(**checkpoint["args"])
        data_stats = checkpoint["data_stats"]
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        # QM9 format (qm9_data.py). ``args`` there are the trainer's CLI args; the
        # architecture lives in ``model_args``.
        model_args = argparse.Namespace(**checkpoint["model_args"])
        data_stats = checkpoint["stats"]
        state_dict = checkpoint["model_state_dict"]
    else:
        raise KeyError(
            "Unsupported checkpoint format. Expected either "
            "{model, data_stats, args} or {model_state_dict, stats, model_args}."
        )
    model = build_model(model_args, data_stats)
    model.load_state_dict(state_dict)
    model.set_data_stats(data_stats)
    model = model.to(device).eval()
    return model, data_stats, model_args
