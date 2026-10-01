"""Differentiable FP2-fingerprint reward for the GLASS / multiMFM steering engine.

The GEOM-Drugs experiment (paper Table 3) steers with

    r(X_1) = -|| sigmoid(head(X_1)) - c ||^2 ,

where ``head`` is the TFG-Flow structure head and ``c`` the target fingerprint.
With the GLASS value ``V_t = log E[exp(lambda * r(X_1)) | X_t]`` this is exactly
the ``exp(-lambda * L2)`` tilt of TFG-Flow.

The shared steering engine lives in :mod:`multimfm.glass_guidance` (ported from
the QM9 side of the same codebase, which has no fingerprint reward). Rather than
duplicating 1,200 lines of engine, this module *adds* the fingerprint reward to
it through :func:`install_fingerprint_reward`, which is exactly the three
additive hunks the GEOM checkout carried on top of the QM9 driver:

1. ``terminal_reward`` learns the ``"fingerprint"`` reward name;
2. ``grouped_rewards`` repeats a 2-D per-molecule target ``[B, n_bits]`` along
   the batch axis instead of flattening it;
3. the two helpers below.

The patch is idempotent, and a no-op if :mod:`multimfm.glass_guidance` ever
grows native fingerprint support, so it keeps working if the hunks are merged
upstream into the shared module.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from multimfm import glass_guidance as engine

__all__ = [
    "fingerprint_prediction",
    "fingerprint_reward",
    "fingerprint_tanimoto_reward",
    "soft_tanimoto",
    "install_fingerprint_reward",
    "engine_supports_fingerprint",
]


def fingerprint_prediction(
    state: TensorDict, regressor, soft: bool | None = None
) -> torch.Tensor:
    """Predicted FP2 bit-probabilities ``[B, n_bits]`` from the structure head.

    Differentiable w.r.t. ``coords`` always, and w.r.t. ``atomics`` when it
    requires grad (soft-embedding path, mirroring ``tfg_property_prediction``).
    Coordinates are consumed in the model's normalized (COM-centered,
    /normalizer) frame -- the same frame the structure head was trained on.

    Args:
        soft: Force the soft (``True``) or hard (``False``) atom-type path
            instead of choosing it from ``state["atomics"].requires_grad``.
            ``None`` (the default) keeps the original auto-detecting behaviour,
            so every recorded run is unaffected. Used by
            ``multimfm.geom.diagnostics fd-check``, which has to evaluate the
            *same* function under ``torch.no_grad()`` for a finite difference.
    """
    if regressor is None:
        raise ValueError("A structure-fingerprint regressor is required.")
    coords = state["coords"].float()
    mask = (~state["padding_mask"]).long()
    atomics = state["atomics"].float()
    use_soft = atomics.requires_grad if soft is None else bool(soft)
    if use_soft:
        temperature = float(getattr(regressor, "soft_atom_temperature", 0.25))
        probs = torch.softmax(atomics / max(temperature, 1e-4), dim=-1)
        h = probs @ regressor.atom_emb.weight
        h = h * mask.unsqueeze(-1)
        h, _ = regressor.gnn.forward(h, coords, mask.bool())
        h = h * mask.unsqueeze(-1)
        h = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        logits = regressor.predictor(h)
    else:
        atom_types = atomics.argmax(dim=-1).long()
        logits = regressor(coords, atom_types, mask)
    return torch.sigmoid(logits)


def fingerprint_reward(
    state: TensorDict, regressor, target_fp, soft: bool | None = None
) -> torch.Tensor:
    """``r(X_1) = -||sigmoid(head(X_1)) - target_fp||^2``, shape ``[B]``.

    ``target_fp`` is expected already aligned with ``state``'s batch
    (``[B, n_bits]``). ``soft`` is forwarded to :func:`fingerprint_prediction`.
    """
    pred = fingerprint_prediction(state, regressor, soft=soft)
    target = torch.as_tensor(target_fp, dtype=pred.dtype, device=pred.device)
    return -((pred - target) ** 2).sum(dim=-1)


def soft_tanimoto(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable Tanimoto between a predicted bit-probability vector and a target.

    The reward the steering engine has always used is ``-||p - c||^2``, which is a poor
    surrogate for the quantity Table 3 actually reports: over composition candidates its
    Spearman correlation with the counted FP2 Tanimoto is 0.375, against 0.708 for this
    relaxation (`results/geom/head_tanimoto_direction.json`). Optimising a surrogate that
    ranks candidates half as well as an available alternative is a defect in the objective,
    not in the method, and it has never been tried in-trajectory.
    """
    inter = (pred * target).sum(-1)
    return inter / (pred.sum(-1) + target.sum(-1) - inter + 1e-9)


def fingerprint_tanimoto_reward(
    state: TensorDict, regressor, target_fp, soft: bool | None = None
) -> torch.Tensor:
    """``r(X_1) = soft_tanimoto(sigmoid(head(X_1)), target_fp)``, shape ``[B]``."""
    pred = fingerprint_prediction(state, regressor, soft=soft)
    target = torch.as_tensor(target_fp, dtype=pred.dtype, device=pred.device)
    return soft_tanimoto(pred, target)


def engine_supports_fingerprint() -> bool:
    """True when :mod:`multimfm.glass_guidance` already knows the reward."""
    return bool(getattr(engine, "_HAS_FINGERPRINT_REWARD", False))


def install_fingerprint_reward() -> None:
    """Teach the shared steering engine the ``"fingerprint"`` reward.

    Idempotent; call it before :func:`multimfm.glass_guidance.guided_sample`.
    """
    if engine_supports_fingerprint():
        return

    base_terminal_reward = engine.terminal_reward

    def terminal_reward(state, *, reward_name, target_x, property_name,
                        property_target, property_regressor=None):
        if reward_name == "fingerprint":
            return fingerprint_reward(state, property_regressor, property_target)
        if reward_name == "fingerprint_tanimoto":
            return fingerprint_tanimoto_reward(state, property_regressor, property_target)
        return base_terminal_reward(
            state,
            reward_name=reward_name,
            target_x=target_x,
            property_name=property_name,
            property_target=property_target,
            property_regressor=property_regressor,
        )

    def grouped_rewards(state, *, batch_size, posterior_samples, reward_name,
                        target_x, property_name, property_target,
                        property_regressor=None):
        if torch.is_tensor(property_target) and property_target.ndim > 0:
            if property_target.ndim == 2:
                # Per-molecule vector target (e.g. fingerprint [B, n_bits]) ->
                # expand along the batch axis to [B * posterior_samples, n_bits].
                property_target = property_target.repeat_interleave(
                    posterior_samples, dim=0
                )
            else:
                property_target = property_target.reshape(-1).repeat_interleave(
                    posterior_samples
                )
        rewards = engine.terminal_reward(
            state,
            reward_name=reward_name,
            target_x=target_x,
            property_name=property_name,
            property_target=property_target,
            property_regressor=property_regressor,
        )
        return rewards.view(batch_size, posterior_samples)

    terminal_reward.__doc__ = base_terminal_reward.__doc__
    engine.terminal_reward = terminal_reward
    engine.grouped_rewards = grouped_rewards
    engine.fingerprint_prediction = fingerprint_prediction
    engine.fingerprint_reward = fingerprint_reward
    engine._HAS_FINGERPRINT_REWARD = True
