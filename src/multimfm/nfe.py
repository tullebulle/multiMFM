"""Measured network-evaluation counts (NFE) for the QM9 steering drivers.

``NFECounter`` attaches forward hooks to the networks a sampler uses and counts,
per network, the number of *rows* (molecules) pushed through each forward call.
Dividing by the number of generated molecules gives NFE per molecule, the unit used
to compare steering methods at matched budget.

A forward call that runs with autograd enabled and has at least one input that
requires grad is also counted in ``<name>_grad_rows``: every such forward is
back-propagated exactly once by the QM9 samplers (value gradient, DPS gradient), so
``grad_rows`` is the number of rows that also get a backward pass through that
network. Summary conventions (per generated molecule):

* ``nfe_generative``      forward passes of generative networks (base flow model +
                          meta flow map), the primary NFE axis;
* ``nfe_generative_bwd``  of those, the ones that are back-propagated;
* ``nfe_generative_fwd_equiv`` = forward + 2 x backward, a rough cost in
                          forward-pass units (a backward pass costs ~2 forwards);
* ``nfe_regressor`` / ``nfe_regressor_bwd``  the same for the guide/oracle regressor
                          (cheap EGNNs; reported, not part of the NFE axis).

The base model and the MFM student are both 4-layer, 64-wide TABASCO transformers
here (Tables 19/20), so one forward of either costs about the same.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

GENERATIVE = ("base", "mfm")


def _rows(args) -> int:
    for a in args:
        if torch.is_tensor(a) and a.ndim >= 1:
            return int(a.shape[0])
        if hasattr(a, "batch_size") and len(getattr(a, "batch_size")) >= 1:
            return int(a.batch_size[0])
    return 0


def _any_requires_grad(args) -> bool:
    for a in args:
        if torch.is_tensor(a) and a.requires_grad:
            return True
    return False


class NFECounter:
    """Count forward rows per named module. Use ``phase`` to split the counts."""

    def __init__(self) -> None:
        self.rows: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self.grad_rows: dict[str, int] = {}
        self._handles = []
        self._phase = "main"

    # ------------------------------------------------------------ attaching
    def attach(self, name: str, module: torch.nn.Module | None) -> None:
        if module is None:
            return

        def hook(_mod, args, kwargs):
            rows = _rows(args) or _rows(tuple(kwargs.values()))
            key = f"{self._phase}:{name}"
            self.rows[key] = self.rows.get(key, 0) + rows
            self.calls[key] = self.calls.get(key, 0) + 1
            if torch.is_grad_enabled() and (
                _any_requires_grad(args) or _any_requires_grad(tuple(kwargs.values()))
            ):
                self.grad_rows[key] = self.grad_rows.get(key, 0) + rows

        self._handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @contextmanager
    def phase(self, name: str):
        prev, self._phase = self._phase, name
        try:
            yield self
        finally:
            self._phase = prev

    # ------------------------------------------------------------ summaries
    def _sum(self, table: dict[str, int], *, phase: str | None, names) -> int:
        total = 0
        for key, value in table.items():
            ph, nm = key.split(":", 1)
            if (phase is None or ph == phase) and nm in names:
                total += value
        return total

    def summary(self, n_molecules: int, *, phase: str | None = None) -> dict:
        """Per-molecule counts (``phase=None``: all phases)."""
        n = max(1, int(n_molecules))
        out = {}
        for nm in ("base", "mfm", "regressor"):
            out[f"nfe_{nm}"] = self._sum(self.rows, phase=phase, names=(nm,)) / n
            out[f"nfe_{nm}_bwd"] = self._sum(self.grad_rows, phase=phase, names=(nm,)) / n
        fwd = self._sum(self.rows, phase=phase, names=GENERATIVE) / n
        bwd = self._sum(self.grad_rows, phase=phase, names=GENERATIVE) / n
        out["nfe_generative"] = fwd
        out["nfe_generative_bwd"] = bwd
        out["nfe_generative_fwd_equiv"] = fwd + 2.0 * bwd
        return out

    def by_phase(self, n_molecules: int) -> dict:
        phases = sorted({k.split(":", 1)[0] for k in self.rows})
        return {ph: self.summary(n_molecules, phase=ph) for ph in phases}

    def raw(self) -> dict:
        return {"rows": dict(self.rows), "calls": dict(self.calls),
                "grad_rows": dict(self.grad_rows)}


def attach_standard(counter: NFECounter, *, model=None, mfm_student=None,
                    regressors=()) -> NFECounter:
    """Hook the base flow model's network, the MFM student and the regressors."""
    if model is not None:
        counter.attach("base", getattr(model, "net", model))
    counter.attach("mfm", mfm_student)
    for reg in regressors:
        # TFG-Flow's Predictor calls ``self.gnn.forward`` directly (no hooks fire),
        # but every prediction, hard or soft-atom, ends in ``predictor`` (nn.Linear).
        counter.attach("regressor", getattr(reg, "predictor", reg))
    return counter
