"""Steer-search for GEOM: keep the best PB-valid look-ahead the guided run already made.

multiMFM-SS on QM9 harvests the look-aheads the value estimator computes anyway, scores
them with the guide, checks PoseBusters validity, and returns the best valid one instead of
the integration endpoint. It costs no extra generative evaluations -- measured on QM9,
multiMFM and multiMFM-SS both spend 14,848 generative forward passes per molecule -- and it
was worth 1.479 -> 1.311 Bohr^3 there at validity 1.00.

GEOM never had it. This adds it, using the ``step_callback`` hook on ``guided_sample``: the
look-aheads arrive through the stats dict that ``value_gradient`` already fills when asked,
so nothing is sampled twice.

The selection rule is the QM9 one: a candidate replaces the incumbent only if it both scores
better under the guide **and** is PoseBusters-valid. Validity is checked on the few
candidates that beat the incumbent rather than on all of them, which is what keeps the
check affordable.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = ["SteerSearch"]


class SteerSearch:
    """Running incumbent over look-aheads, scored by the guide and filtered by validity.

    Usage::

        ss = SteerSearch(head, targets, score_fn, pb_fn, select_min_t=0.3)
        guided_sample(..., step_callback=ss)
        best = ss.finalize(endpoint_state)
    """

    def __init__(self, score_fn, pb_fn=None, *, select_min_t: float = 0.3,
                 max_pb_checks_per_step: int = 4):
        self.score_fn = score_fn              # (coords, atomics, mask) -> [B] higher is better
        self.pb_fn = pb_fn                    # (coords, atomics, mask) -> [B] bool, or None
        self.select_min_t = float(select_min_t)
        self.max_pb_checks = int(max_pb_checks_per_step)
        self.best_score: torch.Tensor | None = None
        self.best_coords: torch.Tensor | None = None
        self.best_atomics: torch.Tensor | None = None
        self.n_considered = 0
        self.n_pb_checks = 0
        self.n_replacements = 0

    def __call__(self, t: float, stats: dict) -> None:
        if t < self.select_min_t:
            return                                   # too early for an endpoint worth keeping
        coords = stats.get("posterior_coords")
        atomics = stats.get("posterior_atomics")
        if coords is None or atomics is None:
            return
        B, K = coords.shape[0], coords.shape[1]
        flat_c = coords.reshape(B * K, *coords.shape[2:])
        flat_a = atomics.reshape(B * K, *atomics.shape[2:])
        mask = stats["posterior_mask"].reshape(B * K, -1) if "posterior_mask" in stats else None
        with torch.no_grad():
            sc = self.score_fn(flat_c, flat_a, mask).reshape(B, K)
        self.n_considered += B * K

        if self.best_score is None:
            self.best_score = torch.full((B,), -float("inf"), device=sc.device, dtype=sc.dtype)
            self.best_coords = torch.zeros_like(coords[:, 0])
            self.best_atomics = torch.zeros_like(atomics[:, 0])

        # Only candidates that beat the incumbent are worth a validity check.
        order = sc.argsort(dim=1, descending=True)[:, : self.max_pb_checks]
        for rank in range(order.shape[1]):
            k = order[:, rank]
            cand_s = sc.gather(1, k[:, None]).squeeze(1)
            better = cand_s > self.best_score
            if not bool(better.any()):
                continue
            idx = torch.arange(B, device=sc.device)
            cc = coords[idx, k]
            aa = atomics[idx, k]
            ok = better
            if self.pb_fn is not None:
                m = mask.reshape(B, K, -1)[idx, k] if mask is not None else None
                valid = self.pb_fn(cc[better], aa[better], None if m is None else m[better])
                self.n_pb_checks += int(better.sum())
                full = torch.zeros_like(better)
                full[better] = valid.to(full.device)
                ok = full
            if bool(ok.any()):
                self.best_score = torch.where(ok, cand_s, self.best_score)
                self.best_coords[ok] = cc[ok]
                self.best_atomics[ok] = aa[ok]
                self.n_replacements += int(ok.sum())

    def finalize(self, endpoint: TensorDict) -> TensorDict:
        """Return the incumbent where one was found, the integration endpoint otherwise."""
        if self.best_score is None:
            return endpoint
        found = torch.isfinite(self.best_score)
        coords = endpoint["coords"].clone()
        atomics = endpoint["atomics"].clone()
        coords[found] = self.best_coords[found].to(coords.dtype)
        atomics[found] = self.best_atomics[found].to(atomics.dtype)
        return TensorDict({"coords": coords, "atomics": atomics,
                           "padding_mask": endpoint["padding_mask"]},
                          batch_size=endpoint["padding_mask"].shape[0]).to(endpoint.device)

    def summary(self) -> dict:
        return {"ss_considered": self.n_considered, "ss_pb_checks": self.n_pb_checks,
                "ss_replacements": self.n_replacements,
                "ss_frac_with_incumbent": (float(torch.isfinite(self.best_score).float().mean())
                                           if self.best_score is not None else 0.0)}
