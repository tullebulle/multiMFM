"""CPU check: steer_search's lazy PB walk returns exactly Algorithm 3's choice.

Algorithm 3 (paper appendix) scans the look-aheads in time order, keeps the PB-valid one
with the smallest error (strict "<", so the earliest wins ties), and at the end returns
the endpoint if nothing was stored, or if the endpoint is PB-valid with a strictly
smaller error than the stored one. ``multimfm.steer_search.lazy_select`` instead sorts
by error and PB-checks lazily. This script compares the two on random error matrices
(with deliberate ties) and random validity patterns for the unbounded store (exact,
including Algorithm 3's earliest-wins tie rule), and on continuous errors for top-K stores
(where every molecule whose choice could differ must be flagged as
``n_truncated_uncertain``; top-K reorders candidates, so exact ties are not ordered by time).

    python scripts/qm9/check_lazy_select.py
"""
from __future__ import annotations

import torch

from multimfm.steer_search import LookaheadStore, lazy_select


class FakeConverter:
    """Molecule "id" = atomics[0, 0] of the (masked, centred) state."""

    def from_batch(self, batch, **_):
        return [int(round(float(a))) for a in batch["atomics"][:, 0, 0]]


class FakePool:
    def __init__(self, valid):
        self.valid_by_id = valid
        self.n_checked = 0

    def valid(self, ids):
        self.n_checked += len(ids)
        return [bool(self.valid_by_id[i]) for i in ids]


def algorithm3(err_t, valid_t, err_end, valid_end):
    """Reference: sequential Algorithm 3 on time-ordered candidates. Returns index or -1."""
    best, best_err = None, float("inf")
    for c in range(err_t.shape[0]):
        if err_t[c] < best_err and valid_t[c]:
            best, best_err = c, float(err_t[c])
    if best is None:
        return -1
    if err_end < best_err and valid_end:
        return -1
    return best


def one_case(gen, B=40, steps=12, N=6, topk=0, p_valid=0.3, ties=True):
    L, A = 3, 2
    C = steps * N
    ids = torch.arange(B * C).view(B, C)                      # time-ordered look-ahead ids
    if ties:   # many exact ties: checks the earliest-wins rule (unbounded store keeps time order)
        err_t = torch.randint(0, 25, (B, C), generator=gen).float() / 5.0
    else:      # continuous errors, as in real runs (top-K reorders candidates, so no tie rule)
        err_t = torch.rand(B, C, generator=gen) * 5.0
    valid_t = torch.rand(B, C, generator=gen) < p_valid
    end_ids = B * C + torch.arange(B)
    err_end = (torch.randint(0, 25, (B,), generator=gen).float() / 5.0 if ties
               else torch.rand(B, generator=gen) * 5.0)
    valid_end = torch.rand(B, generator=gen) < 0.5
    valid_by_id = torch.cat([valid_t.flatten(), valid_end]).tolist()

    store = LookaheadStore(B, L, A, topk)
    for k in range(steps):
        sl = slice(k * N, (k + 1) * N)
        atomics = torch.zeros(B, N, L, A)
        atomics[:, :, 0, 0] = ids[:, sl].float()
        store.add(torch.zeros(B, N, L, 3), atomics, err_t[:, sl], t_value=k / steps)
    coords_la, atomics_la, err_la, t_la = store.tensors()
    final = {"coords": torch.zeros(B, L, 3), "atomics": torch.zeros(B, L, A)}
    final["atomics"][:, 0, 0] = end_ids.float()
    from tensordict import TensorDict
    final = TensorDict({**final, "padding_mask": torch.zeros(B, L, dtype=torch.bool)}, batch_size=B)
    sel, resolved = lazy_select(err_la=err_la, t_la=t_la, coords_la=coords_la, atomics_la=atomics_la,
                                err_end=err_end, final_state=final,
                                mask=torch.zeros(B, L, dtype=torch.bool), converter=FakeConverter(),
                                sanitize=True, pool=FakePool(valid_by_id))
    Cs = err_la.shape[1]
    mismatches, flagged_ok = 0, True
    kth = err_la.max(dim=1).values
    for b in range(B):
        ref = algorithm3(err_t[b], valid_t[b], float(err_end[b]), bool(valid_end[b]))
        ref_id = int(end_ids[b]) if ref == -1 else int(ids[b, ref])
        got_id = int(end_ids[b]) if int(sel[b]) == Cs else int(round(float(atomics_la[b, sel[b], 0, 0])))
        if topk and store.offered[b] > Cs:
            is_end = int(sel[b]) == Cs
            uncertain = (not resolved[b]) or (is_end and float(err_end[b]) > float(kth[b]))
            if ref_id != got_id and not uncertain:
                flagged_ok = False
            continue
        mismatches += ref_id != got_id
    return mismatches, flagged_ok


def main():
    gen = torch.Generator().manual_seed(0)
    total = 0
    for case in range(200):
        m, ok = one_case(gen, topk=0, p_valid=[0.02, 0.1, 0.3, 0.8][case % 4])
        total += m
        assert ok
    print(f"unbounded store: {total} mismatches vs sequential Algorithm 3 over 200 x 40 molecules")
    assert total == 0
    for topk in (4, 16):
        bad = 0
        for case in range(100):
            m, ok = one_case(gen, topk=topk, p_valid=[0.02, 0.1, 0.3][case % 3], ties=False)
            bad += (m != 0) or (not ok)
        print(f"top-{topk} store: {bad} cases where an unflagged molecule differs from Algorithm 3")
        assert bad == 0
    print("OK")


if __name__ == "__main__":
    main()
