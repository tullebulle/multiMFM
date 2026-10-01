#!/usr/bin/env python
"""Recompute the two steering figures on forward-equivalent compute axes.

Both steering comparisons (Figure 4 right on DNA, the QM9 steering-vs-NFE figure)
put "NFE (generative model)" on the x-axis, i.e. forward passes only. The guided
methods (dMFM on DNA; multiMFM / multiMFM-SS and DPS on QM9) also run a backward
pass through the same network to get the value gradient; the search baselines
(best-of-N, Feynman-Kac, beam, MCTS) do not. This script re-reads the stored
counters and rebuilds both figures on

    axis_k = (generative forward passes) + k * (generative forward passes that are
              inside a differentiated graph)

for k = 0 (the published axis), 1, 2, 3, and reports where every guided/search
pair crosses under each k.

k = 1 is the convention this audit recommends, because *all* model parameters are
frozen (``requires_grad_(False)``) in both codebases and the gradient is taken
w.r.t. the input state only, so the backward pass computes activation gradients
and no weight gradients: roughly one forward's worth of matmuls, not two.
k = 2 is what ``multimfm.nfe.NFECounter`` already writes as
``nfe_generative_fwd_equiv`` (a training-style backward). k = 3 is an upper bound.

Writes, next to this file:
    qm9_axes.csv, dna_axes.csv, crossovers.csv, dna_wallclock.csv, summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
QM9_SWEEP = REPO / "results" / "qm9" / "baselines" / "baseline_sweep_v2.csv"
DNA_CURVES = REPO / "results" / "dna" / "fig4" / "panelB_curves.csv"
DNA_RUNS = REPO / "results" / "dna" / "fig4" / "panelB_runs.csv"

# k = backward cost in units of one forward. k=0 is the published axis. k=1.1 is MEASURED on
# the DNA value-gradient path on an H200 ((fwd+bwd)/fwd = 2.01-2.14 over batch 25-1600, Slurm
# 49431669); k=1 is the weight-free-backward lower bound; k=1.6 a midpoint kept for continuity
# with the first draft; k=2 is multimfm.nfe's existing nfe_generative_fwd_equiv; k=3 an upper
# bound.
KS = (0, 1, 1.1, 1.6, 2, 3)


def kname(k) -> str:
    return ("axis_k%g" % k).replace(".", "p")

GUIDED = {"dna": ["dmfm"], "qm9": ["multiMFM", "multiMFM-SS", "DPS"]}
SEARCH = {"dna": ["best_of_n", "fk", "beam", "mcts"],
          "qm9": ["best-of-N", "Feynman-Kac"]}


# --------------------------------------------------------------------- helpers


def crossing(guided: pd.DataFrame, search: pd.DataFrame, xcol: str, ycol: str):
    """First budget at which `search` is at least as good (low) as `guided`.

    Both curves are interpolated linearly in (log x, log y) on the overlap of
    their x ranges; the crossing is the smallest x where search(x) <= guided(x).
    Returns (x_cross, bracket_lo, bracket_hi, verdict) with x_cross = None when
    the two curves do not cross inside the common range.
    """
    g = guided.sort_values(xcol)
    s = search.sort_values(xcol)
    if len(g) < 2 or len(s) < 2:
        return None, None, None, "too few points"
    lo = max(g[xcol].min(), s[xcol].min())
    hi = min(g[xcol].max(), s[xcol].max())
    if not (hi > lo):
        return None, None, None, "no overlapping budget range"
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 4001))
    gy = np.exp(np.interp(np.log(grid), np.log(g[xcol].to_numpy()), np.log(g[ycol].to_numpy())))
    sy = np.exp(np.interp(np.log(grid), np.log(s[xcol].to_numpy()), np.log(s[ycol].to_numpy())))
    better = sy <= gy
    if better.all():
        return lo, None, None, "search better over the whole shared range"
    if not better.any():
        return None, None, None, "guided better over the whole shared range"
    i = int(np.argmax(better))            # first True
    if i == 0:
        return lo, None, None, "search better over the whole shared range"
    x = float(grid[i])
    # honest bracket: the two *measured* search budgets around the crossing
    xs = s[xcol].to_numpy()
    blo = float(xs[xs <= x].max()) if (xs <= x).any() else None
    bhi = float(xs[xs >= x].min()) if (xs >= x).any() else None
    return x, blo, bhi, "crosses inside the shared range"


# ------------------------------------------------------------------------ DNA


def dna_table() -> pd.DataFrame:
    c = pd.read_csv(DNA_CURVES)
    r = pd.read_csv(DNA_RUNS)
    # nfe_base / nfe_student are only in the per-run file. MCTS's NFE is
    # data-dependent (it varies across repeats), and panelB_curves.csv carries the
    # mean over repeats, so average here too.
    split = (r.groupby(["method", "config"])[["nfe_base_per_output", "nfe_student_per_output"]]
              .mean().reset_index())
    d = c.merge(split, on=["method", "config"], how="left")
    assert np.allclose(d.nfe_base_per_output + d.nfe_student_per_output, d.nfe_gen_per_output)

    # Which generative forwards sit inside a differentiated graph?
    #   dmfm : every student forward (dmfm_value_and_grad runs the student under
    #          enable_grad from a leaf that requires grad); the 96 base steps do not.
    #   search baselines: none (all are @torch.no_grad()).
    d["nfe_gen_in_graph"] = np.where(d.method == "dmfm", d.nfe_student_per_output, 0.0)
    for k in KS:
        d[kname(k)] = d.nfe_gen_per_output + k * d.nfe_gen_in_graph
    d["mae"] = d.mae_mean
    d["se"] = d.mae_sem
    return d


# ------------------------------------------------------------------------ QM9


QM9_RUN_TO_METHOD = {
    "best_lookahead_pbvalid": "multiMFM-SS",
    "final_endpoint": "multiMFM",
    "best_of_n_pbvalid": "best-of-N",
    "fk_pbvalid": "Feynman-Kac",
    "dps": "DPS",
}
QM9_MAX_MC = 16
QM9_MAX_NFE = 10_000


def _cfg(pattern: str, config: str):
    m = re.search(pattern, str(config))
    return float(m.group(1)) if m else None


def qm9_table(*, trim: bool = True) -> pd.DataFrame:
    """Same selection as scripts/qm9/plot_steering_nfe_matched.py.

    ``trim=False`` drops the MAX_MC / MAX_NFE cut so the full measured ladders
    (best-of-N to N=512, multiMFM to MC=32) are available; the trim is a plotting
    choice, not a measurement, and the crossings must not depend on it.
    """
    df = pd.read_csv(QM9_SWEEP)
    df = df[df.run.isin(QM9_RUN_TO_METHOD)].copy()
    df["method"] = df.run.map(QM9_RUN_TO_METHOD)
    keep = []
    for _, r in df.iterrows():
        m, c = r.method, str(r.config)
        if m in ("multiMFM", "multiMFM-SS"):
            mc = _cfg(r"MC=([0-9]+)", c)
            ok = _cfg(r"mu=([0-9.]+)", c) == 3.4 and mc is not None and (not trim or mc <= QM9_MAX_MC)
        elif m == "best-of-N":
            ok = _cfg(r"N=([0-9]+)", c) is not None
        elif m == "Feynman-Kac":
            ok = _cfg(r"lambda=([0-9.]+)", c) == 0.01
        elif m == "DPS":
            ok = _cfg(r"mu=([0-9.]+)", c) == 0.1
        else:
            ok = False
        keep.append(ok)
    df = df[pd.Series(keep, index=df.index)]
    if trim:
        df = df[df.nfe_generative <= QM9_MAX_NFE]
    g = (df.groupby(["method", "config"])
           .agg(nfe_gen_per_output=("nfe_generative", "mean"),
                nfe_gen_in_graph=("nfe_generative_bwd", "mean"),
                nfe_regressor=("nfe_regressor", "mean"),
                mae=("oracle_mae", "mean"),
                guide_mae=("guide_mae", "mean"),
                se=("oracle_mae", lambda s: s.std(ddof=1) / np.sqrt(max(len(s), 1))),
                pb=("pb_valid", "mean"),
                seeds=("seed", "nunique"))
           .reset_index())
    for k in KS:
        g[kname(k)] = g.nfe_gen_per_output + k * g.nfe_gen_in_graph
    return g.sort_values(["method", "nfe_gen_per_output"])


# ----------------------------------------------------------------------- main


def crossovers(tab: pd.DataFrame, side: str) -> pd.DataFrame:
    rows = []
    for gm in GUIDED[side]:
        g = tab[tab.method == gm]
        if not len(g):
            continue
        for sm in SEARCH[side]:
            s = tab[tab.method == sm]
            if not len(s):
                continue
            for k in KS:
                col = kname(k)
                x, blo, bhi, verdict = crossing(g, s, col, "mae")
                rows.append(dict(side=side, guided=gm, search=sm, k=k, axis=col,
                                 crossover=x, bracket_lo=blo, bracket_hi=bhi,
                                 verdict=verdict,
                                 guided_axis_min=g[col].min(), guided_axis_max=g[col].max(),
                                 search_axis_min=s[col].min(), search_axis_max=s[col].max()))
    return pd.DataFrame(rows)


def matched_budget(tab: pd.DataFrame, side: str, k: int) -> pd.DataFrame:
    """Every search curve read at each guided point's budget on ``axis_k``.

    The same construction as ``summary.json -> panelB.matched_nfe_comparison``, but
    on a forward-equivalent axis. NaN where the search curve does not reach that
    budget (no extrapolation).
    """
    col = kname(k)
    rows = []
    for gm in GUIDED[side]:
        g = tab[tab.method == gm].sort_values(col)
        for _, gr in g.iterrows():
            rec = {"side": side, "k": k, "guided": gm, "config": gr.config,
                   "axis": float(gr[col]), "guided_mae": float(gr.mae)}
            for sm in SEARCH[side]:
                s = tab[tab.method == sm].sort_values(col)
                x, y = s[col].to_numpy(), s.mae.to_numpy()
                rec[f"{sm}_at_same_budget"] = (
                    float(np.exp(np.interp(np.log(gr[col]), np.log(x), np.log(y))))
                    if len(x) >= 2 and x.min() <= gr[col] <= x.max() else np.nan)
            best = [v for kk, v in rec.items() if kk.endswith("_at_same_budget") and not np.isnan(v)]
            rec["best_search_at_same_budget"] = min(best) if best else np.nan
            rec["guided_wins"] = (bool(rec["guided_mae"] < rec["best_search_at_same_budget"])
                                  if best else None)
            rows.append(rec)
    return pd.DataFrame(rows)


def conservative_measured(tab: pd.DataFrame, side: str, k: int) -> pd.DataFrame:
    """No interpolation: each search method at its largest MEASURED budget <= the
    guided point's budget, so the search method never gets more compute.

    ``z`` is the difference in units of the combined standard error of the two
    independent 3-seed / 3-repeat means; positive means the search method is better.
    """
    col = kname(k)
    rows = []
    for gm in GUIDED[side]:
        for _, gr in tab[tab.method == gm].sort_values(col).iterrows():
            rec = {"side": side, "k": k, "guided": gm, "config": gr.config,
                   "axis": float(gr[col]), "guided_mae": float(gr.mae), "guided_se": float(gr.se)}
            for sm in SEARCH[side]:
                s = tab[(tab.method == sm) & (tab[col] <= gr[col])].sort_values(col)
                if not len(s):
                    rec[f"{sm}_mae"] = np.nan
                    rec[f"{sm}_budget"] = np.nan
                    rec[f"{sm}_z"] = np.nan
                    continue
                b = s.iloc[-1]
                se = float(np.hypot(gr.se, b.se))
                rec[f"{sm}_mae"] = float(b.mae)
                rec[f"{sm}_budget"] = float(b[col])
                rec[f"{sm}_z"] = (float(gr.mae) - float(b.mae)) / se if se > 0 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def reward_model_sensitivity(tab: pd.DataFrame, side: str, reward_col: str,
                             ratios, k: int = 1) -> pd.DataFrame:
    """What if the reward model (DNA C0 guide / QM9 regressor) went on the axis too?

    ``ratio`` is the cost of one reward-model call in units of one generative
    forward. The guided methods make far MORE reward-model calls than best-of-N
    (which needs one per candidate), so this can only move the crossover against
    the guided methods.
    """
    rows = []
    for r in ratios:
        t = tab.copy()
        t["axis"] = t[kname(k)] + float(r) * t[reward_col]
        for gm in GUIDED[side]:
            g = t[t.method == gm]
            if not len(g):
                continue
            for sm in SEARCH[side]:
                s = t[t.method == sm]
                if not len(s):
                    continue
                x, blo, bhi, verdict = crossing(g, s, "axis", "mae")
                rows.append(dict(side=side, k=k, reward_cost_ratio=r, guided=gm, search=sm,
                                 crossover=x, bracket_lo=blo, bracket_hi=bhi, verdict=verdict,
                                 guided_cheapest=g.axis.min()))
    return pd.DataFrame(rows)


def dna_wallclock() -> pd.DataFrame:
    r = pd.read_csv(DNA_RUNS)
    w = (r.groupby(["method", "config"])
          .agg(elapsed_s_mean=("elapsed_s", "mean"),
               elapsed_s_sd=("elapsed_s", lambda s: s.std(ddof=1)),
               n_repeats=("elapsed_s", "size"),
               n_outputs=("n_outputs", "first"),
               nfe_gen_per_output=("nfe_gen_per_output", "first"),
               nfe_backward_per_output=("nfe_backward_per_output", "first"))
          .reset_index())
    w["s_per_output"] = w.elapsed_s_mean / w.n_outputs
    c = pd.read_csv(DNA_CURVES)[["method", "config", "mae_mean", "mae_sem"]]
    w = w.merge(c, on=["method", "config"], how="left")
    # rows in flight, from the drivers' defaults (--batch-rows 2048, --dmfm-batch 25)
    def rows_in_flight(m, cfg):
        if m == "dmfm":
            mc = int(_cfg(r"MC=([0-9]+)", cfg))
            return 25 * mc                        # student forward width; base flow runs 25
        if m == "best_of_n":
            n = int(_cfg(r"N=([0-9]+)", cfg))
            return min(max(1, 2048 // n) * n, 100 * n)
        if m == "fk":
            return 100 * int(_cfg(r"K=([0-9]+)", cfg))
        if m == "beam":
            return 100 * 1 * 4                     # n_outputs * W * L
        if m == "mcts":
            return 100                             # one rollout batch per iteration
        return np.nan
    w["rows_in_flight"] = [rows_in_flight(m, c) for m, c in zip(w.method, w.config)]
    return w.sort_values(["method", "nfe_gen_per_output"])


def main() -> None:
    dna = dna_table()
    qm9 = qm9_table(trim=True)
    qm9_full = qm9_table(trim=False)
    qm9_full["method"] = qm9_full.method  # no-op, keeps column order stable

    dna.to_csv(HERE / "dna_axes.csv", index=False)
    qm9.to_csv(HERE / "qm9_axes.csv", index=False)
    qm9_full.to_csv(HERE / "qm9_axes_untrimmed.csv", index=False)

    cx = pd.concat([crossovers(dna, "dna"),
                    crossovers(qm9, "qm9"),
                    crossovers(qm9_full, "qm9").assign(side="qm9_untrimmed")],
                   ignore_index=True)
    cx.to_csv(HERE / "crossovers.csv", index=False)

    mb = pd.concat([matched_budget(dna, "dna", k) for k in KS]
                   + [matched_budget(qm9_full, "qm9", k) for k in KS], ignore_index=True)
    mb.to_csv(HERE / "matched_budget.csv", index=False)

    cm = pd.concat([conservative_measured(dna, "dna", k) for k in KS]
                   + [conservative_measured(qm9_full, "qm9", k) for k in KS], ignore_index=True)
    cm.to_csv(HERE / "conservative_measured.csv", index=False)

    # The QM9 regressor has 928,263 parameters against the base flow net's 338,121 and
    # the MFM student's 363,465, so a regressor call is NOT obviously cheap; the DNA C0
    # guide has 88,773 against the base DiT's 2,912,836 (3.0%).
    qm9_reg = qm9_full.copy()
    # steer_search rows carry only the trajectory-phase regressor count; the selection
    # phase costs one more call per harvested look-ahead (mfm variant: 115 per MC).
    qm9_reg["reward_calls"] = np.where(
        qm9_reg.method == "multiMFM-SS", 2.0 * qm9_reg.nfe_regressor, qm9_reg.nfe_regressor)
    rs = pd.concat([
        reward_model_sensitivity(dna.assign(reward_calls=dna.nfe_guide_per_output),
                                 "dna", "reward_calls", (0.0, 0.03, 0.1, 1.0)),
        reward_model_sensitivity(qm9_reg, "qm9", "reward_calls", (0.0, 0.25, 1.0, 2.74)),
    ], ignore_index=True)
    rs.to_csv(HERE / "reward_model_sensitivity.csv", index=False)

    wc = dna_wallclock()
    wc.to_csv(HERE / "dna_wallclock.csv", index=False)

    summary = {
        "convention": {
            "axis_k": "generative forwards + k * generative forwards inside a differentiated graph",
            "k=0": "the published axis (nfe_generative / nfe_gen_per_output)",
            "k=1": "lower bound: frozen params, input-only gradient => no weight-gradient matmuls",
            "k=1.1": "MEASURED: (fwd+bwd)/fwd = 2.01-2.14 on the DNA value-gradient path, "
                     "batch 25-1600, NVIDIA H200, Slurm 49431669 (backward_cost_h200.json)",
            "k=1.6": "midpoint kept from the first draft, before the measurement landed",
            "k=2": "multimfm.nfe's existing nfe_generative_fwd_equiv (training-style backward)",
            "k=3": "upper bound",
            "note": "every verdict in docs/nfe_accounting_audit.md is the same for all k in [1, 2]",
        },
        "dna_axis": {r.method + "|" + r.config: {kname(k): float(r[kname(k)]) for k in KS}
                     for _, r in dna.iterrows()},
        "headline_crossovers": {
            f"{r.side}|{r.guided}|{r.search}|k={r.k}": {
                "crossover": None if pd.isna(r.crossover) else float(r.crossover),
                "measured_bracket": [None if pd.isna(r.bracket_lo) else float(r.bracket_lo),
                                     None if pd.isna(r.bracket_hi) else float(r.bracket_hi)],
                "verdict": r.verdict}
            for _, r in cx.iterrows()
            if r.side in ("dna", "qm9")
            and (r.guided, r.search) in (("dmfm", "fk"), ("dmfm", "best_of_n"),
                                         ("multiMFM-SS", "best-of-N"), ("multiMFM", "best-of-N"))
        },
        "guided_wins_by_setting": {
            f"{r.side}|{r.guided}|{r.config}|k={r.k}": r.guided_wins
            for _, r in mb.iterrows()
        },
    }
    # Fold in the GPU measurement if measure_backward_cost.py has run.
    for tag in ("", "_h200"):
        bc = HERE / f"backward_cost{tag}.json"
        if not bc.exists():
            continue
        raw = json.loads(bc.read_text())
        dna_r = {b: round(v["ratio_fwdbwd_over_fwd"], 3)
                 for b, v in (raw.get("dna", {}).get("by_batch") or {}).items()}
        qm9_r = {b: round(v["ratio_fwdbwd_over_fwd"], 3)
                 for b, v in (raw.get("qm9", {}).get("by_batch") or {}).items()
                 if "ratio_fwdbwd_over_fwd" in v}
        summary.setdefault("measured", {})[f"backward_cost{tag}"] = {
            "device": raw.get("meta", {}).get("device_name"),
            "dna_fwd_plus_bwd_over_fwd_by_batch": dna_r,
            "dna_implied_k": [round(r - 1, 2) for r in dna_r.values()],
            "qm9_guided_step_over_billed_forwards": qm9_r,
            "qm9_note": "includes the off-axis reward regressor's forward and backward; "
                        "a multiplier of 3.0 is exactly axis_k2",
            "qm9_regressor_over_base_forward": {
                k: round(v["regressor_over_base"], 3)
                for k, v in (raw.get("qm9") or {}).items()
                if k.startswith("regressor_vs_base") and "regressor_over_base" in v},
        }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("=== DNA panel B, axes ===")
    print(dna[["method", "config", "nfe_gen_per_output", "nfe_gen_in_graph",
               "nfe_guide_per_output", *[kname(k) for k in KS],
               "mae", "se"]].to_string(index=False))
    print("\n=== QM9 steering, axes (trimmed as plotted) ===")
    print(qm9[["method", "config", "nfe_gen_per_output", "nfe_gen_in_graph", "nfe_regressor",
               *[kname(k) for k in KS], "mae", "se", "pb"]].to_string(index=False))
    print("\n=== QM9 steering, axes (untrimmed) ===")
    print(qm9_full[["method", "config", "nfe_gen_per_output", "nfe_gen_in_graph",
                    *[kname(k) for k in KS], "mae", "se"]].to_string(index=False))
    print("\n=== crossovers ===")
    print(cx.to_string(index=False))
    print("\n=== matched-budget comparison (DNA, k=0 published vs k=1 recommended) ===")
    for k in (0, 1, 1.1, 2):
        sub = mb[(mb.side == "dna") & (mb.k == k)]
        print(f"-- k={k}")
        print(sub[["config", "axis", "guided_mae", "best_of_n_at_same_budget",
                   "fk_at_same_budget", "beam_at_same_budget", "mcts_at_same_budget",
                   "best_search_at_same_budget", "guided_wins"]].to_string(index=False))
    print("\n=== matched-budget comparison (QM9) ===")
    for k in (0, 1, 1.1, 2):
        sub = mb[(mb.side == "qm9") & (mb.k == k) & (mb.guided != "DPS")]
        print(f"-- k={k}")
        print(sub[["guided", "config", "axis", "guided_mae", "best-of-N_at_same_budget",
                   "Feynman-Kac_at_same_budget", "best_search_at_same_budget",
                   "guided_wins"]].to_string(index=False))

    print("\n=== no-interpolation check: search at its largest measured budget <= the guided point ===")
    for side in ("dna", "qm9"):
        for k in (0, 1, 1.1, 2):
            sub = cm[(cm.side == side) & (cm.k == k) & (cm.guided != "DPS")]
            print(f"-- {side} k={k}")
            print(sub.drop(columns=["side", "k"]).to_string(index=False))

    print("\n=== reward-model-on-the-axis sensitivity (k=1) ===")
    print(rs.to_string(index=False))

    print("\n=== DNA wall-clock ===")
    print(wc[["method", "config", "nfe_gen_per_output", "rows_in_flight", "s_per_output",
              "elapsed_s_mean", "elapsed_s_sd", "mae_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
