#!/usr/bin/env python
"""Audit of Table 16 vs Fig 6: the "uniform vs. real" global Pearson r.

Table 16 prints r = 0.1813 (p = 0.819); the same statistic recomputed from Fig 6's own plotted bar
heights is -0.852. This script establishes what the statistic *is* and how much of it is signal.

Findings it computes:
1. The printed pairs (r, p) identify the construction exactly: n = 4 points for the global rows and
   n = 200 (50 positions x 4 bases) for the position-wise rows.
2. Both real and uniform global marginal vectors are constrained to sum to 1, so the centred
   vectors live in a 3-dimensional subspace. For a uniform reference all four multinomial cell
   probabilities are equal, so its sampling noise is *isotropic* in that subspace, and the Pearson
   r against any fixed real vector is the cosine of the angle between a fixed direction and an
   isotropic random direction in R^3 -- which is exactly Uniform(-1, 1).
3. Therefore neither 0.1813 nor -0.852 is "the" value: the statistic carries no information and its
   value is set entirely by the uniform draw.

Writes results/dna/findings_audit/table16_uniform_r.json. CPU, seconds.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Pure-Python t-distribution tail, so this runs on the login node (scipy's compiled extensions
# need a newer libstdc++ than the login image ships; see docs/dna_findings_audit.md).


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * betainc(0.5 * df, 0.5, x)
    return tail if t >= 0 else 1.0 - tail

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "dna" / "findings_audit"


def _ks_uniform(u: np.ndarray) -> float:
    """One-sample KS statistic of `u` against Uniform(0, 1)."""
    u = np.sort(np.asarray(u, float))
    n = len(u)
    i = np.arange(1, n + 1)
    return float(max((i / n - u).max(), (u - (i - 1) / n).max()))


def pearson(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ac, bc = a - a.mean(), b - b.mean()
    r = float(ac @ bc / (np.linalg.norm(ac) * np.linalg.norm(bc)))
    n = len(a)
    t = abs(r) * math.sqrt((n - 2) / max(1e-15, 1 - r * r))
    return r, float(2 * t_sf(t, n - 2))


def implied_n(r: float, p: float) -> float:
    """Sample size implied by a two-sided Pearson (r, p): solves p = 2 sf(t, n-2)."""
    best, bestd = None, np.inf
    for n in range(3, 2001):
        t = abs(r) * math.sqrt((n - 2) / max(1e-15, 1 - r * r))
        pp = 2 * t_sf(t, n - 2)
        d = abs(pp - p)
        if d < bestd:
            best, bestd = n, d
    return {"n": best, "p_at_n": float(2 * t_sf(abs(r) * math.sqrt((best - 2) / (1 - r * r)), best - 2))}


def main() -> None:
    out: dict = {"_what": "Table 16 uniform-vs-real Pearson r: audit of 0.1813 (table) vs -0.852 (Fig 6 data)"}

    # ---- 1. reverse-engineer the printed (r, p) pairs
    printed = json.loads((REPO / "results" / "dna" / "paper_values.json").read_text())["table16_base_marginals"]
    out["paper_table16"] = printed
    out["implied_n_from_printed_p"] = {k: implied_n(*v) for k, v in printed.items() if k != "_columns"}

    # ---- 2. the same statistic on Fig 6's own plotted bar heights
    dig = pd.read_csv(REPO / "results" / "dna" / "figure_digitized" / "fig6_global_marginals_digitized.csv")
    real, gen, uni = dig["real"].to_numpy(), dig["generated"].to_numpy(), dig["uniform"].to_numpy()
    out["fig6_digitized"] = {"base": dig["base"].tolist(), "real": real.tolist(),
                             "generated": gen.tolist(), "uniform": uni.tolist()}
    def tv(a, b):
        return float(0.5 * np.abs(np.asarray(a, float) - np.asarray(b, float)).sum())

    out["fig6_recomputed"] = {
        "global_generated_vs_real": pearson(gen, real),
        "global_uniform_vs_real": pearson(uni, real),
        "total_variation_real_vs_generated": tv(real, gen),
        "total_variation_real_vs_uniform": tv(real, uni),
        "uniform_sums_to": float(uni.sum()),
        "uniform_max_dev_from_0.25": float(np.abs(uni - 0.25).max()),
    }

    # ---- 3. an exactly-uniform reference makes r undefined
    exact = np.full(4, 0.25)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_exact = np.corrcoef(exact, real)[0, 1]
    out["exact_uniform_reference"] = {
        "pearson_r": None if not np.isfinite(r_exact) else float(r_exact),
        "note": "zero variance in the reference -> Pearson r is undefined (nan), so the printed "
                "row can only have come from a finite uniform *sample*",
    }

    # ---- 4. the sampling distribution of the statistic over uniform draws
    rng = np.random.default_rng(0)
    draws = 20000
    n_tokens = 4096 * 50  # the headline Fig 6 / Table 16 set size
    sims = {}
    for label, n_tok in {"4096x50": n_tokens, "1000x50": 1000 * 50, "10000x50": 10000 * 50}.items():
        counts = rng.multinomial(n_tok, np.full(4, 0.25), size=draws) / n_tok
        rc = counts - counts.mean(axis=1, keepdims=True)
        rr = real - real.mean()
        rs = (rc @ rr) / (np.linalg.norm(rc, axis=1) * np.linalg.norm(rr))
        sims[label] = {
            "draws": draws,
            "mean_r": float(rs.mean()), "sd_r": float(rs.std(ddof=1)),
            "quantiles": {q: float(np.quantile(rs, float(q))) for q in ("0.025", "0.25", "0.5", "0.75", "0.975")},
            "frac_abs_r_gt_0.8": float((np.abs(rs) > 0.8).mean()),
            "ks_vs_uniform_m1_1": float(_ks_uniform((rs + 1) / 2)),
            "frac_within_0.02_of_0.1813": float((np.abs(rs - 0.1813) < 0.02).mean()),
            "frac_below_m0.852": float((rs < -0.852).mean()),
        }
    out["null_distribution_over_uniform_draws"] = sims
    out["null_distribution_note"] = (
        "r is the cosine between a fixed centred real-marginal vector and an isotropic random "
        "direction in the 3-dimensional sum-to-zero subspace, i.e. exactly Uniform(-1, 1) and "
        "independent of the number of sequences. Both printed values are ordinary draws from it."
    )

    # ---- 5. what our own regeneration gets, for every (real set, n) it ran
    reg = REPO / "results" / "dna" / "regen" / "fig6_table16" / "correlations.csv"
    if reg.exists():
        c = pd.read_csv(reg)
        u = c[(c.scope == "global") & (c.comparison == "uniform vs. real")]
        out["regen_global_uniform_vs_real"] = {
            "rows": len(u),
            "min": float(u.pearson_r.min()), "max": float(u.pearson_r.max()),
            "headline": [float(x) for x in u[u.headline].pearson_r],
            "all": [[str(a), int(b), float(r)] for a, b, r in zip(u.real_set, u.n_sequences, u.pearson_r)],
        }
        g = c[(c.scope == "global") & (c.comparison == "generated vs. real") & c.headline]
        out["regen_global_generated_vs_real_headline"] = [float(x) for x in g.pearson_r]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table16_uniform_r.json").write_text(json.dumps(out, indent=2))
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
