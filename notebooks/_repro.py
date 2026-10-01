"""Helpers for ``reproduce_from_checkpoints.ipynb``.

Each ``run_*`` function re-runs one paper item from the shipped checkpoints (inference
only, never training) at a small n, then builds a comparison frame against the number
it is expected to reproduce. Expected numbers are read from ``results/`` (the stored
full-size run at the *corrected* configuration) or from the paper where the paper's
number is still the right target. Nothing here tunes anything: every flag is the
stored run's flag, only the sample counts shrink.

Agreement is judged by ``check()``, which calls ``_common.compare`` once per row with
that row's own declared tolerance, and records why that tolerance was chosen
(``tol_basis``). Guided sampling is not bit-reproducible (TF32 matmuls,
nondeterministic CUDA backward), so guided rows are judged against the 95% CI of the
small-n estimate, deterministic rows against a tight absolute tolerance.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO, RESULTS, compare

# REPRO_OUT lets a batch job write to its own tree, so it can run beside an interactive session.
OUT = Path(os.environ.get("REPRO_OUT") or (REPO / "outputs" / "reproduce_from_checkpoints")).resolve()
DNA = RESULTS / "dna"


# --------------------------------------------------------------------------- utilities
def device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def fresh(name: str) -> Path:
    """An empty output directory under outputs/reproduce_from_checkpoints/ (old contents removed,
    so a section can never silently pick up files from an earlier execution)."""
    d = OUT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def boot_ci(x, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def half_width(x, **kw) -> float:
    lo, hi = boot_ci(x, **kw)
    return 0.5 * (hi - lo)


def check(rows: list[dict]) -> pd.DataFrame:
    """``compare()`` row by row, each row with its own tolerance.

    Row keys: ``item``, ``expected`` (number or the printed string), ``ours``, optional
    ``tol`` (absolute; omitted = half a unit in the last printed place of ``expected``),
    ``tol_basis`` (why that tolerance), ``n``, ``note``.
    """
    frames = []
    for r in rows:
        row = {"item": r["item"], "paper": r["expected"], "ours": r["ours"], "note": r.get("note", "")}
        if r.get("decimals") is not None:
            row["decimals"] = r["decimals"]
        df = compare([row], atol=r.get("tol"))
        df["n"] = r.get("n", "")
        df["tol_basis"] = r.get("tol_basis", df["tol_src"].iloc[0])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).rename(columns={"paper": "expected"})
    return out[["item", "expected", "ours", "status", "tol", "tol_basis", "n", "note"]]


def verify_checkpoints() -> dict:
    """``scripts/manifest.py verify``: presence + sha256 of every shipped artifact."""
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "manifest.py"), "verify"],
                       cwd=REPO, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    ok = [ln for ln in lines if ln.startswith("ok")]
    bad = [ln for ln in lines if not ln.startswith("ok") and "verified" not in ln]
    return {"returncode": p.returncode, "n_ok": len(ok), "problems": bad,
            "last_line": lines[-1] if lines else "", "stderr": p.stderr[-2000:]}


# --------------------------------------------------------------------------- Table 11
def run_table11() -> tuple[pd.DataFrame, dict]:
    """Guide / oracle C0 regressors on the 8,294 held-out test windows (deterministic, CPU)."""
    from dmfm import rerun

    r = rerun.table11(device="cpu")
    printed = {"guide": ["0.1588", "0.2306", "0.8774", "0.8838"],
               "oracle": ["0.1620", "0.2347", "0.8724", "0.8764"]}
    rows = []
    for role, vals in printed.items():
        for metric, p in zip(("mae", "rmse", "pearson", "spearman"), vals):
            rows.append({"item": f"Table 11 {role} {metric}", "expected": p, "ours": r[role][metric],
                         "n": r[role]["n"], "note": "paper value; deterministic forward pass"})
    return check(rows), r


# --------------------------------------------------------------------------- Table 16
TV_LABELS = {
    # label in results/dna/base_model/table16_tv.csv -> (reference set, other set, printed in manuscript)
    "FLOOR: train real vs held-out real": ("test", "train", "0.0101"),
    "generated vs held-out real": ("test", "generated", "0.0144"),
    "uniform vs held-out real": ("test", "uniform", "0.1064"),
    "generated vs real (matched n)": ("real", "generated", None),
    "uniform vs real (matched n)": ("real", "uniform", None),
    "generated vs training-split real": ("train", "generated", None),
}


def run_table16(n_gen: int, dev: str) -> tuple[pd.DataFrame, dict]:
    """Total-variation distance between global base marginals (Table 16 as corrected).

    The generated set is re-sampled from the L=50 base DFM exactly as the stored run did
    (seeded CPU noise for 10,000 samples, the first ``n_gen`` used; 128 Euler steps to
    t=1.0). The real sets and the uniform baseline are deterministic functions of the
    data and seeds. TV of the generated rows carries sampling noise when ``n_gen`` <
    4,096, so those rows are judged against twice the bootstrap SD over sequences.
    """
    import torch

    from dmfm import api
    from dmfm.regen.base_marginals import global_marginals, sample_generated, uniform_tokens

    t0 = time.time()
    base, cfg = api.load_base(50, dev)
    gen = sample_generated(base, cfg, n=n_gen, length=50, batch_size=512 if dev != "cpu" else 64,
                           nfe=128, t_max=1.0, seed=0, device=dev, total=10000, verbose=False)
    t_gen = time.time() - t0

    win = api.load_windows(50)
    tokens = win["seqs"].numpy().astype(np.int8)
    test_idx = api.split_indices(50, "test").numpy()
    train_idx = api.split_indices(50, "a_train").numpy()
    perm = torch.randperm(len(test_idx), generator=torch.Generator().manual_seed(0)).numpy()
    sets = {"real": tokens[np.sort(test_idx[perm[:4096]])], "test": tokens[test_idx],
            "train": tokens[train_idx], "uniform": uniform_tokens(10000, 50, seed=1)[:4096]}
    marg = {k: global_marginals(v) for k, v in sets.items()}
    marg["generated"] = global_marginals(gen)

    def tv(p, q):
        return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1))

    # bootstrap over generated sequences, for the rows that involve them
    counts = np.stack([(gen == b).sum(axis=1) for b in range(4)], axis=1).astype(float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_gen, size=(1000, n_gen))
    boot_marg = counts[idx].sum(axis=1) / (n_gen * 50.0)

    stored = pd.read_csv(DNA / "base_model" / "table16_tv.csv").set_index("comparison")
    rows, table = [], []
    for label, (ref, other, printed) in TV_LABELS.items():
        ours = tv(marg[ref], marg[other])
        exp = float(stored.loc[label, "total_variation"])
        if other == "generated":
            sd = float(np.std([0.5 * np.abs(b - marg[ref]).sum() for b in boot_marg], ddof=1))
            tol = max(2 * sd, 5e-5) if n_gen < 4096 else 5e-5
            basis = (f"2 x bootstrap SD over {n_gen} generated sequences" if n_gen < 4096
                     else "deterministic (seeded noise, decode margin ~1): 5e-5")
        else:
            tol, basis = 5e-6, "deterministic (data + seeds): 5e-6"
        rows.append({"item": f"TV {label}", "expected": exp, "ours": ours, "tol": tol, "tol_basis": basis,
                     "n": f"{len(sets[ref])} vs {n_gen if other == 'generated' else len(sets[other])}",
                     "note": "stored n=4096 run (results/dna/base_model/table16_tv.csv)"})
        if printed is not None:
            rows.append({"item": f"TV {label} (as printed)", "expected": printed, "ours": ours,
                         "tol": tol if other == "generated" else None,
                         "tol_basis": basis if other == "generated" else "printed precision",
                         "n": f"{len(sets[ref])} vs {n_gen if other == 'generated' else len(sets[other])}",
                         "note": "corrected manuscript Table 16 (commit 7d5e375)"})
        table.append({"comparison": label, "total_variation": ours})
    info = {"n_generated": n_gen, "seconds_sampling": t_gen, "marginals": {k: v.tolist() for k, v in marg.items()},
            "table": pd.DataFrame(table)}
    return check(rows), info


# --------------------------------------------------------------------------- Tables 4-5
T1_SETS = DNA / "c0_guidance" / "dmfm_s1mc32_n1000"


def run_tables4_5() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Diversity of the n=1000 Table 1 sequence sets at the corrected 1x32 operating point
    (CPU; the same analysis code as the stored 4x8 recompute)."""
    from dmfm import rerun

    out = fresh("tables4_5")
    t0 = time.time()
    rerun.tables4_5(out, run_dir_m1=T1_SETS / "target_m1p0", run_dir_p1=T1_SETS / "target_1p0")
    secs = time.time() - t0
    t4 = pd.read_csv(out / "table6_baselines.csv").set_index("set")
    t5 = pd.read_csv(out / "table7_generated.csv")
    old5 = pd.read_csv(DNA / "dmfm_demos" / "diversity" / "dmfm4_norm" / "table7_generated.csv")
    P = json.loads((DNA / "paper_values.json").read_text())

    rows4 = []
    p4 = P["table4_real_baselines"]
    row_of = {"train_sample": "full_sample", "split_train_sample": "a_train_sample",
              "split_val_sample": "a_val_sample", "split_test_sample": "test_sample"}
    src4 = {"exact_unique": "exact_unique_frac", "nearest_generated": "nearest_gen_frac_mean",
            "pairwise_generated": "pairwise_gen_frac_mean", "pairwise_median": "pairwise_gen_frac_median"}
    for prow, orow in row_of.items():
        vals = dict(zip(p4["_columns"], p4[prow]))
        for c, oc in src4.items():
            rows4.append({"item": f"Table 4 {prow} {c}", "expected": vals[c], "ours": float(t4.loc[orow, oc]),
                          "decimals": 4, "n": int(t4.loc[orow, "n_samples"]),
                          "note": f"paper; ours = parent-disjoint {orow} (posterior-independent)"})

    rows5 = []
    p5 = P["table5_generated_diversity"]
    cols5 = {"guide_score_mean": "guide_score_mean", "nearest_gen": "nearest_gen_frac_mean",
             "pairwise_gen": "pairwise_gen_frac_mean", "pairwise_over_heldout": "pairwise_vs_test_sample",
             "nearest_gen_over_heldout": "nearest_gen_vs_test_sample"}
    for target, key in ((-1.0, "-1_guided"), (1.0, "+1_guided")):
        vals = dict(zip(p5["_columns"], p5[key]))
        new = t5[(t5.target_c0 == target) & (t5.set == "guided")].iloc[0]
        old = old5[(old5.target_c0 == target) & (old5.set == "guided")].iloc[0]
        for c, oc in cols5.items():
            rows5.append({"item": f"Table 5 {key} {c}", "expected": vals[c], "ours": float(new[oc]),
                          "decimals": 4, "n": 1000,
                          "note": f"paper (split65k, GLASS); stored dMFM 4x8 recompute = {float(old[oc]):.4f}"})
    return check(rows4), check(rows5), {"seconds": secs, "t4": t4, "t5": t5, "out": out}


# --------------------------------------------------------------------------- Table 1 / Fig 3
def table1_flags(out: Path, target: float, n: int) -> list[str]:
    """Table 1 at the corrected operating point: dMFM, 1 composed step x MC 32 (NFE 32),
    scale-free gradient rule; the step-matched-student rule picks the diagonal student.
    Every other flag is the stored n=1000 run's (results/dna/c0_guidance/dmfm_s1mc32_n1000/*/args.json)."""
    from dmfm import paths

    return ["--paired", "--posterior", "dmfm", "--grad_normalize", "--student_kind", "auto",
            "--teacher_ckpt", str(paths.base_ckpt(50)), "--c0_ckpt", str(paths.c0_guide()),
            "--out_dir", str(out), "--seed", "0", "--n_samples", str(n), "--batch_size", "8",
            "--target_c0", str(target), "--reward_sigma", "0.15", "--reward_scale", "0.5",
            "--mc", "32", "--mc_chunk", "32", "--nfe_sample", "64", "--nfe_value", "1",
            "--t_max", "0.95", "--guide_t_start", "0.50", "--guide_t_end", "0.95",
            "--guidance_frac", "8.0", "--coeff_cap", "10.0", "--grad_clip", "10.0"]


def run_table1(n: int, dev: str, targets=(1.0, -1.0), extra=()) -> tuple[pd.DataFrame, dict]:
    from dmfm import rerun
    from dmfm.experiments import sample_c0_guidance_dmfm, score_c0_oracle

    S = json.loads((DNA / "c0_guidance" / "table1_dmfm_s1mc32_n1000_summary.json").read_text())
    rows, per, secs = [], {}, {}
    for target in targets:
        tag = "target_1p0" if target > 0 else "target_m1p0"
        key = "+1" if target > 0 else "-1"
        out = fresh(f"table1/{tag}")
        t0 = time.time()
        sample_c0_guidance_dmfm.main(table1_flags(out, target, n) + list(extra))
        score_c0_oracle.main(rerun.paper_argv("table1_oracle", run_glob=str(out / "sample_scores.csv"),
                                              out_dir=str(out / "oracle"), seed=0, device=dev))
        secs[key] = time.time() - t0
        d = pd.read_csv(out / "oracle" / "oracle_per_sample.csv").sort_values("sample_idx")
        ref = pd.read_csv(T1_SETS / tag / "oracle" / "oracle_per_sample.csv").sort_values("sample_idx")
        ref = ref[ref.sample_idx.isin(d.sample_idx)]
        assert (ref.sample_idx.values == d.sample_idx.values).all()
        eu, eg = (d.oracle_unguided - target).abs().values, (d.oracle_guided - target).abs().values
        ru, rg = (ref.oracle_unguided - target).abs().values, (ref.oracle_guided - target).abs().values
        hw = half_width(eg)
        diff = eg - rg
        full = S["n1000"][key]
        per[key] = d
        rows += [
            {"item": f"Table 1 y*={key} unguided oracle MAE", "expected": float(ru.mean()), "ours": float(eu.mean()),
             "tol": 1e-3, "tol_basis": "unguided arm is deterministic: 1e-3", "n": n,
             "note": f"stored n=1000 run, same {n} sample ids"},
            {"item": f"Table 1 y*={key} guided oracle MAE (same ids)", "expected": float(rg.mean()), "ours": float(eg.mean()),
             "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})", "n": n,
             "note": f"stored n=1000 run, same ids; paired diff {diff.mean():+.4f} +- {diff.std(ddof=1) / np.sqrt(n):.4f} (SE)"},
            {"item": f"Table 1 y*={key} guided oracle MAE (n=1000 headline)", "expected": round(full["guided_mae"], 3),
             "ours": float(eg.mean()), "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})", "n": n,
             "note": "corrected Table 1 cell, 1 step x MC 32 (published 4x8 split gave "
                     f"{'0.157' if key == '+1' else '0.153'} n=32 pilot / {'0.193' if key == '+1' else '0.172'} n=1000)"},
            {"item": f"Table 1 y*={key} fraction of pairs improved", "expected": full["frac_pairs_improved"],
             "ours": float((eg < eu).mean()), "tol": max(0.02, 1.96 * np.sqrt(0.02 * 0.98 / n)),
             "tol_basis": "binomial 95% half-width at p=0.98", "n": n, "note": "n=1000 run"},
        ]
    return check(rows), {"per_sample": per, "seconds": secs}


# --------------------------------------------------------------------------- Table 13
def run_table13(n: int, dev: str, extra=()) -> tuple[pd.DataFrame, dict]:
    """Exact motif-reward attainment, L=50, NFE 8 (1 x 8, the only split the module has),
    under the published clip-down rule and the corrected scale-free rule."""
    from dmfm.experiments import operating_point_reward_grid

    out = fresh("table13")
    t0 = time.time()
    operating_point_reward_grid.main(["--length", "50", "--reward", "motif", "--out_dir", str(out),
                                      "--nfe", "8", "--steps_values", "1", "--rules", "clip", "norm",
                                      "--n_samples", str(n), "--device", dev, *extra])
    secs = time.time() - t0
    summ = pd.read_csv(out / "L50" / "summary.csv").set_index("config")
    samp = pd.read_csv(out / "L50" / "sample_scores.csv")
    sdir = DNA / "operating_point" / "table13_grid" / "motif_nfe8" / "L50"
    ssum = pd.read_csv(sdir / "summary.csv").set_index("config")
    ssamp = pd.read_csv(sdir / "sample_scores.csv")
    rows = []
    center = float(summ["target_score"].iloc[0])
    rows.append({"item": "Table 13 calibrated target score (90th pct)", "expected": float(ssum["target_score"].iloc[0]),
                 "ours": center, "tol": 1e-3, "tol_basis": "deterministic calibration: 1e-3", "n": 1024,
                 "note": "stored GPU run, Slurm 49479325"})
    for cfg, label, published in (("dmfm_fm1@norm@mc=8", "scale-free (corrected)", None),
                                  ("dmfm_fm1@clip@mc=8", "clip-down (published rule)", "0.126")):
        mine = samp[samp.config == cfg].sort_values("sample_id")
        ref = ssamp[(ssamp.config == cfg) & (ssamp.sample_id < n)].sort_values("sample_id")
        d = (mine.score_unguided - center).abs().values - (mine.score_guided - center).abs().values
        dr = (ref.score_unguided - center).abs().values - (ref.score_guided - center).abs().values
        hw = half_width(d)
        rows += [
            {"item": f"Table 13 motif L50 {label}: unguided |err|", "expected": float((ref.score_unguided - center).abs().mean()),
             "ours": float((mine.score_unguided - center).abs().mean()), "tol": 1e-3,
             "tol_basis": "unguided arm is deterministic: 1e-3", "n": n, "note": "stored run, same sample ids"},
            {"item": f"Table 13 motif L50 {label}: paired reduction (same ids)", "expected": float(dr.mean()),
             "ours": float(d.mean()), "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})", "n": n,
             "note": "stored n=100 run restricted to the same sample ids"},
            {"item": f"Table 13 motif L50 {label}: paired reduction (n=100)", "expected": float(ssum.loc[cfg, "paired_improvement"]),
             "ours": float(d.mean()), "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})", "n": n,
             "note": "stored n=100 GPU run (results/dna/operating_point/table13_grid/motif_nfe8)"},
        ]
        if published:
            rows.append({"item": f"Table 13 motif L50 {label}: paired reduction (paper)", "expected": published,
                         "ours": float(d.mean()), "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})",
                         "n": n, "note": "published Table 13 cell, 0.126 [0.109, 0.144]"})
    return check(rows), {"summary": summ, "seconds": secs}


# --------------------------------------------------------------------------- Table 18 (cosine)
T18_CONFIGS = [("dmfm4_fm4", 1), ("dmfm4_fm4", 2), ("dmfm4_fm4", 4), ("dmfm4_fm4", 8),
               ("dmfm_fm1", 32), ("dmfm4_fm1", 32), ("glass_euler4", 8)]
T18_LABEL = {"dmfm4_fm4": "dMFM 4-step student, 4 steps", "dmfm_fm1": "dMFM diagonal student, 1 step",
             "dmfm4_fm1": "dMFM 4-step student, 1 step", "glass_euler4": "GLASS 4 Euler steps (base DFM)"}


def run_table18(reward: str, n_probes: int, n_repeats: int, dev: str, extra=()) -> tuple[pd.DataFrame, dict]:
    """Value-gradient cosine similarity with the GLASS-128 RK4-50 reference (Table 18 with the
    corrected metric). Probe states, the reference pool and the nested candidate pools are
    seeded exactly as the stored 32-probe x 4-repeat GPU grid, so probes 0..n_probes-1 and
    repeats 0..n_repeats-1 are the same cells."""
    from dmfm.experiments import operating_point_gradient_grid

    out = fresh(f"table18/{reward}")
    t0 = time.time()
    operating_point_gradient_grid.main(["--length", "50", "--reward", reward, "--output_dir", str(out),
                                        "--device", dev, "--n_probes", str(n_probes), "--n_repeats", str(n_repeats),
                                        "--steps_values", "1", "4", "--mc_values", "1", "2", "4", "8", "32",
                                        "--glass_steps_values", "4", *extra])
    secs = time.time() - t0
    g = pd.read_csv(out / "L50" / "gradient_grid.csv")
    s = pd.read_csv(DNA / "operating_point" / "table18_grid" / reward / "L50" / "gradient_grid.csv")
    s_same = s[(s.probe < n_probes) & (s["repeat"] < n_repeats)]
    rows, diag = [], []
    for cfg, mc in T18_CONFIGS:
        mine = g[(g.config == cfg) & (g.mc == mc)].sort_values(["probe", "repeat"])
        same = s_same[(s_same.config == cfg) & (s_same.mc == mc)].sort_values(["probe", "repeat"])
        full = s[(s.config == cfg) & (s.mc == mc)]
        hw = half_width(mine.cosine_similarity)
        name = f"Table 18 {reward} L50 cos, {T18_LABEL[cfg]}, MC {mc} (NFE {int(mine.nfe.iloc[0])})"
        k = len(mine)
        rows += [
            {"item": name + " [same cells]", "expected": float(same.cosine_similarity.mean()),
             "ours": float(mine.cosine_similarity.mean()), "tol": hw,
             "tol_basis": f"95% CI half-width of ours ({k} cells)", "n": k,
             "note": "stored GPU grid, same probes/repeats"},
            {"item": name + " [32x4 grid]", "expected": float(full.cosine_similarity.mean()),
             "ours": float(mine.cosine_similarity.mean()), "tol": hw,
             "tol_basis": f"95% CI half-width of ours ({k} cells)", "n": k,
             "note": "stored GPU grid, all 32 probes x 4 repeats (results/dna/operating_point/table18_grid)"},
        ]
        diag.append({"config": cfg, "mc": mc, "cos_mean": mine.cosine_similarity.mean(), "mae_mean": mine.mae.mean(),
                     "estimate_l2_mean": mine.estimate_l2.mean(), "reference_l2_median": mine.reference_l2.median()})
    corr = float(np.corrcoef(g.mae, g.estimate_l2)[0, 1])
    return check(rows), {"diag": pd.DataFrame(diag), "corr_mae_vs_estimate_norm": corr, "seconds": secs,
                         "reference_l2_range": (float(g.reference_l2.min()), float(g.reference_l2.max()))}


# --------------------------------------------------------------------------- Fig 5 / Tables 14-15
def run_fig5(n_probes: int, n_repeats: int, dev: str, mcs=(1, 2, 4, 8, 16, 32, 64, 128, 256), **overrides) -> tuple[pd.DataFrame, dict]:
    """Finite-N value-gradient error against an independent GLASS-2048 reference, gc reward,
    L=50, scale-free metrics (relative L1, cosine). This is the estimator the published
    Fig 5 actually ran (GLASS on the base DFM on both sides)."""
    from dmfm import rerun

    out = fresh("fig5")
    t0 = time.time()
    rerun.fig5(out, reward="gc", lengths=(50,), n_probes=n_probes, n_repeats=n_repeats,
               mc_values=mcs, **{"reference_mc": 2048, **overrides})
    secs = time.time() - t0
    mine = pd.read_csv(out / "L50" / "gradient_scale_free_per_state.csv")
    st = pd.read_csv(DNA / "scale_free_mc" / "gc" / "L50" / "gradient_scale_free_per_state.csv")
    st_same = st[st.state < n_probes]
    P = json.loads((DNA / "paper_values.json").read_text())["table15_scale_free_full"]["gc"]
    rows = []
    for mc in mcs:
        m = mine[mine.mc == mc].sort_values("state")
        r = st_same[st_same.mc == mc].sort_values("state")
        spread = float(m.cosine.std(ddof=1)) if len(m) > 1 else 0.0
        tol = max(1.96 * spread / np.sqrt(len(m)), 0.01)
        rows.append({"item": f"Fig 5 gc L50 cosine, N={mc} [same {n_probes} states]", "expected": float(r.cosine.mean()),
                     "ours": float(m.cosine.mean()), "tol": tol,
                     "tol_basis": f"1.96 x SD/sqrt({len(m)}) over states (floor 0.01)", "n": len(m),
                     "note": "stored published-run per-state cells"})
        if str(mc) in P:
            rows.append({"item": f"Fig 5 / Table 15 gc L50 cosine, N={mc} [paper, 32 states]",
                         "expected": f"{P[str(mc)]['50'][1]:.2f}", "ours": float(m.cosine.mean()), "tol": tol,
                         "tol_basis": f"1.96 x SD/sqrt({len(m)}) over states (floor 0.01)", "n": len(m),
                         "note": "published 32-state value; ours covers 4 states only"})
    return check(rows), {"per_state": mine, "seconds": secs}


# --------------------------------------------------------------------------- Table 17
def run_table17(n_sources: int, dev: str, times=(0.0, 0.25, 0.5, 0.75), **overrides) -> tuple[pd.DataFrame, dict]:
    """Reward-free posterior diversity at L=50: one-step diagonal dMFM vs 100-step GLASS."""
    from dmfm import rerun

    out = fresh("table17")
    t0 = time.time()
    rerun.table17(out, lengths=(50,), n_sources=n_sources, n_futures=16, times=tuple(times), **overrides)
    secs = time.time() - t0
    cond = pd.read_csv(out / "posterior_condition_metrics.csv")
    scond = pd.read_csv(DNA / "posterior_diversity" / "posterior_condition_metrics.csv")
    P = json.loads((DNA / "paper_values.json").read_text())["table17_posterior_diversity"]
    rows = []
    for t in times:
        for j, method in enumerate(("dmfm", "glass")):
            m = cond[(cond.length == 50) & (cond.t == t) & (cond.method == method)]
            r = scond[(scond.length == 50) & (scond.t == t) & (scond.method == method) & scond.source_id.isin(m.source_id)]
            hw = max(half_width(m.pairwise_hamming_frac), 0.005)
            rows.append({"item": f"Table 17 L50 t={t:.2f} {method} pairwise Hamming [same sources]",
                         "expected": float(r.pairwise_hamming_frac.mean()), "ours": float(m.pairwise_hamming_frac.mean()),
                         "tol": hw, "tol_basis": f"95% CI half-width of ours ({len(m)} sources, floor 0.005)",
                         "n": len(m), "note": f"stored run, same {len(r)} source ids"})
            rows.append({"item": f"Table 17 L50 t={t:.2f} {method} pairwise Hamming [paper]",
                         "expected": f"{P[f'{t:.2f}']['50'][j]:.3f}", "ours": float(m.pairwise_hamming_frac.mean()),
                         "tol": hw, "tol_basis": f"95% CI half-width of ours ({len(m)} sources, floor 0.005)",
                         "n": len(m), "note": "published value (32 sources)"})
    return check(rows), {"conditions": cond, "seconds": secs}


# --------------------------------------------------------------------------- Table 2 (QM9)
QM9_PROPS = ("cv", "mu", "alpha", "gap", "homo", "lumo")


def _read_per_sample(d: Path) -> pd.DataFrame:
    for name in ("per_sample.csv", "per_sample.csv.gz"):
        if (d / name).exists():
            return pd.read_csv(d / name)
    raise FileNotFoundError(d / "per_sample.csv")


def run_qm9(n: int, props=QM9_PROPS, seed: int = 1, rows_kind=("multimfm", "ss")) -> tuple[pd.DataFrame, dict]:
    """Table 2 at the published 4 x 32 split (already optimal): multiMFM and multiMFM-SS
    (Algorithm 3, MFM look-aheads, the corrected harvest), evaluation seed ``seed`` (target seed
    2026 + seed), first ``n`` molecules. The stored seed-1 n=1000 runs use the same seeds, so
    the first n molecules have the same lengths and targets."""
    from multimfm import qm9_results as q

    summ3 = pd.read_csv(RESULTS / "qm9" / "rerun_3seed" / "rerun_3seed_summary.csv")
    paper = json.loads((RESULTS / "qm9" / "paper_values.json").read_text())["table2"]["rows"]
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4))
    logdir = OUT / "qm9" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    rows, secs, summaries, failures = [], {}, {}, []
    for prop in props:
        for kind in rows_kind:
            od = fresh(f"qm9/{kind}/{prop}_s{seed}_n{n}")
            env = dict(os.environ, SEED=str(seed), TARGET_SEED=str(2026 + seed), OUTPUT_DIR=str(od),
                       PB_WORKERS=cpus, VALUE_BATCH_SIZE="0")
            if kind == "multimfm":
                env["PER_SAMPLE_PB"] = "1"
                cmd = ["bash", "scripts/qm9/run_multimfm.sh", prop, str(n)]
            else:
                env.update(TOPK="0", LOOKAHEAD_SOURCE="mfm")
                # SS mu: the submitted per-property value, except lumo 1.5 -> 3.0 (q.SS_CONFIG_RERUN),
                # exactly as the stored 3-seed runs (scripts/qm9/table2_3seed.sbatch).
                cmd = ["bash", "scripts/run_steer_search.sh", prop, str(n)] + (["3.0"] if prop == "lumo" else [])
            t0 = time.time()
            log_path = logdir / f"{kind}_{prop}.log"
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            secs[(kind, prop)] = time.time() - t0
            if proc.returncode != 0:  # keep going: one failed run must not hide the other eleven
                failures.append((kind, prop, proc.returncode, str(log_path)))
                print(f"FAILED {kind} {prop} (exit {proc.returncode}); log: {log_path}")
                continue
            s = json.loads((od / "summary.json").read_text())
            summaries[(kind, prop)] = s
            run = "guided" if kind == "multimfm" else "best_lookahead_pbvalid"
            mine = _read_per_sample(od)
            mine = mine[mine.run == run].sort_values("sample_idx")
            ref = _read_per_sample(RESULTS / "qm9" / "rerun_3seed" / "runs" / kind / f"{prop}_s{seed}")
            ref = ref[(ref.run == run) & (ref.sample_idx < n)].sort_values("sample_idx")
            assert np.allclose(mine.target.values, ref.target.values, rtol=1e-5), f"{kind} {prop}: targets differ"
            err = np.array([q.to_paper_units(prop, e) for e in mine.oracle_abs_error.values])
            rerr = np.array([q.to_paper_units(prop, e) for e in ref.oracle_abs_error.values])
            hw = half_width(err)
            var = "multimfm" if kind == "multimfm" else "ss"
            r3 = summ3[(summ3.variant == var) & (summ3.property == prop)].iloc[0]
            lab = "multiMFM" if kind == "multimfm" else "multiMFM-SS (Alg. 3)"
            unit = q.UNITS[prop]
            pb = next(r for r in s["summary"] if r.get("run") == run).get("pb_pb_valid")
            rows += [
                {"item": f"Table 2 {lab} {prop} [{unit}] (same molecules)", "expected": float(rerr.mean()),
                 "ours": float(err.mean()), "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})", "n": n,
                 "note": f"stored seed-{seed} n=1000 run, first {n} molecules; PB valid "
                         + (f"{pb:.3f}" if pb is not None else "n/a")},
                {"item": f"Table 2 {lab} {prop} [{unit}] (3-seed n=1000)", "expected": float(r3.oracle_mae_mean),
                 "ours": float(err.mean()), "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})", "n": n,
                 "note": f"3-seed mean, SE {float(r3.oracle_mae_se):.3g}"},
                {"item": f"Table 2 {lab} {prop} [{unit}] (paper)",
                 "expected": paper["multiMFM (Ours)" if kind == "multimfm" else "multiMFM-SS (Ours)"][prop],
                 "ours": float(err.mean()), "tol": hw, "tol_basis": f"95% CI half-width of ours (n={n})", "n": n,
                 "note": "published (seed 8)" + ("" if kind == "multimfm" else "; published SS used the base-endpoint harvest")},
            ]
    return (check(rows) if rows else pd.DataFrame()), {"seconds": secs, "summaries": summaries, "failures": failures}
