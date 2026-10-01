"""Small helpers shared by the DNA notebooks (02-06, 09).

The bootstrap functions replicate, line for line, the CI code of the original scripts in
dirichlet-flow-matching/scripts so that CIs recomputed from results/dna agree with the
stored summaries (and hence with the paper) to floating-point precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import REPO, RESULTS, compare, load_json

DNA = RESULTS / "dna"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 70)
DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


def paper() -> dict:
    """Numbers transcribed from the PDF (results/dna/paper_values.json)."""
    return load_json("dna/paper_values.json")


def cmp(rows: list[dict]) -> pd.DataFrame:
    """``compare`` with a per-row tolerance.

    Each row may carry ``dec`` = number of decimals printed in the paper; the tolerance is then
    half a unit in the last printed place (``compare`` on its own infers decimals from the float,
    which drops trailing zeros, e.g. 0.30 -> 1 decimal). Rows may also carry ``atol`` or ``rtol``
    (relative tolerance, e.g. for p-values printed in scientific notation).
    """
    out = []
    for r in rows:
        r = dict(r)
        dec = r.pop("dec", None)
        atol = r.pop("atol", None)
        rtol = r.pop("rtol", None)
        if atol is None and dec is not None:
            atol = 0.5 * 10 ** (-dec) + 1e-12
        if rtol is not None:
            out.append(compare([r], rtol=rtol, atol=0.0 if atol is None else atol))
        else:
            out.append(compare([r], atol=atol))
    df = pd.concat(out, ignore_index=True)
    return df


def show(df: pd.DataFrame) -> pd.DataFrame:
    """Print a one-line match count and return the frame for display."""
    counts = df["status"].value_counts().to_dict()
    print("status counts:", counts)
    return df


# ---------------------------------------------------------------------------- bootstraps
def bootstrap_choice(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """scripts/score_yeast_c0_oracle.py::bootstrap_mean (Table 1)."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(values), size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_integers(values, n_boot: int, seed: int) -> tuple[float, float]:
    """scripts/sample_parent_reward_attainment.py / rescore_gradient_pairs_*.py::bootstrap_mean."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ---------------------------------------------------------------------------- sequences
def tokens(seqs) -> np.ndarray:
    return np.array([[DNA_TO_INT[c] for c in s] for s in seqs], dtype=np.int8)


def pairwise_hamming_frac(tok: np.ndarray) -> np.ndarray:
    """All i<j normalized Hamming distances (as in analyze_yeast_c0_diversity.py)."""
    n, L = tok.shape
    d = (tok[:, None, :] != tok[None, :, :]).sum(-1)
    iu = np.triu_indices(n, k=1)
    return d[iu] / L


def nearest_generated_frac(tok: np.ndarray) -> np.ndarray:
    n, L = tok.shape
    d = (tok[:, None, :] != tok[None, :, :]).sum(-1).astype(float)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1) / L


# ---------------------------------------------------------------------------- Table 1
TABLE1_RUNS = {
    "n1000": "c0_guidance/n1000",       # primary (authors' decision 2026-09-29)
    "pilot_n32": "c0_guidance/pilot_n32",  # values printed in the submitted PDF
}
TABLE1_TARGETS = {"+1": ("1p0", 1.0), "-1": ("m1p0", -1.0)}


def table1_cells(per_sample: pd.DataFrame, target: float, n_boot: int = 2000, seed: int = 0) -> dict:
    """Every Table 1 cell for one target from ``oracle_per_sample.csv``.

    Replicates scripts/score_yeast_c0_oracle.py exactly: oracle scores are float32, errors
    |oracle - target|, CIs are percentile bootstraps of the mean (2,000 resamples,
    ``np.random.default_rng(0).choice``, re-seeded for each statistic), paired improvement =
    |unguided - target| - |guided - target| per pair.
    """
    u = per_sample["oracle_unguided"].to_numpy(np.float32)
    g = per_sample["oracle_guided"].to_numpy(np.float32)
    t = np.float32(target)
    eu, eg = np.abs(u - t), np.abs(g - t)
    imp = eu - eg
    out = {"n_pairs": int(len(u)),
           "oracle_mean_unguided": float(u.mean()), "oracle_mean_guided": float(g.mean()),
           "unguided_mae": float(eu.mean()), "unguided_mae_ci": list(bootstrap_choice(eu, n_boot, seed)),
           "guided_mae": float(eg.mean()), "guided_mae_ci": list(bootstrap_choice(eg, n_boot, seed)),
           "paired_improvement": float(imp.mean()), "paired_improvement_ci": list(bootstrap_choice(imp, n_boot, seed)),
           "frac_pairs_improved": float((imp > 0).mean())}
    return out


def table1_run(run: str) -> dict:
    """Table 1 cells for both targets of one run, plus the guide-scored summary of the sampler."""
    res = {}
    for key, (tag, target) in TABLE1_TARGETS.items():
        base = DNA / TABLE1_RUNS[run] / f"target_{tag}"
        cells = table1_cells(pd.read_csv(base / "oracle" / "oracle_per_sample.csv"), target)
        stored = pd.read_csv(base / "oracle" / "oracle_summary.csv").set_index("set")
        # consistency with the summary the original scorer wrote
        cells["_max_abs_diff_vs_stored_oracle_summary"] = float(np.nanmax(np.abs(np.array([
            cells["oracle_mean_unguided"] - stored.loc["unguided", "oracle_score_mean"],
            cells["oracle_mean_guided"] - stored.loc["guided", "oracle_score_mean"],
            cells["unguided_mae"] - stored.loc["unguided", "oracle_target_mae"],
            cells["guided_mae"] - stored.loc["guided", "oracle_target_mae"],
            cells["paired_improvement"] - stored.loc["paired_guidance_effect", "oracle_target_mae"],
            *(np.array(cells["unguided_mae_ci"]) - stored.loc["unguided", ["oracle_target_mae_ci95_low", "oracle_target_mae_ci95_high"]].to_numpy(float)),
            *(np.array(cells["guided_mae_ci"]) - stored.loc["guided", ["oracle_target_mae_ci95_low", "oracle_target_mae_ci95_high"]].to_numpy(float)),
            *(np.array(cells["paired_improvement_ci"]) - stored.loc["paired_guidance_effect", ["oracle_target_mae_ci95_low", "oracle_target_mae_ci95_high"]].to_numpy(float)),
        ]))))
        guide = load_json(f"dna/{TABLE1_RUNS[run]}/target_{tag}/summary.json")
        cells["guide_scored"] = {"unguided_mean": guide["unguided"]["mean"], "guided_mean": guide["guided"]["mean"],
                                 "unguided_mae": guide["unguided"]["mean_abs_to_target"],
                                 "guided_mae": guide["guided"]["mean_abs_to_target"],
                                 "guided_frac_within_0.10": guide["guided"]["frac_within_0.10"]}
        res[key] = cells
    return res


def regressor_metrics(role: str) -> dict:
    """Held-out test metrics of the C0 guide/oracle recomputed from test_predictions.csv (Table 11)."""
    df = pd.read_csv(DNA / "c0_regressors" / role / "test_predictions.csv")
    y = df["label"].to_numpy(np.float64)
    p = df["prediction"].to_numpy(np.float64)
    return {"n_test_windows": int(len(df)), "mae": float(np.mean(np.abs(p - y))),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "pearson": float(np.corrcoef(p, y)[0, 1]),
            "spearman": float(np.corrcoef(pd.Series(p).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1]),
            "stored_test_metrics": load_json(f"dna/c0_regressors/{role}/metadata.json")["test_metrics"]}


def fmt_ci(m, ci, d=3):
    return f"{m:.{d}f} [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def build_table1_summary(write: bool = True) -> dict:
    """Assemble results/dna/c0_guidance/table1_n1000_summary.json (Table 1 switched to n = 1000)."""
    import json as _json
    pv = paper()
    n1000, n32 = table1_run("n1000"), table1_run("pilot_n32")
    guide, oracle = regressor_metrics("guide"), regressor_metrics("oracle")
    r = lambda x, d=2: float(f"{x:.{d}f}")
    p1, m1 = n1000["+1"], n1000["-1"]
    table_rows = {k: {"target": k,
                      "mean_oracle_score_unguided": f"{v['oracle_mean_unguided']:.3f}",
                      "mean_oracle_score_guided": f"{v['oracle_mean_guided']:.3f}",
                      "unguided_mae": fmt_ci(v["unguided_mae"], v["unguided_mae_ci"]),
                      "guided_mae": fmt_ci(v["guided_mae"], v["guided_mae_ci"]),
                      "paired_improvement": fmt_ci(v["paired_improvement"], v["paired_improvement_ci"])}
                  for k, v in n1000.items()}
    def _tex_ci(m, ci):
        return f"${m:.3f}\\,[{ci[0]:.3f},\\,{ci[1]:.3f}]$"
    latex_rows = {k: (f"${k}$ & ${v['oracle_mean_unguided']:.3f}$ & ${v['oracle_mean_guided']:.3f}$ & "
                      f"{_tex_ci(v['unguided_mae'], v['unguided_mae_ci'])} & {_tex_ci(v['guided_mae'], v['guided_mae_ci'])} & "
                      f"{_tex_ci(v['paired_improvement'], v['paired_improvement_ci'])} \\\\")
                  for k, v in n1000.items()}
    om = r(oracle["mae"])
    suggested = (f"Guidance reduces oracle MAE from {p1['unguided_mae']:.2f} to {p1['guided_mae']:.2f} for $y^\\star=+1$ "
                 f"and from {m1['unguided_mae']:.2f} to {m1['guided_mae']:.2f} for $y^\\star=-1$. ... The guided errors of "
                 f"{p1['guided_mae']:.2f} and {m1['guided_mae']:.2f} are comparable to the oracle's own MAE of {om:.2f} "
                 f"against measured C0 on the held-out parents.")
    summary = {
        "_what": "Table 1 (Sec. 3.1) recomputed for the n = 1000 run, which replaces the n = 32 pilot printed in "
                 "the submitted PDF (authors' decision 2026-09-29). Built by notebooks/_dna_helpers.py::"
                 "build_table1_summary (executed in notebooks/03_dna_cyclizability.ipynb).",
        "_ci_method": "Percentile bootstrap of the mean over pairs: 2,000 resamples, numpy default_rng(seed=0) "
                      "re-seeded for each statistic, rng.choice with replacement, 2.5/97.5% quantiles; float32 "
                      "oracle scores (exactly scripts/score_yeast_c0_oracle.py --bootstrap 2000 --seed 0, the "
                      "method behind the submitted Table 1).",
        "_oracle": "park_cnn oracle workdir/yeast_parent_c0_oracle_2026-07-26/best_state.pt, reverse-complement "
                   "averaged; trained on b_train parents (disjoint from the guide's a_train).",
        "_sampler": "GLASS-posterior guidance on the parent-disjoint L=50 base flow (see args.json; mc 8, nfe_traj 64, "
                    "nfe_value 4, guidance_frac 8, reward_sigma 0.15, reward_scale 0.5, t in [0.5, 0.95], seed 0).",
        "source_runs": {"n1000": {"+1": "dirichlet-flow-matching/workdir/yeast_parent_glass_n1000_target1p0_g8.0_seed0",
                                  "-1": "dirichlet-flow-matching/workdir/yeast_parent_glass_n1000_targetm1p0_g8.0_seed0",
                                  "slurm_jobs": {"+1": 48651189, "-1": 48651190},
                                  "results": "results/dna/c0_guidance/n1000/target_{1p0,m1p0}/"},
                        "pilot_n32": {"+1": "dirichlet-flow-matching/workdir/yeast_parent_glass_pilot_target1p0_g8.0_seed0",
                                      "-1": "dirichlet-flow-matching/workdir/yeast_parent_glass_pilot_targetm1p0_g8.0_seed0",
                                      "slurm_jobs": {"+1": 35242279, "-1": 35242278},
                                      "results": "results/dna/c0_guidance/pilot_n32/target_{1p0,m1p0}/"}},
        "n1000": n1000,
        "table1_n1000_formatted": table_rows,
        "table1_n1000_latex_rows": latex_rows,
        "submitted_n32": {"recomputed": n32, "printed_in_pdf": {k: pv["table1_c0_guidance"][f"target_{k}"] for k in ("+1", "-1")}},
        "in_text_sec3_1": {
            "oracle_mae_unguided_to_guided": {"+1": [r(p1["unguided_mae"]), r(p1["guided_mae"])],
                                              "-1": [r(m1["unguided_mae"]), r(m1["guided_mae"])]},
            "oracle_mae_unguided_to_guided_3dp": {"+1": [r(p1["unguided_mae"], 3), r(p1["guided_mae"], 3)],
                                                  "-1": [r(m1["unguided_mae"], 3), r(m1["guided_mae"], 3)]},
            "oracle_own_heldout_mae": r(oracle["mae"]),
            "n_pairs_per_target": p1["n_pairs"],
            "frac_pairs_improved": {"+1": p1["frac_pairs_improved"], "-1": m1["frac_pairs_improved"]},
            "submitted_text": {"+1": [1.17, 0.16], "-1": [0.83, 0.15], "oracle_own_mae": 0.16},
            "suggested_sentences": suggested,
        },
        "fig3_caption_and_table11": {
            "guide": guide, "oracle": oracle,
            "caption_values_recomputed": {"guide_pearson": r(guide["pearson"]), "guide_mae": r(guide["mae"], 3),
                                          "oracle_pearson": r(oracle["pearson"]), "oracle_mae": r(oracle["mae"], 3)},
            "caption_values_printed": {k: pv["fig3_caption"][k] for k in ("guide_pearson", "guide_mae", "oracle_pearson", "oracle_mae")},
            "table11_printed": pv["table11_c0_regressors"],
            "_note": "Unchanged by the switch to n = 1000 (same guide/oracle checkpoints, same 8,294 test windows).",
        },
    }
    if write:
        out = DNA / "c0_guidance" / "table1_n1000_summary.json"
        out.write_text(_json.dumps(summary, indent=2) + "\n")
    return summary


# ---------------------------------------------------------------------------- RERUN = True helpers
def _rerun_out_default():
    """``outputs/dna/notebook_rerun`` (gitignored), or ``$DNA_RERUN_OUT`` when set.

    scripts/dna/notebook_rerun.sbatch exports a job-unique ``DNA_RERUN_OUT``, because
    ``rerun_dir(fresh=True)`` deletes its target and two concurrent reruns sharing one root would
    wipe each other's run directories. A relative value is taken relative to the repo root (the
    kernel's working directory is notebooks/, not the repo)."""
    import os
    from pathlib import Path as _P

    raw = os.environ.get("DNA_RERUN_OUT", "").strip()
    if not raw:
        return REPO / "outputs" / "dna" / "notebook_rerun"
    p = _P(raw).expanduser()
    return p if p.is_absolute() else REPO / p


RERUN_OUT = _rerun_out_default()     # gitignored; unchanged default when DNA_RERUN_OUT is unset


def rerun_dir(*parts: str, fresh: bool = True) -> Path:
    """Output directory for a small-n notebook rerun (``outputs/dna/notebook_rerun/...``)."""
    import shutil

    d = RERUN_OUT.joinpath(*parts)
    if fresh and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_experiment(module: str, argv: list) -> float:
    """Run ``dmfm.experiments.<module>.main(argv)`` in this kernel (same code path as the
    ``scripts/dna/*.sbatch`` wrappers, which call ``python -m dmfm.experiments.<module>``).
    Returns the wall-clock seconds."""
    import importlib
    import time

    argv = [str(a) for a in argv]
    print(f"$ python -m dmfm.experiments.{module} " + " ".join(argv), flush=True)
    t0 = time.time()
    importlib.import_module(f"dmfm.experiments.{module}").main(argv)
    dt = time.time() - t0
    print(f"[{module}: {dt:.0f} s]", flush=True)
    return dt


def require_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RERUN = True needs a CUDA GPU (e.g. sbatch on gpu_test; see notebooks/_dna_helpers.py)")
    print("GPU:", torch.cuda.get_device_name(0))
    return torch.device("cuda:0")


def rescore_gradient_mae(root, bootstrap: int = 10000, seed: int = 20260802) -> pd.DataFrame:
    """Line-for-line port of dirichlet-flow-matching/scripts/rescore_gradient_pairs_mae.py (Table 18
    MAE files; not yet in ``dmfm.experiments``). Writes the same CSVs and returns the summary."""
    import torch
    from pathlib import Path as _P

    root = _P(root)
    summaries = []
    for length_dir in sorted(root.glob("L*"), key=lambda path: int(path.name[1:])):
        artifact_dir = length_dir
        if not (artifact_dir / "gradient_pairs.pt").exists():
            artifact_dir = length_dir / length_dir.name
        pairs = torch.load(artifact_dir / "gradient_pairs.pt", map_location="cpu", weights_only=False)
        reference = pairs["reference_gradients"][:, None, None]
        estimate = pairs["estimate_gradients"]
        mae = (estimate - reference).abs().mean(dim=(-1, -2))
        rows = []
        for mc_index, mc in enumerate(pairs["mc_values"].tolist()):
            for probe in range(mae.shape[0]):
                for repeat in range(mae.shape[2]):
                    rows.append({"length": int(length_dir.name[1:]), "probe": probe, "repeat": repeat,
                                 "mc": mc, "mae": mae[probe, mc_index, repeat].item()})
        per_sample = pd.DataFrame(rows)
        per_sample.to_csv(artifact_dir / "gradient_mae_per_condition.csv", index=False)
        state_mean = per_sample.groupby(["mc", "probe"], as_index=False)["mae"].mean()
        summary_rows = []
        for mc, group in state_mean.groupby("mc", sort=True):
            values = group["mae"].to_numpy()
            rng = np.random.default_rng(seed + int(length_dir.name[1:]) + int(mc))
            bootstrap_means = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
            summary_rows.append({"mc": mc, "mean_mae": values.mean(), "median_state_mae": np.median(values),
                                 "std_across_states": values.std(ddof=1) if len(values) > 1 else np.nan,
                                 "ci95_low": np.quantile(bootstrap_means, 0.025),
                                 "ci95_high": np.quantile(bootstrap_means, 0.975),
                                 "n_conditions": len(values), "n_repeats": mae.shape[2]})
        summary = pd.DataFrame(summary_rows)
        summary.insert(0, "length", int(length_dir.name[1:]))
        summary.to_csv(artifact_dir / "gradient_mae_summary.csv", index=False)
        summaries.append(summary)
    out = pd.concat(summaries, ignore_index=True)
    out.to_csv(root / "gradient_mae_summary_all_lengths.csv", index=False)
    return out


def execute_notebook(name: str, rerun: bool = True, out_dir=None, timeout: int = 4 * 3600) -> Path:
    """Execute ``notebooks/<name>`` with the first cell's ``RERUN = False`` switched to ``RERUN = True``
    (papermill-style parameter substitution) and write the executed copy to ``out_dir``
    (default ``outputs/dna/notebook_rerun/``); the tracked RERUN = False notebook is left untouched.

    Command line (used by the Slurm test job):  python notebooks/_dna_helpers.py rerun 03_dna_cyclizability.ipynb
    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from pathlib import Path as _P

    nb_dir = REPO / "notebooks"
    src = nb_dir / name
    nb = nbformat.read(src, as_version=4)
    first = nb.cells[0]
    assert first.cell_type == "code" and "RERUN = False" in first.source, "first cell must set RERUN = False"
    first.source = first.source.replace("RERUN = False", f"RERUN = {bool(rerun)}", 1)
    out_dir = _P(out_dir) if out_dir else RERUN_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + ("_RERUN" if rerun else "") + ".ipynb")
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="multimfm")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_dir)}})
    finally:
        nbformat.write(nb, out)
        print("wrote", out)
    return out


def rerun_flag(default: bool = False) -> bool:
    """``RERUN`` with an environment override, so one notebook file serves both modes.

    The tracked notebooks keep ``RERUN = False`` (and their CPU outputs); the Slurm job that
    executes them on a GPU sets ``DNA_RERUN=1`` and nbconvert then runs the rerun cells
    without the notebook being edited (see ``scripts`` in the report / ``execute_notebook``,
    which patches the literal instead). Needs ``source scripts/env.sh`` for the CUDA/libstdc++
    paths that ``import dmfm`` depends on.
    """
    import os

    raw = os.environ.get("DNA_RERUN")
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


#: Printed by every guided-sampling rerun cell.
RERUN_NOT_BIT_EXACT = (
    "Guided sampling is NOT bit-reproducible: the value gradient is accumulated through a "
    "nondeterministic CUDA backward, and cuBLAS/cuDNN kernels depend on batch shape and GPU "
    "model, so guided sequences/scores differ between runs and from the stored full-size run "
    "(they agree statistically). The unguided branch takes no gradient and is deterministic: "
    "its sequences must match the stored run exactly."
)


def rerun_banner(title: str, *, small: str, stored: str, not_bit_exact: bool = True) -> None:
    """Header of a RERUN = True cell: what is rerun here vs what the stored numbers came from."""
    bar = "=" * 108
    print(bar)
    print(f"RERUN (small n, one GPU): {title}")
    print(f"  this run   : {small}")
    print(f"  stored run : {stored}")
    if not_bit_exact:
        print(f"  caveat     : {RERUN_NOT_BIT_EXACT}")
    print(bar, flush=True)


def stored_vs_small(rows: list[dict]) -> pd.DataFrame:
    """:func:`cmp` with the columns renamed for a rerun: ``paper`` = the stored full-size value,
    ``ours`` = this small-n rerun. Rows take the same ``dec`` / ``atol`` / ``rtol`` keys."""
    return show(cmp(rows)).rename(columns={"paper": "stored (full size)", "ours": "this run (small n)"})


def agree_ci(rows: list[dict]) -> pd.DataFrame:
    """Agreement "within noise" of a small-n rerun with a stored full-size value.

    A small-n estimate of a noisy quantity cannot match to the printed precision; the check is
    a confidence interval. Each row is ``{"item", "stored", "ours", "note" (optional)}`` plus
    either

    * ``ci_ours=(low, high)``: the 95 % bootstrap CI of the small-n estimate; the row agrees if
      the stored full-size value lies inside it (the right test for a mean over fewer samples), or
    * ``ci=(low, high)``: a 95 % CI stored with the full-size run; the row agrees if the small-n
      value lies inside it.

    Any other keys are shown as extra columns. ``status`` is "within CI" / "OUTSIDE CI".
    """
    out = []
    for r in rows:
        r = dict(r)
        item, stored, ours, note = r.pop("item"), r.pop("stored"), r.pop("ours"), r.pop("note", "")
        if "ci_ours" in r:
            lo, hi = (float(v) for v in r.pop("ci_ours"))
            probe, which = stored, "this run"
        else:
            lo, hi = (float(v) for v in r.pop("ci"))
            probe, which = ours, "stored run"
        vals = [stored, ours, lo, hi]
        bad = any(v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in vals)
        status = "n/a" if bad else ("within CI" if lo <= float(probe) <= hi else "OUTSIDE CI")
        out.append({"item": item, "stored (full size)": stored, **r, "this run (small n)": ours,
                    "95% CI": f"[{lo:.4g}, {hi:.4g}]", "CI of": which, "status": status, "note": note})
    df = pd.DataFrame(out)
    print("status counts:", df["status"].value_counts().to_dict())
    return df


def has_module(name: str) -> bool:
    """True if ``name`` is importable (used to pick up dna-core modules once they land)."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


__all__ = ["DNA", "paper", "cmp", "show", "bootstrap_choice", "bootstrap_integers", "tokens",
           "pairwise_hamming_frac", "nearest_generated_frac", "TABLE1_RUNS", "TABLE1_TARGETS",
           "table1_cells", "table1_run", "regressor_metrics", "fmt_ci", "build_table1_summary",
           "RERUN_OUT", "rerun_dir", "run_experiment", "require_gpu", "rescore_gradient_mae", "execute_notebook",
           "rerun_flag", "RERUN_NOT_BIT_EXACT", "rerun_banner", "stored_vs_small", "agree_ci", "has_module"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Execute a DNA notebook with RERUN = True (small n, one GPU).")
    ap.add_argument("cmd", choices=["rerun"])
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    for n in a.notebooks:
        execute_notebook(n, rerun=True, out_dir=a.out_dir)

