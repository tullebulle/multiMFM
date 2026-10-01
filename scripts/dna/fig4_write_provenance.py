#!/usr/bin/env python
"""Write results/dna/fig4/PROVENANCE.md from the collected summary.json.

Run by scripts/dna/fig4_collect.sh; job IDs come from the shards themselves
(each records its SLURM_JOB_ID), so the file cannot drift from the runs.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "dna" / "fig4"

HEADER = """# Figure 4 -- provenance of the rerun

**The models and the results are gone; the producers are not.** Both panels were made in
May 2026 from the `split65k` models
(`yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35`, dMFM
`yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35`) and the
Keras-derived C0 guide `dinko/C0free_torch.pt`; all three were purged, and both run
directories (`workdir/yeast_split_value_mc_2026-05-13_12-48-10`,
`workdir/yeast_split_dmfm_steering_benchmark_2026-05-13_12-51-45`) survive empty.

Both **producer scripts** were recovered on 2026-09-29 from `tullebulle/DNA-MFM@187fe7b`
and are kept verbatim, with their Slurm launchers and sha256, in
`results/dna/original_scripts/` (this corrects the earlier note in
`docs/REPRODUCIBILITY_PLAN.md` that the left panel had no producer):

* `benchmark_yeast_split_value_mc.py` (left panel) settles its protocol -- see
  "Panel A settings" below. Its reward is **not** the cyclizability guide but a synthetic
  `SmoothDNAReward`, and its reference is teacher SDE sampling, not GLASS.
* `benchmark_yeast_split_dmfm_steering.py` (right panel) ran only the dMFM half. Its four
  baseline curves are a hard-coded CSV inside it (lines 28-48; byte-identical to
  `results/dna/benchmarks/fig4_right_hardcoded_baselines.csv`). **No code that produced
  those numbers has ever existed in the repository**, and their `mae_sem` is 0.000000 in
  every row although `n_repeats` is 3.

Authors' decision 3 (2026-09-29): rerun both panels on the current parent-disjoint models,
with real implementations of the baselines. The numbers here are that rerun: panel A is a
faithful port of the original producer, panel B keeps the original's dMFM configuration and
replaces the hard-coded baselines with implementations of best-of-N, Feynman-Kac steering,
beam search and MCTS following Didi et al. (2026). **They are not comparable to the
submitted figure**: different base model, different dMFM, different C0 guide, and baselines
that are now actually measured.

## What produced these files

| file | producer |
|---|---|
| `panelA_value_error.csv`, `panelA_per_condition.csv.gz` | `python -m dmfm.benchmarks.value_error` |
| `panelA_value_error_c0_reward.csv` | the same with `--reward c0 --reference glass` (robustness variant) |
| `panelB_curves.csv`, `panelB_runs.csv` | `python -m dmfm.benchmarks.run_steering --plan sweep` |
| `panelB_pilot.csv`, `pilot_choice.json` | the same with `--plan pilot`: how each baseline's hyperparameters were chosen |
| `calibration.json` | `--mode calibrate`: unguided C0 distribution vs the sampler stochasticity level `eta` |
| `summary.json` | `python -m dmfm.benchmarks.collect` |
| `paper_sentences.md` | hand-written: the canonical Figure 4 caption and Sec. 3.3 sentences, with the placeholders each collected file fills. Source of truth for the wording the tex agent applies. |
| all of the above, on a GPU | `sbatch --array=0-2 scripts/dna/fig4_all.sbatch` (work-stealing workers) |
| figure | `notebooks/06_dna_benchmarks.ipynb` (`RERUN=False` rebuilds it from these files) |

## Which device produced which numbers

Two shard trees, deliberately separate:

| tree | hardware | arithmetic / RNG | submitted by |
|---|---|---|---|
| `outputs/dna/fig4/` | GPU: `kempner_requeue` (H100, account `kempner_albergo_lab`) **and** `gpu_requeue` (mixed A100/H100/H200/RTX Pro, account `albergo_lab` — `kempner_albergo_lab` is not a valid association there) | TF32 matmuls, CUDA RNG | `scripts/dna/fig4_all.sbatch` |
| `outputs/dna/fig4_cpu/` | CPU, `serial_requeue`, account `albergo_lab` | true fp32, CPU RNG | `scripts/dna/fig4_cpu.sbatch` |

The two GPU routes share one tree on purpose: they are the same arithmetic (CUDA, TF32, the
same seeded Philox stream), so skip-if-exists and the locks let whichever queue frees up
first do the work. Each shard records `device_name`, so the exact GPU model behind any
number is recoverable; `gpu_requeue` is a mixed pool while `kempner_requeue` is H100 only.

**A plotted panel is produced entirely on one device type, never mixed.** CUDA and CPU do
not draw the same samples and do not round identically, and there is no A/B run showing
that difference sits below the reported error bars, so mixing would introduce a confound
that could not be ruled out. `dmfm.benchmarks.collect` warns if a shard directory contains
more than one device type; every shard records `device_type`, `tf32_matmul`,
`slurm_account` and `slurm_partition` in its own metadata.

The **pilot** is the one part that may run on either: it only *ranks* baseline
hyperparameter families and no pilot shard is ever plotted. `plan.pilot_scores` still takes
it from a single tree (the first complete one), so the ranking itself is never half CPU and
half GPU, and `plan.frozen_from_pilot` pins the winners to
`outputs/dna/fig4/frozen_hyperparameters.json` the first time a complete pilot is seen, so a
running sweep cannot switch configurations if a second pilot tree finishes later.

**So expect a GPU sweep whose baseline hyperparameters were ranked by a CPU pilot** — the
CPU queue was the one that freed up first. That is deliberate, not an oversight. A pilot
chooses *which* configuration of each baseline to plot; it contributes no plotted number. The
chosen configuration is then measured on the same device as every other point in the panel,
so the comparison the figure makes is unaffected. `pilot_choice.json` records the tree the
ranking came from and the full grid, so the choice can be re-derived on either device.

Models (all `checkpoints/dna/`, group `dna` of `checkpoints/MANIFEST.json`):
base `base/L50/best.pt`, dMFM `dmfm/L50/epoch=95-step=49000.ckpt`,
guide `c0/guide/best_state.pt`, oracle `c0/oracle/best_state.pt`.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    if not summary:
        (OUT / "PROVENANCE.md").write_text(
            HEADER
            + "\n## Status\n\n**The rerun has not been collected yet.** Launch it with\n"
            "`sbatch --array=0-2 scripts/dna/fig4_all.sbatch` (work-stealing workers; one\n"
            "alone does everything in 1-2 GPU-hours, extra tasks just split it and a\n"
            "preempted task resumes), then `bash scripts/dna/fig4_collect.sh`, which\n"
            "aggregates `outputs/dna/fig4/` into this directory, rewrites this file with the\n"
            "job IDs taken from the shards themselves, and re-executes\n"
            "`notebooks/06_dna_benchmarks.ipynb`.\n"
        )
        print(f"wrote {OUT / 'PROVENANCE.md'} (no results yet)")
        return
    lines = [HEADER, "\n## Slurm jobs\n"]
    log = REPO / "outputs" / "dna" / "fig4_jobs.txt"
    if log.exists():
        lines.append("Submissions (from `outputs/dna/fig4_jobs.txt`):\n\n")
        lines += [f"    {ln}\n" for ln in log.read_text().splitlines() if ln.strip()]
        lines.append("\nJob IDs recorded inside the collected shards:\n\n")
    for panel in ("panelA", "panelB", "pilot"):
        block = summary.get(panel)
        if not block:
            lines.append(f"- **{panel}**: not run yet.\n")
            continue
        ids = ", ".join(str(j) for j in block.get("slurm_job_ids", []) if j) or "(run outside Slurm)"
        lines.append(f"- **{panel}**: {block.get('n_shards', 0)} shards, job IDs {ids}\n")
    if "panelA" in summary:
        a = summary["panelA"]["args"]
        f = summary["panelA"]["reference_floor"]
        ref = (
            f"teacher SDE, {a['sde_ref_n']} samples x {a['sde_steps']} Euler-Maruyama steps"
            if a.get("reference", "sde") == "sde"
            else f"GLASS-{a['glass_ref_n']}, {a['glass_nfe']} Euler steps"
        )
        reward = (
            f"synthetic SmoothDNAReward (seed {a['reward_seed']}, gain {a['reward_gain']}) -- the "
            "original producer's reward, NOT the cyclizability guide"
            if a.get("reward", "smooth") == "smooth"
            else f"cyclizability, r(x) = -0.5 ((f_cyc(x) - {a['target_c0']}) / {a['reward_sigma']})^2"
        )
        lines.append(
            "\n## Panel A settings\n\n"
            "All values are the defaults of the recovered producer "
            "`results/dna/original_scripts/benchmark_yeast_split_value_mc.py`.\n\n"
            f"- reward: {reward}\n"
            f"- reference V_t: {ref}. It is itself a Monte Carlo estimate; the `sde_mc` series "
            "measures its own finite-N error and the published panel omits that series.\n"
            f"- conditioning states: {f['n_conditions']} held-out `{a['data_split']}` sequences noised "
            f"to t ~ U({a['tcond_low']}, {a['tcond_high']}). The original used 8; that number is the "
            "shared default of the script and its launcher, not a recorded fact, and the extra "
            "conditions here only add error bars (conditions 0-7 are the original set).\n"
            f"- dMFM at NFE = {a['dmfm_nfe']}; FMAP = teacher ODE from t to 1 in {a['fmap_nfe']} steps; "
            "DPS = the student's own denoiser at the conditioning state. DPS and FMAP are computed "
            "once per condition, so they are exactly flat in N.\n"
            f"- N = {a['n_list'][0]}..{a['n_list'][-1]}; the script's default stops at 256, but the "
            "launcher takes N_LIST from the environment and the published panel has exactly these "
            "points. Each budget is an independently seeded draw, not a nested prefix.\n"
            f"- reference's own MC spread (half vs half of its samples): {f['selfsplit_abs_diff_mean']:.4f} "
            f"nats; mean V_ref {f['V_ref_mean']:.3f} (sd {f['V_ref_sd']:.3f} across conditions)\n"
        )
    if "panelB" in summary:
        b = summary["panelB"]
        lines.append(
            "\n## Panel B settings\n\n"
            f"- {b['n_steps']} base-flow steps, target C0 = {b['target_c0']}, metric "
            "|f_cyc(hard sequence) - y*| under the *guide*; the independent oracle "
            "(reverse-complement averaged) is in the `oracle_*` columns\n"
            "- NFE is measured, not predicted, and counts one forward pass per sampler step; "
            "no parallel execution is assumed. `nfe_backward_per_output` is the extra "
            "backward passes the dMFM needs and the baselines do not\n"
        )
        for row in b.get("matched_nfe_comparison", []):
            others = ", ".join(
                f"{k.replace('_mae_at_same_nfe', '')} {v:.3f}"
                for k, v in row.items()
                if k.endswith("_mae_at_same_nfe") and v == v
            )
            lines.append(
                f"- at {row['nfe']:.0f} NFE (dMFM {row['dmfm_config']}): dMFM {row['dmfm_mae']:.3f} vs {others}\n"
            )
    (OUT / "PROVENANCE.md").write_text("".join(lines))
    print(f"wrote {OUT / 'PROVENANCE.md'}")


if __name__ == "__main__":
    main()
