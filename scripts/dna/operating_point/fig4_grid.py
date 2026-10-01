#!/usr/bin/env python
"""Figure 4 (right): the dMFM curve's (steps, MC) split at matched value-gradient NFE.

The plotted dMFM curve is parameterised by MC at a *fixed* one composed flow-map step
(``dmfm.benchmarks.plan._dmfm`` sets ``nfe_value = 1``), so its x-axis, NFE per output, already
grows only through MC.  That is the split the deficit-rescue measurement prefers, so the
question here is the converse of the one Table 1 faced: would any *larger* number of posterior
steps, paid for with fewer MC draws, move the curve?  This driver answers it by running the
same shard producer (``dmfm.benchmarks.run_steering.run_job``) on jobs whose ``nfe_value`` is
not 1, at the same total ``steps x MC``, so each matched-NFE group is one point of the plotted
curve split four or five ways.

``plan.py`` is not touched: the jobs are constructed here and written to a separate out-dir.
Every split in a group shares ``--index``, hence the shard seed, hence the initial noise.

Convention 1 (``docs/REPRODUCIBILITY_PLAN.md`` section 7 item 8) selects the diagonal student
at one composed step and the 4-step ESD student above it; ``run_steering.run_job`` always loads
the diagonal one, so ``load_dmfm_student`` is wrapped here to follow the convention. Pass
``--student diagonal`` to keep the diagonal student at every step count (a control that
separates the split from the student swap).

    python scripts/dna/operating_point/fig4_grid.py --nfe 16 8 --n-outputs 32 \
        --out-dir results/dna/operating_point/fig4_grid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dmfm import paths
from dmfm.benchmarks import core as bench_core, run_steering
from dmfm.benchmarks.plan import Job

_STUDENT_MODE = "convention"
_LENGTH = 50


def _student(length: int = 50, device=None, *, ckpt=None):
    """Convention-1 student for the step count of the job being run."""
    steps = _STEPS_NOW[0]
    if _STUDENT_MODE == "diagonal":
        chosen = paths.dmfm_ckpt(length)
    else:
        chosen = paths.dmfm_ckpt_for_steps(length, steps)
    return bench_core.load_dmfm_student(length, device, ckpt=chosen)


_STEPS_NOW = [1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nfe", type=int, nargs="+", default=[16, 8],
                    help="value-gradient NFE budgets to split (steps x MC)")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-outputs", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", type=int, default=0, help="shard seed offset shared by every split")
    ap.add_argument("--out-dir", default="results/dna/operating_point/fig4_grid")
    ap.add_argument("--student", choices=["convention", "diagonal"], default="convention")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    global _STUDENT_MODE
    _STUDENT_MODE = args.student
    run_steering.load_dmfm_student = _student

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    for nfe in args.nfe:
        for steps in args.steps:
            if nfe % steps or nfe // steps < 1:
                continue
            mc = nfe // steps
            jobs.append(Job("dmfm", f"NFE={nfe},steps={steps},MC={mc}",
                            {"n_mc": mc, "nfe_value": steps}))
    # run_steering.run_job hard-codes ``paths.dmfm_ckpt`` into every shard's
    # meta["checkpoints"]["dmfm"], whatever student was actually loaded, so that field is NOT
    # evidence of which artifact ran. Record the truth next to the shards instead.
    import json
    (out_dir / "student_mode.json").write_text(json.dumps({
        "student_mode": args.student,
        "resolved": {int(s_): str(paths.dmfm_ckpt(50) if args.student == "diagonal"
                                 else paths.dmfm_ckpt_for_steps(50, int(s_)))
                     for s_ in sorted({j.config["nfe_value"] for j in jobs})},
        "_warning": "each shard's meta.checkpoints.dmfm is hard-coded by run_steering and may "
                    "name the diagonal student even when another one was loaded; this file is "
                    "authoritative for this directory.",
    }, indent=2) + "\n")
    print(f"{len(jobs)} splits: {[j.label for j in jobs]} (student={args.student})", flush=True)
    for job in jobs:
        for repeat in range(args.repeats):
            cli = run_steering.parse_args([
                "--plan", "sweep", "--index", str(args.index), "--repeat", str(repeat),
                "--n-outputs", str(args.n_outputs), "--seed", str(args.seed),
                "--out-dir", str(out_dir), "--save-sequences",
                *(["--device", args.device] if args.device else []),
            ])
            path = run_steering.shard_path(out_dir, job, repeat, cli.n_outputs, cli.seed)
            if path.exists():
                print(f"exists, skipping {path.name}", flush=True)
                continue
            _STEPS_NOW[0] = int(job.config["nfe_value"])
            run_steering.run_job(cli, job, repeat, path)


if __name__ == "__main__":
    sys.exit(main())
