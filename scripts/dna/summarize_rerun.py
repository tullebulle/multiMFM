#!/usr/bin/env python
"""Summarize the RERUN = True executions of notebooks 02-05 (the GPU verification).

    python scripts/dna/summarize_rerun.py [dir]

`dir` defaults to the newest outputs/dna/notebook_rerun/job_*/executed (where
scripts/dna/notebook_rerun.sbatch writes the executed copies), else outputs/dna/notebook_rerun/executed. For each notebook it prints: whether RERUN was True, any cell that
raised, every "status counts:" line the comparison helpers printed, and the rows the rerun cells
marked MISMATCH / OUTSIDE CI -- i.e. any small-n result that does not agree with the stored
full-size number and would indicate a port bug worth chasing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[2]
_JOBS = sorted((REPO / "outputs" / "dna" / "notebook_rerun").glob("job_*/executed"), key=lambda p: p.stat().st_mtime)
DEFAULT = _JOBS[-1] if _JOBS else REPO / "outputs" / "dna" / "notebook_rerun" / "executed"
COUNTS = re.compile(r"status counts:\s*(\{[^}]*\})")


def text_of(cell) -> str:
    out = []
    for o in cell.get("outputs", []):
        if o.output_type == "stream":
            out.append(o.get("text", ""))
        elif o.output_type in ("execute_result", "display_data"):
            out.append(o.get("data", {}).get("text/plain", ""))
    return "".join(out)


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else DEFAULT
    files = sorted(root.glob("*_RERUN.ipynb"))
    if not files:
        print(f"no executed copies in {root}")
        print("run: sbatch scripts/dna/notebook_rerun.sbatch   ")
        return 1
    bad_total = 0
    for f in files:
        nb = nbformat.read(f, as_version=4)
        first = nb.cells[0].source
        rerun_on = "RERUN = True" in first or "DNA_RERUN" in first
        print("=" * 100)
        print(f"{f.name}  ({len(nb.cells)} cells, first cell sets RERUN: {rerun_on})")
        errs, counts, flagged = [], [], []
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            for o in c.get("outputs", []):
                if o.output_type == "error":
                    errs.append((i, o.ename, (o.evalue or "")[:160]))
            t = text_of(c)
            counts += [(i, m.group(1)) for m in COUNTS.finditer(t)]
            for line in t.splitlines():
                if ("MISMATCH" in line or "OUTSIDE CI" in line) and "status counts" not in line:
                    flagged.append((i, line.strip()[:200]))
        print(f"  cells that raised: {errs if errs else 'none'}")
        for i, cnt in counts:
            print(f"  cell {i:>3}: {cnt}")
        if flagged:
            print("  rows marked MISMATCH / OUTSIDE CI:")
            for i, line in flagged:
                print(f"    cell {i:>3}: {line}")
        bad_total += len(errs)
    print("=" * 100)
    print(f"notebooks: {len(files)}; cells that raised: {bad_total}")
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
