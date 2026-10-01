#!/usr/bin/env python
"""Parse-only stand-in for ``python`` in the DNA run wrappers.

Every wrapper runs ``${PY} -u -m dmfm.experiments.<name> <flags>`` with ``PY`` defaulting to
``python`` (see ``_header.sh``). Setting ``PY`` to this file turns a wrapper into a dry run:
each Python invocation parses its flags with the module's own parser and prints (and, with
``DMFM_DRYRUN_OUT=<file>``, appends as a JSON line) the resulting namespace, without loading a
model or touching a GPU::

    PY="python scripts/dna/_dryrun.py" LENGTH=50 bash scripts/dna/train_dmfm.sbatch

``tests/dna/check_script_args.py`` uses this to compare every wrapper's arguments with the
arguments recorded by the original runs.
"""

from __future__ import annotations

import importlib
import json
import os
import sys


def _split(argv: list[str]) -> tuple[str | None, str | None, list[str]]:
    """Return (module, script, rest) from a python command line."""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-m":
            return argv[i + 1], None, argv[i + 2 :]
        if a == "-c":
            return None, "-c", argv[i + 2 :]
        if a in ("-W", "-X"):
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        return None, a, argv[i + 1 :]
    return None, None, []


def parse(module: str, rest: list[str]):
    if module == "dmfm.experiments.train_dmfm":
        from dmfm.utils.parsing import build_train_parser

        return build_train_parser().parse_args(rest)
    mod = importlib.import_module(module)
    if hasattr(mod, "parse_args"):
        return mod.parse_args(rest)
    return None


def main() -> int:
    module, script, rest = _split(sys.argv[1:])
    record: dict = {"argv": sys.argv[1:]}
    if script == "-c":
        # helper one-liners (e.g. resolving a default checkpoint path) run for real
        os.execv(sys.executable, [sys.executable] + sys.argv[1:])
    if module is None:
        print(f"[dryrun] skip (not a module call): {script}", flush=True)
        return 0
    ns = parse(module, rest)
    record["module"] = module
    if ns is None:
        print(f"[dryrun] {module}: no parse_args(); not checked", flush=True)
        record["parsed"] = None
    else:
        parsed = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)) for k, v in vars(ns).items()}
        record["parsed"] = parsed
        print(f"[dryrun] {module}: parsed {len(parsed)} arguments OK", flush=True)
    out = os.environ.get("DMFM_DRYRUN_OUT")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
