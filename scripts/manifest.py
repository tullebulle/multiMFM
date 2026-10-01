#!/usr/bin/env python
"""Track the large paper artifacts (checkpoints, datasets) that live outside git.

The tracked file ``checkpoints/MANIFEST.json`` lists every artifact with its
repo-relative path, size, sha256, and where it came from. Three commands:

    # record (or refresh) entries for files already placed in the repo
    python scripts/manifest.py add checkpoints/dna/base/L50/best.pt \
        --source /n/holylabs/.../best.pt --group dna --note "Table 9 base, L=50"

    # copy a file into the repo and record it in one go
    python scripts/manifest.py fetch --source /n/holylabs/.../best.pt \
        --dest checkpoints/dna/base/L50/best.pt --group dna

    # check that everything listed is present and matches its sha256
    python scripts/manifest.py verify [--group dna] [--fetch-missing]

``fetch`` and ``--fetch-missing`` copy from the recorded ``source``. Once the
artifacts are hosted remotely, ``source`` can be a URL (http/https), which is
downloaded instead.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "checkpoints" / "MANIFEST.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def locked():
    """Serialize read-modify-write of the manifest across concurrent writers."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"artifacts": {}}


def save(manifest: dict) -> None:
    manifest["artifacts"] = dict(sorted(manifest["artifacts"].items()))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return str(p.relative_to(REPO_ROOT))


def copy_from(source: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        shutil.copy2(source, dest)


def record(manifest: dict, path: str, source: str | None, group: str | None, note: str | None) -> None:
    full = REPO_ROOT / path
    entry = manifest["artifacts"].get(path, {})
    entry.update(size=full.stat().st_size, sha256=sha256(full))
    if source:
        entry["source"] = source
    if group:
        entry["group"] = group
    if note:
        entry["note"] = note
    manifest["artifacts"][path] = entry
    print(f"recorded {path} ({entry['size'] / 1e6:.1f} MB)")


def cmd_add(args: argparse.Namespace) -> None:
    entries = []
    for p in args.paths:
        full = Path(p).resolve()
        files = sorted(f for f in full.rglob("*") if f.is_file()) if full.is_dir() else [full]
        for f in files:
            src = args.source if (args.source and not full.is_dir()) else None
            if args.source and full.is_dir():
                src = str(Path(args.source) / f.relative_to(full))
            entries.append((rel(f), src))
    with locked():
        manifest = load()
        for path, src in entries:
            record(manifest, path, src, args.group, args.note)
        save(manifest)


def cmd_fetch(args: argparse.Namespace) -> None:
    dest = REPO_ROOT / rel(args.dest)
    copy_from(args.source, dest)
    with locked():
        manifest = load()
        record(manifest, rel(dest), args.source, args.group, args.note)
        save(manifest)


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = load()
    bad = 0
    for path, entry in manifest["artifacts"].items():
        if args.group and entry.get("group") != args.group:
            continue
        full = REPO_ROOT / path
        if not full.exists() and args.fetch_missing and entry.get("source"):
            print(f"fetching {path} <- {entry['source']}")
            copy_from(entry["source"], full)
        if not full.exists():
            print(f"MISSING  {path}")
            bad += 1
        elif full.stat().st_size != entry["size"] or sha256(full) != entry["sha256"]:
            print(f"MISMATCH {path}")
            bad += 1
        else:
            print(f"ok       {path}")
    print(f"{'all artifacts verified' if not bad else f'{bad} problem(s)'}")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="record files/directories already in the repo")
    a.add_argument("paths", nargs="+")
    a.add_argument("--source", help="original location (file, or directory when adding a directory)")
    a.add_argument("--group")
    a.add_argument("--note")

    f = sub.add_parser("fetch", help="copy a file into the repo and record it")
    f.add_argument("--source", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--group")
    f.add_argument("--note")

    v = sub.add_parser("verify", help="check presence and sha256 of every artifact")
    v.add_argument("--group")
    v.add_argument("--fetch-missing", action="store_true")

    args = parser.parse_args()
    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "fetch":
        cmd_fetch(args)
    else:
        sys.exit(cmd_verify(args))


if __name__ == "__main__":
    main()
