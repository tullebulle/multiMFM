#!/usr/bin/env python
"""Derive src/dmfm/experiments/score_yeast_c0_evo2.py from the TRUE upstream original.

Run it to re-check the derivation (it is idempotent -- it rewrites the same bytes):

    python scripts/dna/regen/derive_evo2_scorer.py

It re-reads the upstream file from the holylabs clone, verifies its sha256, replays the edit
list, and asserts that (a) every anchor matched exactly once, (b) the 13 scoring-path functions
stay AST-identical to the original and (c) only the five packaging functions changed.

Starts from the verbatim upstream file (sha256 059498df...fafbe9) and applies a fixed,
asserted list of surgical edits. Every edit is anchored on an exact substring that must
occur exactly once, so the derivation is mechanical and auditable.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

REPO = Path("/n/netscratch/albergo_lab/Lab/uunneberg/MultiMFM")
ORIG = Path("/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929"
            "/DNA-MFM_187fe7b/scripts/score_yeast_c0_evo2.py")
DEST = REPO / "src/dmfm/experiments/score_yeast_c0_evo2.py"
ORIG_SHA = "059498df13c607cee8736fabb41f64d1b6ddfecab3f1d758c761f76787fafbe9"

text = ORIG.read_text()
assert hashlib.sha256(text.encode()).hexdigest() == ORIG_SHA, "original sha256 mismatch"
before = DEST.read_text()

DOC = '''"""Evo2 plausibility scoring of C0-steered yeast samples and real-data baselines (Table 6).

DERIVED FROM THE ORIGINAL SOURCE, not from a reconstruction. The upstream file is

    github.com/tullebulle/DNA-MFM@187fe7b  scripts/score_yeast_c0_evo2.py
    13,706 bytes, sha256 059498df13c607cee8736fabb41f64d1b6ddfecab3f1d758c761f76787fafbe9

read from the durable clone on holylabs
(``/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/DNA-MFM_187fe7b``,
a read-only source) and kept verbatim at ``results/dna/regen/original/score_yeast_c0_evo2.py``
with its hash in that directory's ``_sources.json``. This module is that file with the edits
listed below and nothing else; the derivation is scripted and re-checkable.

History: in ``dirichlet-flow-matching`` only the bytecode
(``scripts/__pycache__/score_yeast_c0_evo2.cpython-39.pyc``) survived the purge, so this module
was first rebuilt from the Codex session that wrote it and checked against that bytecode (all 22
code objects agreed; disassembly in ``docs/provenance/dna/score_yeast_c0_evo2.dis.txt``). The
upstream source was found later. Re-deriving from it changed **no** executable statement: the
reconstruction and this file are identical under ``ast.dump`` with docstrings stripped.

THE SCORING PATH IS UNTOUCHED. Ignoring docstrings and comments, these functions are
AST-identical to the original -- tokenisation, padding, the optional BOS, windowing, strand
handling, the reduction and everything that is averaged:

    prepare_batch, score_mean_logprobs, score_rows, bootstrap_ci, summarize,
    reverse_complement, tokens_to_seqs, model_forward, load_evo2, generated_rows,
    cap_df, run_metadata, load_args_payload

In particular, per sequence Evo2 scores ``mean_j log p(x_j | x_<j)`` over the L-1 predictable
positions (L-1+1 with ``--prepend_bos``); with ``--reverse_complement`` (the default) the forward
and reverse-complement means are averaged with weight 0.5 each; NLL = -mean_logprob; the CI is a
2,000-draw nonparametric bootstrap of the per-sequence NLL; ``evo2_perplexity_from_mean_nll`` is
``exp(mean NLL)``; and ``evo2_nll_delta_vs_split_test`` is measured against ``split_test_sample``.

EXHAUSTIVE LIST OF DEVIATIONS (imports, paths and CLI wiring only; five functions touched):

1. ``parse_args`` gains an ``argv`` parameter and passes it to ``parse_args(argv)``, so the
   module is callable in-process (notebooks, tests) as well as from the command line.
2. ``parse_args`` defaults become repo-relative via :mod:`dmfm.paths`. The originals were
   cwd-relative paths into the purged split65k tree and are kept in comments beside each:
   ``--train_pt data/yeast_mid50.pt``,
   ``--split_pt data/yeast_mid50_split_seed0_train65000_val8000.pt``,
   ``--out_dir workdir/yeast_split_dmfm_c0_evo2``.
3. ``parse_args`` gains ``--train_split`` / ``--val_split`` / ``--test_split``. The original
   hard-coded the split keys ``train`` / ``val`` / ``test`` inside ``baseline_rows``; those keys
   exist only in the purged split65k split file. The defaults here are the parent-disjoint
   equivalents ``a_train`` / ``a_val`` / ``test``. **Passing ``--train_split train --val_split val
   --test_split test`` restores the original behaviour exactly.**
4. ``baseline_rows`` gains a ``split_names`` parameter carrying those keys. Its default is the
   ORIGINAL triple ``("train", "val", "test")``, so calling it positionally is unchanged.
5. ``discover_csvs``: its default glob pointed at the purged split65k sample directories
   (``workdir/yeast_split_dmfm_c0_samples_target_*/sample_scores.csv``) and now points at the
   shipped parent-disjoint n=1,000 C0 sets (``DEFAULT_RUN_GLOB``). Its local variable ``paths``
   is renamed ``paths_found`` because ``dmfm.paths`` is now imported under that name.
6. ``load_train_tokens``: ``torch.load`` -> ``dmfm.utils.torch_io.torch_load``
   (``weights_only=False``). The original ran under torch 2.7, where that was the default; the
   payloads are plain tensors or a dict of tensors, so nothing about the data changes.
7. ``main`` gains the same ``argv`` parameter, passes ``split_names`` through, and names the
   default glob in its "no files matched" error.
8. Imports: ``from utils.yeast_splits import ...`` -> ``dmfm.utils.yeast_splits``; plus
   ``dmfm.paths`` and ``dmfm.utils.torch_io``.

None of the above touches a number Table 6 reports. What DOES change Table 6 is the input:
the paper's split65k sample sets, split file and models are all purged, so the shipped defaults
score the parent-disjoint n=1,000 C0 sets instead. See ``results/dna/regen/PROVENANCE.md``.

Provenance of the published Table 6: run ``workdir/yeast_split_dmfm_c0_evo2_2026-06-10``
(outputs purged; its summary CSV survives in ``results/dna/june_split65k_recovered/``) with

    PYTHONPATH=. .envs/evo2/bin/python scripts/score_yeast_c0_evo2.py \\
      --run_glob 'workdir/yeast_split_dmfm_c0_samples_target_p1_2026-06-02/sample_scores.csv' \\
      --run_glob 'workdir/yeast_split_dmfm_c0_samples_target_m1_2026-06-02/sample_scores.csv' \\
      --train_pt data/yeast_mid50.pt --split_pt data/yeast_mid50_split_seed0_train65000_val8000.pt \\
      --out_dir workdir/yeast_split_dmfm_c0_evo2_2026-06-10 --targets -1 1 --baseline_n 1000 \\
      --batch_size 1 --bootstrap 2000 --model_name evo2_7b_base --device cuda:0

Run it in the separate Evo2 environment (``scripts/dna/regen/evo2_environment.yml`` plus
``evo2_postinstall.py``); ``evo2`` is imported lazily in :func:`load_evo2`, so this module
imports fine without it.
"""
'''

EDITS: list[tuple[str, str]] = [
    # 1. module docstring, right after the shebang
    ("#!/usr/bin/env python\nfrom __future__ import annotations\n",
     "#!/usr/bin/env python\n" + DOC + "from __future__ import annotations\n"),

    # 8. imports
    ("from utils.yeast_splits import load_yeast_split_indices\n",
     "from dmfm import paths\n"
     "from dmfm.utils.torch_io import torch_load\n"
     "from dmfm.utils.yeast_splits import load_yeast_split_indices\n"),

    # 5. default run glob constant
    ('RC_TRANS = str.maketrans("ACGTacgt", "TGCAtgca")\n',
     'RC_TRANS = str.maketrans("ACGTacgt", "TGCAtgca")\n\n'
     '# Original default: "workdir/yeast_split_dmfm_c0_samples_target_*/sample_scores.csv"\n'
     '# (split65k sample dirs, purged). Port default: the shipped parent-disjoint n=1,000 C0 sets.\n'
     'DEFAULT_RUN_GLOB = str(paths.RESULTS / "c0_guidance" / "n1000" / "target_*" / "sample_scores.csv")\n'),

    # 1. argv parameter
    ("def parse_args() -> argparse.Namespace:\n",
     "def parse_args(argv: list[str] | None = None) -> argparse.Namespace:\n"),

    # 5. run_glob help mentions the new default
    ('        help="Glob(s) for sample_scores.csv. Can be passed multiple times.",\n',
     '        help="Glob(s) for sample_scores.csv. Can be passed multiple times. "\n'
     '        f"Default: {paths.rel(DEFAULT_RUN_GLOB)}",\n'),

    # 2. + 3. repo-relative defaults and the new split-key flags
    ('    p.add_argument("--train_pt", default="data/yeast_mid50.pt")\n'
     '    p.add_argument("--split_pt", default="data/yeast_mid50_split_seed0_train65000_val8000.pt")\n'
     '    p.add_argument("--out_dir", default="workdir/yeast_split_dmfm_c0_evo2")\n',
     '    # Original default: data/yeast_mid50.pt (split65k, purged).\n'
     '    p.add_argument("--train_pt", default=str(paths.data_pt(50)))\n'
     '    # Original default: data/yeast_mid50_split_seed0_train65000_val8000.pt (split65k, purged).\n'
     '    p.add_argument("--split_pt", default=str(paths.split_pt(50)))\n'
     '    # Added in the port; the original hard-coded train/val/test inside baseline_rows.\n'
     '    p.add_argument("--train_split", default="a_train", help="Split key for split_train_sample (original: train).")\n'
     '    p.add_argument("--val_split", default="a_val", help="Split key for split_val_sample (original: val).")\n'
     '    p.add_argument("--test_split", default="test", help="Split key for split_test_sample (original: test).")\n'
     '    # Original default: workdir/yeast_split_dmfm_c0_evo2\n'
     '    p.add_argument("--out_dir", default=str(paths.OUTPUTS / "yeast_parent_c0_evo2"))\n'),

    # 1. parse argv
    ("    return p.parse_args()\n", "    return p.parse_args(argv)\n"),

    # 6. torch_load
    ('    payload = torch.load(path, map_location="cpu")\n',
     '    payload = torch_load(path, map_location="cpu")\n'),

    # 5. discover_csvs: local rename + default glob
    ('    paths: list[str] = []\n'
     '    for pat in patterns or ["workdir/yeast_split_dmfm_c0_samples_target_*/sample_scores.csv"]:\n'
     '        paths.extend(glob.glob(pat))\n'
     '    return sorted(set(paths))\n',
     '    # Local renamed from `paths` because `dmfm.paths` is imported under that name.\n'
     '    paths_found: list[str] = []\n'
     '    for pat in patterns or [DEFAULT_RUN_GLOB]:\n'
     '        paths_found.extend(glob.glob(pat))\n'
     '    return sorted(set(paths_found))\n'),

    # 4. baseline_rows split_names parameter (default = the original keys)
    ('    max_per_set: int | None,\n'
     '    seed: int,\n'
     ') -> pd.DataFrame:\n'
     '    full = load_train_tokens(train_pt)\n'
     '    references = {\n'
     '        "full_sample": full,\n'
     '        "split_train_sample": full[load_yeast_split_indices(split_pt, "train").numpy()],\n'
     '        "split_val_sample": full[load_yeast_split_indices(split_pt, "val").numpy()],\n'
     '        "split_test_sample": full[load_yeast_split_indices(split_pt, "test").numpy()],\n'
     '    }\n',
     '    max_per_set: int | None,\n'
     '    seed: int,\n'
     '    split_names: tuple[str, str, str] = ("train", "val", "test"),\n'
     ') -> pd.DataFrame:\n'
     '    """Real-data reference sets. ``split_names`` (added in the port) defaults to the original keys."""\n'
     '    train_split, val_split, test_split = split_names\n'
     '    full = load_train_tokens(train_pt)\n'
     '    references = {\n'
     '        "full_sample": full,\n'
     '        "split_train_sample": full[load_yeast_split_indices(split_pt, train_split).numpy()],\n'
     '        "split_val_sample": full[load_yeast_split_indices(split_pt, val_split).numpy()],\n'
     '        "split_test_sample": full[load_yeast_split_indices(split_pt, test_split).numpy()],\n'
     '    }\n'),

    # 7. main
    ("def main() -> None:\n    args = parse_args()\n",
     "def main(argv: list[str] | None = None) -> None:\n    args = parse_args(argv)\n"),
    ('        raise FileNotFoundError(f"No sample_scores.csv files matched: {args.run_glob}")\n',
     '        raise FileNotFoundError(\n'
     '            f"No sample_scores.csv files matched: {args.run_glob or [DEFAULT_RUN_GLOB]}"\n'
     '        )\n'),
    ('            baseline_rows(args.train_pt, args.split_pt, args.baseline_n, args.max_per_set, args.seed),\n',
     '            baseline_rows(\n'
     '                args.train_pt, args.split_pt, args.baseline_n, args.max_per_set, args.seed,\n'
     '                split_names=(args.train_split, args.val_split, args.test_split),\n'
     '            ),\n'),
]

for i, (old, new) in enumerate(EDITS, 1):
    n = text.count(old)
    assert n == 1, f"edit {i}: anchor occurs {n} times, expected exactly 1:\n{old[:120]!r}"
    text = text.replace(old, new, 1)

DEST.write_text(text)


# ---- proof that the executable content did not change relative to the previous port -------
class StripDocs(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node
    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip


def norm(s: str) -> str:
    return ast.dump(StripDocs().visit(ast.parse(s)), indent=1)


print("re-derived module is AST-identical to the previous (bytecode-reconstructed) port:",
      norm(text) == norm(before))

orig_fns = {n.name: ast.dump(n, indent=1)
            for n in StripDocs().visit(ast.parse(ORIG.read_text())).body
            if isinstance(n, ast.FunctionDef)}
new_fns = {n.name: ast.dump(n, indent=1)
           for n in StripDocs().visit(ast.parse(text)).body
           if isinstance(n, ast.FunctionDef)}
identical = [k for k in orig_fns if orig_fns[k] == new_fns[k]]
changed = [k for k in orig_fns if orig_fns[k] != new_fns[k]]
print("functions AST-identical to the ORIGINAL:", identical)
print("functions changed (must be exactly the 5 packaging ones):", changed)
assert set(changed) == {"parse_args", "load_train_tokens", "discover_csvs", "baseline_rows", "main"}, changed

print("py_compile:", subprocess.run(
    ["/n/home04/uunneberg/micromamba/envs/multimfm/bin/python", "-m", "py_compile", str(DEST)],
    capture_output=True, text=True).returncode == 0)
print("bytes:", len(text.encode()))
