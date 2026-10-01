#!/usr/bin/env python
"""Make the rebuilt ``evo2`` environment importable without FlashAttention.

Run once after ``micromamba create -f scripts/dna/regen/evo2_environment.yml``, with that
environment's interpreter::

    ~/micromamba/envs/evo2/bin/python scripts/dna/regen/evo2_postinstall.py

Why this is needed (2026-09-29). Evo 2 7B runs without FlashAttention -- ``vtx``'s own README
says so ("Installing without Transformer Engine (Evo 2 7b only) ... work without TE in pure
bfloat16"), ``vortex/model/attention.py`` already wraps ``from vortex.ops import ...`` in
``try/except ImportError`` and has a non-flash code path, and the recovered record of the
paper's own Evo 2 runs says they used neither Transformer Engine nor flash-attn
(``scripts/dna/regen/evo2_environment.yml``). But ``vtx`` 1.0.8 cannot actually be imported
that way:

1. ``vortex/ops/__init__.py`` imports ``attn_interface``, which does a bare
   ``import flash_attn_2_cuda`` at module level. Anything that touches ``vortex.ops`` --
   including ``vortex/model/rotary.py``, which is *not* guarded -- therefore fails before
   ``attention.py``'s fallback can catch it. Patch 1 wraps that import and leaves the five
   ``local_flash_attn_*`` names as ``None``, which is exactly what ``attention.py``'s own
   except-branch does.
2. ``configs/evo2-7b-8k.yml`` (the config of ``evo2_7b_base``) sets ``use_flash_attn: True``,
   which sends the attention layers down the flash path. Patch 2 sets it to ``False``.

Neither FlashAttention wheel route works on this cluster: the published
``flash_attn ... cp312 ... torch2.7`` wheels and the ``flash_attn_2_cuda`` binaries bundled in
``vortex/ops/depr_attn/`` are built against GLIBC 2.32, and the nodes run GLIBC 2.28
(Rocky 8). Building from source needs ``nvcc`` and hours on a GPU node.

Numerics: FlashAttention is exact attention, so this changes only the floating-point reduction
order inside the five attention layers (indices 3, 10, 17, 24, 31 of Evo 2 7B), not the
algorithm.

The script is idempotent, prints what it changed, and leaves a marker file
``vortex/.multimfm_no_flash_attn`` in site-packages.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

OPS_INIT_PATCHED = '''"""Patched by scripts/dna/regen/evo2_postinstall.py (MultiMFM): FlashAttention is optional.

vortex/model/attention.py already treats these names as optional (``except ImportError`` ->
None), but the original unguarded import here broke every other consumer of ``vortex.ops``
(e.g. vortex/model/rotary.py). Behaviour with FlashAttention installed is unchanged.
"""

try:
    from .attn_interface import (
        local_flash_attn_func,
        local_flash_attn_kvpacked_func,
        local_flash_attn_qkvpacked_func,
        local_flash_attn_varlen_kvpacked_func,
        local_flash_attn_varlen_qkvpacked_func,
        local_flash_attn_with_kvcache,
    )
except ImportError:  # no flash_attn_2_cuda in this environment
    local_flash_attn_func = None
    local_flash_attn_kvpacked_func = None
    local_flash_attn_qkvpacked_func = None
    local_flash_attn_varlen_kvpacked_func = None
    local_flash_attn_varlen_qkvpacked_func = None
    local_flash_attn_with_kvcache = None
'''

MARKER = "Patched by scripts/dna/regen/evo2_postinstall.py"


def site_packages() -> Path:
    import sysconfig

    return Path(sysconfig.get_paths()["purelib"])


def patch_ops_init(sp: Path) -> str:
    path = sp / "vortex" / "ops" / "__init__.py"
    if not path.exists():
        return f"SKIP  {path} (vtx not installed?)"
    text = path.read_text()
    if MARKER in text:
        return f"ok    {path} (already patched)"
    backup = path.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(OPS_INIT_PATCHED)
    return f"PATCH {path} (original kept as {backup.name})"


def patch_config(sp: Path, config: str = "evo2-7b-8k.yml") -> str:
    path = sp / "evo2" / "configs" / config
    if not path.exists():
        return f"SKIP  {path} (evo2 not installed?)"
    lines = path.read_text().splitlines(keepends=True)
    out, changed = [], False
    for line in lines:
        if line.startswith("use_flash_attn:") and "False" not in line:
            out.append("use_flash_attn: False  # " + MARKER + "\n")
            changed = True
        else:
            out.append(line)
    if not changed:
        return f"ok    {path} (use_flash_attn already False)"
    backup = path.with_suffix(".yml.orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text("".join(out))
    return f"PATCH {path} use_flash_attn -> False (original kept as {backup.name})"


def main() -> int:
    sp = site_packages()
    print(f"site-packages: {sp}")
    try:
        import flash_attn_2_cuda  # noqa: F401
    except Exception as exc:
        print(f"flash_attn_2_cuda unavailable ({type(exc).__name__}); patching for the no-flash path")
    else:
        print("flash_attn_2_cuda imports fine; patches are still applied and are no-ops for it")
    print(patch_ops_init(sp))
    print(patch_config(sp))
    (sp / "vortex" / ".multimfm_no_flash_attn").write_text(MARKER + "\n")

    import evo2  # noqa: F401
    from evo2 import Evo2  # noqa: F401

    print("import evo2.Evo2: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
