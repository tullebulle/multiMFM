"""Turn a compiled-dependency import failure into a legible message.

The test modules import ``dmfm``, which pulls in scipy and torch. When one of
their compiled extensions cannot load, collection dies inside a third-party
import with something like

    ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found

which says nothing about what to do. This hook makes the session stop with the
error and the things worth checking instead.

Observed once on 2026-09-29 while several agents were installing packages
concurrently (the Evo 2 environment work), and not reproducible afterwards in a
bare shell — so a transient inconsistency in the environment is a real
possibility alongside a genuinely unactivated environment. The dynamic loader
reads ``LD_LIBRARY_PATH`` at process start, so nothing here can repair it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_HINT = """
A compiled dependency of the test suite failed to import:

    {err}

Worth checking, in order:

1. Is the `multimfm` environment active? `scripts/env.sh` also puts the
   environment's own libstdc++ and the torch wheel's bundled CUDA runtime on
   LD_LIBRARY_PATH, which some setups need:

       cd {root}
       source scripts/env.sh
       pytest tests/

2. Is something installing into the environment right now? A concurrent
   pip/conda install can leave shared libraries briefly inconsistent. If so,
   simply rerunning once it finishes is enough.

3. Does the import fail outside pytest too?

       python -c "import scipy.linalg, torch"

   If that also fails, the environment itself needs repair rather than the
   tests.

LD_LIBRARY_PATH is read by the dynamic loader at process start, so it cannot be
set from inside this session.
"""


def pytest_configure(config) -> None:
    """Probe the compiled dependencies before any test module is imported."""
    try:
        import scipy.linalg  # noqa: F401
        import torch  # noqa: F401
    except ImportError as err:
        raise pytest.UsageError(_HINT.format(err=err, root=REPO_ROOT)) from err
