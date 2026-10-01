"""Toy multimodal data: coloured particles x = (c, y).

RECONSTRUCTED (original ``norway/data.py`` was purged; no copy survives).
The interface is fixed by the recovered code:

* ``train_mfm_diagonal.py`` imports ``NUM_CLASSES, Y_DIM, sample_flag,
  sample_uniform_color_square`` and calls ``sample(batch, generator=g, device=d)``
  expecting ``(c1, y1)`` with integer class labels ``c1`` (fed to ``F.one_hot``).
* the Fig. 1 notebook imports ``COLOR_NAMES, NUM_CLASSES`` and plots classes
  ``0, 1, 2`` with colours ``blue, white, red`` (model order B, W, R, App. F.7.2).

The distribution used for the paper is ``uniform_color_square``:
p1 = U({B, W, R}) x U([-1, 1]^2)  (App. F.7.1).  ``french_flag`` (the original
toy's first dataset, three vertical stripes coloured B | W | R) is kept only
because the recovered trainers list it; it is not used for any paper figure.
"""

from __future__ import annotations

import torch

NUM_CLASSES = 3
Y_DIM = 2
COLOR_NAMES = ("blue", "white", "red")  # model colour order (B, W, R)


def sample_uniform_color_square(
    n: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """c ~ U{0, 1, 2}, y ~ U([-1, 1]^2), independent."""
    c = torch.randint(0, NUM_CLASSES, (n,), generator=generator, device=device)
    y = 2.0 * torch.rand(n, Y_DIM, generator=generator, device=device) - 1.0
    return c, y


def sample_flag(
    n: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """French flag: y ~ U([-1, 1]^2), colour fixed by the vertical stripe of y1."""
    y = 2.0 * torch.rand(n, Y_DIM, generator=generator, device=device) - 1.0
    c = torch.bucketize(y[:, 0], torch.tensor([-1.0 / 3.0, 1.0 / 3.0], device=device))
    return c, y


DATA_SAMPLERS = {
    "french_flag": sample_flag,
    "uniform_color_square": sample_uniform_color_square,
}
