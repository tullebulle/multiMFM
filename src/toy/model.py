"""Tiny multimodal DiT for joint (continuous, discrete) flow matching.

The state at time t is x_t = (c_t, y_t) where:
    c_t in R^K              (linear interpolant of N(0, I_K) and onehot(c_1))
    y_t in R^2              (linear interpolant of N(0, I_2) and y_1)

Architecture: each modality becomes a single token, tokens attend to each
other in a few transformer blocks with adaLN-Zero conditioning on t, and
two heads produce:
    y_pred:   endpoint prediction for the continuous part        (B, 2)
    z_logits: logits whose softmax denoises the discrete part    (B, 3)
"""

import math
import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero modulation from a conditioning vector."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # 6 modulation params: (scale, shift, gate) for each of attn / mlp.
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1)
        h = self.mlp(h)
        x = x + g2.unsqueeze(1) * h
        return x


class MultimodalDiT(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        y_dim: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.y_dim = y_dim
        self.d_model = d_model

        self.y_embed = nn.Linear(y_dim, d_model)
        # c_t is a continuous K-vector (linear interpolant in one-hot space).
        self.c_embed = nn.Linear(num_classes, d_model)
        # One learned modality embedding per token slot.
        self.mod_embed = nn.Parameter(torch.zeros(2, d_model))

        self.t_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, mlp_ratio, d_model) for _ in range(n_layers)]
        )

        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada_out = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)

        self.y_head = nn.Linear(d_model, y_dim)
        self.c_head = nn.Linear(d_model, num_classes)

    def forward(self, t: torch.Tensor, c_t: torch.Tensor, y_t: torch.Tensor):
        """
        t:   (B,) float in [0, 1]
        c_t: (B, num_classes) float — the linear interpolant in one-hot space
        y_t: (B, y_dim) float
        """
        t_emb = self.t_proj(timestep_embedding(t, self.d_model))

        y_tok = self.y_embed(y_t) + self.mod_embed[0]
        c_tok = self.c_embed(c_t) + self.mod_embed[1]
        x = torch.stack([y_tok, c_tok], dim=1)  # (B, 2, d_model)

        for block in self.blocks:
            x = block(x, t_emb)

        s, b = self.ada_out(t_emb).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + s.unsqueeze(1)) + b.unsqueeze(1)

        y_pred = self.y_head(x[:, 0])
        z_logits = self.c_head(x[:, 1])
        return y_pred, z_logits


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 5
    model = MultimodalDiT()
    t = torch.rand(B)
    c = torch.randn(B, 3)
    y = torch.randn(B, 2)
    y_pred, z_logits = model(t, c, y)
    print("y_pred:", tuple(y_pred.shape), "z_logits:", tuple(z_logits.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
