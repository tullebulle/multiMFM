"""Endpoint-predicting Meta Flow Map model for the toy multimodal state.

The base `model.py` architecture predicts endpoint quantities:
    y_pred:   endpoint prediction for the continuous part
    z_logits: endpoint logits for the discrete/color part

This MFM variant keeps that endpoint parameterization, but conditions the
endpoint predictor on:
    s:      source/current time of the map
    u:      target time of the map
    t_cond: observation/conditioning time
    x_cond: observed conditioning state, concat(c_cond, y_cond)

Velocity and maps are derived from the endpoint prediction:
    v_su(x) = (x1_hat(s, u, x, t_cond, x_cond) - x) / (1 - s)
    X_su(x) = x + (u - s) * v_su(x)

The off-diagonal time dependence is represented through a centered embedding
of du = u - s. Therefore when u == s, the du pathway is exactly zero and the
model lies on the diagonal used for GLASS velocity distillation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for scalar times."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def centered_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding with exact zero output at t == 0."""
    emb = timestep_embedding(t, dim)
    zero = timestep_embedding(torch.zeros_like(t), dim)
    return emb - zero


class TimeMLP(nn.Module):
    """Standard time embedding MLP used for the diagonal/source time."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(timestep_embedding(t, self.net[0].in_features))


class ZeroAtZeroTimeMLP(nn.Module):
    """Time MLP whose output is structurally zero when the scalar input is 0."""

    def __init__(self, dim: int, zero_init_last: bool = True):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )
        if zero_init_last:
            nn.init.zeros_(self.net[2].weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(centered_timestep_embedding(t, self.dim))


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


class MultimodalMFM(nn.Module):
    """Endpoint-predicting MFM for x = concat(c, y)."""

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
        self.c_embed = nn.Linear(num_classes, d_model)
        self.y_cond_embed = nn.Linear(y_dim, d_model)
        self.c_cond_embed = nn.Linear(num_classes, d_model)
        self.mod_embed = nn.Parameter(torch.zeros(2, d_model))

        # Diagonal/source time is the old flow time. Off-diagonal dependence
        # enters only through du = u - s, which is zero on the diagonal.
        self.s_proj = TimeMLP(d_model)
        self.du_proj = ZeroAtZeroTimeMLP(d_model, zero_init_last=True)
        self.t_cond_proj = ZeroAtZeroTimeMLP(d_model, zero_init_last=True)

        # Gate x_cond off at initialization via t_cond_emb == 0. Keep this
        # matrix trainable at random init so gradients can open the path for
        # t_cond > 0; t_cond == 0 remains structurally zero because the
        # t_cond embedding is centered and bias-free.
        self.x_cond_gate = nn.Linear(d_model, 2 * d_model, bias=False)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, mlp_ratio, d_model) for _ in range(n_layers)]
        )

        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada_out = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)

        self.y_head = nn.Linear(d_model, y_dim)
        self.c_head = nn.Linear(d_model, num_classes)

    def split_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return c, y from the joint state x = concat(c, y)."""
        return x[:, : self.num_classes], x[:, self.num_classes :]

    def conditioning_embedding(
        self, s: torch.Tensor, u: torch.Tensor, t_cond: torch.Tensor
    ) -> torch.Tensor:
        """Return adaLN conditioning with exact diagonal-safe du contribution."""
        return self.s_proj(s) + self.du_proj(u - s) + self.t_cond_proj(t_cond)

    def forward(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        t_cond: torch.Tensor | None = None,
        x_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict endpoint quantities conditioned on a map from s to u.

        s, u:    (B,) scalar source/target times.
        x:       (B, num_classes + y_dim), current joint state at time s.
        t_cond:  (B,) observation time. Defaults to zero.
        x_cond:  (B, num_classes + y_dim) observation state. Defaults to zero.

        Returns:
            y_pred, z_logits for the predicted endpoint x_1.
        """
        B = x.shape[0]
        if t_cond is None:
            t_cond = torch.zeros(B, device=x.device, dtype=x.dtype)
        if x_cond is None:
            x_cond = torch.zeros_like(x)

        c, y = self.split_x(x)
        c_cond, y_cond = self.split_x(x_cond)

        t_cond_emb = self.t_cond_proj(t_cond)
        cond = self.s_proj(s) + self.du_proj(u - s) + t_cond_emb
        y_gate, c_gate = self.x_cond_gate(t_cond_emb).chunk(2, dim=-1)

        y_tok = self.y_embed(y) + y_gate * self.y_cond_embed(y_cond) + self.mod_embed[0]
        c_tok = self.c_embed(c) + c_gate * self.c_cond_embed(c_cond) + self.mod_embed[1]
        h = torch.stack([y_tok, c_tok], dim=1)

        for block in self.blocks:
            h = block(h, cond)

        scale, shift = self.ada_out(cond).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        y_pred = self.y_head(h[:, 0])
        z_logits = self.c_head(h[:, 1])
        return y_pred, z_logits

    def endpoint(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        t_cond: torch.Tensor | None = None,
        x_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the continuous endpoint estimate concat(softmax(logits), y)."""
        y_pred, z_logits = self.forward(s, u, x, t_cond=t_cond, x_cond=x_cond)
        c_pred = F.softmax(z_logits, dim=-1)
        return torch.cat([c_pred, y_pred], dim=-1)

    def v(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        t_cond: torch.Tensor | None = None,
        x_cond: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Derived velocity from endpoint prediction."""
        x1_hat = self.endpoint(s, u, x, t_cond=t_cond, x_cond=x_cond)
        denom = (1.0 - s).clamp_min(eps).unsqueeze(-1)
        return (x1_hat - x) / denom

    def X(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        t_cond: torch.Tensor | None = None,
        x_cond: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """MFM map from time s to time u: X_su = x + (u - s) v_su."""
        v = self.v(s, u, x, t_cond=t_cond, x_cond=x_cond, eps=eps)
        return x + (u - s).unsqueeze(-1) * v

    def init_from_base(self, base_model: nn.Module) -> "MultimodalMFM":
        """Copy matching endpoint-predictor weights from `model.MultimodalDiT`."""
        self.y_embed.load_state_dict(base_model.y_embed.state_dict())
        self.c_embed.load_state_dict(base_model.c_embed.state_dict())
        self.mod_embed.data.copy_(base_model.mod_embed.data)
        self.s_proj.net.load_state_dict(base_model.t_proj.state_dict())
        self.blocks.load_state_dict(base_model.blocks.state_dict())
        self.norm_out.load_state_dict(base_model.norm_out.state_dict())
        self.ada_out.load_state_dict(base_model.ada_out.state_dict())
        self.y_head.load_state_dict(base_model.y_head.state_dict())
        self.c_head.load_state_dict(base_model.c_head.state_dict())
        return self


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 5
    model = MultimodalMFM()
    s = 0.8 * torch.rand(B)
    u = s.clone()
    t_cond = torch.rand(B)
    x = torch.randn(B, model.num_classes + model.y_dim)
    x_cond = torch.randn_like(x)

    y_pred, z_logits = model(s, u, x, t_cond, x_cond)
    # --- Lines below were not in the recovered transcript (it stopped at line
    # 280); reconstructed smoke test, see docs/provenance/toy/README.md. ---
    print("y_pred:", tuple(y_pred.shape), "z_logits:", tuple(z_logits.shape))
    v_diag = model.v(s, s, x, t_cond, x_cond)
    x_su = model.X(s, s + 0.1, x, t_cond, x_cond)
    print("v:", tuple(v_diag.shape), "X:", tuple(x_su.shape))
    # du pathway is exactly zero on the diagonal, and x_cond is gated off at t_cond == 0.
    assert torch.equal(model.du_proj(torch.zeros(B)), torch.zeros(B, model.d_model))
    assert torch.equal(model.t_cond_proj(torch.zeros(B)), torch.zeros(B, model.d_model))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
