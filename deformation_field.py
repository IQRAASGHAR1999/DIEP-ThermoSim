"""
deformation_field.py — Thermal deformation field network.

This module is a direct methodological port of the deformation field network
used in DynGS-Pro (Asghar et al., KSEM 2026), adapted from 3D scene
dynamics to thermal field dynamics.

In DynGS-Pro, the deformation field is an MLP that takes a canonical
3D position and a time embedding and returns a per-point displacement,
allowing the static Gaussian primitives to model non-rigid scene motion.

Here, the deformation field is an MLP that takes a 2D image position and a
time embedding and returns a thermal residual, allowing the Pennes
forward model to be corrected for patient-specific or instrument-specific
effects that the bulk PDE cannot capture (e.g. local skin texture, sweat,
non-uniform cooling, fine-scale perfusion heterogeneity).

The hybrid pattern is the same: an explicit physical/geometric model plus
a learned correction. This file makes that lineage explicit in code, not
just in prose.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding.

    Maps (x, y, t) in [-1, 1]^3 to a higher-dimensional feature so the MLP
    can fit high-frequency variation. Same idea as NeRF and as the
    positional encoding used in the DynGS-Pro deformation field.
    """

    def __init__(self, n_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        freq_bands = 2.0 ** torch.linspace(0.0, n_freqs - 1, n_freqs)
        self.register_buffer("freq_bands", freq_bands)

    @property
    def output_dim_per_input(self) -> int:
        # for each input dim: optionally include itself, then sin+cos at n_freqs
        return (1 if self.include_input else 0) + 2 * self.n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D] -> [..., D * (1 + 2 * n_freqs)] if include_input else [..., D * 2 * n_freqs]."""
        out = [x] if self.include_input else []
        for f in self.freq_bands:
            out.append(torch.sin(x * f * math.pi))
            out.append(torch.cos(x * f * math.pi))
        return torch.cat(out, dim=-1)


class ThermalDeformationField(nn.Module):
    """An MLP that predicts a scalar thermal residual at (x, y, t).

    Architecture mirrors DynGS-Pro's deformation field network: a small MLP
    operating on positionally-encoded (x, y, t) inputs, with a skip
    connection at the middle layer.

    Inputs are normalised pixel coordinates and a normalised time index;
    the output is a per-pixel temperature residual in Kelvin (typically
    small, e.g. ±0.5 K).
    """

    def __init__(
        self,
        n_freqs_xy: int = 6,
        n_freqs_t: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 4,
        skip_at: int = 2,
        max_residual_K: float = 1.0,
    ):
        super().__init__()
        self.pe_xy = PositionalEncoding(n_freqs=n_freqs_xy, include_input=True)
        self.pe_t = PositionalEncoding(n_freqs=n_freqs_t, include_input=True)
        # input dim: 2 (xy) * pe_xy_dim_per_input + 1 (t) * pe_t_dim_per_input
        in_dim = 2 * self.pe_xy.output_dim_per_input + 1 * self.pe_t.output_dim_per_input
        self.skip_at = skip_at
        self.max_residual_K = max_residual_K

        self.layers = nn.ModuleList()
        cur = in_dim
        for i in range(n_layers):
            if i == skip_at:
                cur = cur + in_dim  # skip-concat the original encoded input
            self.layers.append(nn.Linear(cur, hidden_dim))
            cur = hidden_dim
        self.head = nn.Linear(hidden_dim, 1)

    def encode(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """xy: [..., 2] in [-1, 1], t: [..., 1] in [-1, 1] -> [..., enc_dim]."""
        return torch.cat([self.pe_xy(xy), self.pe_t(t)], dim=-1)

    def forward(self, xy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """xy: [..., 2], t: [..., 1] -> [..., 1] residual in Kelvin."""
        h = enc = self.encode(xy, t)
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, enc], dim=-1)
            h = torch.relu(layer(h))
        out = self.head(h)
        return torch.tanh(out) * self.max_residual_K


def build_pixel_time_grid(H: int, W: int, T: int, device: torch.device | None = None):
    """Construct (x, y, t) coordinates for every (frame, pixel) in [-1, 1]."""
    device = device or torch.device("cpu")
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ts = torch.linspace(-1.0, 1.0, T, device=device)
    tt, yy, xx = torch.meshgrid(ts, ys, xs, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)    # [T, H, W, 2]
    tg = tt.unsqueeze(-1)                 # [T, H, W, 1]
    return xy, tg


def apply_deformation_to_video(
    field: ThermalDeformationField,
    frames: torch.Tensor,
) -> torch.Tensor:
    """Add a per-frame, per-pixel learned residual to a thermal video.

    frames: [B, T, 1, H, W]
    returns: [B, T, 1, H, W] with residual added.
    """
    B, T, C, H, W = frames.shape
    xy, tg = build_pixel_time_grid(H, W, T, device=frames.device)
    inp_xy = xy.reshape(-1, 2)
    inp_t = tg.reshape(-1, 1)
    res = field(inp_xy, inp_t)            # [T*H*W, 1]
    res = res.view(1, T, 1, H, W)        # [1, T, 1, H, W] — matches frames [B, T, 1, H, W]
    return frames + res                   # broadcasts over batch B


def _smoke_test() -> None:
    H, W, T = 16, 16, 6
    B = 2
    field = ThermalDeformationField(hidden_dim=32, n_layers=4)
    n_params = sum(p.numel() for p in field.parameters())
    print(f"ThermalDeformationField: {n_params/1e3:.1f}k params")
    frames = torch.zeros(B, T, 1, H, W)
    out = apply_deformation_to_video(field, frames)
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Residual range: {out.min().item():.4f} K to {out.max().item():.4f} K "
          f"(bounded to ±{field.max_residual_K} K)")


if __name__ == "__main__":
    _smoke_test()
