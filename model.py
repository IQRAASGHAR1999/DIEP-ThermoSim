"""
model.py — Spatio-temporal U-Net for perforator detection from DIRT video.

Architecture:
    Input:  [B, T, 1, H, W]   thermal video (T frames of single-channel IR)
    Reshape via 3D conv encoder over (T, H, W)
    Encoder: 3D convs reducing T → 1, then 2D U-Net decoder over (H, W)
    Heads:
        mask_head:    predicts per-pixel vessel probability  [B, 1, H, W]
        quality_head: predicts per-pixel vessel quality       [B, 1, H, W]

Uncertainty: dropout layers are retained at inference for MC Dropout.

The architecture is small enough to train on a single GPU; we deliberately
chose a compact design because the underlying signal (perforator hotspot
in a thermal video) is low-frequency and a giant model would overfit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deformation_field import ThermalDeformationField, apply_deformation_to_video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _groups(n_channels: int, max_groups: int = 8) -> int:
    """Return largest divisor of n_channels that is <= max_groups.

    GroupNorm requires num_channels % num_groups == 0.
    min(8, n_channels) breaks when n_channels is not divisible by 8
    (e.g. n_channels=12 gives min(8,12)=8, but 12%8 != 0).
    This helper always returns a valid value.
    """
    g = min(max_groups, n_channels)
    while n_channels % g != 0:
        g -= 1
    return g


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TemporalEncoder(nn.Module):
    """Collapses the time dimension via cascaded 3D convs.

    Input:  [B, 1, T, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, in_ch: int = 1, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T, H, W]
        h = self.block(x)
        # collapse time via mean pooling — also a learned T-conv collapse
        return h.mean(dim=2)  # [B, C, H, W]


class DoubleConv2D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )


class UNet2D(nn.Module):
    """Compact 2D U-Net decoder operating on the temporally-collapsed features."""

    def __init__(self, in_ch: int = 16, base: int = 32, dropout: float = 0.1):
        super().__init__()
        self.enc1 = DoubleConv2D(in_ch, base, dropout)
        self.enc2 = DoubleConv2D(base, base * 2, dropout)
        self.enc3 = DoubleConv2D(base * 2, base * 4, dropout)
        self.bottleneck = DoubleConv2D(base * 4, base * 8, dropout)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv2D(base * 8, base * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv2D(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv2D(base * 2, base, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1


class PerforatorNet(nn.Module):
    """Full model: optional thermal deformation field + temporal encoder + 2D U-Net + dual heads.

    The deformation field is a direct port of the architecture used in
    DynGS-Pro (Asghar et al., KSEM 2026), adapted from 3D scene dynamics
    to 2D thermal residuals. See deformation_field.py for details.
    """

    def __init__(self, base: int = 32, hidden_3d: int = 16, dropout: float = 0.1,
                 use_deformation_field: bool = True):
        super().__init__()
        self.use_deformation_field = use_deformation_field
        if use_deformation_field:
            self.deformation = ThermalDeformationField(
                n_freqs_xy=6, n_freqs_t=4,
                hidden_dim=64, n_layers=4, max_residual_K=1.0,
            )
        self.temporal = TemporalEncoder(in_ch=1, hidden=hidden_3d, dropout=dropout)
        self.unet = UNet2D(in_ch=hidden_3d, base=base, dropout=dropout)
        self.mask_head = nn.Conv2d(base, 1, kernel_size=1)
        self.quality_head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, 1, H, W] -> (mask_logit, quality) each [B, 1, H, W]"""
        if self.use_deformation_field:
            x = apply_deformation_to_video(self.deformation, x)
        x = x.permute(0, 2, 1, 3, 4)  # [B, 1, T, H, W]
        feat = self.temporal(x)
        feat = self.unet(feat)
        mask_logit = self.mask_head(feat)
        quality = torch.sigmoid(self.quality_head(feat))
        return mask_logit, quality


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice between sigmoid(logits) and target (both in [0,1])."""
    p = torch.sigmoid(logits)
    dims = (1, 2, 3) if p.ndim == 4 else (1, 2)
    inter = (p * target).sum(dim=dims)
    union = p.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()


def perforator_loss(mask_logit: torch.Tensor,
                    quality_pred: torch.Tensor,
                    gt_mask: torch.Tensor,
                    gt_quality: torch.Tensor) -> dict:
    """Compound loss: BCE + Dice on mask, masked MSE on quality."""
    gt_mask4 = gt_mask.unsqueeze(1)
    gt_quality4 = gt_quality.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(mask_logit, gt_mask4)
    dice = soft_dice_loss(mask_logit, gt_mask4)
    # quality regression weighted by where the GT mask is non-trivial
    weight = gt_mask4.clamp(min=0.05)
    mse = (((quality_pred - gt_quality4) ** 2) * weight).mean()
    total = bce + dice + 0.5 * mse
    return {"total": total, "bce": bce, "dice": dice, "mse_quality": mse}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    B, T, H, W = 2, 8, 32, 32
    x = torch.randn(B, T, 1, H, W)
    gt_mask = torch.zeros(B, H, W)
    gt_mask[:, 14:18, 14:18] = 1.0
    gt_q = gt_mask * 0.7
    model = PerforatorNet(base=16, hidden_3d=8, dropout=0.1)
    mask_logit, quality = model(x)
    print(f"mask_logit shape: {tuple(mask_logit.shape)}")
    print(f"quality shape: {tuple(quality.shape)}")
    losses = perforator_loss(mask_logit, quality, gt_mask, gt_q)
    print(f"Loss: {losses['total'].item():.4f} "
          f"(bce={losses['bce'].item():.4f}, dice={losses['dice'].item():.4f})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params/1e6:.2f}M parameters")


if __name__ == "__main__":
    _smoke_test()
