"""
bioheat.py — Pennes bioheat equation solver on a 3D voxel grid.

Solves:
    rho * c * dT/dt = div(k * grad(T)) + rho_b * w_b * c_b * (T_a - T) + Q_m

Geometry: a 3D tissue block with multiple layers (epidermis, dermis, fat,
muscle) and embedded perforator vessels modelled as cylindrical
high-perfusion sources rising from the muscle layer toward the skin.

Discretization:
- Space: central differences on a regular Cartesian grid (an explicit
  finite-difference method equivalent to first-order finite elements on
  hex meshes for this problem).
- Time: forward Euler, with dt chosen for explicit stability.

This is the synthetic-data engine: every random vessel configuration
produces a paired (thermal-video, ground-truth-mask) sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# Tissue properties — defaults from Werner & Buse 1988, ITIS database
@dataclass
class TissueLayer:
    name: str
    thickness_mm: float
    rho: float
    c: float
    k: float
    w_b: float
    Q_m: float


# 4-layer abdominal skin model (DIEP donor site)
DEFAULT_LAYERS = (
    TissueLayer("epidermis", thickness_mm=0.1, rho=1200, c=3589, k=0.235, w_b=0.0,    Q_m=0.0),
    TissueLayer("dermis",    thickness_mm=1.5, rho=1200, c=3300, k=0.445, w_b=0.0011, Q_m=368.1),
    TissueLayer("fat",       thickness_mm=8.0, rho=1000, c=2674, k=0.185, w_b=0.0001, Q_m=58.0),
    TissueLayer("muscle",    thickness_mm=10.5, rho=1085, c=3768, k=0.510, w_b=0.0009, Q_m=684.2),
)

BLOOD_RHO_C = 1060.0 * 3770.0
T_ARTERY_C = 37.0


@dataclass
class Vessel:
    x_mm: float
    y_mm: float
    radius_mm: float
    depth_mm: float
    quality: float


@dataclass
class SimulationConfig:
    nx: int = 64
    ny: int = 64
    voxel_mm: float = 1.0
    layers: tuple = field(default_factory=lambda: DEFAULT_LAYERS)

    initial_T_C: float = 32.0
    cool_T_C: float = 18.0
    ambient_T_C: float = 22.0
    h_conv: float = 10.0
    t_cool_s: float = 30.0
    t_warm_s: float = 180.0
    fps: int = 5

    dt_s: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_domain(cfg: SimulationConfig, vessels: list[Vessel]):
    """Build per-voxel material fields plus ground-truth surface mask."""
    dx = cfg.voxel_mm * 1e-3
    layer_voxels = [max(1, int(round(L.thickness_mm / cfg.voxel_mm))) for L in cfg.layers]
    nz = sum(layer_voxels)

    rho_c = torch.zeros((nz, cfg.ny, cfg.nx), dtype=torch.float32)
    k = torch.zeros_like(rho_c)
    w_b = torch.zeros_like(rho_c)
    Q_m = torch.zeros_like(rho_c)

    z = 0
    for L, nvox in zip(cfg.layers, layer_voxels):
        rho_c[z:z+nvox] = L.rho * L.c
        k[z:z+nvox] = L.k
        w_b[z:z+nvox] = L.w_b
        Q_m[z:z+nvox] = L.Q_m
        z += nvox

    yy, xx = torch.meshgrid(
        torch.arange(cfg.ny, dtype=torch.float32),
        torch.arange(cfg.nx, dtype=torch.float32),
        indexing="ij",
    )
    gt_mask = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float32)
    gt_quality = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float32)

    for V in vessels:
        cx = V.x_mm / cfg.voxel_mm
        cy = V.y_mm / cfg.voxel_mm
        r_vox = V.radius_mm / cfg.voxel_mm
        depth_vox = max(1, int(round(V.depth_mm / cfg.voxel_mm)))

        cyl = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r_vox ** 2
        z_top = min(depth_vox, nz - 1)
        delta_wb = 0.05 * V.quality
        w_b[z_top:, cyl] = w_b[z_top:, cyl] + delta_wb

        sigma = max(r_vox, 1.0)
        blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        gt_mask = torch.maximum(gt_mask, blob)
        gt_quality = torch.maximum(gt_quality, blob * V.quality)

    return rho_c, k, w_b, Q_m, gt_mask, gt_quality, dx


def _laplacian_xy(T: torch.Tensor, dx: float) -> torch.Tensor:
    """Lateral (x, y) Laplacian with Neumann boundary."""
    lap_x = torch.zeros_like(T)
    lap_x[:, :, 1:-1] = (T[:, :, 2:] - 2 * T[:, :, 1:-1] + T[:, :, :-2])
    lap_x[:, :, 0]  = (T[:, :, 1] - T[:, :, 0])
    lap_x[:, :, -1] = (T[:, :, -2] - T[:, :, -1])
    lap_y = torch.zeros_like(T)
    lap_y[:, 1:-1, :] = (T[:, 2:, :] - 2 * T[:, 1:-1, :] + T[:, :-2, :])
    lap_y[:, 0, :]    = (T[:, 1, :] - T[:, 0, :])
    lap_y[:, -1, :]   = (T[:, -2, :] - T[:, -1, :])
    return (lap_x + lap_y) / (dx ** 2)


def _laplacian_z(T: torch.Tensor, dx: float, T_ghost_top: torch.Tensor) -> torch.Tensor:
    """z-direction Laplacian: Robin BC at top via ghost cell,
    Neumann zero-flux at bottom."""
    lap = torch.zeros_like(T)
    lap[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2])
    lap[0]    = (T[1] - 2 * T[0] + T_ghost_top)
    lap[-1]   = (T[-2] - T[-1])
    return lap / (dx ** 2)


def run_simulation(cfg: SimulationConfig, vessels: list[Vessel]):
    """Run cooling + rewarming. Return (frames, gt_mask, gt_quality)."""
    rho_c, k, w_b, Q_m, gt_mask, gt_quality, dx = build_domain(cfg, vessels)
    device = torch.device(cfg.device)
    rho_c = rho_c.to(device)
    k = k.to(device)
    w_b = w_b.to(device)
    Q_m = Q_m.to(device)

    nz, ny, nx = rho_c.shape
    T = torch.full((nz, ny, nx), cfg.initial_T_C, device=device, dtype=torch.float32)

    k_top = k[0]
    frames = []
    dt = cfg.dt_s
    n_steps_cool = int(round(cfg.t_cool_s / dt))
    n_steps_warm = int(round(cfg.t_warm_s / dt))
    frame_every = max(1, int(round(1.0 / (dt * cfg.fps))))

    def evolve(env_T: float, record: bool, n_steps: int):
        nonlocal T
        for s in range(n_steps):
            T_ghost = T[0] - (2.0 * dx / k_top) * cfg.h_conv * (T[0] - env_T)
            lap = _laplacian_xy(T, dx) + _laplacian_z(T, dx, T_ghost)
            perfusion = BLOOD_RHO_C * w_b * (T_ARTERY_C - T)
            rhs = (k * lap + perfusion + Q_m) / rho_c
            T = T + dt * rhs
            if record and (s % frame_every == 0):
                frames.append(T[0].clone().cpu())

    evolve(cfg.cool_T_C, record=False, n_steps=n_steps_cool)
    evolve(cfg.ambient_T_C, record=True, n_steps=n_steps_warm)

    if not frames:
        frames.append(T[0].clone().cpu())
    return torch.stack(frames, dim=0), gt_mask, gt_quality


def _smoke_test() -> None:
    cfg = SimulationConfig(nx=24, ny=24, t_cool_s=5.0, t_warm_s=15.0, fps=2)
    vessels = [
        Vessel(x_mm=8,  y_mm=10, radius_mm=1.2, depth_mm=8.0, quality=0.9),
        Vessel(x_mm=16, y_mm=14, radius_mm=0.8, depth_mm=6.0, quality=0.5),
    ]
    frames, gt_mask, gt_quality = run_simulation(cfg, vessels)
    print(f"Frames shape: {tuple(frames.shape)}")
    print(f"GT mask shape: {tuple(gt_mask.shape)}, sum {gt_mask.sum().item():.2f}")
    print(f"Temp range: {frames.min().item():.2f} to {frames.max().item():.2f} C")
    last = frames[-1]
    hot_v1 = last[10, 8].item()
    bg = last.mean().item()
    print(f"Late-frame T at strong vessel: {hot_v1:.2f} C vs background {bg:.2f} C")


if __name__ == "__main__":
    _smoke_test()
