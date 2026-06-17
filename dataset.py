"""
dataset.py — Synthetic DIRT dataset generation and PyTorch Dataset wrapper.

Calls the Pennes bioheat solver with randomized vessel layouts to generate
paired (thermal_video, vessel_mask, quality_map) samples.

Workflow:
    generate_dataset(out_dir, n_samples=...)  # offline; saves .pt files
    DIRTDataset(out_dir)                      # PyTorch Dataset

Each saved sample is a single .pt file with keys:
    frames      [T, H, W]    surface temperature time series
    gt_mask     [H, W]       soft vessel projection mask
    gt_quality  [H, W]       per-pixel vessel quality
    meta        dict         vessel list + sim config
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from bioheat import SimulationConfig, Vessel, run_simulation


def _sample_vessels(cfg: SimulationConfig, n_min: int = 2, n_max: int = 6) -> list[Vessel]:
    """Sample a random plausible perforator configuration.

    Constraints based on the DIEP donor-site literature:
    - typical perforator radius 0.3–1.5 mm at the surface
    - originates from muscle, so depth 6–12 mm below skin
    - quality (a clinical heuristic) drawn uniform on [0.3, 1.0]
    - 2–6 perforators per hemiabdomen
    - minimum separation 5 mm to avoid overlap
    """
    n = random.randint(n_min, n_max)
    field_w_mm = cfg.nx * cfg.voxel_mm
    field_h_mm = cfg.ny * cfg.voxel_mm
    vessels: list[Vessel] = []
    attempts = 0
    while len(vessels) < n and attempts < 200:
        attempts += 1
        x = random.uniform(5, field_w_mm - 5)
        y = random.uniform(5, field_h_mm - 5)
        # rejection sampling for minimum separation
        too_close = any(((v.x_mm - x) ** 2 + (v.y_mm - y) ** 2) < 5.0 ** 2 for v in vessels)
        if too_close:
            continue
        vessels.append(Vessel(
            x_mm=x, y_mm=y,
            radius_mm=random.uniform(0.3, 1.5),
            depth_mm=random.uniform(6.0, 12.0),
            quality=random.uniform(0.3, 1.0),
        ))
    return vessels


def generate_dataset(out_dir: str, n_samples: int, seed: int = 0,
                     cfg: SimulationConfig | None = None) -> None:
    """Generate and save n_samples to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    cfg = cfg or SimulationConfig()
    for i in range(n_samples):
        vessels = _sample_vessels(cfg)
        frames, gt_mask, gt_quality = run_simulation(cfg, vessels)
        sample = {
            "frames": frames.cpu(),
            "gt_mask": gt_mask.cpu(),
            "gt_quality": gt_quality.cpu(),
            "meta": {
                "vessels": [asdict(v) for v in vessels],
                "cfg": {k: v for k, v in asdict(cfg).items() if k != "layers"},
            },
        }
        path = out / f"sample_{i:05d}.pt"
        torch.save(sample, path)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  saved {i+1}/{n_samples} → {path.name}")
    print(f"Done. Generated {n_samples} samples in {out}")


class DIRTDataset(Dataset):
    """PyTorch Dataset over the synthetic DIRT samples.

    Returns:
        frames:    FloatTensor [T, 1, H, W]   (single thermal channel)
        gt_mask:   FloatTensor [H, W]
        gt_quality: FloatTensor [H, W]
    """

    def __init__(self, root: str, n_frames: int | None = None, normalize: bool = True):
        self.paths = sorted(Path(root).glob("sample_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No sample_*.pt files in {root}. "
                                    "Run dataset.py --generate first.")
        self.n_frames = n_frames
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        s = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        frames = s["frames"].float()  # [T, H, W]
        if self.n_frames is not None and frames.shape[0] >= self.n_frames:
            # uniformly subsample frames
            idxs = torch.linspace(0, frames.shape[0] - 1, self.n_frames).long()
            frames = frames[idxs]
        if self.normalize:
            # per-sample z-normalization (matches typical DIRT preprocessing)
            mu = frames.mean()
            sd = frames.std() + 1e-6
            frames = (frames - mu) / sd
        frames = frames.unsqueeze(1)  # [T, 1, H, W]
        return frames, s["gt_mask"].float(), s["gt_quality"].float()


def _smoke_test() -> None:
    import tempfile
    cfg = SimulationConfig(nx=24, ny=24, t_cool_s=3.0, t_warm_s=10.0, fps=2)
    with tempfile.TemporaryDirectory() as tmp:
        generate_dataset(tmp, n_samples=2, cfg=cfg)
        ds = DIRTDataset(tmp)
        frames, gt, q = ds[0]
        print(f"Sample frames: {tuple(frames.shape)}")
        print(f"GT mask: {tuple(gt.shape)} max={gt.max().item():.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--out", type=str, default="data/train")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke_test:
        _smoke_test()
        return
    if args.generate:
        generate_dataset(args.out, args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
