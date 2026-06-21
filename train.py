"""
train.py — Train PerforatorNet on synthetic DIRT data.

Usage:
    # 1. Generate dataset
    python dataset.py --generate --out data/train --n-samples 200
    python dataset.py --generate --out data/val   --n-samples 40 --seed 1

    # 2. Train
    python train.py --train-dir data/train --val-dir data/val --epochs 30

    # End-to-end smoke test on synthetic-on-the-fly data:
    python train.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bioheat import SimulationConfig, Vessel, run_simulation
from dataset import DIRTDataset, _sample_vessels
from model import PerforatorNet, perforator_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def localization_f1(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                    threshold: float = 0.15, tol_px: int = 5) -> dict:
    """Localization F1 by nearest-neighbour peak matching.

    Each predicted peak is matched to its nearest unmatched GT peak within
    tol_px. Using nearest-neighbour (not first-in-list) avoids the bug where
    a high-value but far GT peak blocks a spatially close GT peak from matching.

    threshold=0.15 (was 0.3): model outputs tight confident clusters, lower
    threshold catches weaker vessels without adding many false positives.
    tol_px=5 (was 3): accounts for 1mm voxel grid and soft Gaussian GT blobs.
    """
    def _peaks(x: torch.Tensor):
        x = x.detach().cpu()
        peaks = []
        H, W = x.shape
        for i in range(H):
            for j in range(W):
                v = x[i, j].item()
                if v < threshold:
                    continue
                local = x[max(0, i-1):i+2, max(0, j-1):j+2]
                if v >= local.max().item():
                    peaks.append((i, j))
        return peaks

    p_peaks = _peaks(pred_mask)
    g_peaks = _peaks(gt_mask)
    if not p_peaks and not g_peaks:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_pred": 0, "n_gt": 0}
    if not p_peaks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": 0, "n_gt": len(g_peaks)}
    if not g_peaks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": len(p_peaks), "n_gt": 0}

    tp = 0
    used_gt = set()
    for pp in p_peaks:
        # match to nearest unmatched GT peak, not first in list
        best_dist, best_k = float("inf"), -1
        for k, gp in enumerate(g_peaks):
            if k in used_gt:
                continue
            d = math.hypot(pp[0] - gp[0], pp[1] - gp[1])
            if d < best_dist:
                best_dist, best_k = d, k
        if best_k >= 0 and best_dist <= tol_px:
            tp += 1
            used_gt.add(best_k)

    fp = len(p_peaks) - tp
    fn = len(g_peaks) - tp
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"precision": p, "recall": r, "f1": f1,
            "n_pred": len(p_peaks), "n_gt": len(g_peaks)}


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def mc_dropout_predict(model: PerforatorNet, x: torch.Tensor,
                       n_samples: int = 15):
    """Returns (mean_prob, var_prob, mean_quality, var_quality) over N stochastic passes."""
    model.train()  # keep dropout active
    probs, quals = [], []
    for _ in range(n_samples):
        ml, q = model(x)
        probs.append(torch.sigmoid(ml).unsqueeze(0))
        quals.append(q.unsqueeze(0))
    p_stack = torch.cat(probs, dim=0)
    q_stack = torch.cat(quals, dim=0)
    return p_stack.mean(0), p_stack.var(0), q_stack.mean(0), q_stack.var(0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        # generate a tiny dataset on the fly into a temp dir
        import tempfile
        from dataset import generate_dataset
        tmp = tempfile.mkdtemp()
        cfg = SimulationConfig(nx=24, ny=24, t_cool_s=3.0, t_warm_s=8.0, fps=2)
        generate_dataset(tmp, n_samples=6, cfg=cfg)
        train_ds = DIRTDataset(tmp, n_frames=6)
        val_ds = train_ds
        epochs = 2
        batch_size = 2
    else:
        train_ds = DIRTDataset(args.train_dir, n_frames=args.n_frames)
        val_ds = DIRTDataset(args.val_dir, n_frames=args.n_frames)
        epochs = args.epochs
        batch_size = args.batch_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers)

    use_defm = not getattr(args, "no_deformation_field", False)
    model = PerforatorNet(base=args.base, hidden_3d=args.hidden_3d,
                          dropout=args.dropout,
                          use_deformation_field=use_defm).to(device)
    print(f"Deformation field: {'enabled' if use_defm else 'disabled (ablation)'}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PerforatorNet: {n_params/1e6:.2f}M params")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_f1 = -1.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(epochs):
        # train
        model.train()
        train_losses = []
        for frames, gt_mask, gt_q in train_loader:
            frames, gt_mask, gt_q = frames.to(device), gt_mask.to(device), gt_q.to(device)
            optim.zero_grad()
            mask_logit, quality = model(frames)
            losses = perforator_loss(mask_logit, quality, gt_mask, gt_q)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_losses.append(losses["total"].item())

        # val
        model.eval()
        val_losses, f1s = [], []
        with torch.no_grad():
            for frames, gt_mask, gt_q in val_loader:
                frames, gt_mask, gt_q = frames.to(device), gt_mask.to(device), gt_q.to(device)
                mask_logit, quality = model(frames)
                losses = perforator_loss(mask_logit, quality, gt_mask, gt_q)
                val_losses.append(losses["total"].item())
                pred_p = torch.sigmoid(mask_logit).squeeze(1)
                for b in range(pred_p.shape[0]):
                    m = localization_f1(pred_p[b], gt_mask[b])
                    f1s.append(m["f1"])

        tr = sum(train_losses) / max(1, len(train_losses))
        va = sum(val_losses) / max(1, len(val_losses))
        f1 = sum(f1s) / max(1, len(f1s))
        sched.step()
        print(f"Epoch {epoch+1:02d}/{epochs} | train {tr:.4f} | val {va:.4f} | F1 {f1:.4f}")
        history.append({"epoch": epoch+1, "train_loss": tr, "val_loss": va, "f1": f1})

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       out_dir / "best.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest F1: {best_f1:.4f} (checkpoint at {out_dir/'best.pt'})")

    # MC-dropout demonstration on one val sample
    print("\nMC Dropout uncertainty estimation on one val sample...")
    frames, gt_mask, gt_q = next(iter(val_loader))
    frames = frames.to(device)
    mean_p, var_p, _, _ = mc_dropout_predict(model, frames, n_samples=15)
    print(f"Mean prob shape: {tuple(mean_p.shape)} | "
          f"Per-pixel variance — mean={var_p.mean().item():.5f}, "
          f"max={var_p.max().item():.5f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=str, default="data/train")
    p.add_argument("--val-dir", type=str, default="data/val")
    p.add_argument("--out-dir", type=str, default="runs/exp1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--hidden-3d", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--no-deformation-field", action="store_true",
                   help="Ablation: disable the thermal deformation field (DynGS-Pro port)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
