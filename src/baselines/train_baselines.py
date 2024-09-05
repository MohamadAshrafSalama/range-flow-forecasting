"""
Training script for baseline models.

Usage examples:
  python -m src.baselines.train_baselines --model-type lidarcrafter_like
  python -m src.baselines.train_baselines --model-type occworld_style
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.default import Config, get_default_config
from src.data.temporal_dataset import build_datasets
from src.baselines.models import build_baseline


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_lr(epoch: int, total_epochs: int, base_lr: float, warmup: int = 5, min_lr: float = 1e-6):
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scaler: GradScaler,
    epoch: int,
    accumulation_steps: int,
    device: torch.device,
):
    model.train()
    running = {}
    count = 0
    skipped_nonfinite = 0
    skipped_oom = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        try:
            with autocast(dtype=torch.float16):
                loss, loss_dict = model.training_step(batch, epoch=epoch)
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                skipped_oom += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            raise

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k, v in loss_dict.items():
            if math.isfinite(v):
                running[k] = running.get(k, 0.0) + float(v)
        count += 1

    metrics = {f"train/{k}": (v / max(count, 1)) for k, v in running.items()}
    metrics["train/skipped_nonfinite"] = float(skipped_nonfinite)
    metrics["train/skipped_oom"] = float(skipped_oom)
    return metrics


@torch.no_grad()
def validate(model, loader: DataLoader, device: torch.device, max_batches: int = 40):
    model.eval()
    running = {}
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        try:
            with autocast(dtype=torch.float16):
                loss, loss_dict = model.training_step(batch, epoch=0)
                inf = model.forward_inference(
                    current_pc=batch["current_pc"],
                    rel_pose=batch["relative_pose"],
                    action=batch["action_features"],
                    current_mask=batch["current_mask"],
                    num_steps=20,
                )
                gt_ri, gt_valid = model.projector.project(batch["target_pc"], batch["target_mask"])
                rmse = torch.sqrt(
                    (((inf["pred_range"] - gt_ri) ** 2) * gt_valid).sum() / gt_valid.sum().clamp(min=1.0),
                )
                loss_dict["ri_rmse"] = float(rmse.item())
        except RuntimeError:
            continue

        for k, v in loss_dict.items():
            if math.isfinite(v):
                running[k] = running.get(k, 0.0) + float(v)
        count += 1

    if count == 0:
        return {}
    return {f"val/{k}": v / count for k, v in running.items()}


def save_checkpoint(save_path: Path, model, optimizer, scaler, cfg: Config, epoch: int, model_type: str):
    state = {
        "epoch": epoch,
        "model_type": model_type,
        "config": cfg.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(state, str(save_path))


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model-type", type=str, required=True, choices=["lidarcrafter_like", "occworld_style"])
    parser.add_argument("--config", type=str, default=None, help="Path to config json")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config) if args.config else get_default_config()
    if args.epochs is not None:
        cfg.training.num_epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.training.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.training.learning_rate = float(args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.training.seed)

    root_dir = Path(__file__).resolve().parents[2]
    save_dir = Path(args.save_dir) if args.save_dir else (root_dir / "baselines" / "checkpoints" / args.model_type)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    print("Loading nuScenes...")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=cfg.data.version, dataroot=cfg.data.data_root, verbose=True)
    train_ds, val_ds = build_datasets(
        nusc,
        max_points=cfg.data.max_points,
        train_ratio=cfg.data.train_split_ratio,
        prediction_horizon=cfg.data.prediction_horizon,
        seed=cfg.training.seed,
    )
    del nusc

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.training.batch_size // 2),
        shuffle=False,
        num_workers=max(1, cfg.training.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )

    model = build_baseline(args.model_type, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
    )
    scaler = GradScaler()

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from epoch {start_epoch}")

    train_log = []
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t0 = time.time()
        lr = get_lr(epoch, cfg.training.num_epochs, cfg.training.learning_rate, warmup=5, min_lr=cfg.training.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            accumulation_steps=args.accumulation_steps,
            device=device,
        )
        val_metrics = validate(model, val_loader, device=device, max_batches=30) if (epoch + 1) % 5 == 0 else {}
        elapsed = time.time() - t0

        row = {"epoch": epoch, "lr": lr, "time_s": elapsed, **train_metrics, **val_metrics}
        train_log.append(row)
        with open(save_dir / "training_log.json", "w") as f:
            json.dump(train_log, f, indent=1)

        epoch_ckpt = save_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
        save_checkpoint(epoch_ckpt, model, optimizer, scaler, cfg, epoch, args.model_type)
        save_checkpoint(save_dir / "latest.pth", model, optimizer, scaler, cfg, epoch, args.model_type)

        if val_metrics:
            cur = val_metrics.get("val/ri_rmse", float("inf"))
            if cur < best_val:
                best_val = cur
                save_checkpoint(save_dir / "best_model.pth", model, optimizer, scaler, cfg, epoch, args.model_type)

        print(
            f"[{args.model_type}] epoch {epoch:03d}/{cfg.training.num_epochs - 1} "
            f"lr={lr:.6f} train={train_metrics.get('train/total', float('nan')):.5f} "
            f"val_rmse={val_metrics.get('val/ri_rmse', float('nan')):.5f} time={elapsed:.1f}s",
            flush=True,
        )

    print(f"Training done. Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()
