"""
Training script for the range-image world model.

Key features:
  - Range-image representation (32x1024) with UNet and flow matching
  - Robust training with AMP, EMA, gradient clipping
  - Multi-step scheduled sampling for long-horizon rollout
  - DDP support for multi-GPU training
"""

import os
import sys
import csv
import json
import math
import time
import datetime
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.default import Config, get_default_config
from src.model.world_model import build_world_model, ResBlock
from src.model.losses import RangeWorldModelLoss, MultiStepLoss
from src.data.temporal_dataset import build_datasets


# =====================================================================
#  EMA
# =====================================================================

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                if torch.isfinite(p.data).all():
                    self.shadow[name].lerp_(p.data, 1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {
            name: p.data.clone() for name, p in model.named_parameters()
            if p.requires_grad and name in self.shadow
        }
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


# =====================================================================
#  Logger
# =====================================================================

class TrainingLogger:
    """CSV + JSON + TensorBoard logging."""

    def __init__(self, log_dir: str, rank: int = 0):
        self.log_dir = Path(log_dir)
        self.rank = rank
        if rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.log_dir / "metrics.csv"
            self.json_path = self.log_dir / "training_log.json"
            self.json_data = []
            self.csv_writer = None
            self.csv_file = None
            self.tb_writer = None
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(str(self.log_dir / "tb"))
            except ImportError:
                pass

    def log_epoch(self, epoch: int, metrics: dict):
        if self.rank != 0:
            return

        self.json_data.append({"epoch": epoch, **metrics})
        with open(self.json_path, "w") as f:
            json.dump(self.json_data, f, indent=1)

        all_fields = [
            "lr", "time_s",
            "train/total", "train/velocity", "train/x1_huber", "train/edge", "train/valid_bce", "train/freq",
            "train/empty", "train/ray", "train/angular", "train/occupancy", "train/chamfer_reg",
            "train_ms/total",
            "val/total", "val/velocity", "val/x1_huber", "val/edge", "val/valid_bce", "val/freq",
            "val/empty", "val/ray", "val/angular", "val/occupancy", "val/chamfer_reg",
            "val/ri_rmse", "val/free_fp_rate",
        ]
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(
                self.csv_file, fieldnames=["epoch"] + all_fields, extrasaction="ignore",
            )
            self.csv_writer.writeheader()

        row = {"epoch": epoch, **{k: f"{v:.6f}" if isinstance(v, float) else v for k, v in metrics.items()}}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, epoch)

    def close(self):
        if getattr(self, 'csv_file', None):
            self.csv_file.close()
        if getattr(self, 'tb_writer', None):
            self.tb_writer.close()


# =====================================================================
#  Training loop
# =====================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RangeWorldModelLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: Config,
    epoch: int,
    ema,
    rank: int = 0,
) -> dict:
    """Single training epoch. Returns averaged loss dict."""
    model.train()
    running = {}
    count = 0
    accum = cfg.training.accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    oom_count = 0
    nonfinite_count = 0

    for step, batch in enumerate(dataloader):
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bad_batch = False

        try:
            with autocast(dtype=torch.float16):
                raw_model = model.module if hasattr(model, "module") else model
                output = raw_model.forward_train(
                    current_pc=batch["current_pc"],
                    target_pc=batch["target_pc"],
                    rel_pose=batch["relative_pose"],
                    action=batch["action_features"],
                    current_mask=batch["current_mask"],
                    target_mask=batch["target_mask"],
                )
                loss, loss_dict = criterion(output, epoch)

                tcfg = cfg.training
                ode_prob = getattr(tcfg, "ode_consistency_prob", 0.0)
                ode_start = getattr(tcfg, "ode_consistency_start_epoch", 4)
                ode_steps = getattr(tcfg, "ode_consistency_steps", 10)
                if (
                    ode_prob > 0
                    and epoch >= ode_start
                    and torch.rand(1, device=loss.device).item() < ode_prob
                ):
                    gt_range, gt_valid = raw_model.projector.project(
                        batch["target_pc"], batch["target_mask"]
                    )
                    x1_ode = raw_model.forward_ode_integration_train(
                        current_pc=batch["current_pc"],
                        rel_pose=batch["relative_pose"],
                        action=batch["action_features"],
                        current_mask=batch["current_mask"],
                        num_steps=ode_steps,
                        use_midpoint=True,
                    )
                    l_ode = F.smooth_l1_loss(
                        x1_ode * gt_valid, gt_range * gt_valid, beta=0.05
                    )
                    loss = loss + 0.5 * l_ode
                    loss_dict["ode_consistency"] = l_ode.item()
                if not torch.isfinite(loss):
                    nonfinite_count += 1
                    bad_batch = True

            if bad_batch:
                optimizer.zero_grad(set_to_none=True)
                dummy = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
                scaler.scale(dummy).backward()
            else:
                scaler.scale(loss / accum).backward()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[Rank {rank}] OOM at step {step}, skipping batch")
                oom_count += 1
                bad_batch = True
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                dummy = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
                scaler.scale(dummy).backward()
            else:
                raise

        if bad_batch:
            loss_dict = {}

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

        if not bad_batch:
            for k, v in loss_dict.items():
                if math.isfinite(v):
                    running[k] = running.get(k, 0.0) + v
            count += 1

        if rank == 0 and step % cfg.training.log_interval == 0 and count > 0:
            avg_total = running.get("total", 0) / count
            avg_vel = running.get("velocity", 0) / count
            print(
                f"  step {step:4d}/{len(dataloader)} | "
                f"total={avg_total:.5f} vel={avg_vel:.5f}",
                flush=True,
            )

    if rank == 0 and (oom_count > 0 or nonfinite_count > 0):
        print(
            f"[train] skipped batches: oom={oom_count}, nonfinite={nonfinite_count}",
            flush=True,
        )

    if count == 0:
        return {}
    return {f"train/{k}": v / count for k, v in running.items()}


def get_scheduled_sampling_prob(epoch: int, cfg: Config) -> float:
    """Compute scheduled sampling probability for current epoch."""
    tcfg = cfg.training
    if epoch < tcfg.ss_start_epoch:
        return 0.0
    progress = (epoch - tcfg.ss_start_epoch) / max(tcfg.ss_rampup_epochs, 1)
    return min(progress, 1.0) * tcfg.ss_final_prob


def get_active_rollout_steps(epoch: int, cfg: Config) -> int:
    """Progressive curriculum: number of active rollout steps for this epoch."""
    tcfg = cfg.training
    if not getattr(tcfg, "ms_curriculum", False):
        return tcfg.sequence_length
    if epoch < tcfg.ss_start_epoch:
        return 0
    min_steps = getattr(tcfg, "ms_min_steps", 2)
    interval = max(getattr(tcfg, "ms_step_increase_interval", 5), 1)
    epochs_since_start = epoch - tcfg.ss_start_epoch
    active = min_steps + epochs_since_start // interval
    return min(active, tcfg.sequence_length)


def train_multistep_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiStepLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: Config,
    epoch: int,
    ema,
    rank: int = 0,
) -> dict:
    """Multi-step training epoch with scheduled sampling."""
    model.train()
    running = {}
    count = 0
    accum = getattr(cfg.training, "multistep_accumulation_steps", cfg.training.accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    oom_count = 0
    nonfinite_count = 0

    ss_prob = get_scheduled_sampling_prob(epoch, cfg)
    active_K = get_active_rollout_steps(epoch, cfg)

    if rank == 0 and count == 0:
        print(
            f"  [multi-step] active_steps={active_K}, ss_prob={ss_prob:.3f}, accum={accum}",
            flush=True,
        )

    if active_K <= 0:
        return {}

    for step, batch in enumerate(dataloader):
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bad_batch = False

        try:
            with autocast(dtype=torch.float16):
                raw_model = model.module if hasattr(model, "module") else model
                K_data = batch["rel_poses"].shape[1]
                K = min(active_K, K_data)

                frames_pc = [batch["frames_pc"][:, i] for i in range(K + 1)]
                frames_mask = [batch["frames_mask"][:, i] for i in range(K + 1)]
                rel_poses = [batch["rel_poses"][:, i] for i in range(K)]
                actions = [batch["actions"][:, i] for i in range(K)]

                kwargs = dict(
                    frames_pc=frames_pc,
                    frames_mask=frames_mask,
                    rel_poses=rel_poses,
                    actions=actions,
                    sampling_prob=ss_prob,
                )
                if "history_pc" in batch and "history_mask" in batch:
                    kwargs["history_pc"] = batch["history_pc"]
                    kwargs["history_mask"] = batch["history_mask"]
                outputs = raw_model.forward_train_scheduled(**kwargs)
                loss, loss_dict = criterion(
                    outputs, epoch,
                    frames_pc=frames_pc, frames_mask=frames_mask, rel_poses=rel_poses,
                )
                if not torch.isfinite(loss):
                    nonfinite_count += 1
                    bad_batch = True

            if bad_batch:
                optimizer.zero_grad(set_to_none=True)
                dummy = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
                scaler.scale(dummy).backward()
            else:
                scaler.scale(loss / accum).backward()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[Rank {rank}] OOM at step {step} (multi-step), skipping")
                oom_count += 1
                bad_batch = True
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                dummy = sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)
                scaler.scale(dummy).backward()
            else:
                raise

        if bad_batch:
            loss_dict = {}

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

        if not bad_batch:
            for k, v in loss_dict.items():
                if math.isfinite(v):
                    running[k] = running.get(k, 0.0) + v
            count += 1

        if rank == 0 and step % cfg.training.log_interval == 0 and count > 0:
            avg_total = running.get("total", 0) / count
            print(
                f"  [ms] step {step:4d}/{len(dataloader)} | total={avg_total:.5f}",
                flush=True,
            )

    if rank == 0 and (oom_count > 0 or nonfinite_count > 0):
        print(
            f"[train_ms] skipped batches: oom={oom_count}, nonfinite={nonfinite_count}",
            flush=True,
        )

    if count == 0:
        return {}
    return {f"train_ms/{k}": v / count for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RangeWorldModelLoss,
    cfg: Config,
    epoch: int,
) -> dict:
    """Validation epoch on true inference pathway."""
    model.eval()
    running = {}
    count = 0
    skipped = 0

    for batch in dataloader:
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        try:
            with autocast(dtype=torch.float16):
                raw_model = model.module if hasattr(model, "module") else model
                inf_out = raw_model.forward_inference(
                    current_pc=batch["current_pc"],
                    rel_pose=batch["relative_pose"],
                    action=batch["action_features"],
                    current_mask=batch["current_mask"],
                    num_steps=cfg.evaluation.eval_inference_steps,
                )
                gt_range, gt_valid = raw_model.projector.project(
                    batch["target_pc"], batch["target_mask"],
                )

                pred_range = inf_out["pred_range"].float()
                gt_range = gt_range.float()
                gt_valid = gt_valid.float()

                l_x1 = F.smooth_l1_loss(pred_range * gt_valid, gt_range * gt_valid, beta=0.05)
                l_edge = criterion.edge_loss(pred_range, gt_range, gt_valid)
                l_valid = criterion._valid_focal_loss(pred_range, gt_range)
                l_freq = criterion.freq_loss(pred_range, gt_range, gt_valid)
                l_empty = criterion.empty_loss(pred_range, gt_valid)
                l_ray = criterion.ray_loss(pred_range, gt_range, gt_valid)
                l_angular = criterion.angular_loss(pred_range, gt_range, gt_valid)
                l_occupancy = criterion.occupancy_loss(
                    pred_range, gt_range, criterion.valid_threshold,
                )
                if criterion.lambda_chamfer > 0:
                    l_chamfer = criterion.chamfer_reg(pred_range, gt_range, gt_valid)
                else:
                    l_chamfer = pred_range.new_tensor(0.0)
                total = (
                    criterion.lambda_x1 * l_x1
                    + criterion.lambda_edge * l_edge
                    + criterion.lambda_valid * l_valid
                    + criterion.lambda_freq * l_freq
                    + criterion.lambda_empty * l_empty
                    + criterion.lambda_ray * l_ray
                    + criterion.lambda_angular * l_angular
                    + criterion.lambda_occupancy * l_occupancy
                    + criterion.lambda_chamfer * l_chamfer
                )

                rmse = torch.sqrt(
                    ((pred_range - gt_range) ** 2 * gt_valid).sum()
                    / gt_valid.sum().clamp(min=1.0)
                )
                pred_valid = raw_model._compute_validity_mask(
                    inf_out["pred_range"], inf_out["x_0_valid"],
                )
                gt_invalid = gt_valid <= 0.5
                free_fp_rate = (
                    ((pred_valid > 0.5) & gt_invalid).float().sum()
                    / gt_invalid.float().sum().clamp(min=1.0)
                )
                loss_dict = {
                    "total": total.item(),
                    "x1_huber": l_x1.item(),
                    "edge": l_edge.item(),
                    "valid_bce": l_valid.item(),
                    "freq": l_freq.item(),
                    "empty": l_empty.item(),
                    "ray": l_ray.item(),
                    "angular": l_angular.item(),
                    "occupancy": l_occupancy.item(),
                    "chamfer_reg": l_chamfer.item(),
                    "ri_rmse": rmse.item(),
                    "free_fp_rate": free_fp_rate.item(),
                }
        except RuntimeError:
            skipped += 1
            continue

        for k, v in loss_dict.items():
            if math.isfinite(v):
                running[k] = running.get(k, 0.0) + v
        count += 1

    if count == 0:
        return {}
    if skipped > 0:
        print(f"[validate] skipped {skipped} batches due to runtime errors")
    return {f"val/{k}": v / count for k, v in running.items()}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema,
    epoch: int,
    cfg: Config,
    save_path: str,
):
    """Save training checkpoint."""
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": cfg.to_dict(),
    }
    if ema is not None:
        state["ema_shadow"] = ema.shadow
    torch.save(state, save_path)


@torch.no_grad()
def save_visualisation(
    model: nn.Module,
    dataloader: DataLoader,
    save_dir: str,
    epoch: int,
    n_samples: int = 4,
):
    """Save range-image comparison visualisations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    batch = next(iter(dataloader))
    batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    ns = min(n_samples, batch["current_pc"].shape[0])

    train_out = raw_model.forward_train(
        current_pc=batch["current_pc"][:ns],
        target_pc=batch["target_pc"][:ns],
        rel_pose=batch["relative_pose"][:ns],
        action=batch["action_features"][:ns],
        current_mask=batch["current_mask"][:ns],
        target_mask=batch["target_mask"][:ns],
    )
    x_0 = train_out["x_0"].cpu()
    x_1 = train_out["x_1"].cpu()

    inf_out = raw_model.forward_inference(
        current_pc=batch["current_pc"][:ns],
        rel_pose=batch["relative_pose"][:ns],
        action=batch["action_features"][:ns],
        current_mask=batch["current_mask"][:ns],
    )
    x1_pred = inf_out["pred_range"].cpu().clamp(-1, 1)

    for i in range(ns):
        fig, axes = plt.subplots(3, 1, figsize=(16, 6), dpi=100)
        axes[0].imshow(x_0[i, 0].numpy(), cmap="turbo", vmin=-1, vmax=1, aspect="auto")
        axes[0].set_title("Warped Previous (x_0)")
        axes[0].axis("off")
        axes[1].imshow(x_1[i, 0].numpy(), cmap="turbo", vmin=-1, vmax=1, aspect="auto")
        axes[1].set_title("Ground Truth (x_1)")
        axes[1].axis("off")
        axes[2].imshow(x1_pred[i, 0].numpy(), cmap="turbo", vmin=-1, vmax=1, aspect="auto")
        axes[2].set_title("Predicted (x1_hat)")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(save_path / f"epoch{epoch:03d}_sample{i}.png", bbox_inches="tight")
        plt.close(fig)


# =====================================================================
#  LR schedule
# =====================================================================

def get_lr(epoch: int, cfg: Config, base_lr: float) -> float:
    """Linear warmup + cosine annealing."""
    warmup = cfg.training.warmup_epochs
    total = cfg.training.num_epochs
    min_lr = cfg.training.min_lr

    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total - warmup, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train range-image world model")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    distributed = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
        torch.cuda.set_device(0)

    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg = get_default_config()

    base_lr = cfg.training.learning_rate * math.sqrt(world_size)

    torch.manual_seed(cfg.training.seed + rank)
    np.random.seed(cfg.training.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        Path(cfg.training.save_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.training.vis_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.training.log_dir).mkdir(parents=True, exist_ok=True)
        cfg.save(os.path.join(cfg.training.save_dir, "config.json"))

    if rank == 0:
        print("Loading nuScenes...")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg.data.version, dataroot=cfg.data.data_root, verbose=(rank == 0))

    tcfg = cfg.training
    if tcfg.enable_multistep:
        result = build_datasets(
            nusc,
            max_points=cfg.data.max_points,
            train_ratio=cfg.data.train_split_ratio,
            prediction_horizon=cfg.data.prediction_horizon,
            seed=cfg.training.seed,
            sequence_length=tcfg.sequence_length,
            history_frames=getattr(cfg.model, "history_frames", 0) if getattr(cfg.model, "use_history", False) else 0,
        )
        train_ds, val_ds, train_seq_ds, val_seq_ds = result
    else:
        train_ds, val_ds = build_datasets(
            nusc,
            max_points=cfg.data.max_points,
            train_ratio=cfg.data.train_split_ratio,
            prediction_horizon=cfg.data.prediction_horizon,
            seed=cfg.training.seed,
        )
        train_seq_ds = val_seq_ds = None
    del nusc

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.training.num_workers > 0,
        prefetch_factor=4 if cfg.training.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.training.num_workers // 2 or 1,
        pin_memory=True,
        drop_last=False,
    )

    ms_train_loader = None
    if train_seq_ds is not None:
        ms_train_sampler = DistributedSampler(train_seq_ds, shuffle=True) if distributed else None
        ms_num_workers = max(cfg.training.num_workers // 2, 1)
        ms_train_loader = DataLoader(
            train_seq_ds,
            batch_size=tcfg.multistep_batch_size,
            sampler=ms_train_sampler,
            shuffle=(ms_train_sampler is None),
            num_workers=ms_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=ms_num_workers > 0,
            prefetch_factor=4 if ms_num_workers > 0 else None,
        )

    model = build_world_model(cfg).cuda()
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
    )
    scaler = GradScaler()

    ema = None
    if cfg.model.use_ema and rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        ema = EMAModel(raw_model, decay=cfg.model.ema_decay)

    criterion = RangeWorldModelLoss(cfg).cuda()
    ms_criterion = MultiStepLoss(cfg).cuda() if tcfg.enable_multistep else None

    logger = TrainingLogger(cfg.training.log_dir, rank)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if ema and "ema_shadow" in ckpt:
            ema.shadow = {
                k: v.to(torch.cuda.current_device())
                for k, v in ckpt["ema_shadow"].items()
            }
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Range-Image World Model Training")
        print(f"  Parameters: {n_params/1e6:.1f}M")
        print(f"  Epochs: {cfg.training.num_epochs}")
        print(f"  Batch: {cfg.training.batch_size} x {world_size} GPU x {cfg.training.accumulation_steps} accum")
        print(f"  Range image: {cfg.model.ri_H} x {cfg.model.ri_W}")
        print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        t0 = time.time()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        lr = get_lr(epoch, cfg, base_lr)
        set_lr(optimizer, lr)

        if rank == 0:
            print(f"Epoch {epoch:3d}/{cfg.training.num_epochs} | lr={lr:.6f}")

        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg, epoch, ema, rank
        )

        ms_metrics = {}
        if (
            ms_train_loader is not None
            and ms_criterion is not None
            and epoch >= tcfg.ss_start_epoch
        ):
            if rank == 0:
                print("  [multi-step training phase]")
            if hasattr(ms_train_loader, "sampler") and hasattr(ms_train_loader.sampler, "set_epoch"):
                ms_train_loader.sampler.set_epoch(epoch)
            ms_metrics = train_multistep_epoch(
                model, ms_train_loader, ms_criterion, optimizer, scaler,
                cfg, epoch, ema, rank,
            )

        val_metrics = {}
        if (epoch + 1) % cfg.training.eval_interval == 0:
            if ema:
                ema.apply_shadow(model.module if hasattr(model, "module") else model)
            val_metrics = validate(model, val_loader, criterion, cfg, epoch)
            if ema:
                ema.restore(model.module if hasattr(model, "module") else model)

        elapsed = time.time() - t0

        all_metrics = {
            "lr": lr,
            "time_s": elapsed,
            **train_metrics,
            **ms_metrics,
            **val_metrics,
        }

        if rank == 0:
            logger.log_epoch(epoch, all_metrics)
            t_total = train_metrics.get("train/total", 0)
            v_total = val_metrics.get("val/total", 0)
            ms_total = ms_metrics.get("train_ms/total", 0)
            log_msg = f"  -> train={t_total:.5f}"
            if ms_total > 0:
                log_msg += f" | ms={ms_total:.5f}"
            log_msg += f" | val={v_total:.5f} | time={elapsed:.0f}s"
            print(log_msg)

            if (epoch + 1) % cfg.training.save_interval == 0:
                save_path = os.path.join(
                    cfg.training.save_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(model, optimizer, scaler, ema, epoch, cfg, save_path)
                print(f"  Saved checkpoint: {save_path}")

            if val_metrics and val_metrics.get("val/total", float("inf")) < best_val_loss:
                best_val_loss = val_metrics["val/total"]
                save_path = os.path.join(cfg.training.save_dir, "best_model.pth")
                save_checkpoint(model, optimizer, scaler, ema, epoch, cfg, save_path)
                print(f"  New best model: val_total={best_val_loss:.5f}")

            if (epoch + 1) % cfg.training.eval_interval == 0:
                try:
                    save_visualisation(model, val_loader, cfg.training.vis_dir, epoch)
                except Exception as e:
                    print(f"  Visualisation failed: {e}")

    logger.close()
    if distributed:
        dist.destroy_process_group()
    if rank == 0:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()

