"""
Evaluate baseline models on single-step and multi-step forecasting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.default import Config
from src.data.temporal_dataset import build_datasets
from src.baselines.models import build_baseline


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor, subsample: int = 10000):
    if pred.shape[0] > subsample:
        pred = pred[torch.randperm(pred.shape[0], device=pred.device)[:subsample]]
    if gt.shape[0] > subsample:
        gt = gt[torch.randperm(gt.shape[0], device=gt.device)[:subsample]]
    dist = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    return 0.5 * (dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())


def bev_iou(pred: torch.Tensor, gt: torch.Tensor, grid_size: int = 256, range_m: float = 64.0):
    def to_bev(pts):
        x = ((pts[:, 0] + range_m) / (2 * range_m) * grid_size).long().clamp(0, grid_size - 1)
        y = ((pts[:, 1] + range_m) / (2 * range_m) * grid_size).long().clamp(0, grid_size - 1)
        bev = torch.zeros(grid_size, grid_size, device=pts.device)
        bev[x, y] = 1.0
        return bev

    bp = to_bev(pred)
    bg = to_bev(gt)
    inter = (bp * bg).sum()
    union = ((bp + bg) > 0).float().sum().clamp(min=1)
    return (inter / union).item()


def ri_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    diff = (pred - gt) * mask
    mse = (diff ** 2).sum() / mask.sum().clamp(min=1)
    return mse.sqrt().item()


def ri_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    mse = ((pred - gt) * mask).pow(2).sum() / mask.sum().clamp(min=1)
    if mse < 1e-10:
        return 50.0
    return (10 * torch.log10(4.0 / mse)).item()


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metrics = defaultdict(list)
        valid_depth_m = max(
            float(getattr(cfg.model, "validity_relaxed_depth_m", cfg.training.ray_depth_min_m)),
            float(cfg.training.ray_depth_min_m),
        )
        log_max = math.log2(float(cfg.training.ray_depth_max_m) + 1.0)
        self.valid_threshold = (math.log2(valid_depth_m + 1.0) / log_max) * 2.0 - 1.0

    def update(self, pred_range, gt_range, gt_valid, pred_points, pred_mask, gt_points, gt_mask, prefix: str = ""):
        B = pred_range.shape[0]
        for b in range(B):
            m = gt_valid[b]
            self.metrics[f"{prefix}ri_rmse"].append(ri_rmse(pred_range[b], gt_range[b], m))
            self.metrics[f"{prefix}ri_psnr"].append(ri_psnr(pred_range[b], gt_range[b], m))

            pred_valid_map = pred_range[b] > self.valid_threshold
            gt_invalid = gt_valid[b] <= 0.5
            fp_rate = (pred_valid_map & gt_invalid).float().sum() / gt_invalid.float().sum().clamp(min=1)
            self.metrics[f"{prefix}free_fp_rate"].append(fp_rate.item())

            pp = pred_points[b][pred_mask[b]]
            gp = gt_points[b][gt_mask[b]]
            if gp.shape[0] > 0:
                self.metrics[f"{prefix}pred_gt_count_ratio"].append(float(pp.shape[0]) / float(gp.shape[0]))
            eligible = pp.shape[0] > 100 and gp.shape[0] > 100
            self.metrics[f"{prefix}cd_eval_eligible"].append(float(eligible))
            if eligible:
                self.metrics[f"{prefix}chamfer_dist"].append(
                    chamfer_distance(pp, gp, self.cfg.evaluation.chamfer_subsample).item(),
                )
                self.metrics[f"{prefix}bev_iou"].append(
                    bev_iou(pp, gp, self.cfg.evaluation.bev_eval_size, self.cfg.evaluation.bev_eval_range_m),
                )

    def compute(self):
        return {k: float(np.mean(v)) for k, v in self.metrics.items() if len(v) > 0}


def move_batch(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def evaluate_single(model, val_loader, cfg, max_batches: int = 50):
    evaluator = Evaluator(cfg)
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, next(model.parameters()).device)
        out = model.forward_inference(
            current_pc=batch["current_pc"],
            rel_pose=batch["relative_pose"],
            action=batch["action_features"],
            current_mask=batch["current_mask"],
            num_steps=cfg.evaluation.eval_inference_steps,
        )
        gt_ri, gt_valid = model.projector.project(batch["target_pc"], batch["target_mask"])
        gt_points, gt_pt_mask = model.projector.unproject(gt_ri, gt_valid)
        evaluator.update(
            pred_range=out["pred_range"],
            gt_range=gt_ri,
            gt_valid=gt_valid,
            pred_points=out["pred_points"],
            pred_mask=out["pred_mask"],
            gt_points=gt_points,
            gt_mask=gt_pt_mask,
        )
    return evaluator.compute()


@torch.no_grad()
def evaluate_multistep(model, seq_loader, cfg, max_batches: int = 30):
    evaluator = Evaluator(cfg)
    K = cfg.evaluation.eval_rollout_steps
    for i, batch in enumerate(seq_loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, next(model.parameters()).device)
        K_actual = min(K, batch["rel_poses"].shape[1])
        rel_poses = [batch["rel_poses"][:, k] for k in range(K_actual)]
        actions = [batch["actions"][:, k] for k in range(K_actual)]
        rollout = model.forward_rollout_range_space(
            current_pc=batch["frames_pc"][:, 0],
            current_mask=batch["frames_mask"][:, 0],
            rel_poses=rel_poses,
            actions=actions,
            num_steps=cfg.evaluation.eval_inference_steps,
        )
        for k in range(K_actual):
            gt_pc = batch["frames_pc"][:, k + 1]
            gt_mask = batch["frames_mask"][:, k + 1]
            gt_ri, gt_valid = model.projector.project(gt_pc, gt_mask)
            gt_points, gt_pt_mask = model.projector.unproject(gt_ri, gt_valid)
            out = rollout[k]
            evaluator.update(
                pred_range=out["pred_range"],
                gt_range=gt_ri,
                gt_valid=gt_valid,
                pred_points=out["pred_points"],
                pred_mask=out["pred_mask"],
                gt_points=gt_points,
                gt_mask=gt_pt_mask,
                prefix=f"t+{k+1}/",
            )
    return evaluator.compute()


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument("checkpoint", type=str, help="Path to baseline checkpoint")
    parser.add_argument("--model-type", type=str, default=None, choices=["lidarcrafter_like", "occworld_style"])
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--single-step-only", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = Config.from_dict(ckpt["config"])
    cfg.evaluation.eval_rollout_steps = args.rollout_steps

    model_type = args.model_type or ckpt.get("model_type")
    if model_type is None:
        raise ValueError("model type is missing. Provide --model-type.")
    model = build_baseline(model_type, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=cfg.data.version, dataroot=cfg.data.data_root, verbose=True)
    if args.single_step_only:
        _, val_ds = build_datasets(
            nusc,
            max_points=cfg.data.max_points,
            train_ratio=cfg.data.train_split_ratio,
            prediction_horizon=cfg.data.prediction_horizon,
            seed=cfg.training.seed,
        )
        val_seq_ds = None
    else:
        _, val_ds, _, val_seq_ds = build_datasets(
            nusc,
            max_points=cfg.data.max_points,
            train_ratio=cfg.data.train_split_ratio,
            prediction_horizon=cfg.data.prediction_horizon,
            seed=cfg.training.seed,
            sequence_length=args.rollout_steps,
        )
    del nusc

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=max(1, cfg.training.batch_size), shuffle=False, num_workers=4, pin_memory=True,
    )

    print(f"Running baseline evaluation: {model_type}")
    metrics = evaluate_single(model, val_loader, cfg, max_batches=args.max_batches)

    if val_seq_ds is not None:
        seq_loader = torch.utils.data.DataLoader(
            val_seq_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True,
        )
        ms = evaluate_multistep(model, seq_loader, cfg, max_batches=min(args.max_batches, 30))
        metrics.update(ms)

    root_dir = Path(__file__).resolve().parents[2]
    out_path = args.output
    if out_path is None:
        out_path = str(root_dir / "baselines" / "results" / f"{model_type}_eval.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Results:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s}: {v:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
