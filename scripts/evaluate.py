"""
Evaluation script for the range-image world model.

Computes both range-image-space and 3D point-cloud metrics:
  - Range image: RMSE, PSNR
  - 3D: Chamfer Distance, BEV IoU
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.default import Config
from src.model.world_model import build_world_model
from src.data.range_image_utils import RangeImageProjector, build_projector_from_cfg


# =====================================================================
#  Metrics
# =====================================================================

def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor, subsample: int = 10000):
    """
    Chamfer distance between two point clouds.
    pred, gt: (N, 3) and (M, 3)
    Returns: scalar (mean of both directions)
    """
    if pred.shape[0] > subsample:
        idx = torch.randperm(pred.shape[0])[:subsample]
        pred = pred[idx]
    if gt.shape[0] > subsample:
        idx = torch.randperm(gt.shape[0])[:subsample]
        gt = gt[idx]

    dist_p2g = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
    d_p2g = dist_p2g.min(dim=1).values.mean()
    d_g2p = dist_p2g.min(dim=0).values.mean()
    return (d_p2g + d_g2p) / 2


def bev_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    grid_size: int = 256,
    range_m: float = 64.0,
):
    """BEV occupancy IoU between two point clouds."""
    def to_bev(pts):
        x = ((pts[:, 0] + range_m) / (2 * range_m) * grid_size).long().clamp(0, grid_size - 1)
        y = ((pts[:, 1] + range_m) / (2 * range_m) * grid_size).long().clamp(0, grid_size - 1)
        bev = torch.zeros(grid_size, grid_size, device=pts.device)
        bev[x, y] = 1.0
        return bev

    bev_pred = to_bev(pred)
    bev_gt = to_bev(gt)
    inter = (bev_pred * bev_gt).sum()
    union = ((bev_pred + bev_gt) > 0).float().sum()
    return (inter / union.clamp(min=1)).item()


def range_image_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    """RMSE on valid range image pixels."""
    diff = (pred - gt) * mask
    mse = (diff ** 2).sum() / mask.sum().clamp(min=1)
    return mse.sqrt().item()


def range_image_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    """PSNR on valid range image pixels (range [-1,1] -> max_val=2)."""
    mse = ((pred - gt) * mask).pow(2).sum() / mask.sum().clamp(min=1)
    if mse < 1e-10:
        return 50.0
    return (10 * torch.log10(4.0 / mse)).item()  # max_val^2 = 4


# =====================================================================
#  Evaluator
# =====================================================================

class WorldModelEvaluator:
    """Accumulates metrics across batches."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.projector = build_projector_from_cfg(cfg)
        self.metrics = defaultdict(list)
        valid_depth_m = max(
            float(getattr(cfg.model, "validity_relaxed_depth_m", cfg.training.ray_depth_min_m)),
            float(cfg.training.ray_depth_min_m),
        )
        log_max = math.log2(float(cfg.training.ray_depth_max_m) + 1.0)
        self.valid_threshold = (math.log2(valid_depth_m + 1.0) / log_max) * 2.0 - 1.0

    def update(self, pred_range, gt_range, gt_valid, pred_points, pred_mask, gt_points, gt_mask,
               step_prefix: str = ""):
        """Add one batch of predictions. step_prefix: e.g. 'step2/' for multi-step."""
        B = pred_range.shape[0]

        for b in range(B):
            m = gt_valid[b]
            self.metrics[f"{step_prefix}ri_rmse"].append(range_image_rmse(pred_range[b], gt_range[b], m))
            self.metrics[f"{step_prefix}ri_psnr"].append(range_image_psnr(pred_range[b], gt_range[b], m))
            pred_valid_map = pred_range[b] > self.valid_threshold
            gt_invalid = gt_valid[b] <= 0.5
            fp_rate = (
                (pred_valid_map & gt_invalid).float().sum()
                / gt_invalid.float().sum().clamp(min=1)
            )
            self.metrics[f"{step_prefix}free_fp_rate"].append(fp_rate.item())

            pp = pred_points[b][pred_mask[b]]
            gp = gt_points[b][gt_mask[b]]
            if gp.shape[0] > 0:
                self.metrics[f"{step_prefix}pred_gt_count_ratio"].append(
                    float(pp.shape[0]) / float(gp.shape[0]),
                )
            eligible = pp.shape[0] > 100 and gp.shape[0] > 100
            self.metrics[f"{step_prefix}cd_eval_eligible"].append(float(eligible))
            if eligible:
                cd = chamfer_distance(pp, gp, self.cfg.evaluation.chamfer_subsample)
                self.metrics[f"{step_prefix}chamfer_dist"].append(cd.item())
                self.metrics[f"{step_prefix}bev_iou"].append(
                    bev_iou(pp, gp, self.cfg.evaluation.bev_eval_size, self.cfg.evaluation.bev_eval_range_m)
                )

    def compute(self) -> dict:
        """Returns averaged metrics."""
        return {k: np.mean(v) for k, v in self.metrics.items() if len(v) > 0}


# =====================================================================
#  Main evaluation
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    val_loader,
    cfg,
    projector,
    max_batches: int = 50,
    rank: int = 0,
) -> dict:
    """Run evaluation on validation set."""
    model.eval()
    evaluator = WorldModelEvaluator(cfg)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        raw_model = model.module if hasattr(model, "module") else model
        inf_out = raw_model.forward_inference(
            current_pc=batch["current_pc"],
            rel_pose=batch["relative_pose"],
            action=batch["action_features"],
            current_mask=batch["current_mask"],
            num_steps=cfg.evaluation.eval_inference_steps,
        )

        gt_ri, gt_valid = projector.project(batch["target_pc"], batch["target_mask"])
        gt_points, gt_pt_mask = projector.unproject(gt_ri, gt_valid)

        evaluator.update(
            pred_range=inf_out["pred_range"],
            gt_range=gt_ri,
            gt_valid=gt_valid,
            pred_points=inf_out["pred_points"],
            pred_mask=inf_out["pred_mask"],
            gt_points=gt_points,
            gt_mask=gt_pt_mask,
        )

        if rank == 0 and (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{min(max_batches, len(val_loader))} batches")

    return evaluator.compute()


@torch.no_grad()
def evaluate_multistep(
    model,
    seq_loader,
    cfg,
    projector,
    max_batches: int = 50,
    rank: int = 0,
) -> dict:
    """
    Multi-step rollout evaluation (t+1 through t+K).

    Uses the TemporalSequenceDataset which provides K+1 consecutive frames.
    Runs autoregressive rollout and evaluates each step independently.
    """
    model.eval()
    evaluator = WorldModelEvaluator(cfg)
    K = getattr(cfg.evaluation, "eval_rollout_steps", 6)

    for i, batch in enumerate(seq_loader):
        if i >= max_batches:
            break

        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        raw_model = model.module if hasattr(model, "module") else model
        B = batch["frames_pc"].shape[0]
        K_actual = min(K, batch["rel_poses"].shape[1])

        rel_poses = [batch["rel_poses"][:, k] for k in range(K_actual)]
        actions = [batch["actions"][:, k] for k in range(K_actual)]

        rollout_results = raw_model.forward_rollout_range_space(
            current_pc=batch["frames_pc"][:, 0],
            current_mask=batch["frames_mask"][:, 0],
            rel_poses=rel_poses,
            actions=actions,
            num_steps=cfg.evaluation.eval_inference_steps,
            use_midpoint=True,
        )

        for k in range(K_actual):
            gt_pc = batch["frames_pc"][:, k + 1]
            gt_mask = batch["frames_mask"][:, k + 1]
            gt_ri, gt_valid = projector.project(gt_pc, gt_mask)
            gt_points, gt_pt_mask = projector.unproject(gt_ri, gt_valid)

            step_out = rollout_results[k]
            evaluator.update(
                pred_range=step_out["pred_range"],
                gt_range=gt_ri,
                gt_valid=gt_valid,
                pred_points=step_out["pred_points"],
                pred_mask=step_out["pred_mask"],
                gt_points=gt_points,
                gt_mask=gt_pt_mask,
                step_prefix=f"t+{k+1}/",
            )

        if rank == 0 and (i + 1) % 10 == 0:
            print(f"  [multi-step] Evaluated {i+1}/{min(max_batches, len(seq_loader))} batches")

    return evaluator.compute()


def main():
    parser = argparse.ArgumentParser(description="Evaluate range-image world model")
    parser.add_argument("checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--output", type=str, default="./results/eval_results.json")
    parser.add_argument("--single-step-only", action="store_true", help="Skip multi-step evaluation")
    parser.add_argument("--rollout-steps", type=int, default=6, help="Number of rollout steps for multi-step eval")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])

    model = build_world_model(cfg).cuda()
    model.load_state_dict(ckpt["model_state_dict"])

    if "ema_shadow" in ckpt:
        for name, p in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                p.data.copy_(ckpt["ema_shadow"][name])

    from nuscenes.nuscenes import NuScenes
    from src.data.temporal_dataset import build_datasets

    nusc = NuScenes(version=cfg.data.version, dataroot=cfg.data.data_root, verbose=True)

    cfg.evaluation.eval_rollout_steps = args.rollout_steps

    if not args.single_step_only:
        result = build_datasets(
            nusc, max_points=cfg.data.max_points, seed=cfg.training.seed,
            sequence_length=args.rollout_steps,
        )
        _, val_ds, _, val_seq_ds = result
    else:
        _, val_ds = build_datasets(
            nusc, max_points=cfg.data.max_points, seed=cfg.training.seed,
        )
        val_seq_ds = None
    del nusc

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    projector = build_projector_from_cfg(cfg)

    print("Running single-step evaluation...")
    metrics = evaluate(model, val_loader, cfg, projector, args.max_batches)

    if val_seq_ds is not None:
        print(f"\nRunning multi-step evaluation (t+1 to t+{args.rollout_steps})...")
        seq_loader = torch.utils.data.DataLoader(
            val_seq_ds, batch_size=2, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        ms_metrics = evaluate_multistep(
            model, seq_loader, cfg, projector,
            max_batches=min(args.max_batches, 30),
        )
        metrics.update(ms_metrics)

    import json
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s}: {v:.4f}")


if __name__ == "__main__":
    main()
