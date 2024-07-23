"""
Loss functions for the range-image world model.

  1. FM velocity MSE (primary)           -- lambda_velocity * ||v_pred - v_target||^2
  2. x1-prediction Huber (supervision)   -- lambda_x1 * SmoothL1(x1_pred, x_1)
  3. Depth edge consistency              -- lambda_edge * L_sobel(x1_pred, x_1)
  4. Valid-mask BCE                       -- lambda_valid * BCE(x1_pred_valid, x_1_valid)
  5. Frequency sharpness                 -- lambda_freq * L_freq(x1_pred, x_1)
  6. Temporal consistency (multi-step)   -- lambda_temporal * L_temporal(pred_k, pred_{k+1})
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.range_image_utils import build_projector_from_cfg
from src.data.bev_utils import range_to_bev_occupancy, bev_iou_loss


class DepthEdgeLoss(nn.Module):
    """
    Penalises edge/gradient mismatch between predicted and target range images.
    Uses Sobel filters to extract horizontal and vertical depth gradients,
    then computes L1 difference.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude. Uses circular padding on width."""
        padded = F.pad(img, (1, 1, 0, 0), mode="circular")
        padded = F.pad(padded, (0, 0, 1, 1), mode="constant", value=0)
        gx = F.conv2d(padded, self.sobel_x)
        gy = F.conv2d(padded, self.sobel_y)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grad_pred = self._gradient(pred)
        grad_target = self._gradient(target)
        loss = (grad_pred - grad_target).abs()
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)
        return loss.mean()


class FrequencySharpnessLoss(nn.Module):
    """
    Penalises high-frequency content mismatch using FFT.

    Sharpness is encoded in high-frequency components. Matching the
    spectral energy distribution pushes the model to produce crisp
    depth edges rather than blurry averaged predictions.
    """

    def __init__(self, weight_high_freq: float = 2.0):
        super().__init__()
        self.weight_high_freq = weight_high_freq

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask

        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        _, _, H, W_half = pred_mag.shape
        freq_h = torch.arange(H, device=pred.device).float()
        freq_h = torch.min(freq_h, H - freq_h) / (H / 2)
        freq_w = torch.arange(W_half, device=pred.device).float() / W_half
        freq_grid = torch.sqrt(freq_h[:, None] ** 2 + freq_w[None, :] ** 2).clamp(max=1.0)
        freq_weight = 1.0 + (self.weight_high_freq - 1.0) * freq_grid
        freq_weight = freq_weight.unsqueeze(0).unsqueeze(0)

        loss = (freq_weight * (pred_mag - target_mag).abs()).mean()
        return loss


class EmptySpaceLoss(nn.Module):
    """
    Strong L1 penalty for non-(-1) predictions in GT-empty regions.

    Without this, the model fills empty regions with low-magnitude noise
    that creates dense spurious 3D points when unprojected.
    """

    def forward(
        self,
        pred: torch.Tensor,       # (B, 1, H, W) predicted range image
        target_valid: torch.Tensor,  # (B, 1, H, W) GT valid mask
    ) -> torch.Tensor:
        empty_mask = (1.0 - target_valid)
        deviation = (pred - (-1.0)).abs() * empty_mask
        return deviation.sum() / empty_mask.sum().clamp(min=1)


class RayAwareDepthLoss(nn.Module):
    """
    Penalises depth errors proportionally to metric depth.

    In log-depth encoding, a fixed error epsilon in normalised space
    corresponds to a much larger metric error at far range. Weighting
    by sqrt(metric_depth) gives ~1x at 10 m, ~2x at 40 m, ~2.8x at 80 m.
    """

    def __init__(self, max_depth: float = 80.0):
        super().__init__()
        self.log_max = math.log2(max_depth + 1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        unit = (target + 1.0) * 0.5
        approx_depth = torch.pow(2.0, unit * self.log_max) - 1.0
        depth_weight = torch.sqrt(approx_depth.clamp(min=1.0) / 10.0)
        error = (pred - target).abs() * depth_weight * valid_mask
        denom = (valid_mask * depth_weight).sum().clamp(min=1)
        return error.sum() / denom


class AngularCurvatureLoss(nn.Module):
    """
    Matches azimuth-direction second derivatives between prediction and target.

    Suppresses radial ray-casting streaks while preserving legitimate
    hard boundaries, by penalising curvature mismatch rather than
    absolute smoothness.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_curv = torch.roll(pred, shifts=-1, dims=-1) - 2.0 * pred + torch.roll(pred, shifts=1, dims=-1)
        target_curv = torch.roll(target, shifts=-1, dims=-1) - 2.0 * target + torch.roll(target, shifts=1, dims=-1)
        diff = (pred_curv - target_curv).abs() * valid_mask
        return diff.sum() / valid_mask.sum().clamp(min=1)


class OccupancyDiceTverskyLoss(nn.Module):
    """
    Occupancy-aware overlap loss in range-image space.

    Combines Dice and Tversky losses to reduce both false positives in
    free space and missed occupied pixels around boundaries.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        dice_mix: float = 0.5,
        logit_scale: float = 10.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_mix = dice_mix
        self.logit_scale = logit_scale
        self.smooth = smooth

    def forward(
        self,
        pred_range: torch.Tensor,
        target_range: torch.Tensor,
        valid_threshold: float,
    ) -> torch.Tensor:
        prob = torch.sigmoid((pred_range - valid_threshold) * self.logit_scale)
        target = (target_range > valid_threshold).float()

        dims = (1, 2, 3)
        tp = (prob * target).sum(dim=dims)
        fp = (prob * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - prob) * target).sum(dim=dims)

        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        l_dice = 1.0 - dice.mean()
        l_tversky = 1.0 - tversky.mean()
        return self.dice_mix * l_dice + (1.0 - self.dice_mix) * l_tversky


class LightweightChamferRegularizer(nn.Module):
    """
    Lightweight differentiable Chamfer regulariser.

    Uses continuous metric depth on fixed rays so gradients flow to
    predicted range values. GT occupancy mask defines the region of interest;
    both point sets are aggressively subsampled.
    """

    def __init__(
        self,
        projector,
        max_pred_points: int = 768,
        max_gt_points: int = 768,
        norm_scale_m: float = 50.0,
    ):
        super().__init__()
        self.projector = projector
        self.max_pred_points = int(max_pred_points)
        self.max_gt_points = int(max_gt_points)
        self.norm_scale_m = float(max(norm_scale_m, 1.0))

    def _subsample(self, pts: torch.Tensor, max_points: int) -> torch.Tensor:
        n = pts.shape[0]
        if n <= max_points:
            return pts
        idx = torch.randperm(n, device=pts.device)[:max_points]
        return pts[idx]

    def forward(
        self,
        pred_range: torch.Tensor,
        target_range: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        B = pred_range.shape[0]
        device = pred_range.device
        dirs = self.projector._get_ray_directions(device).float()  # (H, W, 3)

        pred_depth = self.projector.decode_depth(pred_range[:, 0].float().clamp(-1, 1))
        target_depth = self.projector.decode_depth(target_range[:, 0].float().clamp(-1, 1))
        pred_xyz = dirs.unsqueeze(0) * pred_depth.unsqueeze(-1)
        target_xyz = dirs.unsqueeze(0) * target_depth.unsqueeze(-1)

        losses = []
        for b in range(B):
            gt_mask = target_valid[b, 0] > 0.5
            if gt_mask.sum() < 16:
                continue

            gt_pts = target_xyz[b][gt_mask]
            pred_pts = pred_xyz[b][gt_mask]
            if pred_pts.shape[0] < 16:
                pred_pts = pred_xyz[b].reshape(-1, 3)

            gt_pts = self._subsample(gt_pts, self.max_gt_points)
            pred_pts = self._subsample(pred_pts, self.max_pred_points)
            if gt_pts.shape[0] < 8 or pred_pts.shape[0] < 8:
                continue

            p = pred_pts / self.norm_scale_m
            g = gt_pts / self.norm_scale_m
            dist = torch.cdist(p.unsqueeze(0), g.unsqueeze(0)).squeeze(0)
            cd = 0.5 * (dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())
            losses.append(cd)

        if len(losses) == 0:
            return pred_range.new_tensor(0.0)
        return torch.stack(losses).mean().to(dtype=pred_range.dtype)


class RangeWorldModelLoss(nn.Module):
    """
    Combined loss for the range-image world model.

    Takes the output dict from RangeWorldModel.forward_train() and computes:
      total = lambda_v * L_velocity + lambda_x1 * L_x1 + lambda_edge * L_edge
              + lambda_valid * L_valid + lambda_freq * L_freq + ...
    """

    def __init__(self, cfg):
        super().__init__()
        tcfg = cfg.training
        self.lambda_v = tcfg.lambda_velocity
        self.lambda_x1 = tcfg.lambda_x1
        self.lambda_edge = tcfg.lambda_edge
        self.lambda_valid = tcfg.lambda_valid
        self.lambda_freq = getattr(tcfg, "lambda_freq", 0.02)
        self.lambda_empty = getattr(tcfg, "lambda_empty", 0.5)
        self.lambda_ray = getattr(tcfg, "lambda_ray", 0.3)
        self.lambda_angular = getattr(tcfg, "lambda_angular", 0.05)
        self.lambda_occupancy = getattr(tcfg, "lambda_occupancy", 0.1)
        self.lambda_chamfer = getattr(tcfg, "lambda_chamfer", 0.02)
        self.lambda_bev_iou = getattr(tcfg, "lambda_bev_iou", 0.05)
        self.lambda_occ_flow = getattr(tcfg, "lambda_occ_flow", 0.0)
        self.velocity_t_weight = getattr(tcfg, "velocity_t_weight", 0.0)

        self.edge_loss = DepthEdgeLoss()
        self.freq_loss = FrequencySharpnessLoss(weight_high_freq=2.0)
        self.empty_loss = EmptySpaceLoss()
        self.ray_loss = RayAwareDepthLoss(
            max_depth=getattr(tcfg, "ray_depth_max_m", 80.0),
        )
        self.angular_loss = AngularCurvatureLoss()
        self.occupancy_loss = OccupancyDiceTverskyLoss(
            alpha=getattr(tcfg, "occ_tversky_alpha", 0.7),
            beta=getattr(tcfg, "occ_tversky_beta", 0.3),
            dice_mix=getattr(tcfg, "occ_dice_mix", 0.5),
        )
        self.chamfer_reg = LightweightChamferRegularizer(
            projector=build_projector_from_cfg(cfg),
            max_pred_points=getattr(tcfg, "chamfer_max_pred_points", 768),
            max_gt_points=getattr(tcfg, "chamfer_max_gt_points", 768),
            norm_scale_m=getattr(tcfg, "chamfer_norm_scale_m", 50.0),
        )
        self.chamfer_start_epoch = getattr(tcfg, "chamfer_start_epoch", 6)
        self.chamfer_every_n_steps = max(int(getattr(tcfg, "chamfer_every_n_steps", 6)), 1)
        self.chamfer_only_first_step = bool(getattr(tcfg, "chamfer_only_first_step", False))
        self._step_counter = 0
        mcfg = cfg.model
        self._bev_ri_H = getattr(mcfg, "ri_H", 32)
        self._bev_ri_W = getattr(mcfg, "ri_W", 1024)
        self._bev_size = getattr(mcfg, "bev_size", 64)
        self._bev_range_m = getattr(mcfg, "bev_range_m", 64.0)
        self._bev_max_depth = getattr(tcfg, "ray_depth_max_m", 80.0)
        valid_depth_m = max(
            float(getattr(cfg.model, "validity_relaxed_depth_m", tcfg.ray_depth_min_m)),
            float(tcfg.ray_depth_min_m),
        )
        log_max = math.log2(float(tcfg.ray_depth_max_m) + 1.0)
        self.valid_threshold = (math.log2(valid_depth_m + 1.0) / log_max) * 2.0 - 1.0

    def _valid_focal_loss(
        self,
        pred_range: torch.Tensor,
        target_range: torch.Tensor,
    ) -> torch.Tensor:
        pred_valid_logit = (pred_range - self.valid_threshold) * 10.0
        target_is_valid = (target_range > self.valid_threshold).float()

        gamma = 2.0
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            pred_valid_logit, target_is_valid, reduction="none",
        )
        pred_prob = torch.sigmoid(pred_valid_logit)
        p_t = pred_prob * target_is_valid + (1 - pred_prob) * (1 - target_is_valid)
        focal_weight = (1 - p_t) ** gamma
        return (focal_weight * bce_per_pixel).mean()

    def forward(self, output: dict, epoch: int = 0) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            output: dict from RangeWorldModel.forward_train()
            epoch:  current epoch

        Returns:
            total_loss: scalar
            loss_dict:  per-component losses for logging
        """
        v_pred = output["v_pred"]
        v_target = output["v_target"]
        x1_pred = output["x1_pred"]
        x_1 = output["x_1"]
        target_valid = output["target_valid"]
        self._step_counter += 1

        weight = 1.0 + target_valid
        if self.velocity_t_weight > 0 and "t" in output:
            t = output["t"].view(-1, 1, 1, 1)
            weight = weight * (1.0 + self.velocity_t_weight * t)
        l_velocity = (weight * (v_pred - v_target) ** 2).mean()

        l_x1 = F.smooth_l1_loss(
            x1_pred * target_valid,
            x_1 * target_valid,
            beta=0.05,
        )

        l_edge = self.edge_loss(x1_pred, x_1, target_valid)
        l_valid = self._valid_focal_loss(x1_pred, x_1)
        l_freq = self.freq_loss(x1_pred, x_1, target_valid)
        l_empty = self.empty_loss(x1_pred, target_valid)
        l_ray = self.ray_loss(x1_pred, x_1, target_valid)
        l_angular = self.angular_loss(x1_pred, x_1, target_valid)
        l_occupancy = self.occupancy_loss(x1_pred, x_1, self.valid_threshold)

        apply_chamfer = (
            self.lambda_chamfer > 0
            and epoch >= self.chamfer_start_epoch
            and (self._step_counter % self.chamfer_every_n_steps == 0)
        )
        if self.chamfer_only_first_step:
            apply_chamfer = apply_chamfer and int(output.get("step_idx", 0)) == 0
        if apply_chamfer:
            l_chamfer = self.chamfer_reg(x1_pred, x_1, target_valid)
        else:
            l_chamfer = x1_pred.new_tensor(0.0)

        l_bev_iou = x1_pred.new_tensor(0.0)
        if self.lambda_bev_iou > 0:
            pred_valid_bev = (x1_pred > self.valid_threshold).float()
            bev_pred, _ = range_to_bev_occupancy(
                x1_pred, pred_valid_bev,
                ri_H=self._bev_ri_H, ri_W=self._bev_ri_W,
                max_depth=self._bev_max_depth, bev_size=self._bev_size, bev_range_m=self._bev_range_m,
            )
            bev_gt, bev_gt_valid = range_to_bev_occupancy(
                x_1, target_valid,
                ri_H=self._bev_ri_H, ri_W=self._bev_ri_W,
                max_depth=self._bev_max_depth, bev_size=self._bev_size, bev_range_m=self._bev_range_m,
            )
            if bev_gt_valid.sum() > 10:
                l_bev_iou = bev_iou_loss(bev_pred, bev_gt, bev_gt_valid)

        l_occ_flow = x1_pred.new_tensor(0.0)
        if self.lambda_occ_flow > 0 and "occ_logit" in output:
            occ_logit = output["occ_logit"]
            occ_target = (x_1 > self.valid_threshold).float()
            l_occ_flow = F.binary_cross_entropy_with_logits(occ_logit, occ_target)

        total = (
            self.lambda_v * l_velocity
            + self.lambda_x1 * l_x1
            + self.lambda_edge * l_edge
            + self.lambda_valid * l_valid
            + self.lambda_freq * l_freq
            + self.lambda_empty * l_empty
            + self.lambda_ray * l_ray
            + self.lambda_angular * l_angular
            + self.lambda_occupancy * l_occupancy
            + self.lambda_chamfer * l_chamfer
            + self.lambda_bev_iou * l_bev_iou
            + self.lambda_occ_flow * l_occ_flow
        )

        loss_dict = {
            "total": total.item(),
            "velocity": l_velocity.item(),
            "x1_huber": l_x1.item(),
            "edge": l_edge.item(),
            "valid_bce": l_valid.item(),
            "freq": l_freq.item(),
            "empty": l_empty.item(),
            "ray": l_ray.item(),
            "angular": l_angular.item(),
            "occupancy": l_occupancy.item(),
            "chamfer_reg": l_chamfer.item(),
            "bev_iou": l_bev_iou.item(),
            "occ_flow": l_occ_flow.item(),
        }

        return total, loss_dict


# =====================================================================
#  Temporal Consistency Loss (for multi-step training)
# =====================================================================

class TemporalConsistencyLoss(nn.Module):
    """
    Penalises temporal inconsistency across consecutive predictions
    in multi-step autoregressive rollout.

    Two components:
      (a) Depth smoothness: predicted range images at consecutive steps
          should change smoothly (penalise jitter / flickering).
      (b) Structure preservation: the overall valid/invalid pattern should
          remain consistent.
    """

    def __init__(self, valid_threshold: float):
        super().__init__()
        self.valid_threshold = valid_threshold

    def forward(
        self,
        pred_k: torch.Tensor,
        pred_k1: torch.Tensor,
        gt_k: torch.Tensor,
        gt_k1: torch.Tensor,
        valid_k: torch.Tensor,
        valid_k1: torch.Tensor,
    ) -> torch.Tensor:
        gt_delta = gt_k1 - gt_k
        pred_delta = pred_k1 - pred_k

        joint_valid = valid_k * valid_k1
        delta_diff = (pred_delta - gt_delta).abs()
        delta_loss = (delta_diff * joint_valid).sum() / joint_valid.sum().clamp(min=1)

        gt_occ_change = (gt_k1 > self.valid_threshold).float() - (gt_k > self.valid_threshold).float()
        pred_occ_change = (pred_k1 > self.valid_threshold).float() - (pred_k > self.valid_threshold).float()
        occ_loss = (pred_occ_change - gt_occ_change).abs().mean()

        return delta_loss + 0.5 * occ_loss


class MultiStepLoss(nn.Module):
    """
    Wraps the single-step RangeWorldModelLoss for multi-step training.

    For K-step scheduled-sampling training, computes:
      - Single-step loss for each step (with decay for later steps)
      - Temporal consistency between consecutive predictions
      - Higher penalty for steps where scheduled sampling was used
    """

    def __init__(self, cfg):
        super().__init__()
        self.single_step_loss = RangeWorldModelLoss(cfg)
        self.temporal_loss = TemporalConsistencyLoss(self.single_step_loss.valid_threshold)
        self.lambda_temporal = getattr(cfg.training, "lambda_temporal", 0.1)
        self.step_decay = getattr(cfg.training, "multistep_decay", 0.8)
        self.own_pred_boost = getattr(cfg.training, "multistep_own_pred_boost", 1.15)
        self.future_weight_power = getattr(cfg.training, "multistep_future_weight_power", 0.0)
        self.t3_t6_boost = getattr(cfg.training, "multistep_t3_t6_boost", 1.0)
        self.ss_start_epoch = getattr(cfg.training, "ss_start_epoch", 0)
        self.ramp_epochs = max(getattr(cfg.training, "multistep_ramp_epochs", 12), 1)
        self.lambda_jump = getattr(cfg.training, "lambda_jump", 0.0)
        self.lambda_cycle = getattr(cfg.training, "lambda_cycle", 0.0)
        self.projector = build_projector_from_cfg(cfg)
        self._valid_threshold = self.single_step_loss.valid_threshold

    def _cycle_consistency_loss(
        self,
        pred_range: torch.Tensor,
        frame_0_pc: torch.Tensor,
        frame_0_mask: torch.Tensor,
        rel_poses: list,
    ) -> Optional[torch.Tensor]:
        """Warp pred t+K back to t, compare with frame 0 (cycle consistency)."""
        K = len(rel_poses)
        if K == 0:
            return None
        rel_0_K = rel_poses[0].clone()
        for k in range(1, K):
            rel_0_K = torch.bmm(rel_0_K, rel_poses[k])
        R = rel_0_K[:, :3, :3]
        t_vec = rel_0_K[:, :3, 3:4]
        R_inv = R.transpose(1, 2)
        t_inv = -torch.bmm(R_inv, t_vec)
        rel_inv = rel_0_K.clone()
        rel_inv[:, :3, :3] = R_inv
        rel_inv[:, :3, 3:4] = t_inv

        with torch.no_grad():
            valid = (pred_range > self._valid_threshold).float()
            pred_pts, pred_pt_mask = self.projector.unproject(pred_range, valid)
            warped_ri, warped_valid = self.projector.warp_range_image(pred_pts, pred_pt_mask, rel_inv)
            gt_ri, gt_valid = self.projector.project(frame_0_pc, frame_0_mask)
        joint = warped_valid * gt_valid
        if joint.sum() < 10:
            return None
        diff = (warped_ri - gt_ri).abs() * joint
        return (diff.sum() / joint.sum().clamp(min=1)).mean()

    def forward(
        self,
        outputs: list[dict],
        epoch: int = 0,
        frames_pc: Optional[list] = None,
        frames_mask: Optional[list] = None,
        rel_poses: Optional[list] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: list of K output dicts from forward_train_scheduled()
            epoch:   current epoch

        Returns:
            total_loss: scalar (sum over all steps)
            loss_dict:  aggregated metrics
        """
        step_outputs = [o for o in outputs if not o.get("is_jump", False)]
        jump_outputs = [o for o in outputs if o.get("is_jump", False)]
        K = len(step_outputs)
        if K == 0:
            step_outputs = outputs
            jump_outputs = []
            K = len(step_outputs)

        total = torch.tensor(0.0, device=step_outputs[0]["v_pred"].device)
        agg_dict = {}

        step_terms = []
        step_weights = []
        for k, out in enumerate(step_outputs):
            w = self.step_decay ** k
            if K > 1 and self.future_weight_power > 0:
                horizon_ratio = k / float(K - 1)
                w *= (1.0 + horizon_ratio) ** self.future_weight_power

            if k >= 3 and self.t3_t6_boost > 1.0:
                w *= self.t3_t6_boost

            if out.get("used_own_prediction", False):
                w *= self.own_pred_boost

            step_loss, step_dict = self.single_step_loss(out, epoch)
            step_terms.append(step_loss)
            step_weights.append(w)

            for key, val in step_dict.items():
                agg_dict[f"step{k}/{key}"] = val

        weight_norm = max(sum(step_weights), 1e-8)
        for step_loss, w in zip(step_terms, step_weights):
            total = total + (w / weight_norm) * step_loss

        l_temporal = torch.tensor(0.0, device=total.device)
        temporal_count = 0
        for k in range(K - 1):
            tc = self.temporal_loss(
                pred_k=step_outputs[k]["x1_pred"],
                pred_k1=step_outputs[k + 1]["x1_pred"],
                gt_k=step_outputs[k]["x_1"],
                gt_k1=step_outputs[k + 1]["x_1"],
                valid_k=step_outputs[k]["target_valid"],
                valid_k1=step_outputs[k + 1]["target_valid"],
            )
            l_temporal = l_temporal + tc
            temporal_count += 1

        if temporal_count > 0:
            l_temporal = l_temporal / temporal_count
            total = total + self.lambda_temporal * l_temporal

        if self.lambda_jump > 0 and jump_outputs:
            l_jump, jump_dict = self.single_step_loss(jump_outputs[0], epoch)
            total = total + self.lambda_jump * l_jump
            for key, val in jump_dict.items():
                agg_dict[f"jump/{key}"] = val

        if (self.lambda_cycle > 0 and K >= 2 and frames_pc is not None
                and frames_mask is not None and rel_poses is not None):
            l_cycle = self._cycle_consistency_loss(
                pred_range=step_outputs[K - 1]["x1_pred"],
                frame_0_pc=frames_pc[0],
                frame_0_mask=frames_mask[0],
                rel_poses=rel_poses[:K],
            )
            if l_cycle is not None:
                total = total + self.lambda_cycle * l_cycle
                agg_dict["cycle"] = l_cycle.item()

        ramp_weight = 1.0
        if epoch >= self.ss_start_epoch:
            ramp_weight = min(
                (epoch - self.ss_start_epoch + 1) / float(self.ramp_epochs),
                1.0,
            )
        total = total * ramp_weight

        agg_dict["temporal"] = l_temporal.item()
        agg_dict["ramp_weight"] = float(ramp_weight)
        agg_dict["step_weight_norm"] = float(weight_norm)
        agg_dict["future_weight_power"] = float(self.future_weight_power)
        agg_dict["total"] = total.item()

        return total, agg_dict

