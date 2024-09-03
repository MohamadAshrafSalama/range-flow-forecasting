"""
Baseline models for LiDAR forecasting comparisons.

Implemented baselines:
- LiDARCrafter-like conditional diffusion on range images
- OccWorld-style occupancy-aware deterministic UNet
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.range_image_utils import build_projector_from_cfg
from src.model.world_model import RangeUNet, ActionEncoder


def _depth_to_normalised(depth_m: float, max_depth_m: float) -> float:
    depth_m = max(float(depth_m), 1e-6)
    log_max = math.log2(float(max_depth_m) + 1.0)
    return (math.log2(depth_m + 1.0) / log_max) * 2.0 - 1.0


def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class BaselineBase(nn.Module):
    """Common utilities shared by all baselines."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.projector = build_projector_from_cfg(cfg)
        self.num_inference_steps = getattr(cfg.evaluation, "eval_inference_steps", 20)
        self.valid_threshold = _depth_to_normalised(
            getattr(cfg.model, "validity_relaxed_depth_m", cfg.training.ray_depth_min_m),
            cfg.training.ray_depth_max_m,
        )

    def _prepare_conditions(
        self,
        current_pc: torch.Tensor,
        current_mask: torch.Tensor,
        rel_pose: torch.Tensor,
    ):
        return self.projector.warp_range_image(current_pc, current_mask, rel_pose)

    def _pred_valid_from_depth(
        self,
        pred_range: torch.Tensor,
        prior_valid: Optional[torch.Tensor] = None,
        occ_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        depth_valid = pred_range > self.valid_threshold
        if prior_valid is not None:
            depth_valid = depth_valid | ((prior_valid > 0.5) & (pred_range > (self.valid_threshold - 0.08)))
        if occ_prob is not None:
            depth_valid = depth_valid & (occ_prob > 0.45)
        return depth_valid.float()

    def _build_emb(self, unet: RangeUNet, action_encoder: Optional[ActionEncoder], t: torch.Tensor, action: torch.Tensor):
        emb = unet.time_embed(t)
        if action_encoder is not None:
            emb = emb + action_encoder(action)
        return emb

    @torch.no_grad()
    def forward_rollout_range_space(
        self,
        current_pc: torch.Tensor,
        current_mask: torch.Tensor,
        rel_poses: list,
        actions: list,
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
    ) -> list:
        K = len(rel_poses)
        assert len(actions) == K
        outputs = []
        pc = current_pc
        mask = current_mask
        for k in range(K):
            out = self.forward_inference(
                current_pc=pc,
                rel_pose=rel_poses[k],
                action=actions[k],
                current_mask=mask,
                num_steps=num_steps,
                use_midpoint=use_midpoint,
            )
            outputs.append(out)
            pc = out["pred_points"]
            mask = out["pred_mask"]
        return outputs


class LiDARCrafterLikeDiffusionBaseline(BaselineBase):
    """Conditional diffusion baseline in range-image space."""

    def __init__(self, cfg):
        super().__init__(cfg)
        mcfg = cfg.model
        emb_dim = mcfg.base_channels * 4
        self.unet = RangeUNet(
            in_channels=2,
            out_channels=1,
            base_channels=mcfg.base_channels,
            channel_mult=mcfg.channel_mult,
            num_res_blocks=mcfg.num_res_blocks,
            attention_resolutions=mcfg.attention_resolutions,
            num_heads=mcfg.num_heads,
            emb_dim=emb_dim,
            dropout=mcfg.dropout,
            use_circular=mcfg.use_circular_pad,
            use_activation_checkpointing=getattr(mcfg, "use_activation_checkpointing", False),
        )
        self.action_encoder = ActionEncoder(mcfg.action_dim, emb_dim) if mcfg.use_action_conditioning else None

        self.train_timesteps = 1000
        betas = _linear_beta_schedule(self.train_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def _q_sample(self, x_start: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor):
        a_bar = self.alpha_bars[t_idx].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x_start + torch.sqrt(1.0 - a_bar) * noise

    def training_step(self, batch: dict, epoch: int = 0):
        x_cond, x_cond_valid = self._prepare_conditions(
            batch["current_pc"], batch["current_mask"], batch["relative_pose"],
        )
        x_target, target_valid = self.projector.project(
            batch["target_pc"], batch["target_mask"],
        )
        B = x_target.shape[0]
        device = x_target.device

        t_idx = torch.randint(0, self.train_timesteps, (B,), device=device)
        t_norm = t_idx.float() / float(self.train_timesteps - 1)
        noise = torch.randn_like(x_target)
        x_noisy = self._q_sample(x_target, t_idx, noise)

        unet_input = torch.cat([x_noisy, x_cond], dim=1)
        emb = self._build_emb(self.unet, self.action_encoder, t_norm, batch["action_features"])
        pred_noise = self.unet.forward_with_emb(unet_input, emb)

        weight = 1.0 + target_valid
        loss = (weight * (pred_noise - noise) ** 2).mean()
        loss_dict = {
            "total": float(loss.item()),
            "eps_mse": float(((pred_noise - noise) ** 2).mean().item()),
            "valid_ratio": float(target_valid.mean().item()),
        }
        return loss, loss_dict

    def _inference_timestep_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        steps = max(int(num_steps), 2)
        ts = torch.linspace(self.train_timesteps - 1, 0, steps, device=device).long()
        ts = torch.unique_consecutive(ts)
        if ts[-1].item() != 0:
            ts = torch.cat([ts, torch.zeros(1, device=device, dtype=torch.long)])
        return ts

    @torch.no_grad()
    def forward_inference(
        self,
        current_pc: torch.Tensor,
        rel_pose: torch.Tensor,
        action: torch.Tensor,
        current_mask: torch.Tensor,
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
    ) -> dict:
        if num_steps is None:
            num_steps = self.num_inference_steps
        x_cond, x_cond_valid = self._prepare_conditions(current_pc, current_mask, rel_pose)
        B = x_cond.shape[0]
        device = x_cond.device

        x = torch.randn_like(x_cond)
        ts = self._inference_timestep_schedule(num_steps, device)

        for i in range(ts.shape[0] - 1):
            t_cur = ts[i]
            t_prev = ts[i + 1]
            t_norm = torch.full((B,), float(t_cur.item()) / float(self.train_timesteps - 1), device=device)
            emb = self._build_emb(self.unet, self.action_encoder, t_norm, action)

            eps = self.unet.forward_with_emb(torch.cat([x, x_cond], dim=1), emb)
            a_bar_t = self.alpha_bars[t_cur]
            a_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
            x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1.0 - a_bar_prev) * eps

        pred_range = x.clamp(-1, 1)
        pred_valid = self._pred_valid_from_depth(pred_range, prior_valid=x_cond_valid)
        pred_points, pred_mask = self.projector.unproject(pred_range, pred_valid)
        return {
            "pred_range": pred_range,
            "pred_points": pred_points,
            "pred_mask": pred_mask,
            "x_0": x_cond,
            "x_0_valid": x_cond_valid,
        }


class OccWorldStyleDeterministicBaseline(BaselineBase):
    """Occupancy-first deterministic UNet baseline."""

    def __init__(self, cfg):
        super().__init__(cfg)
        mcfg = cfg.model
        emb_dim = mcfg.base_channels * 4
        self.unet = RangeUNet(
            in_channels=2,
            out_channels=2,  # depth + occupancy-logit
            base_channels=mcfg.base_channels,
            channel_mult=mcfg.channel_mult,
            num_res_blocks=mcfg.num_res_blocks,
            attention_resolutions=mcfg.attention_resolutions,
            num_heads=mcfg.num_heads,
            emb_dim=emb_dim,
            dropout=mcfg.dropout,
            use_circular=mcfg.use_circular_pad,
            use_activation_checkpointing=getattr(mcfg, "use_activation_checkpointing", False),
        )
        self.action_encoder = ActionEncoder(mcfg.action_dim, emb_dim) if mcfg.use_action_conditioning else None

    def training_step(self, batch: dict, epoch: int = 0):
        x_cond, x_cond_valid = self._prepare_conditions(
            batch["current_pc"], batch["current_mask"], batch["relative_pose"],
        )
        x_target, target_valid = self.projector.project(
            batch["target_pc"], batch["target_mask"],
        )
        B = x_target.shape[0]
        t_zeros = torch.zeros(B, device=x_target.device)
        emb = self._build_emb(self.unet, self.action_encoder, t_zeros, batch["action_features"])

        model_in = torch.cat([x_cond, x_cond_valid], dim=1)
        out = self.unet.forward_with_emb(model_in, emb)
        depth_pred = out[:, :1].tanh()
        occ_logit = out[:, 1:2]

        depth_loss = F.smooth_l1_loss(
            depth_pred * target_valid,
            x_target * target_valid,
            beta=0.05,
        )
        occ_target = (x_target > self.valid_threshold).float()
        occ_bce = F.binary_cross_entropy_with_logits(occ_logit, occ_target)
        occ_prob = torch.sigmoid(occ_logit)
        inter = (occ_prob * occ_target).sum(dim=(1, 2, 3))
        denom = occ_prob.sum(dim=(1, 2, 3)) + occ_target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()

        loss = depth_loss + 0.5 * occ_bce + 0.25 * dice_loss
        loss_dict = {
            "total": float(loss.item()),
            "depth": float(depth_loss.item()),
            "occ_bce": float(occ_bce.item()),
            "occ_dice": float(dice_loss.item()),
        }
        return loss, loss_dict

    @torch.no_grad()
    def forward_inference(
        self,
        current_pc: torch.Tensor,
        rel_pose: torch.Tensor,
        action: torch.Tensor,
        current_mask: torch.Tensor,
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
    ) -> dict:
        x_cond, x_cond_valid = self._prepare_conditions(current_pc, current_mask, rel_pose)
        B = x_cond.shape[0]
        t_zeros = torch.zeros(B, device=x_cond.device)
        emb = self._build_emb(self.unet, self.action_encoder, t_zeros, action)

        out = self.unet.forward_with_emb(torch.cat([x_cond, x_cond_valid], dim=1), emb)
        depth_pred = out[:, :1].tanh()
        occ_prob = torch.sigmoid(out[:, 1:2])
        pred_valid = self._pred_valid_from_depth(
            depth_pred, prior_valid=x_cond_valid, occ_prob=occ_prob,
        )
        pred_range = torch.where(pred_valid > 0.5, depth_pred, depth_pred.new_full(depth_pred.shape, -1.0))
        pred_points, pred_mask = self.projector.unproject(pred_range, pred_valid)
        return {
            "pred_range": pred_range,
            "pred_points": pred_points,
            "pred_mask": pred_mask,
            "x_0": x_cond,
            "x_0_valid": x_cond_valid,
        }


def build_baseline(model_type: str, cfg) -> BaselineBase:
    model_type = model_type.lower().strip()
    if model_type == "lidarcrafter_like":
        return LiDARCrafterLikeDiffusionBaseline(cfg)
    if model_type == "occworld_style":
        return OccWorldStyleDeterministicBaseline(cfg)
    raise ValueError(f"Unknown baseline type: {model_type}")
