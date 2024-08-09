"""
Latent Flow Matching world model.

Uses RangeImageVAE to compress range images to latent space, then
applies Flow Matching in the latent space. Decodes to range for loss.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.default import Config
from src.data.range_image_utils import build_projector_from_cfg
from src.model.range_vae import RangeImageVAE, RangeVAEConfig
from src.model.world_model import RangeWorldModel, FlowMatchingScheduler


class LatentRangeWorldModel(RangeWorldModel):
    """
    Flow Matching in VAE latent space.
    Encodes x_0, x_1 to latent, does FM in latent, decodes for loss.
    """

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)  # skip RangeWorldModel __init__ to override
        self.cfg = cfg
        mcfg = cfg.model

        self.projector = build_projector_from_cfg(cfg)
        self.fm = FlowMatchingScheduler()

        vae_cfg = RangeVAEConfig(
            in_channels=1,
            ri_H=mcfg.ri_H,
            ri_W=mcfg.ri_W,
            latent_channels=getattr(mcfg, "vae_latent_channels", 8),
            downsample=getattr(mcfg, "vae_downsample", 4),
        )
        self.vae = RangeImageVAE(vae_cfg, use_circular=mcfg.use_circular_pad)

        anchor_ch = getattr(mcfg, "anchor_channels", 0)
        self.use_anchor = anchor_ch > 0
        self.use_history = False
        self.history_encoder = None

        total_in = vae_cfg.latent_channels * 2 + (anchor_ch if self.use_anchor else 0)
        emb_dim = mcfg.base_channels * 4

        from src.model.world_model import RangeUNet
        self.unet = RangeUNet(
            in_channels=total_in,
            out_channels=vae_cfg.latent_channels,
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

        if mcfg.use_action_conditioning:
            from src.model.world_model import ActionEncoder
            self.action_encoder = ActionEncoder(mcfg.action_dim, emb_dim)
        else:
            self.action_encoder = None

        self.sigma_prior = mcfg.sigma_prior
        self.num_inference_steps = mcfg.num_inference_steps
        self.enable_inference_postprocess = getattr(mcfg, "enable_inference_postprocess", True)
        self.postprocess_outlier_threshold = getattr(mcfg, "postprocess_outlier_threshold", 0.2)
        self.postprocess_structure_weight = 0.0

        relaxed_depth_m = max(
            float(getattr(mcfg, "validity_relaxed_depth_m", self.projector.min_depth)),
            self.projector.min_depth,
        )
        strict_depth_m = max(float(getattr(mcfg, "validity_strict_depth_m", 2.0)), relaxed_depth_m)
        self.validity_relaxed_threshold = self._depth_to_normalised(relaxed_depth_m)
        self.validity_strict_threshold = self._depth_to_normalised(strict_depth_m)
        self.validity_smooth_base = 0.25
        self.validity_smooth_with_prior = 0.35

    def _depth_to_normalised(self, depth_m: float) -> float:
        depth_m = max(float(depth_m), 1e-6)
        return (math.log2(depth_m + 1.0) / self.projector._log_max) * 2.0 - 1.0

    def _prepare_conditions(self, current_pc, current_mask, rel_pose):
        return self.projector.warp_range_image(current_pc, current_mask, rel_pose)

    def forward_train(
        self,
        current_pc: torch.Tensor,
        target_pc: torch.Tensor,
        rel_pose: torch.Tensor,
        action: torch.Tensor,
        current_mask: torch.Tensor,
        target_mask: torch.Tensor,
        anchor_range: Optional[torch.Tensor] = None,
        history_encoded: Optional[torch.Tensor] = None,
    ) -> dict:
        B = current_pc.shape[0]
        device = current_pc.device

        x_0, x_0_valid = self._prepare_conditions(current_pc, current_mask, rel_pose)
        x_1, x_1_valid = self.projector.project(target_pc, target_mask)

        if self.use_anchor and anchor_range is None:
            anchor_range, _ = self.projector.project(current_pc, current_mask)

        z_0_mu, z_0_logvar = self.vae.encode(x_0)
        z_1_mu, z_1_logvar = self.vae.encode(x_1)
        z_0 = self.vae.reparameterize(z_0_mu, z_0_logvar)
        z_1 = z_1_mu

        noise = torch.randn_like(z_0) * 0.02
        z_0_noisy = z_0 + noise

        t = self.fm.sample_t(B, device)
        z_t = self.fm.interpolate(z_0_noisy, z_1, t)
        v_target = self.fm.target_velocity(z_0_noisy, z_1)

        parts = [z_t, z_0]
        if self.use_anchor and anchor_range is not None:
            z_anchor_mu, _ = self.vae.encode(anchor_range)
            parts.append(z_anchor_mu)
        unet_input = torch.cat(parts, dim=1)

        emb_t = self.unet.time_embed(t)
        if self.action_encoder is not None:
            emb = emb_t + self.action_encoder(action)
        else:
            emb = emb_t

        v_pred = self.unet.forward_with_emb(unet_input, emb)
        z1_pred = self.fm.predict_x1(z_t, v_pred, t)

        x1_pred = self.vae.decode(z1_pred).clamp(-1, 1)

        return {
            "v_pred": v_pred,
            "v_target": v_target,
            "x1_pred": x1_pred,
            "x_1": x_1,
            "x_0": x_0,
            "t": t,
            "target_valid": x_1_valid,
            "prior_valid": x_0_valid,
        }

    @torch.no_grad()
    def forward_inference(
        self,
        current_pc: torch.Tensor,
        rel_pose: torch.Tensor,
        action: torch.Tensor,
        current_mask: torch.Tensor,
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
        anchor_range: Optional[torch.Tensor] = None,
    ) -> dict:
        if num_steps is None:
            num_steps = self.num_inference_steps
        B = current_pc.shape[0]
        device = current_pc.device

        x_0, x_0_valid = self._prepare_conditions(current_pc, current_mask, rel_pose)
        z_0_mu, z_0_logvar = self.vae.encode(x_0)
        z = z_0_mu

        if self.use_anchor and anchor_range is None:
            anchor_range, _ = self.projector.project(current_pc, current_mask)
        z_anchor = self.vae.encode(anchor_range)[0] if (self.use_anchor and anchor_range is not None) else None

        emb_base = self.action_encoder(action) if self.action_encoder is not None else torch.zeros(B, self.unet.emb_dim, device=device)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)
            parts = [z, z_0_mu]
            if z_anchor is not None:
                parts.append(z_anchor)
            unet_input = torch.cat(parts, dim=1)
            emb = self.unet.time_embed(t) + emb_base
            v1 = self.unet.forward_with_emb(unet_input, emb)
            if use_midpoint and step < num_steps - 1:
                z_mid = z + v1 * dt
                t_mid = torch.full((B,), t_val + dt, device=device)
                parts_mid = [z_mid, z_0_mu]
                if z_anchor is not None:
                    parts_mid.append(z_anchor)
                unet_input_mid = torch.cat(parts_mid, dim=1)
                emb_mid = self.unet.time_embed(t_mid) + emb_base
                v2 = self.unet.forward_with_emb(unet_input_mid, emb_mid)
                z = z + 0.5 * (v1 + v2) * dt
            else:
                z = z + v1 * dt

        pred_range = self.vae.decode(z).clamp(-1, 1)
        pred_valid = (pred_range > self.validity_relaxed_threshold).float()
        pred_points, pred_pt_mask = self.projector.unproject(pred_range, pred_valid)

        return {
            "pred_range": pred_range,
            "pred_points": pred_points,
            "pred_mask": pred_pt_mask,
            "x_0": x_0,
            "x_0_valid": x_0_valid,
        }

    @torch.no_grad()
    def forward_inference_from_range(
        self,
        x_0: torch.Tensor,
        x_0_valid: torch.Tensor,
        action: torch.Tensor,
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
        anchor_range: Optional[torch.Tensor] = None,
        skip_postprocess: bool = False,
    ) -> dict:
        if num_steps is None:
            num_steps = self.num_inference_steps
        B = x_0.shape[0]
        device = x_0.device

        z_0_mu, _ = self.vae.encode(x_0)
        z = z_0_mu
        z_anchor = self.vae.encode(anchor_range)[0] if (self.use_anchor and anchor_range is not None) else None
        emb_base = self.action_encoder(action) if self.action_encoder is not None else torch.zeros(B, self.unet.emb_dim, device=device)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)
            parts = [z, z_0_mu]
            if z_anchor is not None:
                parts.append(z_anchor)
            unet_input = torch.cat(parts, dim=1)
            emb = self.unet.time_embed(t) + emb_base
            v1 = self.unet.forward_with_emb(unet_input, emb)
            if use_midpoint and step < num_steps - 1:
                z_mid = z + v1 * dt
                t_mid = torch.full((B,), t_val + dt, device=device)
                parts_mid = [z_mid, z_0_mu]
                if z_anchor is not None:
                    parts_mid.append(z_anchor)
                unet_input_mid = torch.cat(parts_mid, dim=1)
                emb_mid = self.unet.time_embed(t_mid) + emb_base
                v2 = self.unet.forward_with_emb(unet_input_mid, emb_mid)
                z = z + 0.5 * (v1 + v2) * dt
            else:
                z = z + v1 * dt

        pred_range = self.vae.decode(z).clamp(-1, 1)
        pred_valid = (pred_range > self.validity_relaxed_threshold).float()
        pred_points, pred_pt_mask = self.projector.unproject(pred_range, pred_valid)
        return {
            "pred_range": pred_range,
            "pred_points": pred_points,
            "pred_mask": pred_pt_mask,
            "x_0": x_0,
            "x_0_valid": x_0_valid,
        }
