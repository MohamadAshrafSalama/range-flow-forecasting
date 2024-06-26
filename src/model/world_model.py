"""
Range-image world model with flow matching.

Predicts the next LiDAR frame as a range image using a conditional UNet
with optimal-transport conditional flow matching (OT-CFM). The structured
prior is the ego-motion-warped previous frame projected to range-image space.

Architecture:
  prev_frame -> [ego warp] -> [project] -> x_0 (structured prior)
  next_frame -> [project] -> x_1 (target)
  x_t = (1-t)*x_0 + t*x_1     (FM interpolation)
  v_theta(x_t, t, cond=x_0, action) -> predict v = x_1 - x_0

UNet design:
  - Circular padding on horizontal axis (360 degree wrap-around)
  - Adaptive Group Norm (AdaGN) for timestep + action conditioning
  - Self-attention at low resolutions
  - Channel multiplier [1, 2, 4, 8]: 64 -> 128 -> 256 -> 512
"""

from __future__ import annotations

import math
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from src.data.range_image_utils import (
    RangeImageProjector,
    CircularPad2d,
    CircularConv2d,
    build_projector_from_cfg,
)

# =====================================================================
#  Building Blocks
# =====================================================================


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep / continuous scalar."""

    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaGN(nn.Module):
    """
    Adaptive Group Normalisation.
    Applies  (1 + scale) * GN(x) + shift  where (scale, shift) are
    predicted from a conditioning embedding.
    """

    def __init__(self, emb_dim: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, out_channels, eps=1e-5, affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.gn(x)
        scale, shift = self.proj(emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return h * (1 + scale) + shift


class ResBlock(nn.Module):
    """
    Residual block with AdaGN timestep conditioning and circular conv.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout: float = 0.0,
        use_circular: bool = True,
    ):
        super().__init__()
        Conv = CircularConv2d if use_circular else lambda i, o, k=3, **kw: nn.Conv2d(i, o, k, padding=k // 2, **kw)

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = Conv(in_ch, out_ch)
        self.ada_gn = AdaGN(emb_dim, out_ch)
        self.conv2 = Conv(out_ch, out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        nn.init.zeros_(self.conv2.conv.weight if isinstance(self.conv2, CircularConv2d) else self.conv2.weight)
        nn.init.zeros_(self.conv2.conv.bias if isinstance(self.conv2, CircularConv2d) else self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.ada_gn(h, emb)
        h = self.conv2(self.drop(self.act(h)))
        return h + self.skip(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention on spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H * W)
        return x + self.proj_out(out).view(B, C, H, W)


class Downsample2d(nn.Module):
    """Strided convolution downsampling (factor 2)."""

    def __init__(self, channels: int, use_circular: bool = True):
        super().__init__()
        if use_circular:
            self.pad = CircularPad2d(1)
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)
        else:
            self.pad = None
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad is not None:
            x = self.pad(x)
        return self.conv(x)


class Upsample2d(nn.Module):
    """Nearest-neighbour upsample (factor 2) + conv."""

    def __init__(self, channels: int, use_circular: bool = True):
        super().__init__()
        Conv = CircularConv2d if use_circular else lambda i, o, k=3, **kw: nn.Conv2d(i, o, k, padding=k // 2, **kw)
        self.conv = Conv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# =====================================================================
#  Range-Image UNet
# =====================================================================


class RangeUNet(nn.Module):
    """
    UNet for range-image generation / velocity prediction.

    4 resolution levels with [1x, 1/2x, 1/4x, 1/8x]
    Channels:  base x [1, 2, 4, 8]  ->  64, 128, 256, 512
    Attention at 1/4x and 1/8x resolutions.

    Conditioning via AdaGN: the embedding = timestep_emb + action_emb.
    Previous frame is concatenated as extra input channel(s).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (8, 16),
        num_heads: int = 8,
        emb_dim: int = 256,
        dropout: float = 0.0,
        use_circular: bool = True,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_activation_checkpointing = use_activation_checkpointing
        ch = base_channels

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(ch),
            nn.Linear(ch, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # ---- Encoder ---- #
        self.input_conv = CircularConv2d(in_channels, ch) if use_circular else nn.Conv2d(in_channels, ch, 3, padding=1)

        self.enc_res = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.enc_down = nn.ModuleList()

        skip_channels = [ch]
        cur_ch = ch
        ds = 1

        for level, mult in enumerate(channel_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.enc_res.append(ResBlock(cur_ch, out_ch, emb_dim, dropout, use_circular))
                if ds in attention_resolutions:
                    self.enc_attn.append(MultiHeadSelfAttention(out_ch, num_heads))
                else:
                    self.enc_attn.append(nn.Identity())
                cur_ch = out_ch
                skip_channels.append(cur_ch)

            if level < len(channel_mult) - 1:
                self.enc_down.append(Downsample2d(cur_ch, use_circular))
                skip_channels.append(cur_ch)
                ds *= 2
            else:
                self.enc_down.append(None)

        self._n_enc_levels = len(channel_mult)

        # ---- Middle ---- #
        mid_ch = cur_ch
        self.mid_block1 = ResBlock(mid_ch, mid_ch, emb_dim, dropout, use_circular)
        self.mid_attn = MultiHeadSelfAttention(mid_ch, num_heads)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, emb_dim, dropout, use_circular)

        # ---- Decoder ---- #
        self.dec_res = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.dec_up = nn.ModuleList()

        for level in reversed(range(len(channel_mult))):
            mult = channel_mult[level]
            out_ch = ch * mult
            n_blocks = num_res_blocks + 1
            for i in range(n_blocks):
                skip_ch = skip_channels.pop()
                self.dec_res.append(ResBlock(cur_ch + skip_ch, out_ch, emb_dim, dropout, use_circular))
                if ds in attention_resolutions:
                    self.dec_attn.append(MultiHeadSelfAttention(out_ch, num_heads))
                else:
                    self.dec_attn.append(nn.Identity())
                cur_ch = out_ch

            if level > 0:
                self.dec_up.append(Upsample2d(cur_ch, use_circular))
                ds //= 2
            else:
                self.dec_up.append(None)

        # ---- Output ---- #
        self.output_norm = nn.GroupNorm(32, cur_ch)
        self.output_conv = nn.Conv2d(cur_ch, out_channels, 1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

        self._num_res_blocks = num_res_blocks
        self._channel_mult = channel_mult

    def _should_checkpoint(self) -> bool:
        return self.use_activation_checkpointing and self.training and torch.is_grad_enabled()

    def _run_resblock(self, block: nn.Module, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self._should_checkpoint():
            return activation_checkpoint(lambda x, e: block(x, e), h, emb, use_reentrant=False)
        return block(h, emb)

    def _run_single_input(self, block: nn.Module, h: torch.Tensor) -> torch.Tensor:
        if isinstance(block, nn.Identity):
            return h
        if self._should_checkpoint():
            return activation_checkpoint(block, h, use_reentrant=False)
        return block(h)

    def forward_with_emb(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward with pre-computed embedding (used by world model)."""
        h = self.input_conv(x)
        skips = [h]

        block_idx = 0
        for level in range(self._n_enc_levels):
            for _ in range(self._num_res_blocks):
                h = self._run_resblock(self.enc_res[block_idx], h, emb)
                h = self._run_single_input(self.enc_attn[block_idx], h)
                skips.append(h)
                block_idx += 1
            if self.enc_down[level] is not None:
                h = self._run_single_input(self.enc_down[level], h)
                skips.append(h)

        h = self._run_resblock(self.mid_block1, h, emb)
        h = self._run_single_input(self.mid_attn, h)
        h = self._run_resblock(self.mid_block2, h, emb)

        block_idx = 0
        up_idx = 0
        for level in reversed(range(self._n_enc_levels)):
            n_blocks = self._num_res_blocks + 1
            for _ in range(n_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self._run_resblock(self.dec_res[block_idx], h, emb)
                h = self._run_single_input(self.dec_attn[block_idx], h)
                block_idx += 1
            if self.dec_up[up_idx] is not None:
                h = self._run_single_input(self.dec_up[up_idx], h)
            up_idx += 1

        return self.output_conv(F.silu(self.output_norm(h)))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t_emb)
        return self.forward_with_emb(x, emb)


# =====================================================================
#  Action Encoder
# =====================================================================


class ActionEncoder(nn.Module):
    """Encodes the 5D ego-motion action vector to an embedding."""

    def __init__(self, action_dim: int = 5, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


class StepIndexEncoder(nn.Module):
    """
    Encodes step index k (0..K-1) for multi-step conditioning.

    The model can adapt its strategy based on which rollout step it is
    predicting (e.g. be more corrective for later steps with noisier inputs).
    """

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, step_ratio: torch.Tensor) -> torch.Tensor:
        """step_ratio: (B,) in [0, 1], e.g. k/K."""
        h = self.sin_emb(step_ratio)
        return self.proj(h)


# =====================================================================
#  Flow Matching Scheduler
# =====================================================================


class FlowMatchingScheduler:
    """
    Optimal-transport conditional flow matching scheduler.

    x_0 : structured prior (warped prev frame)
    x_1 : target (ground truth next frame)
    x_t = (1-t) * x_0 + t * x_1
    v_target = x_1 - x_0   (constant velocity, independent of t)
    x_1_pred = x_t + (1-t) * v_pred
    """

    @staticmethod
    def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample t with logit-normal distribution.

        Logit-normal concentrates samples near t=0 and t=1 where the
        model needs to be most accurate (start and end of the ODE).
        """
        z = torch.randn(batch_size, device=device)
        t = torch.sigmoid(z)
        return t.clamp(1e-4, 1 - 1e-4)

    @staticmethod
    def interpolate(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = (1-t)*x_0 + t*x_1"""
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x_0 + t * x_1

    @staticmethod
    def target_velocity(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """u = x_1 - x_0  (constant, independent of t)"""
        return x_1 - x_0

    @staticmethod
    def predict_x1(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_1 = x_t + (1-t)*v_pred"""
        t = t.view(-1, 1, 1, 1)
        return x_t + (1 - t) * v_pred


# =====================================================================
#  Full World Model
# =====================================================================


class RangeWorldModel(nn.Module):
    """
    Range-image world model with flow matching.

    Given the current LiDAR frame and ego-motion, predicts the next frame.

    Training:
        1. Warp current frame to next frame coords -> x_0 (structured prior)
        2. Project ground truth next frame -> x_1
        3. FM interpolation: x_t = (1-t)*x_0 + t*x_1
        4. Predict velocity v_theta(x_t, t, condition=x_0, action)
        5. Loss = MSE(v_theta, x_1 - x_0)

    Inference:
        1. Warp current frame -> x_0
        2. ODE integration from x_0 for n_steps (Euler or midpoint)
        3. Unproject predicted range image -> 3D point cloud

    Multi-step:
        - Autoregressive rollout: chain single-step predictions
        - Scheduled sampling: randomly use own predictions as input during training
          to close the train-inference distribution gap
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model

        self.projector = build_projector_from_cfg(cfg)
        self.fm = FlowMatchingScheduler()

        anchor_ch = getattr(mcfg, "anchor_channels", 0)
        self.use_anchor = anchor_ch > 0
        self.use_history = getattr(mcfg, "use_history", False)
        self.history_frames = getattr(mcfg, "history_frames", 2) if self.use_history else 0
        history_ch = 0
        if self.use_history and self.history_frames > 0:
            from src.model.history_encoder import HistoryEncoder
            self.history_encoder = HistoryEncoder(
                num_history=self.history_frames,
                in_channels=1,
                out_channels=4,
                ri_H=mcfg.ri_H,
                ri_W=mcfg.ri_W,
                downsample=4,
                use_circular=mcfg.use_circular_pad,
            )
            history_ch = 4
        else:
            self.history_encoder = None
        total_in = mcfg.in_channels + mcfg.cond_channels + anchor_ch + history_ch
        emb_dim = mcfg.base_channels * 4  # 256 for base=64

        self.use_occupancy_flow = getattr(mcfg, "use_occupancy_flow", False)
        out_ch = mcfg.out_channels + (1 if self.use_occupancy_flow else 0)

        self.unet = RangeUNet(
            in_channels=total_in,
            out_channels=out_ch,
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
            self.action_encoder = ActionEncoder(mcfg.action_dim, emb_dim)
        else:
            self.action_encoder = None

        self.use_step_index_conditioning = getattr(mcfg, "use_step_index_conditioning", False)
        if self.use_step_index_conditioning:
            self.step_index_encoder = StepIndexEncoder(emb_dim)
        else:
            self.step_index_encoder = None

        if self.use_history and self.history_encoder is not None:
            self.history_placeholder = nn.Parameter(torch.zeros(1, history_ch, 1, 1))
            nn.init.normal_(self.history_placeholder, std=0.02)
        else:
            self.history_placeholder = None

        self.sigma_prior = mcfg.sigma_prior
        self.inference_noise_std = getattr(mcfg, "inference_noise_std", 0.0)
        self.num_inference_steps = mcfg.num_inference_steps
        self.enable_inference_postprocess = getattr(mcfg, "enable_inference_postprocess", True)
        self.postprocess_outlier_threshold = getattr(mcfg, "postprocess_outlier_threshold", 0.2)
        self.postprocess_structure_weight = getattr(mcfg, "postprocess_structure_weight", 0.02)
        self.postprocess_structure_max_deviation = getattr(
            mcfg, "postprocess_structure_max_deviation", 0.12,
        )
        relaxed_depth_m = max(
            float(getattr(mcfg, "validity_relaxed_depth_m", self.projector.min_depth)),
            self.projector.min_depth,
        )
        strict_depth_m = max(
            float(getattr(mcfg, "validity_strict_depth_m", 2.0)),
            relaxed_depth_m,
        )
        self.validity_relaxed_threshold = self._depth_to_normalised(relaxed_depth_m)
        self.validity_strict_threshold = self._depth_to_normalised(strict_depth_m)
        self.validity_smooth_base = float(getattr(mcfg, "validity_smooth_base", 0.25))
        self.validity_smooth_with_prior = float(
            getattr(mcfg, "validity_smooth_with_prior", 0.35),
        )

    def _depth_to_normalised(self, depth_m: float) -> float:
        """Convert metric depth in metres to normalised log-depth [-1, 1]."""
        depth_m = max(float(depth_m), 1e-6)
        return (math.log2(depth_m + 1.0) / self.projector._log_max) * 2.0 - 1.0

    def _prepare_conditions(
        self,
        current_pc: torch.Tensor,
        current_mask: torch.Tensor,
        rel_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warp current frame and project to range image.

        Returns:
            x_0: (B, 1, H, W) warped previous frame (structured prior)
            x_0_valid: (B, 1, H, W) validity mask
        """
        x_0, x_0_valid = self.projector.warp_range_image(current_pc, current_mask, rel_pose)
        return x_0, x_0_valid

    def forward_train(
        self,
        current_pc: torch.Tensor,   # (B, N, 3) current frame
        target_pc: torch.Tensor,     # (B, N, 3) next frame
        rel_pose: torch.Tensor,       # (B, 4, 4) ego-motion
        action: torch.Tensor,         # (B, 5) action vector
        current_mask: torch.Tensor,   # (B, N) valid points
        target_mask: torch.Tensor,    # (B, N) valid points
        anchor_range: Optional[torch.Tensor] = None,
        history_encoded: Optional[torch.Tensor] = None,  # (B, C, H, W) from HistoryEncoder
        step_idx: Optional[int] = None,  # 0..K-1 for multi-step; None = single-step
    ) -> dict:
        B = current_pc.shape[0]
        device = current_pc.device

        x_0, x_0_valid = self._prepare_conditions(current_pc, current_mask, rel_pose)
        x_1, x_1_valid = self.projector.project(target_pc, target_mask)

        if self.use_anchor:
            if anchor_range is None:
                anchor_range, _ = self.projector.project(current_pc, current_mask)

        noise = torch.randn_like(x_0) * self.sigma_prior
        x_0_noisy = x_0 + noise

        t = self.fm.sample_t(B, device)
        x_t = self.fm.interpolate(x_0_noisy, x_1, t)
        v_target = self.fm.target_velocity(x_0_noisy, x_1)

        parts = [x_t, x_0]
        if self.use_anchor and anchor_range is not None:
            parts.append(anchor_range)
        if self.use_history and self.history_encoder is not None:
            if history_encoded is not None:
                parts.append(history_encoded)
            elif self.history_placeholder is not None:
                H, W = x_0.shape[2], x_0.shape[3]
                placeholder = self.history_placeholder.expand(B, -1, H, W)
                parts.append(placeholder)
            else:
                H, W = x_0.shape[2], x_0.shape[3]
                parts.append(torch.zeros(B, 4, H, W, device=device, dtype=x_0.dtype))
        unet_input = torch.cat(parts, dim=1)

        emb_t = self.unet.time_embed(t)
        emb = emb_t
        if self.action_encoder is not None:
            emb = emb + self.action_encoder(action)
        if self.step_index_encoder is not None:
            K = getattr(self.cfg.training, "sequence_length", 6)
            step_ratio = torch.full((B,), (step_idx if step_idx is not None else 0) / max(K - 1, 1), device=device)
            emb = emb + self.step_index_encoder(step_ratio)

        unet_out = self.unet.forward_with_emb(unet_input, emb)
        if self.use_occupancy_flow:
            v_pred = unet_out[:, :1]
            occ_logit = unet_out[:, 1:2]
        else:
            v_pred = unet_out
            occ_logit = None

        x1_pred = self.fm.predict_x1(x_t, v_pred, t).clamp(-1, 1)

        out = {
            "v_pred": v_pred,
            "v_target": v_target,
            "x1_pred": x1_pred,
            "x_1": x_1,
            "x_0": x_0,
            "t": t,
            "target_valid": x_1_valid,
            "prior_valid": x_0_valid,
        }
        if occ_logit is not None:
            out["occ_logit"] = occ_logit
        return out

    def forward_ode_integration_train(
        self,
        current_pc: torch.Tensor,
        rel_pose: torch.Tensor,
        action: torch.Tensor,
        current_mask: torch.Tensor,
        num_steps: int = 10,
        use_midpoint: bool = True,
        anchor_range: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ODE integration with gradients (for train-test alignment).
        Returns final x1_pred at end of trajectory.
        """
        B = current_pc.shape[0]
        device = current_pc.device
        x_0, x_0_valid = self._prepare_conditions(current_pc, current_mask, rel_pose)
        if self.use_anchor and anchor_range is None:
            anchor_range, _ = self.projector.project(current_pc, current_mask)

        x = x_0.clone()
        emb_base = torch.zeros(B, self.unet.emb_dim, device=device)
        if self.action_encoder is not None:
            emb_base = self.action_encoder(action)

        dt = 1.0 / num_steps
        H, W = x_0.shape[2], x_0.shape[3]
        history_ch = torch.zeros(B, 4, H, W, device=device, dtype=x_0.dtype)
        if self.use_history and self.history_encoder is not None:
            if self.history_placeholder is not None:
                history_ch = self.history_placeholder.expand(B, -1, H, W)
        use_history_ch = self.use_history and self.history_encoder is not None

        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)
            parts = [x, x_0]
            if self.use_anchor and anchor_range is not None:
                parts.append(anchor_range)
            if use_history_ch:
                parts.append(history_ch)
            unet_input = torch.cat(parts, dim=1)
            emb = self.unet.time_embed(t) + emb_base
            v1 = self.unet.forward_with_emb(unet_input, emb)
            if self.use_occupancy_flow:
                v1 = v1[:, :1]
            if use_midpoint and step < num_steps - 1:
                x_mid = x + v1 * dt
                t_mid = torch.full((B,), t_val + dt, device=device)
                parts_mid = [x_mid, x_0]
                if self.use_anchor and anchor_range is not None:
                    parts_mid.append(anchor_range)
                if use_history_ch:
                    parts_mid.append(history_ch)
                unet_input_mid = torch.cat(parts_mid, dim=1)
                emb_mid = self.unet.time_embed(t_mid) + emb_base
                v2 = self.unet.forward_with_emb(unet_input_mid, emb_mid)
                if self.use_occupancy_flow:
                    v2 = v2[:, :1]
                x = x + 0.5 * (v1 + v2) * dt
            else:
                x = x + v1 * dt

        return x.clamp(-1, 1)

    @torch.no_grad()
    def forward_inference(
        self,
        current_pc: torch.Tensor,   # (B, N, 3)
        rel_pose: torch.Tensor,     # (B, 4, 4)
        action: torch.Tensor,       # (B, 5)
        current_mask: torch.Tensor,  # (B, N)
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
        noise_std: Optional[float] = None,
        anchor_range: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
    ) -> dict:
        if num_steps is None:
            num_steps = self.num_inference_steps
        B = current_pc.shape[0]
        device = current_pc.device

        x_0, x_0_valid = self._prepare_conditions(current_pc, current_mask, rel_pose)

        if self.use_anchor and anchor_range is None:
            anchor_range, _ = self.projector.project(current_pc, current_mask)

        if noise_std is None:
            noise_std = self.inference_noise_std
        if noise_std > 0.0:
            x = x_0 + torch.randn_like(x_0) * noise_std
        else:
            x = x_0.clone()

        emb_base = torch.zeros(B, self.unet.emb_dim, device=device)
        if self.action_encoder is not None:
            emb_base = emb_base + self.action_encoder(action)
        if self.step_index_encoder is not None and step_idx is not None:
            K = getattr(self.cfg.training, "sequence_length", 6)
            step_ratio = torch.full((B,), step_idx / max(K - 1, 1), device=device)
            emb_base = emb_base + self.step_index_encoder(step_ratio)

        H, W = x_0.shape[2], x_0.shape[3]
        history_ch = torch.zeros(B, 4, H, W, device=device, dtype=x_0.dtype)
        if self.use_history and self.history_encoder is not None:
            if self.history_placeholder is not None:
                history_ch = self.history_placeholder.expand(B, -1, H, W)
        use_history_channels = self.use_history and self.history_encoder is not None

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)

            parts = [x, x_0]
            if self.use_anchor and anchor_range is not None:
                parts.append(anchor_range)
            if use_history_channels:
                parts.append(history_ch)
            unet_input = torch.cat(parts, dim=1)
            emb = self.unet.time_embed(t) + emb_base
            v1 = self.unet.forward_with_emb(unet_input, emb)
            if self.use_occupancy_flow:
                v1 = v1[:, :1]

            if use_midpoint and step < num_steps - 1:
                x_mid = x + v1 * dt
                t_mid = torch.full((B,), t_val + dt, device=device)
                parts_mid = [x_mid, x_0]
                if self.use_anchor and anchor_range is not None:
                    parts_mid.append(anchor_range)
                if use_history_channels:
                    parts_mid.append(history_ch)
                unet_input_mid = torch.cat(parts_mid, dim=1)
                emb_mid = self.unet.time_embed(t_mid) + emb_base
                v2 = self.unet.forward_with_emb(unet_input_mid, emb_mid)
                if self.use_occupancy_flow:
                    v2 = v2[:, :1]
                x = x + 0.5 * (v1 + v2) * dt
            else:
                x = x + v1 * dt

        pred_range = x.clamp(-1, 1)
        pred_range = self._postprocess_range_image(pred_range, x_0, x_0_valid)
        pred_valid = self._compute_validity_mask(pred_range, x_0_valid)
        pred_points, pred_pt_mask = self.projector.unproject(pred_range, pred_valid.float())

        return {
            "pred_range": pred_range,
            "pred_points": pred_points,
            "pred_mask": pred_pt_mask,
            "x_0": x_0,
            "x_0_valid": x_0_valid,
        }

    @torch.no_grad()
    def _postprocess_range_image(
        self,
        pred_range: torch.Tensor,
        x_0: torch.Tensor,
        x_0_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lightweight post-processing: remove extreme salt-and-pepper outliers.
        """
        if not self.enable_inference_postprocess:
            return pred_range.clamp(-1, 1)

        B, C, H, W = pred_range.shape

        padded = F.pad(pred_range, (1, 1, 0, 0), mode="circular")
        padded = F.pad(padded, (0, 0, 1, 1), mode="replicate")
        patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)  # (B, C, H, W, 3, 3)
        median_filtered = patches.reshape(B, C, H, W, 9).median(dim=-1).values

        deviation = (pred_range - median_filtered).abs()
        extreme_outlier = deviation > (self.postprocess_outlier_threshold * 2.0)
        blend_weight = extreme_outlier.float()
        pred_range = pred_range * (1 - blend_weight) + median_filtered * blend_weight

        return pred_range.clamp(-1, 1)

    @torch.no_grad()
    def _compute_validity_mask(
        self,
        pred_range: torch.Tensor,
        x_0_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Relaxed validity mask for unprojection.
        """
        prior_support = x_0_valid > 0.5

        depth_valid_relaxed = pred_range > self.validity_relaxed_threshold
        depth_valid_strict = pred_range > self.validity_strict_threshold
        depth_valid = depth_valid_strict | (depth_valid_relaxed & prior_support)

        valid_float = depth_valid.float()
        padded = F.pad(valid_float, (1, 1, 0, 0), mode="circular")
        padded = F.pad(padded, (0, 0, 1, 1), mode="constant", value=0)
        neighbour_count = F.avg_pool2d(padded, 3, stride=1, padding=0) * 9.0
        locally_consistent = neighbour_count >= 1.5

        pred_valid = depth_valid & locally_consistent
        pred_valid = pred_valid | (depth_valid_relaxed & prior_support)

        return pred_valid.float()

    # -----------------------------------------------------------------
    #  Autoregressive Rollout (multi-step inference t+1 .. t+K)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def forward_rollout(
        self,
        current_pc: torch.Tensor,        # (B, N, 3)
        current_mask: torch.Tensor,       # (B, N)
        rel_poses: list[torch.Tensor],    # K x (B, 4, 4)
        actions: list[torch.Tensor],      # K x (B, 5)
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
    ) -> list[dict]:
        K = len(rel_poses)
        assert len(actions) == K

        anchor_range = None
        if self.use_anchor:
            anchor_range, _ = self.projector.project(current_pc, current_mask)

        results = []
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
                anchor_range=anchor_range,
            )
            results.append(out)
            pc = out["pred_points"]
            mask = out["pred_mask"]

        return results

    # -----------------------------------------------------------------
    #  Scheduled Sampling Training (multi-step)
    # -----------------------------------------------------------------

    def forward_train_scheduled(
        self,
        frames_pc: list[torch.Tensor],     # (K+1) x (B, N, 3)
        frames_mask: list[torch.Tensor],    # (K+1) x (B, N)
        rel_poses: list[torch.Tensor],      # K x (B, 4, 4)
        actions: list[torch.Tensor],        # K x (B, 5)
        sampling_prob: float = 0.0,
        history_pc: Optional[torch.Tensor] = None,   # (B, H, N, 3) when use_history
        history_mask: Optional[torch.Tensor] = None,  # (B, H, N)
    ) -> list[dict]:
        """
        Multi-step training with scheduled sampling and anchor conditioning.

        Key design choices for long-horizon performance:
          1. Anchor frame: original observation (frame 0) is passed as context
             to all steps, providing stable scene memory.
          2. Relaxed intermediate validity: predicted range images are passed
             to the next step with minimal filtering to prevent cascading
             information loss.
          3. High scheduled sampling probability trains the model to operate
             robustly on its own imperfect predictions.
        """
        K = len(rel_poses)
        assert len(frames_pc) == K + 1

        anchor_range = None
        if self.use_anchor:
            anchor_range, _ = self.projector.project(frames_pc[0], frames_mask[0])

        history_encoded_all = None
        if self.use_history and self.history_encoder is not None and history_pc is not None and history_mask is not None:
            B, H_hist, N, _ = history_pc.shape
            hist_range_list = []
            for h in range(H_hist):
                rr, _ = self.projector.project(history_pc[:, h], history_mask[:, h])
                hist_range_list.append(rr)
            history_range = torch.stack(hist_range_list, dim=1)
            history_encoded_all = self.history_encoder(history_range)
            history_encoded_all = F.interpolate(
                history_encoded_all,
                size=(self.projector.H, self.projector.W),
                mode="bilinear",
                align_corners=False,
            )

        outputs = []
        prev_pred_range = None
        prev_pred_valid = None

        for k in range(K):
            use_own_prediction = (
                k > 0
                and prev_pred_range is not None
                and prev_pred_valid is not None
                and torch.rand(1).item() < sampling_prob
            )

            if use_own_prediction:
                prev_points, prev_points_mask = self.projector.unproject(
                    prev_pred_range, prev_pred_valid,
                )
                x_0, x_0_valid = self.projector.warp_range_image(
                    prev_points, prev_points_mask, rel_poses[k],
                )

                x_1, x_1_valid = self.projector.project(
                    frames_pc[k + 1], frames_mask[k + 1],
                )

                B = x_0.shape[0]
                device = x_0.device

                step_noise_scale = getattr(self.cfg.training, "step_noise_scale", 0.0)
                sigma = self.sigma_prior * (1.0 + step_noise_scale * k / max(K, 1))
                x_0_noisy = x_0 + torch.randn_like(x_0) * sigma

                t = self.fm.sample_t(B, device)
                x_t = self.fm.interpolate(x_0_noisy, x_1, t)
                v_target = self.fm.target_velocity(x_0_noisy, x_1)

                parts = [x_t, x_0]
                if self.use_anchor and anchor_range is not None:
                    parts.append(anchor_range)
                if self.use_history and self.history_encoder is not None:
                    H, W = x_0.shape[2], x_0.shape[3]
                    if k > 0 and history_encoded_all is not None:
                        parts.append(history_encoded_all)
                    elif self.history_placeholder is not None:
                        parts.append(self.history_placeholder.expand(B, -1, H, W))
                    else:
                        parts.append(torch.zeros(B, 4, H, W, device=device, dtype=x_0.dtype))
                unet_input = torch.cat(parts, dim=1)

                emb_t = self.unet.time_embed(t)
                emb = emb_t
                if self.action_encoder is not None:
                    emb = emb + self.action_encoder(actions[k])
                if self.step_index_encoder is not None:
                    step_ratio = torch.full((B,), k / max(K - 1, 1), device=device)
                    emb = emb + self.step_index_encoder(step_ratio)

                unet_out = self.unet.forward_with_emb(unet_input, emb)
                v_pred = unet_out[:, :1] if self.use_occupancy_flow else unet_out
                x1_pred = self.fm.predict_x1(x_t, v_pred, t).clamp(-1, 1)

                out = {
                    "v_pred": v_pred,
                    "v_target": v_target,
                    "x1_pred": x1_pred,
                    "x_1": x_1,
                    "x_0": x_0,
                    "t": t,
                    "target_valid": x_1_valid,
                    "prior_valid": x_0_valid,
                }
                if self.use_occupancy_flow:
                    out["occ_logit"] = unet_out[:, 1:2]
            else:
                hist_enc = history_encoded_all if (k > 0 and history_encoded_all is not None) else None
                out = self.forward_train(
                    current_pc=frames_pc[k],
                    target_pc=frames_pc[k + 1],
                    rel_pose=rel_poses[k],
                    action=actions[k],
                    current_mask=frames_mask[k],
                    target_mask=frames_mask[k + 1],
                    anchor_range=anchor_range,
                    history_encoded=hist_enc,
                    step_idx=k,
                )

            out["step_idx"] = k
            out["used_own_prediction"] = use_own_prediction
            outputs.append(out)

            prev_pred_range = out["x1_pred"].detach().clamp(-1, 1)
            prev_pred_valid = (prev_pred_range > self.validity_relaxed_threshold).float()

        if K >= 2 and getattr(self.cfg.training, "use_jump_prediction", False):
            jump_out = self._forward_jump(
                frames_pc=frames_pc,
                frames_mask=frames_mask,
                rel_poses=rel_poses,
                actions=actions,
                anchor_range=anchor_range,
                history_encoded=history_encoded_all,
                K=K,
            )
            if jump_out is not None:
                jump_out["is_jump"] = True
                jump_out["step_idx"] = K
                outputs.append(jump_out)

        return outputs

    def _forward_jump(
        self,
        frames_pc: list,
        frames_mask: list,
        rel_poses: list,
        actions: list,
        anchor_range: Optional[torch.Tensor],
        history_encoded: Optional[torch.Tensor],
        K: int,
    ) -> Optional[dict]:
        """Direct t->t+K jump prediction (auxiliary long-horizon supervision)."""
        rel_0_K = rel_poses[0].clone()
        for k in range(1, K):
            rel_0_K = torch.bmm(rel_0_K, rel_poses[k])
        x_0, x_0_valid = self.projector.warp_range_image(
            frames_pc[0], frames_mask[0], rel_0_K,
        )
        x_1, x_1_valid = self.projector.project(frames_pc[K], frames_mask[K])
        B = x_0.shape[0]
        device = x_0.device

        step_noise_scale = getattr(self.cfg.training, "step_noise_scale", 0.0)
        sigma = self.sigma_prior * (1.0 + step_noise_scale)
        x_0_noisy = x_0 + torch.randn_like(x_0) * sigma
        t = self.fm.sample_t(B, device)
        x_t = self.fm.interpolate(x_0_noisy, x_1, t)
        v_target = self.fm.target_velocity(x_0_noisy, x_1)

        parts = [x_t, x_0]
        if self.use_anchor and anchor_range is not None:
            parts.append(anchor_range)
        if self.use_history and self.history_encoder is not None:
            H, W = x_0.shape[2], x_0.shape[3]
            if history_encoded is not None:
                parts.append(history_encoded)
            elif self.history_placeholder is not None:
                parts.append(self.history_placeholder.expand(B, -1, H, W))
            else:
                parts.append(torch.zeros(B, 4, H, W, device=device, dtype=x_0.dtype))
        unet_input = torch.cat(parts, dim=1)

        emb_t = self.unet.time_embed(t)
        emb = emb_t
        if self.action_encoder is not None:
            action_agg = torch.stack(actions[:K], dim=0).mean(dim=0)
            emb = emb + self.action_encoder(action_agg)
        if self.step_index_encoder is not None:
            step_ratio = torch.ones(B, device=device)
            emb = emb + self.step_index_encoder(step_ratio)

        unet_out = self.unet.forward_with_emb(unet_input, emb)
        v_pred = unet_out[:, :1] if self.use_occupancy_flow else unet_out
        x1_pred = self.fm.predict_x1(x_t, v_pred, t).clamp(-1, 1)

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

    # -----------------------------------------------------------------
    #  Range-image inference (from range image, not 3D points)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def forward_inference_from_range(
        self,
        x_0: torch.Tensor,             # (B, 1, H, W) input range image
        x_0_valid: torch.Tensor,        # (B, 1, H, W) validity
        action: torch.Tensor,           # (B, 5)
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
        noise_std: Optional[float] = None,
        anchor_range: Optional[torch.Tensor] = None,
        skip_postprocess: bool = False,
        step_idx: Optional[int] = None,
    ) -> dict:
        """
        Inference directly from a range image.

        Args:
            skip_postprocess: If True, skip aggressive post-processing and
                use relaxed validity. Used for intermediate rollout steps to
                prevent cascading information loss.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        B = x_0.shape[0]
        device = x_0.device

        if noise_std is None:
            noise_std = self.inference_noise_std
        if noise_std > 0.0:
            x = x_0 + torch.randn_like(x_0) * noise_std
        else:
            x = x_0.clone()

        emb_base = torch.zeros(B, self.unet.emb_dim, device=device)
        if self.action_encoder is not None:
            emb_base = emb_base + self.action_encoder(action)
        if self.step_index_encoder is not None and step_idx is not None:
            K = getattr(self.cfg.training, "sequence_length", 6)
            step_ratio = torch.full((B,), step_idx / max(K - 1, 1), device=device)
            emb_base = emb_base + self.step_index_encoder(step_ratio)

        H, W = x_0.shape[2], x_0.shape[3]
        history_ch = torch.zeros(B, 4, H, W, device=device, dtype=x_0.dtype)
        if self.use_history and self.history_encoder is not None:
            if self.history_placeholder is not None:
                history_ch = self.history_placeholder.expand(B, -1, H, W)
        use_history_channels = self.use_history and self.history_encoder is not None

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)

            parts = [x, x_0]
            if self.use_anchor and anchor_range is not None:
                parts.append(anchor_range)
            if use_history_channels:
                parts.append(history_ch)
            unet_input = torch.cat(parts, dim=1)
            emb = self.unet.time_embed(t) + emb_base
            v1 = self.unet.forward_with_emb(unet_input, emb)
            if self.use_occupancy_flow:
                v1 = v1[:, :1]

            if use_midpoint and step < num_steps - 1:
                x_mid = x + v1 * dt
                t_mid = torch.full((B,), t_val + dt, device=device)
                parts_mid = [x_mid, x_0]
                if self.use_anchor and anchor_range is not None:
                    parts_mid.append(anchor_range)
                if use_history_channels:
                    parts_mid.append(history_ch)
                unet_input_mid = torch.cat(parts_mid, dim=1)
                emb_mid = self.unet.time_embed(t_mid) + emb_base
                v2 = self.unet.forward_with_emb(unet_input_mid, emb_mid)
                if self.use_occupancy_flow:
                    v2 = v2[:, :1]
                x = x + 0.5 * (v1 + v2) * dt
            else:
                x = x + v1 * dt

        pred_range = x.clamp(-1, 1)

        if skip_postprocess:
            pred_valid = (pred_range > self.validity_relaxed_threshold).float()
        else:
            pred_range = self._postprocess_range_image(pred_range, x_0, x_0_valid)
            pred_valid = self._compute_validity_mask(pred_range, x_0_valid)

        pred_points, pred_pt_mask = self.projector.unproject(pred_range, pred_valid)

        return {
            "pred_range": pred_range,
            "pred_points": pred_points,
            "pred_mask": pred_pt_mask,
            "x_0": x_0,
            "x_0_valid": x_0_valid,
        }

    @torch.no_grad()
    def forward_rollout_range_space(
        self,
        current_pc: torch.Tensor,        # (B, N, 3)
        current_mask: torch.Tensor,       # (B, N)
        rel_poses: list[torch.Tensor],    # K x (B, 4, 4)
        actions: list[torch.Tensor],      # K x (B, 5)
        num_steps: Optional[int] = None,
        use_midpoint: bool = True,
    ) -> list[dict]:
        """
        Multi-step rollout with anchor conditioning and soft intermediate filtering.

        Intermediate steps use relaxed validity and skip aggressive post-processing,
        preserving maximum information between steps.
        """
        K = len(rel_poses)
        if K == 0:
            return []

        anchor_range = None
        if self.use_anchor:
            anchor_range, _ = self.projector.project(current_pc, current_mask)

        results = []

        out = self.forward_inference(
            current_pc=current_pc,
            rel_pose=rel_poses[0],
            action=actions[0],
            current_mask=current_mask,
            num_steps=num_steps,
            use_midpoint=use_midpoint,
            anchor_range=anchor_range,
            step_idx=0,
        )
        results.append(out)

        for k in range(1, K):
            prev_range = results[-1]["pred_range"]
            relaxed_valid = (prev_range > self.validity_relaxed_threshold).float()
            prev_points, prev_mask = self.projector.unproject(prev_range, relaxed_valid)
            x_0, x_0_valid = self.projector.warp_range_image(
                prev_points, prev_mask, rel_poses[k],
            )

            is_last = (k == K - 1)
            out = self.forward_inference_from_range(
                x_0=x_0,
                x_0_valid=x_0_valid,
                action=actions[k],
                num_steps=num_steps,
                use_midpoint=use_midpoint,
                anchor_range=anchor_range,
                skip_postprocess=(not is_last),
                step_idx=k,
            )
            results.append(out)

        return results


# =====================================================================
#  Factory
# =====================================================================


def build_world_model(cfg) -> RangeWorldModel:
    """Build the range-image world model from config."""
    if getattr(cfg.model, "use_latent_fm", False):
        from src.model.latent_world_model import LatentRangeWorldModel
        model = LatentRangeWorldModel(cfg)
    else:
        model = RangeWorldModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[WorldModel] {n_params / 1e6:.1f}M trainable parameters")
    return model
