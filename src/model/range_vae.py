"""
2D Range-Image VAE for compressing LiDAR range images.

Compresses range images (B, 1, 32, 1024) with log-depth [-1, 1] and circular
azimuth to latent (B, latent_ch, H', W') with downsampling factor 4 or 8.
Uses CircularConv2d and CircularPad2d for 360 degree consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.range_image_utils import CircularConv2d, CircularPad2d


@dataclass
class RangeVAEConfig:
    """Configuration for RangeImageVAE."""

    in_channels: int = 1
    ri_H: int = 32
    ri_W: int = 1024

    latent_channels: int = 8
    downsample: int = 4  # 4 or 8

    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2

    def __post_init__(self):
        assert self.downsample in (4, 8), "downsample must be 4 or 8"
        if self.downsample == 4:
            self.num_down_levels = 2
        else:
            self.num_down_levels = 3


# =====================================================================
#  Building Blocks
# =====================================================================


class ResBlockVAE(nn.Module):
    """Residual block for VAE (no conditioning)."""

    def __init__(self, in_ch: int, out_ch: int, use_circular: bool = True):
        super().__init__()
        Conv = (
            CircularConv2d
            if use_circular
            else lambda i, o, k=3, **kw: nn.Conv2d(i, o, k, padding=k // 2, **kw)
        )

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = Conv(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = Conv(out_ch, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        nn.init.zeros_(self.conv2.conv.weight if isinstance(self.conv2, CircularConv2d) else self.conv2.weight)
        nn.init.zeros_(self.conv2.conv.bias if isinstance(self.conv2, CircularConv2d) else self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample2dVAE(nn.Module):
    """Strided conv downsampling (factor 2)."""

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


class Upsample2dVAE(nn.Module):
    """Nearest-neighbour upsample (factor 2) + circular conv."""

    def __init__(self, channels: int, use_circular: bool = True):
        super().__init__()
        Conv = (
            CircularConv2d
            if use_circular
            else lambda i, o, k=3, **kw: nn.Conv2d(i, o, k, padding=k // 2, **kw)
        )
        self.conv = Conv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# =====================================================================
#  Range-Image VAE
# =====================================================================


class RangeImageVAE(nn.Module):
    """
    2D range-image VAE with circular convolution for 360 degree LiDAR consistency.

    Input:  (B, 1, H, W) log-depth normalised to [-1, 1], circular on W
    Latent: (B, latent_ch, H', W') where H'=H//downsample, W'=W//downsample
    Output: (B, 1, H, W) reconstructed range image
    """

    def __init__(self, cfg: RangeVAEConfig | None = None, use_circular: bool = True):
        super().__init__()
        cfg = cfg or RangeVAEConfig()
        self.cfg = cfg
        self.use_circular = use_circular

        ch = cfg.base_channels
        channel_mult = list(cfg.channel_mult)
        num_levels = cfg.num_down_levels

        while len(channel_mult) < num_levels + 1:
            channel_mult.append(channel_mult[-1] * 2)
        channel_mult = channel_mult[: num_levels + 1]

        # ---- Encoder ---- #
        self.enc_conv_in = (
            CircularConv2d(cfg.in_channels, ch)
            if use_circular
            else nn.Conv2d(cfg.in_channels, ch, 3, padding=1)
        )

        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        cur_ch = ch

        for level in range(num_levels):
            out_ch = ch * channel_mult[level]
            for _ in range(cfg.num_res_blocks):
                self.enc_blocks.append(ResBlockVAE(cur_ch, out_ch, use_circular))
                cur_ch = out_ch
            self.enc_downs.append(Downsample2dVAE(cur_ch, use_circular))

        out_ch = ch * channel_mult[num_levels]
        self.enc_final = nn.Sequential(
            ResBlockVAE(cur_ch, out_ch, use_circular),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
        )
        cur_ch = out_ch

        self.enc_to_mu = (
            CircularConv2d(cur_ch, cfg.latent_channels, 3)
            if use_circular
            else nn.Conv2d(cur_ch, cfg.latent_channels, 3, padding=1)
        )
        self.enc_to_logvar = (
            CircularConv2d(cur_ch, cfg.latent_channels, 3)
            if use_circular
            else nn.Conv2d(cur_ch, cfg.latent_channels, 3, padding=1)
        )

        # ---- Decoder ---- #
        self.dec_conv_in = (
            CircularConv2d(cfg.latent_channels, cur_ch, 3)
            if use_circular
            else nn.Conv2d(cfg.latent_channels, cur_ch, 3, padding=1)
        )

        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()

        for level in range(num_levels - 1, -1, -1):
            self.dec_ups.append(Upsample2dVAE(cur_ch, use_circular))
            out_ch = ch * channel_mult[level]
            for _ in range(cfg.num_res_blocks):
                self.dec_blocks.append(ResBlockVAE(cur_ch, out_ch, use_circular))
                cur_ch = out_ch

        self.dec_final = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            nn.SiLU(),
            (
                CircularConv2d(cur_ch, cfg.in_channels, 3)
                if use_circular
                else nn.Conv2d(cur_ch, cfg.in_channels, 3, padding=1)
            ),
        )

        self._latent_shape = None

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode range image to latent mean and log-variance."""
        h = self.enc_conv_in(x)

        for blocks, down in zip(
            [
                self.enc_blocks[i : i + self.cfg.num_res_blocks]
                for i in range(0, len(self.enc_blocks), self.cfg.num_res_blocks)
            ],
            self.enc_downs,
        ):
            for block in blocks:
                h = block(h)
            h = down(h)

        h = self.enc_final(h)
        mu = self.enc_to_mu(h)
        logvar = self.enc_to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to range image."""
        h = self.dec_conv_in(z)

        ups = list(self.dec_ups)
        blocks_list = [
            self.dec_blocks[i : i + self.cfg.num_res_blocks]
            for i in range(0, len(self.dec_blocks), self.cfg.num_res_blocks)
        ]

        for up, blocks in zip(ups, blocks_list):
            h = up(h)
            for block in blocks:
                h = block(h)

        return self.dec_final(h)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x: (B, 1, H, W) range image, log-depth [-1, 1], circular on W

        Returns:
            recon: (B, 1, H, W) reconstructed range image
            mu:    (B, latent_ch, H', W') latent mean
            logvar:(B, latent_ch, H', W') latent log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @property
    def latent_spatial_shape(self) -> Tuple[int, int]:
        """(H', W') of the latent."""
        d = self.cfg.downsample
        return self.cfg.ri_H // d, self.cfg.ri_W // d
