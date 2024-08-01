"""
History conditioning module for multi-step LiDAR forecasting.

Encodes H past range images (downsampled) and provides a compact
context embedding for concatenation with UNet input.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.range_image_utils import CircularConv2d, CircularPad2d


class HistoryEncoder(nn.Module):
    """
    Encodes H past range images into a compact spatial feature map.

    Each history frame is downsampled (e.g. 32x1024 -> 8x256), then
    concatenated along channel dim and processed by a light CNN.
    Output: (B, out_channels, H', W') for concatenation with UNet input.
    """

    def __init__(
        self,
        num_history: int = 2,
        in_channels: int = 1,
        out_channels: int = 4,
        ri_H: int = 32,
        ri_W: int = 1024,
        downsample: int = 4,
        use_circular: bool = True,
    ):
        super().__init__()
        self.num_history = num_history
        self.downsample = downsample
        self.out_H = ri_H // downsample
        self.out_W = ri_W // downsample

        Conv = (
            CircularConv2d
            if use_circular
            else lambda i, o, k=3, **kw: nn.Conv2d(i, o, k, padding=k // 2, **kw)
        )

        self.per_frame = nn.Sequential(
            Conv(in_channels, 16, 3),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.AvgPool2d(downsample, stride=downsample),
            Conv(16, 32, 3),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )

        fuse_in = 32 * num_history
        self.fuse = nn.Sequential(
            Conv(fuse_in, 64, 3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            Conv(64, out_channels, 3),
        )

    def forward(
        self,
        history_range: torch.Tensor,  # (B, H, 1, ri_H, ri_W) or (B*H, 1, ri_H, ri_W)
        history_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_range: (B, num_history, 1, H, W) range images
        Returns:
            (B, out_channels, H', W') encoded history features
        """
        B = history_range.shape[0]
        H = history_range.shape[1]
        assert H == self.num_history

        feats = []
        for i in range(H):
            x = history_range[:, i]  # (B, 1, ri_H, ri_W)
            f = self.per_frame(x)  # (B, 32, H', W')
            feats.append(f)

        fused = torch.cat(feats, dim=1)  # (B, 32*H, H', W')
        return self.fuse(fused)  # (B, out_channels, H', W')
