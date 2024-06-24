"""
BEV (Bird's Eye View) occupancy utilities for LiDAR world models.

Projects range images to 2D BEV occupancy grids for auxiliary supervision
and IoU proxy loss. Follows evaluation convention (x forward, y left).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def range_to_bev_occupancy(
    range_img: torch.Tensor,  # (B, 1, H, W) log-depth [-1, 1]
    valid_mask: torch.Tensor,  # (B, 1, H, W)
    ri_H: int = 32,
    ri_W: int = 1024,
    fov_up_deg: float = 10.0,
    fov_down_deg: float = -30.0,
    max_depth: float = 80.0,
    bev_size: int = 64,
    bev_range_m: float = 64.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project range image to BEV occupancy grid.

    BEV convention: x forward, y left (nuScenes). Grid indexed [y, x] with
    origin at center. bev_occ[b, 0, y, x] = 1 if occupied.

    Returns:
        bev_occ: (B, 1, bev_size, bev_size) float [0, 1] occupancy
        bev_valid: (B, 1, bev_size, bev_size) valid mask
    """
    B = range_img.shape[0]
    device = range_img.device

    fov_up = math.radians(fov_up_deg)
    fov_down = math.radians(fov_down_deg)

    log_max = math.log2(max_depth + 1.0)
    depth = ((range_img[:, 0] + 1.0) * 0.5).clamp(0, 1)
    depth = torch.pow(2.0, depth * log_max) - 1.0

    elev = torch.linspace(fov_up, fov_down, ri_H, device=device)
    azim = torch.linspace(0, 2 * math.pi, ri_W + 1, device=device)[:-1]
    elev_g, azim_g = torch.meshgrid(elev, azim, indexing="ij")

    cos_e = torch.cos(elev_g)
    x = depth * cos_e * torch.cos(azim_g)  # forward
    y = depth * cos_e * torch.sin(azim_g)  # left

    valid = (valid_mask[:, 0] > 0.5) & (depth > 1.0) & (depth < max_depth - 1.0)

    scale = bev_size / (2 * bev_range_m)
    cx = bev_size // 2
    cy = bev_size // 2

    xi = (x * scale + cx).long().clamp(0, bev_size - 1)
    yi = (-y * scale + cy).long().clamp(0, bev_size - 1)

    bev_occ = torch.zeros(B, 1, bev_size, bev_size, device=device, dtype=range_img.dtype)
    bev_valid = torch.zeros(B, 1, bev_size, bev_size, device=device, dtype=torch.float32)

    for b in range(B):
        v = valid[b]
        xb, yb = xi[b][v], yi[b][v]
        if xb.numel() > 0:
            bev_occ[b, 0, yb, xb] = 1.0
            bev_valid[b, 0, yb, xb] = 1.0

    return bev_occ, bev_valid


def points_to_bev_occupancy(
    points: torch.Tensor,  # (B, N, 3) x,y,z in ego frame
    pt_mask: torch.Tensor,  # (B, N)
    bev_size: int = 64,
    bev_range_m: float = 64.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize 3D points to BEV occupancy.
    Returns (bev_occ, bev_valid) each (B, 1, bev_size, bev_size).
    """
    B, N, _ = points.shape
    device = points.device
    scale = bev_size / (2 * bev_range_m)
    cx = cy = bev_size // 2

    x, y = points[..., 0], points[..., 1]
    xi = (x * scale + cx).long().clamp(0, bev_size - 1)
    yi = (-y * scale + cy).long().clamp(0, bev_size - 1)

    bev_occ = torch.zeros(B, 1, bev_size, bev_size, device=device)
    bev_valid = torch.zeros(B, 1, bev_size, bev_size, device=device)

    for b in range(B):
        v = pt_mask[b]
        xb, yb = xi[b][v], yi[b][v]
        if xb.numel() > 0:
            bev_occ[b, 0, yb, xb] = 1.0
            bev_valid[b, 0, yb, xb] = 1.0

    return bev_occ, bev_valid


def bev_iou_loss(
    pred_occ: torch.Tensor,  # (B, 1, H, W) logits or [0,1]
    gt_occ: torch.Tensor,  # (B, 1, H, W) binary
    valid: torch.Tensor,  # (B, 1, H, W)
) -> torch.Tensor:
    """
    Soft IoU loss for BEV occupancy (differentiable proxy).
    pred_occ and gt_occ in [0, 1]; valid masks where to compute.
    """
    pred = pred_occ.clamp(0, 1) * valid
    gt = gt_occ * valid
    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = (pred + gt - pred * gt).sum(dim=(1, 2, 3)).clamp(min=1e-6)
    iou = (inter / union).mean()
    return 1.0 - iou
