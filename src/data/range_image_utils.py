"""
Range image utilities for LiDAR scene forecasting.

Handles spherical projection of 3D LiDAR points to/from 2D range images,
log-depth encoding, circular padding, and ego-motion warping in range-image space.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeImageProjector:
    """
    Projects 3D LiDAR points to/from 2D range images via spherical projection.

    Range image layout:
        - Rows (H): elevation angles, top to bottom (fov_up -> fov_down)
        - Cols (W): azimuth angles, 360 degree circular (left to right)
        - Values: log-compressed depth normalized to [-1, 1]
        - Invalid pixels (no return): -1

    nuScenes HDL-32E: 32 beams, FOV [-30 deg, +10 deg], 360 deg azimuth
    """

    def __init__(
        self,
        H: int = 32,
        W: int = 1024,
        fov_up: float = 10.0,
        fov_down: float = -30.0,
        max_depth: float = 80.0,
        min_depth: float = 1.45,
        beam_elevations_deg: Optional[Tuple[float, ...]] = None,
    ):
        self.H = H
        self.W = W
        self.fov_up = math.radians(fov_up)
        self.fov_down = math.radians(fov_down)
        self.fov_range = self.fov_up - self.fov_down
        self.max_depth = max_depth
        self.min_depth = min_depth
        self._log_max = math.log2(max_depth + 1.0)

        # Pre-compute ray directions for unprojection
        self._ray_dirs = None
        self._beam_elevations_base = None       # (H,) radians, sorted high->low
        self._beam_boundaries_asc_base = None   # (H-1,) radians, low->high
        self._beam_cache_device = None
        self._beam_elevations_device = None
        self._beam_boundaries_device = None

        if beam_elevations_deg is not None and len(beam_elevations_deg) > 0:
            beam = torch.as_tensor(beam_elevations_deg, dtype=torch.float32)
            if beam.numel() == self.H:
                beam = torch.sort(beam, descending=True).values
                self._beam_elevations_base = torch.deg2rad(beam)
                if self.H > 1:
                    asc = self._beam_elevations_base.flip(0)
                    self._beam_boundaries_asc_base = 0.5 * (asc[:-1] + asc[1:])
                self.fov_up = float(self._beam_elevations_base[0].item())
                self.fov_down = float(self._beam_elevations_base[-1].item())
                self.fov_range = max(self.fov_up - self.fov_down, 1e-6)
            else:
                print(
                    f"[RangeImageProjector] beam profile length {beam.numel()} "
                    f"!= H ({self.H}); fallback to linear FOV mapping.",
                )

    # ---- depth encoding / decoding ---- #

    def encode_depth(self, metric: torch.Tensor) -> torch.Tensor:
        """Metric depth [m] -> normalised log-depth in [-1, 1]."""
        log_d = torch.log2(metric.clamp(min=1e-6) + 1.0) / self._log_max  # [0, 1]
        return log_d * 2.0 - 1.0  # [-1, 1]

    def decode_depth(self, normalised: torch.Tensor) -> torch.Tensor:
        """Normalised log-depth [-1, 1] -> metric depth [m]."""
        unit = (normalised + 1.0) * 0.5  # [0, 1]
        metric = torch.pow(2.0, unit * self._log_max) - 1.0
        return metric.clamp(min=0.0)

    # ---- projection ---- #

    def _prepare_beam_cache(self, device: torch.device):
        if self._beam_elevations_base is None:
            return
        if self._beam_cache_device == device:
            return
        self._beam_elevations_device = self._beam_elevations_base.to(device=device)
        if self._beam_boundaries_asc_base is None:
            self._beam_boundaries_device = None
        else:
            self._beam_boundaries_device = self._beam_boundaries_asc_base.to(device=device)
        self._beam_cache_device = device

    def _get_beam_elevations(self, device: torch.device) -> Optional[torch.Tensor]:
        self._prepare_beam_cache(device)
        return self._beam_elevations_device

    def _get_beam_boundaries(self, device: torch.device) -> Optional[torch.Tensor]:
        self._prepare_beam_cache(device)
        return self._beam_boundaries_device

    def _elevation_to_row(self, elev: torch.Tensor) -> torch.Tensor:
        boundaries = self._get_beam_boundaries(elev.device)
        if boundaries is None:
            return torch.zeros_like(elev, dtype=torch.long)
        idx_asc = torch.bucketize(elev, boundaries)
        row = self.H - 1 - idx_asc
        return row.long().clamp(0, self.H - 1)

    def _get_ray_directions(self, device: torch.device) -> torch.Tensor:
        """Returns (H, W, 3) unit ray direction vectors."""
        if self._ray_dirs is not None and self._ray_dirs.device == device:
            return self._ray_dirs

        beam_elev = self._get_beam_elevations(device)
        if beam_elev is not None:
            elev = beam_elev
        else:
            elev = torch.linspace(self.fov_up, self.fov_down, self.H, device=device)
        azim = torch.linspace(0, 2 * math.pi, self.W + 1, device=device)[:-1]

        elev_grid, azim_grid = torch.meshgrid(elev, azim, indexing="ij")
        cos_e = torch.cos(elev_grid)
        dirs = torch.stack(
            [
                cos_e * torch.cos(azim_grid),  # x
                cos_e * torch.sin(azim_grid),  # y
                torch.sin(elev_grid),           # z
            ],
            dim=-1,
        )  # (H, W, 3)
        self._ray_dirs = dirs
        return dirs

    @torch.no_grad()
    def project(
        self,
        points: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to a range image.

        Args:
            points: (B, N, 3) point coordinates in ego frame
            masks:  (B, N) boolean valid mask (optional)

        Returns:
            range_img: (B, 1, H, W) log-depth normalised to [-1, 1]
            valid:     (B, 1, H, W) binary mask
        """
        B, N, _ = points.shape
        device = points.device

        if masks is None:
            masks = torch.ones(B, N, dtype=torch.bool, device=device)

        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r_xy = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        depth = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-8)

        elev = torch.atan2(z, r_xy)                  # [-pi/2, pi/2]
        azim = torch.atan2(y, x)                       # [-pi, pi]
        azim = azim % (2 * math.pi)                    # [0, 2*pi)

        if self._beam_elevations_base is not None:
            row = self._elevation_to_row(elev)
        else:
            row = (1.0 - (elev - self.fov_down) / self.fov_range) * (self.H - 1)
            row = row.round().long().clamp(0, self.H - 1)
        col = azim / (2 * math.pi) * self.W
        col = torch.floor(col).long() % self.W

        if self._beam_elevations_base is not None:
            tol = math.radians(0.75)
            in_fov = (elev >= (self.fov_down - tol)) & (elev <= (self.fov_up + tol))
        else:
            in_fov = (elev >= self.fov_down) & (elev <= self.fov_up)
        valid_pts = (
            masks
            & (depth >= self.min_depth)
            & (depth <= self.max_depth)
            & in_fov
        )

        range_img = torch.full(
            (B, self.H, self.W), fill_value=self.max_depth + 10.0,
            device=device, dtype=points.dtype,
        )

        for b in range(B):
            v = valid_pts[b]
            r_b, c_b, d_b = row[b][v], col[b][v], depth[b][v]
            if r_b.numel() == 0:
                continue
            idx = r_b * self.W + c_b
            flat = range_img[b].view(-1)
            d_b = d_b.to(flat.dtype)
            flat.scatter_reduce_(0, idx, d_b, reduce="amin", include_self=True)

        valid_mask = range_img < (self.max_depth + 5.0)
        range_img[~valid_mask] = 0.0

        encoded = self.encode_depth(range_img)
        encoded[~valid_mask] = -1.0  # invalid pixels = -1

        return encoded.unsqueeze(1), valid_mask.unsqueeze(1).float()

    @torch.no_grad()
    def unproject(
        self,
        range_img: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unproject range image to 3D points.

        Args:
            range_img:  (B, 1, H, W) log-depth normalised to [-1, 1]
            valid_mask: (B, 1, H, W) optional binary mask

        Returns:
            points: (B, N_max, 3) 3D points
            pt_mask: (B, N_max) boolean mask
        """
        B = range_img.shape[0]
        device = range_img.device

        metric_depth = self.decode_depth(range_img[:, 0])  # (B, H, W)

        if valid_mask is not None:
            valid = valid_mask[:, 0] > 0.5
        else:
            valid = metric_depth > self.min_depth

        dirs = self._get_ray_directions(device)  # (H, W, 3)

        xyz = dirs.unsqueeze(0) * metric_depth.unsqueeze(-1)  # (B, H, W, 3)

        N_max = valid.sum(dim=(1, 2)).max().item()
        if N_max == 0:
            N_max = 1

        points = torch.zeros(B, N_max, 3, device=device, dtype=range_img.dtype)
        pt_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for b in range(B):
            v = valid[b]
            pts = xyz[b][v]
            n = min(pts.shape[0], N_max)
            points[b, :n] = pts[:n]
            pt_mask[b, :n] = True

        return points, pt_mask

    @torch.no_grad()
    def warp_range_image(
        self,
        points_3d: torch.Tensor,
        masks: torch.Tensor,
        rel_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ego-motion warping: transform 3D points then re-project.

        Args:
            points_3d: (B, N, 3) current frame points
            masks:     (B, N) validity
            rel_pose:  (B, 4, 4) transform from current to next ego frame

        Returns:
            warped_ri: (B, 1, H, W) warped range image
            warped_vm: (B, 1, H, W) valid mask
        """
        R = rel_pose[:, :3, :3]  # (B, 3, 3)
        t = rel_pose[:, :3, 3:]  # (B, 3, 1)
        warped = torch.bmm(points_3d, R.transpose(1, 2)) + t.transpose(1, 2)
        return self.project(warped, masks)


# ---- Circular padding layers ---- #


class CircularPad2d(nn.Module):
    """
    Pad with circular wrapping on horizontal axis (360 degree LiDAR geometry)
    and constant/zero on vertical axis.
    """

    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.padding
        x = F.pad(x, (p, p, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, p, p), mode="constant", value=-1.0)
        return x


class CircularConv2d(nn.Module):
    """Conv2d with circular padding on horizontal axis."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.pad = CircularPad2d(kernel_size // 2) if kernel_size > 1 else None
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad is not None:
            x = self.pad(x)
        return self.conv(x)


def build_projector_from_cfg(cfg) -> RangeImageProjector:
    """Factory from config object."""
    H = getattr(cfg.model, "ri_H", 32)
    beam_profile = None
    if getattr(cfg.model, "use_calibrated_beams", False):
        candidate = getattr(cfg.model, "beam_elevations_deg", None)
        if candidate is not None and len(candidate) == H:
            beam_profile = tuple(float(x) for x in candidate)
    return RangeImageProjector(
        H=H,
        W=getattr(cfg.model, "ri_W", 1024),
        fov_up=getattr(cfg.model, "fov_up", 10.0),
        fov_down=getattr(cfg.model, "fov_down", -30.0),
        max_depth=getattr(cfg.training, "ray_depth_max_m", 80.0),
        min_depth=getattr(cfg.training, "ray_depth_min_m", 1.45),
        beam_elevations_deg=beam_profile,
    )
