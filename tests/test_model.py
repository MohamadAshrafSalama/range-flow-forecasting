"""
Smoke tests for the range-image world model.

Runs minimal forward passes with random tensors to verify that shapes,
imports, and module wiring are correct without requiring a GPU or dataset.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.default import Config, get_default_config
from src.model.world_model import build_world_model, RangeUNet, FlowMatchingScheduler
from src.data.range_image_utils import CircularConv2d, CircularPad2d


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _small_config() -> Config:
    """Return a Config with tiny dimensions suitable for CPU smoke tests."""
    cfg = get_default_config()
    cfg.model.ri_H = 16
    cfg.model.ri_W = 64
    cfg.model.base_channels = 16
    cfg.model.channel_mult = (1, 2)
    cfg.model.num_res_blocks = 1
    cfg.model.attention_resolutions = ()
    cfg.model.num_heads = 1
    cfg.model.use_history = False
    cfg.model.use_action_conditioning = False
    cfg.model.use_anchor = False
    cfg.model.model_type = "flow_matching"
    cfg.training.ray_depth_max_m = 80.0
    cfg.training.ray_depth_min_m = 1.0
    cfg.data.lidar_h_fov_deg = 360.0
    cfg.data.lidar_v_fov_min_deg = -30.0
    cfg.data.lidar_v_fov_max_deg = 10.0
    return cfg


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

def test_circular_pad():
    pad = CircularPad2d(1)
    x = torch.randn(1, 3, 8, 16)
    out = pad(x)
    assert out.shape == (1, 3, 10, 18), f"Unexpected shape: {out.shape}"


def test_circular_conv():
    conv = CircularConv2d(4, 8, kernel_size=3)
    x = torch.randn(2, 4, 16, 64)
    out = conv(x)
    assert out.shape == (2, 8, 16, 64), f"Unexpected shape: {out.shape}"


def test_flow_matching_scheduler():
    fm = FlowMatchingScheduler()
    B = 4
    x0 = torch.randn(B, 1, 16, 64)
    x1 = torch.randn(B, 1, 16, 64)
    t = fm.sample_t(B, device=torch.device("cpu"))
    assert t.shape == (B,)
    assert (t >= 0).all() and (t <= 1).all()

    xt = fm.interpolate(x0, x1, t)
    assert xt.shape == x0.shape

    v = fm.target_velocity(x0, x1)
    assert v.shape == x0.shape


def test_range_unet_forward():
    cfg = _small_config()
    model = build_world_model(cfg)
    model.eval()

    B = 2
    H, W = cfg.model.ri_H, cfg.model.ri_W
    in_ch = model.unet.in_channels if hasattr(model.unet, "in_channels") else 2

    x = torch.randn(B, in_ch, H, W)
    t = torch.rand(B)
    emb = model.unet.time_embed(t)
    out = model.unet.forward_with_emb(x, emb)
    assert out.shape == (B, 1, H, W), f"Unexpected output shape: {out.shape}"


def test_world_model_forward_train():
    cfg = _small_config()
    model = build_world_model(cfg)
    model.eval()

    B = 2
    N = 512
    current_pc = torch.rand(B, N, 3) * 20 - 10
    target_pc = torch.rand(B, N, 3) * 20 - 10
    current_mask = torch.ones(B, N, dtype=torch.bool)
    target_mask = torch.ones(B, N, dtype=torch.bool)
    rel_pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    action = torch.zeros(B, cfg.model.action_dim)

    with torch.no_grad():
        out = model.forward_train(
            current_pc=current_pc,
            target_pc=target_pc,
            rel_pose=rel_pose,
            action=action,
            current_mask=current_mask,
            target_mask=target_mask,
        )

    assert "v_pred" in out
    assert "v_target" in out
    assert "x1_pred" in out
    H, W = cfg.model.ri_H, cfg.model.ri_W
    assert out["x1_pred"].shape == (B, 1, H, W), f"Bad x1_pred shape: {out['x1_pred'].shape}"


def test_world_model_forward_inference():
    cfg = _small_config()
    cfg.model.num_inference_steps = 2
    model = build_world_model(cfg)
    model.eval()

    B = 1
    N = 256
    current_pc = torch.rand(B, N, 3) * 20 - 10
    current_mask = torch.ones(B, N, dtype=torch.bool)
    rel_pose = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
    action = torch.zeros(B, cfg.model.action_dim)

    with torch.no_grad():
        out = model.forward_inference(
            current_pc=current_pc,
            rel_pose=rel_pose,
            action=action,
            current_mask=current_mask,
            num_steps=2,
        )

    assert "pred_range" in out
    assert "pred_points" in out
    assert "pred_mask" in out
    H, W = cfg.model.ri_H, cfg.model.ri_W
    assert out["pred_range"].shape == (B, 1, H, W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
