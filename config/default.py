"""
Configuration for the range-image world model.

Uses a range-image UNet with flow matching and a structured prior
(ego-motion-warped previous frame) for LiDAR scene forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class DataConfig:
    data_root: str = "/comm_priv/autonomous_zhang/nuscenes/"
    version: str = "v1.0-trainval"
    max_points: int = 35000  # raw points before projection (more is fine)
    train_split_ratio: float = 0.85
    prediction_horizon: int = 1
    skip_first_n_samples: int = 0
    load_intensity: bool = False  # whether to load intensity channel


@dataclass
class ModelConfig:
    # ---- range image ---- #
    ri_H: int = 32            # range image height (beams)
    ri_W: int = 1024          # range image width (azimuth bins)
    fov_up: float = 10.0      # degrees
    fov_down: float = -30.0   # degrees
    use_calibrated_beams: bool = True
    # nuScenes HDL-32E vertical beam calibration (degrees).
    # Order can be arbitrary; projector sorts top->bottom internally.
    beam_elevations_deg: tuple = (
        -30.67, -9.33, -29.33, -8.00, -28.00, -6.67, -26.67, -5.33,
        -25.33, -4.00, -24.00, -2.67, -22.67, -1.33, -21.33,  0.00,
        -20.00,  1.33, -18.67,  2.67, -17.33,  4.00, -16.00,  5.33,
        -14.67,  6.67, -13.33,  8.00, -12.00,  9.33, -10.67, 10.67,
    )
    in_channels: int = 1      # depth only  (2 if intensity is added)
    cond_channels: int = 1    # concatenated condition channels (warped prev)
    anchor_channels: int = 1  # anchor frame context channel (original observation memory)
    out_channels: int = 1     # velocity prediction channels

    # ---- history conditioning ---- #
    use_history: bool = True          # history only when k>0; k=0 uses learnable placeholder
    history_frames: int = 2
    use_step_index_conditioning: bool = True   # step k as conditioning (like time t in FM)
    use_bev_head: bool = False
    bev_size: int = 64
    bev_range_m: float = 64.0
    use_latent_fm: bool = False
    vae_latent_channels: int = 8
    vae_downsample: int = 4
    use_occupancy_flow: bool = False

    # ---- UNet ---- #
    base_channels: int = 64
    channel_mult: tuple = (1, 2, 4, 8)   # 64, 128, 256, 512
    num_res_blocks: int = 2
    attention_resolutions: tuple = (4, 8)  # at 4x and 8x downsampled
    num_heads: int = 8
    dropout: float = 0.0
    use_scale_shift_norm: bool = True       # AdaGN
    use_circular_pad: bool = True           # 360 degree wrap-around
    use_activation_checkpointing: bool = True

    # ---- flow matching ---- #
    sigma_prior: float = 0.02     # noise added to structured prior x_0
    inference_noise_std: float = 0.0  # disable stochastic noise during inference by default
    num_inference_steps: int = 50
    ode_method: str = "euler"     # euler | midpoint

    # ---- inference stabilisation ---- #
    enable_inference_postprocess: bool = True
    postprocess_outlier_threshold: float = 0.2
    postprocess_structure_weight: float = 0.0
    postprocess_structure_max_deviation: float = 0.12
    validity_relaxed_depth_m: float = 1.45
    validity_strict_depth_m: float = 2.0
    validity_smooth_base: float = 0.25
    validity_smooth_with_prior: float = 0.35

    # ---- action conditioning ---- #
    action_dim: int = 5    # [v_fwd, yaw_rate, dx, dy, dtheta]
    use_action_conditioning: bool = True

    # ---- EMA ---- #
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 50
    num_workers: int = 8
    seed: int = 42
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    accumulation_steps: int = 2
    use_cosine_schedule: bool = True
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # ---- loss weights ---- #
    lambda_velocity: float = 1.0      # primary FM velocity MSE
    lambda_x1: float = 0.5            # x1-prediction Huber
    lambda_edge: float = 0.2          # depth edge / gradient loss
    lambda_valid: float = 0.2         # valid-mask BCE
    lambda_freq: float = 0.02         # frequency sharpness loss
    lambda_empty: float = 0.5         # empty-space penalty
    lambda_ray: float = 0.3           # ray-aware depth loss
    lambda_angular: float = 0.05      # azimuth anti-radial smoothness
    lambda_occupancy: float = 0.03    # occupancy Dice/Tversky loss
    lambda_chamfer: float = 0.02      # lightweight point-set regulariser
    lambda_bev_iou: float = 0.0       # BEV IoU proxy loss
    lambda_occ_flow: float = 0.0      # occupancy flow auxiliary
    lambda_temporal: float = 0.15     # temporal consistency
    multistep_decay: float = 1.0      # step weight decay over rollout
    multistep_future_weight_power: float = 0.5
    multistep_own_pred_boost: float = 1.3
    multistep_ramp_epochs: int = 8
    step_noise_scale: float = 0.6     # scale prior noise by step when using own pred
    lambda_jump: float = 0.15         # jump prediction auxiliary loss
    lambda_cycle: float = 0.1         # cycle consistency (warp-back) loss
    occ_tversky_alpha: float = 0.7
    occ_tversky_beta: float = 0.3
    occ_dice_mix: float = 0.5
    chamfer_start_epoch: int = 6
    chamfer_every_n_steps: int = 8
    chamfer_max_pred_points: int = 512
    chamfer_max_gt_points: int = 512
    chamfer_norm_scale_m: float = 50.0
    chamfer_only_first_step: bool = False

    # ---- ODE consistency (train-test alignment) ---- #
    ode_consistency_prob: float = 0.2
    ode_consistency_start_epoch: int = 4
    ode_consistency_steps: int = 10

    # ---- t-weighting (emphasize large t, long horizon) ---- #
    velocity_t_weight: float = 0.5
    multistep_t3_t6_boost: float = 1.8
    use_jump_prediction: bool = True
    use_cycle_consistency: bool = True

    # ---- multi-step / scheduled sampling ---- #
    enable_multistep: bool = True
    sequence_length: int = 6
    ss_start_epoch: int = 5
    ss_final_prob: float = 0.85
    ss_rampup_epochs: int = 30
    multistep_batch_size: int = 2
    multistep_accumulation_steps: int = 2

    # ---- progressive multi-step curriculum ---- #
    ms_curriculum: bool = True
    ms_min_steps: int = 2
    ms_step_increase_interval: int = 3

    # ---- depth range ---- #
    ray_depth_min_m: float = 1.45
    ray_depth_max_m: float = 80.0

    # ---- directories ---- #
    save_dir: str = "./checkpoints"
    vis_dir: str = "./visualizations"
    log_dir: str = "./checkpoints/logs"
    save_interval: int = 1
    eval_interval: int = 5
    log_interval: int = 10


@dataclass
class EvalConfig:
    chamfer_subsample: int = 10000
    bev_eval_size: int = 256
    bev_eval_range_m: float = 64.0
    eval_inference_steps: int = 50
    eval_rollout_steps: int = 6
    results_dir: str = "./results"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        import dataclasses
        def _conv(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _conv(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, tuple):
                return list(obj)
            return obj
        return _conv(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for section in ("data", "model", "training", "evaluation"):
            sub = d.get(section, {})
            obj = getattr(cfg, section)
            for k, v in sub.items():
                if hasattr(obj, k):
                    expected = type(getattr(obj, k))
                    if expected is tuple and isinstance(v, list):
                        v = tuple(v)
                    setattr(obj, k, v)
        return cfg

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def get_default_config() -> Config:
    """Returns default config."""
    return Config()
