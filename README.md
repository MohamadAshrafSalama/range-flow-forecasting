# range-flow-forecasting

LiDAR scene forecasting via optimal transport conditional flow matching on range images.

The model predicts the next LiDAR sweep from a current sweep and an ego-motion estimate. It uses the ego-warped previous frame as a structured prior and learns a flow from that prior to the true target distribution. Autoregressive rollout enables multi-step prediction (t+1 through t+6).

---

## Method

### Range-Image Representation

Each LiDAR sweep is projected to a 32x1024 range image using spherical projection. Depth values are log-normalised to [-1, 1]. Invalid rays (no return) are masked during both training and evaluation. Circular convolutions are used throughout to respect the 360-degree azimuthal geometry.

### Ego-Motion Warping as Structured Prior

Rather than sampling from Gaussian noise, the flow is initialised from the ego-warped previous frame. The warped frame is obtained by projecting the current point cloud into the ego frame of the next timestep using the relative SE(3) pose. This structured prior concentrates probability mass near the correct answer and accelerates convergence.

### Optimal Transport Conditional Flow Matching

Flow matching defines a probability path between the prior x_0 (warped frame) and the target x_1 (true next frame). A UNet learns the velocity field v(x_t, t) such that integrating it from t=0 to t=1 transforms x_0 into x_1. The training objective is:

    L = || v_pred(x_t, t) - (x_1 - x_0) ||^2

weighted by validity masks to ignore invalid depth regions. Midpoint Runge-Kutta integration is used at inference.

### Architecture

- **UNet backbone**: Residual blocks with Adaptive Group Normalisation (AdaGN) for time conditioning. Spatial attention at configurable resolutions.
- **Circular convolutions**: `CircularPad2d` + standard `Conv2d` to handle wraparound at the azimuthal boundary.
- **Action encoder**: Linear projection of ego-motion velocity features into the time embedding space.
- **History encoder** (optional): Lightweight CNN that encodes H past frames into a spatial feature map concatenated with the UNet input.
- **Latent flow matching** (optional): Flow matching in a learned VAE latent space (`RangeImageVAE`) instead of raw range-image space.

### Training

- Mixed-precision (AMP) with gradient scaling
- Gradient accumulation across configurable steps
- Cosine learning rate schedule with linear warmup
- EMA (Exponential Moving Average) of weights for stable evaluation checkpoints
- DDP (Distributed Data Parallel) for multi-GPU training
- Scheduled sampling for multi-step consistency training

---

## Repository Structure

```
range-flow-forecasting/
├── config/
│   └── default.py          # DataConfig, ModelConfig, TrainingConfig, EvalConfig
├── src/
│   ├── data/
│   │   ├── range_image_utils.py   # Spherical projection, warping, circular ops
│   │   ├── temporal_dataset.py    # nuScenes pair and sequence datasets
│   │   └── bev_utils.py           # BEV occupancy projection and IoU loss
│   ├── model/
│   │   ├── world_model.py         # UNet, FlowMatchingScheduler, RangeWorldModel
│   │   ├── losses.py              # Composite training loss (depth, edge, ray, BEV)
│   │   ├── history_encoder.py     # Multi-frame history conditioning
│   │   ├── range_vae.py           # 2D VAE for latent flow matching
│   │   └── latent_world_model.py  # LatentRangeWorldModel
│   └── baselines/
│       ├── models.py              # Diffusion and deterministic baselines
│       ├── train_baselines.py
│       └── evaluate_baselines.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/
│   └── test_model.py
├── environment.yml
├── pyproject.toml
└── Dockerfile
```

---

## Setup

### Conda

```bash
conda env create -f environment.yml
conda activate range-flow
pip install -e .
```

### Docker

```bash
docker build -t range-flow .
docker run --gpus all -v /path/to/nuscenes:/data/nuscenes range-flow bash
```

---

## Data

This project uses the [nuScenes](https://www.nuscenes.org/) dataset with the HDL-32E LiDAR sensor (32 beams, 1024 azimuthal columns).

Set `cfg.data.data_root` to your nuScenes root directory, e.g. `/data/nuscenes`.

---

## Training

Single-GPU:

```bash
python scripts/train.py \
    --data-root /data/nuscenes \
    --save-dir ./checkpoints/run_01 \
    --epochs 100 \
    --batch-size 8
```

Multi-GPU (DDP):

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --data-root /data/nuscenes \
    --save-dir ./checkpoints/run_01 \
    --epochs 100 \
    --batch-size 4
```

Resume from checkpoint:

```bash
python scripts/train.py --resume ./checkpoints/run_01/latest.pth
```

Latent flow matching (VAE-based):

```bash
python scripts/train.py --model-type latent_flow_matching \
    --data-root /data/nuscenes \
    --save-dir ./checkpoints/run_latent
```

---

## Evaluation

Single-step evaluation:

```bash
python scripts/evaluate.py ./checkpoints/run_01/best_model.pth \
    --max-batches 100 \
    --output ./results/eval.json
```

Multi-step rollout evaluation (t+1 to t+6):

```bash
python scripts/evaluate.py ./checkpoints/run_01/best_model.pth \
    --rollout-steps 6 \
    --output ./results/eval_multistep.json
```

### Metrics

| Metric | Description |
|---|---|
| `ri_rmse` | Root mean squared error on valid range pixels |
| `ri_psnr` | Peak signal-to-noise ratio on valid range pixels |
| `chamfer_dist` | Bidirectional Chamfer distance between predicted and GT point clouds |
| `bev_iou` | Bird's-eye-view occupancy IoU at 256x256 grid |
| `free_fp_rate` | False-positive rate in free-space regions |

Multi-step metrics are reported per horizon: `t+1/chamfer_dist`, `t+2/chamfer_dist`, etc.

---

## Visualisation

Generate range-image and BEV comparison figures:

```bash
python scripts/visualize.py ./checkpoints/run_01/best_model.pth \
    --output ./demo_output \
    --n-samples 16 \
    --num-steps 50
```

Multi-step rollout visualisation:

```bash
python scripts/visualize.py ./checkpoints/run_01/best_model.pth \
    --output ./demo_output \
    --rollout-steps 6
```

---

## Baseline Comparisons

Two reference baselines are included:

- **`lidarcrafter_like`**: Conditional DDPM on range images (noise-conditioned diffusion)
- **`occworld_style`**: Deterministic UNet predicting depth and occupancy jointly

Train a baseline:

```bash
python -m src.baselines.train_baselines \
    --model-type lidarcrafter_like \
    --epochs 60 \
    --batch-size 8
```

Evaluate a baseline:

```bash
python -m src.baselines.evaluate_baselines \
    baselines/checkpoints/lidarcrafter_like/best_model.pth \
    --model-type lidarcrafter_like \
    --output baselines/results/lidarcrafter_eval.json
```

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT
