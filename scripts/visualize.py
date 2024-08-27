"""
Demo visualisation for the range-image world model.

Generates:
  1. Side-by-side range image comparisons (warped prior, GT, predicted)
  2. BEV scatter plots of unprojected 3D points
  3. Multi-step (t+1 to t+6) rollout visualisations
  4. MP4 video of sequential predictions
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.default import Config
from src.model.world_model import build_world_model
from src.data.range_image_utils import RangeImageProjector, build_projector_from_cfg


def make_range_image_figure(x_0, x_1, x1_pred, sample_idx: int = 0):
    """Create 3-row range image comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(18, 7), dpi=120)

    kwargs = dict(cmap="turbo", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    axes[0].imshow(x_0[sample_idx, 0].cpu().numpy(), **kwargs)
    axes[0].set_title("Ego-Warped Previous Frame (Structured Prior x0)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(x_1[sample_idx, 0].cpu().numpy(), **kwargs)
    axes[1].set_title("Ground Truth Next Frame (x1)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(x1_pred[sample_idx, 0].cpu().numpy(), **kwargs)
    axes[2].set_title("Predicted Next Frame (x1_hat)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def make_bev_figure(
    gt_pts,
    pred_pts,
    gt_mask=None,
    pred_mask=None,
    sample_idx: int = 0,
    range_m: float = 60.0,
):
    """Create BEV scatter comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)

    gt = gt_pts[sample_idx]
    pred = pred_pts[sample_idx]
    if gt_mask is not None:
        gt = gt[gt_mask[sample_idx]]
    if pred_mask is not None:
        pred = pred[pred_mask[sample_idx]]
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()

    for ax, pts, title in [
        (axes[0], gt, "Ground Truth BEV"),
        (axes[1], pred, "Predicted BEV"),
    ]:
        mask = np.abs(pts).max(axis=1) < range_m
        pts = pts[mask]
        if pts.shape[0] == 0:
            ax.set_xlim(-range_m, range_m)
            ax.set_ylim(-range_m, range_m)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True, alpha=0.2)
            continue
        colors = np.clip((pts[:, 2] + 3) / 8, 0, 1)
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap="viridis", s=0.15, alpha=0.7)
        ax.set_xlim(-range_m, range_m)
        ax.set_ylim(-range_m, range_m)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


@torch.no_grad()
def run_demo(
    model,
    val_loader,
    projector: RangeImageProjector,
    output_dir: str,
    n_samples: int = 8,
    num_inference_steps: int = 20,
):
    """Generate demo outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    out_dir = Path(output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    sample_count = 0

    for batch in val_loader:
        if sample_count >= n_samples:
            break

        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        B = batch["current_pc"].shape[0]

        raw_model = model.module if hasattr(model, "module") else model

        train_out = raw_model.forward_train(
            current_pc=batch["current_pc"],
            target_pc=batch["target_pc"],
            rel_pose=batch["relative_pose"],
            action=batch["action_features"],
            current_mask=batch["current_mask"],
            target_mask=batch["target_mask"],
        )

        inf_out = raw_model.forward_inference(
            current_pc=batch["current_pc"],
            rel_pose=batch["relative_pose"],
            action=batch["action_features"],
            current_mask=batch["current_mask"],
            num_steps=num_inference_steps,
        )

        x_0 = train_out["x_0"]
        x_1 = train_out["x_1"]
        x1_pred = inf_out["pred_range"]

        gt_pts, gt_mask = projector.unproject(x_1, train_out["target_valid"])

        for i in range(min(B, n_samples - sample_count)):
            fig = make_range_image_figure(x_0, x_1, x1_pred, i)
            fig.savefig(frames_dir / f"range_{sample_count:03d}.png", bbox_inches="tight")
            plt.close(fig)
            all_frames.append(str(frames_dir / f"range_{sample_count:03d}.png"))

            fig = make_bev_figure(
                gt_pts, inf_out["pred_points"], gt_mask, inf_out["pred_mask"], i,
            )
            fig.savefig(frames_dir / f"bev_{sample_count:03d}.png", bbox_inches="tight")
            plt.close(fig)

            sample_count += 1
            print(f"  Generated sample {sample_count}/{n_samples}")

    if len(all_frames) >= 4:
        from PIL import Image
        imgs = [Image.open(f) for f in all_frames[:4]]
        w, h = imgs[0].size
        grid = Image.new("RGB", (w * 2, h * 2))
        for idx, img in enumerate(imgs):
            grid.paste(img, ((idx % 2) * w, (idx // 2) * h))
        grid.save(out_dir / "summary_grid.png")

    try:
        import subprocess
        frame_pattern = str(frames_dir / "range_%03d.png")
        video_path = str(out_dir / "demo_comparison.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", "2",
                "-i", frame_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                video_path,
            ],
            capture_output=True, timeout=60,
        )
        print(f"  Video saved: {video_path}")
    except Exception as e:
        print(f"  Could not create video: {e}")

    print(f"\nDemo output saved to {out_dir}")


@torch.no_grad()
def run_multistep_demo(
    model,
    seq_loader,
    projector: RangeImageProjector,
    output_dir: str,
    n_samples: int = 4,
    rollout_steps: int = 6,
    num_inference_steps: int = 20,
):
    """Generate multi-step rollout visualisations (t+1 through t+K)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    out_dir = Path(output_dir)
    ms_dir = out_dir / "multistep"
    ms_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0

    for batch in seq_loader:
        if sample_count >= n_samples:
            break

        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        raw_model = model.module if hasattr(model, "module") else model
        B = batch["frames_pc"].shape[0]
        K = min(rollout_steps, batch["rel_poses"].shape[1])

        rel_poses = [batch["rel_poses"][:, k] for k in range(K)]
        actions = [batch["actions"][:, k] for k in range(K)]

        rollout_results = raw_model.forward_rollout_range_space(
            current_pc=batch["frames_pc"][:, 0],
            current_mask=batch["frames_mask"][:, 0],
            rel_poses=rel_poses,
            actions=actions,
            num_steps=num_inference_steps,
            use_midpoint=True,
        )

        for i in range(min(B, n_samples - sample_count)):
            fig, axes = plt.subplots(K, 2, figsize=(20, 2.5 * K), dpi=100)
            if K == 1:
                axes = axes[None, :]
            kwargs = dict(cmap="turbo", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")

            for k in range(K):
                gt_ri, gt_valid = projector.project(
                    batch["frames_pc"][:, k + 1], batch["frames_mask"][:, k + 1]
                )
                pred_ri = rollout_results[k]["pred_range"]

                axes[k, 0].imshow(gt_ri[i, 0].cpu().numpy(), **kwargs)
                axes[k, 0].set_title(f"GT t+{k+1}", fontsize=10)
                axes[k, 0].axis("off")

                axes[k, 1].imshow(pred_ri[i, 0].cpu().numpy(), **kwargs)
                axes[k, 1].set_title(f"Pred t+{k+1}", fontsize=10)
                axes[k, 1].axis("off")

            fig.suptitle(f"Multi-Step Rollout (Sample {sample_count})", fontsize=13)
            plt.tight_layout()
            fig.savefig(ms_dir / f"rollout_range_{sample_count:03d}.png", bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, K, figsize=(5 * K, 10), dpi=100)
            range_m = 60.0

            for k in range(K):
                gt_ri, gt_valid = projector.project(
                    batch["frames_pc"][:, k + 1], batch["frames_mask"][:, k + 1]
                )
                gt_pts, gt_mask = projector.unproject(gt_ri, gt_valid)
                pred_pts = rollout_results[k]["pred_points"]
                pred_mask = rollout_results[k]["pred_mask"]

                for row, (pts, pts_mask, label) in enumerate(
                    [(gt_pts, gt_mask, "GT"), (pred_pts, pred_mask, "Pred")]
                ):
                    ax = axes[row, k] if K > 1 else axes[row]
                    p = pts[i][pts_mask[i]].cpu().numpy()
                    valid = np.abs(p).max(axis=1) < range_m
                    p = p[valid]
                    if p.shape[0] == 0:
                        ax.set_xlim(-range_m, range_m)
                        ax.set_ylim(-range_m, range_m)
                        ax.set_aspect("equal")
                        ax.set_title(f"{label} t+{k+1}", fontsize=10)
                        ax.grid(True, alpha=0.15)
                        continue
                    colors = np.clip((p[:, 2] + 3) / 8, 0, 1)
                    ax.scatter(p[:, 0], p[:, 1], c=colors, cmap="viridis", s=0.1, alpha=0.7)
                    ax.set_xlim(-range_m, range_m)
                    ax.set_ylim(-range_m, range_m)
                    ax.set_aspect("equal")
                    ax.set_title(f"{label} t+{k+1}", fontsize=10)
                    ax.grid(True, alpha=0.15)

            fig.suptitle(f"BEV Multi-Step Rollout (Sample {sample_count})", fontsize=13)
            plt.tight_layout()
            fig.savefig(ms_dir / f"rollout_bev_{sample_count:03d}.png", bbox_inches="tight")
            plt.close(fig)

            sample_count += 1
            print(f"  [multi-step] Generated sample {sample_count}/{n_samples}")

    print(f"Multi-step demo saved to {ms_dir}")


def main():
    parser = argparse.ArgumentParser(description="Demo visualisation for range-image world model")
    parser.add_argument("checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--output", type=str, default="./demo_output")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=6, help="Number of multi-step rollout steps")
    parser.add_argument("--single-step-only", action="store_true", help="Skip multi-step demo")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])

    model = build_world_model(cfg).cuda()
    model.load_state_dict(ckpt["model_state_dict"])

    if "ema_shadow" in ckpt:
        for name, p in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                p.data.copy_(ckpt["ema_shadow"][name])

    projector = build_projector_from_cfg(cfg)

    from nuscenes.nuscenes import NuScenes
    from src.data.temporal_dataset import build_datasets

    nusc = NuScenes(version=cfg.data.version, dataroot=cfg.data.data_root, verbose=True)

    if args.single_step_only:
        _, val_ds = build_datasets(nusc, max_points=cfg.data.max_points, seed=cfg.training.seed)
        val_seq_ds = None
    else:
        result = build_datasets(
            nusc, max_points=cfg.data.max_points, seed=cfg.training.seed,
            sequence_length=args.rollout_steps,
        )
        _, val_ds, _, val_seq_ds = result
    del nusc

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True,
    )

    print("Generating single-step visualisations...")
    run_demo(model, val_loader, projector, args.output, args.n_samples, args.num_steps)

    if val_seq_ds is not None:
        print(f"\nGenerating multi-step rollout visualisations (t+1 to t+{args.rollout_steps})...")
        seq_loader = torch.utils.data.DataLoader(
            val_seq_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True,
        )
        run_multistep_demo(
            model, seq_loader, projector, args.output,
            n_samples=min(args.n_samples, 4),
            rollout_steps=args.rollout_steps,
            num_inference_steps=args.num_steps,
        )


if __name__ == "__main__":
    main()
