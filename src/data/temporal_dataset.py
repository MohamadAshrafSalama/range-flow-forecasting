"""
Temporal pair dataset for range-image scene forecasting.

Loads consecutive LiDAR frame pairs from nuScenes, returning raw 3D point
clouds in ego frame (metric coordinates, NOT normalised). The model handles
range-image projection and encoding internally.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix


class TemporalPairDataset(Dataset):
    """
    Each item is a dict:
        current_pc      (max_points, 3)   float32   current frame in ego_t  [metres]
        current_mask    (max_points,)      bool      valid-point mask
        target_pc       (max_points, 3)   float32   next frame in ego_{t+1}  [metres]
        target_mask     (max_points,)      bool
        relative_pose   (4, 4)            float32   ego_t -> ego_{t+1}  [metres]
        action_features (5,)              float32   [v_fwd, yaw_rate, dx, dy, dtheta]
        dt              scalar            float32   time gap (seconds)

    Coordinates are in raw metres. The model's RangeImageProjector handles
    depth encoding.
    """

    def __init__(
        self,
        nusc: NuScenes,
        sample_tokens: list,
        max_points: int = 35000,
        prediction_horizon: int = 1,
    ):
        super().__init__()
        self.max_points = max_points
        self.prediction_horizon = prediction_horizon
        self.dataroot = nusc.dataroot

        # ---- build valid temporal pairs ----
        self.pairs = []
        for token in sample_tokens:
            cur = nusc.get("sample", token)
            nxt = cur
            for _ in range(prediction_horizon):
                if nxt["next"] == "":
                    nxt = None
                    break
                nxt = nusc.get("sample", nxt["next"])
            if nxt is not None:
                self.pairs.append((token, nxt["token"]))

        print(
            f"[TemporalPairDataset] {len(self.pairs)} valid pairs "
            f"(horizon={prediction_horizon}) from {len(sample_tokens)} samples"
        )

        # ---- pre-extract per-sample metadata ----
        needed_tokens = set()
        for a, b in self.pairs:
            needed_tokens.add(a)
            needed_tokens.add(b)

        cs_cache: dict = {}
        self._sample_meta: dict = {}

        for tok in needed_tokens:
            sample = nusc.get("sample", tok)
            sd_token = sample["data"]["LIDAR_TOP"]
            sd = nusc.get("sample_data", sd_token)

            cs_token = sd["calibrated_sensor_token"]
            if cs_token not in cs_cache:
                cs = nusc.get("calibrated_sensor", cs_token)
                cs_cache[cs_token] = transform_matrix(
                    np.array(cs["translation"]),
                    Quaternion(cs["rotation"]),
                )
            T_cs = cs_cache[cs_token]

            pose_rec = nusc.get("ego_pose", sd["ego_pose_token"])
            T_ego = transform_matrix(
                np.array(pose_rec["translation"]),
                Quaternion(pose_rec["rotation"]),
            )

            self._sample_meta[tok] = {
                "path": os.path.join(self.dataroot, sd["filename"]),
                "T_cs": T_cs.astype(np.float64),
                "T_ego": T_ego.astype(np.float64),
                "timestamp": sd["timestamp"],
            }

    def _load_lidar_ego(self, sample_token: str):
        """Load LIDAR_TOP, transform to ego frame. Returns (N, 3)."""
        meta = self._sample_meta[sample_token]
        pts = np.fromfile(meta["path"], dtype=np.float32).reshape(-1, 5)[:, :3]
        T_cs = meta["T_cs"]
        pts_ego = (T_cs[:3, :3] @ pts.T + T_cs[:3, 3:]).T
        return pts_ego, meta["T_ego"], meta["timestamp"]

    def _pad_pc(self, pc: np.ndarray):
        """Pad / subsample to max_points."""
        N = pc.shape[0]
        mask = np.ones(self.max_points, dtype=bool)
        if N >= self.max_points:
            idx = np.random.choice(N, self.max_points, replace=False)
            pc = pc[idx]
        else:
            pad = np.zeros((self.max_points - N, 3), dtype=pc.dtype)
            pc = np.vstack([pc, pad])
            mask[N:] = False
        return pc, mask

    @staticmethod
    def _action_from_pose(T_rel: np.ndarray, dt: float) -> np.ndarray:
        dx = T_rel[0, 3]
        dy = T_rel[1, 3]
        dtheta = np.arctan2(T_rel[1, 0], T_rel[0, 0])
        dist = np.sqrt(dx**2 + dy**2)
        dt_safe = max(dt, 1e-3)
        v_forward = dist / dt_safe
        yaw_rate = dtheta / dt_safe
        return np.array([v_forward, yaw_rate, dx, dy, dtheta], dtype=np.float32)

    def __getitem__(self, idx):
        cur_token, nxt_token = self.pairs[idx]

        cur_pts, T_cur, ts_cur = self._load_lidar_ego(cur_token)
        nxt_pts, T_nxt, ts_nxt = self._load_lidar_ego(nxt_token)

        T_rel = np.linalg.inv(T_nxt) @ T_cur

        dt = abs(ts_nxt - ts_cur) / 1e6

        action = self._action_from_pose(T_rel, dt)

        cur_pts, cur_mask = self._pad_pc(cur_pts)
        nxt_pts, nxt_mask = self._pad_pc(nxt_pts)

        return {
            "current_pc": torch.tensor(cur_pts, dtype=torch.float32),
            "current_mask": torch.tensor(cur_mask, dtype=torch.bool),
            "target_pc": torch.tensor(nxt_pts, dtype=torch.float32),
            "target_mask": torch.tensor(nxt_mask, dtype=torch.bool),
            "relative_pose": torch.tensor(T_rel, dtype=torch.float32),
            "action_features": torch.tensor(action, dtype=torch.float32),
            "dt": torch.tensor(dt, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.pairs)


# =====================================================================
#  Multi-step Sequence Dataset (for scheduled sampling training)
# =====================================================================

class TemporalSequenceDataset(Dataset):
    """
    Returns sequences of (K+1) consecutive frames for multi-step training.

    Each item is a dict:
        frames_pc       list of (K+1) x (max_points, 3)  float32
        frames_mask     list of (K+1) x (max_points,)    bool
        rel_poses       list of K x (4, 4)               float32
        actions         list of K x (5,)                  float32
        dts             list of K scalars                 float32
    """

    def __init__(
        self,
        nusc: NuScenes,
        sample_tokens: list,
        max_points: int = 35000,
        sequence_length: int = 6,   # K = number of future steps (returns K+1 frames)
        history_frames: int = 0,    # B: H past frames before current (for history conditioning)
    ):
        super().__init__()
        self.max_points = max_points
        self.sequence_length = sequence_length
        self.history_frames = max(0, history_frames)
        self.dataroot = nusc.dataroot

        self.sequences = []  # list of (H+K+1)-tuples when history>0, else (K+1)-tuples
        token_set = set(sample_tokens)
        for token in sample_tokens:
            seq = [token]
            cur = nusc.get("sample", token)
            valid = True

            for _ in range(history_frames):
                if not cur.get("prev", ""):
                    valid = False
                    break
                prev_s = nusc.get("sample", cur["prev"])
                if prev_s["token"] not in token_set:
                    valid = False
                    break
                seq.insert(0, prev_s["token"])
                cur = prev_s

            cur = nusc.get("sample", token)
            for _ in range(sequence_length):
                if cur["next"] == "":
                    valid = False
                    break
                nxt = nusc.get("sample", cur["next"])
                if nxt["token"] not in token_set:
                    valid = False
                    break
                seq.append(nxt["token"])
                cur = nxt

            expected_len = sequence_length + 1 + history_frames
            if valid and len(seq) == expected_len:
                self.sequences.append(tuple(seq))

        print(
            f"[TemporalSequenceDataset] {len(self.sequences)} valid sequences "
            f"(K={sequence_length}) from {len(sample_tokens)} samples"
        )

        needed_tokens = set()
        for seq in self.sequences:
            needed_tokens.update(seq)

        cs_cache: dict = {}
        self._sample_meta: dict = {}

        for tok in needed_tokens:
            sample = nusc.get("sample", tok)
            sd_token = sample["data"]["LIDAR_TOP"]
            sd = nusc.get("sample_data", sd_token)

            cs_token = sd["calibrated_sensor_token"]
            if cs_token not in cs_cache:
                cs = nusc.get("calibrated_sensor", cs_token)
                cs_cache[cs_token] = transform_matrix(
                    np.array(cs["translation"]),
                    Quaternion(cs["rotation"]),
                )
            T_cs = cs_cache[cs_token]

            pose_rec = nusc.get("ego_pose", sd["ego_pose_token"])
            T_ego = transform_matrix(
                np.array(pose_rec["translation"]),
                Quaternion(pose_rec["rotation"]),
            )

            self._sample_meta[tok] = {
                "path": os.path.join(self.dataroot, sd["filename"]),
                "T_cs": T_cs.astype(np.float64),
                "T_ego": T_ego.astype(np.float64),
                "timestamp": sd["timestamp"],
            }

    def _load_lidar_ego(self, sample_token: str):
        meta = self._sample_meta[sample_token]
        pts = np.fromfile(meta["path"], dtype=np.float32).reshape(-1, 5)[:, :3]
        T_cs = meta["T_cs"]
        pts_ego = (T_cs[:3, :3] @ pts.T + T_cs[:3, 3:]).T
        return pts_ego, meta["T_ego"], meta["timestamp"]

    def _pad_pc(self, pc: np.ndarray):
        N = pc.shape[0]
        mask = np.ones(self.max_points, dtype=bool)
        if N >= self.max_points:
            idx = np.random.choice(N, self.max_points, replace=False)
            pc = pc[idx]
        else:
            pad = np.zeros((self.max_points - N, 3), dtype=pc.dtype)
            pc = np.vstack([pc, pad])
            mask[N:] = False
        return pc, mask

    @staticmethod
    def _action_from_pose(T_rel: np.ndarray, dt: float) -> np.ndarray:
        dx = T_rel[0, 3]
        dy = T_rel[1, 3]
        dtheta = np.arctan2(T_rel[1, 0], T_rel[0, 0])
        dist = np.sqrt(dx**2 + dy**2)
        dt_safe = max(dt, 1e-3)
        v_forward = dist / dt_safe
        yaw_rate = dtheta / dt_safe
        return np.array([v_forward, yaw_rate, dx, dy, dtheta], dtype=np.float32)

    def __getitem__(self, idx):
        seq_tokens = self.sequences[idx]
        K = self.sequence_length
        H = self.history_frames
        total = len(seq_tokens)

        frame_data = []
        for tok in seq_tokens:
            pts, T_ego, ts = self._load_lidar_ego(tok)
            pts, mask = self._pad_pc(pts)
            frame_data.append({
                "pts": pts,
                "mask": mask,
                "T_ego": T_ego,
                "ts": ts,
            })

        frames_pc = []
        frames_mask = []
        rel_poses = []
        actions = []
        dts = []

        start = H
        for i in range(start, total):
            frames_pc.append(torch.tensor(frame_data[i]["pts"], dtype=torch.float32))
            frames_mask.append(torch.tensor(frame_data[i]["mask"], dtype=torch.bool))

        for i in range(start, total - 1):
            T_cur = frame_data[i]["T_ego"]
            T_nxt = frame_data[i + 1]["T_ego"]
            T_rel = np.linalg.inv(T_nxt) @ T_cur
            dt = abs(frame_data[i + 1]["ts"] - frame_data[i]["ts"]) / 1e6
            action = self._action_from_pose(T_rel, dt)
            rel_poses.append(torch.tensor(T_rel, dtype=torch.float32))
            actions.append(torch.tensor(action, dtype=torch.float32))
            dts.append(torch.tensor(dt, dtype=torch.float32))

        out = {
            "frames_pc": torch.stack(frames_pc),
            "frames_mask": torch.stack(frames_mask),
            "rel_poses": torch.stack(rel_poses),
            "actions": torch.stack(actions),
            "dts": torch.stack(dts),
        }
        if H > 0:
            history_pc = torch.stack([
                torch.tensor(frame_data[i]["pts"], dtype=torch.float32)
                for i in range(H)
            ])
            history_mask = torch.stack([
                torch.tensor(frame_data[i]["mask"], dtype=torch.bool)
                for i in range(H)
            ])
            out["history_pc"] = history_pc
            out["history_mask"] = history_mask
        return out

    def __len__(self):
        return len(self.sequences)


def build_datasets(
    nusc: NuScenes,
    max_points: int = 35000,
    train_ratio: float = 0.85,
    prediction_horizon: int = 1,
    seed: int = 42,
    sequence_length: int = 0,
    history_frames: int = 0,
):
    """Split by scene, returns (train_ds, val_ds) or (train_ds, val_ds, train_seq_ds, val_seq_ds)."""
    scenes = list(nusc.scene)
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    n_train = int(len(scenes) * train_ratio)
    train_scenes = set(s["token"] for s in scenes[:n_train])

    train_tokens, val_tokens = [], []
    for s in nusc.sample:
        if s["scene_token"] in train_scenes:
            train_tokens.append(s["token"])
        else:
            val_tokens.append(s["token"])

    print(f"[build_datasets] train scenes={n_train}, val={len(scenes)-n_train}")
    print(f"  train samples={len(train_tokens)}, val samples={len(val_tokens)}")

    train_ds = TemporalPairDataset(
        nusc, train_tokens, max_points, prediction_horizon
    )
    val_ds = TemporalPairDataset(
        nusc, val_tokens, max_points, prediction_horizon
    )

    if sequence_length > 0:
        train_seq_ds = TemporalSequenceDataset(
            nusc, train_tokens, max_points, sequence_length,
            history_frames=history_frames,
        )
        val_seq_ds = TemporalSequenceDataset(
            nusc, val_tokens, max_points, sequence_length,
            history_frames=history_frames,
        )
        return train_ds, val_ds, train_seq_ds, val_seq_ds

    return train_ds, val_ds
