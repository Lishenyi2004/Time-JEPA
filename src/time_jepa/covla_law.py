"""Longitudinal kinematics on CoVLA's sampled, two-frame tubelets.

The state head predicts normalized targets; this loss operates in m/s, m/s²
and seconds. No trajectory, shot target, or held-out annotation is used.
"""

import hashlib
import json
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from src.time_jepa.physical_targets import (
    map_video_frames_to_state_indices,
    read_payload_field,
)

PROTOCOL = "covla_longitudinal_trapezoid_video_pts_v2"
MAX_DT_SECONDS = 0.6


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def load_law_series(states_path):
    """Keep epoch timestamps in float64 until after differencing (raw: ms)."""
    ids, timestamps, values, valid = [], [], [], []
    with open(states_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            frame, payload = next(iter(json.loads(line).items()))
            ids.append(int(frame))
            timestamp = payload.get("timestamp")
            timestamps.append(
                float(timestamp)
                if isinstance(timestamp, (int, float)) and math.isfinite(timestamp)
                else float("nan")
            )
            row = [read_payload_field(payload, key) for key in ("vEgo", "aEgo")]
            values.append([item[0] for item in row])
            valid.append([item[1] for item in row])
    if not ids:
        raise ValueError(f"Empty state file: {states_path}")
    order = np.argsort(ids)
    ids = np.asarray(ids, dtype=np.int64)[order]
    if np.any(np.diff(ids) <= 0):
        raise ValueError(f"Duplicate state frame IDs: {states_path}")
    timestamps = np.asarray(timestamps, dtype=np.float64)[order]
    finite = timestamps[np.isfinite(timestamps)]
    if not len(finite):
        raise ValueError(f"No finite millisecond timestamps: {states_path}")
    times = (timestamps - finite[0]) / 1000.0
    return ids, times, np.asarray(values, dtype=np.float32)[order], np.asarray(valid, dtype=bool)[order]


def tubelet_law_metadata(frame_indices, num_video_frames, series, tubelet_size=2, video_frame_times=None):
    """Law mask requires BOTH channels on ALL four frames of each pair."""
    ids, times, values, valid = series
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if tubelet_size != 2 or len(frame_indices) % tubelet_size:
        raise ValueError("The frozen law protocol requires two-frame tubelets")
    mapped = map_video_frames_to_state_indices(frame_indices, ids, num_video_frames)
    if video_frame_times is not None:
        # STATE timestamps can stall or jump. Use decoded video PTS at the
        # mapped STATE frame IDs, and reject annotations outside the video.
        video_frame_times = np.asarray(video_frame_times, dtype=np.float64)
        if len(video_frame_times) != num_video_frames:
            raise ValueError("Video PTS must contain one timestamp per frame")
        if np.any(ids[mapped] < 0) or np.any(ids[mapped] >= num_video_frames):
            raise ValueError("Mapped STATE frame ID is outside video PTS")
        frame_times = video_frame_times[ids[mapped]].reshape(-1, tubelet_size)
    else:
        frame_times = times[mapped].reshape(-1, tubelet_size)
    complete = valid[mapped].reshape(-1, tubelet_size, 2).all(axis=(1, 2))
    complete &= np.isfinite(frame_times).all(axis=1)
    complete &= np.diff(frame_times, axis=1).min(axis=1) > 0
    centers = frame_times.mean(axis=1)
    dt = np.diff(centers)
    pair_valid = complete[:-1] & complete[1:]
    pair_valid &= np.isfinite(dt) & (dt > 0) & (dt <= MAX_DT_SECONDS)
    # Reject overlapping/repeated/reversed sampled frames, including padding.
    mapped_ids = ids[mapped].reshape(-1, tubelet_size)
    pair_valid &= mapped_ids[:-1, -1] < mapped_ids[1:, 0]
    target = values[mapped].reshape(-1, tubelet_size, 2).mean(axis=1)
    return {
        "law_dt": torch.from_numpy(np.where(pair_valid, dt, 0).astype(np.float32)),
        "law_valid": torch.from_numpy(pair_valid),
        "law_target": torch.from_numpy(target),
        "law_center_time": torch.from_numpy(centers),
    }


def longitudinal_law_loss(prediction, mean, std, dt, pair_valid, scale, beta=0.5):
    """Return differentiable loss and detached metrics; v/a are columns 0/1."""
    if not math.isfinite(float(scale)) or scale <= 0 or beta <= 0:
        raise ValueError("Law scale and SmoothL1 beta must be positive and finite")
    if prediction.ndim != 3 or prediction.shape[-1] < 2:
        raise ValueError("Expected physical predictions [batch, tubelet, channels>=2]")
    expected_shape = prediction.shape[:1] + (prediction.shape[1] - 1,)
    if dt.shape != expected_shape or pair_valid.shape != expected_shape:
        raise ValueError(f"Law metadata shape mismatch: {dt.shape}, {expected_shape}")
    dt = dt.to(device=prediction.device, dtype=torch.float32)
    pair_valid = pair_valid.to(device=prediction.device, dtype=torch.bool)
    pair_valid = pair_valid & torch.isfinite(dt) & (dt > 0) & (dt <= MAX_DT_SECONDS)
    dt = torch.where(pair_valid, dt, torch.zeros_like(dt))
    physical = prediction.float() * std.to(prediction.device).float() + mean.to(prediction.device).float()
    delta_v = physical[:, 1:, 0] - physical[:, :-1, 0]
    integral_a = 0.5 * (physical[:, 1:, 1] + physical[:, :-1, 1]) * dt
    residual = (delta_v - integral_a)[pair_valid]
    if not torch.isfinite(residual).all():
        raise FloatingPointError("Nonfinite prediction on a valid longitudinal-law pair")
    if residual.numel():
        loss = F.smooth_l1_loss(residual / scale, torch.zeros_like(residual), beta=beta)
        mae = residual.detach().abs().mean()
    else:
        # Empty indexing keeps a zero gradient connection, even if invalid
        # positions contain NaNs; NaN * 0 would not be safe here.
        loss = residual.sum()
        mae = loss.detach()
    return loss, {
        "law_loss": float(loss.detach()),
        "law_residual_mae_mps": float(mae),
        "law_valid_pairs": int(pair_valid.sum()),
        "law_total_pairs": pair_valid.numel(),
    }
