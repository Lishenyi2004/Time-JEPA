"""Pose-only articulated-arm geometry used by the GT gate and evaluator.

This module intentionally reimplements only the small position-smoothing path
needed for the cosine-law audit.  It never decodes video and does not read any
model prediction while choosing the formula.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.time_jepa.time_sampling import nearest_time_indices


SIDES = ("left", "right")


def _point(joints: dict, name: str, min_views: int) -> np.ndarray | None:
    value = joints.get(name)
    if not isinstance(value, dict):
        return None
    if float(value.get("num_views_for_3d", 0)) < float(min_views):
        return None
    point = np.asarray(
        [value.get("x"), value.get("y"), value.get("z")], dtype=np.float64
    )
    return point if np.isfinite(point).all() else None


def _smooth_positions(
    values: np.ndarray, valid: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    if window is None or int(window) <= 1:
        return values.copy(), valid.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    output = np.full_like(values, np.nan)
    output_valid = np.zeros_like(valid)
    for index in range(values.shape[0]):
        lo = max(0, index - radius)
        hi = min(values.shape[0], index + radius + 1)
        mask = valid[lo:hi]
        if mask.sum() < max(2, (hi - lo + 1) // 2):
            continue
        output[index] = np.median(values[lo:hi][mask], axis=0)
        output_valid[index] = True
    return output, output_valid


def cosine_law_distance(
    theta: np.ndarray,
    proximal_length: np.ndarray | float,
    distal_length: np.ndarray | float,
    clip_angle: bool = True,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    if clip_angle:
        theta = np.clip(theta, 0.0, math.pi)
    proximal = np.asarray(proximal_length, dtype=np.float64)
    distal = np.asarray(distal_length, dtype=np.float64)
    squared = proximal**2 + distal**2 - 2.0 * proximal * distal * np.cos(theta)
    return np.sqrt(np.maximum(squared, 0.0))


@lru_cache(maxsize=16)
def load_take_arm_geometry(
    pose_json_path: str,
    pose_fps: float = 30.0,
    min_views: int = 2,
    smooth_window: int = 5,
) -> dict:
    """Load smoothed arms and freeze one pair of median lengths per side."""

    pose_path = Path(pose_json_path).expanduser().resolve()
    with pose_path.open() as handle:
        payload = json.load(handle)
    frame_ids = np.asarray(sorted(int(key) for key in payload), dtype=np.int64)
    times = frame_ids.astype(np.float64) / float(pose_fps)
    names = [
        f"{side}-{joint}"
        for side in SIDES
        for joint in ("shoulder", "elbow", "wrist")
    ]
    positions = {
        name: np.full((len(frame_ids), 3), np.nan, dtype=np.float64) for name in names
    }
    position_valid = {name: np.zeros(len(frame_ids), dtype=bool) for name in names}
    for row_index, frame_id in enumerate(frame_ids.tolist()):
        persons = payload.get(str(frame_id), [])
        if not persons:
            continue
        joints = persons[0].get("annotation3D", {})
        for name in names:
            point = _point(joints, name, min_views=min_views)
            if point is not None:
                positions[name][row_index] = point
                position_valid[name][row_index] = True
    del payload
    for name in names:
        positions[name], position_valid[name] = _smooth_positions(
            positions[name], position_valid[name], smooth_window
        )

    arms = {}
    for side in SIDES:
        shoulder = positions[f"{side}-shoulder"]
        elbow = positions[f"{side}-elbow"]
        wrist = positions[f"{side}-wrist"]
        valid = (
            position_valid[f"{side}-shoulder"]
            & position_valid[f"{side}-elbow"]
            & position_valid[f"{side}-wrist"]
        )
        proximal_vector = shoulder - elbow
        distal_vector = wrist - elbow
        proximal_length = np.linalg.norm(proximal_vector, axis=1)
        distal_length = np.linalg.norm(distal_vector, axis=1)
        denominator = proximal_length * distal_length
        valid &= np.isfinite(denominator) & (denominator > 1e-12)
        theta = np.full(len(frame_ids), np.nan, dtype=np.float64)
        distance = np.full(len(frame_ids), np.nan, dtype=np.float64)
        cosine = np.sum(proximal_vector[valid] * distal_vector[valid], axis=1)
        cosine /= denominator[valid]
        theta[valid] = np.arccos(np.clip(cosine, -1.0, 1.0))
        distance[valid] = np.linalg.norm(wrist[valid] - shoulder[valid], axis=1)
        if not np.any(valid):
            median_proximal = median_distal = np.nan
        else:
            median_proximal = float(np.median(proximal_length[valid]))
            median_distal = float(np.median(distal_length[valid]))
        arms[side] = {
            "theta": theta,
            "distance": distance,
            "proximal_length": proximal_length,
            "distal_length": distal_length,
            "valid": valid,
            "median_proximal_length_m": median_proximal,
            "median_distal_length_m": median_distal,
        }
    return {
        "pose_path": str(pose_path),
        "pose_fps": float(pose_fps),
        "min_views": int(min_views),
        "smooth_window": int(smooth_window),
        "frame_ids": frame_ids,
        "times": times,
        "arms": arms,
    }


def _pool_frames(
    values: np.ndarray, valid: np.ndarray, tubelet_size: int
) -> tuple[np.ndarray, np.ndarray]:
    usable = (len(values) // int(tubelet_size)) * int(tubelet_size)
    values = values[:usable].reshape(-1, int(tubelet_size))
    valid = valid[:usable].reshape(-1, int(tubelet_size)).astype(np.float64)
    counts = valid.sum(axis=1)
    pooled = np.nansum(np.nan_to_num(values) * valid, axis=1) / np.maximum(counts, 1.0)
    return pooled, counts > 0


def pool_window_arm_targets(
    geometry: dict,
    sample_times: np.ndarray,
    tubelet_size: int = 2,
    max_time_delta: float = 0.05,
) -> dict:
    """Pool theta/distance onto the exact 5-fps, 32-token probe grid."""

    indices, time_valid = nearest_time_indices(
        sample_times, geometry["times"], max_delta=max_time_delta
    )
    output = {}
    for side in SIDES:
        source = geometry["arms"][side]
        sampled_valid = source["valid"][indices] & time_valid.astype(bool)
        theta, theta_valid = _pool_frames(
            source["theta"][indices], sampled_valid, tubelet_size
        )
        distance, distance_valid = _pool_frames(
            source["distance"][indices], sampled_valid, tubelet_size
        )
        output[side] = {
            "theta": theta,
            "distance": distance,
            "valid": theta_valid & distance_valid,
            "median_proximal_length_m": source["median_proximal_length_m"],
            "median_distal_length_m": source["median_distal_length_m"],
        }
    return output


def morphology_for_pose(
    pose_json_path: str,
    pose_fps: float = 30.0,
    min_views: int = 2,
    smooth_window: int = 5,
) -> dict[str, dict[str, float]]:
    geometry = load_take_arm_geometry(
        str(Path(pose_json_path).expanduser().resolve()),
        float(pose_fps),
        int(min_views),
        int(smooth_window),
    )
    return {
        side: {
            "proximal_length_m": float(
                geometry["arms"][side]["median_proximal_length_m"]
            ),
            "distal_length_m": float(
                geometry["arms"][side]["median_distal_length_m"]
            ),
        }
        for side in SIDES
    }

