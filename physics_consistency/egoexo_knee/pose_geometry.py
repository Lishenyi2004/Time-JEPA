"""Smoothed hip-knee-ankle geometry for the EgoExo knee GT gate."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from physics_consistency.egoexo.pose_geometry import (
    SIDES,
    _point,
    _pool_frames,
    _smooth_positions,
    cosine_law_distance,
)
from src.time_jepa.time_sampling import nearest_time_indices


@lru_cache(maxsize=16)
def load_take_leg_geometry(
    pose_json_path: str,
    pose_fps: float = 30.0,
    min_views: int = 2,
    smooth_window: int = 5,
) -> dict:
    """Load both legs and freeze median thigh/shin lengths per take."""

    pose_path = Path(pose_json_path).expanduser().resolve()
    with pose_path.open() as handle:
        payload = json.load(handle)
    frame_ids = np.asarray(sorted(int(key) for key in payload), dtype=np.int64)
    times = frame_ids.astype(np.float64) / float(pose_fps)
    names = [
        f"{side}-{joint}"
        for side in SIDES
        for joint in ("hip", "knee", "ankle")
    ]
    positions = {
        name: np.full((len(frame_ids), 3), np.nan, dtype=np.float64)
        for name in names
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

    legs = {}
    for side in SIDES:
        hip = positions[f"{side}-hip"]
        knee = positions[f"{side}-knee"]
        ankle = positions[f"{side}-ankle"]
        valid = (
            position_valid[f"{side}-hip"]
            & position_valid[f"{side}-knee"]
            & position_valid[f"{side}-ankle"]
        )
        thigh_vector = hip - knee
        shin_vector = ankle - knee
        thigh_length = np.linalg.norm(thigh_vector, axis=1)
        shin_length = np.linalg.norm(shin_vector, axis=1)
        denominator = thigh_length * shin_length
        valid &= np.isfinite(denominator) & (denominator > 1e-12)
        theta = np.full(len(frame_ids), np.nan, dtype=np.float64)
        distance = np.full(len(frame_ids), np.nan, dtype=np.float64)
        cosine = np.sum(thigh_vector[valid] * shin_vector[valid], axis=1)
        cosine /= denominator[valid]
        theta[valid] = np.arccos(np.clip(cosine, -1.0, 1.0))
        distance[valid] = np.linalg.norm(ankle[valid] - hip[valid], axis=1)
        if np.any(valid):
            median_thigh = float(np.median(thigh_length[valid]))
            median_shin = float(np.median(shin_length[valid]))
        else:
            median_thigh = median_shin = np.nan
        legs[side] = {
            "theta": theta,
            "distance": distance,
            "proximal_length": thigh_length,
            "distal_length": shin_length,
            "valid": valid,
            "median_proximal_length_m": median_thigh,
            "median_distal_length_m": median_shin,
        }
    return {
        "pose_path": str(pose_path),
        "pose_fps": float(pose_fps),
        "min_views": int(min_views),
        "smooth_window": int(smooth_window),
        "frame_ids": frame_ids,
        "times": times,
        "legs": legs,
    }


def pool_window_leg_targets(
    geometry: dict,
    sample_times: np.ndarray,
    tubelet_size: int = 2,
    max_time_delta: float = 0.05,
) -> dict:
    """Pool knee angle and hip-ankle distance onto the probe tubelet grid."""

    indices, time_valid = nearest_time_indices(
        sample_times, geometry["times"], max_delta=max_time_delta
    )
    output = {}
    for side in SIDES:
        source = geometry["legs"][side]
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
    geometry = load_take_leg_geometry(
        str(Path(pose_json_path).expanduser().resolve()),
        float(pose_fps),
        int(min_views),
        int(smooth_window),
    )
    return {
        side: {
            "proximal_length_m": float(
                geometry["legs"][side]["median_proximal_length_m"]
            ),
            "distal_length_m": float(
                geometry["legs"][side]["median_distal_length_m"]
            ),
        }
        for side in SIDES
    }

