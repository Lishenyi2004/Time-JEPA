"""Dependency-light construction of raw shoulder-to-hand reach rate."""

from __future__ import annotations

import numpy as np

from physics_consistency.agibot_reach_rate.common import ARMS, P_SLICES


def _central_indices(length: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(int(length), dtype=np.int64)
    return np.maximum(index - 1, 0), np.minimum(
        index + 1, int(length) - 1
    )


def derive_reach_rate_series(
    series: dict[str, np.ndarray],
    shoulder_points: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw shoulder-to-hand distance, its derivative, and validity."""

    timestamps = np.asarray(series["timestamps"], dtype=np.float64)
    position = np.asarray(series["position"], dtype=np.float64)
    previous, following = _central_indices(len(timestamps))
    delta_t = timestamps[following] - timestamps[previous]
    reach = np.zeros((len(timestamps), len(ARMS)), dtype=np.float64)
    rate = np.zeros_like(reach)
    valid = np.zeros_like(reach, dtype=np.float32)
    for arm_index, arm in enumerate(ARMS):
        shoulder = np.asarray(shoulder_points[arm], dtype=np.float64)
        if shoulder.shape != (3,) or not np.all(np.isfinite(shoulder)):
            raise ValueError(f"Invalid {arm} shoulder point: {shoulder}")
        arm_position = position[:, P_SLICES[arm]]
        reach[:, arm_index] = np.linalg.norm(
            arm_position - shoulder[None, :], axis=-1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            arm_rate = (
                reach[following, arm_index] - reach[previous, arm_index]
            ) / delta_t
        arm_velocity_valid = np.all(
            np.asarray(series["velocity_valid"][:, P_SLICES[arm]]) > 0,
            axis=-1,
        )
        arm_valid = (
            (np.asarray(series["position_valid"][:, arm_index]) > 0)
            & (np.asarray(series["position_valid"][previous, arm_index]) > 0)
            & (np.asarray(series["position_valid"][following, arm_index]) > 0)
            & arm_velocity_valid
            & np.isfinite(arm_rate)
            & (delta_t > 0.0)
        )
        rate[:, arm_index] = np.where(arm_valid, arm_rate, 0.0)
        valid[:, arm_index] = arm_valid.astype(np.float32)
    return reach.astype(np.float32), rate.astype(np.float32), valid


__all__ = ["derive_reach_rate_series"]
