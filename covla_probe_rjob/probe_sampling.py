"""Frame index sampling, video↔state alignment, and tubelet target pooling."""

import math

import numpy as np
import torch


def normalize_future_horizons_seconds(
    future_horizons_seconds=None,
    future_horizon_seconds=1.0,
):
    """Return a strictly increasing tuple of positive forecast horizons.

    ``future_horizon_seconds`` is retained as the single-endpoint compatibility
    path.  New multi-horizon configs should set ``future_horizons_seconds`` to a
    comma-separated string or a YAML/Python list.
    """
    values = future_horizons_seconds
    if values is None or (isinstance(values, str) and not values.strip()):
        values = [future_horizon_seconds]
    elif isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    elif not isinstance(values, (list, tuple, np.ndarray)):
        values = [values]

    horizons = tuple(float(value) for value in values)
    if not horizons:
        raise ValueError("At least one future horizon is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in horizons):
        raise ValueError(
            "Future horizons must be finite and strictly positive, got "
            f"{horizons}"
        )
    if any(right <= left for left, right in zip(horizons, horizons[1:])):
        raise ValueError(
            "Future horizons must be unique and strictly increasing, got "
            f"{horizons}"
        )
    return horizons


def map_video_frame_to_state_index(video_frame_idx, num_video_frames, num_state_frames):
    if num_video_frames <= 1 or num_state_frames <= 1:
        return 0
    ratio = float(video_frame_idx) / float(num_video_frames - 1)
    return int(np.clip(round(ratio * (num_state_frames - 1)), 0, num_state_frames - 1))


def map_video_frames_to_state_indices(video_frame_indices, frame_ids, num_video_frames):
    video_frame_indices = np.asarray(video_frame_indices, dtype=np.int64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if len(frame_ids) == 0:
        raise RuntimeError("Cannot map video frames to an empty state series")

    if len(frame_ids) == num_video_frames and np.array_equal(frame_ids, np.arange(num_video_frames)):
        return np.clip(video_frame_indices, 0, len(frame_ids) - 1)

    if frame_ids[0] <= video_frame_indices.min() and frame_ids[-1] >= video_frame_indices.max():
        right = np.searchsorted(frame_ids, video_frame_indices, side="left")
        right = np.clip(right, 0, len(frame_ids) - 1)
        left = np.clip(right - 1, 0, len(frame_ids) - 1)
        choose_left = np.abs(frame_ids[left] - video_frame_indices) <= np.abs(frame_ids[right] - video_frame_indices)
        return np.where(choose_left, left, right).astype(np.int64)

    return np.array(
        [
            map_video_frame_to_state_index(idx, num_video_frames, len(frame_ids))
            for idx in video_frame_indices
        ],
        dtype=np.int64,
    )


def compute_effective_frame_stride(num_video_frames, num_frames, frame_span=None, frame_stride=0):
    if num_frames <= 1:
        return 1

    if frame_stride is not None and frame_stride > 0:
        stride = int(frame_stride)
    else:
        span = frame_span if frame_span is not None else num_video_frames
        stride = int(round(float(max(1, span - 1)) / float(num_frames - 1)))
        stride = max(1, stride)

    max_uniform_stride = max(1, (num_video_frames - 1) // (num_frames - 1))
    return min(stride, max_uniform_stride)


def sample_frame_indices(
    num_video_frames,
    num_frames,
    frame_span=None,
    frame_stride=0,
    random_clip=False,
    rng=None,
    reserve_after=0,
):
    if num_frames <= 1:
        return np.array([0], dtype=np.int64)

    reserve_after = max(0, int(reserve_after))
    usable_num_frames = int(num_video_frames) - reserve_after
    if usable_num_frames <= 0:
        raise ValueError(
            f"Cannot reserve {reserve_after} future frames from a {num_video_frames}-frame video"
        )
    stride = compute_effective_frame_stride(usable_num_frames, num_frames, frame_span, frame_stride)
    clip_len = (num_frames - 1) * stride + 1
    if clip_len > usable_num_frames:
        raise ValueError(
            "Video is too short for the requested context plus future reservation: "
            f"video_frames={num_video_frames} context_frames={num_frames} "
            f"context_stride={stride} reserve_after={reserve_after}"
        )
    max_start = usable_num_frames - clip_len
    if random_clip:
        start = rng.randint(0, max_start) if rng is not None else np.random.randint(0, max_start + 1)
    else:
        start = max_start // 2
    return (start + np.arange(num_frames, dtype=np.int64) * stride).astype(np.int64)


def future_row_is_eligible(row, num_frames, fps, horizon_seconds, slice_stride_seconds=None):
    """Return whether a timestamp-sampled row has a label at t_last + horizon.

    AgiBot and EgoExo rows point at full episode/take assets while their split keys
    identify fixed windows.  Filtering only removes terminal windows whose future
    query would exceed the underlying asset; it does not alter split membership.
    """
    horizon_seconds = float(horizon_seconds)
    if horizon_seconds <= 0:
        return True
    duration = row.get("duration")
    if duration is None:
        return True
    if row.get("window_idx") is not None:
        stride_seconds = (
            row.get("slice_stride_seconds")
            if row.get("slice_stride_seconds") is not None
            else slice_stride_seconds
        )
        if stride_seconds is None:
            stride_seconds = float(num_frames) / float(fps)
        start_time = float(row["window_idx"]) * float(stride_seconds)
    else:
        start_time = 0.0
    context_end = start_time + (int(num_frames) - 1) / float(fps)
    return context_end + horizon_seconds <= float(duration) + 1e-6


def build_tubelet_targets(values, valid, state_indices, tubelet_size):
    per_frame_target = torch.from_numpy(values[state_indices])
    per_frame_valid = torch.from_numpy(valid[state_indices])
    target = per_frame_target.reshape(-1, tubelet_size, per_frame_target.size(-1))
    valid_mask = per_frame_valid.reshape(-1, tubelet_size, per_frame_valid.size(-1))
    valid_sum = valid_mask.sum(dim=1).clamp_min(1.0)
    target = (target * valid_mask).sum(dim=1) / valid_sum
    valid_mask = (valid_mask.sum(dim=1) > 0).float()
    return target, valid_mask
