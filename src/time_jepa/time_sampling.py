import math

import numpy as np


def build_sample_times(start_time, frames_per_clip, fps):
    if fps is None or fps <= 0:
        raise ValueError(f"fps must be positive for timestamp sampling, got {fps}")
    return float(start_time) + np.arange(int(frames_per_clip), dtype=np.float64) / float(fps)


def clip_span_seconds(frames_per_clip, fps):
    if frames_per_clip <= 0:
        raise ValueError(f"frames_per_clip must be positive, got {frames_per_clip}")
    if fps is None or fps <= 0:
        raise ValueError(f"fps must be positive for timestamp sampling, got {fps}")
    return float(frames_per_clip - 1) / float(fps)


def max_start_time(duration_seconds, frames_per_clip, fps):
    return max(0.0, float(duration_seconds) - clip_span_seconds(frames_per_clip, fps))


def times_to_frame_indices(sample_times, video_fps, num_frames):
    if num_frames <= 0:
        raise ValueError("Cannot sample frames from an empty video")
    if video_fps is None or video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")
    indices = np.rint(np.asarray(sample_times, dtype=np.float64) * float(video_fps))
    return np.clip(indices, 0, int(num_frames) - 1).astype(np.int64)


def nearest_time_indices(query_times, source_times, max_delta=None):
    query_times = np.asarray(query_times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    if source_times.size == 0:
        raise ValueError("Cannot index into an empty time series")

    right = np.searchsorted(source_times, query_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.clip(right - 1, 0, source_times.size - 1)
    choose_left = np.abs(source_times[left] - query_times) <= np.abs(source_times[right] - query_times)
    indices = np.where(choose_left, left, right).astype(np.int64)
    deltas = np.abs(source_times[indices] - query_times)
    if max_delta is None:
        valid = np.ones_like(deltas, dtype=np.float32)
    else:
        valid = (deltas <= float(max_delta)).astype(np.float32)
    return indices, valid


def infer_video_fps(vr, fallback=None):
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 0.0
    if not math.isfinite(fps) or fps <= 0:
        fps = fallback
    if fps is None or fps <= 0:
        raise ValueError("Could not infer a positive video fps")
    return float(fps)
