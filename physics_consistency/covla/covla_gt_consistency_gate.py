#!/usr/bin/env python3
"""Ground-truth gate for the standard COVLA weather-OOD protocol.

The gate is intentionally restricted to probe_train and probe_val.  It audits
the video/state/trajectory time grids, selects the COVLA coordinate convention
on probe_train, validates the frozen convention on probe_val, and reports the
oracle floor for trajectory and velocity-acceleration closure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np


ALLOWED_SPLITS = {"probe_train", "probe_val"}
TRAJECTORY_POINT_INDICES = (1, 19, 20, 39, 40, 59)
PRIMARY_HORIZONS = ((1.0, 20), (2.0, 40), (2.95, 59))
VA_HORIZONS = (0.5, 1.0, 2.0)
SCOPE_SPECS = {
    "raw_20fps": None,
    "tubelet_4fps": 4.0,
    "tubelet_10fps": 10.0,
}


def finite_number(value) -> float:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return math.nan


def indexed_number(payload, field, index) -> float:
    value = payload.get(field)
    if not isinstance(value, (list, tuple)) or not (-len(value) <= index < len(value)):
        return math.nan
    return finite_number(value[index])


def trajectory_xy(payload, point_index) -> tuple[float, float]:
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, (list, tuple)) or not (-len(trajectory) <= point_index < len(trajectory)):
        return math.nan, math.nan
    point = trajectory[point_index]
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return math.nan, math.nan
    return finite_number(point[0]), finite_number(point[1])


@dataclass
class RawSequence:
    video_id: str
    split: str
    frame_ids: np.ndarray
    timestamps_ms: np.ndarray
    v_ego: np.ndarray
    a_ego: np.ndarray
    vx_calib: np.ndarray
    vy_calib: np.ndarray
    ax_calib: np.ndarray
    yaw_rate: np.ndarray
    trajectory: dict[int, np.ndarray]
    video_fps: float
    video_frames: int


@dataclass
class KinematicSequence:
    video_id: str
    split: str
    scope: str
    dt: float
    times: np.ndarray
    v_ego: np.ndarray
    a_ego: np.ndarray
    vx_calib: np.ndarray
    vy_calib: np.ndarray
    ax_calib: np.ndarray
    yaw_rate: np.ndarray
    trajectory: dict[int, np.ndarray]


@dataclass(frozen=True)
class Candidate:
    method: str
    x_field: str
    vy_sign: int
    yaw_sign: int

    @property
    def key(self) -> str:
        return f"{self.method}__{self.x_field}__vy{self.vy_sign:+d}__yaw{self.yaw_sign:+d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_ids(split_dir: Path, split_name: str) -> tuple[list[str], Path]:
    if split_name not in ALLOWED_SPLITS:
        raise ValueError(
            f"COVLA GT Gate only permits {sorted(ALLOWED_SPLITS)}, got {split_name!r}; "
            "test/OOD must not influence protocol selection"
        )
    path = split_dir / f"covla_{split_name}.json"
    with path.open() as handle:
        video_ids = json.load(handle)
    if not isinstance(video_ids, list) or not all(isinstance(item, str) for item in video_ids):
        raise RuntimeError(f"Expected a JSON list of video IDs: {path}")
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError(f"Duplicate video IDs in {path}")
    return video_ids, path


def read_metadata(covla_root: Path) -> dict[str, Path]:
    result = {}
    with (covla_root / "metadata.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            video_id = record.get("video_id") or Path(record["file_name"]).stem
            result[video_id] = covla_root / record["file_name"]
    return result


def parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return math.nan
    return float(Fraction(value))


def ffprobe_one(video_id: str, path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {"video_id": video_id, "path": str(path), "error": proc.stderr.strip()}
    try:
        payload = json.loads(proc.stdout)
        stream = payload["streams"][0]
        return {
            "video_id": video_id,
            "path": str(path),
            "fps": parse_rate(stream.get("avg_frame_rate")),
            "frames": int(stream["nb_frames"]),
            "duration_seconds": float(stream["duration"]),
            "error": None,
        }
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"video_id": video_id, "path": str(path), "error": f"parse error: {exc}"}


def audit_videos(video_ids: Iterable[str], metadata: dict[str, Path], workers: int) -> dict[str, dict]:
    missing = [video_id for video_id in video_ids if video_id not in metadata]
    if missing:
        raise RuntimeError(f"{len(missing)} split IDs are absent from metadata.jsonl; first={missing[:5]}")
    result = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(ffprobe_one, video_id, metadata[video_id]): video_id
            for video_id in video_ids
        }
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            result[record["video_id"]] = record
            if done % 250 == 0 or done == len(futures):
                print(f"[video audit] {done}/{len(futures)}", flush=True)
    return result


class StateAudit:
    def __init__(self):
        self.frame_delta = Counter()
        self.timestamp_delta_ms = Counter()
        self.unique_timestamp_delta_ms = Counter()
        self.trajectory_lengths = Counter()
        self.state_counts = Counter()
        self.trajectory_p0_norm = []
        self.trajectory_step_dt_speed = []
        self.trajectory_step_dt_vego = []
        self.timestamp_spans_seconds = []

    def add(self, frame_ids, timestamps, trajectory_lengths, p0_norm, step_distance, vxy, v_ego):
        if len(frame_ids) > 1:
            self.frame_delta.update(int(item) for item in np.diff(frame_ids))
        valid_ts = timestamps[np.isfinite(timestamps)]
        if len(valid_ts) > 1:
            self.timestamp_delta_ms.update(int(round(item)) for item in np.diff(valid_ts))
            unique_ts = np.unique(valid_ts)
            if len(unique_ts) > 1:
                self.unique_timestamp_delta_ms.update(int(round(item)) for item in np.diff(unique_ts))
            self.timestamp_spans_seconds.append(float((valid_ts[-1] - valid_ts[0]) / 1000.0))
        self.trajectory_lengths.update(int(item) for item in trajectory_lengths)
        self.state_counts[len(frame_ids)] += 1
        valid_p0 = p0_norm[np.isfinite(p0_norm)]
        if len(valid_p0):
            self.trajectory_p0_norm.append(valid_p0.astype(np.float32))
        valid_speed = np.isfinite(step_distance) & np.isfinite(vxy) & (vxy > 1.0)
        estimate = step_distance[valid_speed] / vxy[valid_speed]
        estimate = estimate[(estimate > 0.0) & (estimate < 0.25)]
        if len(estimate):
            self.trajectory_step_dt_speed.append(estimate.astype(np.float32))
        valid_vego = np.isfinite(step_distance) & np.isfinite(v_ego) & (np.abs(v_ego) > 1.0)
        estimate = step_distance[valid_vego] / np.abs(v_ego[valid_vego])
        estimate = estimate[(estimate > 0.0) & (estimate < 0.25)]
        if len(estimate):
            self.trajectory_step_dt_vego.append(estimate.astype(np.float32))

    @staticmethod
    def _array_summary(chunks) -> dict:
        if not chunks:
            return {"count": 0}
        values = np.concatenate(chunks).astype(np.float64, copy=False)
        return distribution_summary(values, signed=False)

    def summary(self) -> dict:
        return {
            "state_count_histogram": dict(sorted(self.state_counts.items())),
            "frame_delta_histogram": dict(sorted(self.frame_delta.items())),
            "timestamp_delta_ms_histogram": dict(sorted(self.timestamp_delta_ms.items())),
            "unique_timestamp_delta_ms_histogram": dict(sorted(self.unique_timestamp_delta_ms.items())),
            "trajectory_length_histogram": dict(sorted(self.trajectory_lengths.items())),
            "timestamp_span_seconds": distribution_summary(np.asarray(self.timestamp_spans_seconds), signed=False),
            "trajectory_point0_norm_m": self._array_summary(self.trajectory_p0_norm),
            "trajectory_step_seconds_from_speed_norm": self._array_summary(self.trajectory_step_dt_speed),
            "trajectory_step_seconds_from_vEgo": self._array_summary(self.trajectory_step_dt_vego),
        }


def load_state_sequence(
    covla_root: Path,
    video_id: str,
    split: str,
    video_record: dict,
    audit: StateAudit,
) -> RawSequence:
    frame_ids = []
    timestamps = []
    v_ego = []
    a_ego = []
    vx_calib = []
    vy_calib = []
    ax_calib = []
    yaw_rate = []
    trajectories = {point: [] for point in TRAJECTORY_POINT_INDICES}
    trajectory_lengths = []
    p0_norm = []
    step_distance = []
    path = covla_root / "states" / f"{video_id}.jsonl"
    with path.open() as handle:
        for line in handle:
            obj = json.loads(line)
            frame_id_text, payload = next(iter(obj.items()))
            frame_ids.append(int(frame_id_text))
            timestamps.append(finite_number(payload.get("timestamp")))
            v_ego.append(finite_number(payload.get("vEgo")))
            a_ego.append(finite_number(payload.get("aEgo")))
            vx_calib.append(indexed_number(payload, "velocities_calib", 0))
            vy_calib.append(indexed_number(payload, "velocities_calib", 1))
            ax_calib.append(indexed_number(payload, "accelerations_calib", 0))
            yaw_rate.append(indexed_number(payload, "angular_velocities_calib", 2))
            trajectory = payload.get("trajectory")
            trajectory_lengths.append(len(trajectory) if isinstance(trajectory, (list, tuple)) else 0)
            p0 = np.asarray(trajectory_xy(payload, 0), dtype=np.float64)
            p1 = np.asarray(trajectory_xy(payload, 1), dtype=np.float64)
            p0_norm.append(float(np.linalg.norm(p0)) if np.all(np.isfinite(p0)) else math.nan)
            step_distance.append(float(np.linalg.norm(p1 - p0)) if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)) else math.nan)
            for point in TRAJECTORY_POINT_INDICES:
                trajectories[point].append(trajectory_xy(payload, point))

    order = np.argsort(np.asarray(frame_ids, dtype=np.int64))

    def ordered(values, dtype=np.float32):
        return np.asarray(values, dtype=dtype)[order]

    result = RawSequence(
        video_id=video_id,
        split=split,
        frame_ids=ordered(frame_ids, np.int64),
        timestamps_ms=ordered(timestamps, np.float64),
        v_ego=ordered(v_ego),
        a_ego=ordered(a_ego),
        vx_calib=ordered(vx_calib),
        vy_calib=ordered(vy_calib),
        ax_calib=ordered(ax_calib),
        yaw_rate=ordered(yaw_rate),
        trajectory={point: ordered(values) for point, values in trajectories.items()},
        video_fps=float(video_record.get("fps", math.nan)),
        video_frames=int(video_record.get("frames", 0)),
    )
    vxy = np.hypot(result.vx_calib, result.vy_calib)
    audit.add(
        result.frame_ids,
        result.timestamps_ms,
        ordered(trajectory_lengths, np.int64),
        ordered(p0_norm),
        ordered(step_distance),
        vxy,
        result.v_ego,
    )
    return result


def load_sequences(covla_root, split, video_ids, video_audit, audit, limit=0) -> list[RawSequence]:
    selected_ids = video_ids[:limit] if limit > 0 else video_ids
    sequences = []
    for index, video_id in enumerate(selected_ids, start=1):
        video_record = video_audit[video_id]
        if video_record.get("error"):
            raise RuntimeError(f"ffprobe failed for {video_id}: {video_record['error']}")
        sequences.append(load_state_sequence(covla_root, video_id, split, video_record, audit))
        if index % 100 == 0 or index == len(selected_ids):
            print(f"[state load:{split}] {index}/{len(selected_ids)}", flush=True)
    return sequences


def nearest_state_indices(video_indices: np.ndarray, frame_ids: np.ndarray) -> np.ndarray:
    right = np.searchsorted(frame_ids, video_indices, side="left")
    right = np.clip(right, 0, len(frame_ids) - 1)
    left = np.clip(right - 1, 0, len(frame_ids) - 1)
    choose_left = np.abs(frame_ids[left] - video_indices) <= np.abs(frame_ids[right] - video_indices)
    return np.where(choose_left, left, right).astype(np.int64)


def masked_pair_mean(values: np.ndarray) -> np.ndarray:
    shape = (values.shape[0] // 2, 2) + values.shape[1:]
    paired = values.reshape(shape)
    valid = np.isfinite(paired)
    count = valid.sum(axis=1)
    summed = np.where(valid, paired, 0.0).sum(axis=1)
    result = np.full(summed.shape, np.nan, dtype=np.float64)
    np.divide(summed, count, out=result, where=count > 0)
    return result.astype(np.float32)


def build_scope(raw: RawSequence, scope: str) -> KinematicSequence:
    target_fps = SCOPE_SPECS[scope]
    if target_fps is None:
        return KinematicSequence(
            video_id=raw.video_id,
            split=raw.split,
            scope=scope,
            dt=1.0 / raw.video_fps,
            times=raw.frame_ids.astype(np.float64) / raw.video_fps,
            v_ego=raw.v_ego,
            a_ego=raw.a_ego,
            vx_calib=raw.vx_calib,
            vy_calib=raw.vy_calib,
            ax_calib=raw.ax_calib,
            yaw_rate=raw.yaw_rate,
            trajectory=raw.trajectory,
        )
    stride_float = raw.video_fps / target_fps
    stride = int(round(stride_float))
    if not math.isclose(stride, stride_float, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"Non-integral frame stride for {raw.video_id}: {raw.video_fps=} {target_fps=}")
    num_frames = 64
    clip_len = (num_frames - 1) * stride + 1
    if raw.video_frames < clip_len:
        raise RuntimeError(f"Video too short for standard 64-frame scope: {raw.video_id}")
    start = (raw.video_frames - clip_len) // 2
    video_indices = start + np.arange(num_frames, dtype=np.int64) * stride
    state_indices = nearest_state_indices(video_indices, raw.frame_ids)
    pair_video_indices = video_indices.reshape(-1, 2)

    def pool(values):
        return masked_pair_mean(values[state_indices])

    return KinematicSequence(
        video_id=raw.video_id,
        split=raw.split,
        scope=scope,
        dt=float(np.mean(np.diff(pair_video_indices.mean(axis=1))) / raw.video_fps),
        times=pair_video_indices.mean(axis=1).astype(np.float64) / raw.video_fps,
        v_ego=pool(raw.v_ego),
        a_ego=pool(raw.a_ego),
        vx_calib=pool(raw.vx_calib),
        vy_calib=pool(raw.vy_calib),
        ax_calib=pool(raw.ax_calib),
        yaw_rate=pool(raw.yaw_rate),
        trajectory={point: pool(values) for point, values in raw.trajectory.items()},
    )


def horizon_layout(horizon: float, dt: float) -> tuple[int, float, int]:
    ratio = horizon / dt
    full = int(math.floor(ratio + 1e-10))
    fraction = ratio - full
    if fraction < 1e-8:
        fraction = 0.0
    required = full + (1 if fraction > 0.0 else 0)
    return full, fraction, required


def rotate_components(vx, vy, phi):
    cosine = np.cos(phi)
    sine = np.sin(phi)
    return cosine * vx - sine * vy, sine * vx + cosine * vy


def rollout_uniform(
    vx: np.ndarray,
    vy: np.ndarray,
    omega: np.ndarray,
    dt: float,
    horizon: float,
    method: str,
    vy_sign: int = 1,
    yaw_sign: int = 1,
    rotate: bool = True,
) -> np.ndarray:
    if method not in {"euler", "trapezoid"}:
        raise ValueError(f"Unsupported integration method: {method}")
    full, fraction, required = horizon_layout(horizon, dt)
    count = len(vx) - required
    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    starts = np.arange(count, dtype=np.int64)
    pos_x = np.zeros(count, dtype=np.float64)
    pos_y = np.zeros(count, dtype=np.float64)
    phi = np.zeros(count, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64) * float(vy_sign)
    vx = np.asarray(vx, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64) * float(yaw_sign)

    for step in range(full):
        current = starts + step
        if method == "euler":
            if rotate:
                rx, ry = rotate_components(vx[current], vy[current], phi)
            else:
                rx, ry = vx[current], vy[current]
            pos_x += rx * dt
            pos_y += ry * dt
            if rotate:
                phi += omega[current] * dt
        else:
            following = current + 1
            phi_next = phi + (0.5 * (omega[current] + omega[following]) * dt if rotate else 0.0)
            if rotate:
                rx0, ry0 = rotate_components(vx[current], vy[current], phi)
                rx1, ry1 = rotate_components(vx[following], vy[following], phi_next)
            else:
                rx0, ry0 = vx[current], vy[current]
                rx1, ry1 = vx[following], vy[following]
            pos_x += 0.5 * (rx0 + rx1) * dt
            pos_y += 0.5 * (ry0 + ry1) * dt
            phi = phi_next

    if fraction > 0.0:
        current = starts + full
        following = current + 1
        partial_dt = fraction * dt
        if method == "euler":
            if rotate:
                rx, ry = rotate_components(vx[current], vy[current], phi)
            else:
                rx, ry = vx[current], vy[current]
            pos_x += rx * partial_dt
            pos_y += ry * partial_dt
        else:
            vx_end = vx[current] + fraction * (vx[following] - vx[current])
            vy_end = vy[current] + fraction * (vy[following] - vy[current])
            omega_end = omega[current] + fraction * (omega[following] - omega[current])
            phi_end = phi + (0.5 * (omega[current] + omega_end) * partial_dt if rotate else 0.0)
            if rotate:
                rx0, ry0 = rotate_components(vx[current], vy[current], phi)
                rx1, ry1 = rotate_components(vx_end, vy_end, phi_end)
            else:
                rx0, ry0 = vx[current], vy[current]
                rx1, ry1 = vx_end, vy_end
            pos_x += 0.5 * (rx0 + rx1) * partial_dt
            pos_y += 0.5 * (ry0 + ry1) * partial_dt

    return np.stack([pos_x, pos_y], axis=1)


def integrate_scalar(values: np.ndarray, dt: float, horizon: float, method: str) -> np.ndarray:
    zeros = np.zeros_like(values, dtype=np.float64)
    return rollout_uniform(values, zeros, zeros, dt, horizon, method, rotate=False)[:, 0]


def value_at_horizon(values: np.ndarray, dt: float, horizon: float) -> np.ndarray:
    full, fraction, required = horizon_layout(horizon, dt)
    count = len(values) - required
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    starts = np.arange(count, dtype=np.int64)
    if fraction == 0.0:
        return np.asarray(values[starts + full], dtype=np.float64)
    current = np.asarray(values[starts + full], dtype=np.float64)
    following = np.asarray(values[starts + full + 1], dtype=np.float64)
    return current + fraction * (following - current)


def mean_absolute_over_window(values: np.ndarray, dt: float, horizon: float) -> np.ndarray:
    return integrate_scalar(np.abs(values), dt, horizon, "trapezoid") / horizon


def distribution_summary(values: np.ndarray, signed: bool = True) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    magnitude = np.abs(values)
    result = {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if signed else float(np.mean(magnitude)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p50": float(np.quantile(magnitude, 0.50)),
        "p90": float(np.quantile(magnitude, 0.90)),
        "p95": float(np.quantile(magnitude, 0.95)),
        "p99": float(np.quantile(magnitude, 0.99)),
    }
    if signed:
        result["mae"] = float(np.mean(magnitude))
        result["signed_mean"] = float(np.mean(values))
    return result


class ResidualAggregate:
    def __init__(self):
        self.count = 0
        self.potential = 0
        self.sum_abs_x = 0.0
        self.sum_abs_y = 0.0
        self.sum_norm = 0.0
        self.sum_target_abs_x = 0.0
        self.sum_target_abs_y = 0.0
        self.sum_target_norm = 0.0

    def add(self, prediction: np.ndarray, target: np.ndarray, extra_mask=None):
        count = min(len(prediction), len(target))
        self.potential += count
        if count == 0:
            return
        prediction = prediction[:count]
        target = target[:count]
        valid = np.all(np.isfinite(prediction), axis=1) & np.all(np.isfinite(target), axis=1)
        if extra_mask is not None:
            valid &= np.asarray(extra_mask[:count], dtype=bool)
        if not np.any(valid):
            return
        residual = prediction[valid] - target[valid]
        selected_target = target[valid]
        self.count += int(valid.sum())
        self.sum_abs_x += float(np.abs(residual[:, 0]).sum())
        self.sum_abs_y += float(np.abs(residual[:, 1]).sum())
        self.sum_norm += float(np.linalg.norm(residual, axis=1).sum())
        self.sum_target_abs_x += float(np.abs(selected_target[:, 0]).sum())
        self.sum_target_abs_y += float(np.abs(selected_target[:, 1]).sum())
        self.sum_target_norm += float(np.linalg.norm(selected_target, axis=1).sum())

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0, "potential": self.potential, "valid_rate": 0.0}
        return {
            "count": self.count,
            "potential": self.potential,
            "valid_rate": self.count / max(1, self.potential),
            "x_mae_m": self.sum_abs_x / self.count,
            "y_mae_m": self.sum_abs_y / self.count,
            "norm_mae_m": self.sum_norm / self.count,
            "target_mean_abs_x_m": self.sum_target_abs_x / self.count,
            "target_mean_abs_y_m": self.sum_target_abs_y / self.count,
            "target_mean_norm_m": self.sum_target_norm / self.count,
            "x_relative_mae": self.sum_abs_x / max(self.sum_target_abs_x, 1e-12),
            "y_relative_mae": self.sum_abs_y / max(self.sum_target_abs_y, 1e-12),
            "norm_relative_mae": self.sum_norm / max(self.sum_target_norm, 1e-12),
        }


def x_values(sequence: KinematicSequence, x_field: str) -> np.ndarray:
    if x_field == "vEgo":
        return sequence.v_ego
    if x_field == "velocities_calib[0]":
        return sequence.vx_calib
    raise ValueError(x_field)


def lateral_thresholds(sequences: list[KinematicSequence]) -> dict[float, float]:
    result = {}
    for horizon, point_index in PRIMARY_HORIZONS:
        chunks = [np.abs(sequence.trajectory[point_index][:, 1]) for sequence in sequences]
        values = np.concatenate(chunks)
        values = values[np.isfinite(values)]
        result[horizon] = float(np.quantile(values, 0.75))
    return result


def motion_thresholds(sequences: list[KinematicSequence]) -> dict:
    turn, acceleration, speed = [], [], []
    for sequence in sequences:
        turn.append(mean_absolute_over_window(sequence.yaw_rate, sequence.dt, 1.0))
        acceleration.append(mean_absolute_over_window(sequence.a_ego, sequence.dt, 1.0))
        speed.append(mean_absolute_over_window(sequence.v_ego, sequence.dt, 1.0))

    def values(chunks):
        array = np.concatenate(chunks)
        return array[np.isfinite(array)]

    turn_values = values(turn)
    acceleration_values = values(acceleration)
    speed_values = values(speed)
    return {
        "definition": {
            "turning_strength": "mean absolute angular_velocities_calib[2] over the first 1.0 s",
            "acceleration_strength": "mean absolute aEgo over the first 1.0 s",
            "speed_strength": "mean absolute vEgo over the first 1.0 s",
            "quantile_source": "probe_train, deterministic centered 4fps/64-frame/tubelet=2 view",
        },
        "active_turning_q75_rad_per_s": float(np.quantile(turn_values, 0.75)),
        "active_acceleration_q75_m_per_s2": float(np.quantile(acceleration_values, 0.75)),
        "low_speed_q10_m_per_s": float(np.quantile(speed_values, 0.10)),
        "high_speed_q75_m_per_s": float(np.quantile(speed_values, 0.75)),
        "sample_count": int(len(speed_values)),
    }


def candidate_grid() -> list[Candidate]:
    return [
        Candidate(method, x_field, vy_sign, yaw_sign)
        for method in ("trapezoid", "euler")
        for x_field in ("vEgo", "velocities_calib[0]")
        for vy_sign in (-1, 1)
        for yaw_sign in (-1, 1)
    ]


def evaluate_candidates(
    sequences: list[KinematicSequence],
    active_lateral_thresholds: dict[float, float],
) -> list[dict]:
    records = []
    for candidate in candidate_grid():
        by_horizon = {}
        for horizon, point_index in PRIMARY_HORIZONS:
            overall = ResidualAggregate()
            active = ResidualAggregate()
            for sequence in sequences:
                prediction = rollout_uniform(
                    x_values(sequence, candidate.x_field),
                    sequence.vy_calib,
                    sequence.yaw_rate,
                    sequence.dt,
                    horizon,
                    candidate.method,
                    candidate.vy_sign,
                    candidate.yaw_sign,
                    rotate=True,
                )
                target = sequence.trajectory[point_index][: len(prediction)]
                overall.add(prediction, target)
                active_mask = np.abs(target[:, 1]) >= active_lateral_thresholds[horizon]
                active.add(prediction, target, active_mask)
            by_horizon[str(horizon)] = {
                "overall": overall.summary(),
                "active_lateral_q75": active.summary(),
            }
        score_terms = []
        for metrics in by_horizon.values():
            overall_relative = metrics["overall"].get("norm_relative_mae", math.inf)
            active_y_relative = metrics["active_lateral_q75"].get("y_relative_mae", math.inf)
            score_terms.append(0.5 * overall_relative + 0.5 * active_y_relative)
        score = float(np.mean(score_terms))
        records.append(
            {
                "candidate": candidate.key,
                "method": candidate.method,
                "x_field": candidate.x_field,
                "vy_sign": candidate.vy_sign,
                "yaw_sign": candidate.yaw_sign,
                "selection_score": score,
                "horizons": by_horizon,
            }
        )
    records.sort(key=lambda item: (item["selection_score"], item["method"] != "trapezoid", item["candidate"]))
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def candidate_from_record(record: dict) -> Candidate:
    return Candidate(record["method"], record["x_field"], int(record["vy_sign"]), int(record["yaw_sign"]))


class ResidualCollector:
    def __init__(self):
        self.potential = 0
        self.prediction = []
        self.target = []
        self.turn_strength = []
        self.acceleration_strength = []
        self.speed_strength = []

    def add(self, prediction, target, turn_strength, acceleration_strength, speed_strength):
        count = min(len(prediction), len(target))
        self.potential += count
        if count == 0:
            return
        self.prediction.append(np.asarray(prediction[:count], dtype=np.float32))
        self.target.append(np.asarray(target[:count], dtype=np.float32))
        self.turn_strength.append(np.asarray(turn_strength[:count], dtype=np.float32))
        self.acceleration_strength.append(np.asarray(acceleration_strength[:count], dtype=np.float32))
        self.speed_strength.append(np.asarray(speed_strength[:count], dtype=np.float32))

    def summary(self, thresholds: dict) -> dict:
        if not self.prediction:
            return {"potential": self.potential, "conditions": {}}
        prediction = np.concatenate(self.prediction).astype(np.float64, copy=False)
        target = np.concatenate(self.target).astype(np.float64, copy=False)
        turn = np.concatenate(self.turn_strength).astype(np.float64, copy=False)
        acceleration = np.concatenate(self.acceleration_strength).astype(np.float64, copy=False)
        speed = np.concatenate(self.speed_strength).astype(np.float64, copy=False)
        base_valid = np.all(np.isfinite(prediction), axis=1) & np.all(np.isfinite(target), axis=1)
        conditions = {
            "overall": np.ones(len(prediction), dtype=bool),
            "active_turning": turn >= thresholds["active_turning_q75_rad_per_s"],
            "active_acceleration": acceleration >= thresholds["active_acceleration_q75_m_per_s2"],
            "low_speed": speed <= thresholds["low_speed_q10_m_per_s"],
            "high_speed": speed >= thresholds["high_speed_q75_m_per_s"],
        }
        result = {}
        for name, condition in conditions.items():
            valid = base_valid & condition
            condition_potential = int(np.sum(condition & np.isfinite(turn) & np.isfinite(acceleration) & np.isfinite(speed)))
            result[name] = residual_metrics(
                prediction[valid],
                target[valid],
                potential=self.potential if name == "overall" else condition_potential,
            )
        return {"potential": self.potential, "conditions": result}


def residual_metrics(prediction: np.ndarray, target: np.ndarray, potential: int) -> dict:
    if len(prediction) == 0:
        return {"count": 0, "potential": int(potential), "valid_rate": 0.0}
    residual = prediction - target
    norm = np.linalg.norm(residual, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    x = distribution_summary(residual[:, 0], signed=True)
    y = distribution_summary(residual[:, 1], signed=True)
    norm_metrics = distribution_summary(norm, signed=False)
    norm_metrics["mae"] = float(np.mean(norm))
    target_abs_x = float(np.mean(np.abs(target[:, 0])))
    target_abs_y = float(np.mean(np.abs(target[:, 1])))
    target_mean_norm = float(np.mean(target_norm))
    return {
        "count": int(len(prediction)),
        "potential": int(potential),
        "valid_rate": float(len(prediction) / max(1, potential)),
        "x": x,
        "y": y,
        "norm": norm_metrics,
        "target_mean_abs_x_m": target_abs_x,
        "target_mean_abs_y_m": target_abs_y,
        "target_mean_norm_m": target_mean_norm,
        "x_relative_mae": x["mae"] / max(target_abs_x, 1e-12),
        "y_relative_mae": y["mae"] / max(target_abs_y, 1e-12),
        "norm_relative_mae": norm_metrics["mae"] / max(target_mean_norm, 1e-12),
    }


class ScalarCollector:
    def __init__(self):
        self.potential = 0
        self.residual = []
        self.change = []
        self.acceleration_strength = []

    def add(self, residual, change, acceleration_strength):
        count = min(len(residual), len(change))
        self.potential += count
        self.residual.append(np.asarray(residual[:count], dtype=np.float32))
        self.change.append(np.asarray(change[:count], dtype=np.float32))
        self.acceleration_strength.append(np.asarray(acceleration_strength[:count], dtype=np.float32))

    def summary(self, thresholds):
        residual = np.concatenate(self.residual).astype(np.float64, copy=False)
        change = np.concatenate(self.change).astype(np.float64, copy=False)
        acceleration = np.concatenate(self.acceleration_strength).astype(np.float64, copy=False)
        base = np.isfinite(residual) & np.isfinite(change)
        conditions = {
            "overall": np.ones(len(residual), dtype=bool),
            "active_acceleration": acceleration >= thresholds["active_acceleration_q75_m_per_s2"],
        }
        result = {}
        for name, condition in conditions.items():
            valid = base & condition
            values = residual[valid]
            changes = change[valid]
            potential = self.potential if name == "overall" else int(np.sum(condition & np.isfinite(acceleration)))
            metrics = distribution_summary(values, signed=True)
            mean_abs_change = float(np.mean(np.abs(changes))) if len(changes) else math.nan
            metrics.update(
                {
                    "potential": int(potential),
                    "valid_rate": float(len(values) / max(1, potential)),
                    "mean_abs_velocity_change_m_per_s": mean_abs_change,
                    "relative_mae": metrics.get("mae", math.nan) / max(mean_abs_change, 1e-12),
                }
            )
            result[name] = metrics
        return {"potential": self.potential, "conditions": result}


def motion_features(sequence: KinematicSequence, count: int):
    turn = mean_absolute_over_window(sequence.yaw_rate, sequence.dt, 1.0)[:count]
    acceleration = mean_absolute_over_window(sequence.a_ego, sequence.dt, 1.0)[:count]
    speed = mean_absolute_over_window(sequence.v_ego, sequence.dt, 1.0)[:count]
    return turn, acceleration, speed


def compact_per_video_metrics(prediction, target) -> dict:
    count = min(len(prediction), len(target))
    prediction = prediction[:count]
    target = target[:count]
    valid = np.all(np.isfinite(prediction), axis=1) & np.all(np.isfinite(target), axis=1)
    if not np.any(valid):
        return {"count": 0}
    residual = prediction[valid] - target[valid]
    target_valid = target[valid]
    return {
        "count": int(valid.sum()),
        "x_mae_m": float(np.mean(np.abs(residual[:, 0]))),
        "y_mae_m": float(np.mean(np.abs(residual[:, 1]))),
        "norm_mae_m": float(np.mean(np.linalg.norm(residual, axis=1))),
        "target_mean_norm_m": float(np.mean(np.linalg.norm(target_valid, axis=1))),
    }


def evaluate_split(
    raw_sequences: list[RawSequence],
    selected: Candidate,
    thresholds: dict,
) -> tuple[dict, dict, list[dict]]:
    trajectory_collectors = defaultdict(ResidualCollector)
    va_collectors = defaultdict(ScalarCollector)
    per_video_rows = []

    for raw_index, raw in enumerate(raw_sequences, start=1):
        for scope in SCOPE_SPECS:
            sequence = build_scope(raw, scope)
            for horizon, point_index in PRIMARY_HORIZONS:
                target = sequence.trajectory[point_index]
                selected_prediction = rollout_uniform(
                    x_values(sequence, selected.x_field),
                    sequence.vy_calib,
                    sequence.yaw_rate,
                    sequence.dt,
                    horizon,
                    selected.method,
                    selected.vy_sign,
                    selected.yaw_sign,
                    rotate=True,
                )
                no_rotation_prediction = rollout_uniform(
                    x_values(sequence, selected.x_field),
                    sequence.vy_calib,
                    sequence.yaw_rate,
                    sequence.dt,
                    horizon,
                    selected.method,
                    selected.vy_sign,
                    selected.yaw_sign,
                    rotate=False,
                )
                count = len(selected_prediction)
                turn, acceleration, speed = motion_features(sequence, count)
                for formula, prediction in (
                    ("selected_se2", selected_prediction),
                    ("no_rotation_2d", no_rotation_prediction),
                ):
                    collector = trajectory_collectors[(scope, str(horizon), formula)]
                    collector.add(prediction, target, turn, acceleration, speed)
                    compact = compact_per_video_metrics(prediction, target)
                    per_video_rows.append(
                        {
                            "split": raw.split,
                            "video_id": raw.video_id,
                            "scope": scope,
                            "constraint": "trajectory_closure",
                            "formula": formula,
                            "horizon_seconds": horizon,
                            **compact,
                        }
                    )

            for horizon in VA_HORIZONS:
                endpoint = value_at_horizon(sequence.v_ego, sequence.dt, horizon)
                count = len(endpoint)
                start = sequence.v_ego[:count].astype(np.float64)
                change = endpoint - start
                acceleration_strength = mean_absolute_over_window(sequence.a_ego, sequence.dt, horizon)[:count]
                for method in ("trapezoid", "euler"):
                    predicted_change = integrate_scalar(sequence.a_ego, sequence.dt, horizon, method)
                    residual = change - predicted_change
                    va_collectors[(scope, str(horizon), method)].add(residual, change, acceleration_strength)
                    valid = np.isfinite(residual) & np.isfinite(change)
                    metrics = distribution_summary(residual[valid], signed=True)
                    per_video_rows.append(
                        {
                            "split": raw.split,
                            "video_id": raw.video_id,
                            "scope": scope,
                            "constraint": "velocity_acceleration",
                            "formula": method,
                            "horizon_seconds": horizon,
                            "count": int(valid.sum()),
                            "x_mae_m": metrics.get("mae"),
                            "y_mae_m": None,
                            "norm_mae_m": metrics.get("mae"),
                            "target_mean_norm_m": float(np.mean(np.abs(change[valid]))) if np.any(valid) else None,
                        }
                    )
        if raw_index % 100 == 0 or raw_index == len(raw_sequences):
            print(f"[GT evaluate:{raw.split}] {raw_index}/{len(raw_sequences)}", flush=True)

    trajectory_result = defaultdict(lambda: defaultdict(dict))
    for (scope, horizon, formula), collector in trajectory_collectors.items():
        trajectory_result[scope][horizon][formula] = collector.summary(thresholds)
    va_result = defaultdict(lambda: defaultdict(dict))
    for (scope, horizon, method), collector in va_collectors.items():
        va_result[scope][horizon][method] = collector.summary(thresholds)
    return nested_plain_dict(trajectory_result), nested_plain_dict(va_result), per_video_rows


def nested_plain_dict(value):
    if isinstance(value, defaultdict):
        value = dict(value)
    if isinstance(value, dict):
        return {str(key): nested_plain_dict(item) for key, item in value.items()}
    return value


def endpoint_semantics(raw_sequences: list[RawSequence]) -> dict:
    result = {}
    for point_index in (19, 20, 39, 40, 59):
        chunks = [sequence.trajectory[point_index] for sequence in raw_sequences]
        values = np.concatenate(chunks).astype(np.float64, copy=False)
        valid = np.all(np.isfinite(values), axis=1)
        values = values[valid]
        result[str(point_index)] = {
            "point_index": point_index,
            "zero_based_time_seconds_at_20fps": point_index / 20.0,
            "count": int(len(values)),
            "mean_abs_x_m": float(np.mean(np.abs(values[:, 0]))),
            "mean_abs_y_m": float(np.mean(np.abs(values[:, 1]))),
            "mean_norm_m": float(np.mean(np.linalg.norm(values, axis=1))),
        }
    return result


def video_audit_summary(records: dict[str, dict], expected_fps: float) -> dict:
    errors = {key: value["error"] for key, value in records.items() if value.get("error")}
    valid = [value for value in records.values() if not value.get("error")]
    patterns = Counter(
        (round(value["fps"], 8), value["frames"], round(value["duration_seconds"], 6))
        for value in valid
    )
    return {
        "count": len(records),
        "valid_count": len(valid),
        "errors": errors,
        "patterns": [
            {"fps": key[0], "frames": key[1], "duration_seconds": key[2], "count": count}
            for key, count in sorted(patterns.items())
        ],
        "all_expected_fps": bool(valid)
        and not errors
        and all(math.isclose(item["fps"], expected_fps, abs_tol=1e-8) for item in valid),
    }


def candidate_rows_for_csv(split: str, ranking: list[dict]) -> list[dict]:
    rows = []
    for item in ranking:
        row = {
            "split": split,
            "rank": item["rank"],
            "candidate": item["candidate"],
            "selection_score": item["selection_score"],
            "method": item["method"],
            "x_field": item["x_field"],
            "vy_sign": item["vy_sign"],
            "yaw_sign": item["yaw_sign"],
        }
        for horizon, metrics in item["horizons"].items():
            row[f"h{horizon}_norm_relative"] = metrics["overall"].get("norm_relative_mae")
            row[f"h{horizon}_active_y_relative"] = metrics["active_lateral_q75"].get("y_relative_mae")
        rows.append(row)
    return rows


def choose_va_method(va_train: dict) -> dict:
    scores = {}
    for method in ("trapezoid", "euler"):
        terms = []
        for horizon in VA_HORIZONS:
            metrics = va_train["tubelet_4fps"][str(horizon)][method]["conditions"]["overall"]
            terms.append(metrics["mae"])
        scores[method] = float(np.mean(terms))
    selected = min(scores, key=lambda key: (scores[key], key != "trapezoid"))
    return {"selected_method": selected, "train_tubelet_4fps_mean_mae": scores}


def gate_decision(
    video_summary: dict,
    state_summary: dict,
    trajectory_val: dict,
    selected: Candidate,
    val_ranking: list[dict],
) -> dict:
    checks = []

    def add(name, passed, value, criterion, required=True):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "required": required,
                "value": value,
                "criterion": criterion,
            }
        )

    add(
        "video_fps",
        video_summary["all_expected_fps"],
        video_summary["patterns"],
        "all probe_train/probe_val videos are readable at exactly 20 fps",
    )
    step_median = state_summary["trajectory_step_seconds_from_speed_norm"].get("p50", math.nan)
    add(
        "trajectory_step",
        math.isfinite(step_median) and 0.045 <= step_median <= 0.055,
        step_median,
        "median trajectory point interval inferred from speed is within [0.045, 0.055] s",
    )
    for horizon, _ in PRIMARY_HORIZONS:
        metrics = trajectory_val["tubelet_4fps"][str(horizon)]["selected_se2"]["conditions"]["overall"]
        add(
            f"val_valid_h{horizon}",
            metrics["valid_rate"] >= 0.99,
            metrics["valid_rate"],
            "probe_val valid rate >= 0.99",
        )
        add(
            f"val_relative_oracle_h{horizon}",
            metrics["norm_relative_mae"] <= 0.15,
            metrics["norm_relative_mae"],
            "probe_val 2D oracle MAE / mean trajectory magnitude <= 0.15",
        )
    turning_improvements = 0
    turning_deltas = {}
    for horizon, _ in PRIMARY_HORIZONS:
        selected_metrics = trajectory_val["tubelet_4fps"][str(horizon)]["selected_se2"]["conditions"]["active_turning"]
        baseline_metrics = trajectory_val["tubelet_4fps"][str(horizon)]["no_rotation_2d"]["conditions"]["active_turning"]
        delta = baseline_metrics["y"]["mae"] - selected_metrics["y"]["mae"]
        turning_deltas[str(horizon)] = delta
        turning_improvements += int(delta > 0.0)
    add(
        "active_turning_specificity",
        turning_improvements >= 2,
        turning_deltas,
        "SE(2) lowers active-turning lateral MAE versus no-rotation in at least 2/3 horizons",
    )
    selected_val_record = next(item for item in val_ranking if item["candidate"] == selected.key)
    add(
        "validation_rank_stability",
        selected_val_record["rank"] <= 4,
        selected_val_record["rank"],
        "probe_train-frozen candidate remains within top 4/16 on probe_val",
        required=False,
    )
    required_pass = all(item["passed"] for item in checks if item["required"])
    return {
        "status": "PASS" if required_pass else "FAIL",
        "checks": checks,
        "note": "The decision concerns GT formula legality only; it is not a model result.",
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test():
    count = 50
    dt = 0.1
    vx = np.full(count, 2.0)
    zero = np.zeros(count)
    straight = rollout_uniform(vx, zero, zero, dt, 1.0, "trapezoid")
    if not np.allclose(straight, np.asarray([2.0, 0.0]), atol=1e-10):
        raise AssertionError("straight-line trapezoid integration failed")
    fractional = rollout_uniform(vx, zero, zero, dt, 0.95, "trapezoid")
    if not np.allclose(fractional, np.asarray([1.9, 0.0]), atol=1e-10):
        raise AssertionError("fractional-horizon integration failed")
    omega = np.full(count, 0.2)
    circular = rollout_uniform(vx, zero, omega, dt, 1.0, "trapezoid")[0]
    expected = np.asarray([2.0 / 0.2 * math.sin(0.2), 2.0 / 0.2 * (1.0 - math.cos(0.2))])
    if not np.allclose(circular, expected, atol=2e-4):
        raise AssertionError(f"circular integration failed: {circular=} {expected=}")
    print("self-test passed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--covla-root",
        type=Path,
        default=Path("data/CoVLA-Dataset"),
    )
    parser.add_argument("--split-dir", type=Path, default=Path("time-jepa/splits/covla_weather_ood"))
    parser.add_argument("--selection-split", default="probe_train")
    parser.add_argument("--validation-split", default="probe_val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("physics_consistency/covla/results/gt_gate"),
    )
    parser.add_argument("--expected-video-fps", type=float, default=20.0)
    parser.add_argument("--ffprobe-workers", type=int, default=16)
    parser.add_argument("--max-train-videos", type=int, default=0)
    parser.add_argument("--max-val-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.selection_split not in ALLOWED_SPLITS or args.validation_split not in ALLOWED_SPLITS:
        raise ValueError("Only probe_train and probe_val are permitted")
    if args.selection_split == args.validation_split:
        raise ValueError("Selection and validation splits must differ")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ids, train_split_path = load_split_ids(args.split_dir, args.selection_split)
    val_ids, val_split_path = load_split_ids(args.split_dir, args.validation_split)
    if set(train_ids) & set(val_ids):
        raise RuntimeError("probe_train and probe_val overlap")
    if args.max_train_videos > 0:
        train_ids = train_ids[: args.max_train_videos]
    if args.max_val_videos > 0:
        val_ids = val_ids[: args.max_val_videos]

    metadata = read_metadata(args.covla_root)
    all_ids = train_ids + val_ids
    video_records = audit_videos(all_ids, metadata, args.ffprobe_workers)
    video_summary = video_audit_summary(video_records, args.expected_video_fps)
    if video_summary["errors"]:
        raise RuntimeError(f"Video audit errors: {list(video_summary['errors'].items())[:5]}")

    train_audit = StateAudit()
    val_audit = StateAudit()
    train_raw = load_sequences(args.covla_root, args.selection_split, train_ids, video_records, train_audit)
    val_raw = load_sequences(args.covla_root, args.validation_split, val_ids, video_records, val_audit)
    state_summary = {
        args.selection_split: train_audit.summary(),
        args.validation_split: val_audit.summary(),
        "combined": combine_state_audits(train_audit, val_audit).summary(),
    }

    train_main_scope = [build_scope(sequence, "tubelet_4fps") for sequence in train_raw]
    val_main_scope = [build_scope(sequence, "tubelet_4fps") for sequence in val_raw]
    active_lateral_thresholds = lateral_thresholds(train_main_scope)
    thresholds = motion_thresholds(train_main_scope)
    thresholds["active_lateral_q75_m_by_horizon"] = {
        str(key): value for key, value in active_lateral_thresholds.items()
    }

    print("[candidate selection] probe_train", flush=True)
    train_ranking = evaluate_candidates(train_main_scope, active_lateral_thresholds)
    selected = candidate_from_record(train_ranking[0])
    print(f"[candidate selection] frozen={selected.key}", flush=True)
    print("[candidate validation] probe_val", flush=True)
    val_ranking = evaluate_candidates(val_main_scope, active_lateral_thresholds)

    trajectory_train, va_train, per_video_train = evaluate_split(train_raw, selected, thresholds)
    trajectory_val, va_val, per_video_val = evaluate_split(val_raw, selected, thresholds)
    va_selection = choose_va_method(va_train)

    combined_state_summary = state_summary["combined"]
    decision = gate_decision(video_summary, combined_state_summary, trajectory_val, selected, val_ranking)
    selected_val_record = next(item for item in val_ranking if item["candidate"] == selected.key)

    payload = {
        "metadata": {
            "gate": "covla_weather_ood_gt_kinematic_consistency",
            "selection_split": args.selection_split,
            "validation_split": args.validation_split,
            "forbidden_selection_splits": ["test", "weather_ood_test"],
            "covla_root": str(args.covla_root.resolve()),
            "split_dir": str(args.split_dir.resolve()),
            "selection_split_sha256": sha256_file(train_split_path),
            "validation_split_sha256": sha256_file(val_split_path),
            "selection_video_count": len(train_raw),
            "validation_video_count": len(val_raw),
            "is_full_run": args.max_train_videos == 0 and args.max_val_videos == 0,
            "scope_definitions": {
                "raw_20fps": "all contiguous state/video frame IDs at verified 20 fps",
                "tubelet_4fps": "deterministic centered 64 frames, stride 5, adjacent sampled-frame pairs averaged",
                "tubelet_10fps": "legacy sensitivity: deterministic centered 64 frames, stride 2, adjacent sampled-frame pairs averaged",
            },
        },
        "video_audit": video_summary,
        "state_and_time_audit": state_summary,
        "trajectory_endpoint_semantics": {
            "probe_train": endpoint_semantics(train_raw),
            "probe_val": endpoint_semantics(val_raw),
            "frozen_primary_points": [
                {"horizon_seconds": horizon, "trajectory_point_index": point}
                for horizon, point in PRIMARY_HORIZONS
            ],
            "interpretation": "trajectory[0] is the current origin; at verified 20 fps, point i denotes i/20 seconds. Therefore 20/40 are exact 1/2 s; 59 is 2.95 s. Existing 19/39 labels must not be called exact 1/2 s.",
        },
        "train_frozen_thresholds": thresholds,
        "coordinate_and_integrator_selection": {
            "objective": "mean over 1.0/2.0/2.95 s of 0.5*(overall 2D MAE / mean path norm) + 0.5*(active-lateral y MAE / mean |trajectory y|)",
            "scope": "probe_train deterministic tubelet_4fps only",
            "selected": {
                "candidate": selected.key,
                "method": selected.method,
                "x_field": selected.x_field,
                "vy_sign": selected.vy_sign,
                "yaw_sign": selected.yaw_sign,
                "trajectory_y_sign": 1,
                "train_rank": 1,
                "validation_rank_without_reselection": selected_val_record["rank"],
            },
            "probe_train_ranking": train_ranking,
            "probe_val_frozen_audit_ranking": val_ranking,
        },
        "trajectory_closure": {
            args.selection_split: trajectory_train,
            args.validation_split: trajectory_val,
        },
        "velocity_acceleration_closure": {
            "selection": va_selection,
            args.selection_split: va_train,
            args.validation_split: va_val,
        },
        "gate_decision": decision,
        "outputs": {
            "metrics_json": str((args.output_dir / "gt_gate_metrics.json").resolve()),
            "candidate_csv": str((args.output_dir / "gt_gate_candidate_ranking.csv").resolve()),
            "per_video_csv": str((args.output_dir / "gt_gate_per_video.csv").resolve()),
            "video_metadata_csv": str((args.output_dir / "gt_gate_video_metadata.csv").resolve()),
        },
    }

    metrics_path = args.output_dir / "gt_gate_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    write_csv(
        args.output_dir / "gt_gate_candidate_ranking.csv",
        candidate_rows_for_csv(args.selection_split, train_ranking)
        + candidate_rows_for_csv(args.validation_split, val_ranking),
    )
    write_csv(args.output_dir / "gt_gate_per_video.csv", per_video_train + per_video_val)
    video_rows = []
    split_by_id = {video_id: args.selection_split for video_id in train_ids}
    split_by_id.update({video_id: args.validation_split for video_id in val_ids})
    state_rows_by_id = {sequence.video_id: len(sequence.frame_ids) for sequence in train_raw + val_raw}
    for video_id in all_ids:
        row = video_records[video_id]
        video_rows.append(
            {
                "split": split_by_id[video_id],
                "video_id": video_id,
                "fps": row.get("fps"),
                "video_frames": row.get("frames"),
                "state_rows": state_rows_by_id.get(video_id),
                "duration_seconds": row.get("duration_seconds"),
                "error": row.get("error"),
            }
        )
    write_csv(args.output_dir / "gt_gate_video_metadata.csv", video_rows)
    print(f"[done] {metrics_path}", flush=True)
    print(f"[gate] {decision['status']}", flush=True)


def combine_state_audits(first: StateAudit, second: StateAudit) -> StateAudit:
    combined = StateAudit()
    for name in (
        "frame_delta",
        "timestamp_delta_ms",
        "unique_timestamp_delta_ms",
        "trajectory_lengths",
        "state_counts",
    ):
        getattr(combined, name).update(getattr(first, name))
        getattr(combined, name).update(getattr(second, name))
    for name in (
        "trajectory_p0_norm",
        "trajectory_step_dt_speed",
        "trajectory_step_dt_vego",
        "timestamp_spans_seconds",
    ):
        getattr(combined, name).extend(getattr(first, name))
        getattr(combined, name).extend(getattr(second, name))
    return combined


if __name__ == "__main__":
    main()
