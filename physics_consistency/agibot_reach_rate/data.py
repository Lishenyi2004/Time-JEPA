"""AgiBot rows, q/velocity heads, and independently derived reach-rate GT."""

from __future__ import annotations

import fcntl
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from physics_consistency.agibot.bank_data import (
    load_protocol_rows,
    load_single_view_splits,
)
from physics_consistency.agibot_reach_rate.common import (
    ARMS,
    P_SLICES,
    TARGET_DEFINITION,
    TARGET_FIELDS,
    TARGET_NAMES,
    atomic_json_dump,
    canonical_hash,
)
from physics_consistency.agibot_reach_rate.reach_math import (
    derive_reach_rate_series,
)
from src.time_jepa.agibot_derived_targets import (
    END_ARM_POSITION_SLICE,
    END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS,
    SERIES_DERIVED_MIN_STATE_LEN,
    derive_agibot_series_field,
)

from probe_datasets import (
    AgiBotProbeDataset,
    agibot_sample_times,
    agibot_times_to_frame_indices,
    build_agibot_tubelet_targets,
    nearest_time_indices,
)


JOINT_POSITION_SLICE = slice(40, 54)
Q_FIELDS = tuple(f"state/joint/position[{index}]" for index in range(14))
P_FIELDS = tuple(f"state/end/arm_position[{index}]" for index in range(6))


@lru_cache(maxsize=64)
def load_reach_series(parquet_path: str) -> dict[str, np.ndarray]:
    """Load the raw q/endpoint stream and exact v2 signed-velocity targets."""

    import pandas as pd

    frame = pd.read_parquet(
        parquet_path, columns=["timestamp", "observation.state"]
    )
    if len(frame) == 0:
        raise RuntimeError(f"Empty AgiBot parquet: {parquet_path}")
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    states = np.stack(
        [np.asarray(value, dtype=np.float64) for value in frame["observation.state"]],
        axis=0,
    )
    order = np.argsort(timestamps)
    timestamps, states = timestamps[order], states[order]
    if states.shape[1] < SERIES_DERIVED_MIN_STATE_LEN:
        raise RuntimeError(
            f"AgiBot reach rate requires state dim >= {SERIES_DERIVED_MIN_STATE_LEN}, "
            f"got {states.shape[1]} in {parquet_path}"
        )

    q = np.asarray(states[:, JOINT_POSITION_SLICE], dtype=np.float32)
    q_valid = np.isfinite(q).astype(np.float32)
    position = np.asarray(states[:, END_ARM_POSITION_SLICE], dtype=np.float32)
    position_valid = np.stack(
        [
            np.isfinite(position[:, P_SLICES[arm]]).all(axis=1)
            for arm in ARMS
        ],
        axis=-1,
    ).astype(np.float32)
    velocity_values, velocity_valid = [], []
    for field in END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS:
        values, valid = derive_agibot_series_field(timestamps, states, field)
        velocity_values.append(values)
        velocity_valid.append(valid)
    velocity = np.stack(velocity_values, axis=-1).astype(np.float32)
    velocity_valid = np.stack(velocity_valid, axis=-1).astype(np.float32)
    return {
        "timestamps": timestamps,
        "q": q,
        "q_valid": q_valid,
        "position": position,
        "position_valid": position_valid,
        "velocity": velocity,
        "velocity_valid": velocity_valid,
    }


def _pool_sampled(
    values: np.ndarray,
    valid: np.ndarray,
    state_indices: np.ndarray,
    time_valid: np.ndarray,
    tubelet_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if valid.ndim == 1:
        valid = valid[:, None]
    sampled_values = values[state_indices]
    sampled_valid = valid[state_indices] * time_valid[:, None]
    token_count = len(state_indices) // int(tubelet_size)
    width = sampled_values.shape[-1]
    sampled_values = sampled_values.reshape(token_count, int(tubelet_size), width)
    sampled_valid = sampled_valid.reshape(token_count, int(tubelet_size), width)
    counts = np.maximum(sampled_valid.sum(axis=1), 1.0)
    pooled = (sampled_values * sampled_valid).sum(axis=1) / counts
    pooled_valid = (sampled_valid.sum(axis=1) > 0).astype(np.float32)
    return pooled.astype(np.float32), pooled_valid


def build_reach_rate_targets(
    parquet_path: str | Path,
    sample_times: np.ndarray,
    sampling: dict,
    shoulder_points: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Build q/v head targets plus independent endpoint/reach-rate GT."""

    sample_times = np.asarray(sample_times, dtype=np.float64)
    tubelet = int(sampling["tubelet_size"])
    if len(sample_times) % tubelet:
        raise ValueError("sample_times must be divisible by tubelet_size")
    series = load_reach_series(str(Path(parquet_path).expanduser().resolve()))
    max_delta = float(sampling.get("max_time_delta", 0.05))
    state_indices, time_valid = nearest_time_indices(
        sample_times, series["timestamps"], max_delta=max_delta
    )
    target_values = np.concatenate([series["q"], series["velocity"]], axis=-1)
    target_valid = np.concatenate(
        [series["q_valid"], series["velocity_valid"]], axis=-1
    )
    target, valid = _pool_sampled(
        target_values, target_valid, state_indices, time_valid, tubelet
    )

    position_component_valid = np.concatenate(
        [
            np.repeat(
                series["position_valid"][:, arm_index : arm_index + 1],
                3,
                axis=1,
            )
            for arm_index in range(len(ARMS))
        ],
        axis=1,
    )
    endpoint, endpoint_component_valid = _pool_sampled(
        series["position"],
        position_component_valid,
        state_indices,
        time_valid,
        tubelet,
    )
    endpoint_valid = np.stack(
        [
            np.all(endpoint_component_valid[:, P_SLICES[arm]] > 0, axis=-1)
            for arm in ARMS
        ],
        axis=-1,
    ).astype(np.float32)

    _, reach_rate, reach_valid = derive_reach_rate_series(
        series, shoulder_points
    )
    pooled_reach_rate, pooled_reach_valid = _pool_sampled(
        reach_rate, reach_valid, state_indices, time_valid, tubelet
    )
    token_times = sample_times.reshape(-1, tubelet).mean(axis=1)
    return {
        "target": torch.from_numpy(target),
        "valid": torch.from_numpy(valid),
        "endpoint_position": torch.from_numpy(endpoint),
        "endpoint_valid": torch.from_numpy(endpoint_valid),
        "reach_rate_gt": torch.from_numpy(pooled_reach_rate),
        "reach_rate_valid": torch.from_numpy(pooled_reach_valid),
        "target_sample_times": torch.from_numpy(
            token_times.astype(np.float32)
        ),
    }


class AgiBotReachRateDataset(AgiBotProbeDataset):
    def __init__(
        self,
        rows,
        sampling,
        shoulder_points,
        transform,
        random_clip=False,
        stats=None,
        data_config=None,
    ):
        data_config = data_config or {}
        super().__init__(
            rows=rows,
            target_fields=TARGET_FIELDS,
            num_frames=int(sampling["num_frames"]),
            transform=transform,
            random_clip=random_clip,
            stats=None,
            tubelet_size=int(sampling["tubelet_size"]),
            video_num_threads=int(data_config.get("video_num_threads", 1)),
            agibot_fps=float(sampling["fps"]),
            agibot_max_time_delta=float(sampling.get("max_time_delta", 0.05)),
            agibot_video_backend=data_config.get("video_backend", "ffmpeg"),
            agibot_ffmpeg_timeout=float(data_config.get("ffmpeg_timeout", 120)),
            agibot_slice_stride_seconds=float(
                data_config.get("slice_stride_seconds", 12.8)
            ),
        )
        self.sampling = dict(sampling)
        self.shoulder_points = {
            arm: np.asarray(shoulder_points[arm], dtype=np.float64)
            for arm in ARMS
        }
        self.reach_stats = stats

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get("window_idx") is not None and row.get("slice_stride_seconds") is None:
            row = {**row, "slice_stride_seconds": self.agibot_slice_stride_seconds}
        sample_times = agibot_sample_times(
            row, self.num_frames, self.agibot_fps, self.random_clip
        )
        frame_indices = agibot_times_to_frame_indices(
            sample_times,
            video_fps=float(row.get("raw_fps", 30.0)),
            num_frames=int(row.get("length", 0)),
        )
        video = self.transform(
            [frame for frame in self._read_clip(row, frame_indices)]
        )
        payload = build_reach_rate_targets(
            row["parquet_path"],
            sample_times,
            self.sampling,
            self.shoulder_points,
        )
        target = payload["target"]
        if self.reach_stats is not None:
            mean, std = self.reach_stats
            target = (target - mean) / std
        payload.update(
            {
                "video": video,
                "target": target.float(),
                "valid": payload["valid"].float(),
                "frame_indices": torch.from_numpy(frame_indices.copy()),
                "target_frame_indices": torch.from_numpy(
                    frame_indices.copy()
                ).reshape(-1, self.tubelet_size),
                "sample_times": torch.from_numpy(
                    sample_times.astype(np.float32)
                ),
                "video_id": row["video_id"],
                "episode_key": row["episode_key"],
                "window_key": row.get("window_key", row["episode_key"]),
                "task_id": int(row["task_id"]),
                "view_key": row["video_key"],
            }
        )
        return payload


def stats_fingerprint(
    rows: list[dict],
    sampling: dict,
    seed: int,
    max_rows: int,
) -> dict:
    payload = {
        "schema": 1,
        "target_definition": TARGET_DEFINITION,
        "target_names": TARGET_NAMES,
        "window_views": sorted(row["video_id"] for row in rows),
        "sampling": {
            "num_frames": int(sampling["num_frames"]),
            "fps": float(sampling["fps"]),
            "tubelet_size": int(sampling["tubelet_size"]),
            "max_time_delta": float(sampling.get("max_time_delta", 0.05)),
        },
        "stats_seed": int(seed),
        "max_rows": int(max_rows),
    }
    return {"sha256": canonical_hash(payload), "payload": payload}


def compute_stats(
    rows: list[dict],
    sampling: dict,
    shoulder_points: dict[str, np.ndarray],
    seed: int = 239,
    max_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_by_window = {}
    for row in rows:
        key = row.get("window_key", row.get("episode_key", row["video_id"]))
        selected_by_window.setdefault(key, row)
    selected = list(selected_by_window.values())
    rng = random.Random(int(seed))
    if max_rows > 0 and len(selected) > max_rows:
        selected = rng.sample(selected, int(max_rows))
    sums = torch.zeros(len(TARGET_NAMES), dtype=torch.float64)
    sq_sums = torch.zeros_like(sums)
    counts = torch.zeros_like(sums)
    for row in selected:
        sample_times = agibot_sample_times(
            row, int(sampling["num_frames"]), float(sampling["fps"]), False
        )
        payload = build_reach_rate_targets(
            row["parquet_path"], sample_times, sampling, shoulder_points
        )
        target = payload["target"].double()
        valid = payload["valid"].double()
        sums += (target * valid).sum(0)
        sq_sums += (target.square() * valid).sum(0)
        counts += valid.sum(0)
    if (counts < 2).any():
        missing = [
            TARGET_NAMES[index]
            for index in torch.where(counts < 2)[0].tolist()
        ]
        raise RuntimeError(f"Insufficient normalization samples for {missing}")
    mean = sums / counts
    variance = (sq_sums - sums.square() / counts) / (counts - 1).clamp_min(1)
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return mean.float(), std.float(), counts


def get_or_compute_stats(
    rows: list[dict],
    sampling: dict,
    shoulder_points: dict[str, np.ndarray],
    cache_path: str | Path,
    seed: int = 239,
    max_rows: int = 0,
    distributed: bool = False,
    rank: int = 0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    cache_path = Path(cache_path).expanduser().resolve()
    fingerprint = stats_fingerprint(rows, sampling, seed, max_rows)
    payload = None
    if not distributed or rank == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache_path.exists():
                candidate = json.loads(cache_path.read_text())
                if candidate.get("fingerprint", {}).get("sha256") == fingerprint["sha256"]:
                    payload = candidate
            if payload is None:
                mean, std, counts = compute_stats(
                    rows, sampling, shoulder_points, seed, max_rows
                )
                payload = {
                    "schema": 1,
                    "fingerprint": fingerprint,
                    "target_names": list(TARGET_NAMES),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "counts": counts.tolist(),
                }
                atomic_json_dump(payload, cache_path)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        packed = torch.tensor(
            [payload["mean"], payload["std"]], dtype=torch.float32
        )
    else:
        packed = torch.empty(2, len(TARGET_NAMES), dtype=torch.float32)
    if distributed:
        communication_device = (
            device if device.type == "cuda" else torch.device("cpu")
        )
        packed = packed.to(communication_device)
        dist.broadcast(packed, src=0)
        packed = packed.cpu()
        objects = [payload]
        dist.broadcast_object_list(objects, src=0)
        payload = objects[0]
    return packed[0].clone(), packed[1].clone(), payload


def collect_reach_targets(
    rows: list[dict],
    sampling: dict,
    shoulder_points: dict[str, np.ndarray],
    progress_every: int = 0,
    progress_label: str = "reach targets",
) -> dict:
    tensors = {
        key: []
        for key in (
            "target",
            "valid",
            "endpoint_position",
            "endpoint_valid",
            "reach_rate_gt",
            "reach_rate_valid",
            "target_sample_times",
        )
    }
    task_ids, episode_keys, window_keys = [], [], []
    for row_index, row in enumerate(rows):
        sample_times = agibot_sample_times(
            row, int(sampling["num_frames"]), float(sampling["fps"]), False
        )
        payload = build_reach_rate_targets(
            row["parquet_path"], sample_times, sampling, shoulder_points
        )
        token_count = len(payload["target"])
        for key in tensors:
            tensors[key].append(payload[key].numpy())
        task_ids.extend([int(row["task_id"])] * token_count)
        episode_keys.extend([row["episode_key"]] * token_count)
        window_keys.extend(
            [row.get("window_key", row["episode_key"])] * token_count
        )
        if progress_every and (row_index + 1) % int(progress_every) == 0:
            print(
                f"{progress_label}: {row_index + 1}/{len(rows)} windows",
                flush=True,
            )
    output = {
        key: np.concatenate(value, axis=0)
        for key, value in tensors.items()
    }
    output.update(
        {
            "task_id": np.asarray(task_ids, dtype=np.int64),
            "episode_key": np.asarray(episode_keys, dtype=str),
            "window_key": np.asarray(window_keys, dtype=str),
        }
    )
    return output


def collect_qp_targets(
    rows: list[dict],
    sampling: dict,
    progress_every: int = 0,
) -> dict:
    fields = (*Q_FIELDS, *P_FIELDS)
    target, valid = [], []
    for row_index, row in enumerate(rows):
        sample_times = agibot_sample_times(
            row, int(sampling["num_frames"]), float(sampling["fps"]), False
        )
        values, mask = build_agibot_tubelet_targets(
            row["parquet_path"],
            sample_times,
            fields,
            int(sampling["tubelet_size"]),
            max_time_delta=float(sampling.get("max_time_delta", 0.05)),
        )
        target.append(values.numpy())
        valid.append(mask.numpy())
        if progress_every and (row_index + 1) % int(progress_every) == 0:
            print(
                f"q/p calibration targets: {row_index + 1}/{len(rows)} windows",
                flush=True,
            )
    return {
        "target": np.concatenate(target, axis=0),
        "valid": np.concatenate(valid, axis=0),
    }


def load_gate_rows(config: dict) -> dict[str, list[dict]]:
    data = config["data"]
    view = data.get("gate_view", "observation.images.top_head")
    names = (
        str(config.get("selection_split", "encoder_train")),
        str(config.get("threshold_split", "probe_train")),
        str(config.get("validation_split", "probe_val")),
    )
    loaded = load_single_view_splits(data, view, names)
    return {
        "selection": loaded[names[0]],
        "threshold": loaded[names[1]],
        "validation": loaded[names[2]],
    }


__all__ = [
    "AgiBotReachRateDataset",
    "P_FIELDS",
    "Q_FIELDS",
    "build_reach_rate_targets",
    "collect_qp_targets",
    "collect_reach_targets",
    "compute_stats",
    "derive_reach_rate_series",
    "get_or_compute_stats",
    "load_gate_rows",
    "load_protocol_rows",
    "load_reach_series",
]
