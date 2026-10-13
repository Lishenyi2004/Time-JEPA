"""Lightweight, video-free data path used only by the train/val GT Gate.

This intentionally avoids importing the probe model registry (and therefore
torch/timm/wandb).  Its sampling and pooling definitions match the training
dataset, while all inputs are raw metadata and parquet state streams.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from physics_consistency.agibot_reach_rate.common import (
    ARMS,
    P_SLICES,
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


JOINT_POSITION_SLICE = slice(40, 54)


def _task_ids(value) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item.strip()]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(item) for item in value]


def _episode_key(row: dict) -> str:
    session = row.get("session_id")
    episode = f"episode_{int(row['episode_index']):06d}"
    if session:
        return f"task_{row['task_id']}/{session}/{episode}"
    return f"task_{row['task_id']}/{episode}"


def _discover_task_dirs(root: Path, task_id: int) -> list[tuple[str | None, Path]]:
    task_root = root / f"task_{task_id}"
    old_style = task_root / "data"
    if (old_style / "meta" / "info.json").is_file():
        return [(None, old_style)]
    return [
        (data_dir.parent.name, data_dir)
        for data_dir in sorted(task_root.glob("*/data"))
        if (data_dir / "meta" / "info.json").is_file()
    ]


def _feature_shape(info: dict, video_key: str) -> tuple[int | None, int | None]:
    shape = (
        info.get("features", {}).get(video_key, {}).get("shape") or []
    )
    if len(shape) < 2:
        return None, None
    return int(shape[0]), int(shape[1])


def _feature_fps(info: dict, video_key: str) -> float:
    feature = info.get("features", {}).get(video_key, {})
    return float(
        feature.get("video_info", {}).get(
            "video.fps", info.get("fps", 30)
        )
    )


def read_metadata(
    root: str | Path,
    tasks,
    video_key: str,
    use_meta_episodes_only: bool = True,
    allowed_episode_keys: set[str] | None = None,
) -> list[dict]:
    root = Path(root).expanduser().resolve()
    rows = []
    for task_id in _task_ids(tasks):
        task_dirs = _discover_task_dirs(root, task_id)
        if not task_dirs:
            raise FileNotFoundError(
                f"No AgiBot data directory for task_{task_id} under {root}"
            )
        for session_id, task_dir in task_dirs:
            info_path = task_dir / "meta" / "info.json"
            episodes_path = task_dir / "meta" / "episodes.jsonl"
            if not episodes_path.is_file():
                raise FileNotFoundError(
                    f"Missing AgiBot episodes metadata: {episodes_path}"
                )
            info = json.loads(info_path.read_text())
            if video_key not in info.get("features", {}):
                raise ValueError(
                    f"video_key {video_key!r} missing from {info_path}"
                )
            raw_fps = _feature_fps(info, video_key)
            height, width = _feature_shape(info, video_key)
            data_template = info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.parquet",
            )
            video_template = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4",
            )
            records = [
                json.loads(line)
                for line in episodes_path.read_text().splitlines()
                if line.strip()
            ]
            if not use_meta_episodes_only:
                known = {int(row["episode_index"]) for row in records}
                for path in sorted(
                    (task_dir / "data" / "chunk-000").glob(
                        "episode_*.parquet"
                    )
                ):
                    episode_index = int(path.stem.split("_")[-1])
                    if episode_index not in known:
                        records.append(
                            {"episode_index": episode_index, "length": None}
                        )
            for record in records:
                episode_index = int(record["episode_index"])
                identity = {
                    "task_id": str(task_id),
                    "session_id": session_id,
                    "episode_index": episode_index,
                }
                episode_key = _episode_key(identity)
                if (
                    allowed_episode_keys is not None
                    and episode_key not in allowed_episode_keys
                ):
                    continue
                chunk = episode_index // int(info.get("chunks_size", 1000))
                parquet_path = task_dir / data_template.format(
                    episode_chunk=chunk,
                    episode_index=episode_index,
                )
                video_path = task_dir / video_template.format(
                    episode_chunk=chunk,
                    episode_index=episode_index,
                    video_key=video_key,
                )
                # Mirror the formal probe metadata filter exactly even though
                # the Gate itself never decodes the video.
                if not parquet_path.is_file() or not video_path.is_file():
                    continue
                length = record.get("length")
                if length is None:
                    import pandas as pd

                    length = len(
                        pd.read_parquet(
                            parquet_path, columns=["timestamp"]
                        )
                    )
                rows.append(
                    {
                        **identity,
                        "video_id": f"{episode_key}/{video_key}",
                        "episode_key": episode_key,
                        "video_key": video_key,
                        "video_path": str(video_path),
                        "parquet_path": str(parquet_path),
                        "length": int(length),
                        "raw_fps": raw_fps,
                        "height": height,
                        "width": width,
                        "duration": max(
                            0.0,
                            (float(length) - 1.0) / raw_fps,
                        )
                        if length
                        else 0.0,
                    }
                )
    if not rows:
        raise RuntimeError(
            f"No usable AgiBot rows under {root} for {video_key}"
        )
    return rows


def split_rows(
    rows: list[dict],
    split_dir: str | Path,
    split_names: tuple[str, ...],
    prefix: str,
    keys_by_split: dict[str, set[str]] | None = None,
) -> dict[str, list[dict]]:
    split_dir = Path(split_dir)
    if keys_by_split is None:
        keys_by_split = {}
        for name in split_names:
            path = split_dir / f"{prefix}_{name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing AgiBot split file: {path}")
            keys_by_split[name] = set(json.loads(path.read_text()))
    has_windows = any(
        "/window_" in str(key)
        for keys in keys_by_split.values()
        for key in keys
    )
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_key"], []).append(row)
    output = {}
    for name in split_names:
        selected, missing = [], []
        for key in sorted(keys_by_split[name]):
            episode_key = (
                key.rsplit("/window_", 1)[0] if has_windows else key
            )
            matches = by_episode.get(episode_key)
            if not matches:
                missing.append(key)
                continue
            for source in matches:
                row = dict(source)
                if has_windows:
                    row["window_idx"] = int(
                        key.rsplit("/window_", 1)[1]
                    )
                    row["window_key"] = key
                    row["video_id"] = f"{key}/{row['video_key']}"
                selected.append(row)
        if missing:
            raise RuntimeError(
                f"{len(missing)} keys missing from {name}; "
                f"first={missing[:5]}"
            )
        output[name] = selected
    return output


def load_gate_rows(config: dict) -> dict[str, list[dict]]:
    data = config["data"]
    view = data.get("gate_view", "observation.images.top_head")
    split_names = (
        str(config.get("selection_split", "encoder_train")),
        str(config.get("threshold_split", "probe_train")),
        str(config.get("validation_split", "probe_val")),
    )
    prefix = data.get("split_prefix", "agibot_top_head")
    limits = data.get("limits", {})
    keys_by_split = {}
    for name in split_names:
        path = Path(data["split_dir"]) / f"{prefix}_{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing AgiBot split file: {path}")
        keys = sorted(set(json.loads(path.read_text())))
        limit = int(limits.get(name, 0) or 0)
        if limit > 0:
            keys = keys[:limit]
        keys_by_split[name] = set(keys)
    allowed_episode_keys = {
        key.rsplit("/window_", 1)[0]
        for keys in keys_by_split.values()
        for key in keys
    }
    metadata = read_metadata(
        data["agibot_root"],
        data["tasks"],
        view,
        bool(data.get("use_meta_episodes_only", True)),
        allowed_episode_keys,
    )
    splits = split_rows(
        metadata,
        data["split_dir"],
        split_names,
        prefix,
        keys_by_split,
    )
    for name in split_names:
        for row in splits[name]:
            row["slice_stride_seconds"] = float(
                data.get("slice_stride_seconds", 12.8)
            )
        if not splits[name]:
            raise RuntimeError(f"AgiBot Gate split {name} is empty")
    return {
        "selection": splits[split_names[0]],
        "threshold": splits[split_names[1]],
        "validation": splits[split_names[2]],
    }


def sample_times(row: dict, sampling: dict) -> np.ndarray:
    num_frames = int(sampling["num_frames"])
    fps = float(sampling["fps"])
    if row.get("window_idx") is not None:
        stride = float(
            row.get("slice_stride_seconds", num_frames / fps)
        )
        start = float(row["window_idx"]) * stride
    else:
        span = (num_frames - 1) / fps
        start = max(0.0, float(row.get("duration", 0.0)) - span) / 2.0
    return start + np.arange(num_frames, dtype=np.float64) / fps


def nearest_time_indices(
    query_times: np.ndarray,
    source_times: np.ndarray,
    max_delta: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_times, dtype=np.float64)
    source = np.asarray(source_times, dtype=np.float64)
    right = np.clip(
        np.searchsorted(source, query, side="left"), 0, len(source) - 1
    )
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_left = np.abs(source[left] - query) <= np.abs(
        source[right] - query
    )
    indices = np.where(choose_left, left, right).astype(np.int64)
    delta = np.abs(source[indices] - query)
    valid = (
        np.ones_like(delta, dtype=np.float32)
        if max_delta is None
        else (delta <= float(max_delta)).astype(np.float32)
    )
    return indices, valid


@lru_cache(maxsize=64)
def load_reach_series(parquet_path: str) -> dict[str, np.ndarray]:
    import pandas as pd

    frame = pd.read_parquet(
        parquet_path, columns=["timestamp", "observation.state"]
    )
    if len(frame) == 0:
        raise RuntimeError(f"Empty AgiBot parquet: {parquet_path}")
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    states = np.stack(
        [
            np.asarray(value, dtype=np.float64)
            for value in frame["observation.state"]
        ],
        axis=0,
    )
    order = np.argsort(timestamps)
    timestamps, states = timestamps[order], states[order]
    if states.shape[1] < SERIES_DERIVED_MIN_STATE_LEN:
        raise RuntimeError(
            f"AgiBot state dim {states.shape[1]} is too small in "
            f"{parquet_path}"
        )
    q = np.asarray(states[:, JOINT_POSITION_SLICE], dtype=np.float32)
    position = np.asarray(
        states[:, END_ARM_POSITION_SLICE], dtype=np.float32
    )
    velocity_values, velocity_valid = [], []
    for field in END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS:
        value, valid = derive_agibot_series_field(
            timestamps, states, field
        )
        velocity_values.append(value)
        velocity_valid.append(valid)
    return {
        "timestamps": timestamps,
        "q": q,
        "q_valid": np.isfinite(q).astype(np.float32),
        "position": position,
        "position_valid": np.stack(
            [
                np.isfinite(position[:, P_SLICES[arm]]).all(axis=1)
                for arm in ARMS
            ],
            axis=-1,
        ).astype(np.float32),
        "velocity": np.stack(
            velocity_values, axis=-1
        ).astype(np.float32),
        "velocity_valid": np.stack(
            velocity_valid, axis=-1
        ).astype(np.float32),
    }


def _pool(
    values: np.ndarray,
    valid: np.ndarray,
    indices: np.ndarray,
    time_valid: np.ndarray,
    tubelet_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if valid.ndim == 1:
        valid = valid[:, None]
    sampled = values[indices]
    mask = valid[indices] * time_valid[:, None]
    token_count = len(indices) // int(tubelet_size)
    sampled = sampled.reshape(token_count, int(tubelet_size), -1)
    mask = mask.reshape(token_count, int(tubelet_size), -1)
    counts = np.maximum(mask.sum(axis=1), 1.0)
    return (
        ((sampled * mask).sum(axis=1) / counts).astype(np.float32),
        (mask.sum(axis=1) > 0).astype(np.float32),
    )


def build_targets(
    row: dict,
    sampling: dict,
    shoulder_points: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    times = sample_times(row, sampling)
    tubelet = int(sampling["tubelet_size"])
    series = load_reach_series(row["parquet_path"])
    indices, time_valid = nearest_time_indices(
        times,
        series["timestamps"],
        sampling.get("max_time_delta", 0.05),
    )
    target, valid = _pool(
        np.concatenate([series["q"], series["velocity"]], axis=-1),
        np.concatenate(
            [series["q_valid"], series["velocity_valid"]], axis=-1
        ),
        indices,
        time_valid,
        tubelet,
    )
    position_valid = np.concatenate(
        [
            np.repeat(
                series["position_valid"][:, arm_index : arm_index + 1],
                3,
                axis=1,
            )
            for arm_index in range(2)
        ],
        axis=1,
    )
    endpoint, endpoint_component_valid = _pool(
        series["position"],
        position_valid,
        indices,
        time_valid,
        tubelet,
    )
    endpoint_valid = np.stack(
        [
            np.all(
                endpoint_component_valid[:, P_SLICES[arm]] > 0,
                axis=-1,
            )
            for arm in ARMS
        ],
        axis=-1,
    ).astype(np.float32)
    output = {
        "target": target,
        "valid": valid,
        "endpoint_position": endpoint,
        "endpoint_valid": endpoint_valid,
        "target_sample_times": times.reshape(-1, tubelet)
        .mean(axis=1)
        .astype(np.float32),
    }
    if shoulder_points is not None:
        _, rate, rate_valid = derive_reach_rate_series(
            series, shoulder_points
        )
        output["reach_rate_gt"], output["reach_rate_valid"] = _pool(
            rate,
            rate_valid,
            indices,
            time_valid,
            tubelet,
        )
    return output


def collect_qp_targets(
    rows: list[dict],
    sampling: dict,
    progress_every: int = 0,
) -> dict:
    targets, valid = [], []
    for index, row in enumerate(rows):
        payload = build_targets(row, sampling, None)
        targets.append(
            np.concatenate(
                [payload["target"][:, :14], payload["endpoint_position"]],
                axis=-1,
            )
        )
        endpoint_component_valid = np.repeat(
            payload["endpoint_valid"], 3, axis=-1
        ).reshape(len(payload["endpoint_valid"]), 6)
        valid.append(
            np.concatenate(
                [payload["valid"][:, :14], endpoint_component_valid],
                axis=-1,
            )
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"q/p calibration targets: {index + 1}/{len(rows)} windows",
                flush=True,
            )
    return {
        "target": np.concatenate(targets, axis=0),
        "valid": np.concatenate(valid, axis=0),
    }


def collect_reach_targets(
    rows: list[dict],
    sampling: dict,
    shoulder_points: dict[str, np.ndarray],
    progress_every: int = 0,
    progress_label: str = "reach targets",
) -> dict:
    keys = (
        "target",
        "valid",
        "endpoint_position",
        "endpoint_valid",
        "reach_rate_gt",
        "reach_rate_valid",
        "target_sample_times",
    )
    values = {key: [] for key in keys}
    task_ids, episode_keys, window_keys = [], [], []
    for index, row in enumerate(rows):
        payload = build_targets(row, sampling, shoulder_points)
        token_count = len(payload["target"])
        for key in keys:
            values[key].append(payload[key])
        task_ids.extend([int(row["task_id"])] * token_count)
        episode_keys.extend([row["episode_key"]] * token_count)
        window_keys.extend(
            [row.get("window_key", row["episode_key"])] * token_count
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"{progress_label}: {index + 1}/{len(rows)} windows",
                flush=True,
            )
    output = {
        key: np.concatenate(items, axis=0)
        for key, items in values.items()
    }
    output.update(
        {
            "task_id": np.asarray(task_ids, dtype=np.int64),
            "episode_key": np.asarray(episode_keys, dtype=str),
            "window_key": np.asarray(window_keys, dtype=str),
        }
    )
    return output


__all__ = [
    "collect_qp_targets",
    "collect_reach_targets",
    "load_gate_rows",
]
