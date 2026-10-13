"""CoVLA/KITTI metadata, split loading, and probe Dataset classes."""

import json
import random
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset

from probe_sampling import (
    build_tubelet_targets,
    compute_effective_frame_stride,
    map_video_frames_to_state_indices,
    normalize_future_horizons_seconds,
    sample_frame_indices,
)
from probe_targets import load_agibot_state_series, load_kitti_oxts_series, load_state_series, load_trajectory_xy_series


def parse_int_list(value):
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_ratio_list(value, expected_len=4):
    ratios = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(ratios) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated ratios, got {value!r}")
    total = sum(ratios)
    if total <= 0:
        raise ValueError(f"Split ratios must sum to a positive value, got {value!r}")
    return [ratio / total for ratio in ratios]


def read_metadata(covla_root):
    covla_root = Path(covla_root)
    metadata_path = covla_root / "metadata.jsonl"
    rows = []
    with metadata_path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            video_path = covla_root / rec["file_name"]
            video_id = rec.get("video_id", video_path.stem)
            states_path = covla_root / "states" / f"{video_id}.jsonl"
            if video_path.exists() and states_path.exists():
                rows.append(
                    {
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "states_path": str(states_path),
                    }
                )
    if not rows:
        raise RuntimeError(f"No usable CoVLA rows found under {covla_root}")
    return rows


def split_rows(rows, train_ratio=0.8, val_ratio=0.1, seed=0):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def load_split_ids(split_dir):
    split_dir = Path(split_dir)
    split_ids = {}
    for split_path in sorted(split_dir.glob("covla_*.json")):
        if split_path.name.startswith("covla_split_manifest"):
            continue
        split_name = split_path.stem[len("covla_") :]
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        with split_path.open("r") as f:
            split_ids[split_name] = set(json.load(f))
    return split_ids


def split_rows_from_files(rows, split_dir, split_names=("train", "val", "test")):
    split_ids = load_split_ids(split_dir)
    by_id = {row["video_id"]: row for row in rows}
    splits = {}
    missing_split_names = [name for name in split_names if name not in split_ids]
    if missing_split_names:
        available = sorted(split_ids)
        raise FileNotFoundError(
            f"Missing split files for {missing_split_names} under {split_dir}; available splits: {available}"
        )
    for split_name in split_names:
        ids = split_ids[split_name]
        missing = sorted(ids - set(by_id))
        if missing:
            raise RuntimeError(f"{len(missing)} ids from {split_name} split are missing from metadata; first={missing[:5]}")
        splits[split_name] = [by_id[video_id] for video_id in sorted(ids)]
    return [splits[name] for name in split_names]


def load_kitti_split_rows(split_dir, split_names=("probe_train", "probe_val", "test")):
    split_dir = Path(split_dir)
    splits = []
    for split_name in split_names:
        split_path = split_dir / f"kitti_{split_name}.json"
        if not split_path.exists():
            available = sorted(p.name for p in split_dir.glob("kitti_*.json"))
            raise FileNotFoundError(
                f"Missing KITTI split file: {split_path}; available KITTI split files: {available}"
            )
        with split_path.open("r") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise RuntimeError(f"KITTI split file must contain a list: {split_path}")
        splits.append(rows)
    return splits


def _agibot_feature_shape(info, video_key):
    feature = info.get("features", {}).get(video_key, {})
    shape = feature.get("shape") or []
    if len(shape) < 2:
        return None, None
    return int(shape[0]), int(shape[1])


def _agibot_feature_fps(info, video_key):
    feature = info.get("features", {}).get(video_key, {})
    video_info = feature.get("video_info", {})
    return float(video_info.get("video.fps", info.get("fps", 30)))


def discover_agibot_task_data_dirs(root, task_id):
    task_root = Path(root) / f"task_{task_id}"
    old_style = task_root / "data"
    if (old_style / "meta" / "info.json").exists():
        return [(None, old_style)]
    return [
        (data_dir.parent.name, data_dir)
        for data_dir in sorted(task_root.glob("*/data"))
        if (data_dir / "meta" / "info.json").exists()
    ]


def read_agibot_metadata(
    root,
    tasks=(3641, 3777),
    video_key="observation.images.top_head",
    use_meta_episodes_only=True,
):
    root = Path(root).expanduser().resolve()
    rows = []
    for task in parse_int_list(tasks):
        task_id = str(task)
        task_data_dirs = discover_agibot_task_data_dirs(root, task_id)
        if not task_data_dirs:
            raise FileNotFoundError(
                f"No AgiBot data directories found for task_{task_id} under {root}; "
                "expected task_<id>/data or task_<id>/<session>/data"
            )
        for session_id, task_dir in task_data_dirs:
            info_path = task_dir / "meta" / "info.json"
            episodes_path = task_dir / "meta" / "episodes.jsonl"
            if not episodes_path.exists():
                raise FileNotFoundError(f"AgiBot episodes.jsonl not found: {episodes_path}")
            with info_path.open("r") as f:
                info = json.load(f)
            if video_key not in info.get("features", {}):
                raise ValueError(f"AgiBot video_key {video_key!r} not found in {info_path}")
            raw_fps = _agibot_feature_fps(info, video_key)
            height, width = _agibot_feature_shape(info, video_key)
            data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
            video_template = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )

            episode_records = []
            with episodes_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episode_records.append(json.loads(line))
            if not use_meta_episodes_only:
                known = {int(record["episode_index"]) for record in episode_records}
                for parquet_path in sorted((task_dir / "data" / "chunk-000").glob("episode_*.parquet")):
                    episode_index = int(parquet_path.stem.split("_")[-1])
                    if episode_index not in known:
                        episode_records.append({"episode_index": episode_index, "length": None})

            for record in episode_records:
                episode_index = int(record["episode_index"])
                episode_chunk = episode_index // int(info.get("chunks_size", 1000))
                parquet_path = task_dir / data_template.format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
                video_path = task_dir / video_template.format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=video_key,
                )
                if not parquet_path.exists() or not video_path.exists():
                    continue
                length = record.get("length")
                if length is None:
                    import pandas as pd

                    length = len(pd.read_parquet(parquet_path, columns=["timestamp"]))
                duration = max(0.0, (float(length) - 1.0) / raw_fps) if length else 0.0
                ep_key = agibot_episode_key(
                    {
                        "task_id": task_id,
                        "session_id": session_id,
                        "episode_index": episode_index,
                    }
                )
                rows.append(
                    {
                        "video_id": f"{ep_key}/{video_key}",
                        "task_id": task_id,
                        "session_id": session_id,
                        "episode_index": episode_index,
                        "episode_key": ep_key,
                        "video_key": video_key,
                        "video_path": str(video_path),
                        "parquet_path": str(parquet_path),
                        "length": int(length),
                        "raw_fps": raw_fps,
                        "height": height,
                        "width": width,
                        "duration": duration,
                    }
                )
    if not rows:
        raise RuntimeError(f"No usable AgiBot rows found under {root} tasks={tasks} video_key={video_key}")
    return rows


def agibot_episode_key(row):
    """Camera-agnostic episode key matching make_agibot_splits.py."""
    session_id = row.get("session_id")
    episode = f"episode_{int(row['episode_index']):06d}"
    if session_id:
        return f"task_{row['task_id']}/{session_id}/{episode}"
    return f"task_{row['task_id']}/{episode}"


def agibot_window_key(row, window_idx):
    return f"{agibot_episode_key(row)}/window_{int(window_idx):06d}"


def is_agibot_window_key(key):
    return "/window_" in str(key)


def load_agibot_split_keys(split_dir, split_names, prefix="agibot"):
    split_dir = Path(split_dir)
    split_keys = {}
    for split_name in split_names:
        split_path = split_dir / f"{prefix}_{split_name}.json"
        if not split_path.exists():
            available = sorted(p.name for p in split_dir.glob(f"{prefix}_*.json"))
            raise FileNotFoundError(
                f"Missing AgiBot split file: {split_path}; available: {available}"
            )
        with split_path.open("r") as f:
            split_keys[split_name] = set(json.load(f))
    return split_keys


def split_agibot_rows_from_files(rows, split_dir, split_names=("probe_train", "probe_val", "test"), prefix="agibot"):
    """Partition AgiBot rows using persisted episode-key split files, so the
    probe shares the exact episode protocol used by the encoder (no leakage,
    reproducible). Mirrors split_rows_from_files for CoVLA."""
    split_keys = load_agibot_split_keys(split_dir, split_names, prefix=prefix)
    has_window_keys = any(
        is_agibot_window_key(key)
        for keys in split_keys.values()
        for key in keys
    )
    by_key = {}
    for row in rows:
        by_key.setdefault(agibot_episode_key(row), []).append(row)
    splits = {}
    for split_name in split_names:
        keys = split_keys[split_name]
        if has_window_keys:
            selected = []
            missing = []
            for key in sorted(keys):
                episode_key = key.rsplit("/window_", 1)[0]
                if episode_key not in by_key:
                    missing.append(key)
                    continue
                window_idx = int(key.rsplit("/window_", 1)[1])
                for row in by_key[episode_key]:
                    row = dict(row)
                    row["window_idx"] = window_idx
                    row["window_key"] = key
                    row["video_id"] = f"{key}/{row['video_key']}"
                    selected.append(row)
            if missing:
                raise RuntimeError(
                    f"{len(missing)} window keys from AgiBot {split_name} split are missing "
                    f"from metadata; first={missing[:5]}"
                )
            splits[split_name] = selected
        else:
            missing = sorted(keys - set(by_key))
            if missing:
                raise RuntimeError(
                    f"{len(missing)} episodes from AgiBot {split_name} split are missing "
                    f"from metadata; first={missing[:5]}"
                )
            splits[split_name] = [row for key in sorted(keys) for row in by_key[key]]
    return [splits[name] for name in split_names]


def split_agibot_rows(rows, ratios="70,15,5,10", seed=0, split_names=("encoder_train", "probe_train", "probe_val", "test")):
    ratios = parse_ratio_list(ratios, expected_len=len(split_names))
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    counts = [int(n * ratio) for ratio in ratios[:-1]]
    counts.append(n - sum(counts))
    splits = {}
    cursor = 0
    for split_name, count in zip(split_names, counts):
        splits[split_name] = rows[cursor : cursor + count]
        cursor += count
    return [splits[name] for name in split_names]


def agibot_sample_times(row, num_frames, fps, random_clip, rng=None, future_horizon_seconds=0.0):
    if row.get("window_idx") is not None:
        stride_seconds = row.get("slice_stride_seconds", float(num_frames) / float(fps))
        start_time = float(row["window_idx"]) * float(stride_seconds)
        return start_time + np.arange(int(num_frames), dtype=np.float64) / float(fps)
    span = (int(num_frames) - 1) / float(fps)
    max_start = max(
        0.0,
        float(row.get("duration", 0.0)) - span - max(0.0, float(future_horizon_seconds)),
    )
    if random_clip and max_start > 0:
        if rng is None:
            start_time = float(np.random.uniform(0.0, max_start))
        else:
            start_time = float(rng.random() * max_start)
    else:
        start_time = max_start / 2.0
    return start_time + np.arange(int(num_frames), dtype=np.float64) / float(fps)


def agibot_times_to_frame_indices(sample_times, video_fps, num_frames):
    indices = np.rint(np.asarray(sample_times, dtype=np.float64) * float(video_fps))
    return np.clip(indices, 0, int(num_frames) - 1).astype(np.int64)


def nearest_time_indices(query_times, source_times, max_delta=None):
    query_times = np.asarray(query_times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
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


def build_agibot_tubelet_targets(parquet_path, sample_times, target_fields, tubelet_size, max_time_delta=None):
    timestamps, values, valid = load_agibot_state_series(parquet_path, tuple(target_fields))
    state_indices, time_valid = nearest_time_indices(sample_times, timestamps, max_delta=max_time_delta)
    per_frame_target = torch.from_numpy(values[state_indices])
    per_frame_valid = torch.from_numpy(valid[state_indices] * time_valid[:, None])
    target = per_frame_target.reshape(-1, tubelet_size, per_frame_target.size(-1))
    valid_mask = per_frame_valid.reshape(-1, tubelet_size, per_frame_valid.size(-1))
    valid_sum = valid_mask.sum(dim=1).clamp_min(1.0)
    target = (target * valid_mask).sum(dim=1) / valid_sum
    valid_mask = (valid_mask.sum(dim=1) > 0).float()
    return target.float(), valid_mask.float()


@lru_cache(maxsize=512)
def probe_video_metadata(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed with code {proc.returncode}: {stderr}")
    payload = json.loads(proc.stdout.decode("utf-8"))
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream: {video_path}")
    stream = streams[0]

    def parse_rate(value):
        numerator, denominator = str(value or "0/1").split("/", 1)
        denominator = float(denominator)
        return float(numerator) / denominator if denominator else 0.0

    width = int(stream["width"])
    height = int(stream["height"])
    video_fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate")) or 30.0
    raw_num_frames = stream.get("nb_frames")
    if raw_num_frames not in (None, "", "N/A"):
        num_frames = int(raw_num_frames)
    else:
        raw_duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
        if raw_duration in (None, "", "N/A"):
            raise RuntimeError(f"ffprobe could not determine frame count or duration: {video_path}")
        num_frames = max(1, int(round(float(raw_duration) * video_fps)))
    if width <= 0 or height <= 0 or num_frames <= 0:
        raise RuntimeError(
            f"Invalid ffprobe metadata for {video_path}: "
            f"width={width} height={height} fps={video_fps} frames={num_frames}"
        )
    return video_fps, num_frames, height, width


def read_ffmpeg_frames(video_path, frame_indices, height, width, timeout=120, seek_time=None):
    unique_indices, inverse = np.unique(np.asarray(frame_indices, dtype=np.int64), return_inverse=True)
    select_expr = "+".join(f"eq(n\\,{int(idx)})" for idx in unique_indices.tolist())
    frame_bytes = int(height) * int(width) * 3
    expected_bytes = len(unique_indices) * frame_bytes
    cmd = [
        "ffmpeg",
        "-v",
        "error",
    ]
    if seek_time is not None and float(seek_time) > 0:
        cmd += ["-ss", f"{float(seek_time):.6f}"]
    cmd += [
        "-i",
        str(video_path),
        "-vf",
        f"select={select_expr}",
        "-vsync",
        "0",
        "-frames:v",
        str(len(unique_indices)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=None if timeout is None else float(timeout),
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {stderr}")
    if len(proc.stdout) != expected_bytes:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg returned {len(proc.stdout)} bytes, expected {expected_bytes}; stderr={stderr}")
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(len(unique_indices), int(height), int(width), 3)
    return frames[inverse].copy()


class CoVLAProbeDataset(Dataset):
    def __init__(
        self,
        rows,
        target_fields,
        num_frames,
        transform,
        random_clip=False,
        stats=None,
        frame_span=None,
        frame_stride=0,
        tubelet_size=2,
        video_num_threads=1,
        target_mode="state",
        trajectory_point_indices=(-1,),
        future_horizon_seconds=1.0,
        future_horizons_seconds=None,
    ):
        self.rows = rows
        self.target_fields = tuple(target_fields)
        self.num_frames = num_frames
        self.transform = transform
        self.random_clip = random_clip
        self.stats = stats
        self.frame_span = frame_span
        self.frame_stride = frame_stride
        self.tubelet_size = tubelet_size
        self.video_num_threads = video_num_threads
        self.target_mode = target_mode
        self.trajectory_point_indices = tuple(trajectory_point_indices)
        self.future_horizons_seconds = normalize_future_horizons_seconds(
            future_horizons_seconds,
            future_horizon_seconds,
        )
        self.future_horizon_seconds = max(self.future_horizons_seconds)
        if self.tubelet_size <= 0:
            raise ValueError(f"tubelet_size must be positive, got {self.tubelet_size}")
        if self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                f"num_frames must be divisible by tubelet_size for tubelet targets, "
                f"got {self.num_frames=} {self.tubelet_size=}"
            )

    def __len__(self):
        return len(self.rows)

    def effective_frame_stride(self, num_video_frames):
        return compute_effective_frame_stride(num_video_frames, self.num_frames, self.frame_span, self.frame_stride)

    def _sample_indices(self, num_video_frames, reserve_after=0):
        return sample_frame_indices(
            num_video_frames,
            self.num_frames,
            frame_span=self.frame_span,
            frame_stride=self.frame_stride,
            random_clip=self.random_clip,
            reserve_after=reserve_after,
        )

    def __getitem__(self, index):
        row = self.rows[index]
        vr = VideoReader(row["video_path"], num_threads=self.video_num_threads, ctx=cpu(0))
        if len(vr) <= 0:
            raise RuntimeError(f"Empty video: {row['video_path']}")
        video_fps = float(vr.get_avg_fps())
        future_frame_offsets = (
            np.asarray(
                [
                    max(1, int(round(horizon * video_fps)))
                    for horizon in self.future_horizons_seconds
                ],
                dtype=np.int64,
            )
            if self.target_mode == "future_state"
            else np.asarray([], dtype=np.int64)
        )
        if self.target_mode == "future_state" and np.any(np.diff(future_frame_offsets) <= 0):
            raise RuntimeError(
                "Configured future horizons collapse to duplicate video frames at "
                f"fps={video_fps}: horizons={self.future_horizons_seconds}, "
                f"offsets={future_frame_offsets.tolist()}"
            )
        reserve_after = int(future_frame_offsets[-1]) if len(future_frame_offsets) else 0
        indices = self._sample_indices(len(vr), reserve_after=reserve_after)
        video = vr.get_batch(indices).asnumpy()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        video = self.transform(video)

        if self.target_mode == "trajectory_xy":
            frame_ids, values, valid = load_trajectory_xy_series(
                row["states_path"],
                point_indices_tuple=self.trajectory_point_indices,
            )
        elif self.target_mode in ("state", "future_state"):
            frame_ids, values, valid = load_state_series(row["states_path"], self.target_fields)
        else:
            raise ValueError(f"Unsupported CoVLA target_mode: {self.target_mode}")
        if self.target_mode == "future_state":
            future_frame_indices = int(indices[-1]) + future_frame_offsets
            if int(future_frame_indices[-1]) >= len(vr):
                raise RuntimeError(
                    f"Future target escaped reserved video range for {row['video_id']}: "
                    f"last_context={int(indices[-1])} offsets={future_frame_offsets.tolist()} "
                    f"frames={len(vr)}"
                )
            target_state_index = map_video_frames_to_state_indices(
                future_frame_indices, frame_ids, len(vr)
            )
            current_state_index = map_video_frames_to_state_indices(
                np.asarray([int(indices[-1])]), frame_ids, len(vr)
            )
            target, valid_mask = build_tubelet_targets(values, valid, target_state_index, 1)
            current_target, current_valid = build_tubelet_targets(values, valid, current_state_index, 1)
            target_frame_indices = torch.from_numpy(future_frame_indices.copy()).reshape(-1, 1)
        else:
            state_indices = map_video_frames_to_state_indices(indices, frame_ids, len(vr))
            target, valid_mask = build_tubelet_targets(values, valid, state_indices, self.tubelet_size)
            target_frame_indices = torch.from_numpy(indices.copy()).reshape(-1, self.tubelet_size)

        if self.stats is not None:
            mean, std = self.stats
            target = (target - mean) / std

        sample = {
            "video": video,
            "target": target,
            "valid": valid_mask,
            "frame_indices": torch.from_numpy(indices.copy()),
            "target_frame_indices": target_frame_indices,
            "video_id": row["video_id"],
        }
        if self.target_mode == "future_state":
            sample.update(
                {
                    "current_target": current_target,
                    "current_valid": current_valid,
                    "future_horizon_seconds": torch.tensor(self.future_horizon_seconds),
                    "future_horizons_seconds": torch.tensor(
                        self.future_horizons_seconds,
                        dtype=torch.float32,
                    ),
                }
            )
        return sample


def kitti_frame_path(image_dir, frame_idx):
    return Path(image_dir) / f"{int(frame_idx):010d}.png"


class AgiBotProbeDataset(Dataset):
    def __init__(
        self,
        rows,
        target_fields,
        num_frames,
        transform,
        random_clip=False,
        stats=None,
        frame_span=None,
        frame_stride=0,
        tubelet_size=2,
        video_num_threads=1,
        target_mode="state",
        trajectory_point_indices=(-1,),
        agibot_fps=5.0,
        agibot_max_time_delta=0.05,
        agibot_video_backend="ffmpeg",
        agibot_ffmpeg_timeout=120.0,
        agibot_slice_stride_seconds=12.8,
        future_horizon_seconds=1.0,
        future_horizons_seconds=None,
    ):
        del frame_span, frame_stride, trajectory_point_indices
        self.rows = rows
        self.target_fields = tuple(target_fields)
        self.num_frames = int(num_frames)
        self.transform = transform
        self.random_clip = bool(random_clip)
        self.stats = stats
        self.tubelet_size = int(tubelet_size)
        self.video_num_threads = int(video_num_threads)
        self.agibot_fps = float(agibot_fps)
        self.agibot_max_time_delta = agibot_max_time_delta
        self.agibot_video_backend = str(agibot_video_backend or "ffmpeg").lower()
        self.agibot_ffmpeg_timeout = agibot_ffmpeg_timeout
        self.agibot_slice_stride_seconds = float(agibot_slice_stride_seconds)
        self.target_mode = str(target_mode)
        self.future_horizons_seconds = normalize_future_horizons_seconds(
            future_horizons_seconds,
            future_horizon_seconds,
        )
        self.future_horizon_seconds = max(self.future_horizons_seconds)
        if self.target_mode not in ("state", "future_state"):
            raise ValueError(f"Unsupported AgiBot target_mode: {self.target_mode}")
        if self.agibot_video_backend not in ("auto", "decord", "ffmpeg"):
            raise ValueError("agibot_video_backend must be one of: auto, decord, ffmpeg")
        if self.tubelet_size <= 0:
            raise ValueError(f"tubelet_size must be positive, got {self.tubelet_size}")
        if self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                f"num_frames must be divisible by tubelet_size for tubelet targets, "
                f"got {self.num_frames=} {self.tubelet_size=}"
            )

    def __len__(self):
        return len(self.rows)

    def effective_frame_stride(self, num_video_frames):
        raw_fps = float(self.rows[0].get("raw_fps", 30.0)) if self.rows else 30.0
        return max(1, int(round(raw_fps / self.agibot_fps)))

    def _read_decord_clip(self, row, frame_indices):
        vr = VideoReader(row["video_path"], num_threads=self.video_num_threads, ctx=cpu(0))
        if len(vr) <= 0:
            raise RuntimeError(f"Empty video: {row['video_path']}")
        frame_indices = np.clip(frame_indices, 0, len(vr) - 1).astype(np.int64)
        return vr.get_batch(frame_indices.tolist()).asnumpy()

    def _read_ffmpeg_clip(self, row, frame_indices):
        height = row.get("height")
        width = row.get("width")
        if height is None or width is None:
            raise RuntimeError(f"AgiBot row is missing video shape metadata: {row['video_path']}")
        first_frame_idx = int(frame_indices[0])
        relative_indices = frame_indices - first_frame_idx
        seek_time = float(first_frame_idx) / float(row.get("raw_fps", 30.0))
        return read_ffmpeg_frames(
            row["video_path"],
            relative_indices,
            height=height,
            width=width,
            timeout=self.agibot_ffmpeg_timeout,
            seek_time=seek_time,
        )

    def _read_clip(self, row, frame_indices):
        if self.agibot_video_backend in ("auto", "decord"):
            try:
                return self._read_decord_clip(row, frame_indices)
            except Exception:
                if self.agibot_video_backend == "decord":
                    raise
        return self._read_ffmpeg_clip(row, frame_indices)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get("window_idx") is not None and row.get("slice_stride_seconds") is None:
            row = dict(row)
            row["slice_stride_seconds"] = self.agibot_slice_stride_seconds
        sample_times = agibot_sample_times(
            row,
            self.num_frames,
            self.agibot_fps,
            self.random_clip,
            future_horizon_seconds=(
                self.future_horizon_seconds if self.target_mode == "future_state" else 0.0
            ),
        )
        frame_indices = agibot_times_to_frame_indices(
            sample_times,
            video_fps=float(row.get("raw_fps", 30.0)),
            num_frames=int(row.get("length", 0)),
        )
        video = self._read_clip(row, frame_indices)
        video = self.transform([frame for frame in video])

        if self.target_mode == "future_state":
            future_times = float(sample_times[-1]) + np.asarray(
                self.future_horizons_seconds,
                dtype=np.float64,
            )
            target, valid_mask = build_agibot_tubelet_targets(
                row["parquet_path"], future_times, self.target_fields, 1,
                max_time_delta=self.agibot_max_time_delta,
            )
            current_target, current_valid = build_agibot_tubelet_targets(
                row["parquet_path"], np.asarray([sample_times[-1]]), self.target_fields, 1,
                max_time_delta=self.agibot_max_time_delta,
            )
            future_frame_indices = agibot_times_to_frame_indices(
                future_times,
                video_fps=float(row.get("raw_fps", 30.0)),
                num_frames=int(row.get("length", 0)),
            )
            target_frame_indices = torch.from_numpy(future_frame_indices.copy()).reshape(-1, 1)
        else:
            target, valid_mask = build_agibot_tubelet_targets(
                row["parquet_path"], sample_times, self.target_fields, self.tubelet_size,
                max_time_delta=self.agibot_max_time_delta,
            )
            target_frame_indices = torch.from_numpy(frame_indices.copy()).reshape(-1, self.tubelet_size)
        if self.stats is not None:
            mean, std = self.stats
            target = (target - mean) / std

        sample = {
            "video": video,
            "target": target,
            "valid": valid_mask,
            "frame_indices": torch.from_numpy(frame_indices.copy()),
            "target_frame_indices": target_frame_indices,
            "sample_times": torch.from_numpy(sample_times.astype(np.float32)),
            "video_id": row["video_id"],
        }
        if self.target_mode == "future_state":
            sample.update(
                {
                    "current_target": current_target,
                    "current_valid": current_valid,
                    "future_horizon_seconds": torch.tensor(self.future_horizon_seconds),
                    "future_horizons_seconds": torch.tensor(
                        self.future_horizons_seconds,
                        dtype=torch.float32,
                    ),
                }
            )
        return sample


class KITTIProbeDataset(Dataset):
    def __init__(
        self,
        rows,
        target_fields,
        num_frames,
        transform,
        random_clip=False,
        stats=None,
        frame_span=None,
        frame_stride=0,
        tubelet_size=2,
        video_num_threads=1,
        target_mode="state",
        trajectory_point_indices=(-1,),
    ):
        self.rows = rows
        self.target_fields = tuple(target_fields)
        self.num_frames = num_frames
        self.transform = transform
        self.random_clip = random_clip
        self.stats = stats
        self.frame_span = frame_span
        self.frame_stride = frame_stride
        self.tubelet_size = tubelet_size
        self.video_num_threads = video_num_threads
        if self.tubelet_size <= 0:
            raise ValueError(f"tubelet_size must be positive, got {self.tubelet_size}")
        if self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                f"num_frames must be divisible by tubelet_size for tubelet targets, "
                f"got {self.num_frames=} {self.tubelet_size=}"
            )

    def __len__(self):
        return len(self.rows)

    def effective_frame_stride(self, num_video_frames):
        return compute_effective_frame_stride(num_video_frames, self.num_frames, self.frame_span, self.frame_stride)

    def _indices_from_row(self, row):
        if "input_frame_indices" in row:
            indices = np.array(row["input_frame_indices"], dtype=np.int64)
            if len(indices) != self.num_frames:
                raise RuntimeError(
                    f"KITTI row {row.get('clip_id')} has {len(indices)} frame indices, "
                    f"expected {self.num_frames}"
                )
            return indices
        num_frames = int(row.get("num_raw_frames", 0))
        if num_frames <= 0:
            image_dir = Path(row["image_dir"])
            num_frames = sum(1 for _ in image_dir.glob("*.png"))
        return sample_frame_indices(
            num_frames,
            self.num_frames,
            frame_span=self.frame_span,
            frame_stride=self.frame_stride,
            random_clip=self.random_clip,
        )

    def __getitem__(self, index):
        row = self.rows[index]
        indices = self._indices_from_row(row)
        image_dir = Path(row["image_dir"])
        clip = []
        for frame_idx in indices:
            path = kitti_frame_path(image_dir, frame_idx)
            if not path.exists():
                raise FileNotFoundError(f"Missing KITTI image frame: {path}")
            with Image.open(path) as image:
                clip.append(image.convert("RGB").copy())
        video = self.transform(clip)

        frame_ids, values, valid = load_kitti_oxts_series(row["oxts_dir"], self.target_fields)
        state_indices = map_video_frames_to_state_indices(indices, frame_ids, int(row.get("num_raw_frames", len(frame_ids))))
        target, valid_mask = build_tubelet_targets(values, valid, state_indices, self.tubelet_size)

        if self.stats is not None:
            mean, std = self.stats
            target = (target - mean) / std

        target_frame_indices = torch.from_numpy(indices.copy()).reshape(-1, self.tubelet_size)
        return {
            "video": video,
            "target": target,
            "valid": valid_mask,
            "frame_indices": torch.from_numpy(indices.copy()),
            "target_frame_indices": target_frame_indices,
            "video_id": row.get("clip_id", row.get("sequence_id", str(index))),
        }

from src.time_jepa.egoexo_pose_targets import build_egoexo_tubelet_targets
from src.time_jepa.egoexo_time_dataset import read_egoexo_rows
from src.time_jepa.egoexo_views import select_egoexo_video_candidate
from src.time_jepa.time_sampling import build_sample_times, infer_video_fps, times_to_frame_indices


def load_egoexo_split_rows(
    root,
    split_dir,
    split_names=("probe_train", "probe_val", "test"),
    prefix="egoexo_bestexo",
    video_view="best_exo",
    frames_per_clip=64,
    fps=5.0,
    slice_stride_seconds=12.8,
    manifest_path=None,
):
    outputs = []
    for split_name in split_names:
        path = Path(split_dir) / f"{prefix}_{split_name}.json"
        with path.open("r") as handle:
            keys = json.load(handle)
        outputs.append(read_egoexo_rows(
            root=root,
            video_view=video_view,
            frames_per_clip=frames_per_clip,
            fps=fps,
            slice_stride_seconds=slice_stride_seconds,
            episode_allowlist=keys,
            manifest_path=manifest_path,
            split_prefix=prefix,
        ))
    return outputs


class EgoExoProbeDataset(Dataset):
    def __init__(
        self,
        rows,
        target_fields,
        num_frames,
        transform,
        random_clip=False,
        stats=None,
        frame_span=None,
        frame_stride=0,
        tubelet_size=2,
        video_num_threads=1,
        target_mode="state",
        trajectory_point_indices=(-1,),
        egoexo_fps=5.0,
        egoexo_max_time_delta=0.05,
        egoexo_pose_fps=30.0,
        egoexo_min_views=2,
        egoexo_smooth_window=5,
        egoexo_video_backend="auto",
        egoexo_ffmpeg_timeout=120.0,
        future_horizon_seconds=1.0,
        future_horizons_seconds=None,
    ):
        del frame_span, frame_stride, trajectory_point_indices
        self.rows = rows
        self.target_fields = tuple(target_fields)
        self.num_frames = int(num_frames)
        self.transform = transform
        self.random_view = bool(random_clip)
        self.stats = stats
        self.tubelet_size = int(tubelet_size)
        self.video_num_threads = int(video_num_threads)
        self.egoexo_fps = float(egoexo_fps)
        self.egoexo_max_time_delta = egoexo_max_time_delta
        self.egoexo_pose_fps = float(egoexo_pose_fps)
        self.egoexo_min_views = int(egoexo_min_views)
        self.egoexo_smooth_window = int(egoexo_smooth_window)
        self.egoexo_video_backend = str(egoexo_video_backend or "auto").lower()
        self.egoexo_ffmpeg_timeout = egoexo_ffmpeg_timeout
        self.target_mode = str(target_mode)
        self.future_horizons_seconds = normalize_future_horizons_seconds(
            future_horizons_seconds,
            future_horizon_seconds,
        )
        self.future_horizon_seconds = max(self.future_horizons_seconds)
        if self.target_mode not in ("state", "future_state"):
            raise ValueError(f"Unsupported EgoExo target_mode: {self.target_mode}")
        if self.egoexo_video_backend not in ("auto", "decord", "ffmpeg"):
            raise ValueError("egoexo_video_backend must be one of: auto, decord, ffmpeg")

    def __len__(self):
        return len(self.rows)

    def effective_frame_stride(self, num_video_frames):
        del num_video_frames
        return max(1, int(round(30.0 / self.egoexo_fps)))

    def _read_clip(self, row, sample_times):
        if self.egoexo_video_backend in ("auto", "decord"):
            try:
                reader = VideoReader(row["video_path"], num_threads=self.video_num_threads, ctx=cpu(0))
                if len(reader) <= 0:
                    raise RuntimeError(f"Empty video: {row['video_path']}")
                video_fps = infer_video_fps(reader, fallback=30.0)
                frame_indices = times_to_frame_indices(sample_times, video_fps, len(reader))
                video = reader.get_batch(frame_indices.tolist()).asnumpy()
                return video, frame_indices
            except Exception:
                if self.egoexo_video_backend == "decord":
                    raise
        video_fps, num_video_frames, height, width = probe_video_metadata(row["video_path"])
        frame_indices = times_to_frame_indices(sample_times, video_fps, num_video_frames)
        first_frame_idx = int(frame_indices[0])
        video = read_ffmpeg_frames(
            row["video_path"],
            frame_indices - first_frame_idx,
            height=height,
            width=width,
            timeout=self.egoexo_ffmpeg_timeout,
            seek_time=float(first_frame_idx) / video_fps,
        )
        return video, frame_indices

    def __getitem__(self, index):
        row = select_egoexo_video_candidate(self.rows[index], random_view=self.random_view)
        sample_times = build_sample_times(row["start_time"], self.num_frames, self.egoexo_fps)
        video, frame_indices = self._read_clip(row, sample_times)
        video = self.transform([frame for frame in video])
        if self.target_mode == "future_state":
            future_times = float(sample_times[-1]) + np.asarray(
                self.future_horizons_seconds,
                dtype=np.float64,
            )
            target, valid = build_egoexo_tubelet_targets(
                row["pose_path"], future_times, self.target_fields, 1,
                self.egoexo_max_time_delta, self.egoexo_pose_fps,
                self.egoexo_min_views, self.egoexo_smooth_window,
            )
            current_target, current_valid = build_egoexo_tubelet_targets(
                row["pose_path"], np.asarray([sample_times[-1]]), self.target_fields, 1,
                self.egoexo_max_time_delta, self.egoexo_pose_fps,
                self.egoexo_min_views, self.egoexo_smooth_window,
            )
            target_frame_indices = torch.tensor(
                np.rint(future_times * self.egoexo_pose_fps).astype(np.int64),
                dtype=torch.long,
            ).reshape(-1, 1)
        else:
            target, valid = build_egoexo_tubelet_targets(
                row["pose_path"], sample_times, self.target_fields, self.tubelet_size,
                self.egoexo_max_time_delta, self.egoexo_pose_fps,
                self.egoexo_min_views, self.egoexo_smooth_window,
            )
            target_frame_indices = torch.from_numpy(frame_indices.copy()).reshape(-1, self.tubelet_size)
        if self.stats is not None:
            mean, std = self.stats
            target = (target - mean) / std
        sample = {
            "video": video,
            "target": target,
            "valid": valid,
            "frame_indices": torch.from_numpy(frame_indices.copy()),
            "target_frame_indices": target_frame_indices,
            "sample_times": torch.from_numpy(sample_times.astype(np.float32)),
            "video_id": f"{row['window_key']}/{row['video_view']}",
            "video_view": row["video_view"],
            "physical_video_view": row.get("physical_video_view", row["video_view"]),
        }
        if self.target_mode == "future_state":
            sample.update(
                {
                    "current_target": current_target,
                    "current_valid": current_valid,
                    "future_horizon_seconds": torch.tensor(self.future_horizon_seconds),
                    "future_horizons_seconds": torch.tensor(
                        self.future_horizons_seconds,
                        dtype=torch.float32,
                    ),
                }
            )
        return sample
