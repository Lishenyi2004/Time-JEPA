"""Combined COVLA state+trajectory dataset and seed-invariant stats cache."""

from __future__ import annotations

import fcntl
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from decord import VideoReader, cpu
from torch.utils.data import Dataset

try:
    from .bank_common import (
        STATE_FIELDS,
        TARGET_NAMES,
        TRAJECTORY_POINTS,
        atomic_json_dump,
        canonical_hash,
    )
except ImportError:
    from bank_common import (
        STATE_FIELDS,
        TARGET_NAMES,
        TRAJECTORY_POINTS,
        atomic_json_dump,
        canonical_hash,
    )
from probe_sampling import (
    build_tubelet_targets,
    compute_effective_frame_stride,
    map_video_frames_to_state_indices,
    sample_frame_indices,
)
from probe_targets import load_state_series, load_trajectory_xy_series


def load_combined_series(states_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_ids, state_values, state_valid = load_state_series(states_path, STATE_FIELDS)
    trajectory_ids, trajectory_values, trajectory_valid = load_trajectory_xy_series(
        states_path, point_indices_tuple=TRAJECTORY_POINTS
    )
    if np.array_equal(state_ids, trajectory_ids):
        common_ids = state_ids
        state_pos = np.arange(len(common_ids))
        trajectory_pos = state_pos
    else:
        common_ids, state_pos, trajectory_pos = np.intersect1d(
            state_ids, trajectory_ids, assume_unique=False, return_indices=True
        )
        if not len(common_ids):
            raise RuntimeError(f"State and trajectory frame grids do not overlap: {states_path}")
    state_values = state_values[state_pos]
    state_valid = state_valid[state_pos]
    trajectory_values = trajectory_values[trajectory_pos]
    trajectory_valid = trajectory_valid[trajectory_pos]
    values = np.concatenate(
        [state_values[:, :3], trajectory_values, state_values[:, 3:]], axis=1
    ).astype(np.float32, copy=False)
    valid = np.concatenate(
        [state_valid[:, :3], trajectory_valid, state_valid[:, 3:]], axis=1
    ).astype(np.float32, copy=False)
    if values.shape[1] != len(TARGET_NAMES):
        raise AssertionError(f"Combined target width mismatch: {values.shape}")
    return common_ids, values, valid


class CovlaPhysicsBankDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        num_frames: int,
        transform,
        random_clip: bool = False,
        stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        frame_span: int | None = None,
        frame_stride: int = 0,
        tubelet_size: int = 2,
        video_num_threads: int = 1,
    ):
        self.rows = list(rows)
        self.num_frames = int(num_frames)
        self.transform = transform
        self.random_clip = bool(random_clip)
        self.stats = stats
        self.frame_span = frame_span
        self.frame_stride = int(frame_stride)
        self.tubelet_size = int(tubelet_size)
        self.video_num_threads = int(video_num_threads)
        if self.num_frames % self.tubelet_size:
            raise ValueError("num_frames must be divisible by tubelet_size")

    def __len__(self) -> int:
        return len(self.rows)

    def effective_frame_stride(self, video_frames: int) -> int:
        return compute_effective_frame_stride(
            video_frames, self.num_frames, self.frame_span, self.frame_stride
        )

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        reader = VideoReader(row["video_path"], num_threads=self.video_num_threads, ctx=cpu(0))
        if len(reader) <= 0:
            raise RuntimeError(f"Empty video: {row['video_path']}")
        indices = sample_frame_indices(
            len(reader),
            self.num_frames,
            frame_span=self.frame_span,
            frame_stride=self.frame_stride,
            random_clip=self.random_clip,
        )
        frames = reader.get_batch(indices.tolist()).asnumpy()
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)
        video = self.transform(video)

        frame_ids, values, valid = load_combined_series(row["states_path"])
        state_indices = map_video_frames_to_state_indices(indices, frame_ids, len(reader))
        raw_target, valid_mask = build_tubelet_targets(
            values, valid, state_indices, self.tubelet_size
        )
        target = raw_target
        if self.stats is not None:
            mean, std = self.stats
            target = (raw_target - mean) / std
        return {
            "video": video,
            "target": target,
            "valid": valid_mask,
            "frame_indices": torch.from_numpy(indices.copy()),
            "target_frame_indices": torch.from_numpy(indices.copy()).reshape(
                -1, self.tubelet_size
            ),
            "video_id": row["video_id"],
        }


def stats_fingerprint(rows: list[dict], sampling: dict, seed: int, max_videos: int) -> dict:
    payload = {
        "schema": 1,
        "target_names": TARGET_NAMES,
        "video_ids": sorted(row["video_id"] for row in rows),
        "sampling": {
            "num_frames": int(sampling["num_frames"]),
            "frame_span": sampling.get("frame_span"),
            "frame_stride": int(sampling.get("frame_stride", 0)),
            "tubelet_size": int(sampling["tubelet_size"]),
        },
        "stats_seed": int(seed),
        "max_videos": int(max_videos),
    }
    return {"sha256": canonical_hash(payload), "payload": payload}


def compute_stats(
    rows: list[dict], sampling: dict, seed: int = 239, max_videos: int = 2000
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = list(rows)
    rng = random.Random(int(seed))
    if max_videos > 0 and len(selected) > max_videos:
        selected = rng.sample(selected, int(max_videos))
    width = len(TARGET_NAMES)
    sums = torch.zeros(width, dtype=torch.float64)
    sq_sums = torch.zeros(width, dtype=torch.float64)
    counts = torch.zeros(width, dtype=torch.float64)
    for row in selected:
        reader = VideoReader(row["video_path"], num_threads=1, ctx=cpu(0))
        if len(reader) <= 0:
            continue
        indices = sample_frame_indices(
            len(reader),
            int(sampling["num_frames"]),
            frame_span=sampling.get("frame_span"),
            frame_stride=int(sampling.get("frame_stride", 0)),
            random_clip=True,
            rng=rng,
        )
        frame_ids, values, valid = load_combined_series(row["states_path"])
        state_indices = map_video_frames_to_state_indices(indices, frame_ids, len(reader))
        target, mask = build_tubelet_targets(
            values, valid, state_indices, int(sampling["tubelet_size"])
        )
        target = target.double()
        mask = mask.double()
        sums += (target * mask).sum(0)
        sq_sums += (target.square() * mask).sum(0)
        counts += mask.sum(0)
    if (counts < 2).any():
        missing = [TARGET_NAMES[i] for i in torch.where(counts < 2)[0].tolist()]
        raise RuntimeError(f"Insufficient valid normalization samples for {missing}")
    mean = sums / counts
    variance = (sq_sums - sums.square() / counts) / (counts - 1).clamp_min(1)
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return mean.float(), std.float(), counts


def get_or_compute_stats(
    rows: list[dict],
    sampling: dict,
    cache_path: str | Path,
    seed: int = 239,
    max_videos: int = 2000,
    distributed: bool = False,
    rank: int = 0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    cache_path = Path(cache_path).expanduser().resolve()
    fingerprint = stats_fingerprint(rows, sampling, seed, max_videos)
    payload = None
    if not distributed or rank == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache_path.exists():
                with cache_path.open() as handle:
                    candidate = json.load(handle)
                if candidate.get("fingerprint", {}).get("sha256") == fingerprint["sha256"]:
                    payload = candidate
            if payload is None:
                mean, std, counts = compute_stats(rows, sampling, seed, max_videos)
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
        packed = torch.tensor([payload["mean"], payload["std"]], dtype=torch.float32)
    else:
        packed = torch.empty(2, len(TARGET_NAMES), dtype=torch.float32)
    if distributed:
        communication_device = device if device.type == "cuda" else torch.device("cpu")
        packed = packed.to(communication_device)
        dist.broadcast(packed, src=0)
        packed = packed.cpu()
        objects = [payload]
        dist.broadcast_object_list(objects, src=0)
        payload = objects[0]
    return packed[0].clone(), packed[1].clone(), payload
