"""AgiBot RGB/action/state adapter for the official four-frame PSG-JEPA model."""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as vision_functional

from src.time_jepa.agibot_derived_targets import (
    WHOLEBODY_CARTESIAN_V2_FIELDS,
    derive_agibot_series_field,
    is_agibot_series_derived_field,
)

from .protocol import (
    DEFAULT_ACTION_DIM,
    DEFAULT_MICROCLIP_FRAMES,
    AgiBotPSGProtocol,
    AgiBotPSGRecord,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DIRECT_STATE_FIELDS = {
    **{f"state/end/arm_position[{index}]": 34 + index for index in range(6)},
    "state/left_effector/position": 0,
    "state/right_effector/position": 1,
}


@dataclass(frozen=True)
class VectorStats:
    mean: torch.Tensor
    std: torch.Tensor
    fields: tuple[str, ...] | None = None
    meta: dict[str, object] | None = None


@dataclass(frozen=True)
class EpisodeArrays:
    timestamps: np.ndarray
    frame_indices: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    target_valid: np.ndarray


def load_vector_stats(
    path: str | Path,
    *,
    expected_dim: int,
    expected_fields: Sequence[str] | None = None,
) -> VectorStats:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {path}")
    with path.open("r") as handle:
        payload = json.load(handle)
    mean = torch.as_tensor(payload.get("mean"), dtype=torch.float32)
    std = torch.as_tensor(payload.get("std"), dtype=torch.float32)
    if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
        raise ValueError(
            f"Stats {path} have shapes mean={tuple(mean.shape)}, std={tuple(std.shape)}; "
            f"expected ({expected_dim},)"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"Stats contain non-finite values: {path}")
    if bool((std <= 0).any()):
        raise ValueError(f"Stats contain non-positive standard deviations: {path}")
    meta = payload.get("meta") or {}
    fields_value = meta.get("target_fields")
    fields = tuple(str(field) for field in fields_value) if fields_value else None
    if expected_fields is not None and fields != tuple(expected_fields):
        raise ValueError(
            f"Stats target fields do not match the PSG state target panel: {path}"
        )
    return VectorStats(
        mean=mean,
        std=std.clamp_min(1.0e-6),
        fields=fields,
        meta=dict(meta),
    )


def derive_target_matrix(
    timestamps: np.ndarray,
    states: np.ndarray,
    target_fields: Sequence[str] = WHOLEBODY_CARTESIAN_V2_FIELDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact existing 18-D AgiBot panel at the original frame rate."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or len(states) != len(timestamps):
        raise ValueError(
            f"Expected aligned timestamps [N] and states [N,D], got "
            f"{timestamps.shape} and {states.shape}"
        )
    values = np.zeros((len(states), len(target_fields)), dtype=np.float32)
    valid = np.zeros_like(values)
    for column, field in enumerate(target_fields):
        if is_agibot_series_derived_field(field):
            field_values, field_valid = derive_agibot_series_field(
                timestamps,
                states,
                field,
            )
        elif field in _DIRECT_STATE_FIELDS:
            state_column = _DIRECT_STATE_FIELDS[field]
            if state_column >= states.shape[1]:
                raise ValueError(
                    f"State field {field!r} needs column {state_column}, "
                    f"but state dimension is {states.shape[1]}"
                )
            field_values = states[:, state_column]
            field_valid = np.isfinite(field_values).astype(np.float32)
            field_values = np.where(field_valid > 0, field_values, 0.0)
        else:
            raise ValueError(f"Unsupported PSG static target field: {field!r}")
        values[:, column] = np.asarray(field_values, dtype=np.float32)
        valid[:, column] = np.asarray(field_valid, dtype=np.float32)
    return values, valid


@lru_cache(maxsize=64)
def load_episode_arrays(
    parquet_path: str,
    target_fields: tuple[str, ...] = tuple(WHOLEBODY_CARTESIAN_V2_FIELDS),
) -> EpisodeArrays:
    dataframe = pd.read_parquet(
        parquet_path,
        columns=["timestamp", "frame_index", "action", "observation.state"],
    )
    if dataframe.empty:
        raise RuntimeError(f"Empty AgiBot parquet: {parquet_path}")
    timestamps = dataframe["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = dataframe["frame_index"].to_numpy(dtype=np.int64)
    actions = np.stack(
        [np.asarray(value, dtype=np.float32) for value in dataframe["action"]],
        axis=0,
    )
    states = np.stack(
        [np.asarray(value, dtype=np.float32) for value in dataframe["observation.state"]],
        axis=0,
    )
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    frame_indices = frame_indices[order]
    actions = actions[order]
    states = states[order]
    if actions.ndim != 2 or actions.shape[1] != DEFAULT_ACTION_DIM:
        raise ValueError(
            f"Expected AgiBot actions [N,{DEFAULT_ACTION_DIM}], got {actions.shape} "
            f"in {parquet_path}"
        )
    if not np.array_equal(np.diff(frame_indices), np.ones(len(frame_indices) - 1, dtype=np.int64)):
        raise ValueError(f"Non-contiguous AgiBot frame_index in {parquet_path}")
    targets, target_valid = derive_target_matrix(timestamps, states, target_fields)
    return EpisodeArrays(
        timestamps=timestamps,
        frame_indices=frame_indices,
        actions=actions,
        targets=targets,
        target_valid=target_valid,
    )


def nearest_indices(
    query_times: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_delta: float,
) -> np.ndarray:
    query_times = np.asarray(query_times, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps):
        raise ValueError("timestamps must be a non-empty one-dimensional array")
    insertion = np.searchsorted(timestamps, query_times, side="left")
    upper = np.clip(insertion, 0, len(timestamps) - 1)
    lower = np.clip(insertion - 1, 0, len(timestamps) - 1)
    choose_upper = np.abs(timestamps[upper] - query_times) < np.abs(
        timestamps[lower] - query_times
    )
    indices = np.where(choose_upper, upper, lower).astype(np.int64)
    deltas = np.abs(timestamps[indices] - query_times)
    if bool((deltas > float(max_delta)).any()):
        worst = int(np.argmax(deltas))
        raise ValueError(
            "AgiBot timestamp alignment exceeded tolerance: "
            f"query={query_times[worst]:.6f}, matched={timestamps[indices[worst]]:.6f}, "
            f"delta={deltas[worst]:.6f}, max_delta={max_delta}"
        )
    return indices


def pack_action_blocks(
    raw_actions: np.ndarray,
    observation_rows: np.ndarray,
    *,
    microclip_offsets: Sequence[int],
    actions_per_step: int,
    action_stats: VectorStats | None = None,
) -> torch.Tensor:
    """Pack six dense 30 Hz commands for each 5 Hz model transition.

    The fourth action in every four-frame microclip is zero padding.  The
    official predictor consumes only the first three action embeddings.
    """

    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    observation_rows = np.asarray(observation_rows, dtype=np.int64)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != DEFAULT_ACTION_DIM:
        raise ValueError(f"Expected raw actions [N,{DEFAULT_ACTION_DIM}], got {raw_actions.shape}")
    if observation_rows.ndim != 1:
        raise ValueError("observation_rows must be one-dimensional")
    effective_dim = int(actions_per_step) * DEFAULT_ACTION_DIM
    packed = torch.zeros(
        len(microclip_offsets),
        DEFAULT_MICROCLIP_FRAMES,
        effective_dim,
        dtype=torch.float32,
    )
    mean = action_stats.mean.numpy() if action_stats is not None else None
    std = action_stats.std.numpy() if action_stats is not None else None
    for microclip_index, offset in enumerate(microclip_offsets):
        for step in range(DEFAULT_MICROCLIP_FRAMES - 1):
            start = int(observation_rows[offset + step])
            stop = int(observation_rows[offset + step + 1])
            block = raw_actions[start:stop]
            if block.shape != (actions_per_step, DEFAULT_ACTION_DIM):
                raise ValueError(
                    f"Action interval at microclip={microclip_index}, step={step} "
                    f"has shape {block.shape}; expected "
                    f"({actions_per_step}, {DEFAULT_ACTION_DIM})"
                )
            if mean is not None and std is not None:
                block = (block - mean[None, :]) / std[None, :]
            packed[microclip_index, step] = torch.from_numpy(
                np.ascontiguousarray(block.reshape(-1))
            )
    return packed


class PSGClipTransform:
    """Apply one spatial crop consistently to all 64 frames in a window."""

    def __init__(
        self,
        image_size: int = 224,
        *,
        training: bool,
        scale: tuple[float, float] = (0.3, 1.0),
        ratio: tuple[float, float] = (0.75, 1.35),
    ):
        self.image_size = int(image_size)
        self.training = bool(training)
        self.scale = tuple(float(value) for value in scale)
        self.ratio = tuple(float(value) for value in ratio)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB frames [T,H,W,3], got {frames.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        if self.training:
            top, left, height, width = RandomResizedCrop.get_params(
                tensor[0],
                scale=self.scale,
                ratio=self.ratio,
            )
            tensor = vision_functional.resized_crop(
                tensor,
                top,
                left,
                height,
                width,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            tensor = vision_functional.resize(
                tensor,
                self.image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = vision_functional.center_crop(
                tensor,
                [self.image_size, self.image_size],
            )
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        return (tensor - self.mean) / self.std


def _decode_decord(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), num_threads=-1, ctx=cpu(0))
    if int(frame_indices[-1]) >= len(reader):
        raise IndexError(
            f"Requested frame {int(frame_indices[-1])} from {video_path}, "
            f"which contains {len(reader)} frames"
        )
    return reader.get_batch(frame_indices.tolist()).asnumpy()


def _decode_ffmpeg(
    video_path: Path,
    frame_indices: np.ndarray,
    *,
    raw_fps: float,
    height: int,
    width: int,
    timeout: float,
) -> np.ndarray:
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    first_frame = int(frame_indices[0])
    relative = frame_indices - first_frame
    unique_indices, inverse = np.unique(relative, return_inverse=True)
    select_expression = "+".join(f"eq(n\\,{int(index)})" for index in unique_indices)
    command = [
        os.environ.get("FFMPEG_BINARY", "ffmpeg"),
        "-v",
        "error",
        "-ss",
        f"{first_frame / float(raw_fps):.9f}",
        "-i",
        str(video_path),
        "-vf",
        f"select={select_expression}",
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
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(timeout),
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}")
    frame_bytes = int(height) * int(width) * 3
    expected_bytes = len(unique_indices) * frame_bytes
    if len(process.stdout) != expected_bytes:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg returned {len(process.stdout)} bytes for {video_path}; "
            f"expected {expected_bytes}. stderr={stderr}"
        )
    decoded = np.frombuffer(process.stdout, dtype=np.uint8).reshape(
        len(unique_indices),
        int(height),
        int(width),
        3,
    )
    return decoded[inverse].copy()


def decode_video_frames(
    record: AgiBotPSGRecord,
    frame_indices: np.ndarray,
    *,
    backend: str,
    ffmpeg_timeout: float,
) -> np.ndarray:
    if not record.video_path.is_file():
        raise FileNotFoundError(f"AgiBot video not found: {record.video_path}")
    backend = str(backend).lower()
    if backend not in {"auto", "decord", "ffmpeg"}:
        raise ValueError(f"Unsupported video backend: {backend}")
    if backend in {"auto", "decord"}:
        try:
            return _decode_decord(record.video_path, frame_indices)
        except Exception as error:
            if backend == "decord":
                raise
            warnings.warn(
                f"Decord failed for {record.video_path}; falling back to ffmpeg: {error}"
            )
    return _decode_ffmpeg(
        record.video_path,
        frame_indices,
        raw_fps=record.raw_fps,
        height=record.height,
        width=record.width,
        timeout=ffmpeg_timeout,
    )


class AgiBotPSGWindowDataset(Dataset):
    """Return 16 official four-frame PSG sequences per frozen 64-frame window."""

    target_fields = tuple(WHOLEBODY_CARTESIAN_V2_FIELDS)

    def __init__(
        self,
        protocol: AgiBotPSGProtocol,
        *,
        action_stats_path: str | Path | None,
        state_stats_path: str | Path | None = None,
        image_size: int = 224,
        training: bool = True,
        video_backend: str = "ffmpeg",
        ffmpeg_timeout: float = 120.0,
        decoder: Callable[[AgiBotPSGRecord, np.ndarray], np.ndarray] | None = None,
    ):
        self.protocol = protocol
        self.records = tuple(protocol.iter_records())
        if len(self.records) != protocol.num_view_windows:
            raise RuntimeError(
                f"Resolved {len(self.records)} view windows, expected {protocol.num_view_windows}"
            )
        self.microclip_offsets = tuple(
            range(0, protocol.window_frames, DEFAULT_MICROCLIP_FRAMES)
        )
        self.action_stats = (
            load_vector_stats(action_stats_path, expected_dim=DEFAULT_ACTION_DIM)
            if action_stats_path is not None
            else None
        )
        if self.action_stats is not None:
            action_meta = self.action_stats.meta or {}
            expected_windows = len(protocol.sample_keys)
            if action_meta.get("kind") != "psg_raw_action_40d":
                raise ValueError(
                    f"Action stats have unexpected kind: {action_meta.get('kind')!r}"
                )
            if int(action_meta.get("num_physical_windows", -1)) != expected_windows:
                raise ValueError(
                    "Action stats do not cover the complete frozen allowlist: "
                    f"got {action_meta.get('num_physical_windows')}, "
                    f"expected {expected_windows}"
                )
            if int(action_meta.get("sample_seed", -1)) != protocol.sample_seed:
                raise ValueError(
                    "Action stats sample seed does not match the frozen protocol: "
                    f"got {action_meta.get('sample_seed')}, expected {protocol.sample_seed}"
                )
        resolved_state_stats = state_stats_path or protocol.state_stats_path
        self.state_stats = (
            load_vector_stats(
                resolved_state_stats,
                expected_dim=len(self.target_fields),
                expected_fields=self.target_fields,
            )
            if resolved_state_stats is not None
            else None
        )
        if self.state_stats is not None:
            state_meta = self.state_stats.meta or {}
            row_count = state_meta.get("row_count")
            if row_count is not None and int(row_count) != protocol.num_view_windows:
                raise ValueError(
                    "State stats row_count does not match the frozen view-window set: "
                    f"got {row_count}, expected {protocol.num_view_windows}"
                )
        self.transform = PSGClipTransform(
            image_size=image_size,
            training=training,
            scale=protocol.random_resize_scale,
            ratio=protocol.random_resize_aspect_ratio,
        )
        self.video_backend = str(video_backend)
        self.ffmpeg_timeout = float(ffmpeg_timeout)
        self.decoder = decoder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        arrays = load_episode_arrays(str(record.parquet_path), self.target_fields)
        sample_times = record.start_time + np.arange(
            self.protocol.window_frames,
            dtype=np.float64,
        ) / self.protocol.model_fps
        observation_rows = nearest_indices(
            sample_times,
            arrays.timestamps,
            max_delta=self.protocol.max_time_delta,
        )
        expected_stride = self.protocol.action_steps_per_model_step
        row_strides = np.diff(observation_rows)
        if not np.array_equal(
            row_strides,
            np.full_like(row_strides, expected_stride),
        ):
            raise ValueError(
                f"AgiBot 5 Hz rows are not separated by {expected_stride} raw rows for "
                f"{record.sample_key}: unique strides={np.unique(row_strides).tolist()}"
            )
        video_frame_indices = arrays.frame_indices[observation_rows]
        if self.decoder is None:
            frames = decode_video_frames(
                record,
                video_frame_indices,
                backend=self.video_backend,
                ffmpeg_timeout=self.ffmpeg_timeout,
            )
        else:
            frames = self.decoder(record, video_frame_indices)
        pixels = self.transform(frames).reshape(
            len(self.microclip_offsets),
            DEFAULT_MICROCLIP_FRAMES,
            3,
            self.transform.image_size,
            self.transform.image_size,
        )
        action = pack_action_blocks(
            arrays.actions,
            observation_rows,
            microclip_offsets=self.microclip_offsets,
            actions_per_step=expected_stride,
            action_stats=self.action_stats,
        )

        target = torch.from_numpy(arrays.targets[observation_rows]).float()
        target_valid = torch.from_numpy(arrays.target_valid[observation_rows]).float()
        if self.state_stats is not None:
            target = (target - self.state_stats.mean) / self.state_stats.std
        target = target.masked_fill(target_valid <= 0, float("nan"))
        observation = target.reshape(
            len(self.microclip_offsets),
            DEFAULT_MICROCLIP_FRAMES,
            len(self.target_fields),
        )
        observation_valid = target_valid.reshape_as(observation)

        return {
            "pixels": pixels,
            "action": action,
            "observation": observation,
            "observation_valid": observation_valid,
            "sample_key": record.sample_key,
            "video_key": record.video_key,
            "video_frame_indices": torch.from_numpy(video_frame_indices.copy()),
            "microclip_offsets": torch.tensor(self.microclip_offsets, dtype=torch.long),
        }


def collate_psg_windows(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Flatten base-window and microclip axes without changing optimizer steps."""

    if not samples:
        raise ValueError("Cannot collate an empty PSG batch")
    tensor_keys = ("pixels", "action", "observation", "observation_valid")
    batch: dict[str, object] = {}
    microclips = int(torch.as_tensor(samples[0]["pixels"]).shape[0])
    for key in tensor_keys:
        tensors = [torch.as_tensor(sample[key]) for sample in samples]
        if any(tensor.shape[0] != microclips for tensor in tensors):
            raise ValueError(f"Inconsistent microclip count while collating {key}")
        batch[key] = torch.cat(tensors, dim=0)
    batch["sample_key"] = tuple(
        str(sample["sample_key"])
        for sample in samples
        for _ in range(microclips)
    )
    batch["video_key"] = tuple(
        str(sample["video_key"])
        for sample in samples
        for _ in range(microclips)
    )
    batch["microclip_index"] = torch.arange(microclips, dtype=torch.long).repeat(
        len(samples)
    )
    return batch
