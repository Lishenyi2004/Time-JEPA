import json
import os
import pathlib
import warnings
from logging import getLogger

import numpy as np
import torch
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.time_jepa.agibot_time_dataset import _read_ffmpeg_frames
from src.time_jepa.egoexo_pose_targets import build_egoexo_tubelet_targets, normalize_egoexo_targets
from src.time_jepa.time_sampling import build_sample_times, infer_video_fps, times_to_frame_indices
from src.time_jepa.egoexo_views import (
    normalize_egoexo_video_views,
    resolve_egoexo_video_candidate,
    select_egoexo_video_candidate,
)


logger = getLogger()


EGOEXO_TARGET_TEMPORAL_MODES = (
    "tubelet_aligned",
    "clip_mean_broadcast",
)


def normalize_egoexo_target_temporal_mode(mode):
    if mode is None:
        return "tubelet_aligned"
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "aligned": "tubelet_aligned",
        "tubelet": "tubelet_aligned",
        "tubelet_aligned": "tubelet_aligned",
        "clip_level": "clip_mean_broadcast",
        "clip_mean": "clip_mean_broadcast",
        "clip_mean_broadcast": "clip_mean_broadcast",
    }
    if normalized not in aliases:
        supported = ", ".join(EGOEXO_TARGET_TEMPORAL_MODES)
        raise ValueError(
            f"Unsupported EgoExo target temporal mode {mode!r}; "
            f"expected one of: {supported}"
        )
    return aliases[normalized]


def broadcast_egoexo_clip_mean_targets(target, valid):
    """Broadcast each dimension's raw valid-tubelet mean to its valid positions."""
    if not isinstance(target, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise TypeError("target and valid must both be torch.Tensor instances")
    if target.ndim != 2 or valid.ndim != 2:
        raise ValueError(
            "EgoExo target and valid tensors must both have shape [time, dimensions]"
        )
    if target.shape != valid.shape:
        raise ValueError(
            f"EgoExo target/valid shape mismatch: {tuple(target.shape)} vs "
            f"{tuple(valid.shape)}"
        )

    valid_mask = valid > 0
    if valid_mask.any() and not torch.isfinite(target[valid_mask]).all():
        raise ValueError("EgoExo target contains a non-finite value at a valid position")

    safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
    valid_counts = valid_mask.sum(dim=0)
    clip_mean = safe_target.sum(dim=0) / valid_counts.clamp_min(1).to(target.dtype)
    broadcast = clip_mean.unsqueeze(0).expand_as(target)
    return torch.where(valid_mask, broadcast, torch.zeros_like(target))


def _load_manifest(root, manifest_path=None, split_prefix="egoexo_bestexo"):
    if manifest_path is None:
        manifest_path = pathlib.Path("time-jepa/splits") / f"{split_prefix}_split_manifest.json"
    manifest_path = pathlib.Path(manifest_path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = pathlib.Path.cwd() / manifest_path
    with manifest_path.open("r") as handle:
        manifest = json.load(handle)
    expected_root = pathlib.Path(manifest.get("root", root)).expanduser().resolve()
    if expected_root != pathlib.Path(root).expanduser().resolve():
        raise RuntimeError(f"EgoExo manifest root mismatch: {expected_root} vs {root}")
    return manifest, manifest_path.resolve()


def egoexo_window_key(take_uid, window_idx):
    return f"{take_uid}/window_{int(window_idx):06d}"


def parse_egoexo_window_key(key):
    take_uid, window_text = str(key).rsplit("/window_", 1)
    return take_uid, int(window_text)


def read_egoexo_rows(
    root,
    video_view="best_exo",
    frames_per_clip=64,
    fps=5.0,
    slice_stride_seconds=12.8,
    episode_allowlist=None,
    manifest_path=None,
    split_prefix="egoexo_bestexo",
):
    root = pathlib.Path(root).expanduser().resolve()
    requested_views = normalize_egoexo_video_views(video_view)
    manifest, resolved_manifest_path = _load_manifest(root, manifest_path, split_prefix)
    take_index = manifest.get("take_index", {})
    keys = list(episode_allowlist or [])
    if not keys:
        keys = [
            egoexo_window_key(uid, window_idx)
            for uid, take in take_index.items()
            for window_idx in range(int(take.get("num_windows", 0)))
        ]
    rows = []
    missing = []
    for key in keys:
        try:
            take_uid, window_idx = parse_egoexo_window_key(key)
        except Exception:
            missing.append(str(key))
            continue
        take = take_index.get(take_uid)
        if take is None:
            missing.append(str(key))
            continue
        video_candidates = [
            resolve_egoexo_video_candidate(root, take, requested_view)
            for requested_view in requested_views
        ]
        missing_views = [
            candidate["video_role"]
            for candidate in video_candidates
            if not candidate.get("video_path") or not os.path.isfile(candidate["video_path"])
        ]
        if missing_views:
            missing.append(f"{key}:{','.join(str(view) for view in missing_views)}")
            continue
        first_candidate = video_candidates[0]
        rows.append({
            **take,
            "window_idx": window_idx,
            "window_key": str(key),
            "start_time": float(window_idx) * float(slice_stride_seconds),
            "duration": float(take["duration_sec"]),
            **first_candidate,
            "video_candidates": video_candidates,
            "requested_video_views": list(requested_views),
            "manifest_path": str(resolved_manifest_path),
            "frames_per_clip": int(frames_per_clip),
            "fps": float(fps),
            "slice_stride_seconds": float(slice_stride_seconds),
        })
    if missing:
        raise RuntimeError(f"{len(missing)} EgoExo split keys/views are missing; first={missing[:5]}")
    if not rows:
        raise RuntimeError("No EgoExo rows found")
    return rows


class EgoExoTimeJEPADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        video_view="best_exo",
        frames_per_clip=64,
        fps=5.0,
        target_fields=None,
        tubelet_size=2,
        transform=None,
        shared_transform=None,
        random_clip_sampling=False,
        slice_stride_seconds=12.8,
        max_time_delta=0.05,
        pose_fps=30.0,
        min_views=2,
        smooth_window=5,
        video_backend="auto",
        ffmpeg_timeout=120,
        return_physical_targets=True,
        episode_allowlist=None,
        manifest_path=None,
        split_prefix="egoexo_bestexo",
        target_temporal_mode="tubelet_aligned",
    ):
        self.root = pathlib.Path(root).expanduser().resolve()
        self.video_view = video_view
        self.frames_per_clip = int(frames_per_clip)
        self.fps = float(fps)
        self.target_fields = normalize_egoexo_targets(target_fields)
        self.tubelet_size = int(tubelet_size)
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = bool(random_clip_sampling)
        self.slice_stride_seconds = float(slice_stride_seconds)
        self.max_time_delta = max_time_delta
        self.pose_fps = float(pose_fps)
        self.min_views = int(min_views)
        self.smooth_window = int(smooth_window)
        self.video_backend = str(video_backend).lower()
        self.ffmpeg_timeout = ffmpeg_timeout
        self.return_physical_targets = bool(return_physical_targets)
        self.target_temporal_mode = normalize_egoexo_target_temporal_mode(
            target_temporal_mode
        )
        self.rows = read_egoexo_rows(
            root=self.root,
            video_view=video_view,
            frames_per_clip=frames_per_clip,
            fps=fps,
            slice_stride_seconds=slice_stride_seconds,
            episode_allowlist=episode_allowlist,
            manifest_path=manifest_path,
            split_prefix=split_prefix,
        )
        self.samples = [row["video_path"] for row in self.rows]
        self.labels = [0] * len(self.rows)
        self.sample_weights = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        for _ in range(10):
            sample = self.get_item_video(index)
            if sample is not None:
                return sample
            index = np.random.randint(len(self.rows))
        raise RuntimeError("Failed to load EgoExo sample after 10 attempts")

    def _read_video(self, row, sample_times):
        video_path = row["video_path"]
        if self.video_backend in ("auto", "decord"):
            try:
                reader = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
                video_fps = infer_video_fps(reader, fallback=30.0)
                indices = times_to_frame_indices(sample_times, video_fps, len(reader))
                return reader.get_batch(indices.tolist()).asnumpy(), indices, "decord"
            except Exception:
                if self.video_backend == "decord":
                    raise
        reader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        video_fps = infer_video_fps(reader, fallback=30.0)
        indices = times_to_frame_indices(sample_times, video_fps, len(reader))
        height, width = reader[0].shape[:2]
        first = int(indices[0])
        frames = _read_ffmpeg_frames(
            video_path,
            indices - first,
            height,
            width,
            timeout=self.ffmpeg_timeout,
            seek_time=first / video_fps,
        )
        return frames, indices, "ffmpeg"

    def get_item_video(self, index):
        row = select_egoexo_video_candidate(self.rows[index], random_view=True)
        try:
            sample_times = build_sample_times(row["start_time"], self.frames_per_clip, self.fps)
            buffer, frame_indices, backend = self._read_video(row, sample_times)
            if self.return_physical_targets:
                target, valid = build_egoexo_tubelet_targets(
                    row["pose_path"],
                    sample_times,
                    target_fields=self.target_fields,
                    tubelet_size=self.tubelet_size,
                    max_time_delta=self.max_time_delta,
                    pose_fps=self.pose_fps,
                    min_views=self.min_views,
                    smooth_window=self.smooth_window,
                )
                if self.target_temporal_mode == "clip_mean_broadcast":
                    target = broadcast_egoexo_clip_mean_targets(target, valid)
        except Exception as exc:
            warnings.warn(f"Failed to build EgoExo sample {row['window_key']}: {exc}")
            return None
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = [self.transform(buffer)] if self.transform is not None else [buffer]
        if not self.return_physical_targets:
            return buffer, 0, [frame_indices]
        physical = {
            "domain": "egoexo",
            "target_temporal_mode": self.target_temporal_mode,
            "target": [target],
            "valid": [valid],
            "video_id": f"{row['window_key']}/{row['video_view']}",
            "take_uid": row["take_uid"],
            "task_id": row.get("task_id", ""),
            "video_key": row["video_view"],
            "physical_video_view": row.get("physical_video_view", row["video_view"]),
            "video_backend": backend,
            "start_time": float(row["start_time"]),
            "sample_times": [torch.from_numpy(sample_times.astype(np.float32))],
        }
        return buffer, 0, [frame_indices], physical


def make_egoexo_time_jepa_dataset(
    data_paths,
    batch_size,
    frames_per_clip=64,
    dataset_fpcs=None,
    fps=5.0,
    num_clips=1,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    egoexo_video_view="best_exo",
    egoexo_target_fields=None,
    physical_tubelet_size=2,
    egoexo_slice_stride_seconds=12.8,
    egoexo_max_time_delta=0.05,
    egoexo_pose_fps=30.0,
    egoexo_min_views=2,
    egoexo_smooth_window=5,
    egoexo_video_backend="auto",
    egoexo_ffmpeg_timeout=120,
    return_physical_targets=True,
    egoexo_episode_allowlist=None,
    egoexo_manifest_path=None,
    egoexo_split_prefix="egoexo_bestexo",
    egoexo_target_temporal_mode="tubelet_aligned",
    **_,
):
    if num_clips != 1:
        raise NotImplementedError("EgoExoTimeJEPADataset supports num_clips=1")
    if dataset_fpcs is not None:
        frames_per_clip = int(dataset_fpcs[0])
    root = data_paths[0] if isinstance(data_paths, (list, tuple)) else data_paths
    dataset = EgoExoTimeJEPADataset(
        root=root,
        video_view=egoexo_video_view,
        frames_per_clip=frames_per_clip,
        fps=fps,
        target_fields=egoexo_target_fields,
        tubelet_size=physical_tubelet_size,
        transform=transform,
        shared_transform=shared_transform,
        slice_stride_seconds=egoexo_slice_stride_seconds,
        max_time_delta=egoexo_max_time_delta,
        pose_fps=egoexo_pose_fps,
        min_views=egoexo_min_views,
        smooth_window=egoexo_smooth_window,
        video_backend=egoexo_video_backend,
        ffmpeg_timeout=egoexo_ffmpeg_timeout,
        return_physical_targets=return_physical_targets,
        episode_allowlist=egoexo_episode_allowlist,
        manifest_path=egoexo_manifest_path,
        split_prefix=egoexo_split_prefix,
        target_temporal_mode=egoexo_target_temporal_mode,
    )
    if log_dir:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        dataset = MonitoredDataset(dataset, str(log_dir / f"resource_file_{rank}_%w.csv"), 10.0, 5.0)
    if datasets_weights is not None:
        sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader_cls = torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    loader = loader_cls(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info(
        "EgoExoTimeJEPADataset created rows=%s targets=%s target_temporal_mode=%s",
        len(dataset),
        egoexo_target_fields,
        egoexo_target_temporal_mode,
    )
    return dataset, loader, sampler


def compute_egoexo_stats(
    rows,
    target_fields,
    frames_per_clip,
    tubelet_size,
    fps,
    seed=0,
    max_videos=500,
    max_time_delta=0.05,
    pose_fps=30.0,
    min_views=2,
    smooth_window=5,
):
    rows = list(rows)
    rng = np.random.RandomState(seed)
    if max_videos > 0 and len(rows) > max_videos:
        rows = [rows[index] for index in rng.choice(len(rows), max_videos, replace=False)]
    fields = normalize_egoexo_targets(target_fields)
    sums = torch.zeros(len(fields), dtype=torch.float64)
    sq_sums = torch.zeros(len(fields), dtype=torch.float64)
    counts = torch.zeros(len(fields), dtype=torch.float64)
    for row in rows:
        sample_times = build_sample_times(row["start_time"], frames_per_clip, fps)
        target, valid = build_egoexo_tubelet_targets(
            row["pose_path"], sample_times, fields, tubelet_size, max_time_delta,
            pose_fps, min_views, smooth_window,
        )
        target = target.double()
        valid = valid.double()
        sums += (target * valid).sum(0)
        sq_sums += ((target ** 2) * valid).sum(0)
        counts += valid.sum(0)
    if (counts < 2).any():
        missing = [fields[i] for i, count in enumerate(counts.tolist()) if count < 2]
        raise RuntimeError(f"Could not compute EgoExo stats for {missing}")
    mean = sums / counts
    variance = (sq_sums - sums * sums / counts) / (counts - 1.0).clamp_min(1.0)
    return mean.float(), torch.sqrt(variance.clamp_min(1e-12)).clamp_min(1e-6).float(), counts.long()
