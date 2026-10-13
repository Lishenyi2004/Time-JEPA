"""Target normalization statistics (per-rank compute + broadcast)."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from decord import VideoReader, cpu

from probe_datasets import agibot_sample_times, build_agibot_tubelet_targets
from probe_sampling import (
    build_tubelet_targets,
    map_video_frames_to_state_indices,
    normalize_future_horizons_seconds,
    sample_frame_indices,
)
from probe_targets import load_kitti_oxts_series, load_state_series, load_trajectory_xy_series


def compute_stats(
    rows,
    target_fields,
    num_frames,
    dataset_name="covla",
    frame_span=None,
    frame_stride=0,
    tubelet_size=2,
    target_mode="state",
    trajectory_point_indices=(-1,),
    agibot_fps=5.0,
    agibot_max_time_delta=0.05,
    egoexo_fps=5.0,
    egoexo_max_time_delta=0.05,
    egoexo_pose_fps=30.0,
    egoexo_min_views=2,
    egoexo_smooth_window=5,
    future_horizon_seconds=1.0,
    future_horizons_seconds=None,
    seed=0,
    max_videos=2000,
):
    rows = list(rows)
    rng = random.Random(seed)
    if max_videos > 0 and len(rows) > max_videos:
        rows = rng.sample(rows, max_videos)

    trajectory_point_indices = tuple(trajectory_point_indices)
    num_outputs = 2 * len(trajectory_point_indices) if target_mode == "trajectory_xy" else len(target_fields)
    future_horizons = normalize_future_horizons_seconds(
        future_horizons_seconds,
        future_horizon_seconds,
    )
    future_max_horizon = max(future_horizons)
    stats_shape = (
        (len(future_horizons), num_outputs)
        if target_mode == "future_state"
        else (num_outputs,)
    )
    sums = torch.zeros(stats_shape, dtype=torch.float64)
    sq_sums = torch.zeros(stats_shape, dtype=torch.float64)
    counts = torch.zeros(stats_shape, dtype=torch.float64)

    for row in rows:
        if dataset_name == "agibot":
            sample_times = agibot_sample_times(
                row,
                num_frames,
                agibot_fps,
                random_clip=True,
                rng=rng,
                future_horizon_seconds=(
                    future_max_horizon if target_mode == "future_state" else 0.0
                ),
            )
            target_times = (
                float(sample_times[-1]) + np.asarray(future_horizons, dtype=np.float64)
                if target_mode == "future_state"
                else sample_times
            )
            target, valid_mask = build_agibot_tubelet_targets(
                row["parquet_path"], target_times, target_fields,
                1 if target_mode == "future_state" else tubelet_size,
                max_time_delta=agibot_max_time_delta,
            )
        elif dataset_name == "egoexo":
            from src.time_jepa.egoexo_pose_targets import build_egoexo_tubelet_targets
            from src.time_jepa.time_sampling import build_sample_times
            sample_times = build_sample_times(row["start_time"], num_frames, egoexo_fps)
            target_times = (
                float(sample_times[-1]) + np.asarray(future_horizons, dtype=np.float64)
                if target_mode == "future_state"
                else sample_times
            )
            target, valid_mask = build_egoexo_tubelet_targets(
                row["pose_path"], target_times, target_fields,
                1 if target_mode == "future_state" else tubelet_size,
                egoexo_max_time_delta, egoexo_pose_fps, egoexo_min_views, egoexo_smooth_window,
            )
        elif dataset_name == "kitti":
            if "input_frame_indices" in row:
                indices = np.array(row["input_frame_indices"], dtype=np.int64)
            else:
                num_raw_frames = int(row.get("num_raw_frames", 0))
                if num_raw_frames <= 0:
                    num_raw_frames = sum(1 for _ in Path(row["image_dir"]).glob("*.png"))
                indices = sample_frame_indices(
                    num_raw_frames,
                    num_frames,
                    frame_span=frame_span,
                    frame_stride=frame_stride,
                    random_clip=True,
                    rng=rng,
                )
            frame_ids, series, valid = load_kitti_oxts_series(row["oxts_dir"], tuple(target_fields))
            state_indices = map_video_frames_to_state_indices(
                indices,
                frame_ids,
                int(row.get("num_raw_frames", len(frame_ids))),
            )
            target, valid_mask = build_tubelet_targets(series, valid, state_indices, tubelet_size)
        else:
            vr = VideoReader(row["video_path"], num_threads=1, ctx=cpu(0))
            if len(vr) <= 0:
                continue

            video_fps = float(vr.get_avg_fps())
            future_frame_offsets = (
                np.asarray(
                    [
                        max(1, int(round(float(horizon) * video_fps)))
                        for horizon in future_horizons
                    ],
                    dtype=np.int64,
                )
                if target_mode == "future_state"
                else np.asarray([], dtype=np.int64)
            )
            if target_mode == "future_state" and np.any(np.diff(future_frame_offsets) <= 0):
                raise RuntimeError(
                    "Configured future horizons collapse to duplicate video frames at "
                    f"fps={video_fps}: horizons={future_horizons}, "
                    f"offsets={future_frame_offsets.tolist()}"
                )
            indices = sample_frame_indices(
                len(vr),
                num_frames,
                frame_span=frame_span,
                frame_stride=frame_stride,
                random_clip=True,
                rng=rng,
                reserve_after=(
                    int(future_frame_offsets[-1])
                    if len(future_frame_offsets)
                    else 0
                ),
            )
            if target_mode == "trajectory_xy":
                frame_ids, series, valid = load_trajectory_xy_series(
                    row["states_path"],
                    point_indices_tuple=trajectory_point_indices,
                )
            elif target_mode in ("state", "future_state"):
                frame_ids, series, valid = load_state_series(row["states_path"], tuple(target_fields))
            else:
                raise ValueError(f"Unsupported target_mode for CoVLA stats: {target_mode}")
            if target_mode == "future_state":
                state_indices = map_video_frames_to_state_indices(
                    int(indices[-1]) + future_frame_offsets,
                    frame_ids,
                    len(vr),
                )
                target, valid_mask = build_tubelet_targets(series, valid, state_indices, 1)
            else:
                state_indices = map_video_frames_to_state_indices(indices, frame_ids, len(vr))
                target, valid_mask = build_tubelet_targets(series, valid, state_indices, tubelet_size)
        target = target.double()
        valid_mask = valid_mask.double()
        if target_mode == "future_state":
            sums += target * valid_mask
            sq_sums += (target**2) * valid_mask
            counts += valid_mask
        else:
            sums += (target * valid_mask).sum(dim=0)
            sq_sums += ((target**2) * valid_mask).sum(dim=0)
            counts += valid_mask.sum(dim=0)

    if (counts < 2).any():
        names = (
            [
                f"trajectory_{point_index}_{axis}"
                for point_index in trajectory_point_indices
                for axis in ("x", "y")
            ]
            if target_mode == "trajectory_xy"
            else target_fields
        )
        if target_mode == "future_state":
            missing = [
                f"{name}@{horizon:g}s"
                for horizon_index, horizon in enumerate(future_horizons)
                for name_index, name in enumerate(names)
                if counts[horizon_index, name_index].item() < 2
            ]
        else:
            missing = [names[i] for i, count in enumerate(counts.tolist()) if count < 2]
        raise RuntimeError(f"Could not compute target normalization stats for fields: {missing}")

    mean = sums / counts
    var = (sq_sums - (sums * sums) / counts) / (counts - 1.0).clamp_min(1.0)
    std = torch.sqrt(var.clamp_min(1e-12)).clamp_min(1e-6)
    mean = mean.float()
    std = std.float()
    return mean, std


def compute_or_broadcast_stats(
    rows,
    target_fields,
    num_frames,
    dataset_name="covla",
    frame_span=None,
    frame_stride=0,
    tubelet_size=2,
    target_mode="state",
    trajectory_point_indices=(-1,),
    agibot_fps=5.0,
    agibot_max_time_delta=0.05,
    egoexo_fps=5.0,
    egoexo_max_time_delta=0.05,
    egoexo_pose_fps=30.0,
    egoexo_min_views=2,
    egoexo_smooth_window=5,
    future_horizon_seconds=1.0,
    future_horizons_seconds=None,
    seed=0,
    distributed=False,
    rank=0,
    device=torch.device("cpu"),
    max_videos=2000,
):
    if not distributed or rank == 0:
        target_mean, target_std = compute_stats(
            rows,
            target_fields,
            num_frames=num_frames,
            dataset_name=dataset_name,
            frame_span=frame_span,
            frame_stride=frame_stride,
            tubelet_size=tubelet_size,
            target_mode=target_mode,
            trajectory_point_indices=trajectory_point_indices,
            agibot_fps=agibot_fps,
            agibot_max_time_delta=agibot_max_time_delta,
            egoexo_fps=egoexo_fps, egoexo_max_time_delta=egoexo_max_time_delta,
            egoexo_pose_fps=egoexo_pose_fps, egoexo_min_views=egoexo_min_views,
            egoexo_smooth_window=egoexo_smooth_window,
            future_horizon_seconds=future_horizon_seconds,
            future_horizons_seconds=future_horizons_seconds,
            seed=seed,
            max_videos=max_videos,
        )
        packed = torch.stack([target_mean, target_std], dim=0)
    else:
        num_outputs = 2 * len(tuple(trajectory_point_indices)) if target_mode == "trajectory_xy" else len(target_fields)
        if target_mode == "future_state":
            future_horizons = normalize_future_horizons_seconds(
                future_horizons_seconds,
                future_horizon_seconds,
            )
            packed = torch.empty(
                2,
                len(future_horizons),
                num_outputs,
                dtype=torch.float32,
            )
        else:
            packed = torch.empty(2, num_outputs, dtype=torch.float32)

    if distributed:
        comm_device = device if device.type == "cuda" else torch.device("cpu")
        packed = packed.to(comm_device)
        dist.broadcast(packed, src=0)
        packed = packed.cpu()

    return packed[0].clone(), packed[1].clone()
