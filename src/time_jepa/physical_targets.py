import json
import hashlib
import math
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from decord import VideoReader, cpu


TARGET_ALIASES = {
    "yaw_rate": "angular_velocities_calib[2]",
    "omega_z": "angular_velocities_calib[2]",
}

HFLIP_SIGN_FIELDS = {
    "angular_velocities_calib[2]",
    "angular_velocities_device[2]",
    "steeringAngleDeg",
    "yaw_rate",
    "omega_z",
}


def normalize_target_field(field):
    field = field.strip()
    return TARGET_ALIASES.get(field, field)


def normalize_target_fields(fields):
    if isinstance(fields, str):
        fields = [f.strip() for f in fields.split(",") if f.strip()]
    return [normalize_target_field(f) for f in fields]


def hflip_sign_vector(target_fields):
    signs = []
    for field in target_fields:
        normalized = normalize_target_field(field)
        signs.append(-1.0 if normalized in HFLIP_SIGN_FIELDS else 1.0)
    return torch.tensor(signs, dtype=torch.float32)


def read_payload_field(payload, field):
    field = normalize_target_field(field)
    if "[" in field and field.endswith("]"):
        base, index_text = field[:-1].split("[", 1)
        value = payload.get(base)
        try:
            index = int(index_text)
        except ValueError:
            return 0.0, 0.0
        is_valid = (
            isinstance(value, (list, tuple))
            and -len(value) <= index < len(value)
            and isinstance(value[index], (int, float))
            and math.isfinite(value[index])
        )
        return (float(value[index]) if is_valid else 0.0), (1.0 if is_valid else 0.0)

    value = payload.get(field)
    is_valid = value is not None and isinstance(value, (int, float)) and math.isfinite(value)
    return (float(value) if is_valid else 0.0), (1.0 if is_valid else 0.0)


@lru_cache(maxsize=512)
def load_state_series(states_path, target_fields_tuple):
    target_fields = normalize_target_fields(target_fields_tuple)
    frame_ids, values, valid = [], [], []
    with open(states_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frame_id_str, payload = next(iter(obj.items()))
            frame_ids.append(int(frame_id_str))
            row_values, row_valid = [], []
            for field in target_fields:
                value, is_valid = read_payload_field(payload, field)
                row_values.append(value)
                row_valid.append(is_valid)
            values.append(row_values)
            valid.append(row_valid)
    if not values:
        raise RuntimeError(f"Empty states file: {states_path}")
    frame_ids = np.array(frame_ids, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    valid = np.array(valid, dtype=np.float32)
    order = np.argsort(frame_ids)
    return frame_ids[order], values[order], valid[order]


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


def build_tubelet_targets(values, valid, state_indices, tubelet_size):
    per_frame_target = torch.from_numpy(values[state_indices])
    per_frame_valid = torch.from_numpy(valid[state_indices])
    if per_frame_target.size(0) % tubelet_size != 0:
        raise ValueError(
            f"Number of sampled frames must be divisible by tubelet_size; "
            f"got {per_frame_target.size(0)} and {tubelet_size}"
        )
    target = per_frame_target.reshape(-1, tubelet_size, per_frame_target.size(-1))
    valid_mask = per_frame_valid.reshape(-1, tubelet_size, per_frame_valid.size(-1))
    valid_sum = valid_mask.sum(dim=1).clamp_min(1.0)
    target = (target * valid_mask).sum(dim=1) / valid_sum
    valid_mask = (valid_mask.sum(dim=1) > 0).float()
    return target.float(), valid_mask.float()


def build_targets_for_clip(states_path, frame_indices, num_video_frames, target_fields, tubelet_size):
    frame_ids, values, valid = load_state_series(str(states_path), tuple(target_fields))
    state_indices = map_video_frames_to_state_indices(frame_indices, frame_ids, num_video_frames)
    return build_tubelet_targets(values, valid, state_indices, tubelet_size)


def read_covla_rows(data_paths):
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    rows = []
    for data_path in data_paths:
        dataset_root, metadata_path = resolve_covla_paths(data_path)
        with open(metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                file_name = record.get("file_name")
                if not file_name:
                    continue
                video_id = record.get("video_id") or Path(file_name).stem
                video_path = Path(file_name)
                if not video_path.is_absolute():
                    video_path = dataset_root / video_path
                states_path = dataset_root / "states" / f"{video_id}.jsonl"
                if not states_path.exists():
                    states_path = video_path.parent.parent / "states" / f"{video_id}.jsonl"
                if video_path.exists() and states_path.exists():
                    rows.append(
                        {
                            "video_id": video_id,
                            "video_path": str(video_path),
                            "states_path": str(states_path),
                        }
                    )
    if not rows:
        raise RuntimeError(f"No CoVLA rows with states found from: {data_paths}")
    return rows


def resolve_covla_paths(data_path):
    data_path = Path(os.path.expanduser(str(data_path))).resolve()
    if data_path.is_dir():
        dataset_root = data_path
        metadata_path = dataset_root / "metadata.jsonl"
    else:
        metadata_path = data_path
        dataset_root = metadata_path.parent
    if not metadata_path.exists():
        raise FileNotFoundError(f"CoVLA metadata file not found: {metadata_path}")
    return dataset_root, metadata_path


def sample_clip_indices_from_vr(
    vr,
    frames_per_clip,
    frame_step=None,
    fps=None,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    rng=None,
    strict_time_sampling=False,
):
    if rng is None:
        rng = np.random
    fstp = frame_step
    if duration is not None or fps is not None:
        video_fps = math.ceil(vr.get_avg_fps())
        if duration is not None:
            fstp = int(duration * video_fps / frames_per_clip)
        else:
            fstp = video_fps // fps
    if fstp is None or fstp <= 0:
        raise ValueError(f"Invalid frame step for stats sampling: {fstp}")

    clip_len = int(frames_per_clip * fstp)
    partition_len = len(vr) // num_clips
    clip_indices = []
    for i in range(num_clips):
        if fps is not None and strict_time_sampling:
            total_duration = max(0.0, float(len(vr) - 1) / float(video_fps))
            partition_start = (total_duration * i) / float(num_clips)
            partition_end = (total_duration * (i + 1)) / float(num_clips)
            clip_span = float(frames_per_clip - 1) / float(fps)
            if partition_end - partition_start > clip_span:
                max_start = partition_end - clip_span
                start_time = rng.uniform(partition_start, max_start) if random_clip_sampling else partition_start
            elif allow_clip_overlap and total_duration > clip_span and num_clips > 1:
                max_start = total_duration - clip_span
                start_time = max_start * i / float(num_clips - 1)
            else:
                start_time = partition_start
            sample_times = start_time + np.arange(frames_per_clip, dtype=np.float64) / float(fps)
            indices = np.rint(sample_times * float(video_fps))
            clip_indices.append(np.clip(indices, 0, len(vr) - 1).astype(np.int64))
            continue

        if partition_len > clip_len:
            end_indx = clip_len
            if random_clip_sampling:
                end_indx = rng.randint(clip_len, partition_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=frames_per_clip)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
            indices = indices + i * partition_len
        else:
            if not allow_clip_overlap:
                indices = np.linspace(0, partition_len, num=max(1, partition_len // fstp))
                pad = frames_per_clip - len(indices)
                if pad > 0:
                    indices = np.concatenate((indices, np.ones(pad) * partition_len))
                indices = np.clip(indices, 0, max(0, partition_len - 1)).astype(np.int64)
                indices = indices + i * partition_len
            else:
                sample_len = min(clip_len, len(vr)) - 1
                indices = np.linspace(0, sample_len, num=max(1, sample_len // fstp))
                pad = frames_per_clip - len(indices)
                if pad > 0:
                    indices = np.concatenate((indices, np.ones(pad) * sample_len))
                indices = np.clip(indices, 0, max(0, sample_len - 1)).astype(np.int64)
                clip_step = 0
                if len(vr) > clip_len and num_clips > 1:
                    clip_step = (len(vr) - clip_len) // (num_clips - 1)
                indices = indices + i * clip_step
        clip_indices.append(indices.astype(np.int64))
    return clip_indices


def compute_physical_stats(
    rows,
    target_fields,
    frames_per_clip,
    tubelet_size,
    frame_step=None,
    fps=None,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    seed=0,
    max_videos=2000,
    strict_time_sampling=False,
):
    rows = list(rows)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    if max_videos > 0 and len(rows) > max_videos:
        rows = rng.sample(rows, max_videos)

    target_fields = normalize_target_fields(target_fields)
    sums = torch.zeros(len(target_fields), dtype=torch.float64)
    sq_sums = torch.zeros(len(target_fields), dtype=torch.float64)
    counts = torch.zeros(len(target_fields), dtype=torch.float64)

    for row in rows:
        try:
            vr = VideoReader(row["video_path"], num_threads=1, ctx=cpu(0))
        except Exception:
            continue
        if len(vr) <= 0:
            continue
        clip_indices = sample_clip_indices_from_vr(
            vr,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            fps=fps,
            duration=duration,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            rng=np_rng,
            strict_time_sampling=strict_time_sampling,
        )
        for indices in clip_indices:
            target, valid_mask = build_targets_for_clip(
                row["states_path"],
                frame_indices=indices,
                num_video_frames=len(vr),
                target_fields=target_fields,
                tubelet_size=tubelet_size,
            )
            target = target.double()
            valid_mask = valid_mask.double()
            sums += (target * valid_mask).sum(dim=0)
            sq_sums += ((target**2) * valid_mask).sum(dim=0)
            counts += valid_mask.sum(dim=0)

    if (counts < 2).any():
        missing = [target_fields[i] for i, count in enumerate(counts.tolist()) if count < 2]
        raise RuntimeError(f"Could not compute physical target stats for fields: {missing}")

    mean = sums / counts
    var = (sq_sums - (sums * sums) / counts) / (counts - 1.0).clamp_min(1.0)
    std = torch.sqrt(var.clamp_min(1e-12)).clamp_min(1e-6)
    return mean.float(), std.float(), counts.long()


def _stats_cache_matches(payload, meta):
    cached_meta = payload.get("meta", {})
    for key, value in meta.items():
        if cached_meta.get(key) != value:
            return False
    return True


def _rows_fingerprint(rows):
    video_ids = sorted(str(row.get("video_id", "")) for row in rows)
    digest = hashlib.sha1("\n".join(video_ids).encode("utf-8")).hexdigest()
    return {
        "row_count": len(video_ids),
        "video_ids_sha1": digest,
    }


def get_or_compute_physical_stats(
    cache_path,
    rows,
    target_fields,
    frames_per_clip,
    tubelet_size,
    frame_step=None,
    fps=None,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    seed=0,
    max_videos=2000,
    strict_time_sampling=False,
    distributed=False,
    rank=0,
    device=torch.device("cpu"),
):
    target_fields = normalize_target_fields(target_fields)
    row_meta = _rows_fingerprint(rows)
    meta = {
        "target_fields": target_fields,
        "frames_per_clip": int(frames_per_clip),
        "tubelet_size": int(tubelet_size),
        "frame_step": None if frame_step is None else int(frame_step),
        "fps": None if fps is None else int(fps),
        "duration": duration,
        "num_clips": int(num_clips),
        "random_clip_sampling": bool(random_clip_sampling),
        "allow_clip_overlap": bool(allow_clip_overlap),
        "strict_time_sampling": bool(strict_time_sampling),
        "seed": int(seed),
        "max_videos": int(max_videos),
        **row_meta,
    }
    cache_path = Path(cache_path).expanduser() if cache_path else None

    if not distributed or rank == 0:
        payload = None
        if cache_path and cache_path.exists():
            with cache_path.open("r") as f:
                payload = json.load(f)
            if not _stats_cache_matches(payload, meta):
                payload = None
        if payload is None:
            mean, std, counts = compute_physical_stats(
                rows=rows,
                target_fields=target_fields,
                frames_per_clip=frames_per_clip,
                tubelet_size=tubelet_size,
                frame_step=frame_step,
                fps=fps,
                duration=duration,
                num_clips=num_clips,
                random_clip_sampling=random_clip_sampling,
                allow_clip_overlap=allow_clip_overlap,
                seed=seed,
                max_videos=max_videos,
                strict_time_sampling=strict_time_sampling,
            )
            payload = {
                "meta": meta,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "counts": counts.tolist(),
            }
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                with tmp_path.open("w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, cache_path)
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
    else:
        mean = torch.empty(len(target_fields), dtype=torch.float32)
        std = torch.empty(len(target_fields), dtype=torch.float32)

    if distributed:
        comm_device = device if device.type == "cuda" else torch.device("cpu")
        packed = torch.stack([mean, std], dim=0).to(comm_device)
        dist.broadcast(packed, src=0)
        mean, std = packed[0].cpu(), packed[1].cpu()

    return mean, std
