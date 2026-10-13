"""AgiBot multiview rows, 20-D q/p targets, and seed-invariant stats."""

from __future__ import annotations

import fcntl
import json
import random
from pathlib import Path

import torch
import torch.distributed as dist

try:
    from .bank_common import TARGET_FIELDS, TARGET_NAMES, atomic_json_dump, canonical_hash
except ImportError:
    from bank_common import TARGET_FIELDS, TARGET_NAMES, atomic_json_dump, canonical_hash

from probe_datasets import (
    AgiBotProbeDataset,
    agibot_sample_times,
    build_agibot_tubelet_targets,
    read_agibot_metadata,
    split_agibot_rows_from_files,
)


def _limit_rows(rows: list[dict], limit: int) -> list[dict]:
    return list(rows[: int(limit)]) if int(limit or 0) > 0 else list(rows)


def load_single_view_splits(
    data_config: dict,
    video_key: str,
    split_names: tuple[str, ...] | list[str],
) -> dict[str, list[dict]]:
    rows = read_agibot_metadata(
        data_config["agibot_root"],
        tasks=data_config["tasks"],
        video_key=video_key,
        use_meta_episodes_only=bool(data_config.get("use_meta_episodes_only", True)),
    )
    split_rows = split_agibot_rows_from_files(
        rows,
        data_config["split_dir"],
        split_names=tuple(split_names),
        prefix=data_config.get("split_prefix", "agibot_top_head"),
    )
    output = {}
    limits = data_config.get("limits", {})
    for split_name, items in zip(split_names, split_rows):
        prepared = []
        for row in items:
            row = dict(row)
            row["slice_stride_seconds"] = float(
                data_config.get("slice_stride_seconds", 12.8)
            )
            prepared.append(row)
        output[split_name] = _limit_rows(prepared, limits.get(split_name, 0))
        if not output[split_name]:
            raise RuntimeError(f"AgiBot {split_name}/{video_key} is empty after limits")
    return output


def load_protocol_rows(config: dict) -> dict:
    data = config["data"]
    train_split = data["train_split"]
    val_split = data["val_split"]
    physical_split = "test" if config["evaluation"]["mode"] == "test" else val_split
    needed_splits = tuple(dict.fromkeys((train_split, val_split, physical_split)))
    requested_eval_views = (
        (*data["id_views"], *data["ood_views"])
        if physical_split == "test"
        else tuple(data["id_views"])
    )
    # Pilot/smoke must not even enumerate fisheye OOD metadata.  The final
    # protocol is the only phase allowed to construct those rows.
    views = tuple(dict.fromkeys((*data["train_views"], *requested_eval_views)))
    by_view = {
        view: load_single_view_splits(data, view, needed_splits)
        for view in views
    }
    train_rows = [row for view in data["train_views"] for row in by_view[view][train_split]]
    val_rows = [row for view in data["train_views"] for row in by_view[view][val_split]]
    eval_rows = {}
    eval_views = data["id_views"] if physical_split == val_split else (*data["id_views"], *data["ood_views"])
    for view in eval_views:
        role = "id" if view in data["id_views"] else "ood"
        eval_rows[f"{role}__{view.rsplit('.', 1)[-1]}"] = by_view[view][physical_split]

    train_episodes = {row["episode_key"] for row in train_rows}
    val_episodes = {row["episode_key"] for row in val_rows}
    if train_episodes & val_episodes:
        raise RuntimeError("probe_train/probe_val episode overlap")
    if physical_split == "test":
        eval_episodes = {row["episode_key"] for rows in eval_rows.values() for row in rows}
        overlap = (train_episodes | val_episodes) & eval_episodes
        if overlap:
            raise RuntimeError(f"Final test overlaps train/val episodes: {sorted(overlap)[:5]}")
    return {
        "train": train_rows,
        "val": val_rows,
        "eval": eval_rows,
        "physical_split": physical_split,
    }


class AgiBotPhysicsBankDataset(AgiBotProbeDataset):
    def __init__(self, rows, sampling, transform, random_clip=False, stats=None, data_config=None):
        data_config = data_config or {}
        super().__init__(
            rows=rows,
            target_fields=TARGET_FIELDS,
            num_frames=int(sampling["num_frames"]),
            transform=transform,
            random_clip=random_clip,
            stats=stats,
            tubelet_size=int(sampling["tubelet_size"]),
            video_num_threads=int(data_config.get("video_num_threads", 1)),
            agibot_fps=float(sampling["fps"]),
            agibot_max_time_delta=float(sampling.get("max_time_delta", 0.05)),
            agibot_video_backend=data_config.get("video_backend", "ffmpeg"),
            agibot_ffmpeg_timeout=float(data_config.get("ffmpeg_timeout", 120)),
            agibot_slice_stride_seconds=float(data_config.get("slice_stride_seconds", 12.8)),
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item.update(
            {
                "episode_key": row["episode_key"],
                "window_key": row.get("window_key", row["episode_key"]),
                "task_id": int(row["task_id"]),
                "view_key": row["video_key"],
            }
        )
        return item


def stats_fingerprint(rows: list[dict], sampling: dict, seed: int, max_rows: int) -> dict:
    payload = {
        "schema": 1,
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
    rows: list[dict], sampling: dict, seed: int = 239, max_rows: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # q/p labels are camera invariant.  The three training cameras therefore
    # share one statistics sample per physical window, avoiding three identical
    # parquet reads without changing the fitted distribution.
    selected_by_window = {}
    for row in rows:
        key = row.get("window_key", row.get("episode_key", row["video_id"]))
        selected_by_window.setdefault(key, row)
    selected = list(selected_by_window.values())
    rng = random.Random(int(seed))
    if max_rows > 0 and len(selected) > max_rows:
        selected = rng.sample(selected, int(max_rows))
    width = len(TARGET_NAMES)
    sums = torch.zeros(width, dtype=torch.float64)
    sq_sums = torch.zeros(width, dtype=torch.float64)
    counts = torch.zeros(width, dtype=torch.float64)
    for row in selected:
        sample_times = agibot_sample_times(
            row,
            int(sampling["num_frames"]),
            float(sampling["fps"]),
            random_clip=False,
        )
        target, valid = build_agibot_tubelet_targets(
            row["parquet_path"],
            sample_times,
            TARGET_FIELDS,
            int(sampling["tubelet_size"]),
            max_time_delta=float(sampling.get("max_time_delta", 0.05)),
        )
        target, valid = target.double(), valid.double()
        sums += (target * valid).sum(0)
        sq_sums += (target.square() * valid).sum(0)
        counts += valid.sum(0)
    if (counts < 2).any():
        missing = [TARGET_NAMES[index] for index in torch.where(counts < 2)[0].tolist()]
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
                mean, std, counts = compute_stats(rows, sampling, seed, max_rows)
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


def collect_raw_targets(
    rows: list[dict],
    sampling: dict,
    progress_every: int = 0,
    progress_label: str = "targets",
) -> dict:
    """Collect pooled q/p GT without decoding video, preserving row metadata."""

    targets, valid, task_ids, episode_keys, window_keys = [], [], [], [], []
    for row_index, row in enumerate(rows):
        sample_times = agibot_sample_times(
            row,
            int(sampling["num_frames"]),
            float(sampling["fps"]),
            random_clip=False,
        )
        target, mask = build_agibot_tubelet_targets(
            row["parquet_path"],
            sample_times,
            TARGET_FIELDS,
            int(sampling["tubelet_size"]),
            max_time_delta=float(sampling.get("max_time_delta", 0.05)),
        )
        targets.append(target.numpy())
        valid.append(mask.numpy())
        task_ids.extend([int(row["task_id"])] * len(target))
        episode_keys.extend([row["episode_key"]] * len(target))
        window_keys.extend([row.get("window_key", row["episode_key"])] * len(target))
        if progress_every > 0 and (row_index + 1) % int(progress_every) == 0:
            print(
                f"{progress_label}: {row_index + 1}/{len(rows)} windows",
                flush=True,
            )
    import numpy as np

    return {
        "target": np.concatenate(targets, axis=0),
        "valid": np.concatenate(valid, axis=0),
        "task_id": np.asarray(task_ids, dtype=np.int64),
        "episode_key": np.asarray(episode_keys, dtype=str),
        "window_key": np.asarray(window_keys, dtype=str),
    }
