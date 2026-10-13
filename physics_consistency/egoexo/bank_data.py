"""EgoExo multiview rows, four scalar targets, and seed-invariant stats."""

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

from probe_datasets import EgoExoProbeDataset, load_egoexo_split_rows
from src.time_jepa.egoexo_pose_targets import build_egoexo_tubelet_targets
from src.time_jepa.time_sampling import build_sample_times


def _limit_rows(rows: list[dict], limit: int) -> list[dict]:
    return list(rows[: int(limit)]) if int(limit or 0) > 0 else list(rows)


def _load_rows(data: dict, split_name: str, video_view) -> list[dict]:
    (rows,) = load_egoexo_split_rows(
        data["egoexo_root"],
        data["split_dir"],
        split_names=(split_name,),
        prefix=data["split_prefix"],
        video_view=video_view,
        frames_per_clip=int(data.get("num_frames", 64)),
        fps=float(data.get("fps", 5.0)),
        slice_stride_seconds=float(data.get("slice_stride_seconds", 12.8)),
        manifest_path=data["manifest_path"],
    )
    rows = _limit_rows(rows, data.get("limits", {}).get(split_name, 0))
    if not rows:
        raise RuntimeError(f"EgoExo {split_name}/{video_view} is empty")
    return rows


def load_protocol_rows(config: dict) -> dict:
    data = {
        **config["data"],
        "num_frames": config["sampling"]["num_frames"],
        "fps": config["sampling"]["fps"],
    }
    train_split, val_split = data["train_split"], data["val_split"]
    physical_split = "test" if config["evaluation"]["mode"] == "test" else val_split

    # One physical row with three candidates; each training access samples one
    # of view01/02/04.  Labels and stats remain deduplicated by physical window.
    train_rows = _load_rows(data, train_split, list(data["train_views"]))

    # Validation is explicitly averaged over all three ID cameras rather than
    # silently using whichever candidate happens to be first.
    val_rows = [
        row
        for view in data["id_views"]
        for row in _load_rows(data, val_split, view)
    ]
    eval_rows = {}
    eval_views = (
        tuple(data["id_views"])
        if physical_split == val_split
        else (*data["id_views"], *data["ood_views"])
    )
    for view in eval_views:
        role = "id" if view in data["id_views"] else "ood"
        eval_rows[f"{role}__{view}"] = _load_rows(data, physical_split, view)

    train_takes = {row["take_uid"] for row in train_rows}
    val_takes = {row["take_uid"] for row in val_rows}
    if train_takes & val_takes:
        raise RuntimeError("probe_train/probe_val take overlap")
    if physical_split == "test":
        eval_takes = {
            row["take_uid"] for rows in eval_rows.values() for row in rows
        }
        overlap = (train_takes | val_takes) & eval_takes
        if overlap:
            raise RuntimeError(f"Final test overlaps train/val takes: {sorted(overlap)[:5]}")
    return {
        "train": train_rows,
        "val": val_rows,
        "eval": eval_rows,
        "physical_split": physical_split,
    }


class EgoExoPhysicsBankDataset(EgoExoProbeDataset):
    def __init__(
        self,
        rows,
        sampling,
        transform,
        random_view=False,
        stats=None,
        data_config=None,
        pose_config=None,
    ):
        data_config = data_config or {}
        pose_config = pose_config or {}
        super().__init__(
            rows=rows,
            target_fields=TARGET_FIELDS,
            num_frames=int(sampling["num_frames"]),
            transform=transform,
            random_clip=random_view,
            stats=stats,
            tubelet_size=int(sampling["tubelet_size"]),
            video_num_threads=int(data_config.get("video_num_threads", 1)),
            egoexo_fps=float(sampling["fps"]),
            egoexo_max_time_delta=float(sampling.get("max_time_delta", 0.05)),
            egoexo_pose_fps=float(pose_config.get("fps", 30.0)),
            egoexo_min_views=int(pose_config.get("min_views", 2)),
            egoexo_smooth_window=int(pose_config.get("smooth_window", 5)),
            egoexo_video_backend=data_config.get("video_backend", "auto"),
            egoexo_ffmpeg_timeout=float(data_config.get("ffmpeg_timeout", 120)),
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        item.update(
            {
                "take_uid": str(row["take_uid"]),
                "window_key": str(row["window_key"]),
                "task_id": str(row.get("task_id", "")),
                "participant_uid": str(row.get("participant_uid", "")),
                "capture_uid": str(row.get("capture_uid", "")),
                "view_key": str(item.get("video_view", row.get("video_view", ""))),
                "pose_path": str(Path(row["pose_path"]).expanduser().resolve()),
            }
        )
        return item


def stats_fingerprint(
    rows: list[dict], sampling: dict, pose: dict, seed: int, max_rows: int
) -> dict:
    payload = {
        "schema": 1,
        "target_names": TARGET_NAMES,
        "physical_windows": sorted({str(row["window_key"]) for row in rows}),
        "sampling": {
            "num_frames": int(sampling["num_frames"]),
            "fps": float(sampling["fps"]),
            "tubelet_size": int(sampling["tubelet_size"]),
            "max_time_delta": float(sampling.get("max_time_delta", 0.05)),
        },
        "pose": {
            "fps": float(pose.get("fps", 30.0)),
            "min_views": int(pose.get("min_views", 2)),
            "smooth_window": int(pose.get("smooth_window", 5)),
        },
        "stats_seed": int(seed),
        "max_rows": int(max_rows),
    }
    return {"sha256": canonical_hash(payload), "payload": payload}


def compute_stats(
    rows: list[dict],
    sampling: dict,
    pose: dict,
    seed: int = 239,
    max_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Camera-invariant targets: one row per physical window even if callers
    # accidentally provide repeated camera rows.
    selected_by_window = {}
    for row in rows:
        selected_by_window.setdefault(str(row["window_key"]), row)
    selected = list(selected_by_window.values())
    rng = random.Random(int(seed))
    if max_rows > 0 and len(selected) > max_rows:
        selected = rng.sample(selected, int(max_rows))
    width = len(TARGET_NAMES)
    sums = torch.zeros(width, dtype=torch.float64)
    square_sums = torch.zeros(width, dtype=torch.float64)
    counts = torch.zeros(width, dtype=torch.float64)
    for row in selected:
        sample_times = build_sample_times(
            row["start_time"], int(sampling["num_frames"]), float(sampling["fps"])
        )
        target, valid = build_egoexo_tubelet_targets(
            row["pose_path"],
            sample_times,
            TARGET_FIELDS,
            int(sampling["tubelet_size"]),
            float(sampling.get("max_time_delta", 0.05)),
            float(pose.get("fps", 30.0)),
            int(pose.get("min_views", 2)),
            int(pose.get("smooth_window", 5)),
        )
        target, valid = target.double(), valid.double()
        sums += (target * valid).sum(0)
        square_sums += (target.square() * valid).sum(0)
        counts += valid.sum(0)
    if (counts < 2).any():
        missing = [TARGET_NAMES[index] for index in torch.where(counts < 2)[0].tolist()]
        raise RuntimeError(f"Insufficient normalization samples for {missing}")
    mean = sums / counts
    variance = (square_sums - sums.square() / counts) / (counts - 1).clamp_min(1)
    std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
    return mean.float(), std.float(), counts


def get_or_compute_stats(
    rows: list[dict],
    sampling: dict,
    pose: dict,
    cache_path: str | Path,
    seed: int = 239,
    max_rows: int = 0,
    distributed: bool = False,
    rank: int = 0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    cache_path = Path(cache_path).expanduser().resolve()
    fingerprint = stats_fingerprint(rows, sampling, pose, seed, max_rows)
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
                mean, std, counts = compute_stats(rows, sampling, pose, seed, max_rows)
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

