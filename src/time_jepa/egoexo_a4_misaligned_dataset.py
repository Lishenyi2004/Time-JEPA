"""Isolated EgoExo A4 dataset with deterministic video--target misalignment.

This module intentionally does not modify the production EgoExo dataset used
by running jobs.  It is loaded only by the dedicated ``vjepa_2_1_a4`` app.
"""

import hashlib
import json
import os
import pathlib
import warnings
from logging import getLogger

import numpy as np
import torch

from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.time_jepa.egoexo_pose_targets import build_egoexo_tubelet_targets
from src.time_jepa.egoexo_time_dataset import EgoExoTimeJEPADataset
from src.time_jepa.egoexo_views import select_egoexo_video_candidate
from src.time_jepa.time_sampling import build_sample_times


logger = getLogger(__name__)

A4_EGOEXO_MISALIGNMENT_MODE = "take_derangement"


def build_group_derangement(group_ids, seed):
    """Return a seeded permutation with no fixed or same-group assignments.

    Group members are placed in contiguous blocks and circularly shifted by
    the size of the largest block.  When no group contains more than half of
    the samples, this is a bijection and no block can overlap itself.
    """
    group_ids = [str(group_id) for group_id in group_ids]
    num_samples = len(group_ids)
    if num_samples < 2:
        raise ValueError("A4 target derangement requires at least two samples")

    grouped_indices = {}
    for index, group_id in enumerate(group_ids):
        grouped_indices.setdefault(group_id, []).append(index)
    max_group_size = max(len(indices) for indices in grouped_indices.values())
    if max_group_size * 2 > num_samples:
        raise ValueError(
            "Cannot construct a same-group-free target derangement when one "
            f"group contains more than half the samples: max_group_size="
            f"{max_group_size}, num_samples={num_samples}"
        )

    rng = np.random.RandomState(int(seed))
    group_order = sorted(grouped_indices)
    rng.shuffle(group_order)
    ordered_indices = []
    for group_id in group_order:
        indices = np.asarray(grouped_indices[group_id], dtype=np.int64)
        rng.shuffle(indices)
        ordered_indices.extend(indices.tolist())

    source_indices = np.empty(num_samples, dtype=np.int64)
    for position, video_index in enumerate(ordered_indices):
        source_indices[video_index] = ordered_indices[
            (position + max_group_size) % num_samples
        ]

    expected = np.arange(num_samples, dtype=np.int64)
    if not np.array_equal(np.sort(source_indices), expected):
        raise RuntimeError("A4 target mapping is not a permutation")
    if np.any(source_indices == expected):
        raise RuntimeError("A4 target mapping unexpectedly contains a fixed point")
    same_group = [
        index
        for index, source_index in enumerate(source_indices.tolist())
        if group_ids[index] == group_ids[source_index]
    ]
    if same_group:
        raise RuntimeError(
            "A4 target mapping unexpectedly contains same-group pairs; "
            f"first={same_group[:5]}"
        )
    return source_indices


def build_take_derangement(rows, seed):
    take_uids = [row["take_uid"] for row in rows]
    return build_group_derangement(take_uids, seed=seed)


def audit_take_derangement(rows, source_indices, seed):
    source_indices = np.asarray(source_indices, dtype=np.int64)
    num_samples = len(rows)
    if source_indices.shape != (num_samples,):
        raise ValueError(
            f"Expected {num_samples} source indices, got {source_indices.shape}"
        )
    permutation_ok = np.array_equal(
        np.sort(source_indices),
        np.arange(num_samples, dtype=np.int64),
    )
    fixed_points = int(np.sum(source_indices == np.arange(num_samples)))
    same_take_pairs = sum(
        rows[index]["take_uid"] == rows[int(source_index)]["take_uid"]
        for index, source_index in enumerate(source_indices.tolist())
    )
    pair_lines = [
        f"{rows[index]['window_key']}->{rows[int(source_index)]['window_key']}"
        for index, source_index in enumerate(source_indices.tolist())
    ]
    return {
        "mode": A4_EGOEXO_MISALIGNMENT_MODE,
        "seed": int(seed),
        "num_samples": num_samples,
        "num_unique_video_windows": len(
            {row["window_key"] for row in rows}
        ),
        "num_unique_target_windows": len(set(source_indices.tolist())),
        "permutation_ok": bool(permutation_ok),
        "fixed_points": fixed_points,
        "same_take_pairs": int(same_take_pairs),
        "target_marginal_preserved_by_bijection": bool(permutation_ok),
        "valid_mask_marginal_preserved_by_bijection": bool(permutation_ok),
        "pairing_sha256": hashlib.sha256(
            "\n".join(pair_lines).encode("utf-8")
        ).hexdigest(),
        "pairing_preview": pair_lines[:10],
    }


def write_pairing_manifest(path, rows, source_indices, seed):
    """Write the full reproducibility manifest atomically."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_take_derangement(rows, source_indices, seed=seed)
    payload = {
        **audit,
        "pairs": [
            {
                "video_window": rows[index]["window_key"],
                "video_take_uid": rows[index]["take_uid"],
                "target_window": rows[int(source_index)]["window_key"],
                "target_take_uid": rows[int(source_index)]["take_uid"],
            }
            for index, source_index in enumerate(source_indices.tolist())
        ],
    }
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)
    return audit


class EgoExoA4MisalignedDataset(EgoExoTimeJEPADataset):
    """Decode video from one window and full-9D targets from another take."""

    def __init__(
        self,
        *args,
        misalignment_mode=A4_EGOEXO_MISALIGNMENT_MODE,
        misalignment_seed=239,
        pairing_manifest_path=None,
        **kwargs,
    ):
        normalized_mode = str(misalignment_mode).strip().lower().replace("-", "_")
        if normalized_mode != A4_EGOEXO_MISALIGNMENT_MODE:
            raise ValueError(
                "EgoExo A4 only supports misalignment_mode="
                f"{A4_EGOEXO_MISALIGNMENT_MODE!r}, got {misalignment_mode!r}"
            )
        target_temporal_mode = kwargs.get("target_temporal_mode", "tubelet_aligned")
        if str(target_temporal_mode).strip().lower().replace("-", "_") not in (
            "aligned",
            "tubelet",
            "tubelet_aligned",
        ):
            raise ValueError(
                "A4 M2d must use tubelet-aligned targets before video--target "
                "misalignment"
            )
        super().__init__(*args, **kwargs)
        if not self.return_physical_targets:
            raise ValueError("A4 M2d requires return_physical_targets=True")

        self.misalignment_mode = normalized_mode
        self.misalignment_seed = int(misalignment_seed)
        self.physical_source_indices = build_take_derangement(
            self.rows,
            seed=self.misalignment_seed,
        )
        self.pairing_audit = audit_take_derangement(
            self.rows,
            self.physical_source_indices,
            seed=self.misalignment_seed,
        )
        if pairing_manifest_path:
            write_pairing_manifest(
                pairing_manifest_path,
                self.rows,
                self.physical_source_indices,
                seed=self.misalignment_seed,
            )
        logger.info("EgoExo A4 pairing audit: %s", self.pairing_audit)

    def get_item_video(self, index):
        video_row = select_egoexo_video_candidate(
            self.rows[index],
            random_view=True,
        )
        target_index = int(self.physical_source_indices[index])
        target_row = self.rows[target_index]
        try:
            video_sample_times = build_sample_times(
                video_row["start_time"],
                self.frames_per_clip,
                self.fps,
            )
            target_sample_times = build_sample_times(
                target_row["start_time"],
                self.frames_per_clip,
                self.fps,
            )
            buffer, frame_indices, backend = self._read_video(
                video_row,
                video_sample_times,
            )
            target, valid = build_egoexo_tubelet_targets(
                target_row["pose_path"],
                target_sample_times,
                target_fields=self.target_fields,
                tubelet_size=self.tubelet_size,
                max_time_delta=self.max_time_delta,
                pose_fps=self.pose_fps,
                min_views=self.min_views,
                smooth_window=self.smooth_window,
            )
        except Exception as exc:
            warnings.warn(
                "Failed to build EgoExo A4 sample "
                f"{video_row['window_key']}->{target_row['window_key']}: {exc}"
            )
            return None

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = [self.transform(buffer)] if self.transform is not None else [buffer]
        physical = {
            "domain": "egoexo",
            "target_temporal_mode": "tubelet_aligned",
            "target_pairing_mode": self.misalignment_mode,
            "target": [target],
            "valid": [valid],
            "video_id": f"{video_row['window_key']}/{video_row['video_view']}",
            "physical_video_id": target_row["window_key"],
            "physical_misaligned": True,
            "take_uid": video_row["take_uid"],
            "physical_take_uid": target_row["take_uid"],
            "task_id": video_row.get("task_id", ""),
            "physical_task_id": target_row.get("task_id", ""),
            "video_key": video_row["video_view"],
            "physical_video_view": "pose_target_from_other_take",
            "video_backend": backend,
            "start_time": float(video_row["start_time"]),
            "physical_start_time": float(target_row["start_time"]),
            "sample_times": [
                torch.from_numpy(video_sample_times.astype(np.float32))
            ],
            "physical_sample_times": [
                torch.from_numpy(target_sample_times.astype(np.float32))
            ],
        }
        return buffer, 0, [frame_indices], physical


def make_egoexo_a4_misaligned_dataset(
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
    egoexo_misalignment_mode=A4_EGOEXO_MISALIGNMENT_MODE,
    egoexo_misalignment_seed=239,
    pairing_manifest_path=None,
):
    if num_clips != 1:
        raise NotImplementedError("EgoExo A4 supports num_clips=1")
    if dataset_fpcs is not None:
        frames_per_clip = int(dataset_fpcs[0])
    root = data_paths[0] if isinstance(data_paths, (list, tuple)) else data_paths
    dataset = EgoExoA4MisalignedDataset(
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
        misalignment_mode=egoexo_misalignment_mode,
        misalignment_seed=egoexo_misalignment_seed,
        pairing_manifest_path=pairing_manifest_path,
    )
    if log_dir:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        dataset = MonitoredDataset(
            dataset,
            str(log_dir / f"resource_file_{rank}_%w.csv"),
            10.0,
            5.0,
        )
    if datasets_weights is not None:
        sampler = DistributedWeightedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    loader_cls = (
        torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    )
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
        "EgoExoA4MisalignedDataset created rows=%s targets=%s audit=%s",
        len(dataset),
        egoexo_target_fields,
        getattr(dataset, "pairing_audit", None),
    )
    return dataset, loader, sampler
