import os
import pathlib
import random
import warnings
from logging import getLogger

import numpy as np
import torch

from src.datasets.covla_video_dataset import CoVLAVideoDataset
from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.masks.multiseq_multiblock3d import MaskCollator
from src.time_jepa.physical_targets import (
    build_targets_for_clip,
    hflip_sign_vector,
    normalize_target_fields,
)


logger = getLogger()


def _normalize_misalignment_mode(enabled, mode):
    if not enabled:
        return "none"
    mode = str(mode or "derangement").strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
        "disabled": "none",
        "on": "derangement",
        "true": "derangement",
        "shuffle_no_self": "derangement",
        "permutation": "shuffle",
    }
    return aliases.get(mode, mode)


def _build_misaligned_indices(num_items, mode, seed=0, shift=1):
    indices = list(range(num_items))
    if mode == "none":
        return indices
    if num_items < 2:
        raise ValueError("physical target misalignment requires at least two samples")

    rng = random.Random(int(seed))
    if mode == "shuffle":
        shuffled = list(indices)
        rng.shuffle(shuffled)
        return shuffled

    if mode == "derangement":
        order = list(indices)
        rng.shuffle(order)
        return {order[i]: order[(i + 1) % num_items] for i in range(num_items)}

    if mode == "shift":
        offset = int(shift) % num_items
        if offset == 0:
            raise ValueError("physical target misalignment shift must not be a multiple of dataset length")
        return [(i + offset) % num_items for i in indices]

    raise ValueError(
        "Unsupported physical target misalignment mode "
        f"{mode!r}; expected one of none, shuffle, derangement, shift"
    )


def _relative_source_indices(frame_indices, source_num_frames, reference_num_frames):
    frame_indices = np.asarray(frame_indices, dtype=np.float32)
    if source_num_frames <= 1 or reference_num_frames <= 1:
        return np.zeros_like(frame_indices, dtype=np.int64)
    scale = float(source_num_frames - 1) / float(reference_num_frames - 1)
    return np.clip(np.rint(frame_indices * scale), 0, source_num_frames - 1).astype(np.int64)


class CoVLATimeJEPADataset(CoVLAVideoDataset):
    """CoVLA video dataset that attaches tubelet-level physical targets."""

    def __init__(
        self,
        *args,
        physical_target_fields=None,
        physical_tubelet_size=2,
        physical_misalignment_enabled=False,
        physical_misalignment_mode="none",
        physical_misalignment_seed=0,
        physical_misalignment_shift=1,
        return_physical_targets=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_physical_targets = bool(return_physical_targets)
        self.physical_target_fields = normalize_target_fields(physical_target_fields or [])
        self.physical_tubelet_size = physical_tubelet_size
        self.hflip_sign = hflip_sign_vector(self.physical_target_fields)
        self.states_paths = []
        for sample, video_id in zip(self.samples, self.video_ids):
            states_path = pathlib.Path(sample).parent.parent / "states" / f"{video_id}.jsonl"
            self.states_paths.append(str(states_path))
        self.physical_misalignment_mode = _normalize_misalignment_mode(
            physical_misalignment_enabled,
            physical_misalignment_mode,
        )
        self.physical_misalignment_seed = int(physical_misalignment_seed)
        self.physical_misalignment_shift = int(physical_misalignment_shift)
        source_indices = _build_misaligned_indices(
            len(self.samples),
            mode=self.physical_misalignment_mode,
            seed=self.physical_misalignment_seed,
            shift=self.physical_misalignment_shift,
        )
        if isinstance(source_indices, dict):
            source_indices = [source_indices[i] for i in range(len(self.samples))]
        self.physical_source_indices = source_indices
        if self.physical_misalignment_mode != "none":
            preview = [
                f"{self.video_ids[i]}->{self.video_ids[self.physical_source_indices[i]]}"
                for i in range(min(5, len(self.video_ids)))
            ]
            logger.info(
                "CoVLA physical targets misaligned: mode=%s seed=%s shift=%s samples=%s preview=%s",
                self.physical_misalignment_mode,
                self.physical_misalignment_seed,
                self.physical_misalignment_shift,
                len(self.samples),
                preview,
            )

    def get_item_video(self, index):
        loaded_sample = super().get_item_video(index)
        if not loaded_sample:
            return

        buffer, label, clip_indices = loaded_sample
        if not self.return_physical_targets:
            return buffer, label, clip_indices

        hflip_flags = [False for _ in buffer]
        normalized_buffer = []
        for clip_idx, clip in enumerate(buffer):
            if isinstance(clip, tuple) and len(clip) == 2:
                clip_tensor, clip_meta = clip
                normalized_buffer.append(clip_tensor)
                hflip_flags[clip_idx] = bool(clip_meta.get("hflip", False))
            else:
                normalized_buffer.append(clip)
        buffer = normalized_buffer
        sample = self.samples[index]
        video_id = self.video_ids[index]
        source_index = self.physical_source_indices[index]
        source_sample = self.samples[source_index]
        source_video_id = self.video_ids[source_index]
        states_path = self.states_paths[source_index]
        if not os.path.exists(states_path):
            warnings.warn(f"CoVLA states path not found for {source_video_id}: {states_path}")
            return

        try:
            num_video_frames = self._get_num_video_frames(sample)
            source_num_video_frames = self._get_num_video_frames(source_sample)
            physical_targets, physical_valid = [], []
            for indices, hflip in zip(clip_indices, hflip_flags):
                target_indices = indices
                target_num_video_frames = num_video_frames
                if source_index != index:
                    target_indices = _relative_source_indices(
                        indices,
                        source_num_frames=source_num_video_frames,
                        reference_num_frames=num_video_frames,
                    )
                    target_num_video_frames = source_num_video_frames
                target, valid = build_targets_for_clip(
                    states_path=states_path,
                    frame_indices=target_indices,
                    num_video_frames=target_num_video_frames,
                    target_fields=self.physical_target_fields,
                    tubelet_size=self.physical_tubelet_size,
                )
                if hflip:
                    target = target * self.hflip_sign.view(1, -1)
                physical_targets.append(target)
                physical_valid.append(valid)
        except Exception as exc:
            warnings.warn(f"Failed to build CoVLA physical targets for {video_id}: {exc}")
            return

        physical = {
            "target": physical_targets,
            "valid": physical_valid,
            "hflip": hflip_flags,
            "video_id": video_id,
            "physical_video_id": source_video_id,
            "physical_misaligned": source_index != index,
        }
        return buffer, label, clip_indices, physical

    def _get_num_video_frames(self, sample):
        if not hasattr(self, "_num_video_frames_cache"):
            self._num_video_frames_cache = {}
        cached = self._num_video_frames_cache.get(sample)
        if cached is not None:
            return cached
        from decord import VideoReader, cpu

        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
        num_frames = len(vr)
        self._num_video_frames_cache[sample] = num_frames
        return num_frames


class TimeJEPAMaskCollator(MaskCollator):
    """Mask collator that preserves CoVLA physical targets."""

    def __call__(self, batch):
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            if len(sample) >= 3 and isinstance(sample[2], (list, tuple)):
                try:
                    fpc = len(sample[2][-1])
                except (TypeError, IndexError):
                    fpc = 1
            else:
                fpc = 1
            if fpc in filtered_batches:
                filtered_batches[fpc].append(sample)

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue

            video_part = [sample[:3] for sample in fpc_batch]
            collated_batch = torch.utils.data.default_collate(video_part)
            physical_part = [sample[3] for sample in fpc_batch]
            collated_physical = torch.utils.data.default_collate(physical_part)
            collated_batch = tuple(list(collated_batch) + [collated_physical])

            collated_masks_pred, collated_masks_enc = [], []
            for mask_generator in self.mask_generators[fpc]:
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            fpc_collations.append((collated_batch, collated_masks_enc, collated_masks_pred))

        return fpc_collations


def make_covla_time_jepa_dataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
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
    physical_target_fields=None,
    physical_tubelet_size=2,
    physical_misalignment_enabled=False,
    physical_misalignment_mode="none",
    physical_misalignment_seed=0,
    physical_misalignment_shift=1,
    strict_time_sampling=False,
    return_physical_targets=True,
):
    dataset = CoVLATimeJEPADataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        physical_target_fields=physical_target_fields,
        physical_tubelet_size=physical_tubelet_size,
        physical_misalignment_enabled=physical_misalignment_enabled,
        physical_misalignment_mode=physical_misalignment_mode,
        physical_misalignment_seed=physical_misalignment_seed,
        physical_misalignment_shift=physical_misalignment_shift,
        strict_time_sampling=strict_time_sampling,
        return_physical_targets=return_physical_targets,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("CoVLATimeJEPADataset dataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    loader_cls = torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    data_loader = loader_cls(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("CoVLATimeJEPADataset unsupervised data loader created")
    return dataset, data_loader, dist_sampler
