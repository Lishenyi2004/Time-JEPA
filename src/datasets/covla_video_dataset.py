# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import pathlib
import warnings
from logging import getLogger

import torch

from src.datasets.utils.dataloader import (
    ConcatIndices,
    MonitoredDataset,
    NondeterministicDataLoader,
)
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.datasets.video_dataset import VideoDataset

logger = getLogger()


def make_covla_video_dataset(
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
    strict_time_sampling=False,
):
    dataset = CoVLAVideoDataset(
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
        strict_time_sampling=strict_time_sampling,
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

    logger.info("CoVLAVideoDataset dataset created")
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
    logger.info("CoVLAVideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class CoVLAVideoDataset(VideoDataset):
    """CoVLA raw video dataset compatible with the existing V-JEPA collator.

    This first adapter intentionally returns only video clips, labels, and
    sampled frame indices, matching ``VideoDataset``. Frame-level CoVLA JSONL
    annotations can be added later without changing the video-only baseline.
    """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,
        strict_time_sampling=False,
    ):
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Must specify one frames-per-clip value per CoVLA data path")
            self.dataset_fpcs = dataset_fpcs

        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.strict_time_sampling = bool(strict_time_sampling)

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

        samples, labels = [], []
        self.video_ids = []
        self.num_samples_per_dataset = []

        for data_path in self.data_paths:
            dataset_root, metadata_path = self._resolve_covla_paths(data_path)
            dataset_samples, dataset_video_ids = self._load_metadata(dataset_root, metadata_path)
            samples += dataset_samples
            labels += [0] * len(dataset_samples)
            self.video_ids += dataset_video_ids
            self.num_samples_per_dataset.append(len(dataset_samples))

        if not samples:
            raise ValueError(f"No CoVLA videos found from data paths: {self.data_paths}")

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)
        self.sample_weights = None
        if self.datasets_weights is not None:
            if len(self.datasets_weights) != len(self.num_samples_per_dataset):
                raise ValueError("Must specify one CoVLA dataset weight per data path")
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    @staticmethod
    def _resolve_covla_paths(data_path):
        data_path = os.path.abspath(os.path.expanduser(data_path))
        if os.path.isdir(data_path):
            dataset_root = data_path
            metadata_path = os.path.join(dataset_root, "metadata.jsonl")
        else:
            metadata_path = data_path
            dataset_root = os.path.dirname(metadata_path)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"CoVLA metadata file not found: {metadata_path}")
        return dataset_root, metadata_path

    @staticmethod
    def _load_metadata(dataset_root, metadata_path):
        samples, video_ids = [], []
        with open(metadata_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rel_video_path = record.get("file_name")
                video_id = record.get("video_id")
                if not rel_video_path:
                    warnings.warn(f"Skipping CoVLA metadata line {line_no} without file_name")
                    continue
                video_path = rel_video_path
                if not os.path.isabs(video_path):
                    video_path = os.path.join(dataset_root, video_path)
                if not os.path.exists(video_path):
                    warnings.warn(f"Skipping missing CoVLA video: {video_path}")
                    continue
                samples.append(video_path)
                video_ids.append(video_id or os.path.splitext(os.path.basename(video_path))[0])
        return samples, video_ids
