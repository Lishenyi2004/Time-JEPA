# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data="ImageNet",
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
    subset_file=None,
    clip_len=None,
    dataset_fpcs=None,
    frame_sample_rate=None,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    datasets_weights=None,
    persistent_workers=False,
    deterministic=True,
    log_dir=None,
    physical_target_fields=None,
    physical_tubelet_size=2,
    physical_misalignment_enabled=False,
    physical_misalignment_mode="none",
    physical_misalignment_seed=0,
    physical_misalignment_shift=1,
    strict_time_sampling=False,
    agibot_tasks=None,
    agibot_video_key="observation.images.top_head",
    agibot_target_fields=None,
    agibot_slice_mode="random",
    agibot_slice_stride_seconds=None,
    agibot_max_windows_per_episode=None,
    agibot_use_meta_episodes_only=True,
    agibot_max_time_delta=None,
    agibot_video_backend="auto",
    agibot_ffmpeg_timeout=120,
    return_physical_targets=True,
    agibot_episode_allowlist=None,
    agibot_view_episode_allowlists=None,
    agibot_video_keys=None,
    egoexo_video_view="best_exo",
    egoexo_target_fields=None,
    egoexo_slice_stride_seconds=12.8,
    egoexo_max_time_delta=0.05,
    egoexo_pose_fps=30.0,
    egoexo_min_views=2,
    egoexo_smooth_window=5,
    egoexo_video_backend="auto",
    egoexo_ffmpeg_timeout=120,
    egoexo_episode_allowlist=None,
    egoexo_manifest_path=None,
    egoexo_split_prefix="egoexo_bestexo",
    egoexo_target_temporal_mode="tubelet_aligned",
):
    if data.lower() == "imagenet":
        from src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            subset_file=subset_file,
        )

    elif data.lower() == "videodataset":
        from src.datasets.video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
            strict_time_sampling=strict_time_sampling,
        )

    elif data.lower() in ("covlavideodataset", "covla_video", "covla"):
        from src.datasets.covla_video_dataset import make_covla_video_dataset

        dataset, data_loader, dist_sampler = make_covla_video_dataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
            strict_time_sampling=strict_time_sampling,
        )

    elif data.lower() in ("covlatimejepadataset", "covla_time_jepa", "covla_time"):
        from src.time_jepa.covla_time_dataset import make_covla_time_jepa_dataset

        dataset, data_loader, dist_sampler = make_covla_time_jepa_dataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
            physical_target_fields=physical_target_fields,
            physical_tubelet_size=physical_tubelet_size,
            physical_misalignment_enabled=physical_misalignment_enabled,
            physical_misalignment_mode=physical_misalignment_mode,
            physical_misalignment_seed=physical_misalignment_seed,
            physical_misalignment_shift=physical_misalignment_shift,
            strict_time_sampling=strict_time_sampling,
            return_physical_targets=return_physical_targets,
        )

    elif data.lower() in ("agibottimejepadataset", "agibot_time_jepa", "agibot_time"):
        from src.time_jepa.agibot_time_dataset import make_agibot_time_jepa_dataset

        if agibot_video_keys is not None:
            raise NotImplementedError(
                "AgiBot multi-view training is handled at the application/config level; "
                "data_manager.init_data expects a single agibot_video_key"
            )
        dataset, data_loader, dist_sampler = make_agibot_time_jepa_dataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
            agibot_tasks=agibot_tasks,
            agibot_video_key=agibot_video_key,
            agibot_target_fields=agibot_target_fields,
            physical_tubelet_size=physical_tubelet_size,
            agibot_slice_mode=agibot_slice_mode,
            agibot_slice_stride_seconds=agibot_slice_stride_seconds,
            agibot_max_windows_per_episode=agibot_max_windows_per_episode,
            agibot_use_meta_episodes_only=agibot_use_meta_episodes_only,
            agibot_max_time_delta=agibot_max_time_delta,
            agibot_video_backend=agibot_video_backend,
            agibot_ffmpeg_timeout=agibot_ffmpeg_timeout,
            return_physical_targets=return_physical_targets,
            agibot_episode_allowlist=agibot_episode_allowlist,
            agibot_view_episode_allowlists=agibot_view_episode_allowlists,
        )

    elif data.lower() in ("egoexotimejepadataset", "egoexo_time_jepa", "egoexo_time"):
        from src.time_jepa.egoexo_time_dataset import make_egoexo_time_jepa_dataset

        dataset, data_loader, dist_sampler = make_egoexo_time_jepa_dataset(
            data_paths=root_path, batch_size=batch_size, frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs, fps=fps, num_clips=num_clips, transform=transform,
            shared_transform=shared_transform, rank=rank, world_size=world_size,
            datasets_weights=datasets_weights, collator=collator, drop_last=drop_last,
            num_workers=num_workers, pin_mem=pin_mem, persistent_workers=persistent_workers,
            deterministic=deterministic, log_dir=log_dir, egoexo_video_view=egoexo_video_view,
            egoexo_target_fields=egoexo_target_fields, physical_tubelet_size=physical_tubelet_size,
            egoexo_slice_stride_seconds=egoexo_slice_stride_seconds,
            egoexo_max_time_delta=egoexo_max_time_delta, egoexo_pose_fps=egoexo_pose_fps,
            egoexo_min_views=egoexo_min_views, egoexo_smooth_window=egoexo_smooth_window,
            egoexo_video_backend=egoexo_video_backend, egoexo_ffmpeg_timeout=egoexo_ffmpeg_timeout,
            return_physical_targets=return_physical_targets,
            egoexo_episode_allowlist=egoexo_episode_allowlist,
            egoexo_manifest_path=egoexo_manifest_path, egoexo_split_prefix=egoexo_split_prefix,
            egoexo_target_temporal_mode=egoexo_target_temporal_mode,
        )

    return (data_loader, dist_sampler)
