"""Isolated EgoExo target corruptions for sensor-robustness experiments.

The production EgoExo dataset remains unchanged.  This module is used only by
the dedicated ``vjepa_2_1_sensor_robustness`` application.
"""

import hashlib
import math
import pathlib
from logging import getLogger

import numpy as np
import torch

from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.time_jepa.egoexo_pose_targets import build_egoexo_tubelet_targets
from src.time_jepa.egoexo_time_dataset import EgoExoTimeJEPADataset


logger = getLogger(__name__)

EGOEXO_SENSOR_CORRUPTION_MODES = (
    "time_offset",
    "sampling_density",
    "sampling_density_mean_broadcast",
    "gaussian_noise",
    "composite_debug",
)


def normalize_sensor_corruption_config(config):
    """Validate and canonicalize one target-corruption configuration."""
    if not isinstance(config, dict):
        raise TypeError("EgoExo target_corruption must be a mapping")

    mode = str(config.get("mode", "")).strip().lower().replace("-", "_")
    aliases = {
        "offset": "time_offset",
        "temporal_offset": "time_offset",
        "density": "sampling_density",
        "subsample": "sampling_density",
        "density_mean_broadcast": "sampling_density_mean_broadcast",
        "block_mean_broadcast": "sampling_density_mean_broadcast",
        "sampling_density_block_mean": "sampling_density_mean_broadcast",
        "noise": "gaussian_noise",
        "composite": "composite_debug",
    }
    mode = aliases.get(mode, mode)
    if mode not in EGOEXO_SENSOR_CORRUPTION_MODES:
        expected = ", ".join(EGOEXO_SENSOR_CORRUPTION_MODES)
        raise ValueError(
            f"Unsupported EgoExo sensor corruption mode {mode!r}; expected: {expected}"
        )

    offset_seconds = float(config.get("offset_seconds", 0.0))
    sampling_density = float(config.get("sampling_density", 1.0))
    sampling_phase = int(config.get("sampling_phase", 0))
    noise_std = float(config.get("noise_std", 0.0))
    noise_seed = int(config.get("noise_seed", 239))

    if not math.isfinite(offset_seconds):
        raise ValueError("target_corruption.offset_seconds must be finite")
    if not math.isfinite(sampling_density) or not 0.0 < sampling_density <= 1.0:
        raise ValueError("target_corruption.sampling_density must be in (0, 1]")
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("target_corruption.noise_std must be finite and non-negative")

    period = int(round(1.0 / sampling_density))
    if not math.isclose(sampling_density, 1.0 / period, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "target_corruption.sampling_density must be the reciprocal of an integer"
        )
    if not 0 <= sampling_phase < period:
        raise ValueError(
            "target_corruption.sampling_phase must be in [0, 1 / sampling_density)"
        )

    active = {
        "time_offset": not math.isclose(offset_seconds, 0.0, abs_tol=1e-12),
        "sampling_density": not math.isclose(sampling_density, 1.0, abs_tol=1e-12),
        "gaussian_noise": not math.isclose(noise_std, 0.0, abs_tol=1e-12),
    }
    if mode == "composite_debug":
        if not all(active.values()):
            raise ValueError(
                "composite_debug must enable offset, sparse sampling, and noise"
            )
    elif mode == "sampling_density_mean_broadcast":
        if sampling_phase != 0:
            raise ValueError(
                "sampling_density_mean_broadcast currently requires sampling_phase=0"
            )
        enabled_modes = [name for name, enabled in active.items() if enabled]
        if enabled_modes != ["sampling_density"]:
            raise ValueError(
                "mode='sampling_density_mean_broadcast' must enable exactly "
                f"sampling_density; got {enabled_modes}"
            )
    else:
        enabled_modes = [name for name, enabled in active.items() if enabled]
        if enabled_modes != [mode]:
            raise ValueError(
                f"mode={mode!r} must enable exactly that corruption; got {enabled_modes}"
            )

    return {
        "mode": mode,
        "offset_seconds": offset_seconds,
        "sampling_density": sampling_density,
        "sampling_phase": sampling_phase,
        "noise_std": noise_std,
        "noise_seed": noise_seed,
        "noise_realization": "fixed_per_window",
    }


def apply_target_sampling_density(target, valid, density, phase=0):
    """Keep a regular fraction of tubelet targets and mask all other entries."""
    if target.shape != valid.shape or target.ndim != 2:
        raise ValueError("target and valid must have the same [time, dimensions] shape")
    density = float(density)
    period = int(round(1.0 / density))
    if not 0.0 < density <= 1.0 or not math.isclose(
        density, 1.0 / period, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("density must be the reciprocal of a positive integer")
    phase = int(phase)
    if not 0 <= phase < period:
        raise ValueError("phase must be smaller than the sampling period")

    keep = (torch.arange(target.shape[0], device=valid.device) % period) == phase
    sparse_valid = valid * keep[:, None].to(valid.dtype)
    sparse_target = torch.where(
        sparse_valid > 0,
        target,
        torch.zeros_like(target),
    )
    return sparse_target, sparse_valid


def apply_target_sampling_density_mean_broadcast(target, valid, density, phase=0):
    """Average each regular temporal block and broadcast it to valid positions.

    This is the density-matched analogue of A5's clip-mean broadcast control:
    the target is reduced to one value per block, but the resulting value is
    returned at every original valid tubelet position.  Consequently the
    downstream masked loss still averages over the full valid sequence rather
    than reweighting the remaining observations.
    """
    if target.shape != valid.shape or target.ndim != 2:
        raise ValueError("target and valid must have the same [time, dimensions] shape")
    density = float(density)
    period = int(round(1.0 / density))
    if not 0.0 < density <= 1.0 or not math.isclose(
        density, 1.0 / period, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("density must be the reciprocal of a positive integer")
    phase = int(phase)
    if not 0 <= phase < period:
        raise ValueError("phase must be smaller than the sampling period")
    if phase != 0:
        raise ValueError("mean-broadcast density currently requires phase=0")

    if bool(valid.any()) and not bool(torch.isfinite(target[valid > 0]).all()):
        raise ValueError("target contains a non-finite value at a valid position")

    output = torch.zeros_like(target)
    safe_target = torch.where(valid > 0, target, torch.zeros_like(target))
    num_steps = target.shape[0]
    for block_start in range(0, num_steps, period):
        block_end = min(block_start + period, num_steps)
        block_valid = valid[block_start:block_end]
        block_target = safe_target[block_start:block_end]
        counts = block_valid.sum(dim=0)
        means = (block_target * block_valid).sum(dim=0) / counts.clamp_min(1.0)
        output[block_start:block_end] = torch.where(
            block_valid > 0,
            means.unsqueeze(0).expand_as(block_target),
            torch.zeros_like(block_target),
        )
    return output, valid.clone()


def _window_noise_seed(base_seed, window_key):
    payload = f"{int(base_seed)}|{window_key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def add_fixed_standardized_gaussian_noise(
    target,
    valid,
    target_std,
    normalized_std,
    seed,
    window_key,
):
    """Add a reproducible per-window N(0, normalized_std^2) sensor error.

    ``target`` is in raw physical units.  Multiplication by the clean training
    standard deviation makes the requested noise level exact after the normal
    Time-JEPA target normalization performed by the training loop.
    """
    if target.shape != valid.shape or target.ndim != 2:
        raise ValueError("target and valid must have the same [time, dimensions] shape")
    target_std = torch.as_tensor(target_std, dtype=target.dtype, device=target.device)
    if target_std.ndim != 1 or target_std.numel() != target.shape[-1]:
        raise ValueError(
            f"Expected {target.shape[-1]} target std values, got {tuple(target_std.shape)}"
        )
    if not torch.isfinite(target_std).all() or (target_std <= 0).any():
        raise ValueError("target_std must contain finite positive values")
    normalized_std = float(normalized_std)
    if not math.isfinite(normalized_std) or normalized_std < 0.0:
        raise ValueError("normalized_std must be finite and non-negative")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_window_noise_seed(seed, window_key))
    noise = torch.randn(
        target.shape,
        generator=generator,
        dtype=target.dtype,
        device="cpu",
    ).to(target.device)
    corrupted = target + noise * target_std.view(1, -1) * normalized_std
    return torch.where(valid > 0, corrupted, torch.zeros_like(target))


class EgoExoSensorRobustnessDataset(EgoExoTimeJEPADataset):
    """EgoExo video with corruption applied only to fine-tuning targets."""

    def __init__(self, *args, target_corruption, target_std=None, **kwargs):
        temporal_mode = kwargs.get("target_temporal_mode", "tubelet_aligned")
        if str(temporal_mode).strip().lower().replace("-", "_") not in (
            "aligned",
            "tubelet",
            "tubelet_aligned",
        ):
            raise ValueError(
                "Sensor robustness experiments require tubelet-aligned clean targets"
            )
        super().__init__(*args, **kwargs)
        if not self.return_physical_targets:
            raise ValueError("Sensor robustness experiments require physical targets")
        self.target_corruption = normalize_sensor_corruption_config(target_corruption)
        if self.target_corruption["noise_std"] > 0.0:
            if target_std is None:
                raise ValueError("Gaussian target corruption requires clean target_std")
            self.target_std = torch.as_tensor(target_std, dtype=torch.float32)
        else:
            self.target_std = None

    def get_item_video(self, index):
        sample = super().get_item_video(index)
        if sample is None:
            return None
        buffer, label, frame_indices, physical = sample
        row = self.rows[index]
        corruption = self.target_corruption

        video_sample_times = physical["sample_times"][0].cpu().numpy().astype(np.float64)
        target_sample_times = video_sample_times + corruption["offset_seconds"]
        if not math.isclose(corruption["offset_seconds"], 0.0, abs_tol=1e-12):
            target, valid = build_egoexo_tubelet_targets(
                row["pose_path"],
                target_sample_times,
                target_fields=self.target_fields,
                tubelet_size=self.tubelet_size,
                max_time_delta=self.max_time_delta,
                pose_fps=self.pose_fps,
                min_views=self.min_views,
                smooth_window=self.smooth_window,
            )
        else:
            target = physical["target"][0]
            valid = physical["valid"][0]

        if corruption["mode"] in ("sampling_density", "composite_debug"):
            if corruption["sampling_density"] < 1.0:
                target, valid = apply_target_sampling_density(
                    target,
                    valid,
                    density=corruption["sampling_density"],
                    phase=corruption["sampling_phase"],
                )
        elif corruption["mode"] == "sampling_density_mean_broadcast":
            target, valid = apply_target_sampling_density_mean_broadcast(
                target,
                valid,
                density=corruption["sampling_density"],
                phase=corruption["sampling_phase"],
            )
        if corruption["noise_std"] > 0.0:
            target = add_fixed_standardized_gaussian_noise(
                target,
                valid,
                target_std=self.target_std,
                normalized_std=corruption["noise_std"],
                seed=corruption["noise_seed"],
                window_key=row["window_key"],
            )

        physical["target"] = [target]
        physical["valid"] = [valid]
        physical["target_sample_times"] = [
            torch.from_numpy(target_sample_times.astype(np.float32))
        ]
        physical["target_corruption_mode"] = corruption["mode"]
        physical["target_offset_seconds"] = corruption["offset_seconds"]
        physical["target_sampling_density"] = corruption["sampling_density"]
        physical["target_sampling_phase"] = corruption["sampling_phase"]
        physical["target_noise_std"] = corruption["noise_std"]
        physical["target_noise_seed"] = corruption["noise_seed"]
        return buffer, label, frame_indices, physical


def make_egoexo_sensor_robustness_dataset(
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
    egoexo_target_corruption=None,
    egoexo_target_std=None,
):
    if num_clips != 1:
        raise NotImplementedError("EgoExo sensor robustness supports num_clips=1")
    if dataset_fpcs is not None:
        frames_per_clip = int(dataset_fpcs[0])
    root = data_paths[0] if isinstance(data_paths, (list, tuple)) else data_paths
    dataset = EgoExoSensorRobustnessDataset(
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
        target_corruption=egoexo_target_corruption,
        target_std=egoexo_target_std,
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
        "EgoExoSensorRobustnessDataset created rows=%s targets=%s corruption=%s",
        len(dataset),
        egoexo_target_fields,
        dataset.target_corruption,
    )
    return dataset, loader, sampler
