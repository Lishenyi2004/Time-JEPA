import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from src.time_jepa.agibot_derived_targets import (
    agibot_target_definition_version,
    is_agibot_series_derived_field,
)
from src.time_jepa.agibot_time_dataset import (
    compute_agibot_stats,
    normalize_agibot_targets,
    read_agibot_rows,
)
from src.time_jepa.egoexo_pose_targets import (
    EGOEXO_TARGET_DEFINITION_VERSION,
    normalize_egoexo_targets,
)
from src.time_jepa.egoexo_time_dataset import (
    compute_egoexo_stats,
    read_egoexo_rows,
)
from src.time_jepa.egoexo_views import normalize_egoexo_video_views
from src.time_jepa.physical_targets import (
    compute_physical_stats,
    normalize_target_fields,
    read_covla_rows,
)


DOMAIN_DATASET_TYPES = {
    "covla": "covla_time_jepa",
    "agibot": "agibot_time_jepa",
    "egoexo": "egoexo_time_jepa",
}


def normalize_time_jepa_mode(cfg):
    mode = str((cfg or {}).get("mode", "single")).strip().lower()
    aliases = {
        "covla": "single",
        "single_domain": "single",
        "multi": "multi_domain",
        "multidomain": "multi_domain",
    }
    return aliases.get(mode, mode)


def normalize_domains_config(cfg, data_cfg=None):
    cfg = cfg or {}
    mode = normalize_time_jepa_mode(cfg)
    if mode != "multi_domain":
        return mode, {}
    domains = cfg.get("domains")
    if not domains:
        raise ValueError("time_jepa.mode=multi_domain requires time_jepa.domains")
    normalized = {}
    for name, domain_cfg in domains.items():
        domain_cfg = dict(domain_cfg or {})
        domain = str(domain_cfg.get("domain", name)).strip().lower()
        if domain not in DOMAIN_DATASET_TYPES:
            raise ValueError(f"Unsupported Time-JEPA domain {domain!r}")
        domain_cfg["name"] = str(name)
        domain_cfg["domain"] = domain
        domain_cfg.setdefault("dataset_type", DOMAIN_DATASET_TYPES[domain])
        domain_cfg.setdefault("sample_weight", 1.0)
        domain_cfg.setdefault("lambda_time", domain_cfg.get("lambda_phys", cfg.get("lambda_phys", 0.0)))
        domain_cfg.setdefault("stats_max_videos", cfg.get("stats_max_videos", 2000))
        domain_cfg.setdefault("stats_seed", cfg.get("stats_seed", 0))
        domain_cfg.setdefault("smooth_l1_beta", cfg.get("smooth_l1_beta", 1.0))
        normalized[str(name)] = domain_cfg
    return mode, normalized


def domain_target_fields(domain_cfg):
    domain = domain_cfg["domain"]
    if domain == "covla":
        return normalize_target_fields(domain_cfg.get("targets", ["vEgo", "aEgo", "yaw_rate"]))
    if domain == "agibot":
        return normalize_agibot_targets(domain_cfg.get("targets", {"type": "bimanual_compact_v1"}))
    if domain == "egoexo":
        return normalize_egoexo_targets(domain_cfg.get("targets", {"type": "whole_body_compact_v2"}))
    raise ValueError(f"Unsupported Time-JEPA domain {domain!r}")


def domain_root_paths(domain_cfg):
    root = domain_cfg.get("root", domain_cfg.get("datasets", domain_cfg.get("data_path")))
    if root is None:
        raise ValueError(f"Missing root/datasets for domain {domain_cfg['name']}")
    if isinstance(root, (list, tuple)):
        return list(root)
    return [root]


def _rows_fingerprint(rows, key):
    values = sorted(
        f"{row.get(key, '')}|{row.get('window_idx', '')}|{row.get('start_time', '')}"
        for row in rows
    )
    return {
        "row_count": len(values),
        f"{key}_sha1": hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest(),
    }


def _cache_matches(payload, meta):
    cached_meta = payload.get("meta", {})
    return all(cached_meta.get(key) == value for key, value in meta.items())


def _load_matching_stats_cache(cache_path, meta):
    if not cache_path or not cache_path.exists():
        return None
    try:
        with cache_path.open("r") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    return payload if _cache_matches(payload, meta) else None


def _write_stats_cache(cache_path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, cache_path)


def _wait_for_stats_cache(cache_path, meta, timeout_seconds, poll_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        payload = _load_matching_stats_cache(cache_path, meta)
        if payload is not None:
            return payload
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.0f}s waiting for rank 0 to write stats cache: "
                f"{cache_path}"
            )
        time.sleep(poll_seconds)


def _broadcast_stats(mean, std, target_dim, distributed, rank, device):
    if not distributed:
        return mean, std
    if rank != 0:
        mean = torch.empty(target_dim, dtype=torch.float32)
        std = torch.empty(target_dim, dtype=torch.float32)
    comm_device = device if device.type == "cuda" else torch.device("cpu")
    packed = torch.stack([mean, std], dim=0).to(comm_device)
    dist.broadcast(packed, src=0)
    return packed[0].cpu(), packed[1].cpu()


def get_or_compute_domain_stats(
    domain_cfg,
    frames_per_clip,
    tubelet_size,
    fps,
    frame_step=None,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    strict_time_sampling=False,
    cache_path=None,
    distributed=False,
    rank=0,
    device=torch.device("cpu"),
):
    target_fields = domain_target_fields(domain_cfg)
    domain = domain_cfg["domain"]
    cache_path = Path(cache_path or domain_cfg.get("stats_cache", "")).expanduser() if (cache_path or domain_cfg.get("stats_cache")) else None
    max_videos = int(domain_cfg.get("stats_max_videos", 2000))
    seed = int(domain_cfg.get("stats_seed", 0))

    if domain == "covla":
        rows = read_covla_rows(domain_root_paths(domain_cfg))
        row_meta = _rows_fingerprint(rows, "video_id")
        meta = {
            "domain": domain,
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
            "seed": seed,
            "max_videos": max_videos,
            **row_meta,
        }
        compute_fn = lambda: compute_physical_stats(
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
    elif domain == "agibot":
        rows = read_agibot_rows(
            root=domain_root_paths(domain_cfg)[0],
            tasks=domain_cfg.get("tasks", [3641, 3777]),
            video_key=domain_cfg.get("video_key", "observation.images.top_head"),
            use_meta_episodes_only=bool(domain_cfg.get("use_meta_episodes_only", True)),
            max_windows_per_episode=domain_cfg.get("max_windows_per_episode"),
            frames_per_clip=frames_per_clip,
            fps=fps,
            slice_mode=domain_cfg.get("slice_mode", "episode"),
            slice_stride_seconds=domain_cfg.get("slice_stride_seconds"),
            episode_allowlist=domain_cfg.get("episode_allowlist"),
            view_episode_allowlists=domain_cfg.get("view_episode_allowlists"),
        )
        row_meta = _rows_fingerprint(rows, "video_path")
        meta = {
            "domain": domain,
            "target_fields": target_fields,
            "frames_per_clip": int(frames_per_clip),
            "tubelet_size": int(tubelet_size),
            "fps": None if fps is None else int(fps),
            "seed": seed,
            "max_videos": max_videos,
            "video_key": domain_cfg.get("video_key", "observation.images.top_head"),
            "tasks": [str(t) for t in domain_cfg.get("tasks", [3641, 3777])],
            "slice_mode": domain_cfg.get("slice_mode", "episode"),
            "slice_stride_seconds": domain_cfg.get("slice_stride_seconds"),
            **row_meta,
        }
        if any(is_agibot_series_derived_field(field) for field in target_fields):
            meta["target_definition_version"] = agibot_target_definition_version(
                target_fields
            )
        compute_fn = lambda: compute_agibot_stats(
            rows=rows,
            target_fields=target_fields,
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            fps=fps,
            seed=seed,
            max_videos=max_videos,
            max_time_delta=domain_cfg.get("max_time_delta"),
        )
    elif domain == "egoexo":
        egoexo_video_views = domain_cfg.get("video_views", domain_cfg.get("video_view", "best_exo"))
        normalized_egoexo_video_views = list(normalize_egoexo_video_views(egoexo_video_views))
        egoexo_view_meta = (
            {"video_views": normalized_egoexo_video_views}
            if "video_views" in domain_cfg
            else {"video_view": normalized_egoexo_video_views[0]}
        )
        rows = read_egoexo_rows(
            root=domain_root_paths(domain_cfg)[0],
            video_view=egoexo_video_views,
            frames_per_clip=frames_per_clip,
            fps=fps,
            slice_stride_seconds=domain_cfg.get("slice_stride_seconds", 12.8),
            episode_allowlist=domain_cfg.get("episode_allowlist"),
            manifest_path=domain_cfg.get("manifest_path"),
            split_prefix=domain_cfg.get("split_prefix", "egoexo_bestexo"),
        )
        row_meta = _rows_fingerprint(rows, "video_path")
        meta = {
            "domain": domain,
            "target_fields": target_fields,
            "frames_per_clip": int(frames_per_clip),
            "tubelet_size": int(tubelet_size),
            "fps": float(fps),
            "seed": seed,
            "max_videos": max_videos,
            **egoexo_view_meta,
            "target_definition_version": EGOEXO_TARGET_DEFINITION_VERSION,
            "slice_stride_seconds": domain_cfg.get("slice_stride_seconds", 12.8),
            **row_meta,
        }
        compute_fn = lambda: compute_egoexo_stats(
            rows=rows,
            target_fields=target_fields,
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            fps=fps,
            seed=seed,
            max_videos=max_videos,
            max_time_delta=domain_cfg.get("max_time_delta", 0.05),
            pose_fps=domain_cfg.get("pose_fps", 30.0),
            min_views=domain_cfg.get("min_views", 2),
            smooth_window=domain_cfg.get("smooth_window", 5),
        )
    else:
        raise ValueError(f"Unsupported Time-JEPA domain {domain!r}")

    use_cache_sync = distributed and cache_path is not None
    if not distributed or rank == 0:
        payload = _load_matching_stats_cache(cache_path, meta)
        if payload is None:
            mean, std, counts = compute_fn()
            payload = {
                "meta": meta,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "counts": counts.tolist(),
            }
            if cache_path:
                _write_stats_cache(cache_path, payload)
    elif use_cache_sync:
        payload = _wait_for_stats_cache(
            cache_path=cache_path,
            meta=meta,
            timeout_seconds=float(domain_cfg.get("stats_cache_wait_seconds", 7200)),
            poll_seconds=float(domain_cfg.get("stats_cache_poll_seconds", 2)),
        )
    else:
        payload = None

    if use_cache_sync or not distributed:
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        return mean, std

    if rank != 0:
        mean = torch.empty(len(target_fields), dtype=torch.float32)
        std = torch.empty(len(target_fields), dtype=torch.float32)
    else:
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)

    mean, std = _broadcast_stats(
        mean=mean,
        std=std,
        target_dim=len(target_fields),
        distributed=distributed,
        rank=rank,
        device=device,
    )
    return mean, std


def sample_domain_name(domain_names, weights, device, distributed, rank):
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if weights.sum() <= 0:
        raise ValueError("Domain sampling weights must sum to a positive value")
    weights = weights / weights.sum()
    if (not distributed) or rank == 0:
        idx = torch.multinomial(weights, num_samples=1).long()
    else:
        idx = torch.empty(1, dtype=torch.long, device=device)
    if distributed:
        dist.broadcast(idx, src=0)
    return domain_names[int(idx.item())]


def normalize_domain_sampling_config(time_jepa_cfg, default_seed=0):
    """Normalize the optional exact multi-domain scheduling configuration."""
    raw_cfg = (time_jepa_cfg or {}).get("domain_sampling", {}) or {}
    if isinstance(raw_cfg, str):
        raw_cfg = {"mode": raw_cfg}
    if not isinstance(raw_cfg, dict):
        raise ValueError("time_jepa.domain_sampling must be a mapping or mode string")

    mode = str(raw_cfg.get("mode", "stochastic")).strip().lower()
    mode = {
        "random": "stochastic",
        "weighted_random": "stochastic",
        "exact": "quota",
        "fixed_quota": "quota",
    }.get(mode, mode)
    if mode not in ("stochastic", "quota"):
        raise ValueError(
            "time_jepa.domain_sampling.mode must be one of: stochastic, quota"
        )

    epoch_size = raw_cfg.get("epoch_size", "min_loader")
    if isinstance(epoch_size, str):
        epoch_size = epoch_size.strip().lower()
        epoch_size = {
            "min": "min_loader",
            "shortest": "min_loader",
            "max": "max_loader",
            "longest": "max_loader",
            "sum": "sum_loaders",
            "all": "sum_loaders",
        }.get(epoch_size, epoch_size)
        if epoch_size not in ("min_loader", "max_loader", "sum_loaders"):
            try:
                epoch_size = int(epoch_size)
            except ValueError as exc:
                raise ValueError(
                    "time_jepa.domain_sampling.epoch_size must be a positive integer "
                    "or one of: min_loader, max_loader, sum_loaders"
                ) from exc
    if isinstance(epoch_size, bool) or (
        not isinstance(epoch_size, (str, int))
    ):
        raise ValueError(
            "time_jepa.domain_sampling.epoch_size must be a positive integer "
            "or one of: min_loader, max_loader, sum_loaders"
        )
    if isinstance(epoch_size, int) and epoch_size <= 0:
        raise ValueError("time_jepa.domain_sampling.epoch_size must be positive")

    lambda_warmup_unit = str(
        raw_cfg.get("lambda_warmup_unit", "global_step")
    ).strip().lower()
    if lambda_warmup_unit not in ("global_step", "domain_step"):
        raise ValueError(
            "time_jepa.domain_sampling.lambda_warmup_unit must be one of: "
            "global_step, domain_step"
        )
    if lambda_warmup_unit == "domain_step" and mode != "quota":
        raise ValueError(
            "domain_step lambda warmup requires domain_sampling.mode=quota so "
            "completed per-domain steps can be reconstructed exactly on resume"
        )

    quota_basis = str(raw_cfg.get("quota_basis", "weights")).strip().lower()
    quota_basis = {
        "weight": "weights",
        "loader": "loader_lengths",
        "loaders": "loader_lengths",
    }.get(quota_basis, quota_basis)
    if quota_basis not in ("weights", "loader_lengths"):
        raise ValueError(
            "time_jepa.domain_sampling.quota_basis must be one of: "
            "weights, loader_lengths"
        )
    if quota_basis == "loader_lengths" and mode != "quota":
        raise ValueError(
            "loader_lengths quota_basis requires domain_sampling.mode=quota"
        )

    return {
        "mode": mode,
        "epoch_size": epoch_size,
        "shuffle": bool(raw_cfg.get("shuffle", True)),
        "seed": int(raw_cfg.get("seed", default_seed)),
        "lambda_warmup_unit": lambda_warmup_unit,
        "quota_basis": quota_basis,
    }


def resolve_domain_epoch_size(domain_loader_lengths, epoch_size):
    """Resolve a world-size-aware epoch budget after data loaders are built."""
    positive_lengths = {
        name: int(length)
        for name, length in domain_loader_lengths.items()
        if length is not None and int(length) > 0
    }
    if len(positive_lengths) != len(domain_loader_lengths):
        raise ValueError(
            "All multi-domain loaders must expose a positive length to use "
            f"domain_sampling.epoch_size; got {domain_loader_lengths}"
        )
    if isinstance(epoch_size, int):
        return epoch_size
    values = list(positive_lengths.values())
    if epoch_size == "min_loader":
        return min(values)
    if epoch_size == "max_loader":
        return max(values)
    if epoch_size == "sum_loaders":
        return sum(values)
    raise ValueError(f"Unsupported multi-domain epoch_size: {epoch_size!r}")


def compute_domain_step_quotas(domain_names, weights, total_steps, epoch=0):
    """Return exact integer step counts using rotating largest remainders."""
    if len(domain_names) != len(weights):
        raise ValueError("domain_names and weights must have the same length")
    if not domain_names:
        raise ValueError("At least one domain is required")
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    weight_tensor = torch.tensor(weights, dtype=torch.float64)
    if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
        raise ValueError("Domain sampling weights must be finite and non-negative")
    weight_sum = float(weight_tensor.sum())
    if weight_sum <= 0:
        raise ValueError("Domain sampling weights must sum to a positive value")

    exact = weight_tensor / weight_sum * total_steps
    floors = torch.floor(exact).to(torch.int64)
    quotas = floors.tolist()
    remaining = total_steps - sum(quotas)
    fractions = (exact - floors).tolist()
    num_domains = len(domain_names)
    rotation = int(epoch) % num_domains
    remainder_order = sorted(
        range(num_domains),
        key=lambda idx: (-fractions[idx], (idx - rotation) % num_domains),
    )
    for idx in remainder_order[:remaining]:
        quotas[idx] += 1
    return {name: int(quota) for name, quota in zip(domain_names, quotas)}


def build_domain_epoch_schedule(
    domain_names,
    weights,
    total_steps,
    epoch,
    seed,
    shuffle,
    device,
    distributed,
    rank,
):
    """Build one exact schedule on rank 0 and broadcast it to every DDP rank."""
    quotas = compute_domain_step_quotas(
        domain_names=domain_names,
        weights=weights,
        total_steps=total_steps,
        epoch=epoch,
    )
    total_steps = int(total_steps)
    if (not distributed) or rank == 0:
        schedule_cpu = torch.tensor(
            [
                domain_idx
                for domain_idx, domain_name in enumerate(domain_names)
                for _ in range(quotas[domain_name])
            ],
            dtype=torch.long,
        )
        if shuffle and total_steps > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + int(epoch) * 1_000_003)
            schedule_cpu = schedule_cpu[
                torch.randperm(total_steps, generator=generator)
            ]
        schedule = schedule_cpu.to(device)
    else:
        schedule = torch.empty(total_steps, dtype=torch.long, device=device)
    if distributed:
        dist.broadcast(schedule, src=0)
    schedule_names = [domain_names[idx] for idx in schedule.cpu().tolist()]
    return schedule_names, quotas
