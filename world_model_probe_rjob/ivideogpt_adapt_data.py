"""Aligned COVLA/AgiBot/EgoExo loaders for iVideoGPT adaptation."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.time_jepa.agibot_time_dataset import (
    AgiBotTimeJEPADataset,
    normalize_agibot_targets,
)
from src.time_jepa.covla_time_dataset import CoVLATimeJEPADataset
from src.time_jepa.egoexo_pose_targets import normalize_egoexo_targets
from src.time_jepa.egoexo_time_dataset import EgoExoTimeJEPADataset
from src.time_jepa.physical_targets import normalize_target_fields


def sha1_lines(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_split(path: str) -> List[str]:
    with Path(path).open("r") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Expected non-empty split list: {path}")
    values = list(dict.fromkeys(str(value) for value in values))
    return values


def select_agibot_allowlist(
    split_keys: List[str], total_samples: int, video_keys: List[str], seed: int
) -> List[str]:
    """Match the established 15k V-JEPA protocol exactly.

    The existing launcher samples 5,000 of the 7,000 encoder windows with
    seed 239, then exposes each selected window through three camera views.
    """

    if total_samples % len(video_keys):
        raise ValueError("total_samples must be divisible by number of views")
    selected_count = total_samples // len(video_keys)
    if selected_count > len(split_keys):
        raise ValueError("AgiBot split is smaller than the requested sample budget")
    if selected_count == len(split_keys):
        selected = list(split_keys)
    else:
        selected = random.Random(int(seed)).sample(list(split_keys), selected_count)
    return sorted(selected)


def build_input_transform(input_size: int = 64):
    return video_transforms.Compose(
        [
            video_transforms.Resize(
                (int(input_size), int(input_size)), interpolation="bilinear"
            ),
            volume_transforms.ClipToTensor(),
        ]
    )


def _resolve_covla_metadata(root: str) -> Tuple[Path, Path]:
    root_path = Path(root).expanduser().resolve()
    if root_path.is_dir():
        return root_path, root_path / "metadata.jsonl"
    return root_path.parent, root_path


def write_filtered_covla_metadata(
    root: str, split_path: str, output_path: str
) -> Dict[str, object]:
    dataset_root, metadata_path = _resolve_covla_metadata(root)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    split_ids = load_split(split_path)
    allowed = set(split_ids)
    records = []
    found = set()
    with metadata_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            video_id = str(record.get("video_id", ""))
            if video_id not in allowed:
                continue
            video_path = Path(record.get("file_name", ""))
            if not video_path.is_absolute():
                video_path = dataset_root / video_path
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            normalized = dict(record)
            normalized["video_id"] = video_id
            normalized["file_name"] = str(video_path.resolve())
            records.append(normalized)
            found.add(video_id)
    missing = sorted(allowed - found)
    if missing:
        raise RuntimeError(
            f"{len(missing)} COVLA encoder IDs were not found; first={missing[:5]}"
        )
    if len(records) != len(split_ids):
        raise RuntimeError(
            f"Filtered COVLA rows={len(records)} but split IDs={len(split_ids)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, output_path)
    return {
        "source_metadata": str(metadata_path),
        "filtered_metadata": str(output_path.resolve()),
        "num_samples": len(records),
        "split_sha1": sha1_lines(split_ids),
    }


def normalized_target_fields(cfg: Dict) -> List[str]:
    domain = cfg["data"]["domain"]
    targets = cfg["physical"].get("targets")
    if not cfg["physical"]["enabled"]:
        return []
    if domain == "covla":
        return normalize_target_fields(targets)
    if domain == "agibot":
        return normalize_agibot_targets(targets)
    if domain == "egoexo":
        return normalize_egoexo_targets(targets)
    raise ValueError(domain)


class IVideoGPTAdaptDataset(torch.utils.data.Dataset):
    """Strip V-JEPA-specific nesting while preserving aligned targets."""

    def __init__(self, dataset, include_physical: bool):
        self.dataset = dataset
        self.include_physical = bool(include_physical)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        clips = item[0]
        if not isinstance(clips, (list, tuple)) or len(clips) != 1:
            raise RuntimeError("iVideoGPT adaptation requires exactly one 64-frame clip")
        video = clips[0].float().contiguous()
        if tuple(video.shape[:2]) != (3, 64):
            raise RuntimeError(f"Unexpected video shape: {tuple(video.shape)}")
        if tuple(video.shape[-2:]) != (64, 64):
            raise RuntimeError(f"Unexpected spatial shape: {tuple(video.shape[-2:])}")
        result = {"video": video}
        if self.include_physical:
            if len(item) < 4:
                raise RuntimeError("Time mode dataset did not return physical targets")
            physical = item[3]
            target = physical["target"][0].float().contiguous()
            valid = physical["valid"][0].float().contiguous()
            if target.shape != valid.shape or target.shape[0] != 32:
                raise RuntimeError(
                    f"Unexpected physical target/valid shapes: {target.shape}, {valid.shape}"
                )
            result["target"] = target
            result["valid"] = valid
        return result


def collate_adapt_batch(samples: List[Dict[str, torch.Tensor]]):
    batch = {"video": torch.stack([sample["video"] for sample in samples])}
    if "target" in samples[0]:
        batch["target"] = torch.stack([sample["target"] for sample in samples])
        batch["valid"] = torch.stack([sample["valid"] for sample in samples])
    return batch


def _load_stats(cfg: Dict, fields: List[str]) -> Dict[str, object]:
    if not cfg["physical"]["enabled"]:
        return {}
    path = Path(cfg["physical"]["stats_cache"])
    with path.open("r") as handle:
        stats = json.load(handle)
    stats_fields = list(stats.get("meta", {}).get("target_fields", []))
    if stats_fields != fields:
        raise RuntimeError(
            f"Physical stats fields do not match configured targets: {stats_fields} != {fields}"
        )
    if len(stats.get("mean", [])) != len(fields) or len(stats.get("std", [])) != len(fields):
        raise RuntimeError("Physical stats dimensions do not match configured targets")
    if any(float(value) <= 0 for value in stats["std"]):
        raise RuntimeError("Physical stats contain non-positive standard deviations")
    return stats


def build_adapt_dataset(cfg: Dict, filtered_covla_metadata: str | None = None):
    data = cfg["data"]
    optimization = cfg["optimization"]
    include_physical = cfg["physical"]["enabled"]
    fields = normalized_target_fields(cfg)
    transform = build_input_transform(cfg["model"]["input_size"])
    split_keys = load_split(data["encoder_train_split"])
    domain = data["domain"]

    if domain == "covla":
        if not filtered_covla_metadata:
            raise ValueError("COVLA requires a prepared filtered metadata path")
        base = CoVLATimeJEPADataset(
            data_paths=[filtered_covla_metadata],
            datasets_weights=None,
            frames_per_clip=64,
            dataset_fpcs=[64],
            frame_step=None,
            duration=None,
            fps=5.0,
            num_clips=1,
            random_clip_sampling=True,
            allow_clip_overlap=False,
            transform=transform,
            shared_transform=None,
            strict_time_sampling=True,
            physical_target_fields=fields,
            physical_tubelet_size=2,
            return_physical_targets=include_physical,
        )
        expected_samples = int(data["expected_encoder_train_videos"])
        selected_keys = split_keys
    elif domain == "egoexo":
        base = EgoExoTimeJEPADataset(
            root=data["root"],
            video_view=data["video_views"],
            frames_per_clip=64,
            fps=5.0,
            target_fields=fields,
            tubelet_size=2,
            transform=transform,
            shared_transform=None,
            random_clip_sampling=False,
            slice_stride_seconds=float(data.get("slice_stride_seconds", 12.8)),
            max_time_delta=float(data.get("max_time_delta", 0.05)),
            pose_fps=float(data.get("pose_fps", 30.0)),
            min_views=int(data.get("min_views", 2)),
            smooth_window=int(data.get("smooth_window", 5)),
            video_backend=data.get("video_backend", "decord"),
            ffmpeg_timeout=int(data.get("ffmpeg_timeout", 120)),
            return_physical_targets=include_physical,
            episode_allowlist=split_keys,
            manifest_path=data["manifest_path"],
            split_prefix=data.get("split_prefix", "egoexo_exo124_view3ood"),
        )
        expected_samples = int(data["expected_encoder_train_windows"])
        selected_keys = split_keys
    elif domain == "agibot":
        video_keys = list(data["video_keys"])
        selected_keys = select_agibot_allowlist(
            split_keys,
            total_samples=int(data["encoder_train_num_samples"]),
            video_keys=video_keys,
            seed=int(optimization["seed"]),
        )
        base = AgiBotTimeJEPADataset(
            root=data["root"],
            tasks=data["tasks"],
            video_keys=video_keys,
            frames_per_clip=64,
            fps=5.0,
            target_fields=fields,
            tubelet_size=2,
            transform=transform,
            shared_transform=None,
            random_clip_sampling=False,
            slice_mode="sliding",
            slice_stride_seconds=float(data.get("slice_stride_seconds", 12.8)),
            max_windows_per_episode=data.get("max_windows_per_episode"),
            use_meta_episodes_only=bool(data.get("use_meta_episodes_only", True)),
            max_time_delta=float(data.get("max_time_delta", 0.05)),
            video_backend=data.get("video_backend", "ffmpeg"),
            ffmpeg_timeout=int(data.get("ffmpeg_timeout", 120)),
            return_physical_targets=include_physical,
            episode_allowlist=selected_keys,
        )
        expected_samples = 15000
    else:  # pragma: no cover - config validation prevents this
        raise ValueError(domain)

    if len(base) != expected_samples:
        raise RuntimeError(
            f"{domain} dataset has {len(base)} samples, expected {expected_samples}"
        )
    stats = _load_stats(cfg, fields)
    protocol = {
        "domain": domain,
        "num_samples": len(base),
        "encoder_train_split": data["encoder_train_split"],
        "encoder_train_split_size": len(split_keys),
        "encoder_train_split_sha1": sha1_lines(split_keys),
        "selected_key_count": len(selected_keys),
        "selected_key_sha1": sha1_lines(selected_keys),
        "video_keys": data.get("video_keys", data.get("video_views")),
        "target_fields": fields,
        "stats_cache": cfg["physical"].get("stats_cache"),
    }
    return IVideoGPTAdaptDataset(base, include_physical), stats, protocol

