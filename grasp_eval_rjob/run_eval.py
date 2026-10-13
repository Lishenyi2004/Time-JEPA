from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from grasp_eval_rjob.config import (
    load_config,
    protocol_digest,
    protocol_payload,
    validate_config,
)
from grasp_eval_rjob.dataset import GRASPPairDataset, descriptor_dict
from intphys2_eval_rjob.model import VJEPA21PredictiveModel, initialize_model


LOGGER = logging.getLogger("grasp_eval")
SHARD_RE = re.compile(r"rank_(\d+)_of_(\d+)\.pt$")
CODE_FILES = (
    "constants.py",
    "config.py",
    "dataset.py",
    "run_eval.py",
    "score.py",
    "compare_models.py",
)


def _rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world_size:
        raise RuntimeError(f"Invalid rank/world_size: {rank}/{world_size}")
    return rank, world_size, local_rank


def _seed_everything(seed: int, rank: int) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-r{os.environ.get('RANK', '0')}-p{os.getpid()}"
    )
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _atomic_json_dump(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-p{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, destination)


def _atomic_yaml_dump(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-p{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    os.replace(temporary, destination)


def _check_stale_shards(raw_dir: Path, world_size: int) -> None:
    stale = []
    for path in raw_dir.glob("rank_*_of_*.pt"):
        match = SHARD_RE.search(path.name)
        if match and int(match.group(2)) != world_size:
            stale.append(path.name)
    if stale:
        raise RuntimeError(
            "Output directory contains shards from another world size. Move them aside "
            f"before this run: {stale[:12]}"
        )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: _sha256_file(root / name) for name in CODE_FILES}


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=True
    )


@torch.inference_mode()
def evaluate_pair(
    *,
    model: VJEPA21PredictiveModel,
    clips: torch.Tensor,
    evaluation: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Evaluate a P/IP pair and retain every fixed-context window score."""
    if clips.ndim != 5 or clips.shape[0] != 2:
        raise ValueError(f"Expected clips [2,C,T,H,W], got {tuple(clips.shape)}")
    sampled_frames = int(clips.shape[2])
    frames_per_clip = int(evaluation["frames_per_clip"])
    stride = int(evaluation["stride"])
    context_frames = int(evaluation["context_frames"])
    chunk_size = int(evaluation["chunk_size"])
    if sampled_frames < frames_per_clip:
        raise RuntimeError(
            f"Pair has {sampled_frames} sampled frames, fewer than {frames_per_clip}"
        )
    starts = list(range(0, sampled_frames - frames_per_clip + 1, stride))
    # Ordering is video-major and then temporal, matching torch.unfold in the
    # released evaluator after flatten(0, 1).
    windows = [
        clips[video : video + 1, :, start : start + frames_per_clip]
        for video in range(2)
        for start in starts
    ]
    losses: list[float] = []
    use_bfloat16 = bool(evaluation["use_bfloat16"])
    for offset in range(0, len(windows), chunk_size):
        batch = torch.cat(windows[offset : offset + chunk_size], dim=0).to(
            device, non_blocking=True
        )
        with _autocast_context(device, use_bfloat16):
            target = model.encode_target(batch)
            scores = model.surprise_from_target(batch, target, context_frames)
        losses.extend(float(value) for value in scores.float().cpu().tolist())
        del batch, target, scores

    per_video = torch.tensor(losses, dtype=torch.float32).reshape(2, len(starts))
    if not torch.isfinite(per_video).all():
        raise RuntimeError("Non-finite GRASP surprise value")
    return {
        "window_start": torch.tensor(starts, dtype=torch.int16),
        "prediction_start": torch.tensor(
            [value + context_frames for value in starts], dtype=torch.int16
        ),
        "possible_losses": per_video[0],
        "impossible_losses": per_video[1],
    }


def run(config: dict[str, Any], *, overwrite: bool = False) -> Path:
    rank, world_size, local_rank = _rank_info()
    evaluation = config["evaluation"]
    digest = protocol_digest(config)
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _check_stale_shards(raw_dir, world_size)
    shard_path = raw_dir / f"rank_{rank:05d}_of_{world_size:05d}.pt"

    if shard_path.is_file() and not overwrite:
        existing = torch.load(shard_path, map_location="cpu", weights_only=False)
        if existing.get("protocol_digest") == digest:
            LOGGER.info("Matching shard already complete: %s", shard_path)
            return shard_path
        raise RuntimeError(
            f"Existing shard {shard_path} has protocol "
            f"{existing.get('protocol_digest')}, expected {digest}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ViT-G GRASP evaluation")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} GPUs visible"
        )
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if bool(evaluation["use_bfloat16"]) and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {torch.cuda.get_device_name(device)} lacks BF16")

    _seed_everything(int(evaluation["seed"]), rank)
    torch.backends.cudnn.benchmark = True
    LOGGER.info(
        "rank=%d/%d device=%s gpu=%s protocol=%s",
        rank,
        world_size,
        device,
        torch.cuda.get_device_name(device),
        digest,
    )

    dataset = GRASPPairDataset(
        dataset_root=config["dataset_root"],
        frame_step=int(evaluation["frame_step"]),
        resolution=int(config["model"]["resolution"]),
        frames_per_clip=int(evaluation["frames_per_clip"]),
        max_pairs=evaluation.get("max_pairs"),
        decord_num_threads=int(evaluation["decord_num_threads"]),
    )
    indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, indices)
    num_workers = int(evaluation["num_workers"])
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    LOGGER.info("dataset=%d pairs; rank shard=%d pairs", len(dataset), len(indices))

    model = initialize_model(config, device)
    records: list[dict[str, Any]] = []
    started = time.time()
    for local_index, batch in enumerate(loader):
        item_started = time.time()
        dataset_index = indices[local_index]
        descriptor = dataset.pairs[dataset_index]
        clips = batch["clips"].squeeze(0)
        output = evaluate_pair(
            model=model,
            clips=clips,
            evaluation=evaluation,
            device=device,
        )
        records.append(
            {
                **descriptor_dict(descriptor),
                "decoded_frames": batch["decoded_frames"].squeeze(0).cpu(),
                "sampled_frames": batch["sampled_frames"].squeeze(0).cpu(),
                "fps": batch["fps"].squeeze(0).cpu(),
                **output,
            }
        )
        LOGGER.info(
            "rank=%d pair=%d/%d id=%s windows=%d elapsed=%.1fs",
            rank,
            local_index + 1,
            len(indices),
            descriptor.pair_id,
            len(output["window_start"]),
            time.time() - item_started,
        )

    payload = {
        "format_version": 1,
        "protocol_digest": digest,
        "protocol": protocol_payload(config),
        "rank": rank,
        "world_size": world_size,
        "dataset_size": len(dataset),
        "indices": indices,
        "records": records,
        "elapsed_seconds": time.time() - started,
        "code_sha256": _code_sha256(),
    }
    _atomic_torch_save(payload, shard_path)
    if rank == 0:
        _atomic_json_dump(
            {
                "format_version": 1,
                "protocol_digest": digest,
                "protocol": protocol_payload(config),
                "world_size": world_size,
                "code_sha256": _code_sha256(),
                "expected_shards": [
                    f"rank_{value:05d}_of_{world_size:05d}.pt"
                    for value in range(world_size)
                ],
            },
            output_dir / "run_manifest.json",
        )
        resolved = {key: value for key, value in config.items() if key != "config_path"}
        _atomic_yaml_dump(resolved, output_dir / "resolved_config.yaml")
    LOGGER.info("Saved %s", shard_path)
    return shard_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    if args.max_pairs is not None:
        config["evaluation"]["max_pairs"] = args.max_pairs
    if args.num_workers is not None:
        config["evaluation"]["num_workers"] = args.num_workers
    if args.chunk_size is not None:
        config["evaluation"]["chunk_size"] = args.chunk_size
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    validate_config(config)
    run(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
