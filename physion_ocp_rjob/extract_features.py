from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from physion_ocp_rjob.config import (
    load_config,
    protocol_digest,
    protocol_payload,
    validate_config,
)
from physion_ocp_rjob.dataset import PhysionOCPDataset
from physion_ocp_rjob.feature_store import (
    H5FeatureShardWriter,
    atomic_json_dump,
    check_stale_shards,
    matching_complete_shard,
    shard_path,
    task_directory,
)
from physion_ocp_rjob.model import feature_dimension, initialize_encoder


LOGGER = logging.getLogger("physion_ocp_extract")


def rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise RuntimeError(f"Invalid rank/world_size: {rank}/{world_size}")
    return rank, world_size, local_rank


def seed_everything(seed: int, rank: int) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def _build_dataset(config: dict[str, Any], task: dict[str, str]) -> PhysionOCPDataset:
    extraction = config["extraction"]
    split = task["split"]
    max_samples = extraction.get(
        "max_train_samples" if split == "train" else "max_test_samples"
    )
    return PhysionOCPDataset(
        dataset_root=config["dataset_root"],
        split=split,
        temporal_order=task["temporal_order"],
        resolution=int(config["model"]["resolution"]),
        frame_gap_ms=int(extraction["frame_gap_ms"]),
        num_frames=int(extraction["num_frames"]),
        seed=int(extraction["seed"]),
        max_samples=max_samples,
        decord_num_threads=int(extraction["decord_num_threads"]),
    )


def _batch_metadata(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_index": batch["global_index"].cpu().numpy(),
        "label": batch["label"].cpu().numpy(),
        "contact_frame": batch["contact_frame"].cpu().numpy(),
        "frame_indices": batch["frame_indices"].cpu().numpy(),
        "name": list(batch["name"]),
        "scenario": list(batch["scenario"]),
        "relative_path": list(batch["relative_path"]),
    }


@torch.inference_mode()
def _extract_task(
    *,
    config: dict[str, Any],
    task: dict[str, str],
    dataset: PhysionOCPDataset,
    model: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    digest: str,
    overwrite: bool,
) -> Path:
    extraction = config["extraction"]
    split = task["split"]
    temporal_order = task["temporal_order"]
    destination = shard_path(
        config["output_dir"], split, temporal_order, rank, world_size
    )
    indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, indices)
    num_workers = int(extraction["num_workers"])
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(extraction["batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(subset, **loader_kwargs)

    expected_dim = feature_dimension(config, embed_dim=int(model.embed_dim))
    attributes = {
        "protocol_digest": digest,
        "split": split,
        "temporal_order": temporal_order,
        "aggregation": extraction["aggregation"],
        "rank": rank,
        "world_size": world_size,
        "dataset_size": len(dataset),
        "full_split_size": dataset.full_size,
        "raw_split_size": dataset.raw_size,
        "checkpoint": str(Path(config["checkpoint"]).expanduser().resolve()),
        "model_name": config["name"],
    }
    writer = H5FeatureShardWriter(
        destination,
        row_count=len(indices),
        feature_dim=expected_dim,
        num_frames=int(extraction["num_frames"]),
        feature_dtype=str(extraction["feature_dtype"]),
        attributes=attributes,
    )
    started = time.time()
    try:
        for step, batch in enumerate(loader, start=1):
            clips = batch["clip"].to(device, non_blocking=True)
            use_bfloat16 = bool(extraction["use_bfloat16"])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                features = model(clips)
            features = features.float()
            if not bool(torch.isfinite(features).all()):
                raise FloatingPointError(
                    f"Non-finite features in {split}/{temporal_order}, step {step}"
                )
            storage_dtype = (
                torch.float16
                if extraction["feature_dtype"] == "float16"
                else torch.float32
            )
            feature_array = features.to(storage_dtype).cpu().numpy()
            writer.write(feature_array, _batch_metadata(batch))
            if step == 1 or step % 25 == 0 or writer.position == writer.row_count:
                LOGGER.info(
                    "rank=%d task=%s/%s rows=%d/%d elapsed=%.1fs",
                    rank,
                    split,
                    temporal_order,
                    writer.position,
                    writer.row_count,
                    time.time() - started,
                )
            del clips, features, feature_array
        writer.handle.attrs["elapsed_seconds"] = time.time() - started
        return writer.finish()
    except BaseException:
        writer.close_failed()
        raise


def run(config: dict[str, Any], *, overwrite: bool = False) -> list[Path]:
    rank, world_size, local_rank = rank_info()
    digest = protocol_digest(config)
    extraction = config["extraction"]
    tasks_and_datasets: list[tuple[dict[str, str], PhysionOCPDataset]] = []
    pending: list[tuple[dict[str, str], PhysionOCPDataset]] = []
    completed_paths: list[Path] = []
    for task in extraction["tasks"]:
        dataset = _build_dataset(config, task)
        tasks_and_datasets.append((task, dataset))
        directory = task_directory(
            config["output_dir"], task["split"], task["temporal_order"]
        )
        directory.mkdir(parents=True, exist_ok=True)
        check_stale_shards(directory, world_size)
        destination = shard_path(
            config["output_dir"],
            task["split"],
            task["temporal_order"],
            rank,
            world_size,
        )
        if matching_complete_shard(
            destination,
            digest=digest,
            split=task["split"],
            temporal_order=task["temporal_order"],
            rank=rank,
            world_size=world_size,
            dataset_size=len(dataset),
        ) and not overwrite:
            LOGGER.info("Matching shard already complete: %s", destination)
            completed_paths.append(destination)
        elif destination.exists() and not overwrite:
            raise RuntimeError(
                f"Existing shard {destination} is incomplete or belongs to another "
                "protocol. Use a new output_dir, or pass --overwrite after checking it."
            )
        else:
            pending.append((task, dataset))

    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for ViT-G Physion feature extraction. Use the "
                "provided rjob launcher in the jepa environment."
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} GPUs "
                "are visible"
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if bool(extraction["use_bfloat16"]) and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"GPU {torch.cuda.get_device_name(device)} does not support BF16"
            )
        seed_everything(int(extraction["seed"]), rank)
        torch.backends.cudnn.benchmark = True
        LOGGER.info(
            "rank=%d/%d device=%s gpu=%s protocol=%s pending_tasks=%d",
            rank,
            world_size,
            device,
            torch.cuda.get_device_name(device),
            digest,
            len(pending),
        )
        model = initialize_encoder(config, device)
        for task, dataset in pending:
            path = _extract_task(
                config=config,
                task=task,
                dataset=dataset,
                model=model,
                device=device,
                rank=rank,
                world_size=world_size,
                digest=digest,
                overwrite=overwrite,
            )
            completed_paths.append(path)

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    if rank == 0:
        atomic_json_dump(
            {
                "format_version": 1,
                "protocol_digest": digest,
                "protocol": protocol_payload(config),
                "world_size": world_size,
                "tasks": [
                    {
                        **task,
                        "dataset_size": len(dataset),
                        "expected_shards": [
                            f"rank_{value:05d}_of_{world_size:05d}.h5"
                            for value in range(world_size)
                        ],
                    }
                    for task, dataset in tasks_and_datasets
                ],
            },
            output_dir / "feature_manifest.json",
        )
        resolved = {key: value for key, value in config.items() if key != "config_path"}
        resolved_path = output_dir / "resolved_config.yaml"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved_path.with_name(f".{resolved_path.name}.tmp-p{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)
        os.replace(temporary, resolved_path)
    return completed_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Physion OCP YAML config")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    for key in ("max_train_samples", "max_test_samples", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            config["extraction"][key] = value
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    validate_config(config)
    run(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
