#!/usr/bin/env python3
"""Train eleven independent scalar linear heads on one frozen COVLA encoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

try:
    from .bank_common import (
        REPO_ROOT,
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        load_config,
        seed_everything,
        sha256_file,
        target_manifest,
    )
    from .bank_data import CovlaPhysicsBankDataset, get_or_compute_stats
    from .bank_model import IndependentHeadBank, TinySmokeEncoder
    from .consistency_eval import evaluate_prediction_file
except ImportError:
    from bank_common import (
        REPO_ROOT,
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        load_config,
        seed_everything,
        sha256_file,
        target_manifest,
    )
    from bank_data import CovlaPhysicsBankDataset, get_or_compute_stats
    from bank_model import IndependentHeadBank, TinySmokeEncoder
    from consistency_eval import evaluate_prediction_file

from probe_constants import ARCH_SPECS
from probe_datasets import read_metadata, split_rows_from_files
from probe_engine import load_encoder
from probe_optim import build_optimizer_and_schedulers
from probe_transforms import build_transform
from probe_utils import seed_worker, torch_load
from world_model_probe_rjob.physics_consistency_encoder import (
    WORLD_MODEL_BACKENDS,
    build_physics_input_transform,
    build_world_model_encoder,
)


def init_distributed(device_name: str) -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda but CUDA is unavailable")
    if distributed:
        if device_name == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device(device_name)
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
        "device": device,
    }


def unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def print_main(is_main: bool, *items) -> None:
    if is_main:
        print(*items, flush=True)


def reduce_sum(tensor: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _split_limits(config: dict, split: str) -> int:
    return int(config.get("data", {}).get("limits", {}).get(split, 0) or 0)


def load_protocol_rows(config: dict) -> dict[str, list[dict]]:
    data = config["data"]
    all_rows = read_metadata(data["covla_root"])
    names = []
    for name in [data["train_split"], data["val_split"], *data["eval_splits"]]:
        if name not in names:
            names.append(name)
    loaded = split_rows_from_files(all_rows, data["split_dir"], split_names=tuple(names))
    result = {}
    for name, rows in zip(names, loaded):
        limit = _split_limits(config, name)
        result[name] = rows[:limit] if limit > 0 else rows
        if not result[name]:
            raise RuntimeError(f"Split {name!r} is empty after applying limits")
    train_ids = {row["video_id"] for row in result[data["train_split"]]}
    val_ids = {row["video_id"] for row in result[data["val_split"]]}
    if train_ids & val_ids:
        raise RuntimeError("probe_train/probe_val overlap")
    if config["experiment"]["phase"] == "full":
        final_sets = [{row["video_id"] for row in result[name]} for name in data["eval_splits"]]
        for name, ids in zip(data["eval_splits"], final_sets):
            overlap = (train_ids | val_ids) & ids
            if overlap:
                raise RuntimeError(f"Final split {name} overlaps train/val: {sorted(overlap)[:5]}")
        if len(final_sets) == 2 and final_sets[0] & final_sets[1]:
            raise RuntimeError("ID test and weather OOD test overlap")
    return result


def build_encoder(config: dict, device: torch.device, is_main: bool):
    encoder_config = config["encoder"]
    sampling = config["sampling"]
    backend = encoder_config["backend"]
    if backend == "tiny_smoke":
        embed_dim = int(encoder_config.get("embed_dim", 32))
        encoder = TinySmokeEncoder(embed_dim, sampling["tubelet_size"])
        load_report = {"backend": backend, "checkpoint": None}
    elif backend == "vjepa2_1":
        arch_name = encoder_config["arch"]
        if arch_name not in ARCH_SPECS:
            raise ValueError(f"Unknown encoder arch: {arch_name}")
        arch = ARCH_SPECS[arch_name]
        if int(sampling["num_frames"]) != int(arch["num_frames"]):
            raise ValueError("sampling.num_frames does not match encoder architecture")
        if int(sampling["image_size"]) != int(arch["img_size"]):
            raise ValueError("sampling.image_size does not match encoder architecture")
        checkpoint = Path(encoder_config["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint}")
        encoder = arch["builder"](
            img_size=(arch["img_size"], arch["img_size"]),
            num_frames=arch["num_frames"],
            modality_embedding=True,
            interpolate_rope=True,
        )
        load_report = load_encoder(
            encoder, checkpoint, checkpoint_key=encoder_config.get("checkpoint_key", "auto")
        )
        load_report.update(
            {
                "backend": backend,
                "arch": arch_name,
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
            }
        )
        embed_dim = int(arch["embed_dim"])
    elif backend in WORLD_MODEL_BACKENDS:
        encoder, embed_dim, load_report = build_world_model_encoder(config)
    else:
        raise ValueError(f"Unsupported encoder backend: {backend}")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    print_main(is_main, "encoder:", json.dumps(json_safe(load_report), ensure_ascii=False))
    return encoder, embed_dim, load_report


def build_input_transform(config: dict):
    return build_physics_input_transform(config, build_transform)


def build_loader(
    rows,
    config,
    transform,
    stats,
    dist_info,
    split,
    random_clip=False,
    train=False,
):
    sampling = config["sampling"]
    dataset = CovlaPhysicsBankDataset(
        rows,
        num_frames=sampling["num_frames"],
        transform=transform,
        random_clip=random_clip,
        stats=stats,
        frame_span=sampling.get("frame_span"),
        frame_stride=sampling.get("frame_stride", 0),
        tubelet_size=sampling["tubelet_size"],
        video_num_threads=config["runtime"].get("video_num_threads", 1),
    )
    sampler = None
    eval_dataset = dataset
    if train and dist_info["distributed"]:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_info["world_size"],
            rank=dist_info["rank"],
            shuffle=True,
            seed=int(config["experiment"]["seed"]),
            drop_last=True,
        )
    elif not train and dist_info["distributed"]:
        eval_dataset = Subset(
            dataset,
            list(range(dist_info["rank"], len(dataset), dist_info["world_size"])),
        )
    generator = torch.Generator().manual_seed(
        int(config["experiment"]["seed"]) + (0 if train else 10000) + dist_info["rank"]
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=int(config["optimization"]["batch_size"]),
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=int(config["runtime"].get("num_workers", 4)),
        pin_memory=dist_info["device"].type == "cuda",
        drop_last=train,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if train and len(loader) == 0:
        raise RuntimeError(
            "Training loader has zero batches; lower batch_size or increase the train limit"
        )
    return loader, sampler


def optimizer_args(config: dict) -> SimpleNamespace:
    optimization = config["optimization"]
    return SimpleNamespace(
        warmup_epochs=float(optimization.get("warmup_epochs", 1)),
        start_lr=optimization.get("start_lr"),
        lr=float(optimization["lr"]),
        final_lr=float(optimization.get("final_lr", 1e-6)),
        weight_decay=float(optimization.get("weight_decay", 0.01)),
        final_weight_decay=optimization.get("final_weight_decay"),
        epochs=int(optimization["epochs"]),
        lr_schedule=optimization.get("lr_schedule", "cosine"),
    )


def masked_head_losses(prediction, target, valid):
    losses = []
    for target_index in range(len(TARGET_NAMES)):
        mask = valid[..., target_index]
        error = (prediction[..., target_index] - target[..., target_index]).square() * mask
        losses.append(error.sum() / mask.sum().clamp_min(1.0))
    return torch.stack(losses)


@torch.no_grad()
def validate(encoder, bank, loader, stats, device, distributed, amp_dtype, is_main):
    bank.eval()
    normalized_sse = torch.zeros(len(TARGET_NAMES), device=device, dtype=torch.float64)
    raw_abs = torch.zeros_like(normalized_sse)
    raw_sse = torch.zeros_like(normalized_sse)
    counts = torch.zeros_like(normalized_sse)
    mean, std = [item.to(device) for item in stats]
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            prediction = bank(encoder(video))
        normalized_error = prediction.float() - target.float()
        raw_error = normalized_error * std
        normalized_sse += (normalized_error.square() * valid).sum((0, 1)).double()
        raw_abs += (raw_error.abs() * valid).sum((0, 1)).double()
        raw_sse += (raw_error.square() * valid).sum((0, 1)).double()
        counts += valid.sum((0, 1)).double()
    for tensor in (normalized_sse, raw_abs, raw_sse, counts):
        reduce_sum(tensor, distributed)
    losses = normalized_sse / counts.clamp_min(1)
    return {
        name: {
            "normalized_mse": float(losses[index].item()),
            "raw_mae": float((raw_abs[index] / counts[index].clamp_min(1)).item()),
            "raw_rmse": float(torch.sqrt(raw_sse[index] / counts[index].clamp_min(1)).item()),
            "valid_count": int(counts[index].item()),
        }
        for index, name in enumerate(TARGET_NAMES)
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_head_checkpoint(
    path,
    bank,
    target_name,
    optimizer,
    epoch,
    metric,
    stats,
    config,
    encoder_report,
):
    model = unwrap(bank)
    index = TARGET_NAMES.index(target_name)
    atomic_torch_save(
        {
            "schema": 1,
            "target_name": target_name,
            "target_index": index,
            "head": model.heads[target_name].state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "validation": metric,
            "target_mean": stats[0][index].cpu(),
            "target_std": stats[1][index].cpu(),
            "encoder": encoder_report,
            "config": config,
        },
        path,
    )


def load_best_heads(bank, checkpoint_dir: Path, stats=None) -> dict:
    model = unwrap(bank)
    manifest = {}
    for target_name in TARGET_NAMES:
        path = checkpoint_dir / target_name / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint for {target_name}: {path}")
        checkpoint = torch_load(path, map_location="cpu")
        if checkpoint["target_name"] != target_name:
            raise RuntimeError(f"Checkpoint target mismatch: {path}")
        index = TARGET_NAMES.index(target_name)
        if int(checkpoint.get("target_index", -1)) != index:
            raise RuntimeError(f"Checkpoint target index mismatch: {path}")
        if stats is not None:
            expected_mean = stats[0][index].cpu()
            expected_std = stats[1][index].cpu()
            if not torch.allclose(checkpoint["target_mean"].cpu(), expected_mean, atol=1e-7, rtol=1e-6):
                raise RuntimeError(f"Checkpoint target mean drift: {path}")
            if not torch.allclose(checkpoint["target_std"].cpu(), expected_std, atol=1e-7, rtol=1e-6):
                raise RuntimeError(f"Checkpoint target std drift: {path}")
        model.heads[target_name].load_state_dict(checkpoint["head"], strict=True)
        manifest[target_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "epoch": int(checkpoint["epoch"]),
            "validation": checkpoint["validation"],
        }
    return manifest


def stage_reused_heads(
    source_run: str | Path, run_dir: Path, config: dict, expected_stats_fingerprint: str
) -> dict:
    """Copy an already-approved pilot's seed-0 heads into its full eval run."""
    source_run = Path(source_run).expanduser().resolve()
    completion_path = source_run / "COMPLETED.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"Reusable pilot is not complete: {completion_path}")
    with completion_path.open() as handle:
        completion = json.load(handle)
    expected_model = config["experiment"]["model_name"]
    expected_seed = int(config["experiment"]["seed"])
    if completion.get("phase") != "pilot":
        raise ValueError(f"Only a completed pilot may be promoted, got {completion.get('phase')!r}")
    if completion.get("model_name") != expected_model or int(completion.get("probe_seed", -1)) != expected_seed:
        raise ValueError(
            f"Pilot identity mismatch: expected {expected_model}/seed_{expected_seed}, "
            f"got {completion.get('model_name')}/seed_{completion.get('probe_seed')}"
        )
    source_fingerprint = completion.get("stats", {}).get("fingerprint", {}).get("sha256")
    if source_fingerprint != expected_stats_fingerprint:
        raise ValueError(
            f"Pilot stats fingerprint mismatch: {source_fingerprint} != {expected_stats_fingerprint}"
        )
    destination = run_dir / "checkpoints"
    copied = {}
    for target_name in TARGET_NAMES:
        source = source_run / "checkpoints" / target_name / "best.pt"
        if not source.is_file():
            raise FileNotFoundError(f"Pilot head missing: {source}")
        checkpoint = torch_load(source, map_location="cpu")
        current_checkpoint = str(Path(config["encoder"]["checkpoint"]).expanduser().resolve())
        source_checkpoint = str(Path(checkpoint["encoder"]["checkpoint"]).expanduser().resolve())
        if current_checkpoint != source_checkpoint:
            raise ValueError(
                f"Encoder mismatch while promoting {target_name}: {source_checkpoint} != {current_checkpoint}"
            )
        target = destination / target_name / "best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[target_name] = {"source": str(source), "destination": str(target)}
    source_history = source_run / "history.json"
    if source_history.is_file():
        shutil.copy2(source_history, run_dir / "pilot_history.json")
    return {"source_run": str(source_run), "copied_heads": copied}


@torch.no_grad()
def export_predictions(
    encoder,
    bank,
    loader,
    stats,
    device,
    dist_info,
    config,
    split_name,
    output_path,
):
    bank.eval()
    mean, std = [item.to(device) for item in stats]
    amp_dtype = torch.bfloat16 if config["runtime"].get("use_bfloat16", True) else torch.float16
    local = {
        "video_id": [],
        "prediction": [],
        "target": [],
        "valid": [],
        "target_frame_indices": [],
    }
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        normalized_target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            normalized_prediction = bank(encoder(video))
        raw_prediction = normalized_prediction.float() * std + mean
        raw_target = normalized_target.float() * std + mean
        local["video_id"].extend([str(item) for item in batch["video_id"]])
        local["prediction"].append(raw_prediction.cpu().numpy())
        local["target"].append(raw_target.cpu().numpy())
        local["valid"].append(batch["valid"].cpu().numpy())
        local["target_frame_indices"].append(batch["target_frame_indices"].cpu().numpy())
    for key in ("prediction", "target", "valid", "target_frame_indices"):
        local[key] = np.concatenate(local[key], axis=0)
    if dist_info["distributed"]:
        gathered = [None] * dist_info["world_size"] if dist_info["is_main"] else None
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    if not dist_info["is_main"]:
        return None
    combined = {}
    combined["video_id"] = np.asarray(
        [video_id for shard in gathered for video_id in shard["video_id"]], dtype=str
    )
    for key in ("prediction", "target", "valid", "target_frame_indices"):
        combined[key] = np.concatenate([shard[key] for shard in gathered], axis=0)
    order = np.argsort(combined["video_id"])
    for key in combined:
        combined[key] = combined[key][order]
    if len(set(combined["video_id"].tolist())) != len(combined["video_id"]):
        raise RuntimeError(f"Duplicate videos while exporting {split_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **combined,
        target_names=np.asarray(TARGET_NAMES),
        target_mean=stats[0].cpu().numpy(),
        target_std=stats[1].cpu().numpy(),
        video_fps=np.asarray(float(config["sampling"].get("video_fps", 20.0))),
        split_name=np.asarray(split_name),
        model_name=np.asarray(config["experiment"]["model_name"]),
        seed=np.asarray(int(config["experiment"]["seed"])),
    )
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "video_count": int(len(combined["video_id"])),
        "shape": list(combined["prediction"].shape),
    }


def train(config: dict) -> Path:
    seed = int(config["experiment"]["seed"])
    model_name = config["experiment"]["model_name"]
    run_dir = (
        Path(config["output"]["root"]).expanduser().resolve()
        / model_name
        / f"seed_{seed}"
    )
    if config["experiment"]["phase"] != "smoke" and (run_dir / "COMPLETED.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite a completed formal run: {run_dir}. "
            "Choose a new output root or remove it intentionally after auditing."
        )
    dist_info = init_distributed(config["runtime"].get("device", "cuda"))
    is_main = dist_info["is_main"]
    device = dist_info["device"]
    try:
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(json_safe(config), run_dir / "resolved_config.json")
        rows = load_protocol_rows(config)
        data = config["data"]
        print_main(is_main, "run_dir:", run_dir)
        print_main(is_main, "splits:", {key: len(value) for key, value in rows.items()})
        seed_everything(seed)
        sampling = config["sampling"]
        stats_config = config["stats"]
        stats = get_or_compute_stats(
            rows[data["train_split"]],
            sampling,
            stats_config["cache"],
            seed=int(stats_config.get("seed", 239)),
            max_videos=int(stats_config.get("max_videos", 2000)),
            distributed=dist_info["distributed"],
            rank=dist_info["rank"],
            device=device,
        )
        target_mean, target_std, stats_payload = stats
        stats_pair = (target_mean, target_std)
        print_main(is_main, "target_mean:", target_mean.tolist())
        print_main(is_main, "target_std:", target_std.tolist())
        transform = build_input_transform(config)
        train_loader, train_sampler = build_loader(
            rows[data["train_split"]],
            config,
            transform,
            stats_pair,
            dist_info,
            data["train_split"],
            random_clip=True,
            train=True,
        )
        val_loader, _ = build_loader(
            rows[data["val_split"]],
            config,
            transform,
            stats_pair,
            dist_info,
            data["val_split"],
            random_clip=False,
            train=False,
        )
        encoder, embed_dim, encoder_report = build_encoder(config, device, is_main)
        # Reset after encoder construction/loading so the same probe seed means
        # identical head initialization for all three frozen encoder variants.
        seed_everything(seed)
        bank = IndependentHeadBank(
            embed_dim,
            sampling["num_frames"],
            sampling["tubelet_size"],
            TARGET_NAMES,
        ).to(device)
        bank.assert_disjoint()
        if dist_info["distributed"]:
            ddp_kwargs = {"broadcast_buffers": False}
            if device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[dist_info["local_rank"]], output_device=dist_info["local_rank"]
                )
            bank = DistributedDataParallel(bank, **ddp_kwargs)
        optimization = config["optimization"]
        opt_args = optimizer_args(config)
        optimizers, lr_schedulers, wd_schedulers = {}, {}, {}
        model = unwrap(bank)
        for target_name in TARGET_NAMES:
            optimizer, lr_scheduler, wd_scheduler = build_optimizer_and_schedulers(
                opt_args, model.heads[target_name], len(train_loader)
            )
            optimizers[target_name] = optimizer
            lr_schedulers[target_name] = lr_scheduler
            wd_schedulers[target_name] = wd_scheduler
        amp_dtype = (
            torch.bfloat16 if config["runtime"].get("use_bfloat16", True) else torch.float16
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda" and amp_dtype == torch.float16
        )
        best = {name: math.inf for name in TARGET_NAMES}
        history = []
        checkpoint_dir = run_dir / "checkpoints"
        reuse_report = None
        reuse_source = config["experiment"].get("reuse_heads_from")
        if reuse_source:
            if int(optimization["epochs"]) != 0:
                raise ValueError("reuse_heads_from requires optimization.epochs=0")
            if is_main:
                reuse_report = stage_reused_heads(
                    reuse_source,
                    run_dir,
                    config,
                    stats_payload["fingerprint"]["sha256"],
                )
                atomic_json_dump(json_safe(reuse_report), run_dir / "REUSED_PILOT.json")
            if dist_info["distributed"]:
                dist.barrier()
        for epoch in range(int(optimization["epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            bank.train()
            train_sse = torch.zeros(len(TARGET_NAMES), dtype=torch.float64, device=device)
            train_count = torch.zeros_like(train_sse)
            for batch in train_loader:
                video = batch["video"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
                    ):
                        features = encoder(video)
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
                ):
                    prediction = bank(features.detach())
                    losses = masked_head_losses(prediction, target, valid)
                    # Sum, not mean: each disjoint head receives exactly its own
                    # scalar loss gradient, equivalent to eleven separate probes.
                    total_loss = losses.sum()
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    for optimizer in optimizers.values():
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    for optimizer in optimizers.values():
                        optimizer.step()
                for target_name in TARGET_NAMES:
                    if lr_schedulers[target_name] is not None:
                        lr_schedulers[target_name].step()
                    if wd_schedulers[target_name] is not None:
                        wd_schedulers[target_name].step()
                train_sse += (
                    (prediction.float() - target.float()).square() * valid
                ).sum((0, 1)).double()
                train_count += valid.sum((0, 1)).double()
            reduce_sum(train_sse, dist_info["distributed"])
            reduce_sum(train_count, dist_info["distributed"])
            train_losses = train_sse / train_count.clamp_min(1)
            val_metrics = validate(
                encoder,
                bank,
                val_loader,
                stats_pair,
                device,
                dist_info["distributed"],
                amp_dtype,
                is_main,
            )
            epoch_row = {
                "epoch": epoch + 1,
                "train_normalized_mse": {
                    name: float(train_losses[index].item())
                    for index, name in enumerate(TARGET_NAMES)
                },
                "validation": val_metrics,
            }
            if is_main:
                history.append(epoch_row)
                print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
                atomic_json_dump(json_safe(history), run_dir / "history.json")
                for target_name in TARGET_NAMES:
                    value = val_metrics[target_name]["normalized_mse"]
                    if value < best[target_name]:
                        best[target_name] = value
                        save_head_checkpoint(
                            checkpoint_dir / target_name / "best.pt",
                            bank,
                            target_name,
                            optimizers[target_name],
                            epoch + 1,
                            val_metrics[target_name],
                            stats_pair,
                            config,
                            encoder_report,
                        )
        if dist_info["distributed"]:
            dist.barrier()
        best_manifest = load_best_heads(bank, checkpoint_dir, stats_pair)
        if is_main:
            atomic_json_dump(json_safe(best_manifest), run_dir / "best_heads.json")
        outputs = {}
        for split_name in data["eval_splits"]:
            eval_loader, _ = build_loader(
                rows[split_name],
                config,
                transform,
                stats_pair,
                dist_info,
                split_name,
                random_clip=False,
                train=False,
            )
            prediction_path = run_dir / "predictions" / f"{split_name}.npz"
            export = export_predictions(
                encoder,
                bank,
                eval_loader,
                stats_pair,
                device,
                dist_info,
                config,
                split_name,
                prediction_path,
            )
            if is_main:
                consistency = config["consistency"]
                result = evaluate_prediction_file(
                    prediction_path,
                    consistency["gt_gate_metrics"],
                    run_dir / "metrics",
                    permutations=int(consistency["permutations"]),
                    bootstrap_replicates=int(consistency["bootstrap_replicates"]),
                    seed=seed,
                    split_name=split_name,
                )
                outputs[split_name] = {
                    "predictions": export,
                    "metrics": str(run_dir / "metrics" / f"{split_name}_consistency.json"),
                    "S_coh": result["accuracy_matched_null"]["composite"]["S_coh"],
                }
            if dist_info["distributed"]:
                dist.barrier()
        if is_main:
            completion = {
                "status": "COMPLETE",
                "phase": config["experiment"]["phase"],
                "model_name": model_name,
                "probe_seed": seed,
                "target_manifest": target_manifest(),
                "independent_heads": {
                    "count": len(TARGET_NAMES),
                    "separate_layernorm": True,
                    "separate_linear": True,
                    "separate_optimizer": True,
                    "separate_scheduler": True,
                    "separate_best_checkpoint": True,
                    "shared_trainable_parameters": False,
                    "shared_frozen_encoder_forward": True,
                },
                "stats": stats_payload,
                "best_heads": best_manifest,
                "reused_pilot": reuse_report,
                "outputs": outputs,
            }
            atomic_json_dump(json_safe(completion), run_dir / "COMPLETED.json")
            print(json.dumps({"status": "COMPLETE", "run_dir": str(run_dir)}, indent=2))
        return run_dir
    finally:
        if dist_info["distributed"] and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    parser.add_argument(
        "--reuse-heads-from",
        help="Completed matching pilot run; copies its heads and skips retraining (full seed 0).",
    )
    args = parser.parse_args()
    overrides = {}
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = int(args.seed)
    if args.output_root:
        overrides.setdefault("output", {})["root"] = str(Path(args.output_root).expanduser())
    if args.reuse_heads_from:
        overrides.setdefault("experiment", {})["reuse_heads_from"] = str(
            Path(args.reuse_heads_from).expanduser().resolve()
        )
        overrides.setdefault("optimization", {})["epochs"] = 0
    config = load_config(args.config, overrides)
    train(config)


if __name__ == "__main__":
    main()
