#!/usr/bin/env python3
"""Train 20 independent AgiBot q/p scalar probes on one frozen encoder pass."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        load_config,
        seed_everything,
        sha256_file,
        target_manifest,
    )
    from .bank_data import AgiBotPhysicsBankDataset, get_or_compute_stats, load_protocol_rows
    from .bank_model import IndependentHeadBank, TinySmokeEncoder
    from .grounded_fk_eval import evaluate_prediction_file
except ImportError:
    from bank_common import (
        TARGET_NAMES,
        atomic_json_dump,
        json_safe,
        load_config,
        seed_everything,
        sha256_file,
        target_manifest,
    )
    from bank_data import AgiBotPhysicsBankDataset, get_or_compute_stats, load_protocol_rows
    from bank_model import IndependentHeadBank, TinySmokeEncoder
    from grounded_fk_eval import evaluate_prediction_file

from probe_constants import ARCH_SPECS
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


def verify_gt_gate(config: dict) -> dict:
    path = Path(config["kinematics"]["gt_gate_metrics"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"GT gate is required before probe training: {path}. Run gt_fk_gate.py first."
        )
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"GT gate status is not PASS: {path}")
    urdf = Path(config["kinematics"]["urdf"]).expanduser().resolve()
    if str(urdf) != str(Path(payload["urdf"]["path"]).resolve()):
        raise RuntimeError("Training URDF path differs from the passed GT gate")
    actual_sha = sha256_file(urdf)
    if actual_sha != payload["urdf"]["sha256"]:
        raise RuntimeError("Training URDF content differs from the passed GT gate")
    for arm in ("left", "right"):
        if "workspace" not in payload["arms"][arm] or "active_thresholds" not in payload["arms"][arm]:
            raise RuntimeError("GT gate predates frozen workspace/active thresholds; rerun it")
    return {"path": str(path), "sha256": sha256_file(path), "payload": payload}


def build_encoder(config: dict, device: torch.device, is_main: bool):
    encoder_config = config["encoder"]
    sampling = config["sampling"]
    backend = encoder_config["backend"]
    if backend == "tiny_smoke":
        embed_dim = int(encoder_config.get("embed_dim", 32))
        encoder = TinySmokeEncoder(embed_dim, sampling["tubelet_size"])
        report = {"backend": backend, "checkpoint": None}
    elif backend == "vjepa2_1":
        arch_name = encoder_config["arch"]
        if arch_name not in ARCH_SPECS:
            raise ValueError(f"Unknown encoder architecture: {arch_name}")
        arch = ARCH_SPECS[arch_name]
        if int(sampling["num_frames"]) != int(arch["num_frames"]):
            raise ValueError("sampling.num_frames does not match the encoder architecture")
        if int(sampling["image_size"]) != int(arch["img_size"]):
            raise ValueError("sampling.image_size does not match the encoder architecture")
        checkpoint = Path(encoder_config["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint}")
        encoder = arch["builder"](
            img_size=(arch["img_size"], arch["img_size"]),
            num_frames=arch["num_frames"],
            modality_embedding=True,
            interpolate_rope=True,
        )
        report = load_encoder(
            encoder,
            checkpoint,
            checkpoint_key=encoder_config.get("checkpoint_key", "auto"),
        )
        report.update(
            {
                "backend": backend,
                "arch": arch_name,
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
            }
        )
        # These checkpoints are about 17 GB.  Avoid a second full read in every
        # job unless an expected digest is explicitly frozen in the config;
        # load_encoder already reads the complete file once.
        expected_sha = encoder_config.get("sha256")
        if expected_sha:
            actual_sha = sha256_file(checkpoint)
            if actual_sha != expected_sha:
                raise RuntimeError(f"Encoder SHA256 mismatch: {actual_sha} != {expected_sha}")
            report["checkpoint_sha256"] = actual_sha
        embed_dim = int(arch["embed_dim"])
    elif backend in WORLD_MODEL_BACKENDS:
        encoder, embed_dim, report = build_world_model_encoder(config)
    else:
        raise ValueError(f"Unsupported encoder backend: {backend}")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    print_main(is_main, "encoder:", json.dumps(json_safe(report), ensure_ascii=False))
    return encoder, embed_dim, report


def build_input_transform(config: dict):
    return build_physics_input_transform(config, build_transform)


def build_loader(
    rows: list[dict],
    config: dict,
    transform,
    stats,
    dist_info: dict,
    random_clip: bool = False,
    train: bool = False,
):
    dataset = AgiBotPhysicsBankDataset(
        rows,
        config["sampling"],
        transform,
        random_clip=random_clip,
        stats=stats,
        data_config={**config["data"], **config.get("runtime", {})},
    )
    sampler = None
    loader_dataset = dataset
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
        loader_dataset = Subset(
            dataset,
            list(range(dist_info["rank"], len(dataset), dist_info["world_size"])),
        )
    generator = torch.Generator().manual_seed(
        int(config["experiment"]["seed"]) + (0 if train else 10000) + dist_info["rank"]
    )
    loader = DataLoader(
        loader_dataset,
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
        raise RuntimeError("Training loader has zero batches; lower batch_size or increase data")
    return loader, sampler


def optimizer_args(config: dict) -> SimpleNamespace:
    opt = config["optimization"]
    return SimpleNamespace(
        warmup_epochs=float(opt.get("warmup_epochs", 1)),
        start_lr=opt.get("start_lr"),
        lr=float(opt["lr"]),
        final_lr=float(opt.get("final_lr", 1e-6)),
        weight_decay=float(opt.get("weight_decay", 0.01)),
        final_weight_decay=opt.get("final_weight_decay"),
        epochs=int(opt["epochs"]),
        lr_schedule=opt.get("lr_schedule", "cosine"),
    )


def masked_head_losses(prediction, target, valid) -> torch.Tensor:
    losses = []
    for index in range(len(TARGET_NAMES)):
        mask = valid[..., index]
        error = (prediction[..., index] - target[..., index]).square() * mask
        losses.append(error.sum() / mask.sum().clamp_min(1.0))
    return torch.stack(losses)


@torch.no_grad()
def validate(encoder, bank, loader, stats, device, distributed, amp_dtype):
    bank.eval()
    normalized_sse = torch.zeros(len(TARGET_NAMES), device=device, dtype=torch.float64)
    raw_abs = torch.zeros_like(normalized_sse)
    raw_sse = torch.zeros_like(normalized_sse)
    counts = torch.zeros_like(normalized_sse)
    _, std = [item.to(device) for item in stats]
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
    return {
        name: {
            "normalized_mse": float((normalized_sse[index] / counts[index].clamp_min(1)).item()),
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
    path: Path,
    bank,
    target_name: str,
    optimizer,
    epoch: int,
    metric: dict,
    stats,
    config: dict,
    encoder_report: dict,
) -> None:
    index = TARGET_NAMES.index(target_name)
    atomic_torch_save(
        {
            "schema": 1,
            "target_name": target_name,
            "target_index": index,
            "head": unwrap(bank).heads[target_name].state_dict(),
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


def load_best_heads(bank, checkpoint_dir: Path, stats) -> dict:
    model = unwrap(bank)
    manifest = {}
    for index, target_name in enumerate(TARGET_NAMES):
        path = checkpoint_dir / target_name / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint for {target_name}: {path}")
        checkpoint = torch_load(path, map_location="cpu")
        if checkpoint["target_name"] != target_name or int(checkpoint["target_index"]) != index:
            raise RuntimeError(f"Checkpoint target identity mismatch: {path}")
        if not torch.allclose(checkpoint["target_mean"].cpu(), stats[0][index].cpu()):
            raise RuntimeError(f"Checkpoint target mean drift: {path}")
        if not torch.allclose(checkpoint["target_std"].cpu(), stats[1][index].cpu()):
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
    source_run: str | Path,
    run_dir: Path,
    config: dict,
    expected_stats_fingerprint: str,
) -> dict:
    """Promote an approved seed-0 pilot bank without retraining it."""

    source_run = Path(source_run).expanduser().resolve()
    completion_path = source_run / "COMPLETED.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"Reusable pilot is incomplete: {completion_path}")
    completion = json.loads(completion_path.read_text())
    expected_model = config["experiment"]["model_name"]
    expected_seed = int(config["experiment"]["seed"])
    if completion.get("phase") != "pilot" or completion.get("status") != "COMPLETE":
        raise ValueError(f"Only a completed pilot may be promoted: {source_run}")
    if completion.get("model_name") != expected_model or int(completion.get("probe_seed", -1)) != expected_seed:
        raise ValueError("Pilot model/seed identity differs from the destination run")
    source_stats = completion.get("stats", {}).get("fingerprint", {}).get("sha256")
    if source_stats != expected_stats_fingerprint:
        raise ValueError(f"Pilot target statistics differ: {source_stats} != {expected_stats_fingerprint}")
    copied = {}
    current_encoder = str(Path(config["encoder"]["checkpoint"]).expanduser().resolve())
    for name in TARGET_NAMES:
        source = source_run / "checkpoints" / name / "best.pt"
        checkpoint = torch_load(source, map_location="cpu")
        source_encoder = str(Path(checkpoint["encoder"]["checkpoint"]).expanduser().resolve())
        if source_encoder != current_encoder:
            raise ValueError(f"Encoder mismatch while promoting {name}")
        destination = run_dir / "checkpoints" / name / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[name] = {"source": str(source), "destination": str(destination)}
    history = source_run / "history.json"
    if history.is_file():
        shutil.copy2(history, run_dir / "pilot_history.json")
    return {"source_run": str(source_run), "copied_heads": copied}


def _empty_export_shard() -> dict:
    return {
        key: []
        for key in (
            "video_id",
            "episode_key",
            "window_key",
            "task_id",
            "view_key",
            "prediction",
            "target",
            "valid",
            "target_frame_indices",
            "target_sample_times",
        )
    }


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
    output_path: Path,
):
    bank.eval()
    mean, std = [item.to(device) for item in stats]
    amp_dtype = torch.bfloat16 if config["runtime"].get("use_bfloat16", False) else torch.float16
    local = _empty_export_shard()
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        normalized_target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            normalized_prediction = bank(encoder(video))
        local["prediction"].append((normalized_prediction.float() * std + mean).cpu().numpy())
        local["target"].append((normalized_target.float() * std + mean).cpu().numpy())
        local["valid"].append(batch["valid"].cpu().numpy())
        local["target_frame_indices"].append(batch["target_frame_indices"].cpu().numpy())
        tubelet = int(config["sampling"]["tubelet_size"])
        times = batch["sample_times"].reshape(len(video), -1, tubelet).mean(dim=-1)
        local["target_sample_times"].append(times.cpu().numpy())
        local["video_id"].extend(str(item) for item in batch["video_id"])
        local["episode_key"].extend(str(item) for item in batch["episode_key"])
        local["window_key"].extend(str(item) for item in batch["window_key"])
        local["task_id"].extend(int(item) for item in batch["task_id"])
        local["view_key"].extend(str(item) for item in batch["view_key"])
    tensor_keys = (
        "prediction",
        "target",
        "valid",
        "target_frame_indices",
        "target_sample_times",
    )
    for key in tensor_keys:
        local[key] = np.concatenate(local[key], axis=0)
    if dist_info["distributed"]:
        gathered = [None] * dist_info["world_size"] if dist_info["is_main"] else None
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    if not dist_info["is_main"]:
        return None
    string_keys = ("video_id", "episode_key", "window_key", "view_key")
    combined = {
        key: np.asarray([value for shard in gathered for value in shard[key]], dtype=str)
        for key in string_keys
    }
    combined["task_id"] = np.asarray(
        [value for shard in gathered for value in shard["task_id"]], dtype=np.int64
    )
    for key in tensor_keys:
        combined[key] = np.concatenate([shard[key] for shard in gathered], axis=0)
    order = np.argsort(combined["video_id"])
    for key in combined:
        combined[key] = combined[key][order]
    if len(set(combined["video_id"].tolist())) != len(combined["video_id"]):
        raise RuntimeError(f"Duplicate video/window rows while exporting {split_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **combined,
        target_names=np.asarray(TARGET_NAMES),
        target_mean=stats[0].cpu().numpy(),
        target_std=stats[1].cpu().numpy(),
        split_name=np.asarray(split_name),
        model_name=np.asarray(config["experiment"]["model_name"]),
        seed=np.asarray(int(config["experiment"]["seed"])),
        sampling_fps=np.asarray(float(config["sampling"]["fps"])),
    )
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "row_count": int(len(combined["video_id"])),
        "shape": list(combined["prediction"].shape),
    }


def train(config: dict) -> Path:
    seed = int(config["experiment"]["seed"])
    model_name = config["experiment"]["model_name"]
    run_dir = Path(config["output"]["root"]).expanduser().resolve() / model_name / f"seed_{seed}"
    if config["experiment"]["phase"] != "smoke" and (run_dir / "COMPLETED.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed formal run: {run_dir}")
    gate_report = verify_gt_gate(config)
    dist_info = init_distributed(config["runtime"].get("device", "cuda"))
    is_main, device = dist_info["is_main"], dist_info["device"]
    try:
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(json_safe(config), run_dir / "resolved_config.json")
        protocol = load_protocol_rows(config)
        print_main(
            is_main,
            "rows:",
            {
                "train": len(protocol["train"]),
                "val": len(protocol["val"]),
                **{name: len(rows) for name, rows in protocol["eval"].items()},
            },
        )
        seed_everything(seed)
        stats_cfg = config["stats"]
        target_mean, target_std, stats_payload = get_or_compute_stats(
            protocol["train"],
            config["sampling"],
            stats_cfg["cache"],
            seed=int(stats_cfg.get("seed", 239)),
            max_rows=int(stats_cfg.get("max_rows", 0)),
            distributed=dist_info["distributed"],
            rank=dist_info["rank"],
            device=device,
        )
        stats = (target_mean, target_std)
        transform = build_input_transform(config)
        train_loader, train_sampler = build_loader(
            protocol["train"], config, transform, stats, dist_info, random_clip=True, train=True
        )
        val_loader, _ = build_loader(protocol["val"], config, transform, stats, dist_info)
        encoder, embed_dim, encoder_report = build_encoder(config, device, is_main)
        seed_everything(seed)
        probe_config = config.get("probe", {})
        head_type = str(probe_config.get("type", "independent_linear"))
        hidden_dim = int(probe_config.get("hidden_dim", 0))
        bank = IndependentHeadBank(
            embed_dim,
            config["sampling"]["num_frames"],
            config["sampling"]["tubelet_size"],
            TARGET_NAMES,
            head_type=head_type,
            hidden_dim=hidden_dim,
        ).to(device)
        bank.assert_disjoint()
        architecture_report = bank.architecture_report()
        print_main(is_main, "probe:", json.dumps(json_safe(architecture_report), ensure_ascii=False))
        if dist_info["distributed"]:
            kwargs = {"broadcast_buffers": False}
            if device.type == "cuda":
                kwargs.update(device_ids=[dist_info["local_rank"]], output_device=dist_info["local_rank"])
            bank = DistributedDataParallel(bank, **kwargs)
        opt_args = optimizer_args(config)
        optimizers, lr_schedulers, wd_schedulers = {}, {}, {}
        for name in TARGET_NAMES:
            optimizer, lr_scheduler, wd_scheduler = build_optimizer_and_schedulers(
                opt_args, unwrap(bank).heads[name], len(train_loader)
            )
            optimizers[name] = optimizer
            lr_schedulers[name] = lr_scheduler
            wd_schedulers[name] = wd_scheduler
        amp_dtype = torch.bfloat16 if config["runtime"].get("use_bfloat16", False) else torch.float16
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
        )
        best = {name: math.inf for name in TARGET_NAMES}
        history = []
        checkpoint_dir = run_dir / "checkpoints"
        reuse_report = None
        reuse_source = config["experiment"].get("reuse_heads_from")
        if reuse_source:
            if head_type != "independent_linear":
                raise ValueError("reuse_heads_from is only supported for independent_linear")
            if int(config["optimization"]["epochs"]) != 0:
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
        for epoch in range(int(config["optimization"]["epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            bank.train()
            train_sse = torch.zeros(len(TARGET_NAMES), dtype=torch.float64, device=device)
            train_count = torch.zeros_like(train_sse)
            for batch in train_loader:
                video = batch["video"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
                ):
                    features = encoder(video)
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    prediction = bank(features.detach())
                    losses = masked_head_losses(prediction, target, valid)
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
                for name in TARGET_NAMES:
                    if lr_schedulers[name] is not None:
                        lr_schedulers[name].step()
                    if wd_schedulers[name] is not None:
                        wd_schedulers[name].step()
                train_sse += ((prediction.float() - target.float()).square() * valid).sum((0, 1)).double()
                train_count += valid.sum((0, 1)).double()
            reduce_sum(train_sse, dist_info["distributed"])
            reduce_sum(train_count, dist_info["distributed"])
            val_metrics = validate(
                encoder, bank, val_loader, stats, device, dist_info["distributed"], amp_dtype
            )
            epoch_record = {
                "epoch": epoch + 1,
                "train_normalized_mse": {
                    name: float((train_sse[index] / train_count[index].clamp_min(1)).item())
                    for index, name in enumerate(TARGET_NAMES)
                },
                "validation": val_metrics,
            }
            if is_main:
                history.append(epoch_record)
                print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
                atomic_json_dump(json_safe(history), run_dir / "history.json")
                for name in TARGET_NAMES:
                    value = val_metrics[name]["normalized_mse"]
                    if value < best[name]:
                        best[name] = value
                        save_head_checkpoint(
                            checkpoint_dir / name / "best.pt",
                            bank,
                            name,
                            optimizers[name],
                            epoch + 1,
                            val_metrics[name],
                            stats,
                            config,
                            encoder_report,
                        )
        if dist_info["distributed"]:
            dist.barrier()
        best_manifest = load_best_heads(bank, checkpoint_dir, stats)
        if is_main:
            atomic_json_dump(json_safe(best_manifest), run_dir / "best_heads.json")
        outputs = {}
        for split_name, eval_rows in protocol["eval"].items():
            eval_loader, _ = build_loader(eval_rows, config, transform, stats, dist_info)
            prediction_path = run_dir / "predictions" / f"{split_name}.npz"
            export = export_predictions(
                encoder, bank, eval_loader, stats, device, dist_info, config, split_name, prediction_path
            )
            if is_main:
                grounded = evaluate_prediction_file(
                    prediction_path,
                    config["kinematics"]["gt_gate_metrics"],
                    run_dir / "metrics",
                    split_name=split_name,
                )
                outputs[split_name] = {
                    "predictions": export,
                    "metrics": grounded["metrics_path"],
                    "left_fk_mean_m": grounded["arms"]["left"]["grounded_fk_to_gt"].get("mean_euclidean_m"),
                    "right_fk_mean_m": grounded["arms"]["right"]["grounded_fk_to_gt"].get("mean_euclidean_m"),
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
                "probe": architecture_report,
                "independent_heads": {
                    "count": len(TARGET_NAMES),
                    "head_type": head_type,
                    "hidden_dim": architecture_report["hidden_dim"],
                    "separate_layernorm": head_type == "independent_linear",
                    "separate_linear": head_type == "independent_linear",
                    "separate_conv1x1": head_type == "independent_conv1x1",
                    "separate_optimizer": True,
                    "separate_scheduler": True,
                    "separate_best_checkpoint": True,
                    "shared_trainable_parameters": False,
                    "shared_frozen_encoder_forward": True,
                },
                "exploratory": head_type != "independent_linear",
                "interpretation_guard": (
                    "Independent Conv1x1 probe-capacity ablation; all 20 heads remain "
                    "parameter-disjoint and independently checkpointed."
                    if head_type == "independent_conv1x1"
                    else None
                ),
                "gt_gate": {key: gate_report[key] for key in ("path", "sha256")},
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
    parser.add_argument("--reuse-heads-from")
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
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
