#!/usr/bin/env python3
"""Train independent q14/v6 probes and evaluate arm reach-rate consistency."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from physics_consistency.agibot.bank_model import IndependentHeadBank
from physics_consistency.agibot.train_independent_head_bank import (
    build_encoder,
    build_input_transform,
    init_distributed,
    optimizer_args,
    print_main,
    reduce_sum,
    unwrap,
)
from physics_consistency.agibot_reach_rate.common import (
    ARMS,
    PROTOCOL,
    TARGET_DEFINITION,
    TARGET_NAMES,
    atomic_json_dump,
    config_fingerprint,
    json_safe,
    load_config,
    seed_everything,
    sha256_file,
    shoulder_points_from_gate,
    target_manifest,
)
from physics_consistency.agibot_reach_rate.data import (
    AgiBotReachRateDataset,
    get_or_compute_stats,
    load_protocol_rows,
)
from physics_consistency.agibot_reach_rate.grounded_eval import (
    evaluate_prediction_file,
)

from probe_optim import build_optimizer_and_schedulers
from probe_utils import seed_worker, torch_load


def verify_gt_gate(config: dict) -> dict:
    path = Path(config["kinematics"]["gt_gate_metrics"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"GT Gate is required before probe training: {path}"
        )
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Reach-rate GT Gate is not PASS: {path}")
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("Training protocol differs from the GT Gate")
    if payload.get("target_definition") != TARGET_DEFINITION:
        raise RuntimeError("Training targets differ from the GT Gate")

    urdf = Path(config["kinematics"]["urdf"]).expanduser().resolve()
    if urdf != Path(payload["urdf"]["path"]).expanduser().resolve():
        raise RuntimeError("Training URDF path differs from the GT Gate")
    if sha256_file(urdf) != payload["urdf"]["sha256"]:
        raise RuntimeError("Training URDF content differs from the GT Gate")
    calibration = Path(payload["calibration_path"]).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"Missing frozen Gate calibration: {calibration}")
    if payload.get("calibration_sha256") and (
        sha256_file(calibration) != payload["calibration_sha256"]
    ):
        raise RuntimeError("Frozen Gate calibration content changed")
    shoulder_points_from_gate(payload)
    for arm in ARMS:
        stats = payload.get("arms", {}).get(arm, {}).get(
            "reach_rate_train_stats", {}
        )
        required = {
            "mean_mps",
            "std_mps",
            "mean_abs_mps",
            "active_abs_q75_mps",
        }
        if not required.issubset(stats):
            raise RuntimeError(f"GT Gate lacks frozen {arm} reach-rate stats")
    gate_sampling = payload.get("fingerprint", {}).get("payload", {}).get(
        "sampling", {}
    )
    for key in ("num_frames", "fps", "tubelet_size", "max_time_delta"):
        if float(gate_sampling.get(key, np.nan)) != float(
            config["sampling"].get(key, np.nan)
        ):
            raise RuntimeError(
                f"Training sampling field {key!r} differs from the GT Gate"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload": payload,
        "calibration_path": str(calibration),
    }


def build_loader(
    rows: list[dict],
    config: dict,
    transform,
    stats,
    shoulder_points: dict,
    dist_info: dict,
    random_clip: bool = False,
    train: bool = False,
):
    dataset = AgiBotReachRateDataset(
        rows,
        config["sampling"],
        shoulder_points,
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
            list(
                range(
                    dist_info["rank"],
                    len(dataset),
                    dist_info["world_size"],
                )
            ),
        )
    generator = torch.Generator().manual_seed(
        int(config["experiment"]["seed"])
        + (0 if train else 10000)
        + dist_info["rank"]
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
        raise RuntimeError(
            "Training loader has zero batches; lower batch_size or add rows"
        )
    return loader, sampler


def masked_head_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for index in range(len(TARGET_NAMES)):
        mask = valid[..., index]
        error = (prediction[..., index] - target[..., index]).square() * mask
        losses.append(error.sum() / mask.sum().clamp_min(1.0))
    return torch.stack(losses)


@torch.no_grad()
def validate(
    encoder,
    bank,
    loader,
    stats,
    device,
    distributed: bool,
    amp_dtype,
) -> dict:
    bank.eval()
    normalized_sse = torch.zeros(
        len(TARGET_NAMES), dtype=torch.float64, device=device
    )
    raw_abs = torch.zeros_like(normalized_sse)
    raw_sse = torch.zeros_like(normalized_sse)
    counts = torch.zeros_like(normalized_sse)
    _, target_std = [item.to(device) for item in stats]
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            prediction = bank(encoder(video))
        normalized_error = prediction.float() - target.float()
        raw_error = normalized_error * target_std
        normalized_sse += (
            normalized_error.square() * valid
        ).sum((0, 1)).double()
        raw_abs += (raw_error.abs() * valid).sum((0, 1)).double()
        raw_sse += (raw_error.square() * valid).sum((0, 1)).double()
        counts += valid.sum((0, 1)).double()
    for tensor in (normalized_sse, raw_abs, raw_sse, counts):
        reduce_sum(tensor, distributed)
    return {
        name: {
            "normalized_mse": float(
                (
                    normalized_sse[index]
                    / counts[index].clamp_min(1)
                ).item()
            ),
            "raw_mae": float(
                (raw_abs[index] / counts[index].clamp_min(1)).item()
            ),
            "raw_rmse": float(
                torch.sqrt(
                    raw_sse[index] / counts[index].clamp_min(1)
                ).item()
            ),
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
            "protocol": PROTOCOL,
            "target_definition": TARGET_DEFINITION,
            "target_name": target_name,
            "target_index": index,
            "head": unwrap(bank).heads[target_name].state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "validation": metric,
            "target_mean": stats[0][index].cpu(),
            "target_std": stats[1][index].cpu(),
            "encoder": encoder_report,
            "config_fingerprint": config_fingerprint(config),
        },
        path,
    )


def load_best_heads(bank, checkpoint_dir: Path, stats, config: dict) -> dict:
    model = unwrap(bank)
    expected_fingerprint = config_fingerprint(config)
    manifest = {}
    for index, target_name in enumerate(TARGET_NAMES):
        path = checkpoint_dir / target_name / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {path}")
        checkpoint = torch_load(path, map_location="cpu")
        if (
            checkpoint.get("protocol") != PROTOCOL
            or checkpoint.get("target_definition") != TARGET_DEFINITION
            or checkpoint.get("target_name") != target_name
            or int(checkpoint.get("target_index", -1)) != index
            or checkpoint.get("config_fingerprint") != expected_fingerprint
        ):
            raise RuntimeError(f"Checkpoint identity mismatch: {path}")
        if not torch.allclose(
            checkpoint["target_mean"].cpu(), stats[0][index].cpu()
        ):
            raise RuntimeError(f"Checkpoint target mean drift: {path}")
        if not torch.allclose(
            checkpoint["target_std"].cpu(), stats[1][index].cpu()
        ):
            raise RuntimeError(f"Checkpoint target std drift: {path}")
        model.heads[target_name].load_state_dict(
            checkpoint["head"], strict=True
        )
        manifest[target_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "epoch": int(checkpoint["epoch"]),
            "validation": checkpoint["validation"],
        }
    return manifest


def _empty_export_shard() -> dict:
    keys = (
        "video_id",
        "episode_key",
        "window_key",
        "task_id",
        "view_key",
        "prediction",
        "target",
        "valid",
        "endpoint_position",
        "endpoint_valid",
        "reach_rate_gt",
        "reach_rate_valid",
        "target_frame_indices",
        "target_sample_times",
    )
    return {key: [] for key in keys}


@torch.no_grad()
def export_predictions(
    encoder,
    bank,
    loader,
    stats,
    device,
    dist_info,
    config,
    split_name: str,
    output_path: Path,
) -> dict | None:
    bank.eval()
    mean, std = [item.to(device) for item in stats]
    amp_dtype = (
        torch.bfloat16
        if config["runtime"].get("use_bfloat16", False)
        else torch.float16
    )
    local = _empty_export_shard()
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        normalized_target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            normalized_prediction = bank(encoder(video))
        local["prediction"].append(
            (normalized_prediction.float() * std + mean).cpu().numpy()
        )
        local["target"].append(
            (normalized_target.float() * std + mean).cpu().numpy()
        )
        for key in (
            "valid",
            "endpoint_position",
            "endpoint_valid",
            "reach_rate_gt",
            "reach_rate_valid",
            "target_frame_indices",
            "target_sample_times",
        ):
            local[key].append(batch[key].cpu().numpy())
        for key in ("video_id", "episode_key", "window_key", "view_key"):
            local[key].extend(str(item) for item in batch[key])
        local["task_id"].extend(int(item) for item in batch["task_id"])

    tensor_keys = (
        "prediction",
        "target",
        "valid",
        "endpoint_position",
        "endpoint_valid",
        "reach_rate_gt",
        "reach_rate_valid",
        "target_frame_indices",
        "target_sample_times",
    )
    for key in tensor_keys:
        local[key] = np.concatenate(local[key], axis=0)
    if dist_info["distributed"]:
        gathered = (
            [None] * dist_info["world_size"]
            if dist_info["is_main"]
            else None
        )
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    if not dist_info["is_main"]:
        return None

    string_keys = ("video_id", "episode_key", "window_key", "view_key")
    combined = {
        key: np.asarray(
            [value for shard in gathered for value in shard[key]],
            dtype=str,
        )
        for key in string_keys
    }
    combined["task_id"] = np.asarray(
        [value for shard in gathered for value in shard["task_id"]],
        dtype=np.int64,
    )
    for key in tensor_keys:
        combined[key] = np.concatenate(
            [shard[key] for shard in gathered], axis=0
        )
    order = np.argsort(combined["video_id"])
    for key in combined:
        combined[key] = combined[key][order]
    if len(set(combined["video_id"].tolist())) != len(combined["video_id"]):
        raise RuntimeError(f"Duplicate rows while exporting {split_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **combined,
        target_names=np.asarray(TARGET_NAMES),
        target_mean=stats[0].cpu().numpy(),
        target_std=stats[1].cpu().numpy(),
        target_definition=np.asarray(TARGET_DEFINITION),
        protocol=np.asarray(PROTOCOL),
        split_name=np.asarray(split_name),
        model_name=np.asarray(config["experiment"]["model_name"]),
        seed=np.asarray(int(config["experiment"]["seed"])),
        time_shift_tokens=np.asarray(
            int(config["reach_rate"]["time_shift_tokens"])
        ),
    )
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "row_count": int(len(combined["video_id"])),
        "shape": list(combined["prediction"].shape),
    }


def train(config: dict) -> Path:
    seed = int(config["experiment"]["seed"])
    model_name = str(config["experiment"]["model_name"])
    run_dir = (
        Path(config["output"]["root"]).expanduser().resolve()
        / model_name
        / f"seed_{seed}"
    )
    if (
        config["experiment"]["phase"] != "smoke"
        and (run_dir / "COMPLETED.json").exists()
    ):
        raise FileExistsError(f"Refusing to overwrite completed run: {run_dir}")
    gate_report = verify_gt_gate(config)
    gate = gate_report["payload"]
    shoulder_points = shoulder_points_from_gate(gate)
    dist_info = init_distributed(config["runtime"].get("device", "cuda"))
    is_main, device = dist_info["is_main"], dist_info["device"]
    try:
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(
                json_safe(config), run_dir / "resolved_config.json"
            )
        rows = load_protocol_rows(config)
        print_main(
            is_main,
            "rows:",
            {
                "train": len(rows["train"]),
                "val": len(rows["val"]),
                **{
                    name: len(items)
                    for name, items in rows["eval"].items()
                },
            },
        )
        seed_everything(seed)
        stats_config = config["stats"]
        target_mean, target_std, stats_payload = get_or_compute_stats(
            rows["train"],
            config["sampling"],
            shoulder_points,
            stats_config["cache"],
            seed=int(stats_config.get("seed", 239)),
            max_rows=int(stats_config.get("max_rows", 0)),
            distributed=dist_info["distributed"],
            rank=dist_info["rank"],
            device=device,
        )
        stats = (target_mean, target_std)
        transform = build_input_transform(config)
        train_loader, train_sampler = build_loader(
            rows["train"],
            config,
            transform,
            stats,
            shoulder_points,
            dist_info,
            random_clip=True,
            train=True,
        )
        val_loader, _ = build_loader(
            rows["val"],
            config,
            transform,
            stats,
            shoulder_points,
            dist_info,
        )
        encoder, embed_dim, encoder_report = build_encoder(
            config, device, is_main
        )
        seed_everything(seed)
        bank = IndependentHeadBank(
            embed_dim,
            config["sampling"]["num_frames"],
            config["sampling"]["tubelet_size"],
            TARGET_NAMES,
            head_type=config["probe"]["type"],
            hidden_dim=int(config["probe"].get("hidden_dim", 0)),
        ).to(device)
        bank.assert_disjoint()
        architecture_report = bank.architecture_report()
        print_main(
            is_main,
            "probe:",
            json.dumps(json_safe(architecture_report), ensure_ascii=False),
        )
        if dist_info["distributed"]:
            ddp_options = {"broadcast_buffers": False}
            if device.type == "cuda":
                ddp_options.update(
                    device_ids=[dist_info["local_rank"]],
                    output_device=dist_info["local_rank"],
                )
            bank = DistributedDataParallel(bank, **ddp_options)

        optimizers, lr_schedulers, wd_schedulers = {}, {}, {}
        opt_args = optimizer_args(config)
        for name in TARGET_NAMES:
            optimizer, lr_scheduler, wd_scheduler = (
                build_optimizer_and_schedulers(
                    opt_args, unwrap(bank).heads[name], len(train_loader)
                )
            )
            optimizers[name] = optimizer
            lr_schedulers[name] = lr_scheduler
            wd_schedulers[name] = wd_scheduler
        amp_dtype = (
            torch.bfloat16
            if config["runtime"].get("use_bfloat16", False)
            else torch.float16
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda" and amp_dtype == torch.float16,
        )
        best = {name: math.inf for name in TARGET_NAMES}
        history = []
        checkpoint_dir = run_dir / "checkpoints"
        for epoch in range(int(config["optimization"]["epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            bank.train()
            train_sse = torch.zeros(
                len(TARGET_NAMES), dtype=torch.float64, device=device
            )
            train_count = torch.zeros_like(train_sse)
            for batch in train_loader:
                video = batch["video"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    features = encoder(video)
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    prediction = bank(features.detach())
                    losses = masked_head_losses(prediction, target, valid)
                if scaler.is_enabled():
                    scaler.scale(losses.sum()).backward()
                    for optimizer in optimizers.values():
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.sum().backward()
                    for optimizer in optimizers.values():
                        optimizer.step()
                for name in TARGET_NAMES:
                    if lr_schedulers[name] is not None:
                        lr_schedulers[name].step()
                    if wd_schedulers[name] is not None:
                        wd_schedulers[name].step()
                train_sse += (
                    (prediction.float() - target.float()).square() * valid
                ).sum((0, 1)).double()
                train_count += valid.sum((0, 1)).double()
            reduce_sum(train_sse, dist_info["distributed"])
            reduce_sum(train_count, dist_info["distributed"])
            val_metrics = validate(
                encoder,
                bank,
                val_loader,
                stats,
                device,
                dist_info["distributed"],
                amp_dtype,
            )
            record = {
                "epoch": epoch + 1,
                "train_normalized_mse": {
                    name: float(
                        (
                            train_sse[index]
                            / train_count[index].clamp_min(1)
                        ).item()
                    )
                    for index, name in enumerate(TARGET_NAMES)
                },
                "validation": val_metrics,
            }
            if is_main:
                history.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                atomic_json_dump(
                    json_safe(history), run_dir / "history.json"
                )
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
        best_manifest = load_best_heads(
            bank, checkpoint_dir, stats, config
        )
        if is_main:
            atomic_json_dump(
                json_safe(best_manifest), run_dir / "best_heads.json"
            )

        outputs = {}
        for split_name, eval_rows in rows["eval"].items():
            eval_loader, _ = build_loader(
                eval_rows,
                config,
                transform,
                stats,
                shoulder_points,
                dist_info,
            )
            prediction_path = (
                run_dir / "predictions" / f"{split_name}.npz"
            )
            export = export_predictions(
                encoder,
                bank,
                eval_loader,
                stats,
                device,
                dist_info,
                config,
                split_name,
                prediction_path,
            )
            if is_main:
                grounded = evaluate_prediction_file(
                    prediction_path,
                    gate_report["path"],
                    run_dir / "metrics",
                    split_name=split_name,
                    time_shift_tokens=int(
                        config["reach_rate"]["time_shift_tokens"]
                    ),
                )
                outputs[split_name] = {
                    "predictions": export,
                    "metrics": grounded["metrics_path"],
                    "left_active_formula_mae": grounded[
                        "arm_flattened_metrics"
                    ]["left"]["active_formula_mae"],
                    "right_active_formula_mae": grounded[
                        "arm_flattened_metrics"
                    ]["right"]["active_formula_mae"],
                    "macro_active_formula_mae": grounded[
                        "flattened_metrics"
                    ]["active_formula_mae"],
                }
            if dist_info["distributed"]:
                dist.barrier()

        if is_main:
            completion = {
                "status": "COMPLETE",
                "protocol": PROTOCOL,
                "target_definition": TARGET_DEFINITION,
                "primary_metric": config["reach_rate"][
                    "primary_metric"
                ],
                "phase": config["experiment"]["phase"],
                "model_name": model_name,
                "probe_seed": seed,
                "config_fingerprint": config_fingerprint(config),
                "target_manifest": target_manifest(),
                "probe": architecture_report,
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
                "gt_gate": {
                    "path": gate_report["path"],
                    "sha256": gate_report["sha256"],
                    "calibration_path": gate_report[
                        "calibration_path"
                    ],
                },
                "stats": stats_payload,
                "best_heads": best_manifest,
                "outputs": outputs,
            }
            atomic_json_dump(
                json_safe(completion), run_dir / "COMPLETED.json"
            )
            print(
                json.dumps(
                    {"status": "COMPLETE", "run_dir": str(run_dir)},
                    indent=2,
                )
            )
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
    if args.reuse_heads_from:
        raise ValueError(
            "Head reuse is forbidden: all 20 heads must be retrained per seed"
        )
    overrides = {}
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = int(args.seed)
    if args.output_root:
        overrides.setdefault("output", {})["root"] = str(
            Path(args.output_root).expanduser()
        )
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
