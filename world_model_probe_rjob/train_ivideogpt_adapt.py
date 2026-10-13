#!/usr/bin/env python3
"""Domain-adapt the official iVideoGPT causal transformer, then save it.

This stage never trains a downstream evaluation probe.  ``video_only`` uses
only iVideoGPT's native causal token objective.  ``time`` adds the established
aligned physical auxiliary loss; its temporary head is saved for audit but is
not reused by downstream probes.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parent
IVIDEOGPT_ROOT = REPO_ROOT / "iVideoGPT"
for path in (str(REPO_ROOT), str(RUN_DIR), str(IVIDEOGPT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from ivideogpt_adapt_config import parse_args  # noqa: E402
from ivideogpt_adapt_data import (  # noqa: E402
    build_adapt_dataset,
    collate_adapt_batch,
    write_filtered_covla_metadata,
)
from ivideogpt_adapt_model import (  # noqa: E402
    IVideoGPTAdaptModel,
    load_trainable_components,
    optimizer_parameter_groups,
    tokenize_clips,
)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _trainable_count(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _set_learning_rate(optimizer, cfg, completed_updates, total_updates):
    base = float(cfg["lr"])
    start = float(cfg["start_lr"])
    final = float(cfg["final_lr"])
    warmup_updates = int(cfg["warmup_updates"])
    if warmup_updates > 0 and completed_updates < warmup_updates:
        fraction = float(completed_updates) / float(max(1, warmup_updates))
        value = start + (base - start) * fraction
    else:
        progress = float(completed_updates - warmup_updates) / float(
            max(1, total_updates - warmup_updates - 1)
        )
        progress = min(1.0, max(0.0, progress))
        value = final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def _physical_lambda(cfg, completed_updates):
    if not cfg["enabled"]:
        return 0.0
    base = float(cfg["lambda_time"])
    warmup = int(cfg["lambda_warmup_steps"])
    if warmup <= 0:
        return base
    return base * min(1.0, float(completed_updates + 1) / float(warmup))


def _gradient_audit(model, tokenizer):
    report = {
        "tokenizer_grad_tensors": sum(
            parameter.grad is not None for parameter in tokenizer.parameters()
        ),
        "transformer_nonzero_grad_tensors": 0,
        "physical_head_nonzero_grad_tensors": 0,
        "all_grads_finite": True,
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        finite = bool(torch.isfinite(parameter.grad).all().item())
        nonzero = bool(torch.count_nonzero(parameter.grad).item())
        report["all_grads_finite"] = report["all_grads_finite"] and finite
        if nonzero and name.startswith("transformer."):
            report["transformer_nonzero_grad_tensors"] += 1
        if nonzero and name.startswith("physical_head."):
            report["physical_head_nonzero_grad_tensors"] += 1
    if report["tokenizer_grad_tensors"] != 0:
        raise RuntimeError("Frozen tokenizer unexpectedly received gradients")
    if report["transformer_nonzero_grad_tensors"] == 0:
        raise RuntimeError("Transformer did not receive any non-zero gradients")
    if model.mode == "time" and report["physical_head_nonzero_grad_tensors"] == 0:
        raise RuntimeError("Time physical head did not receive any non-zero gradients")
    if not report["all_grads_finite"]:
        raise RuntimeError("Non-finite gradients detected on the first update")
    return report


def _checkpoint_name(epoch, stopped_early, completed_updates):
    if stopped_early:
        return f"step_{completed_updates:08d}"
    return f"epoch_{epoch:03d}"


def save_checkpoint(
    accelerator,
    model,
    cfg,
    protocol,
    epoch,
    completed_updates,
    stopped_early,
    epoch_metrics,
):
    accelerator.wait_for_everyone()
    checkpoint_name = _checkpoint_name(epoch, stopped_early, completed_updates)
    checkpoint_dir = Path(cfg["runtime"]["output_dir"]) / "checkpoints" / checkpoint_name
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)
    transformer_dir = checkpoint_dir / "transformer"
    unwrapped.transformer.save_pretrained(
        transformer_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    if accelerator.is_main_process and unwrapped.physical_head is not None:
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in unwrapped.physical_head.state_dict().items()
        }
        save_file(state, str(checkpoint_dir / "physical_head.safetensors"))

    if accelerator.is_main_process:
        manifest = {
            "format_version": 1,
            "model_family": "ivideogpt_oxe_64_medium",
            "adaptation_mode": cfg["optimization"]["mode"],
            "base_snapshot": cfg["model"]["base_snapshot"],
            "config_path": cfg["config_path"],
            "epoch": int(epoch),
            "completed_updates": int(completed_updates),
            "stopped_early": bool(stopped_early),
            "transformer_checkpoint": str(
                (transformer_dir / "model.safetensors").resolve()
            ),
            "temporary_physical_head_reused_by_probe": False,
            "tokenizer_trained": False,
            "transformer_trained": True,
            "protocol": protocol,
            "epoch_metrics": epoch_metrics,
            "software": {
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
            },
        }
        _atomic_json(checkpoint_dir / "adaptation_manifest.json", manifest)
        _atomic_json(
            Path(cfg["runtime"]["output_dir"]) / "latest_checkpoint.json",
            manifest,
        )
    accelerator.wait_for_everyone()
    return checkpoint_dir


def _prepare_output_and_covla(cfg, accelerator):
    output_dir = Path(cfg["runtime"]["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
    accelerator.wait_for_everyone()

    filtered_metadata = None
    covla_report = None
    if cfg["data"]["domain"] == "covla":
        filtered_metadata = str(output_dir / "protocol" / "covla_encoder_train.jsonl")
        if accelerator.is_main_process:
            covla_report = write_filtered_covla_metadata(
                cfg["data"]["root"],
                cfg["data"]["encoder_train_split"],
                filtered_metadata,
            )
            _atomic_json(output_dir / "protocol" / "covla_filter_report.json", covla_report)
        accelerator.wait_for_everyone()
        if not Path(filtered_metadata).is_file():
            raise RuntimeError(f"Filtered COVLA metadata was not created: {filtered_metadata}")
    return filtered_metadata


def main(argv=None):
    args = parse_args(argv)
    cfg = args.resolved_config
    if args.config_check:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

    optimization = cfg["optimization"]
    runtime = cfg["runtime"]
    accelerator = Accelerator(
        mixed_precision=runtime["mixed_precision"],
        gradient_accumulation_steps=optimization["gradient_accumulation_steps"],
    )
    actual_global_batch = (
        int(optimization["per_device_batch_size"])
        * int(optimization["gradient_accumulation_steps"])
        * int(accelerator.num_processes)
    )
    formal_run = args.max_steps is None and not args.data_check
    if formal_run and accelerator.num_processes != int(runtime["expected_world_size"]):
        raise RuntimeError(
            "Formal adaptation world-size mismatch: "
            f"actual={accelerator.num_processes}, expected={runtime['expected_world_size']}. "
            "Use the default 8-GPU rjob; non-formal smoke tests must set --max-steps."
        )
    if formal_run and actual_global_batch != int(runtime["expected_global_batch_size"]):
        raise RuntimeError(
            "Formal adaptation global-batch mismatch: "
            f"actual={actual_global_batch}, expected={runtime['expected_global_batch_size']}"
        )
    set_seed(int(optimization["seed"]), device_specific=True)
    filtered_metadata = _prepare_output_and_covla(cfg, accelerator)
    dataset, stats, protocol = build_adapt_dataset(cfg, filtered_metadata)
    protocol["per_device_batch_size"] = optimization["per_device_batch_size"]
    protocol["gradient_accumulation_steps"] = optimization[
        "gradient_accumulation_steps"
    ]
    protocol["world_size"] = accelerator.num_processes
    protocol["global_batch_size"] = actual_global_batch
    if accelerator.is_main_process:
        _atomic_json(Path(runtime["output_dir"]) / "protocol_manifest.json", protocol)
        print(json.dumps({"dataset_protocol": protocol}, indent=2, ensure_ascii=False))
    if args.data_check:
        accelerator.wait_for_everyone()
        return

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(optimization["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=optimization["per_device_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=int(runtime["num_workers"]),
        persistent_workers=(
            int(runtime["num_workers"]) > 0 and bool(runtime["persistent_workers"])
        ),
        pin_memory=bool(runtime["pin_memory"]),
        collate_fn=collate_adapt_batch,
        generator=loader_generator,
    )

    tokenizer, transformer, snapshot_root = load_trainable_components(
        cfg["model"]["base_snapshot"],
        attention_implementation=cfg["model"]["attention_implementation"],
        gradient_checkpointing=True,
    )
    target_fields = protocol["target_fields"]
    model = IVideoGPTAdaptModel(
        transformer=transformer,
        mode=optimization["mode"],
        num_physical_outputs=len(target_fields),
        physical_head_type=cfg["physical"].get("head_type", "conv1x1"),
        smooth_l1_beta=float(cfg["physical"].get("smooth_l1_beta", 0.5)),
    )
    tokenizer.to(accelerator.device)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, optimization["weight_decay"]),
        lr=float(optimization["lr"]),
        betas=(0.9, 0.999),
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    updates_per_epoch = math.ceil(
        len(loader) / int(optimization["gradient_accumulation_steps"])
    )
    total_updates = int(optimization["epochs"]) * updates_per_epoch
    if args.max_steps is not None:
        if int(args.max_steps) <= 0:
            raise ValueError("--max-steps must be positive")
        total_updates = min(total_updates, int(args.max_steps))
    optimization["warmup_updates"] = int(optimization["warmup_epochs"]) * updates_per_epoch

    target_mean = target_std = None
    if cfg["physical"]["enabled"]:
        target_mean = torch.tensor(
            stats["mean"], dtype=torch.float32, device=accelerator.device
        ).view(1, 1, -1)
        target_std = torch.tensor(
            stats["std"], dtype=torch.float32, device=accelerator.device
        ).view(1, 1, -1)

    if accelerator.is_main_process:
        startup = {
            "snapshot_root": str(snapshot_root),
            "mode": optimization["mode"],
            "tokenizer_parameters": _parameter_count(tokenizer),
            "tokenizer_trainable_parameters": _trainable_count(tokenizer),
            "adapt_model_parameters": _parameter_count(accelerator.unwrap_model(model)),
            "adapt_model_trainable_parameters": _trainable_count(
                accelerator.unwrap_model(model)
            ),
            "updates_per_epoch": updates_per_epoch,
            "planned_updates": total_updates,
            "world_size": accelerator.num_processes,
            "mixed_precision": accelerator.mixed_precision,
        }
        _atomic_json(Path(runtime["output_dir"]) / "startup_manifest.json", startup)
        print(json.dumps(startup, indent=2, ensure_ascii=False))

    writer = None
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(Path(runtime["output_dir"]) / "tensorboard"))

    completed_updates = 0
    gradient_report = None
    stop_training = False
    for epoch in range(int(optimization["epochs"])):
        model.train()
        epoch_video = 0.0
        epoch_physical = 0.0
        epoch_total = 0.0
        epoch_batches = 0
        epoch_start = time.time()
        for batch in loader:
            videos = batch["video"].to(
                accelerator.device, non_blocking=bool(runtime["pin_memory"])
            )
            tokens, labels = tokenize_clips(
                tokenizer,
                videos,
                accelerator.unwrap_model(model).layout,
                accelerator.unwrap_model(model).sdf_token_id,
            )
            target = valid = None
            if cfg["physical"]["enabled"]:
                target = batch["target"].to(accelerator.device, non_blocking=True)
                valid = batch["valid"].to(accelerator.device, non_blocking=True)
            lambda_time = _physical_lambda(cfg["physical"], completed_updates)
            current_lr = _set_learning_rate(
                optimizer, optimization, completed_updates, total_updates
            )

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(
                        tokens=tokens,
                        labels=labels,
                        clip_batch_size=videos.shape[0],
                        target=target,
                        valid=valid,
                        target_mean=target_mean,
                        target_std=target_std,
                        lambda_time=lambda_time,
                    )
                accelerator.backward(outputs["loss"])
                if accelerator.sync_gradients:
                    if gradient_report is None:
                        gradient_report = _gradient_audit(
                            accelerator.unwrap_model(model), tokenizer
                        )
                    accelerator.clip_grad_norm_(
                        model.parameters(), float(optimization["max_grad_norm"])
                    )
                optimizer.step()
                optimizer.zero_grad()

            reduced_total = accelerator.reduce(outputs["loss"].detach(), reduction="mean")
            reduced_video = accelerator.reduce(
                outputs["loss_video"].detach(), reduction="mean"
            )
            reduced_physical = accelerator.reduce(
                outputs["loss_physical"].detach(), reduction="mean"
            )
            epoch_total += float(reduced_total.item())
            epoch_video += float(reduced_video.item())
            epoch_physical += float(reduced_physical.item())
            epoch_batches += 1

            if accelerator.sync_gradients:
                completed_updates += 1
                if accelerator.is_main_process and (
                    completed_updates % int(runtime["log_every_updates"]) == 0
                    or completed_updates == 1
                ):
                    log = {
                        "update": completed_updates,
                        "epoch": epoch,
                        "loss": float(reduced_total.item()),
                        "loss_video": float(reduced_video.item()),
                        "loss_physical": float(reduced_physical.item()),
                        "lambda_time": lambda_time,
                        "lr": current_lr,
                    }
                    print(json.dumps(log, ensure_ascii=False), flush=True)
                    if writer is not None:
                        for key, value in log.items():
                            if key not in ("update", "epoch"):
                                writer.add_scalar(f"train/{key}", value, completed_updates)
                if completed_updates >= total_updates:
                    stop_training = True
            del outputs, tokens, labels, videos
            if stop_training:
                break

        epoch_metrics = {
            "loss": epoch_total / max(1, epoch_batches),
            "loss_video": epoch_video / max(1, epoch_batches),
            "loss_physical": epoch_physical / max(1, epoch_batches),
            "batches": epoch_batches,
            "wall_time_seconds": time.time() - epoch_start,
        }
        completed_full_epoch = not stop_training or completed_updates >= (
            (epoch + 1) * updates_per_epoch
        )
        save_due = (
            (epoch + 1) % int(runtime["save_every_epochs"]) == 0
            or stop_training
            or epoch + 1 == int(optimization["epochs"])
        )
        if save_due:
            save_checkpoint(
                accelerator,
                model,
                cfg,
                protocol,
                epoch,
                completed_updates,
                stopped_early=not completed_full_epoch,
                epoch_metrics=epoch_metrics,
            )
        if stop_training:
            break

    if accelerator.is_main_process:
        if gradient_report is None:
            raise RuntimeError("Training ended before the first optimizer update")
        _atomic_json(
            Path(runtime["output_dir"]) / "gradient_audit.json", gradient_report
        )
        if writer is not None:
            writer.close()
        print(
            json.dumps(
                {
                    "status": "complete",
                    "completed_updates": completed_updates,
                    "gradient_audit": gradient_report,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
