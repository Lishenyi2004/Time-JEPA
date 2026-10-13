"""Train PSG-Static-g-Port or PSG-Full-g-Port on the matched AgiBot split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader

from src.time_jepa.agibot_derived_targets import WHOLEBODY_CARTESIAN_V2_FIELDS

from .data import AgiBotPSGWindowDataset, collate_psg_windows
from .modeling import verify_upstream_checkout
from .protocol import EXPECTED_UPSTREAM_COMMIT, AgiBotPSGProtocol
from .vitg_modeling import (
    CheckpointLoadReport,
    build_vitg_psg_world_model,
    parameter_counts,
)
from .vitg_objective import psg_vitg_forward


FORMAL_STATE_INDICES = tuple(range(18))
FORMAL_CONFIGURATION_INDICES = (2, 3, 4, 5, 6, 7, 8, 9, 16, 17)
FORMAL_VELOCITY_INDICES = (0, 1, 10, 11, 12, 13, 14, 15)


class AgiBotPSGViTGModule(pl.LightningModule):
    def __init__(self, model, sigreg, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.sigreg = sigreg
        self.cfg = cfg
        self.save_hyperparameters(
            {"config": OmegaConf.to_container(cfg, resolve=True)},
            ignore=("model", "sigreg"),
        )

    def forward(self, batch, stage="predict"):
        return psg_vitg_forward(self, batch, stage, self.cfg)

    def training_step(self, batch, batch_index):
        del batch_index
        return self(batch, stage="train")["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=float(self.cfg.optimizer.lr),
            betas=tuple(float(value) for value in self.cfg.optimizer.betas),
            eps=float(self.cfg.optimizer.eps),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )
        total_steps = max(2, int(self.trainer.estimated_stepping_batches))
        epochs = max(1, int(self.cfg.trainer.max_epochs))
        steps_per_epoch = max(1, math.ceil(total_steps / epochs))
        warmup_steps = min(
            total_steps - 1,
            max(1, int(float(self.cfg.optimizer.warmup_epochs) * steps_per_epoch)),
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=total_steps,
            warmup_start_lr=float(self.cfg.optimizer.start_lr),
            eta_min=float(self.cfg.optimizer.min_lr),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ReproducibilityArtifacts(pl.Callback):
    """Write the run manifest and an encoder-only final artifact on rank zero."""

    def __init__(
        self,
        *,
        output_dir: Path,
        cfg: DictConfig,
        protocol: AgiBotPSGProtocol,
        upstream_root: Path,
        load_report: CheckpointLoadReport,
        counts: dict[str, int],
        optimizer_steps_per_epoch: int,
        local_batches_per_epoch: int,
        export_encoder: bool,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.cfg = cfg
        self.protocol = protocol
        self.upstream_root = upstream_root
        self.load_report = load_report
        self.counts = counts
        self.optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        self.local_batches_per_epoch = int(local_batches_per_epoch)
        self.export_encoder = bool(export_encoder)

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            self.cfg,
            self.output_dir / "config-resolved.yaml",
            resolve=True,
        )
        cfg = self.cfg
        checkpoint_path = Path(cfg.model.checkpoint_path).expanduser().resolve()
        action_stats_path = Path(cfg.data.action_stats_path).expanduser().resolve()
        state_stats_path = Path(cfg.data.state_stats_path).expanduser().resolve()
        payload = {
            "experiment": f"PSG-{str(cfg.loss.grounding.variant).title()}-g-Port",
            "upstream_repository": "https://github.com/Haodong-Yan/PSG-JEPA",
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
            "upstream_root": str(self.upstream_root),
            "initialization": {
                "backbone": "released_vjepa2_1_vitg_384_image_branch",
                "psg_projectors_predictor_and_heads": "from_scratch",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "checkpoint_load": self.load_report.to_dict(),
            },
            "model_construction": {
                "world_model_class": (
                    f"{type(pl_module.model).__module__}."
                    f"{type(pl_module.model).__name__}"
                ),
                "visual_encoder_adapter_class": (
                    f"{type(pl_module.model.encoder).__module__}."
                    f"{type(pl_module.model.encoder).__name__}"
                ),
                "projector_class": (
                    f"{type(pl_module.model.projector).__module__}."
                    f"{type(pl_module.model.projector).__name__}"
                ),
                "predictor_class": (
                    f"{type(pl_module.model.predictor).__module__}."
                    f"{type(pl_module.model.predictor).__name__}"
                ),
                "action_encoder_class": (
                    f"{type(pl_module.model.action_encoder).__module__}."
                    f"{type(pl_module.model.action_encoder).__name__}"
                ),
                "grounding_class": (
                    f"{type(pl_module.model.grounding).__module__}."
                    f"{type(pl_module.model.grounding).__name__}"
                ),
            },
            "data": {
                "params_path": str(self.protocol.params_path),
                "params_sha256": _sha256(self.protocol.params_path),
                "state_stats_path": str(state_stats_path),
                "state_stats_sha256": _sha256(state_stats_path),
                "action_stats_path": str(action_stats_path),
                "action_stats_sha256": _sha256(action_stats_path),
                "sampling_protocol": str(cfg.comparison.get("data_protocol", "quota7000")),
                "unique_physical_windows": len(self.protocol.sample_keys),
                "view_windows": self.protocol.num_view_windows,
                "per_view_windows": {
                    view: len(keys)
                    for view, keys in (
                        self.protocol.view_sample_keys
                        if self.protocol.view_sample_keys is not None
                        else ((view, self.protocol.sample_keys) for view in self.protocol.video_keys)
                    )
                },
                "frames_per_view_window": self.protocol.window_frames,
                "microclips_per_view_window": self.protocol.microclips_per_window,
                "frames_per_psg_sequence": 4,
                "actions_30hz_per_5hz_transition": (
                    self.protocol.action_steps_per_model_step
                ),
            },
            "target_mapping": {
                "fields": list(WHOLEBODY_CARTESIAN_V2_FIELDS),
                "state_idx": list(FORMAL_STATE_INDICES),
                "configuration_delta_idx": list(FORMAL_CONFIGURATION_INDICES),
                "velocity_idx": list(FORMAL_VELOCITY_INDICES),
            },
            "temporal_contract": {
                "visual_encoder": "independent_T1_frames",
                "psg_sequence_frames": 4,
                "psg_sequence_duration_seconds": 0.6,
                "frames_per_view_window_exposed": 64,
                "spatial_pool": "mean_over_24x24_image_tokens",
            },
            "parameter_counts": self.counts,
            "frozen_inactive_image_mode_parameters": list(
                pl_module.model.inactive_parameter_names
            ),
            "batch_contract": {
                "world_size": trainer.world_size,
                "base_windows_per_rank_per_microbatch": int(
                    cfg.loader.base_window_batch_size
                ),
                "gradient_accumulation": int(
                    cfg.trainer.accumulate_grad_batches
                ),
                "effective_global_view_window_batch": (
                    trainer.world_size
                    * int(cfg.loader.base_window_batch_size)
                    * int(cfg.trainer.accumulate_grad_batches)
                ),
                "local_batches_per_epoch": self.local_batches_per_epoch,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            },
        }
        _atomic_json_dump(payload, self.output_dir / "run-manifest.json")

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero or not self.export_encoder:
            return
        output_path = self.output_dir / "encoder_final.pt"
        temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
        torch.save(
            {
                "encoder": pl_module.model.encoder.backbone.state_dict(),
                "projector": pl_module.model.projector.state_dict(),
                "grounding": pl_module.model.grounding.state_dict(),
                "meta": {
                    "experiment": (
                        f"PSG-{pl_module.model.grounding_variant.title()}-g-Port"
                    ),
                    "model_name": str(self.cfg.model.model_name),
                    "image_size": int(self.cfg.model.image_size),
                    "visual_temporal_input": 1,
                    "pooling": "spatial_mean",
                    "embedding_dim": int(self.cfg.wm.embed_dim),
                    "sampling_protocol": str(self.cfg.comparison.get("data_protocol", "quota7000")),
                    "training_view_windows": self.protocol.num_view_windows,
                    "training_params_path": str(self.protocol.params_path),
                    "upstream_world_model_class": (
                        "psgjepa.model.CrossModalJEPA_v2"
                    ),
                },
            },
            temporary,
        )
        os.replace(temporary, output_path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _validate_formal_config(cfg: DictConfig, protocol: AgiBotPSGProtocol) -> None:
    if int(cfg.seed) != 239:
        raise ValueError("The matched Time-JEPA-g comparison is pinned to seed 239")
    if protocol.sample_seed != 239:
        raise ValueError(
            f"Resolved data seed is {protocol.sample_seed}; expected matched seed 239"
        )
    data_protocol = str(cfg.comparison.get("data_protocol", "quota7000"))
    if data_protocol == "quota7000":
        if protocol.num_view_windows != 7000:
            raise ValueError(
                "quota7000 requires exactly 7,000 quota-sampled view-windows, "
                f"got {protocol.num_view_windows}"
            )
        if protocol.view_sample_keys is None:
            raise ValueError("quota7000 requires persisted per-view allowlists")
        expected_view_counts = {
            "observation.images.top_head": 3000,
            "observation.images.hand_left": 2000,
            "observation.images.hand_right": 2000,
        }
        actual_view_counts = {view: len(keys) for view, keys in protocol.view_sample_keys}
    elif data_protocol == "standard15000":
        if protocol.view_sample_keys is not None or len(protocol.sample_keys) != 5000:
            raise ValueError(
                "standard15000 requires the paper's 5,000 physical-window allowlist "
                "expanded across all three views"
            )
        if protocol.num_view_windows != 15000:
            raise ValueError(
                "standard15000 requires exactly 15,000 view-windows, "
                f"got {protocol.num_view_windows}"
            )
        expected_view_counts = {
            "observation.images.top_head": 5000,
            "observation.images.hand_left": 5000,
            "observation.images.hand_right": 5000,
        }
        actual_view_counts = {view: len(protocol.sample_keys) for view in protocol.video_keys}
    else:
        raise ValueError(f"Unsupported formal data protocol: {data_protocol!r}")
    if actual_view_counts != expected_view_counts:
        raise ValueError(
            f"Unexpected per-view exposure: {actual_view_counts}; "
            f"expected {expected_view_counts}"
        )
    if int(cfg.model.image_size) != 384:
        raise ValueError("Formal PSG-g runs require 384x384 inputs")
    if int(cfg.model.patch_size) != 16 or int(cfg.model.tubelet_size) != 2:
        raise ValueError("Formal PSG-g runs require the V-JEPA ViT-g/16 definition")
    if str(cfg.model.model_name) != "vit_giant_xformers":
        raise ValueError("Formal PSG-g runs require vit_giant_xformers")
    if int(cfg.model.state_dim) != 18:
        raise ValueError("The matched wholebody_cartesian_v2 panel is 18-D")
    if int(cfg.wm.embed_dim) != 1408:
        raise ValueError(
            "The ViT-g PSG port keeps z_t at the 1,408-D backbone width, "
            "matching the upstream rule embed_dim == encoder hidden_dim"
        )
    if int(cfg.wm.history_size) + int(cfg.wm.num_preds) != 4:
        raise ValueError("The official PSG temporal recipe requires a four-frame sequence")
    grounding = cfg.loss.grounding
    if tuple(int(value) for value in grounding.state_idx) != FORMAL_STATE_INDICES:
        raise ValueError("PSG-g static grounding must use the complete matched 18-D panel")
    if tuple(int(value) for value in grounding.joint_idx) != FORMAL_CONFIGURATION_INDICES:
        raise ValueError("Unexpected Full-Port configuration-delta target mapping")
    if tuple(int(value) for value in grounding.vel_idx) != FORMAL_VELOCITY_INDICES:
        raise ValueError("Unexpected Full-Port velocity target mapping")
    if set(FORMAL_CONFIGURATION_INDICES) & set(FORMAL_VELOCITY_INDICES):
        raise RuntimeError("Formal configuration and velocity target panels overlap")
    if set(FORMAL_CONFIGURATION_INDICES) | set(FORMAL_VELOCITY_INDICES) != set(
        FORMAL_STATE_INDICES
    ):
        raise RuntimeError("Formal Full-Port target partition does not cover all 18 fields")


def _build_loader(cfg: DictConfig, protocol: AgiBotPSGProtocol) -> DataLoader:
    action_stats_path = Path(cfg.data.action_stats_path).expanduser().resolve()
    if not action_stats_path.is_file():
        raise FileNotFoundError(
            f"Missing matched-training action statistics: {action_stats_path}. Run "
            "python -m external_baselines.psg_jepa.stats with the matched "
            "params-pretrain.yaml first."
        )
    dataset = AgiBotPSGWindowDataset(
        protocol,
        action_stats_path=action_stats_path,
        state_stats_path=cfg.data.state_stats_path,
        image_size=int(cfg.model.image_size),
        training=True,
        video_backend=str(cfg.data.video_backend),
        ffmpeg_timeout=float(cfg.data.ffmpeg_timeout),
    )
    generator = torch.Generator().manual_seed(int(cfg.seed))
    return DataLoader(
        dataset,
        batch_size=int(cfg.loader.base_window_batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg.loader.num_workers),
        persistent_workers=bool(cfg.loader.persistent_workers)
        and int(cfg.loader.num_workers) > 0,
        prefetch_factor=(
            int(cfg.loader.prefetch_factor)
            if int(cfg.loader.num_workers) > 0
            else None
        ),
        pin_memory=bool(cfg.loader.pin_memory),
        collate_fn=collate_psg_windows,
        generator=generator,
    )


def _resolve_training_budget(
    cfg: DictConfig,
    protocol: AgiBotPSGProtocol,
) -> tuple[int, int]:
    devices = cfg.trainer.devices
    if not isinstance(devices, int):
        raise ValueError("Formal PSG-g configs require trainer.devices to be an integer")
    world_size = int(devices) * int(cfg.trainer.num_nodes)
    microbatch = int(cfg.loader.base_window_batch_size)
    accumulation = int(cfg.trainer.accumulate_grad_batches)
    effective_batch = world_size * microbatch * accumulation
    required_batch = int(cfg.comparison.effective_global_window_batch)
    if bool(cfg.comparison.enforce_formal_batch) and effective_batch != required_batch:
        raise ValueError(
            f"Effective view-window batch is {effective_batch}; expected {required_batch}. "
            "For 8 GPUs use accumulation=8; for 4 GPUs use accumulation=16."
        )
    optimizer_steps = protocol.num_view_windows // max(1, effective_batch)
    if optimizer_steps <= 0:
        raise ValueError("Training configuration produces zero optimizer steps")
    local_batches = optimizer_steps * accumulation
    return optimizer_steps, local_batches


def run(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision("high")
    upstream_root = verify_upstream_checkout(cfg.upstream.repo_root)
    protocol = AgiBotPSGProtocol.from_params(cfg.data.params_path)
    _validate_formal_config(cfg, protocol)
    optimizer_steps, formal_local_batches = _resolve_training_budget(cfg, protocol)
    loader = _build_loader(cfg, protocol)

    effective_action_dim = (
        protocol.action_steps_per_model_step * int(cfg.model.raw_action_dim)
    )
    model, sigreg, load_report = build_vitg_psg_world_model(
        upstream_root,
        checkpoint_path=cfg.model.checkpoint_path,
        grounding_variant=str(cfg.loss.grounding.variant),
        effective_action_dim=effective_action_dim,
        state_dim=int(cfg.model.state_dim),
        joint_dim=len(cfg.loss.grounding.joint_idx),
        velocity_dim=len(cfg.loss.grounding.vel_idx),
        image_size=int(cfg.model.image_size),
        patch_size=int(cfg.model.patch_size),
        max_num_frames=int(cfg.model.max_num_frames),
        tubelet_size=int(cfg.model.tubelet_size),
        model_name=str(cfg.model.model_name),
        embed_dim=int(cfg.wm.embed_dim),
        history_size=int(cfg.wm.history_size),
        predictor_depth=int(cfg.predictor.depth),
        predictor_heads=int(cfg.predictor.heads),
        predictor_mlp_dim=int(cfg.predictor.mlp_dim),
        predictor_dim_head=int(cfg.predictor.dim_head),
        predictor_dropout=float(cfg.predictor.dropout),
        predictor_embedding_dropout=float(cfg.predictor.emb_dropout),
        grounding_hidden_dim=int(cfg.loss.grounding.hidden_dim),
        frame_chunk_size=int(cfg.model.frame_chunk_size),
        use_activation_checkpointing=bool(cfg.model.use_activation_checkpointing),
    )
    module = AgiBotPSGViTGModule(model=model, sigreg=sigreg, cfg=cfg)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    artifacts_callback = ReproducibilityArtifacts(
        output_dir=output_dir,
        cfg=cfg,
        protocol=protocol,
        upstream_root=upstream_root,
        load_report=load_report,
        counts=parameter_counts(model),
        optimizer_steps_per_epoch=optimizer_steps,
        local_batches_per_epoch=formal_local_batches,
        export_encoder=bool(cfg.checkpoint.export_encoder),
    )
    callbacks = [artifacts_callback]
    if bool(cfg.checkpoint.save_last):
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=output_dir / "lightning-checkpoints",
                filename="epoch-{epoch:02d}",
                every_n_epochs=int(cfg.checkpoint.every_n_epochs),
                save_top_k=0,
                save_last=True,
                save_on_train_epoch_end=True,
                save_weights_only=False,
            ),
        )
    logger = CSVLogger(save_dir=str(output_dir), name="csv")
    configured_limit = cfg.trainer.limit_train_batches
    limit_train_batches = (
        formal_local_batches
        if str(configured_limit).lower() == "auto"
        else configured_limit
    )
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices),
        num_nodes=int(cfg.trainer.num_nodes),
        strategy=str(cfg.trainer.strategy),
        max_epochs=int(cfg.trainer.max_epochs),
        precision=str(cfg.trainer.precision),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        limit_train_batches=limit_train_batches,
        enable_checkpointing=bool(cfg.checkpoint.save_last),
        use_distributed_sampler=True,
    )
    resume_path = cfg.checkpoint.get("resume_path")
    trainer.fit(
        module,
        train_dataloaders=loader,
        ckpt_path=str(resume_path) if resume_path else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, e.g. trainer.max_epochs=1",
    )
    arguments = parser.parse_args()
    cfg = OmegaConf.load(arguments.config)
    if arguments.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(arguments.overrides))
    OmegaConf.resolve(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
