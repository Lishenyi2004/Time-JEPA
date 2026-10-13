"""Frozen ViT-g feature adapter for the formal PSG ports.

The training export contains the tuned V-JEPA image branch, the official PSG
projector, and the training-only grounding heads.  Probes intentionally retain
only the first two components.  Each observed frame is encoded independently
with the native T=1 image stem, then adjacent projected frame features are
averaged onto the common 32-tubelet probe grid.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.vjepa_2_1.models import vision_transformer as video_vit

from .modeling import import_upstream_modules, verify_upstream_checkout
from .probe_adapter import group_adjacent_frames
from .vitg_modeling import VJEPAFrameEncoder


PSG_VITG_INPUT_FRAMES = 64
PSG_VITG_INPUT_SIZE = 384
PSG_VITG_FEATURE_DIM = 1408
PSG_VITG_TARGET_GROUP_SIZE = 2
PSG_VITG_MODEL_NAME = "vit_giant_xformers"


def _torch_load_export(path: Path) -> Any:
    """Load the tensor-only export without materializing a second 4 GB copy."""

    try:
        return torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(str(path), map_location="cpu")


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    with path.open("r") as handle:
        payload = json.load(handle)
    return {"path": str(path), "present": True, "payload": payload}


def _module_parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _state_parameter_count(state: Mapping[str, Any]) -> int:
    return sum(int(value.numel()) for value in state.values() if torch.is_tensor(value))


def build_vitg_image_backbone(*, use_activation_checkpointing: bool = False) -> nn.Module:
    """Recreate the exact ViT-g image branch used by PSG-g training."""

    return video_vit.__dict__[PSG_VITG_MODEL_NAME](
        img_size=PSG_VITG_INPUT_SIZE,
        patch_size=16,
        num_frames=PSG_VITG_INPUT_FRAMES,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=bool(use_activation_checkpointing),
        is_causal=False,
        use_rope=True,
        init_type="default",
        img_temporal_dim_size=1,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=True,
        modality_embedding=True,
    )


def _normalize_video_layout(
    videos: torch.Tensor,
    *,
    input_frames: int,
    input_size: int,
) -> torch.Tensor:
    """Return shared-probe input as [B,T,C,H,W] without changing values."""

    if videos.ndim != 5:
        raise ValueError(f"PSG-g expects a 5-D video tensor, got {tuple(videos.shape)}")
    if videos.shape[1] == 3:
        videos = videos.permute(0, 2, 1, 3, 4)
    elif videos.shape[2] != 3:
        raise ValueError(
            "PSG-g input must be [B,C,T,H,W] or [B,T,C,H,W] with C=3; "
            f"got {tuple(videos.shape)}"
        )
    if videos.shape[1] != int(input_frames):
        raise ValueError(
            f"PSG-g probe requires {input_frames} frames, got {videos.shape[1]}"
        )
    if tuple(videos.shape[-2:]) != (int(input_size), int(input_size)):
        raise ValueError(
            f"PSG-g probe requires {input_size}x{input_size} input, "
            f"got {tuple(videos.shape[-2:])}"
        )
    return videos


def project_vitg_video_frames(
    encoder: nn.Module,
    projector: nn.Module,
    videos: torch.Tensor,
    *,
    frame_microbatch_size: int,
    input_frames: int = PSG_VITG_INPUT_FRAMES,
    input_size: int = PSG_VITG_INPUT_SIZE,
    feature_dim: int = PSG_VITG_FEATURE_DIM,
) -> torch.Tensor:
    """Encode frames independently while bounding the temporary frame batch.

    Temporal and batch chunks are only an execution detail: projected features
    are restored to their original [B,T,D] order before any grouping.
    """

    if int(frame_microbatch_size) <= 0:
        raise ValueError("frame_microbatch_size must be positive")
    videos = _normalize_video_layout(
        videos,
        input_frames=input_frames,
        input_size=input_size,
    )
    batch, frames, channels, height, width = videos.shape
    projected_batch_chunks = []
    for batch_start in range(0, batch, int(frame_microbatch_size)):
        batch_stop = min(batch, batch_start + int(frame_microbatch_size))
        batch_chunk = videos[batch_start:batch_stop]
        batch_count = batch_stop - batch_start
        frames_per_chunk = max(1, int(frame_microbatch_size) // batch_count)
        projected_time_chunks = []
        for frame_start in range(0, frames, frames_per_chunk):
            frame_stop = min(frames, frame_start + frames_per_chunk)
            frame_count = frame_stop - frame_start
            flat_frames = (
                batch_chunk[:, frame_start:frame_stop]
                .reshape(batch_count * frame_count, channels, height, width)
            )
            output = encoder(
                flat_frames,
                proprio=None,
                interpolate_pos_encoding=True,
            )
            hidden = output.last_hidden_state
            if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (1, int(feature_dim)):
                raise RuntimeError(
                    "PSG-g image encoder returned an unexpected pooled tensor: "
                    f"{tuple(hidden.shape)}"
                )
            projected = projector(hidden[:, 0])
            if tuple(projected.shape) != (
                batch_count * frame_count,
                int(feature_dim),
            ):
                raise RuntimeError(
                    "Official PSG projector returned an unexpected tensor: "
                    f"{tuple(projected.shape)}"
                )
            projected_time_chunks.append(
                projected.reshape(batch_count, frame_count, int(feature_dim))
            )
        projected_batch_chunks.append(torch.cat(projected_time_chunks, dim=1))
    result = torch.cat(projected_batch_chunks, dim=0)
    expected_shape = (batch, frames, int(feature_dim))
    if tuple(result.shape) != expected_shape:
        raise RuntimeError(
            f"PSG-g frame projection produced {tuple(result.shape)}, expected {expected_shape}"
        )
    return result


class PSGJEPAViTGFeatureAdapter(nn.Module):
    """Frozen tuned ViT-g image encoder plus the authors' trained projector."""

    _EXPECTED_EXPERIMENTS = {
        "psg_static": "PSG-Static-g-Port",
        "psg_full": "PSG-Full-g-Port",
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        upstream_root: str | Path,
        expected_variant: str,
        frame_microbatch_size: int = 32,
    ):
        super().__init__()
        if expected_variant not in self._EXPECTED_EXPERIMENTS:
            raise ValueError(
                "ViT-g PSG probes require expected_variant='psg_static' or "
                f"'psg_full', got {expected_variant!r}"
            )
        if int(frame_microbatch_size) <= 0:
            raise ValueError("frame_microbatch_size must be positive")

        upstream_root = verify_upstream_checkout(upstream_root)
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PSG-g encoder export not found: {checkpoint_path}")
        checkpoint = _torch_load_export(checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"PSG-g export must be a mapping: {checkpoint_path}")
        required_keys = {"encoder", "projector", "grounding", "meta"}
        missing_top_level = sorted(required_keys - set(checkpoint))
        if missing_top_level:
            raise KeyError(
                f"PSG-g export is missing top-level keys {missing_top_level}: {checkpoint_path}"
            )
        encoder_state = checkpoint["encoder"]
        projector_state = checkpoint["projector"]
        grounding_state = checkpoint["grounding"]
        meta = dict(checkpoint["meta"])
        if not all(
            isinstance(value, Mapping)
            for value in (encoder_state, projector_state, grounding_state)
        ):
            raise TypeError("PSG-g encoder/projector/grounding exports must be mappings")

        expected_meta = {
            "experiment": self._EXPECTED_EXPERIMENTS[expected_variant],
            "model_name": PSG_VITG_MODEL_NAME,
            "image_size": PSG_VITG_INPUT_SIZE,
            "visual_temporal_input": 1,
            "pooling": "spatial_mean",
            "embedding_dim": PSG_VITG_FEATURE_DIM,
            "upstream_world_model_class": "psgjepa.model.CrossModalJEPA_v2",
        }
        mismatches = {
            key: {"observed": meta.get(key), "expected": expected}
            for key, expected in expected_meta.items()
            if meta.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"PSG-g export metadata mismatch: {mismatches}")

        backbone = build_vitg_image_backbone(use_activation_checkpointing=False)
        encoder_incompatible = backbone.load_state_dict(encoder_state, strict=True)
        upstream = import_upstream_modules(upstream_root)
        projector = upstream.MLP(
            input_dim=PSG_VITG_FEATURE_DIM,
            output_dim=PSG_VITG_FEATURE_DIM,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d,
        )
        projector_incompatible = projector.load_state_dict(projector_state, strict=True)

        self.encoder = VJEPAFrameEncoder(
            backbone,
            frame_chunk_size=int(frame_microbatch_size),
        )
        self.projector = projector
        self.frame_microbatch_size = int(frame_microbatch_size)
        training_manifest = _read_json_if_present(checkpoint_path.parent / "run-manifest.json")
        self.load_report = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_format": "encoder_final.pt tensor export",
            "variant": expected_variant,
            "checkpoint_meta": meta,
            "training_manifest": training_manifest,
            "upstream_root": str(upstream_root),
            "retained_modules": ["encoder.image_backbone", "projector"],
            "excluded_export_modules": ["grounding"],
            "excluded_grounding_keys": sorted(str(key) for key in grounding_state),
            "excluded_grounding_parameters": _state_parameter_count(grounding_state),
            "encoder_missing_keys": list(encoder_incompatible.missing_keys),
            "encoder_unexpected_keys": list(encoder_incompatible.unexpected_keys),
            "projector_missing_keys": list(projector_incompatible.missing_keys),
            "projector_unexpected_keys": list(projector_incompatible.unexpected_keys),
            "encoder_parameters": _module_parameter_count(backbone),
            "projector_parameters": _module_parameter_count(projector),
            "input_layout": "shared [B,C,64,384,384] -> 64 independent T=1 images",
            "frame_feature_layout": "[B,64,1408] after spatial mean and official projector",
            "formal_output_layout": "[B,32,1408] after adjacent-frame mean",
            "frame_microbatch_size": self.frame_microbatch_size,
        }
        del checkpoint, encoder_state, projector_state, grounding_state

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def project_frames(self, videos: torch.Tensor) -> torch.Tensor:
        return project_vitg_video_frames(
            self.encoder,
            self.projector,
            videos,
            frame_microbatch_size=self.frame_microbatch_size,
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        return group_adjacent_frames(
            self.project_frames(videos),
            group_size=PSG_VITG_TARGET_GROUP_SIZE,
        )
