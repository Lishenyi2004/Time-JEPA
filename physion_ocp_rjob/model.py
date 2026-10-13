from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import app.vjepa_2_1.models.vision_transformer as encoder_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper


LOGGER = logging.getLogger(__name__)
VIT_GIANT_EMBED_DIM = 1408


def feature_dimension(config: dict[str, Any], embed_dim: int = VIT_GIANT_EMBED_DIM) -> int:
    model = config["model"]
    extraction = config["extraction"]
    temporal_tokens = int(extraction["num_frames"]) // int(model["tubelet_size"])
    spatial_tokens = (int(model["resolution"]) // int(model["patch_size"])) ** 2
    aggregation = str(extraction["aggregation"])
    if aggregation == "flatten_tokens":
        return temporal_tokens * spatial_tokens * int(embed_dim)
    if aggregation == "spatial_mean":
        return temporal_tokens * int(embed_dim)
    if aggregation == "global_mean":
        return int(embed_dim)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def _component_kwargs(model_config: dict[str, Any], num_frames: int) -> dict[str, Any]:
    return {
        "img_size": int(model_config["resolution"]),
        "patch_size": int(model_config["patch_size"]),
        "num_frames": int(num_frames),
        "tubelet_size": int(model_config["tubelet_size"]),
        "uniform_power": bool(model_config["uniform_power"]),
        "use_sdpa": bool(model_config["use_sdpa"]),
        "use_silu": False,
        "wide_silu": bool(model_config["wide_silu"]),
        "use_activation_checkpointing": False,
        "is_causal": False,
        "use_rope": bool(model_config["use_rope"]),
        "img_temporal_dim_size": model_config.get("img_temporal_dim_size"),
        "n_registers": 0,
        "has_cls_first": False,
        "interpolate_rope": bool(model_config["interpolate_rope"]),
        "modality_embedding": bool(model_config["modality_embedding"]),
    }


def build_target_encoder(
    model_config: dict[str, Any], num_frames: int, device: torch.device
) -> MultiSeqWrapper:
    model_name = str(model_config["model_name"])
    if model_name not in encoder_vit.__dict__:
        raise ValueError(f"Unknown V-JEPA 2.1 encoder model: {model_name}")
    with torch.device(device):
        backbone = encoder_vit.__dict__[model_name](
            **_component_kwargs(model_config, num_frames)
        )
        return MultiSeqWrapper(backbone)


def build_empty_target_encoder(
    model_config: dict[str, Any], num_frames: int
) -> MultiSeqWrapper:
    from accelerate import init_empty_weights

    with init_empty_weights():
        return build_target_encoder(model_config, num_frames, torch.device("cpu"))


def _normalize_checkpoint_key(key: str) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def normalized_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = _normalize_checkpoint_key(key)
        if normalized in output:
            raise RuntimeError(f"Checkpoint key collision after normalization: {key}")
        output[normalized] = value
    return output


def compare_state_shapes(
    module: nn.Module, checkpoint_state: dict[str, torch.Tensor]
) -> dict[str, list[Any]]:
    expected = {key: tuple(value.shape) for key, value in module.state_dict().items()}
    provided = {
        key: tuple(value.shape)
        for key, value in normalized_state_dict(checkpoint_state).items()
    }
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    mismatched = sorted(
        (key, expected[key], provided[key])
        for key in set(expected) & set(provided)
        if expected[key] != provided[key]
    )
    return {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}


def _format_incompatibility(report: dict[str, list[Any]]) -> str:
    parts = ["Checkpoint target_encoder is incompatible"]
    for key in ("missing", "unexpected", "mismatched"):
        values = report[key]
        if values:
            preview = values[:12]
            suffix = " ..." if len(values) > len(preview) else ""
            parts.append(f"{key} ({len(values)}): {preview}{suffix}")
    return "; ".join(parts)


def load_checkpoint_file(path: str | Path, mmap: bool) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        try:
            return torch.load(resolved, mmap=True, **kwargs)
        except RuntimeError as error:
            LOGGER.warning("Checkpoint mmap failed (%s); using regular load", error)
    return torch.load(resolved, **kwargs)


def load_target_encoder_strict(
    checkpoint_path: str | Path,
    target_encoder: MultiSeqWrapper,
    *,
    strict: bool = True,
    mmap: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint_file(checkpoint_path, mmap=mmap)
    if "target_encoder" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} has no 'target_encoder' section")
    state = checkpoint["target_encoder"]
    report = compare_state_shapes(target_encoder, state)
    if any(report.values()):
        if strict:
            raise RuntimeError(_format_incompatibility(report))
        LOGGER.warning(_format_incompatibility(report))
        current = target_encoder.state_dict()
        state = {
            key: value
            for key, value in normalized_state_dict(state).items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        target_encoder.load_state_dict(state, strict=False)
    else:
        target_encoder.load_state_dict(normalized_state_dict(state), strict=True)
    metadata = {
        key: checkpoint.get(key)
        for key in ("epoch", "batch_size", "world_size", "loss", "lr")
        if key in checkpoint
    }
    metadata["target_encoder_tensors"] = len(state)
    del checkpoint, state
    gc.collect()
    return metadata


class VJEPA21OCPEncoder(nn.Module):
    def __init__(
        self,
        target_encoder: MultiSeqWrapper,
        *,
        num_frames: int,
        patch_size: int,
        tubelet_size: int,
        resolution: int,
        aggregation: str,
    ) -> None:
        super().__init__()
        self.target_encoder = target_encoder
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.resolution = int(resolution)
        self.aggregation = str(aggregation)
        self.embed_dim = int(target_encoder.backbone.embed_dim)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5:
            raise ValueError(f"Expected clips [B,C,T,H,W], got {tuple(clips.shape)}")
        if clips.shape[2] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {clips.shape[2]}")
        tokens = self.target_encoder(
            [clips], masks=None, gram_mode=False, training_mode=False
        )[0]
        return aggregate_tokens(
            tokens,
            aggregation=self.aggregation,
            num_frames=self.num_frames,
            resolution=self.resolution,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )


def aggregate_tokens(
    tokens: torch.Tensor,
    *,
    aggregation: str,
    num_frames: int,
    resolution: int,
    patch_size: int,
    tubelet_size: int,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,D], got {tuple(tokens.shape)}")
    temporal_tokens = int(num_frames) // int(tubelet_size)
    spatial_tokens = (int(resolution) // int(patch_size)) ** 2
    expected_tokens = temporal_tokens * spatial_tokens
    if tokens.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Encoder returned {tokens.shape[1]} tokens; expected {temporal_tokens}x"
            f"{spatial_tokens}={expected_tokens}"
        )
    if aggregation == "flatten_tokens":
        return tokens.flatten(1)
    if aggregation == "global_mean":
        return tokens.float().mean(dim=1)
    if aggregation == "spatial_mean":
        batch, _, dim = tokens.shape
        return (
            tokens.reshape(batch, temporal_tokens, spatial_tokens, dim)
            .float()
            .mean(dim=2)
            .flatten(1)
        )
    raise ValueError(f"Unknown aggregation: {aggregation}")


def initialize_encoder(config: dict[str, Any], device: torch.device) -> VJEPA21OCPEncoder:
    model_config = config["model"]
    extraction = config["extraction"]
    target_encoder = build_target_encoder(
        model_config, int(extraction["num_frames"]), device
    )
    metadata = load_target_encoder_strict(
        config["checkpoint"],
        target_encoder,
        strict=bool(model_config["strict_checkpoint"]),
        mmap=bool(model_config["checkpoint_mmap"]),
    )
    LOGGER.info("Loaded target encoder metadata: %s", metadata)
    model = VJEPA21OCPEncoder(
        target_encoder,
        num_frames=int(extraction["num_frames"]),
        patch_size=int(model_config["patch_size"]),
        tubelet_size=int(model_config["tubelet_size"]),
        resolution=int(model_config["resolution"]),
        aggregation=str(extraction["aggregation"]),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
