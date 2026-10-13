from __future__ import annotations

import copy
import gc
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import app.vjepa_2_1.models.predictor as predictor_vit
import app.vjepa_2_1.models.vision_transformer as encoder_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.masks.utils import apply_masks


LOGGER = logging.getLogger(__name__)


def make_temporal_masks(
    *,
    batch_size: int,
    context_frames: int,
    frames_per_clip: int,
    height: int,
    width: int,
    patch_size: int,
    tubelet_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return time-major keep indices for context and future tubelets.

    IntPhys2's released helper implicitly assumes a 224x224 input.  This
    implementation derives the spatial grid from the actual clip, which is
    required for the 384x384 ViT-G checkpoints used here.
    """
    if context_frames % tubelet_size:
        raise ValueError("context_frames must be divisible by tubelet_size")
    if frames_per_clip % tubelet_size:
        raise ValueError("frames_per_clip must be divisible by tubelet_size")
    if height % patch_size or width % patch_size:
        raise ValueError("Input height and width must be divisible by patch_size")
    if not 0 < context_frames < frames_per_clip:
        raise ValueError("context_frames must be between zero and frames_per_clip")

    spatial_tokens = (height // patch_size) * (width // patch_size)
    temporal_tokens = frames_per_clip // tubelet_size
    context_tubelets = context_frames // tubelet_size
    total_tokens = temporal_tokens * spatial_tokens
    context_tokens = context_tubelets * spatial_tokens
    indices = torch.arange(total_tokens, device=device, dtype=torch.long)
    mask_context = indices[:context_tokens].unsqueeze(0).expand(batch_size, -1)
    mask_future = indices[context_tokens:].unsqueeze(0).expand(batch_size, -1)
    return mask_context, mask_future


def _component_kwargs(model_config: dict[str, Any], frames_per_clip: int) -> dict[str, Any]:
    return {
        "img_size": int(model_config["resolution"]),
        "patch_size": int(model_config["patch_size"]),
        "num_frames": int(frames_per_clip),
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


def build_components(
    model_config: dict[str, Any],
    frames_per_clip: int,
    device: torch.device,
) -> tuple[MultiSeqWrapper, MultiSeqWrapper, PredictorMultiSeqWrapper]:
    model_name = str(model_config["model_name"])
    if model_name not in encoder_vit.__dict__:
        raise ValueError(f"Unknown V-JEPA 2.1 encoder model: {model_name}")

    common = _component_kwargs(model_config, frames_per_clip)
    # Construct directly on the destination device.  This avoids retaining a
    # second 1B-parameter CPU copy while loading a ViT-G checkpoint.
    with torch.device(device):
        encoder = MultiSeqWrapper(encoder_vit.__dict__[model_name](**common))
        target_encoder = copy.deepcopy(encoder)
        predictor = PredictorMultiSeqWrapper(
            predictor_vit.vit_predictor(
                img_size=int(model_config["resolution"]),
                patch_size=int(model_config["patch_size"]),
                num_frames=int(frames_per_clip),
                tubelet_size=int(model_config["tubelet_size"]),
                embed_dim=encoder.backbone.embed_dim,
                predictor_embed_dim=int(model_config["pred_embed_dim"]),
                depth=int(model_config["pred_depth"]),
                num_heads=int(model_config["pred_num_heads"]),
                uniform_power=bool(model_config["uniform_power"]),
                use_mask_tokens=True,
                num_mask_tokens=int(model_config["num_mask_tokens"]),
                zero_init_mask_tokens=True,
                use_rope=bool(model_config["use_rope"]),
                use_sdpa=bool(model_config["use_sdpa"]),
                is_causal=False,
                use_silu=False,
                wide_silu=bool(model_config["wide_silu"]),
                use_activation_checkpointing=False,
                return_all_tokens=True,
                chop_last_n_tokens=0,
                n_registers=0,
                has_cls_first=False,
                interpolate_rope=bool(model_config["interpolate_rope"]),
                modality_embedding=bool(model_config["modality_embedding"]),
                img_temporal_dim_size=model_config.get("img_temporal_dim_size"),
            )
        )
    return encoder, target_encoder, predictor


def build_empty_components(
    model_config: dict[str, Any], frames_per_clip: int
) -> tuple[MultiSeqWrapper, MultiSeqWrapper, PredictorMultiSeqWrapper]:
    """Build shape-only modules for checkpoint compatibility checks."""
    from accelerate import init_empty_weights

    with init_empty_weights():
        encoder, target_encoder, predictor = build_components(
            model_config, frames_per_clip, torch.device("cpu")
        )
    return encoder, target_encoder, predictor


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


def normalized_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = _normalize_checkpoint_key(key)
        if normalized in output:
            raise RuntimeError(f"Checkpoint key collision after prefix removal: {key}")
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


def _format_incompatibility(section: str, report: dict[str, list[Any]]) -> str:
    parts = [f"Checkpoint section {section!r} is incompatible"]
    for key in ("missing", "unexpected", "mismatched"):
        values = report[key]
        if values:
            preview = values[:12]
            suffix = " ..." if len(values) > len(preview) else ""
            parts.append(f"{key} ({len(values)}): {preview}{suffix}")
    return "; ".join(parts)


def _load_checkpoint_file(path: str | Path, mmap: bool) -> dict[str, Any]:
    path = str(Path(path).expanduser().resolve())
    kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        try:
            return torch.load(path, mmap=True, **kwargs)
        except RuntimeError as error:
            LOGGER.warning("Checkpoint mmap failed (%s); falling back to regular load", error)
    return torch.load(path, **kwargs)


def load_checkpoint_strict(
    *,
    checkpoint_path: str | Path,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    strict: bool = True,
    mmap: bool = True,
) -> dict[str, Any]:
    checkpoint = _load_checkpoint_file(checkpoint_path, mmap=mmap)
    required = {
        "encoder": encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
    }
    for section, module in required.items():
        if section not in checkpoint:
            raise KeyError(f"Checkpoint has no {section!r} state dictionary")
        state = checkpoint[section]
        report = compare_state_shapes(module, state)
        incompatible = any(report.values())
        if incompatible and strict:
            raise RuntimeError(_format_incompatibility(section, report))
        if incompatible:
            LOGGER.warning(_format_incompatibility(section, report))
            current = module.state_dict()
            state = {
                key: value
                for key, value in normalized_state_dict(state).items()
                if key in current and tuple(value.shape) == tuple(current[key].shape)
            }
            module.load_state_dict(state, strict=False)
        else:
            module.load_state_dict(normalized_state_dict(state), strict=True)
        LOGGER.info("Loaded %s (%d tensors)", section, len(state))

    metadata = {
        key: checkpoint.get(key)
        for key in ("epoch", "batch_size", "world_size", "loss", "lr")
        if key in checkpoint
    }
    del checkpoint
    gc.collect()
    return metadata


class VJEPA21PredictiveModel(nn.Module):
    """V-JEPA 2.1 encoder/predictor exposed as an IntPhys2 surprise model."""

    def __init__(
        self,
        encoder: MultiSeqWrapper,
        target_encoder: MultiSeqWrapper,
        predictor: PredictorMultiSeqWrapper,
        model_config: dict[str, Any],
        frames_per_clip: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.frames_per_clip = int(frames_per_clip)
        self.patch_size = int(model_config["patch_size"])
        self.tubelet_size = int(model_config["tubelet_size"])
        self.embed_dim = int(encoder.backbone.embed_dim)
        self.levels_predictor = int(model_config["levels_predictor"])

    def encode_target(self, clips: torch.Tensor) -> torch.Tensor:
        target = self.target_encoder(
            [clips], masks=None, gram_mode=False, training_mode=True
        )[0]
        expected_dim = self.embed_dim * self.levels_predictor
        if target.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Target encoder returned D={target.shape[-1]}, expected {expected_dim} "
                f"({self.levels_predictor} x {self.embed_dim})"
            )
        chunks = target.split(self.embed_dim, dim=-1)
        return torch.cat(
            [F.layer_norm(chunk, (self.embed_dim,)) for chunk in chunks], dim=-1
        )

    def surprise_from_target(
        self,
        clips: torch.Tensor,
        normalized_target: torch.Tensor,
        context_frames: int,
    ) -> torch.Tensor:
        batch_size, _, frames, height, width = clips.shape
        if frames != self.frames_per_clip:
            raise ValueError(
                f"Expected {self.frames_per_clip} frames, received {frames}"
            )
        mask_context, mask_future = make_temporal_masks(
            batch_size=batch_size,
            context_frames=int(context_frames),
            frames_per_clip=frames,
            height=height,
            width=width,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            device=clips.device,
        )
        masks_context = [[mask_context]]
        masks_future = [[mask_future]]
        target = apply_masks(normalized_target, [mask_future], concat=False)[0]
        context = self.encoder(
            [clips],
            masks=masks_context,
            gram_mode=False,
            training_mode=True,
        )
        predictions, _ = self.predictor(
            context, masks_context, masks_future, mod="video"
        )
        prediction = predictions[0][0]
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Prediction shape {tuple(prediction.shape)} does not match target "
                f"shape {tuple(target.shape)}"
            )
        return F.l1_loss(prediction, target, reduction="none").mean(dim=(1, 2))


def initialize_model(config: dict[str, Any], device: torch.device) -> VJEPA21PredictiveModel:
    model_config = config["model"]
    frames_per_clip = int(config["evaluation"]["frames_per_clip"])
    encoder, target_encoder, predictor = build_components(
        model_config, frames_per_clip, device
    )
    metadata = load_checkpoint_strict(
        checkpoint_path=config["checkpoint"],
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        strict=bool(model_config["strict_checkpoint"]),
        mmap=bool(model_config["checkpoint_mmap"]),
    )
    LOGGER.info("Checkpoint metadata: %s", metadata)

    model = VJEPA21PredictiveModel(
        encoder, target_encoder, predictor, model_config, frames_per_clip
    )
    if model_config["weights_dtype"] == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
