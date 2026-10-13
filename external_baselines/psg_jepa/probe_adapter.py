"""Frozen PSG-JEPA feature and original-state-head adapters.

The checkpoint is the complete ``CrossModalJEPA_v2`` object emitted by
``train.py``.  Formal probes retain the authors' image encoder and projector,
discard the action/prediction path, and expose one feature per two sampled
frames.  The diagnostic adapter additionally retains the authors' trained
state head and applies it at the original per-frame granularity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
import torch.nn as nn

from .modeling import verify_upstream_checkout
from .protocol import EXPECTED_UPSTREAM_COMMIT


PSG_INPUT_FRAMES = 64
PSG_INPUT_SIZE = 224
PSG_FEATURE_DIM = 192
PSG_TARGET_GROUP_SIZE = 2
PSG_STATE_DIM = 18


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_training_manifest(checkpoint_path: Path) -> dict:
    manifest_path = checkpoint_path.parent / "run-manifest.json"
    if not manifest_path.is_file():
        return {"path": str(manifest_path), "present": False}
    with manifest_path.open("r") as handle:
        payload = json.load(handle)
    return {"path": str(manifest_path), "present": True, "payload": payload}


def _normalize_video_layout(videos: torch.Tensor) -> torch.Tensor:
    """Return shared-probe input as ``[B,T,C,H,W]`` without changing values."""

    if videos.ndim != 5:
        raise ValueError(f"PSG-JEPA expects a 5-D video tensor, got {tuple(videos.shape)}")
    if videos.shape[1] == 3:
        videos = videos.permute(0, 2, 1, 3, 4)
    elif videos.shape[2] != 3:
        raise ValueError(
            "PSG-JEPA input must be [B,C,T,H,W] or [B,T,C,H,W] with C=3; "
            f"got {tuple(videos.shape)}"
        )
    if videos.shape[1] != PSG_INPUT_FRAMES:
        raise ValueError(
            f"PSG-JEPA probe requires {PSG_INPUT_FRAMES} frames, got {videos.shape[1]}"
        )
    if tuple(videos.shape[-2:]) != (PSG_INPUT_SIZE, PSG_INPUT_SIZE):
        raise ValueError(
            f"PSG-JEPA probe requires {PSG_INPUT_SIZE}x{PSG_INPUT_SIZE} input, "
            f"got {tuple(videos.shape[-2:])}"
        )
    return videos


def group_adjacent_frames(values: torch.Tensor, group_size: int = PSG_TARGET_GROUP_SIZE) -> torch.Tensor:
    """Average adjacent frame-aligned values into the common tubelet timeline."""

    if values.ndim != 3:
        raise ValueError(f"Expected frame values [B,T,D], got {tuple(values.shape)}")
    if group_size <= 0 or values.shape[1] % group_size:
        raise ValueError(
            f"Cannot group T={values.shape[1]} frames with group_size={group_size}"
        )
    batch, frames, dim = values.shape
    return values.reshape(batch, frames // group_size, group_size, dim).mean(dim=2)


def project_video_frames(
    encoder: nn.Module,
    projector: nn.Module,
    videos: torch.Tensor,
    *,
    frame_microbatch_size: int,
) -> torch.Tensor:
    """Run the official 2-D ViT and projector independently on all 64 frames."""

    if frame_microbatch_size <= 0:
        raise ValueError("frame_microbatch_size must be positive")
    videos = _normalize_video_layout(videos)
    batch, frames, channels, height, width = videos.shape
    flat = videos.reshape(batch * frames, channels, height, width)
    projected_chunks = []
    for start in range(0, flat.shape[0], frame_microbatch_size):
        output = encoder(
            flat[start : start + frame_microbatch_size],
            proprio=None,
            interpolate_pos_encoding=True,
        )
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != PSG_FEATURE_DIM:
            raise RuntimeError(
                "Official PSG encoder returned an unexpected token tensor: "
                f"{tuple(hidden.shape)}"
            )
        projected_chunks.append(projector(hidden[:, 0]))
    projected = torch.cat(projected_chunks, dim=0)
    if projected.shape != (batch * frames, PSG_FEATURE_DIM):
        raise RuntimeError(
            "Official PSG projector returned an unexpected tensor: "
            f"{tuple(projected.shape)}"
        )
    return projected.reshape(batch, frames, PSG_FEATURE_DIM)


def apply_state_head_per_frame(
    frame_features: torch.Tensor,
    state_head: nn.Module,
    *,
    group_size: int = PSG_TARGET_GROUP_SIZE,
) -> torch.Tensor:
    """Apply the nonlinear official head before temporal target grouping."""

    if frame_features.ndim != 3 or frame_features.shape[-1] != PSG_FEATURE_DIM:
        raise ValueError(
            f"Expected PSG frame features [B,T,{PSG_FEATURE_DIM}], "
            f"got {tuple(frame_features.shape)}"
        )
    batch, frames, dim = frame_features.shape
    predictions = state_head(frame_features.reshape(batch * frames, dim))
    if predictions.shape != (batch * frames, PSG_STATE_DIM):
        raise RuntimeError(
            "Official PSG state head returned an unexpected tensor: "
            f"{tuple(predictions.shape)}"
        )
    predictions = predictions.reshape(batch, frames, PSG_STATE_DIM)
    return group_adjacent_frames(predictions, group_size=group_size)


def _module_parameter_count(module: nn.Module | None) -> int:
    return 0 if module is None else sum(parameter.numel() for parameter in module.parameters())


def _validate_loaded_model(model: nn.Module, expected_variant: str) -> tuple[str, nn.Module | None]:
    if type(model).__module__ != "psgjepa.model" or type(model).__name__ != "CrossModalJEPA_v2":
        raise TypeError(
            "Checkpoint is not the official psgjepa.model.CrossModalJEPA_v2 object: "
            f"got {type(model).__module__}.{type(model).__name__}"
        )
    for name in ("encoder", "projector", "predictor", "action_encoder", "pred_proj"):
        if not isinstance(getattr(model, name, None), nn.Module):
            raise TypeError(f"Official PSG checkpoint is missing module {name!r}")
    if bool(getattr(model, "use_state", True)):
        raise ValueError("AgiBot PSG checkpoints must be image-only encoders (use_state=False)")

    config = getattr(model.encoder, "config", None)
    observed = {
        "hidden_size": int(getattr(config, "hidden_size", -1)),
        "image_size": int(getattr(config, "image_size", -1)),
        "patch_size": int(getattr(config, "patch_size", -1)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", -1)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
    }
    expected = {
        "hidden_size": PSG_FEATURE_DIM,
        "image_size": PSG_INPUT_SIZE,
        "patch_size": 14,
        "num_hidden_layers": 12,
        "num_attention_heads": 3,
    }
    if observed != expected:
        raise ValueError(f"Unexpected PSG ViT-Tiny contract: got {observed}, expected {expected}")

    grounding = getattr(model, "grounding", None)
    state_head = getattr(grounding, "state_head", None) if grounding is not None else None
    actual_variant = "psg_static" if isinstance(state_head, nn.Module) else "lewm"
    if expected_variant not in ("psg_static", "lewm"):
        raise ValueError(f"Unsupported expected_variant={expected_variant!r}")
    if actual_variant != expected_variant:
        raise ValueError(
            f"Checkpoint variant mismatch: detected {actual_variant}, expected {expected_variant}"
        )
    return actual_variant, state_head


class PSGJEPAFeatureAdapter(nn.Module):
    """Frozen official PSG/LeWM encoder plus the authors' projector."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        upstream_root: str | Path,
        expected_variant: str,
        frame_microbatch_size: int = 128,
        retain_state_head: bool = False,
    ):
        super().__init__()
        upstream_root = verify_upstream_checkout(upstream_root)
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PSG-JEPA official model checkpoint not found: {checkpoint_path}")

        # This is a locally generated, explicitly trusted complete nn.Module
        # checkpoint. Object deserialization is required to preserve the exact
        # authors' model; the class and architecture are validated immediately.
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actual_variant, state_head = _validate_loaded_model(model, expected_variant)
        manifest = _read_training_manifest(checkpoint_path)
        if manifest.get("present"):
            payload = manifest["payload"]
            if payload.get("upstream_commit") != EXPECTED_UPSTREAM_COMMIT:
                raise ValueError(
                    "Training manifest upstream commit mismatch: "
                    f"{payload.get('upstream_commit')} != {EXPECTED_UPSTREAM_COMMIT}"
                )

        self.encoder = model.encoder
        self.projector = model.projector
        if retain_state_head:
            if state_head is None:
                raise ValueError("retain_state_head=True requires a PSG-static checkpoint")
            self.state_head = state_head
        else:
            self.state_head = None
        self.frame_microbatch_size = int(frame_microbatch_size)
        if self.frame_microbatch_size <= 0:
            raise ValueError("frame_microbatch_size must be positive")

        discarded = {
            "predictor": _module_parameter_count(model.predictor),
            "action_encoder": _module_parameter_count(model.action_encoder),
            "prediction_projector": _module_parameter_count(model.pred_proj),
            "state_head": 0 if retain_state_head else _module_parameter_count(state_head),
        }
        self.load_report = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_object_class": f"{type(model).__module__}.{type(model).__name__}",
            "variant": actual_variant,
            "upstream_root": str(upstream_root),
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
            "training_manifest": manifest,
            "retained_modules": [
                "encoder",
                "projector",
                *(["grounding.state_head"] if retain_state_head else []),
            ],
            "discarded_module_parameters": discarded,
            "encoder_parameters": _module_parameter_count(self.encoder),
            "projector_parameters": _module_parameter_count(self.projector),
            "retained_state_head_parameters": _module_parameter_count(self.state_head),
            "input_layout": "shared [B,C,T,H,W] -> official per-frame [B*T,C,H,W]",
            "input_frames": PSG_INPUT_FRAMES,
            "input_size": PSG_INPUT_SIZE,
            "frame_feature_dim": PSG_FEATURE_DIM,
            "formal_output_layout": "[B,32,192], adjacent-frame mean after projector",
            "frame_microbatch_size": self.frame_microbatch_size,
        }
        del model
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def project_frames(self, videos: torch.Tensor) -> torch.Tensor:
        return project_video_frames(
            self.encoder,
            self.projector,
            videos,
            frame_microbatch_size=self.frame_microbatch_size,
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        return group_adjacent_frames(self.project_frames(videos))


class PSGJEPAStatePredictionAdapter(nn.Module):
    """Return original PSG state-head predictions on the common 32-step grid."""

    def __init__(self, feature_adapter: PSGJEPAFeatureAdapter):
        super().__init__()
        if not isinstance(feature_adapter.state_head, nn.Module):
            raise ValueError("State prediction requires a retained official state head")
        self.feature_adapter = feature_adapter
        self.load_report = {
            **feature_adapter.load_report,
            "output_layout": "[B,32,18] normalized state prediction",
            "head_order": "official nonlinear state head per frame, then adjacent-frame mean",
        }

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        frame_features = self.feature_adapter.project_frames(videos)
        return apply_state_head_per_frame(
            frame_features,
            self.feature_adapter.state_head,
        )
