"""PSG-JEPA objective modules on the released V-JEPA 2.1 ViT-g image branch.

Only the visual encoder is replaced.  PSG's projector, action embedder,
autoregressive predictor, SIGReg module, and grounding heads are imported from
the pinned authors' checkout.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from app.vjepa_2_1.models import vision_transformer as video_vit

from .modeling import StaticGroundingOnly, import_upstream_modules


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_path: str
    checkpoint_key: str
    matched_keys: int
    model_keys: int
    matched_numel: int
    model_numel: int
    matched_fraction: float
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_encoder_prefixes(key: str) -> str:
    """Map released/DDP encoder keys onto a bare VisionTransformer."""

    key = str(key)
    prefixes = ("module.", "encoder.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _load_checkpoint_file(path: Path) -> Any:
    # mmap keeps the unused target-encoder and predictor tensors in the 16 GB
    # release file off resident memory.  The fallback supports older PyTorch.
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_vjepa_encoder_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str | Path,
    *,
    minimum_matched_fraction: float = 0.95,
) -> CheckpointLoadReport:
    """Load only the released online encoder into a bare V-JEPA backbone."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V-JEPA checkpoint not found: {checkpoint_path}")
    checkpoint = _load_checkpoint_file(checkpoint_path)
    checkpoint_key = "root"
    source = checkpoint
    if isinstance(checkpoint, Mapping):
        for candidate in ("encoder", "target_encoder", "state_dict"):
            value = checkpoint.get(candidate)
            if isinstance(value, Mapping):
                source = value
                checkpoint_key = candidate
                break
    if not isinstance(source, Mapping):
        raise TypeError(
            f"Checkpoint {checkpoint_path} does not contain a state mapping"
        )

    current = backbone.state_dict()
    matched = {}
    source_tensor_keys = []
    for raw_key, value in source.items():
        if not torch.is_tensor(value):
            continue
        source_tensor_keys.append(str(raw_key))
        key = _strip_encoder_prefixes(str(raw_key))
        if key in current and tuple(value.shape) == tuple(current[key].shape):
            matched[key] = value

    if not matched:
        raise RuntimeError(
            f"No encoder tensors from {checkpoint_path} match the ViT-g model"
        )
    matched_numel = sum(int(current[key].numel()) for key in matched)
    model_numel = sum(int(value.numel()) for value in current.values())
    matched_fraction = matched_numel / max(1, model_numel)
    required_groups = {
        "native image patch embed": lambda key: key.startswith("patch_embed_img."),
        "image modality embedding": lambda key: key == "img_mod_embed",
        "first transformer block": lambda key: key.startswith("blocks.0."),
        "last transformer block": lambda key: key.startswith("blocks.39."),
        "final encoder normalization": lambda key: key.startswith("norms_block.3."),
    }
    missing_groups = [
        name
        for name, predicate in required_groups.items()
        if not any(predicate(key) for key in matched)
    ]
    if missing_groups:
        raise RuntimeError(
            "Released checkpoint is missing required ViT-g image-encoder groups: "
            f"{missing_groups}"
        )
    if matched_fraction < float(minimum_matched_fraction):
        raise RuntimeError(
            "Too little of the ViT-g backbone was initialized: "
            f"{matched_fraction:.3%} < {minimum_matched_fraction:.3%}"
        )

    incompatible = backbone.load_state_dict(matched, strict=False)
    source_normalized = {_strip_encoder_prefixes(key) for key in source_tensor_keys}
    unexpected = tuple(sorted(source_normalized - set(current)))
    report = CheckpointLoadReport(
        checkpoint_path=str(checkpoint_path),
        checkpoint_key=checkpoint_key,
        matched_keys=len(matched),
        model_keys=len(current),
        matched_numel=matched_numel,
        model_numel=model_numel,
        matched_fraction=matched_fraction,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=unexpected,
    )
    del matched, source, checkpoint
    gc.collect()
    return report


class VJEPAFrameEncoder(nn.Module):
    """Adapt V-JEPA image tokens to the encoder API used by official PSG."""

    def __init__(self, backbone: nn.Module, *, frame_chunk_size: int = 8):
        super().__init__()
        if int(frame_chunk_size) <= 0:
            raise ValueError("frame_chunk_size must be positive")
        if getattr(backbone, "img_temporal_dim_size", None) != 1:
            raise ValueError(
                "PSG-g requires V-JEPA's native T=1 image branch "
                "(img_temporal_dim_size=1)"
            )
        self.backbone = backbone
        self.frame_chunk_size = int(frame_chunk_size)

    @property
    def embed_dim(self) -> int:
        return int(self.backbone.embed_dim)

    def forward(
        self,
        pixels: torch.Tensor,
        proprio=None,
        interpolate_pos_encoding: bool = True,
    ):
        del interpolate_pos_encoding
        if proprio is not None:
            raise ValueError("The formal PSG-g encoder input is pixels-only")
        if pixels.ndim != 4:
            raise ValueError(
                f"Expected flattened PSG frames [N,C,H,W], got {tuple(pixels.shape)}"
            )
        _, channels, _, _ = pixels.shape
        if channels != 3:
            raise ValueError(f"Expected RGB input, got C={channels}")
        pooled_chunks = []
        for start in range(0, len(pixels), self.frame_chunk_size):
            # A 5-D T=1 tensor selects patch_embed_img and mode='img'.  No
            # temporal token or neighboring frame enters this encoder call.
            frame_chunk = pixels[start : start + self.frame_chunk_size].unsqueeze(2)
            tokens = self.backbone(frame_chunk, masks=None, training=False)
            if not torch.is_tensor(tokens) or tokens.ndim != 3:
                raise RuntimeError(
                    "V-JEPA image encoder must return [N,spatial_tokens,D], got "
                    f"{type(tokens).__name__} / "
                    f"{getattr(tokens, 'shape', None)}"
                )
            pooled_chunks.append(tokens.mean(dim=1))
        pooled = torch.cat(pooled_chunks, dim=0)
        # CrossModalJEPA_v2 consumes token 0 as its per-frame representation.
        # A spatial mean is exposed as that single token because V-JEPA has no
        # CLS token.  late_proprio=None follows the official encoder contract.
        return SimpleNamespace(
            last_hidden_state=pooled.unsqueeze(1),
            late_proprio=None,
        )


def build_vitg_psg_world_model(
    repo_root: str | Path,
    *,
    checkpoint_path: str | Path,
    grounding_variant: str,
    effective_action_dim: int,
    state_dim: int = 18,
    joint_dim: int = 10,
    velocity_dim: int = 8,
    image_size: int = 384,
    patch_size: int = 16,
    max_num_frames: int = 64,
    tubelet_size: int = 2,
    model_name: str = "vit_giant_xformers",
    embed_dim: int = 1408,
    history_size: int = 3,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dim_head: int = 64,
    predictor_dropout: float = 0.1,
    predictor_embedding_dropout: float = 0.0,
    grounding_hidden_dim: int = 256,
    frame_chunk_size: int = 8,
    use_activation_checkpointing: bool = True,
) -> tuple[nn.Module, nn.Module, CheckpointLoadReport]:
    """Build Static/Full PSG ports with all shared settings held constant."""

    if grounding_variant not in {"static", "full"}:
        raise ValueError("grounding_variant must be 'static' or 'full'")
    if model_name != "vit_giant_xformers":
        raise ValueError(
            "The formal PSG-g comparison is pinned to vit_giant_xformers, got "
            f"{model_name!r}"
        )
    upstream = import_upstream_modules(repo_root)
    backbone = video_vit.__dict__[model_name](
        img_size=int(image_size),
        patch_size=int(patch_size),
        num_frames=int(max_num_frames),
        tubelet_size=int(tubelet_size),
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
    load_report = load_vjepa_encoder_checkpoint(backbone, checkpoint_path)
    inactive_parameter_names = freeze_inactive_image_mode_parameters(backbone)
    hidden_dim = int(backbone.embed_dim)
    encoder = VJEPAFrameEncoder(backbone, frame_chunk_size=frame_chunk_size)

    predictor = upstream.ARPredictor(
        num_frames=int(history_size),
        input_dim=int(embed_dim),
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=int(predictor_depth),
        heads=int(predictor_heads),
        mlp_dim=int(predictor_mlp_dim),
        dim_head=int(predictor_dim_head),
        dropout=float(predictor_dropout),
        emb_dropout=float(predictor_embedding_dropout),
    )
    action_encoder = upstream.Embedder(
        input_dim=int(effective_action_dim),
        emb_dim=int(embed_dim),
    )
    projector = upstream.MLP(
        input_dim=hidden_dim,
        output_dim=int(embed_dim),
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )
    prediction_projector = upstream.MLP(
        input_dim=hidden_dim,
        output_dim=int(embed_dim),
        hidden_dim=2048,
        norm_fn=nn.BatchNorm1d,
    )

    full_heads = upstream.PSGGroundingHeads(
        embed_dim=int(embed_dim),
        state_dim=int(state_dim),
        joint_dim=int(joint_dim),
        vel_dim=int(velocity_dim),
        use_velocity=True,
        hidden=int(grounding_hidden_dim),
    )
    grounding = (
        StaticGroundingOnly(full_heads.state_head)
        if grounding_variant == "static"
        else full_heads
    )
    # Keep the authors' complete world-model class.  The only substituted
    # component is the visual encoder adapter above.
    model = upstream.CrossModalJEPA_v2(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=prediction_projector,
        use_state=False,
        grounding=grounding,
    )
    model.grounding_variant = grounding_variant
    model.upstream_grounding_loss = upstream.grounding_loss
    model.inactive_parameter_names = tuple(inactive_parameter_names)
    sigreg = upstream.SIGReg(knots=17, num_proj=1024)
    return model, sigreg, load_report


def freeze_inactive_image_mode_parameters(backbone: nn.Module) -> tuple[str, ...]:
    """Freeze checkpoint parameters that cannot participate in a T=1 forward.

    The video patch stem, video modality embedding, and intermediate
    distillation norms remain in the nominal ViT-g state dict for provenance,
    but image-mode PSG never executes them.  Marking them non-trainable avoids
    DDP unused-parameter graph traversal and makes the trainable count honest.
    """

    frozen = []
    for name, parameter in backbone.named_parameters():
        inactive = (
            name.startswith("patch_embed.")
            or name == "video_mod_embed"
            or any(name.startswith(f"norms_block.{index}.") for index in range(3))
        )
        if inactive:
            parameter.requires_grad_(False)
            frozen.append(name)
    return tuple(frozen)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return auditable nominal and trainable component sizes."""

    components = {
        "backbone": model.encoder.backbone,
        "projector": model.projector,
        "action_encoder": model.action_encoder,
        "predictor": model.predictor,
        "prediction_projector": model.pred_proj,
        "grounding": model.grounding,
    }
    counts = {}
    for name, module in components.items():
        counts[f"{name}_parameters"] = sum(
            int(parameter.numel()) for parameter in module.parameters()
        )
        counts[f"{name}_trainable_parameters"] = sum(
            int(parameter.numel())
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    counts["total_parameters"] = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    counts["total_trainable_parameters"] = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return counts
