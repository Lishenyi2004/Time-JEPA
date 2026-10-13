"""Construct the official PSG-JEPA architecture without loading a checkpoint."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

from .protocol import EXPECTED_UPSTREAM_COMMIT


@dataclass(frozen=True)
class UpstreamModules:
    CrossModalJEPA_v2: type
    CrossModalViTEncoder: type
    PSGGroundingHeads: type
    ARPredictor: type
    Embedder: type
    MLP: type
    SIGReg: type
    grounding_loss: object


class StaticGroundingOnly(nn.Module):
    """Keep only the official state-head MLP from ``PSGGroundingHeads``."""

    def __init__(self, state_head: nn.Module):
        super().__init__()
        self.state_head = state_head


def verify_upstream_checkout(
    repo_root: str | Path,
    *,
    expected_commit: str = EXPECTED_UPSTREAM_COMMIT,
) -> Path:
    repo_root = Path(repo_root).expanduser().resolve()
    required = (
        repo_root / "psgjepa" / "model.py",
        repo_root / "psgjepa" / "module.py",
        repo_root / "psgjepa" / "grounding.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"PSG-JEPA checkout is incomplete under {repo_root}; missing={missing}"
        )
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not resolve PSG-JEPA git commit at {repo_root}: {result.stderr.strip()}"
        )
    actual_commit = result.stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            "Unexpected PSG-JEPA checkout commit: "
            f"got {actual_commit}, expected {expected_commit}"
        )
    return repo_root


def import_upstream_modules(repo_root: str | Path) -> UpstreamModules:
    repo_root = verify_upstream_checkout(repo_root)
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from psgjepa.grounding import PSGGroundingHeads, grounding_loss
    from psgjepa.model import CrossModalJEPA_v2, CrossModalViTEncoder
    from psgjepa.module import ARPredictor, Embedder, MLP, SIGReg

    imported_path = Path(sys.modules["psgjepa"].__file__).resolve().parent.parent
    if imported_path != repo_root:
        raise RuntimeError(
            f"Imported psgjepa from {imported_path}, but pinned checkout is {repo_root}"
        )
    return UpstreamModules(
        CrossModalJEPA_v2=CrossModalJEPA_v2,
        CrossModalViTEncoder=CrossModalViTEncoder,
        PSGGroundingHeads=PSGGroundingHeads,
        ARPredictor=ARPredictor,
        Embedder=Embedder,
        MLP=MLP,
        SIGReg=SIGReg,
        grounding_loss=grounding_loss,
    )


def build_official_world_model(
    repo_root: str | Path,
    *,
    state_dim: int,
    effective_action_dim: int,
    enable_static_grounding: bool,
    image_size: int = 224,
    patch_size: int = 14,
    encoder_scale: str = "tiny",
    embed_dim: int = 192,
    history_size: int = 3,
    predictor_depth: int = 6,
    predictor_heads: int = 16,
    predictor_mlp_dim: int = 2048,
    predictor_dim_head: int = 64,
    predictor_dropout: float = 0.1,
    predictor_embedding_dropout: float = 0.0,
    grounding_hidden_dim: int = 256,
) -> tuple[nn.Module, nn.Module]:
    """Build the authors' ViT, projector, predictor, action encoder and state head."""

    import stable_pretraining as spt

    upstream = import_upstream_modules(repo_root)
    raw_vit = spt.backbone.utils.vit_hf(
        encoder_scale,
        patch_size=int(patch_size),
        image_size=int(image_size),
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = int(raw_vit.config.hidden_size)
    encoder = upstream.CrossModalViTEncoder(
        vit_model=raw_vit,
        proprio_dim=0,
        patch_size=int(patch_size),
        state_inject_mode="patch",
    )
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

    grounding = None
    if enable_static_grounding:
        # Construct the upstream head, then retain only H_s.  This avoids
        # transition-head parameters that would be unused under DDP.
        upstream_heads = upstream.PSGGroundingHeads(
            embed_dim=int(embed_dim),
            state_dim=int(state_dim),
            joint_dim=1,
            use_velocity=False,
            hidden=int(grounding_hidden_dim),
        )
        grounding = StaticGroundingOnly(upstream_heads.state_head)

    model = upstream.CrossModalJEPA_v2(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=prediction_projector,
        use_state=False,
        grounding=grounding,
    )
    sigreg = upstream.SIGReg(knots=17, num_proj=1024)
    return model, sigreg
