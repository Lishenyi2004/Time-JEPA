"""Independent scalar heads sharing only a frozen encoder forward pass."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .bank_common import TARGET_NAMES
except ImportError:
    from bank_common import TARGET_NAMES


def spatial_pool_by_time(features: torch.Tensor, temporal_tokens: int) -> torch.Tensor:
    """Pool spatial tokens while preserving the encoder's temporal grid."""

    if features.ndim != 3:
        raise RuntimeError(f"Expected encoder output [B,N,D], got {tuple(features.shape)}")
    batch, tokens, dim = features.shape
    if tokens == temporal_tokens:
        return features
    if tokens % temporal_tokens:
        raise RuntimeError(
            f"Encoder emitted {tokens} tokens, not divisible by {temporal_tokens} temporal tokens"
        )
    return features.reshape(batch, temporal_tokens, tokens // temporal_tokens, dim).mean(dim=2)


class ScalarLinearHead(nn.Module):
    """One target's complete trainable state; no parameters are shared."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(embed_dim))
        self.linear = nn.Linear(int(embed_dim), 1)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(temporal_features)).squeeze(-1)


class ScalarConv1x1Head(nn.Module):
    """One target's non-linear 1x1-conv readout with no shared parameters.

    This follows the standard AgiBot ``Conv1x1Probe`` block
    (Conv1d -> GELU -> Conv1d), but produces one scalar and is instantiated
    independently for every target.  Kernel size 1 deliberately preserves the
    original per-tubelet temporal alignment and does not mix adjacent times.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        embed_dim = int(embed_dim)
        hidden_dim = int(hidden_dim) or embed_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive or zero for embed_dim")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        return self.net(temporal_features.transpose(1, 2)).squeeze(1)


class IndependentHeadBank(nn.Module):
    """A bank that is computationally joint but statistically independent."""

    def __init__(
        self,
        embed_dim: int,
        num_frames: int,
        tubelet_size: int = 2,
        target_names=TARGET_NAMES,
        head_type: str = "independent_linear",
        hidden_dim: int = 0,
    ):
        super().__init__()
        self.target_names = tuple(target_names)
        self.temporal_tokens = int(num_frames) // int(tubelet_size)
        self.head_type = str(head_type)
        self.hidden_dim = int(hidden_dim)
        if self.head_type == "independent_linear":
            build_head = lambda: ScalarLinearHead(int(embed_dim))
        elif self.head_type == "independent_conv1x1":
            build_head = lambda: ScalarConv1x1Head(int(embed_dim), self.hidden_dim)
        else:
            raise ValueError(
                "head_type must be independent_linear or independent_conv1x1, "
                f"got {self.head_type!r}"
            )
        self.heads = nn.ModuleDict({name: build_head() for name in self.target_names})

    def forward_dict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal = spatial_pool_by_time(features, self.temporal_tokens)
        return {name: self.heads[name](temporal) for name in self.target_names}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_dict(features)
        return torch.stack([outputs[name] for name in self.target_names], dim=-1)

    def independent_parameter_ids(self) -> dict[str, set[int]]:
        return {
            name: {id(parameter) for parameter in head.parameters()}
            for name, head in self.heads.items()
        }

    def assert_disjoint(self) -> None:
        ids = self.independent_parameter_ids()
        for index, left in enumerate(self.target_names):
            for right in self.target_names[index + 1 :]:
                if ids[left] & ids[right]:
                    raise RuntimeError(f"Heads {left} and {right} unexpectedly share parameters")

    def architecture_report(self) -> dict:
        per_head = {
            name: sum(parameter.numel() for parameter in head.parameters())
            for name, head in self.heads.items()
        }
        return {
            "type": self.head_type,
            "target_count": len(self.target_names),
            "hidden_dim": (
                next(iter(self.heads.values())).hidden_dim
                if self.head_type == "independent_conv1x1"
                else None
            ),
            "kernel_size": 1 if self.head_type == "independent_conv1x1" else None,
            "activation": "GELU" if self.head_type == "independent_conv1x1" else None,
            "normalization": "LayerNorm" if self.head_type == "independent_linear" else None,
            "per_head_trainable_parameters": per_head,
            "total_trainable_parameters": int(sum(per_head.values())),
            "shared_trainable_parameters": False,
        }


class TinySmokeEncoder(nn.Module):
    """Deterministic frozen encoder for a cheap end-to-end CPU smoke test."""

    def __init__(self, embed_dim: int = 32, tubelet_size: int = 2):
        super().__init__()
        self.tubelet_size = int(tubelet_size)
        self.projection = nn.Linear(3, int(embed_dim), bias=False)
        generator = torch.Generator().manual_seed(20260721)
        with torch.no_grad():
            weight = torch.randn(self.projection.weight.shape, generator=generator)
            self.projection.weight.copy_(weight / weight.shape[1] ** 0.5)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise RuntimeError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
        temporal_rgb = video.mean(dim=(-1, -2)).transpose(1, 2)
        batch, frames, channels = temporal_rgb.shape
        if frames % self.tubelet_size:
            raise RuntimeError("Tiny encoder requires frames divisible by tubelet_size")
        temporal_rgb = temporal_rgb.reshape(
            batch, frames // self.tubelet_size, self.tubelet_size, channels
        ).mean(dim=2)
        return self.projection(temporal_rgb)
