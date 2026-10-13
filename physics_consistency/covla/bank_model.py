"""Independent scalar probe heads sharing only one frozen encoder forward."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .bank_common import TARGET_NAMES
except ImportError:
    from bank_common import TARGET_NAMES


def spatial_pool_by_time(features: torch.Tensor, temporal_tokens: int) -> torch.Tensor:
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
    """One target's complete trainable state; nothing is shared with other heads."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(temporal_features)).squeeze(-1)


class IndependentHeadBank(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_frames: int,
        tubelet_size: int = 2,
        target_names=TARGET_NAMES,
    ):
        super().__init__()
        self.target_names = tuple(target_names)
        self.temporal_tokens = int(num_frames) // int(tubelet_size)
        self.heads = nn.ModuleDict(
            {name: ScalarLinearHead(int(embed_dim)) for name in self.target_names}
        )

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
        parameter_ids = self.independent_parameter_ids()
        for index, left in enumerate(self.target_names):
            for right in self.target_names[index + 1 :]:
                overlap = parameter_ids[left] & parameter_ids[right]
                if overlap:
                    raise RuntimeError(f"Heads {left} and {right} unexpectedly share parameters")


class TinySmokeEncoder(nn.Module):
    """Cheap CPU encoder used only to exercise the complete local pipeline."""

    def __init__(self, embed_dim: int = 32, tubelet_size: int = 2):
        super().__init__()
        self.tubelet_size = int(tubelet_size)
        self.projection = nn.Linear(3, int(embed_dim), bias=False)
        generator = torch.Generator().manual_seed(20260720)
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
