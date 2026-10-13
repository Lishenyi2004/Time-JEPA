"""Four parameter-disjoint scalar heads sharing only a frozen encoder pass."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .bank_common import TARGET_NAMES
except ImportError:
    from bank_common import TARGET_NAMES


def spatial_pool_by_time(features: torch.Tensor, temporal_tokens: int) -> torch.Tensor:
    if features.ndim != 3:
        raise RuntimeError(f"Expected [B,N,D] encoder output, got {tuple(features.shape)}")
    batch, tokens, dim = features.shape
    if tokens == temporal_tokens:
        return features
    if tokens % temporal_tokens:
        raise RuntimeError(
            f"Encoder tokens {tokens} are not divisible by temporal tokens {temporal_tokens}"
        )
    return features.reshape(batch, temporal_tokens, tokens // temporal_tokens, dim).mean(2)


class ScalarLinearHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(embed_dim))
        self.linear = nn.Linear(int(embed_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(features)).squeeze(-1)


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
        output = self.forward_dict(features)
        return torch.stack([output[name] for name in self.target_names], dim=-1)

    def assert_disjoint(self) -> None:
        parameter_ids = {
            name: {id(parameter) for parameter in head.parameters()}
            for name, head in self.heads.items()
        }
        for index, left in enumerate(self.target_names):
            for right in self.target_names[index + 1 :]:
                if parameter_ids[left] & parameter_ids[right]:
                    raise RuntimeError(f"Heads {left} and {right} share trainable parameters")

    def architecture_report(self) -> dict:
        per_head = {
            name: sum(parameter.numel() for parameter in head.parameters())
            for name, head in self.heads.items()
        }
        return {
            "type": "independent_linear",
            "target_count": len(self.target_names),
            "normalization": "separate LayerNorm per target",
            "per_head_trainable_parameters": per_head,
            "total_trainable_parameters": int(sum(per_head.values())),
            "shared_trainable_parameters": False,
            "temporal_mixing": False,
        }


class TinySmokeEncoder(nn.Module):
    def __init__(self, embed_dim: int = 16, tubelet_size: int = 2):
        super().__init__()
        self.tubelet_size = int(tubelet_size)
        self.projection = nn.Linear(3, int(embed_dim), bias=False)
        generator = torch.Generator().manual_seed(20260723)
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
            raise RuntimeError("Frames must be divisible by tubelet size")
        temporal_rgb = temporal_rgb.reshape(
            batch, frames // self.tubelet_size, self.tubelet_size, channels
        ).mean(2)
        return self.projection(temporal_rgb)

