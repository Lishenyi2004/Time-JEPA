"""Experimental physical heads for the isolated EgoExo head sweep.

This module deliberately extends, rather than edits, ``physical_aux.py`` so a
running job that imported the production heads keeps exactly the same code.
"""

import torch
import torch.nn as nn

from src.time_jepa.physical_aux import (
    SUPPORTED_PHYSICAL_HEAD_TYPES as BASE_PHYSICAL_HEAD_TYPES,
)
from src.time_jepa.physical_aux import build_physical_head as build_base_physical_head


EXPERIMENTAL_PHYSICAL_HEAD_TYPES = (
    "temporal_multiscale_fusion",
    "temporal_attention_local",
    "temporal_attention_global",
    "temporal_attention",
)
SUPPORTED_PHYSICAL_HEAD_TYPES = (
    *BASE_PHYSICAL_HEAD_TYPES,
    *EXPERIMENTAL_PHYSICAL_HEAD_TYPES,
)


def _pool_spatial_tokens(x, temporal_tokens, head_name):
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"{head_name} expects one feature tensor per call")
        x = x[0]
    if x.ndim != 3:
        raise ValueError(
            f"{head_name} expects [batch, tokens, channels], got {tuple(x.shape)}"
        )
    batch_size, num_tokens, channels = x.shape
    if num_tokens % temporal_tokens != 0:
        raise RuntimeError(
            f"Cannot reshape {num_tokens} tokens into {temporal_tokens} temporal steps"
        )
    spatial_tokens = num_tokens // temporal_tokens
    return x.reshape(
        batch_size,
        temporal_tokens,
        spatial_tokens,
        channels,
    ).mean(dim=2)


class TubeletDenseMultiScaleFusionHead(nn.Module):
    """H3-strong: dense k3/k5/k9 depthwise branches with learned concat fusion."""

    temporal_kernel_sizes = (3, 5, 9)

    def __init__(
        self,
        embed_dim,
        num_outputs,
        num_frames,
        tubelet_size=2,
        hidden_dim=610,
    ):
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError(
                "num_frames must be divisible by tubelet_size, "
                f"got {num_frames=} {tubelet_size=}"
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.temporal_tokens = num_frames // tubelet_size
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.input_projection = nn.Conv1d(
            embed_dim,
            self.hidden_dim,
            kernel_size=1,
        )
        self.temporal_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=self.hidden_dim,
                )
                for kernel_size in self.temporal_kernel_sizes
            ]
        )
        self.fusion = nn.Conv1d(
            len(self.temporal_kernel_sizes) * self.hidden_dim,
            self.hidden_dim,
            kernel_size=1,
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv1d(
            self.hidden_dim,
            num_outputs,
            kernel_size=1,
        )

    def forward(self, x):
        x = _pool_spatial_tokens(
            x,
            temporal_tokens=self.temporal_tokens,
            head_name=type(self).__name__,
        )
        x = self.norm(x).transpose(1, 2)
        hidden = self.input_projection(x)
        mixed = self.fusion(
            torch.cat(
                [branch(hidden) for branch in self.temporal_branches],
                dim=1,
            )
        )
        hidden = self.activation(hidden + mixed)
        return self.output_projection(hidden).transpose(1, 2)


def _local_attention_mask(num_steps, window_size):
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be a positive odd integer, got {window_size}"
        )
    radius = window_size // 2
    positions = torch.arange(num_steps)
    # MultiheadAttention uses True to denote a blocked query-key pair.
    return (positions[:, None] - positions[None, :]).abs() > radius


class TubeletTemporalSelfAttentionHead(nn.Module):
    """Single-block temporal self-attention head used by the H4 variants.

    ``local`` uses window-9 attention for every head, ``global`` lets every
    head see all temporal tokens, and ``hybrid`` assigns half of the heads to
    each pattern in the same attention layer.
    """

    valid_attention_modes = ("local", "global", "hybrid")

    def __init__(
        self,
        embed_dim,
        num_outputs,
        num_frames,
        tubelet_size=2,
        hidden_dim=416,
        num_heads=8,
        local_window_size=9,
        ffn_ratio=2,
        attention_mode="hybrid",
    ):
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError(
                "num_frames must be divisible by tubelet_size, "
                f"got {num_frames=} {tubelet_size=}"
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got {hidden_dim=} {num_heads=}"
            )
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be positive, got {ffn_ratio}")
        attention_mode = str(attention_mode).strip().lower()
        if attention_mode not in self.valid_attention_modes:
            expected = ", ".join(self.valid_attention_modes)
            raise ValueError(
                f"attention_mode must be one of {expected}; got {attention_mode!r}"
            )
        if attention_mode == "hybrid" and num_heads % 2 != 0:
            raise ValueError(
                f"hybrid attention requires an even num_heads, got {num_heads}"
            )

        self.temporal_tokens = num_frames // tubelet_size
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.local_window_size = int(local_window_size)
        self.attention_mode = attention_mode

        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_projection = nn.Linear(embed_dim, self.hidden_dim)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.temporal_tokens, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.temporal_position, std=0.02)

        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        ffn_hidden_dim = int(self.hidden_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, self.hidden_dim),
        )
        self.output_projection = nn.Linear(self.hidden_dim, num_outputs)

        local_mask = _local_attention_mask(
            self.temporal_tokens,
            self.local_window_size,
        )
        if self.attention_mode == "local":
            attention_mask = local_mask
        elif self.attention_mode == "hybrid":
            attention_mask = torch.zeros(
                self.num_heads,
                self.temporal_tokens,
                self.temporal_tokens,
                dtype=torch.bool,
            )
            attention_mask[: self.num_heads // 2] = local_mask
        else:
            attention_mask = torch.empty(0, dtype=torch.bool)
        self.register_buffer(
            "_attention_mask",
            attention_mask,
            persistent=False,
        )

    def _expanded_attention_mask(self, batch_size):
        if self.attention_mode == "global":
            return None
        if self.attention_mode == "local":
            return self._attention_mask
        return (
            self._attention_mask.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(
                batch_size * self.num_heads,
                self.temporal_tokens,
                self.temporal_tokens,
            )
        )

    def forward(self, x):
        x = _pool_spatial_tokens(
            x,
            temporal_tokens=self.temporal_tokens,
            head_name=type(self).__name__,
        )
        hidden = self.input_projection(self.input_norm(x))
        hidden = hidden + self.temporal_position.to(dtype=hidden.dtype)

        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=self._expanded_attention_mask(hidden.shape[0]),
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return self.output_projection(hidden)


def build_physical_head(
    head_type,
    embed_dim,
    num_outputs,
    num_frames,
    tubelet_size=2,
    hidden_dim=256,
    temporal_dilations=(1, 2, 4),
):
    """Build a base head or an isolated head-sweep extension."""

    head_type = str(head_type or "linear").strip().lower()
    if head_type in BASE_PHYSICAL_HEAD_TYPES:
        return build_base_physical_head(
            head_type=head_type,
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_dim=hidden_dim,
            temporal_dilations=temporal_dilations,
        )
    if head_type == "temporal_multiscale_fusion":
        return TubeletDenseMultiScaleFusionHead(
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_dim=hidden_dim,
        )
    attention_modes = {
        "temporal_attention_local": "local",
        "temporal_attention_global": "global",
        "temporal_attention": "hybrid",
    }
    if head_type in attention_modes:
        return TubeletTemporalSelfAttentionHead(
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_dim=hidden_dim,
            num_heads=8,
            local_window_size=9,
            ffn_ratio=2,
            attention_mode=attention_modes[head_type],
        )
    expected = ", ".join(SUPPORTED_PHYSICAL_HEAD_TYPES)
    raise ValueError(
        f"Unsupported Time-JEPA physical head_type: {head_type!r}; "
        f"expected one of: {expected}"
    )
