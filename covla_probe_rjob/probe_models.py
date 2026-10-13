"""Probe heads and the build_probe factory."""

import probe_paths  # noqa: F401  (ensures REPO_ROOT is on sys.path before src imports)

import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler


def spatial_pool_by_time(x, temporal_tokens):
    bsz, num_tokens, dim = x.shape
    if num_tokens % temporal_tokens == 0:
        spatial_tokens = num_tokens // temporal_tokens
        return x.reshape(bsz, temporal_tokens, spatial_tokens, dim).mean(dim=2)
    return x.mean(dim=1, keepdim=True)


def ensure_time_length(pred, num_time_steps):
    if pred.size(1) == num_time_steps:
        return pred
    raise RuntimeError(f"Probe produced {pred.size(1)} time steps for {num_time_steps} tubelet targets")


class AttentiveSequenceProbe(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, num_outputs, num_time_steps):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=num_time_steps,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=False,
        )
        self.linear = nn.Linear(embed_dim, num_outputs)

    def forward(self, x):
        return self.linear(self.pooler(x))


class LinearProbe(nn.Module):
    def __init__(self, embed_dim, num_outputs, num_frames, tubelet_size=2):
        super().__init__()
        self.temporal_tokens = max(1, num_frames // tubelet_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_outputs)

    def forward(self, x):
        x = spatial_pool_by_time(x, self.temporal_tokens)
        x = self.linear(self.norm(x))
        return ensure_time_length(x, self.temporal_tokens)


class Conv1x1Probe(nn.Module):
    def __init__(self, embed_dim, num_outputs, num_frames, tubelet_size=2, hidden_dim=0):
        super().__init__()
        self.temporal_tokens = max(1, num_frames // tubelet_size)
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, num_outputs, kernel_size=1),
        )

    def forward(self, x):
        x = spatial_pool_by_time(x, self.temporal_tokens).transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return ensure_time_length(x, self.temporal_tokens)


class FutureConv1x1Probe(nn.Module):
    """Direct multi-horizon probe for states strictly beyond the observed clip.

    The frozen ViT is bidirectional inside the observed clip, so its final
    temporal token is contextualized by the full history.  Selecting only this
    endpoint prevents an accidental tubelet-wise label shift from exposing
    target-time video frames.  A single pointwise channel map emits all forecast
    horizons at once, with output shape [B, H, D].
    """

    def __init__(
        self,
        embed_dim,
        num_outputs,
        num_frames,
        tubelet_size=2,
        hidden_dim=0,
        num_future_steps=1,
    ):
        super().__init__()
        self.temporal_tokens = max(1, num_frames // tubelet_size)
        self.num_outputs = int(num_outputs)
        self.num_future_steps = int(num_future_steps)
        if self.num_future_steps <= 0:
            raise ValueError(
                f"num_future_steps must be positive, got {self.num_future_steps}"
            )
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim,
                self.num_future_steps * self.num_outputs,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        x = spatial_pool_by_time(x, self.temporal_tokens)[:, -1:, :].transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return x.reshape(x.size(0), self.num_future_steps, self.num_outputs)


class AlignedTemporalProbe(nn.Module):
    """Temporal counterpart to Conv1x1Probe with only the kernels changed.

    Both Conv1d layers use kernel size 3, so the two-layer stack has an
    effective five-tubelet receptive field while preserving the input/output
    time length.  The layer count, hidden width, activation, spatial pooling,
    and forward layout otherwise match Conv1x1Probe exactly.
    """

    def __init__(self, embed_dim, num_outputs, num_frames, tubelet_size=2, hidden_dim=0):
        super().__init__()
        self.temporal_tokens = max(1, num_frames // tubelet_size)
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, num_outputs, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = spatial_pool_by_time(x, self.temporal_tokens).transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return ensure_time_length(x, self.temporal_tokens)


class TemporalProbe(nn.Module):
    def __init__(self, embed_dim, num_outputs, num_frames, tubelet_size=2, hidden_dim=0):
        super().__init__()
        self.temporal_tokens = max(1, num_frames // tubelet_size)
        hidden_dim = hidden_dim or embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.linear = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        x = spatial_pool_by_time(x, self.temporal_tokens)
        x = self.norm(x).transpose(1, 2)
        x = self.temporal(x).transpose(1, 2)
        x = self.linear(x)
        return ensure_time_length(x, self.temporal_tokens)


def build_probe(args, arch, num_outputs, num_future_steps=1):
    probe_type = args.probe_type
    if probe_type in ("official_ssv2", "attentive"):
        return AttentiveSequenceProbe(
            embed_dim=arch["embed_dim"],
            num_heads=arch["num_heads"],
            depth=args.probe_depth,
            num_outputs=num_outputs,
            num_time_steps=max(1, arch["num_frames"] // args.tubelet_size),
        )
    if probe_type == "linear":
        return LinearProbe(arch["embed_dim"], num_outputs, num_frames=arch["num_frames"], tubelet_size=args.tubelet_size)
    if probe_type == "conv1x1":
        return Conv1x1Probe(
            arch["embed_dim"],
            num_outputs,
            num_frames=arch["num_frames"],
            tubelet_size=args.tubelet_size,
            hidden_dim=args.probe_hidden_dim,
        )
    if probe_type == "future_conv1x1":
        return FutureConv1x1Probe(
            arch["embed_dim"],
            num_outputs,
            num_frames=arch["num_frames"],
            tubelet_size=args.tubelet_size,
            hidden_dim=args.probe_hidden_dim,
            num_future_steps=num_future_steps,
        )
    if probe_type == "temporal_aligned":
        return AlignedTemporalProbe(
            arch["embed_dim"],
            num_outputs,
            num_frames=arch["num_frames"],
            tubelet_size=args.tubelet_size,
            hidden_dim=args.probe_hidden_dim,
        )
    if probe_type == "temporal":
        return TemporalProbe(
            arch["embed_dim"],
            num_outputs,
            num_frames=arch["num_frames"],
            tubelet_size=args.tubelet_size,
            hidden_dim=args.probe_hidden_dim,
        )
    raise ValueError(f"Unsupported probe type: {probe_type}")
