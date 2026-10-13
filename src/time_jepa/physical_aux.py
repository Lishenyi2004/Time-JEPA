import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_PHYSICAL_HEAD_TYPES = (
    "linear",
    "conv1x1",
    "temporal_bottleneck_k1",
    "temporal_bottleneck_k3",
    "temporal_multiscale",
)


def _pool_spatial_tokens(x, temporal_tokens, head_name):
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"{head_name} expects one feature tensor per call")
        x = x[0]
    bsz, num_tokens, dim = x.shape
    if num_tokens % temporal_tokens != 0:
        raise RuntimeError(
            f"Cannot reshape {num_tokens} tokens into {temporal_tokens} temporal steps"
        )
    spatial_tokens = num_tokens // temporal_tokens
    return x.reshape(bsz, temporal_tokens, spatial_tokens, dim).mean(dim=2)


class TubeletPhysicalHead(nn.Module):
    """Lightweight per-tubelet physical readout used only during training."""

    def __init__(self, embed_dim, num_outputs, num_frames, tubelet_size=2):
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames must be divisible by tubelet_size, got {num_frames=} {tubelet_size=}")
        self.temporal_tokens = num_frames // tubelet_size
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_outputs)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError("TubeletPhysicalHead expects one feature tensor per call")
            x = x[0]
        bsz, num_tokens, dim = x.shape
        if num_tokens % self.temporal_tokens != 0:
            raise RuntimeError(
                f"Cannot reshape {num_tokens} tokens into {self.temporal_tokens} temporal steps"
            )
        spatial_tokens = num_tokens // self.temporal_tokens
        x = x.reshape(bsz, self.temporal_tokens, spatial_tokens, dim).mean(dim=2)
        return self.linear(self.norm(x))


class TubeletConv1x1PhysicalHead(nn.Module):
    """Per-tubelet physical readout that mirrors the frozen-probe conv1x1 head."""

    def __init__(self, embed_dim, num_outputs, num_frames, tubelet_size=2):
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames must be divisible by tubelet_size, got {num_frames=} {tubelet_size=}")
        self.temporal_tokens = num_frames // tubelet_size
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, num_outputs, kernel_size=1),
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError("TubeletConv1x1PhysicalHead expects one feature tensor per call")
            x = x[0]
        bsz, num_tokens, dim = x.shape
        if num_tokens % self.temporal_tokens != 0:
            raise RuntimeError(
                f"Cannot reshape {num_tokens} tokens into {self.temporal_tokens} temporal steps"
            )
        spatial_tokens = num_tokens // self.temporal_tokens
        x = x.reshape(bsz, self.temporal_tokens, spatial_tokens, dim).mean(dim=2)
        x = self.net(x.transpose(1, 2)).transpose(1, 2)
        return x


class TubeletBottleneckTemporalHead(nn.Module):
    """Low-rank auxiliary head with parameter-controlled temporal mixing."""

    def __init__(
        self,
        embed_dim,
        num_outputs,
        num_frames,
        tubelet_size=2,
        hidden_dim=256,
        temporal_kernel_size=1,
    ):
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames must be divisible by tubelet_size, got {num_frames=} {tubelet_size=}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if temporal_kernel_size <= 0 or temporal_kernel_size % 2 == 0:
            raise ValueError(
                "temporal_kernel_size must be a positive odd integer, "
                f"got {temporal_kernel_size}"
            )
        self.temporal_tokens = num_frames // tubelet_size
        self.hidden_dim = int(hidden_dim)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.input_projection = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.temporal_mixer = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            groups=hidden_dim,
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv1d(hidden_dim, num_outputs, kernel_size=1)

    def forward(self, x):
        x = _pool_spatial_tokens(
            x,
            temporal_tokens=self.temporal_tokens,
            head_name=type(self).__name__,
        )
        x = self.norm(x).transpose(1, 2)
        hidden = self.input_projection(x)
        hidden = self.activation(hidden + self.temporal_mixer(hidden))
        return self.output_projection(hidden).transpose(1, 2)


class TubeletMultiScaleTemporalHead(nn.Module):
    """Bottleneck head with lightweight parallel dilated temporal convolutions."""

    def __init__(
        self,
        embed_dim,
        num_outputs,
        num_frames,
        tubelet_size=2,
        hidden_dim=256,
        temporal_dilations=(1, 2, 4),
    ):
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames must be divisible by tubelet_size, got {num_frames=} {tubelet_size=}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        temporal_dilations = tuple(int(dilation) for dilation in temporal_dilations)
        if not temporal_dilations or any(dilation <= 0 for dilation in temporal_dilations):
            raise ValueError(
                "temporal_dilations must contain positive integers, "
                f"got {temporal_dilations}"
            )
        if len(set(temporal_dilations)) != len(temporal_dilations):
            raise ValueError(f"temporal_dilations must be unique, got {temporal_dilations}")
        self.temporal_tokens = num_frames // tubelet_size
        self.hidden_dim = int(hidden_dim)
        self.temporal_dilations = temporal_dilations
        self.norm = nn.LayerNorm(embed_dim)
        self.input_projection = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.temporal_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden_dim,
                )
                for dilation in temporal_dilations
            ]
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv1d(hidden_dim, num_outputs, kernel_size=1)

    def forward(self, x):
        x = _pool_spatial_tokens(
            x,
            temporal_tokens=self.temporal_tokens,
            head_name=type(self).__name__,
        )
        x = self.norm(x).transpose(1, 2)
        hidden = self.input_projection(x)
        mixed = torch.stack(
            [branch(hidden) for branch in self.temporal_branches],
            dim=0,
        ).mean(dim=0)
        hidden = self.activation(hidden + mixed)
        return self.output_projection(hidden).transpose(1, 2)


def build_physical_head(
    head_type,
    embed_dim,
    num_outputs,
    num_frames,
    tubelet_size=2,
    hidden_dim=256,
    temporal_dilations=(1, 2, 4),
):
    head_type = str(head_type or "linear").strip().lower()
    if head_type == "linear":
        return TubeletPhysicalHead(
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
    if head_type == "conv1x1":
        return TubeletConv1x1PhysicalHead(
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
    if head_type in ("temporal_bottleneck_k1", "temporal_bottleneck_k3"):
        temporal_kernel_size = 1 if head_type.endswith("k1") else 3
        return TubeletBottleneckTemporalHead(
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_dim=hidden_dim,
            temporal_kernel_size=temporal_kernel_size,
        )
    if head_type == "temporal_multiscale":
        return TubeletMultiScaleTemporalHead(
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_dim=hidden_dim,
            temporal_dilations=temporal_dilations,
        )
    expected = ", ".join(SUPPORTED_PHYSICAL_HEAD_TYPES)
    raise ValueError(
        f"Unsupported Time-JEPA physical head_type: {head_type!r}; expected one of: {expected}"
    )


def masked_smooth_l1_loss(pred, target, valid, beta=1.0):
    pred = pred.float()
    target = target.float()
    valid = valid.float()
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom


def masked_l1_metric(pred, target, valid):
    pred = pred.float()
    target = target.float()
    valid = valid.float()
    loss = torch.abs(pred - target)
    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom
