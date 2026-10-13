"""Common interface for frozen world-model feature encoders."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EncoderSpec:
    """Shape and ordering contract between an encoder and the physical probe."""

    name: str
    input_frames: int
    input_size: int
    feature_dim: int
    encoder_tubelet_size: int
    spatial_tokens_per_frame: int
    target_group_size: int

    @property
    def output_tokens(self):
        return (
            self.input_frames
            // self.encoder_tubelet_size
            * self.spatial_tokens_per_frame
        )

    @property
    def probe_time_steps(self):
        return self.input_frames // self.target_group_size

    def to_dict(self):
        return {
            **asdict(self),
            "output_tokens": self.output_tokens,
            "probe_time_steps": self.probe_time_steps,
            "output_layout": "[batch, time-major-patch-token, feature]",
        }


class FeatureEncoderAdapter(nn.Module, ABC):
    """Base class for encoders that expose temporal-major patch features."""

    spec: EncoderSpec
    load_report: dict

    @abstractmethod
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Return features shaped ``[B, N, D]`` in temporal-major order."""

