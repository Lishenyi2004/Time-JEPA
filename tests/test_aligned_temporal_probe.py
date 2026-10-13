import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROBE_DIR = Path(__file__).resolve().parents[1] / "covla_probe_rjob"
sys.path.insert(0, str(PROBE_DIR))

from probe_models import AlignedTemporalProbe, build_probe  # noqa: E402


def test_aligned_temporal_probe_matches_conv_layout_with_kernel_three():
    probe = AlignedTemporalProbe(
        embed_dim=8,
        num_outputs=3,
        num_frames=10,
        tubelet_size=2,
        hidden_dim=6,
    )

    layers = list(probe.net)
    assert len(layers) == 3
    assert isinstance(layers[0], nn.Conv1d)
    assert isinstance(layers[1], nn.GELU)
    assert isinstance(layers[2], nn.Conv1d)
    assert layers[0].kernel_size == (3,)
    assert layers[0].padding == (1,)
    assert layers[2].kernel_size == (3,)
    assert layers[2].padding == (1,)

    # Five temporal tubelets, with four spatial tokens per tubelet.
    features = torch.randn(2, 5 * 4, 8)
    predictions = probe(features)
    assert predictions.shape == (2, 5, 3)


def test_aligned_temporal_probe_has_five_tubelet_receptive_field():
    probe = AlignedTemporalProbe(
        embed_dim=1,
        num_outputs=1,
        num_frames=7,
        tubelet_size=1,
        hidden_dim=1,
    )
    with torch.no_grad():
        for layer in probe.net:
            if isinstance(layer, nn.Conv1d):
                layer.weight.fill_(1.0)
                layer.bias.zero_()

    features = torch.zeros(1, 7, 1)
    features[0, 3, 0] = 1.0
    predictions = probe(features)

    active_time_steps = torch.nonzero(predictions[0, :, 0], as_tuple=True)[0].tolist()
    assert active_time_steps == [1, 2, 3, 4, 5]


def test_build_probe_supports_temporal_aligned():
    args = SimpleNamespace(
        probe_type="temporal_aligned",
        tubelet_size=2,
        probe_hidden_dim=6,
    )
    arch = {"embed_dim": 8, "num_frames": 10}

    probe = build_probe(args, arch, num_outputs=3)

    assert isinstance(probe, AlignedTemporalProbe)
