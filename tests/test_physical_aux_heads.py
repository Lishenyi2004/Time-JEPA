import torch
import torch.nn as nn

from src.time_jepa.physical_aux import (
    TubeletBottleneckTemporalHead,
    TubeletMultiScaleTemporalHead,
    build_physical_head,
)


def _set_unit_temporal_weights(head):
    head.norm = nn.Identity()
    with torch.no_grad():
        head.input_projection.weight.fill_(1.0)
        head.input_projection.bias.zero_()
        head.output_projection.weight.fill_(1.0)
        head.output_projection.bias.zero_()
        if isinstance(head, TubeletBottleneckTemporalHead):
            head.temporal_mixer.weight.fill_(1.0)
            head.temporal_mixer.bias.zero_()
        else:
            for branch in head.temporal_branches:
                branch.weight.fill_(1.0)
                branch.bias.zero_()


def _active_time_steps(head, num_steps=11, impulse_step=5):
    _set_unit_temporal_weights(head)
    features = torch.zeros(1, num_steps, 1)
    features[0, impulse_step, 0] = 1.0
    predictions = head(features)
    return torch.nonzero(predictions[0, :, 0], as_tuple=True)[0].tolist()


def test_temporal_auxiliary_heads_preserve_tubelet_layout():
    for head_type in (
        "temporal_bottleneck_k1",
        "temporal_bottleneck_k3",
        "temporal_multiscale",
    ):
        head = build_physical_head(
            head_type=head_type,
            embed_dim=8,
            num_outputs=3,
            num_frames=12,
            tubelet_size=2,
            hidden_dim=6,
            temporal_dilations=(1, 2, 4),
        )
        # Six temporal tubelets, with four spatial tokens per tubelet.
        features = torch.randn(2, 6 * 4, 8)
        predictions = head(features)
        assert predictions.shape == (2, 6, 3)


def test_h1_kernel_one_does_not_mix_neighboring_tubelets():
    head = TubeletBottleneckTemporalHead(
        embed_dim=1,
        num_outputs=1,
        num_frames=11,
        tubelet_size=1,
        hidden_dim=1,
        temporal_kernel_size=1,
    )
    assert _active_time_steps(head) == [5]


def test_h2_kernel_three_mixes_immediate_neighboring_tubelets():
    head = TubeletBottleneckTemporalHead(
        embed_dim=1,
        num_outputs=1,
        num_frames=11,
        tubelet_size=1,
        hidden_dim=1,
        temporal_kernel_size=3,
    )
    assert _active_time_steps(head) == [4, 5, 6]


def test_h3_multiscale_head_uses_all_configured_dilations():
    head = TubeletMultiScaleTemporalHead(
        embed_dim=1,
        num_outputs=1,
        num_frames=11,
        tubelet_size=1,
        hidden_dim=1,
        temporal_dilations=(1, 2, 4),
    )
    assert _active_time_steps(head) == [1, 3, 4, 5, 6, 7, 9]


def test_unknown_temporal_head_type_is_rejected():
    try:
        build_physical_head(
            head_type="temporal_unknown",
            embed_dim=8,
            num_outputs=3,
            num_frames=12,
            tubelet_size=2,
        )
    except ValueError as exc:
        assert "temporal_unknown" in str(exc)
    else:
        raise AssertionError("Unknown Time-JEPA head type should raise ValueError")
