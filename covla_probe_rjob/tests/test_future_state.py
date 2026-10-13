from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROBE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROBE_ROOT.parent
for path in (str(PROBE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from probe_config import parse_args
from probe_engine import evaluate, masked_mse
from probe_models import FutureConv1x1Probe
from probe_sampling import (
    future_row_is_eligible,
    normalize_future_horizons_seconds,
    sample_frame_indices,
)


def test_sampling_reserves_strictly_future_frames():
    indices = sample_frame_indices(
        num_video_frames=600,
        num_frames=64,
        frame_stride=4,
        random_clip=False,
        reserve_after=20,
    )
    assert len(indices) == 64
    assert int(indices[-1]) + 20 < 600


def test_sampling_rejects_context_that_cannot_leave_future_room():
    try:
        sample_frame_indices(
            num_video_frames=70,
            num_frames=64,
            frame_stride=1,
            reserve_after=20,
        )
    except ValueError as exc:
        assert "too short" in str(exc)
    else:
        raise AssertionError("Expected a short context+future video to be rejected")


def test_terminal_timestamp_windows_are_filtered():
    row = {"window_idx": 1, "slice_stride_seconds": 12.8, "duration": 26.0}
    assert not future_row_is_eligible(row, 64, 5.0, 1.0, 12.8)
    row["duration"] = 27.0
    assert future_row_is_eligible(row, 64, 5.0, 1.0, 12.8)


def test_future_horizons_are_positive_unique_and_increasing():
    assert normalize_future_horizons_seconds("0.2,0.4,1.0") == (0.2, 0.4, 1.0)
    for invalid in ((0.4, 0.2), (0.2, 0.2), (0.0, 1.0)):
        try:
            normalize_future_horizons_seconds(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid future horizons: {invalid}")


def test_future_probe_emits_one_state_per_horizon():
    probe = FutureConv1x1Probe(
        embed_dim=8,
        num_outputs=3,
        num_frames=4,
        tubelet_size=2,
        hidden_dim=4,
        num_future_steps=5,
    )
    features = torch.randn(2, 2 * 5, 8)
    assert probe(features).shape == (2, 5, 3)


def test_multi_horizon_loss_weights_horizons_equally():
    pred = torch.zeros(2, 2, 1)
    target = torch.tensor([[[1.0], [2.0]], [[3.0], [100.0]]])
    valid = torch.tensor([[[1.0], [1.0]], [[1.0], [0.0]]])

    loss = masked_mse(pred, target, valid, balance_time_steps=True)
    # Horizon 1 MSE is (1 + 9) / 2 = 5; horizon 2 MSE is 4 / 1 = 4.
    assert np.isclose(loss.item(), 4.5)


def test_all_eighteen_future_configs_parse_and_match_protocol():
    config_root = PROBE_ROOT / "configs" / "future_state_1s"
    configs = sorted(config_root.glob("*/*/*.yaml"))
    assert len(configs) == 18
    for config in configs:
        args = parse_args(["--config", str(config)])
        assert args.target_mode == "future_state"
        assert args.probe_type == "future_conv1x1"
        assert args.future_horizon_seconds == 1.0
        assert normalize_future_horizons_seconds(
            args.future_horizons_seconds,
            args.future_horizon_seconds,
        ) == (0.2, 0.4, 0.6, 0.8, 1.0)
        assert args.checkpoint_key == "auto"
    for domain in ("covla", "agibot", "egoexo"):
        for task in ("readout", "shot"):
            mix = vars(parse_args([
                "--config", str(config_root / "mix_e19" / domain / f"{task}.yaml")
            ]))
            zero = vars(parse_args([
                "--config", str(config_root / "zero" / domain / f"{task}.yaml")
            ]))
            differences = {key for key in mix if mix[key] != zero[key]}
            assert differences == {"config", "encoder_checkpoint", "output_dir"}


class _FutureMetricDataset(Dataset):
    def __init__(self):
        self.targets = [1.0, 3.0]
        self.current = [0.5, 2.5]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "video": torch.zeros(1),
            "target": torch.tensor([[self.targets[index]]]),
            "valid": torch.ones(1, 1),
            "current_target": torch.tensor([[self.current[index]]]),
            "current_valid": torch.ones(1, 1),
        }


class _IdentityEncoder(torch.nn.Module):
    def forward(self, videos):
        return videos


class _ZeroProbe(torch.nn.Module):
    def forward(self, features):
        return torch.zeros(features.shape[0], 1, 1, device=features.device)


class _MultiHorizonMetricDataset(Dataset):
    def __init__(self):
        self.targets = [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]
        self.current = [[[0.5]], [[2.5]]]
        self.current_valid = [[[1.0]], [[0.0]]]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "video": torch.zeros(1),
            "target": torch.tensor(self.targets[index]),
            "valid": torch.ones(2, 1),
            "current_target": torch.tensor(self.current[index]),
            "current_valid": torch.tensor(self.current_valid[index]),
        }


class _MultiHorizonZeroProbe(torch.nn.Module):
    def forward(self, features):
        return torch.zeros(features.shape[0], 2, 1, device=features.device)


def test_future_evaluation_reports_mean_and_persistence_baselines():
    metrics = evaluate(
        _IdentityEncoder(),
        _ZeroProbe(),
        DataLoader(_FutureMetricDataset(), batch_size=2),
        torch.device("cpu"),
        target_mean=torch.tensor([0.0]),
        target_std=torch.tensor([1.0]),
        is_main=False,
    )
    assert metrics["baselines"]["train_mean"] is not None
    persistence = metrics["baselines"]["persistence"]
    assert persistence is not None
    assert np.isclose(persistence["mae"][0], 0.5)
    assert persistence["valid_count"] == [2.0]
    assert np.isclose(persistence["paired_model_rmse"][0], np.sqrt(5.0))


def test_multi_horizon_metrics_keep_model_and_persistence_on_same_mask():
    metrics = evaluate(
        _IdentityEncoder(),
        _MultiHorizonZeroProbe(),
        DataLoader(_MultiHorizonMetricDataset(), batch_size=2),
        torch.device("cpu"),
        target_mean=torch.zeros(2, 1),
        target_std=torch.ones(2, 1),
        is_main=False,
        future_horizons_seconds=(0.5, 1.0),
    )

    assert metrics["top_level_metric_horizon_seconds"] == 1.0
    assert metrics["future_horizons_seconds"] == [0.5, 1.0]
    assert metrics["valid_count"] == [2.0]
    endpoint_persistence = metrics["baselines"]["persistence"]
    assert endpoint_persistence["valid_count"] == [1.0]
    assert np.isclose(endpoint_persistence["rmse"][0], 1.5)
    assert np.isclose(endpoint_persistence["paired_model_rmse"][0], 2.0)
    assert np.isclose(endpoint_persistence["model_skill"][0], 1.0 - 4.0 / 2.25)

    first_horizon = metrics["per_horizon"]["0.5"]
    assert first_horizon["valid_count"] == [2.0]
    first_persistence = first_horizon["baselines"]["persistence"]
    assert first_persistence["valid_count"] == [1.0]
    assert np.isclose(first_persistence["rmse"][0], 0.5)
    assert np.isclose(first_persistence["paired_model_rmse"][0], 1.0)


if __name__ == "__main__":
    for test_name in sorted(name for name in globals() if name.startswith("test_")):
        print(f"RUN {test_name}")
        globals()[test_name]()
    print("PASS")
