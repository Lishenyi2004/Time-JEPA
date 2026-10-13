import math
import sys
from pathlib import Path

import torch


PROBE_DIR = Path(__file__).resolve().parents[1] / "covla_probe_rjob"
sys.path.insert(0, str(PROBE_DIR))

from probe_engine import _regression_metrics_from_sums, evaluate  # noqa: E402


def _metrics(pred, target, train_mean, train_std, valid=None):
    pred = torch.as_tensor(pred, dtype=torch.float64).reshape(-1, 1)
    target = torch.as_tensor(target, dtype=torch.float64).reshape(-1, 1)
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = torch.as_tensor(valid, dtype=torch.float64).reshape(-1, 1)
    error = pred - target
    return _regression_metrics_from_sums(
        abs_error_sum=(error.abs() * valid).sum(dim=0),
        sq_error_sum=(error.square() * valid).sum(dim=0),
        pred_sum=(pred * valid).sum(dim=0),
        target_sum=(target * valid).sum(dim=0),
        pred_sq_sum=(pred.square() * valid).sum(dim=0),
        target_sq_sum=(target.square() * valid).sum(dim=0),
        pred_target_sum=(pred * target * valid).sum(dim=0),
        valid_sum=valid.sum(dim=0),
        train_target_mean=torch.tensor([train_mean], dtype=torch.float64),
        train_target_std=torch.tensor([train_std], dtype=torch.float64),
    )


def test_agreement_metrics_are_one_for_perfect_predictions():
    metrics = _metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], train_mean=2.0, train_std=1.0)

    for name in ("pearson", "ccc", "r2", "skill_train_mean"):
        torch.testing.assert_close(metrics[name], torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(metrics["rmse"], torch.zeros(1, dtype=torch.float64))


def test_ccc_penalizes_affine_miscalibration_that_pearson_ignores():
    metrics = _metrics([3.0, 5.0, 7.0], [1.0, 2.0, 3.0], train_mean=2.0, train_std=1.0)

    torch.testing.assert_close(metrics["pearson"], torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(metrics["ccc"], torch.tensor([8.0 / 37.0], dtype=torch.float64))
    torch.testing.assert_close(metrics["r2"], torch.tensor([-13.5], dtype=torch.float64))
    torch.testing.assert_close(metrics["skill_train_mean"], torch.tensor([-13.5], dtype=torch.float64))
    torch.testing.assert_close(metrics["rmse"], torch.tensor([math.sqrt(29.0 / 3.0)], dtype=torch.float64))


def test_train_mean_skill_uses_a_different_baseline_from_r2():
    metrics = _metrics([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], train_mean=0.0, train_std=1.0)

    torch.testing.assert_close(metrics["r2"], torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(metrics["skill_train_mean"], torch.tensor([6.0 / 7.0], dtype=torch.float64))
    torch.testing.assert_close(metrics["ccc"], torch.zeros(1, dtype=torch.float64))


def test_metrics_respect_valid_mask_and_remain_finite_for_constant_targets():
    masked = _metrics(
        [1.0, 2.0, 999.0],
        [1.0, 2.0, 3.0],
        train_mean=1.5,
        train_std=0.5,
        valid=[1.0, 1.0, 0.0],
    )
    torch.testing.assert_close(masked["ccc"], torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(masked["r2"], torch.ones(1, dtype=torch.float64))

    constant = _metrics([2.0, 2.0], [2.0, 2.0], train_mean=2.0, train_std=1.0)
    for value in constant.values():
        assert torch.isfinite(value).all()


def test_evaluate_exposes_new_metrics_in_result_dictionary():
    target = torch.tensor([[[1.0], [2.0], [3.0]]])
    loader = [{"video": target.clone(), "target": target.clone(), "valid": torch.ones_like(target)}]

    metrics = evaluate(
        encoder=torch.nn.Identity(),
        probe=torch.nn.Identity(),
        loader=loader,
        device=torch.device("cpu"),
        target_mean=torch.zeros(1),
        target_std=torch.ones(1),
        is_main=False,
    )

    assert {"ccc", "r2", "skill_train_mean"}.issubset(metrics)
    assert metrics["ccc"] == [1.0]
    assert metrics["r2"] == [1.0]
    assert metrics["skill_train_mean"] == [1.0]
