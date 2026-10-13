"""Numerical metrics for AgiBot current arm reach rate."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    train_mean: float,
    train_std: float,
    train_mean_abs: float,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
    )
    if not np.any(mask):
        return {"count": 0}
    pred, truth = prediction[mask], target[mask]
    error = pred - truth
    absolute = np.abs(error)
    pred_center = pred - pred.mean()
    truth_center = truth - truth.mean()
    covariance = float(np.sum(pred_center * truth_center))
    pred_ss = float(np.sum(pred_center**2))
    truth_ss = float(np.sum(truth_center**2))
    pearson = (
        covariance / np.sqrt(pred_ss * truth_ss)
        if pred_ss > 0.0 and truth_ss > 0.0
        else np.nan
    )
    pred_var = float(np.mean(pred_center**2))
    truth_var = float(np.mean(truth_center**2))
    ccc_denominator = (
        pred_var + truth_var + float((pred.mean() - truth.mean()) ** 2)
    )
    ccc = (
        2.0 * float(np.mean(pred_center * truth_center)) / ccc_denominator
        if ccc_denominator > 0.0
        else np.nan
    )
    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(error**2)))
    baseline_sse = float(np.sum((truth - float(train_mean)) ** 2))
    return {
        "count": int(len(pred)),
        "mae": mae,
        "rmse": rmse,
        "nmae_train_mean_abs": mae / max(float(train_mean_abs), 1e-12),
        "nrmse_train_std": rmse / max(float(train_std), 1e-12),
        "bias": float(np.mean(error)),
        "pearson": float(np.clip(pearson, -1.0, 1.0)),
        "ccc": float(np.clip(ccc, -1.0, 1.0)),
        "r2": (
            float(1.0 - np.sum(error**2) / truth_ss)
            if truth_ss > 0.0
            else np.nan
        ),
        "skill_vs_train_mean": (
            float(1.0 - np.sum(error**2) / baseline_sse)
            if baseline_sse > 0.0
            else np.nan
        ),
        "p50_absolute": float(np.quantile(absolute, 0.50)),
        "p90_absolute": float(np.quantile(absolute, 0.90)),
        "p95_absolute": float(np.quantile(absolute, 0.95)),
    }


def direction_angle_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = (
        np.asarray(valid, dtype=bool)
        & np.all(np.isfinite(prediction), axis=-1)
        & np.all(np.isfinite(target), axis=-1)
    )
    if not np.any(mask):
        return {"count": 0}
    pred, truth = prediction[mask], target[mask]
    pred_norm = np.linalg.norm(pred, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    nonzero = (pred_norm > 1e-12) & (truth_norm > 1e-12)
    pred, truth = pred[nonzero], truth[nonzero]
    pred_norm, truth_norm = pred_norm[nonzero], truth_norm[nonzero]
    if not len(pred):
        return {"count": 0}
    cosine = np.sum(pred * truth, axis=-1) / (pred_norm * truth_norm)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "count": int(len(angle)),
        "mean_deg": float(np.mean(angle)),
        "median_deg": float(np.median(angle)),
        "p90_deg": float(np.quantile(angle, 0.90)),
        "p95_deg": float(np.quantile(angle, 0.95)),
    }


def paired_preference(
    correct_prediction: np.ndarray,
    wrong_prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict:
    correct_prediction = np.asarray(correct_prediction, dtype=np.float64)
    wrong_prediction = np.asarray(wrong_prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(correct_prediction)
        & np.isfinite(wrong_prediction)
        & np.isfinite(target)
    )
    if not np.any(mask):
        return {"count": 0}
    correct_error = np.abs(correct_prediction[mask] - target[mask])
    wrong_error = np.abs(wrong_prediction[mask] - target[mask])
    margin = wrong_error - correct_error
    return {
        "count": int(len(margin)),
        "strict_accuracy": float(np.mean(margin > 0.0)),
        "tie_aware_accuracy": float(
            np.mean((margin > 0.0) + 0.5 * (margin == 0.0))
        ),
        "mean_margin_wrong_minus_correct": float(np.mean(margin)),
        "median_margin_wrong_minus_correct": float(np.median(margin)),
    }


def macro_average(items: list[dict], keys: tuple[str, ...]) -> dict:
    usable = [item for item in items if item.get("count", 0)]
    output = {"component_count": len(usable)}
    for key in keys:
        values = [
            float(item[key])
            for item in usable
            if key in item and np.isfinite(item[key])
        ]
        output[key] = float(np.mean(values)) if values else np.nan
    return output
