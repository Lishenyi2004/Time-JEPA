"""Dependency-light numerical metrics shared by the EgoExo gate and evaluator."""

from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def binary_discrimination(correct: np.ndarray, negative: np.ndarray) -> dict:
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    valid = np.isfinite(correct) & np.isfinite(negative)
    correct, negative = correct[valid], negative[valid]
    if not len(correct):
        return {"count": 0}
    labels = np.concatenate([np.ones(len(correct)), np.zeros(len(negative))])
    scores = -np.concatenate([correct, negative])
    ranks = average_ranks(scores)
    positives = labels == 1
    n_pos, n_neg = int(np.sum(positives)), int(np.sum(~positives))
    auroc = (
        float(np.sum(ranks[positives])) - n_pos * (n_pos + 1) / 2.0
    ) / max(n_pos * n_neg, 1)
    margin = negative - correct
    return {
        "count": int(len(correct)),
        "auroc": float(auroc),
        "paired_strict_accuracy": float(np.mean(margin > 0)),
        "paired_tie_aware_accuracy": float(
            np.mean((margin > 0) + 0.5 * (margin == 0))
        ),
        "mean_margin_negative_minus_correct_m": float(np.mean(margin)),
        "median_margin_negative_minus_correct_m": float(np.median(margin)),
    }


def error_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if valid is None:
        valid = np.ones(np.broadcast_shapes(prediction.shape, target.shape), dtype=bool)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(prediction) & np.isfinite(target)
    prediction = np.broadcast_to(prediction, mask.shape)[mask]
    target = np.broadcast_to(target, mask.shape)[mask]
    if not len(prediction):
        return {"count": 0}
    absolute = np.abs(prediction - target)
    squared = (prediction - target) ** 2
    mean_target_magnitude = float(np.mean(np.abs(target)))
    return {
        "count": int(len(absolute)),
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(squared))),
        "p50_m": float(np.quantile(absolute, 0.50)),
        "p90_m": float(np.quantile(absolute, 0.90)),
        "p95_m": float(np.quantile(absolute, 0.95)),
        "p99_m": float(np.quantile(absolute, 0.99)),
        "mean_target_magnitude_m": mean_target_magnitude,
        "mae_over_mean_target": float(np.mean(absolute) / max(mean_target_magnitude, 1e-12)),
    }


def scalar_regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    train_mean: float,
    train_std: float,
    unit: str,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(prediction) & np.isfinite(target)
    pred, truth = prediction[mask], target[mask]
    if not len(pred):
        return {"count": 0, "unit": unit}
    delta = pred - truth
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta**2)))
    pred_center = pred - pred.mean()
    target_center = truth - truth.mean()
    covariance_sum = float(np.sum(pred_center * target_center))
    pred_ss = float(np.sum(pred_center**2))
    target_ss = float(np.sum(target_center**2))
    pearson = (
        covariance_sum / np.sqrt(pred_ss * target_ss)
        if pred_ss > 0 and target_ss > 0
        else np.nan
    )
    pred_var = float(np.mean(pred_center**2))
    target_var = float(np.mean(target_center**2))
    ccc_denominator = pred_var + target_var + float((pred.mean() - truth.mean()) ** 2)
    ccc = (
        2.0 * float(np.mean(pred_center * target_center)) / ccc_denominator
        if ccc_denominator > 0
        else np.nan
    )
    baseline_sse = float(np.sum((truth - float(train_mean)) ** 2))
    return {
        "count": int(len(pred)),
        "unit": unit,
        "mae": mae,
        "rmse": rmse,
        "train_std_nrmse": rmse / max(float(train_std), 1e-12),
        "pearson": float(np.clip(pearson, -1.0, 1.0)),
        "ccc": float(np.clip(ccc, -1.0, 1.0)),
        "r2": float(1.0 - np.sum(delta**2) / target_ss) if target_ss > 0 else np.nan,
        "skill_vs_train_mean": (
            float(1.0 - np.sum(delta**2) / baseline_sse)
            if baseline_sse > 0
            else np.nan
        ),
    }


def hard_negative_summary(
    scores: dict[str, np.ndarray],
    active: np.ndarray | None = None,
) -> dict:
    names = [name for name in scores if name != "correct"]
    correct = np.asarray(scores["correct"], dtype=np.float64)
    if active is None:
        active = np.ones(correct.shape, dtype=bool)
    active = np.asarray(active, dtype=bool)
    pairwise = {}
    overall_aurocs, active_aurocs = [], []
    candidates = []
    for name in names:
        negative = np.asarray(scores[name], dtype=np.float64)
        overall = binary_discrimination(correct, negative)
        conditioned = binary_discrimination(correct[active], negative[active])
        pairwise[name] = {"overall": overall, "active_flexion": conditioned}
        if overall.get("count", 0):
            overall_aurocs.append(overall["auroc"])
        if conditioned.get("count", 0):
            active_aurocs.append(conditioned["auroc"])
        candidates.append(negative)
    matrix = np.stack(candidates, axis=-1)
    common = np.isfinite(correct) & np.all(np.isfinite(matrix), axis=-1)
    if np.any(common):
        correct_values = correct[common]
        negatives = matrix[common]
        strict = float(np.mean(np.all(correct_values[:, None] < negatives, axis=1)))
        all_values = np.concatenate([correct_values[:, None], negatives], axis=1)
        minima = np.min(all_values, axis=1, keepdims=True)
        tied = np.isclose(all_values, minima, rtol=1e-12, atol=1e-12)
        tie_recall = float(np.mean(tied[:, 0] / np.sum(tied, axis=1)))
        rank = 1.0 + np.sum(negatives < correct_values[:, None], axis=1)
        rank += 0.5 * np.sum(negatives == correct_values[:, None], axis=1)
        mrr = float(np.mean(1.0 / rank))
        hardest_margin = float(np.mean(np.min(negatives, axis=1) - correct_values))
    else:
        strict = tie_recall = mrr = hardest_margin = np.nan
    return {
        "negative_names": names,
        "retrieval_count": int(np.sum(common)),
        "strict_recall_at_1": strict,
        "tie_aware_expected_recall_at_1": tie_recall,
        "midrank_mrr": mrr,
        "overall_macro_auroc": (
            float(np.mean(overall_aurocs)) if overall_aurocs else np.nan
        ),
        "conditioned_macro_auroc": (
            float(np.mean(active_aurocs)) if active_aurocs else np.nan
        ),
        "mean_margin_hardest_negative_minus_correct_m": hardest_margin,
        "pairwise": pairwise,
    }

