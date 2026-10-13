from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def binary_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ordering = np.argsort(probabilities, kind="mergesort")
    sorted_values = probabilities[ordering]
    sorted_ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and sorted_values[end] == sorted_values[start]:
            end += 1
        # Ranks are one-based; ties receive their average rank.
        sorted_ranks[start:end] = 0.5 * ((start + 1) + end)
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[ordering] = sorted_ranks
    rank_sum_positive = float(ranks[positives].sum())
    return (
        rank_sum_positive - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    scenarios: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scenarios = np.asarray(scenarios, dtype=object)
    if not (len(labels) == len(probabilities) == len(scenarios)):
        raise ValueError("labels, probabilities, and scenarios must have equal length")
    predictions = (probabilities >= 0.5).astype(np.int64)

    def one_group(mask: np.ndarray) -> dict[str, Any]:
        group_labels = labels[mask]
        group_probabilities = probabilities[mask]
        group_predictions = predictions[mask]
        return {
            "n": int(mask.sum()),
            "positive_fraction": float(group_labels.mean()) if len(group_labels) else None,
            "accuracy": float((group_predictions == group_labels).mean())
            if len(group_labels)
            else None,
            "auroc": binary_auroc(group_labels, group_probabilities)
            if len(group_labels)
            else float("nan"),
        }

    overall = one_group(np.ones(len(labels), dtype=bool))
    per_scenario = {
        str(scenario): one_group(scenarios == scenario)
        for scenario in sorted(set(scenarios.tolist()))
    }
    scenario_accuracies = [
        value["accuracy"]
        for value in per_scenario.values()
        if value["accuracy"] is not None
    ]
    scenario_aurocs = [
        value["auroc"]
        for value in per_scenario.values()
        if value["auroc"] is not None and math.isfinite(value["auroc"])
    ]
    return {
        **overall,
        "scenario_macro_accuracy": float(np.mean(scenario_accuracies)),
        "scenario_macro_auroc": float(np.mean(scenario_aurocs))
        if scenario_aurocs
        else float("nan"),
        "per_scenario": per_scenario,
    }


def stratified_kfold_indices(
    labels: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=np.int64)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    rng = np.random.default_rng(int(seed))
    fold_members: list[list[int]] = [[] for _ in range(n_splits)]
    for label in sorted(np.unique(labels).tolist()):
        members = np.flatnonzero(labels == label)
        if len(members) < n_splits:
            raise ValueError(
                f"Class {label} has {len(members)} samples, fewer than {n_splits} folds"
            )
        rng.shuffle(members)
        for offset, member in enumerate(members.tolist()):
            fold_members[offset % n_splits].append(member)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(labels), dtype=np.int64)
    for members in fold_members:
        validation = np.asarray(sorted(members), dtype=np.int64)
        training_mask = np.ones(len(labels), dtype=bool)
        training_mask[validation] = False
        folds.append((all_indices[training_mask], validation))
    return folds


def feature_statistics(
    features: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot compute feature statistics on an empty subset")
    feature_dim = int(features.shape[1])
    total = np.zeros(feature_dim, dtype=np.float64)
    square_total = np.zeros(feature_dim, dtype=np.float64)
    for start in range(0, len(indices), max(1, int(batch_size))):
        rows = indices[start : start + max(1, int(batch_size))]
        chunk = np.asarray(features[rows], dtype=np.float32)
        total += chunk.sum(axis=0, dtype=np.float64)
        square_total += np.square(chunk).sum(axis=0, dtype=np.float64)
    mean = total / len(indices)
    variance = np.maximum(square_total / len(indices) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1.0e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _materialize_standardized(
    features: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> torch.Tensor:
    indices = np.asarray(indices, dtype=np.int64)
    array = np.empty((len(indices), features.shape[1]), dtype=np.float32)
    for start in range(0, len(indices), max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), len(indices))
        array[start:end] = features[indices[start:end]]
    tensor = torch.from_numpy(array).to(device)
    mean_tensor = torch.from_numpy(mean).to(device)
    scale_tensor = torch.from_numpy(scale).to(device)
    tensor.sub_(mean_tensor).div_(scale_tensor)
    return tensor


def _fit_lbfgs(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    c_value: float,
    max_iter: int,
    tolerance_grad: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    if len(torch.unique(labels)) != 2:
        raise ValueError("Binary logistic regression requires both labels")
    linear = torch.nn.Linear(features.shape[1], 1, device=features.device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.LBFGS(
        linear.parameters(),
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=float(tolerance_grad),
        tolerance_change=1.0e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    sample_count = int(len(labels))

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = linear(features).squeeze(1)
        data_loss = F.binary_cross_entropy_with_logits(logits, labels)
        regularizer = linear.weight.square().sum() / (
            2.0 * float(c_value) * sample_count
        )
        loss = data_loss + regularizer
        loss.backward()
        return loss

    final_loss = float(optimizer.step(closure).detach().cpu())
    state = optimizer.state.get(linear.weight, {})
    diagnostics = {
        "objective": final_loss,
        "iterations": int(state.get("n_iter", -1)),
        "function_evaluations": int(state.get("func_evals", -1)),
    }
    return (
        linear.weight.detach().float().cpu().reshape(-1),
        linear.bias.detach().float().cpu().reshape(()),
        diagnostics,
    )


def _standardized_probabilities(
    features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> np.ndarray:
    with torch.inference_mode():
        probabilities = torch.sigmoid(features @ weight.to(features.device) + bias.to(features.device))
    return probabilities.float().cpu().numpy()


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    value = float(metrics[metric_name])
    return value if math.isfinite(value) else -float("inf")


@dataclass
class FittedProbe:
    mean: np.ndarray
    scale: np.ndarray
    weight: torch.Tensor
    bias: torch.Tensor
    c_value: float
    cv_results: list[dict[str, Any]]
    fit_diagnostics: dict[str, Any]
    train_metrics: dict[str, Any]

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": torch.from_numpy(self.mean),
            "scale": torch.from_numpy(self.scale),
            "weight": self.weight,
            "bias": self.bias,
            "c_value": self.c_value,
            "cv_results": self.cv_results,
            "fit_diagnostics": self.fit_diagnostics,
            "train_metrics": self.train_metrics,
        }


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    scenarios: np.ndarray,
    train_indices: np.ndarray,
    probe_config: dict[str, Any],
    device: torch.device,
) -> FittedProbe:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    scenarios = np.asarray(scenarios, dtype=object)
    subset_labels = labels[train_indices]
    folds = stratified_kfold_indices(
        subset_labels,
        int(probe_config["cross_validation_folds"]),
        int(probe_config["seed"]),
    )
    c_values = [float(value) for value in probe_config["c_values"]]
    fold_scores: dict[float, list[float]] = {value: [] for value in c_values}
    fold_diagnostics: dict[float, list[dict[str, Any]]] = {
        value: [] for value in c_values
    }
    selection_metric = str(probe_config["selection_metric"])
    for fold_id, (local_train, local_validation) in enumerate(folds):
        fold_train = train_indices[local_train]
        fold_validation = train_indices[local_validation]
        if bool(probe_config["standardize"]):
            mean, scale = feature_statistics(features, fold_train)
        else:
            mean = np.zeros(features.shape[1], dtype=np.float32)
            scale = np.ones(features.shape[1], dtype=np.float32)
        x_train = _materialize_standardized(
            features, fold_train, mean, scale, device
        )
        x_validation = _materialize_standardized(
            features, fold_validation, mean, scale, device
        )
        y_train = torch.from_numpy(labels[fold_train].astype(np.float32)).to(device)
        for c_value in c_values:
            weight, bias, diagnostics = _fit_lbfgs(
                x_train,
                y_train,
                c_value=c_value,
                max_iter=int(probe_config["max_iter"]),
                tolerance_grad=float(probe_config["tolerance_grad"]),
            )
            probabilities = _standardized_probabilities(
                x_validation, weight, bias
            )
            metrics = classification_metrics(
                labels[fold_validation], probabilities, scenarios[fold_validation]
            )
            score = _metric_value(metrics, selection_metric)
            fold_scores[c_value].append(score)
            fold_diagnostics[c_value].append(
                {
                    "fold": fold_id,
                    "score": score,
                    **diagnostics,
                }
            )
        del x_train, x_validation, y_train
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cv_results = [
        {
            "c_value": c_value,
            "mean_score": float(np.mean(fold_scores[c_value])),
            "std_score": float(np.std(fold_scores[c_value])),
            "folds": fold_diagnostics[c_value],
        }
        for c_value in c_values
    ]
    best = max(cv_results, key=lambda item: item["mean_score"])
    best_c = float(best["c_value"])

    if bool(probe_config["standardize"]):
        mean, scale = feature_statistics(features, train_indices)
    else:
        mean = np.zeros(features.shape[1], dtype=np.float32)
        scale = np.ones(features.shape[1], dtype=np.float32)
    x_train = _materialize_standardized(features, train_indices, mean, scale, device)
    y_train = torch.from_numpy(labels[train_indices].astype(np.float32)).to(device)
    weight, bias, fit_diagnostics = _fit_lbfgs(
        x_train,
        y_train,
        c_value=best_c,
        max_iter=int(probe_config["max_iter"]),
        tolerance_grad=float(probe_config["tolerance_grad"]),
    )
    train_probabilities = _standardized_probabilities(x_train, weight, bias)
    train_metrics = classification_metrics(
        labels[train_indices], train_probabilities, scenarios[train_indices]
    )
    del x_train, y_train
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return FittedProbe(
        mean=mean,
        scale=scale,
        weight=weight,
        bias=bias,
        c_value=best_c,
        cv_results=cv_results,
        fit_diagnostics=fit_diagnostics,
        train_metrics=train_metrics,
    )


def predict_probe(
    fitted: FittedProbe,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    probabilities = np.empty(len(indices), dtype=np.float32)
    weight = fitted.weight.to(device)
    bias = fitted.bias.to(device)
    mean = torch.from_numpy(fitted.mean).to(device)
    scale = torch.from_numpy(fitted.scale).to(device)
    with torch.inference_mode():
        for start in range(0, len(indices), max(1, int(batch_size))):
            end = min(start + max(1, int(batch_size)), len(indices))
            array = np.asarray(features[indices[start:end]], dtype=np.float32)
            tensor = torch.from_numpy(array).to(device)
            tensor.sub_(mean).div_(scale)
            values = torch.sigmoid(tensor @ weight + bias)
            probabilities[start:end] = values.float().cpu().numpy()
    return probabilities
