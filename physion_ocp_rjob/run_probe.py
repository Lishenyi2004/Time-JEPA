from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physion_ocp_rjob.config import load_config, protocol_digest, validate_config
from physion_ocp_rjob.dataset import SCENARIOS
from physion_ocp_rjob.feature_store import FeatureSet, load_feature_set
from physion_ocp_rjob.probe import (
    FittedProbe,
    classification_metrics,
    fit_probe,
    predict_probe,
)


LOGGER = logging.getLogger("physion_ocp_probe")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("probe.device=cuda, but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _probe_digest(probe_config: dict[str, Any], feature_digest: str) -> str:
    payload = json.dumps(
        {"feature_protocol": feature_digest, "probe": probe_config},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-p{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(temporary, destination)


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-p{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _save_fitted_probe(
    fitted: FittedProbe,
    destination: Path,
    *,
    config: dict[str, Any],
    probe_config: dict[str, Any],
    feature_digest: str,
    probe_digest: str,
    training_scenarios: list[str],
) -> None:
    payload = {
        "format_version": 1,
        "model_name": config["name"],
        "feature_protocol_digest": feature_digest,
        "probe_protocol_digest": probe_digest,
        "probe_config": probe_config,
        "training_scenarios": training_scenarios,
        "pipeline": fitted.state_dict(),
    }
    _atomic_torch_save(payload, destination)


def _assert_feature_alignment(
    reference: dict[str, np.ndarray], candidate: FeatureSet, reference_order: str
) -> None:
    for field in ("global_index", "label", "name", "scenario", "relative_path"):
        left = reference[field]
        right = getattr(candidate, field)
        if not np.array_equal(left, right):
            raise RuntimeError(
                f"Test metadata differs between {reference_order} and "
                f"{candidate.temporal_order}: {field}"
            )


def _metrics_rows(
    protocol: str, temporal_order: str, metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = [
        {
            "protocol": protocol,
            "temporal_order": temporal_order,
            "scenario": "all",
            "n": metrics["n"],
            "positive_fraction": metrics["positive_fraction"],
            "accuracy": metrics["accuracy"],
            "auroc": metrics["auroc"],
            "scenario_macro_accuracy": metrics["scenario_macro_accuracy"],
            "scenario_macro_auroc": metrics["scenario_macro_auroc"],
        }
    ]
    for scenario, values in metrics["per_scenario"].items():
        rows.append(
            {
                "protocol": protocol,
                "temporal_order": temporal_order,
                "scenario": scenario,
                "n": values["n"],
                "positive_fraction": values["positive_fraction"],
                "accuracy": values["accuracy"],
                "auroc": values["auroc"],
                "scenario_macro_accuracy": "",
                "scenario_macro_auroc": "",
            }
        )
    return rows


def _predictions_csv(feature_set: FeatureSet, probabilities: np.ndarray) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "global_index",
            "stimulus_name",
            "scenario",
            "label",
            "probability_contact",
            "prediction",
            "contact_frame",
            "frame_indices",
            "relative_path",
        ]
    )
    predictions = (probabilities >= 0.5).astype(np.int64)
    for index in range(len(feature_set.label)):
        writer.writerow(
            [
                int(feature_set.global_index[index]),
                feature_set.name[index],
                feature_set.scenario[index],
                int(feature_set.label[index]),
                float(probabilities[index]),
                int(predictions[index]),
                int(feature_set.contact_frame[index]),
                " ".join(str(int(value)) for value in feature_set.frame_indices[index]),
                feature_set.relative_path[index],
            ]
        )
    return output.getvalue()


def _metrics_csv(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    output = StringIO()
    fieldnames = [
        "protocol",
        "temporal_order",
        "scenario",
        "n",
        "positive_fraction",
        "accuracy",
        "auroc",
        "scenario_macro_accuracy",
        "scenario_macro_auroc",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(_json_safe(row))
    return output.getvalue()


def _summary_markdown(results: dict[str, Any]) -> str:
    lines = [
        f"# Physion OCP: {results['model_name']}",
        "",
        f"- Probe protocol: `{results['protocol']}`",
        f"- Feature protocol: `{results['feature_protocol_digest']}`",
        f"- Probe protocol digest: `{results['probe_protocol_digest']}`",
        f"- Feature dimension: {results['feature_dimension']}",
        "",
        "| Input order | N | Accuracy | AUROC | Scenario-macro accuracy | Scenario-macro AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for order, payload in results["test"].items():
        metrics = payload["metrics"]
        lines.append(
            f"| {order} | {metrics['n']} | {metrics['accuracy']:.4f} | "
            f"{metrics['auroc']:.4f} | {metrics['scenario_macro_accuracy']:.4f} | "
            f"{metrics['scenario_macro_auroc']:.4f} |"
        )
    if results.get("ordered_minus_shuffled"):
        delta = results["ordered_minus_shuffled"]
        lines.extend(
            [
                "",
                "Ordered minus shuffled:",
                "",
                f"- accuracy: {delta['accuracy']:+.4f}",
                f"- AUROC: {delta['auroc']:+.4f}",
                f"- scenario-macro accuracy: {delta['scenario_macro_accuracy']:+.4f}",
                f"- scenario-macro AUROC: {delta['scenario_macro_auroc']:+.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "The encoder is frozen. The readout is a standardized one-layer logistic "
            f"probe with {results['probe_config']['cross_validation_folds']}-fold "
            "stratified C selection; temporal shuffle uses the same fitted probe.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config: dict[str, Any],
    *,
    device_override: str | None = None,
    protocol_override: str | None = None,
) -> Path:
    feature_digest = protocol_digest(config)
    probe_config = dict(config["probe"])
    if device_override is not None:
        probe_config["device"] = device_override
    if protocol_override is not None:
        probe_config["protocol"] = protocol_override
    device = _resolve_device(str(probe_config["device"]))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    np.random.seed(int(probe_config["seed"]))
    torch.manual_seed(int(probe_config["seed"]))
    LOGGER.info("Loading ordered training features")
    train = load_feature_set(
        config["output_dir"],
        "train",
        str(probe_config["train_order"]),
        expected_digest=feature_digest,
    )
    if train.features.ndim != 2:
        raise RuntimeError(f"Expected flattened feature matrix, got {train.features.shape}")
    protocol = str(probe_config["protocol"])
    output_dir = (
        Path(config["output_dir"]).expanduser().resolve() / "probe" / protocol
    )
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    current_probe_digest = _probe_digest(probe_config, feature_digest)
    all_train_indices = np.arange(len(train.label), dtype=np.int64)
    fitted_probes: dict[str, FittedProbe] = {}
    probe_reports: dict[str, Any] = {}

    if protocol == "all_scenarios":
        LOGGER.info(
            "Fitting all-scenarios probe: n=%d d=%d device=%s",
            len(train.label),
            train.features.shape[1],
            device,
        )
        fitted = fit_probe(
            train.features,
            train.label,
            train.scenario,
            all_train_indices,
            probe_config,
            device,
        )
        fitted_probes["all"] = fitted
        probe_reports["all"] = {
            "training_scenarios": sorted(set(train.scenario.tolist())),
            "n_train": len(train.label),
            "selected_c": fitted.c_value,
            "cv_results": fitted.cv_results,
            "fit_diagnostics": fitted.fit_diagnostics,
            "train_metrics": fitted.train_metrics,
        }
        _save_fitted_probe(
            fitted,
            models_dir / "all.pt",
            config=config,
            probe_config=probe_config,
            feature_digest=feature_digest,
            probe_digest=current_probe_digest,
            training_scenarios=probe_reports["all"]["training_scenarios"],
        )
    elif protocol == "all_but_one":
        available_scenarios = set(train.scenario.tolist())
        missing = set(SCENARIOS) - available_scenarios
        if missing:
            raise RuntimeError(f"Training features are missing Physion scenarios: {missing}")
        for held_out in SCENARIOS:
            indices = np.flatnonzero(train.scenario != held_out)
            training_scenarios = sorted(set(train.scenario[indices].tolist()))
            LOGGER.info(
                "Fitting all-but-one probe held_out=%s n=%d d=%d",
                held_out,
                len(indices),
                train.features.shape[1],
            )
            fitted = fit_probe(
                train.features,
                train.label,
                train.scenario,
                indices,
                probe_config,
                device,
            )
            fitted_probes[held_out] = fitted
            probe_reports[held_out] = {
                "held_out_scenario": held_out,
                "training_scenarios": training_scenarios,
                "n_train": len(indices),
                "selected_c": fitted.c_value,
                "cv_results": fitted.cv_results,
                "fit_diagnostics": fitted.fit_diagnostics,
                "train_metrics": fitted.train_metrics,
            }
            _save_fitted_probe(
                fitted,
                models_dir / f"held_out_{held_out}.pt",
                config=config,
                probe_config=probe_config,
                feature_digest=feature_digest,
                probe_digest=current_probe_digest,
                training_scenarios=training_scenarios,
            )
    else:  # guarded by config validation
        raise ValueError(f"Unknown probe protocol: {protocol}")

    results: dict[str, Any] = {
        "format_version": 1,
        "model_name": config["name"],
        "protocol": protocol,
        "feature_protocol_digest": feature_digest,
        "probe_protocol_digest": current_probe_digest,
        "feature_dimension": int(train.features.shape[1]),
        "train_samples": int(len(train.label)),
        "probe_config": probe_config,
        "probes": probe_reports,
        "test": {},
    }
    metric_rows: list[dict[str, Any]] = []
    reference_metadata: dict[str, np.ndarray] | None = None
    reference_order: str | None = None
    for temporal_order in probe_config["test_orders"]:
        LOGGER.info("Loading and scoring test features: %s", temporal_order)
        test = load_feature_set(
            config["output_dir"],
            "test",
            str(temporal_order),
            expected_digest=feature_digest,
        )
        if test.features.shape[1] != train.features.shape[1]:
            raise RuntimeError(
                f"Feature dimension mismatch: train={train.features.shape[1]}, "
                f"test={test.features.shape[1]}"
            )
        if reference_metadata is None:
            reference_metadata = {
                field: getattr(test, field).copy()
                for field in ("global_index", "label", "name", "scenario", "relative_path")
            }
            reference_order = str(temporal_order)
        else:
            assert reference_order is not None
            _assert_feature_alignment(reference_metadata, test, reference_order)
        probabilities = np.empty(len(test.label), dtype=np.float32)
        if protocol == "all_scenarios":
            probabilities[:] = predict_probe(
                fitted_probes["all"],
                test.features,
                np.arange(len(test.label), dtype=np.int64),
                device,
                batch_size=int(probe_config["prediction_batch_size"]),
            )
        else:
            for scenario in SCENARIOS:
                indices = np.flatnonzero(test.scenario == scenario)
                if not len(indices):
                    raise RuntimeError(f"Test features have no {scenario} examples")
                probabilities[indices] = predict_probe(
                    fitted_probes[scenario],
                    test.features,
                    indices,
                    device,
                    batch_size=int(probe_config["prediction_batch_size"]),
                )
        metrics = classification_metrics(test.label, probabilities, test.scenario)
        results["test"][str(temporal_order)] = {"metrics": metrics}
        metric_rows.extend(_metrics_rows(protocol, str(temporal_order), metrics))
        _atomic_text(
            _predictions_csv(test, probabilities),
            output_dir / f"predictions_{temporal_order}.csv",
        )
        # Do not retain test feature matrices across orders.
        del test, probabilities

    if "ordered" in results["test"] and "shuffled" in results["test"]:
        ordered = results["test"]["ordered"]["metrics"]
        shuffled = results["test"]["shuffled"]["metrics"]
        results["ordered_minus_shuffled"] = {
            metric: float(ordered[metric]) - float(shuffled[metric])
            for metric in (
                "accuracy",
                "auroc",
                "scenario_macro_accuracy",
                "scenario_macro_auroc",
            )
        }
    _atomic_text(_metrics_csv(metric_rows), output_dir / "metrics.csv")
    _atomic_text(_summary_markdown(results), output_dir / "summary.md")
    _atomic_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        output_dir / "results.json",
    )
    LOGGER.info("Saved probe results under %s", output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default=None)
    parser.add_argument(
        "--protocol", choices=("all_scenarios", "all_but_one"), default=None
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    validate_config(config)
    run(
        config,
        device_override=args.device,
        protocol_override=args.protocol,
    )


if __name__ == "__main__":
    main()
