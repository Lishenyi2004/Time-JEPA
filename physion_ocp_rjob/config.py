from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "model": {
        "model_name": "vit_giant_xformers",
        "resolution": 384,
        "patch_size": 16,
        "tubelet_size": 2,
        "img_temporal_dim_size": None,
        "uniform_power": True,
        "use_rope": True,
        "use_sdpa": True,
        "interpolate_rope": True,
        "modality_embedding": True,
        "wide_silu": True,
        "strict_checkpoint": True,
        "checkpoint_mmap": True,
    },
    "extraction": {
        "num_frames": 4,
        "frame_gap_ms": 150,
        "aggregation": "spatial_mean",
        "batch_size": 4,
        "num_workers": 4,
        "decord_num_threads": 1,
        "use_bfloat16": True,
        "feature_dtype": "float16",
        "seed": 239,
        "max_train_samples": None,
        "max_test_samples": None,
        "tasks": [
            {"split": "train", "temporal_order": "ordered"},
            {"split": "test", "temporal_order": "ordered"},
            {"split": "test", "temporal_order": "shuffled"},
        ],
    },
    "probe": {
        "protocol": "all_scenarios",
        "train_order": "ordered",
        "test_orders": ["ordered", "shuffled"],
        "cross_validation_folds": 5,
        "selection_metric": "accuracy",
        "c_values": [1.0e-6, 1.0e-5, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0],
        # Match the released Physion v1.5 LogisticRegression upper bound.  LBFGS
        # normally stops much earlier once the gradient tolerance is satisfied.
        "max_iter": 20000,
        "tolerance_grad": 1.0e-7,
        "prediction_batch_size": 128,
        "standardize": True,
        "seed": 239,
        "device": "cuda",
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    config = _deep_update(copy.deepcopy(DEFAULTS), user_config)
    config["config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for required in ("name", "checkpoint", "dataset_root", "output_dir"):
        if not config.get(required):
            raise ValueError(f"Missing required config field: {required}")

    model = config["model"]
    extraction = config["extraction"]
    probe = config["probe"]
    positive_ints = {
        "model.resolution": model["resolution"],
        "model.patch_size": model["patch_size"],
        "model.tubelet_size": model["tubelet_size"],
        "extraction.num_frames": extraction["num_frames"],
        "extraction.frame_gap_ms": extraction["frame_gap_ms"],
        "extraction.batch_size": extraction["batch_size"],
        "probe.cross_validation_folds": probe["cross_validation_folds"],
        "probe.max_iter": probe["max_iter"],
        "probe.prediction_batch_size": probe["prediction_batch_size"],
    }
    for field, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer; got {value!r}")
    if extraction["num_workers"] < 0 or extraction["decord_num_threads"] <= 0:
        raise ValueError("num_workers must be non-negative and decord_num_threads positive")
    if model["resolution"] % model["patch_size"]:
        raise ValueError("model.resolution must be divisible by model.patch_size")
    if extraction["num_frames"] % model["tubelet_size"]:
        raise ValueError("extraction.num_frames must be divisible by tubelet_size")
    if extraction["frame_gap_ms"] % 10:
        raise ValueError("Physion frame_gap_ms must be divisible by 10 ms")
    if extraction["aggregation"] not in {
        "flatten_tokens",
        "spatial_mean",
        "global_mean",
    }:
        raise ValueError(f"Unknown feature aggregation: {extraction['aggregation']}")
    if extraction["feature_dtype"] not in {"float16", "float32"}:
        raise ValueError("feature_dtype must be float16 or float32")
    for field in ("max_train_samples", "max_test_samples"):
        value = extraction.get(field)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"extraction.{field} must be null or a positive integer")

    tasks = extraction.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("extraction.tasks must be a non-empty list")
    seen_tasks = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError("Each extraction task must be a mapping")
        split = task.get("split")
        order = task.get("temporal_order")
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown Physion split: {split!r}")
        if order not in {"ordered", "shuffled"}:
            raise ValueError(f"Unknown temporal order: {order!r}")
        key = (split, order)
        if key in seen_tasks:
            raise ValueError(f"Duplicate extraction task: {key}")
        seen_tasks.add(key)

    if probe["protocol"] not in {"all_scenarios", "all_but_one"}:
        raise ValueError("probe.protocol must be all_scenarios or all_but_one")
    if probe["train_order"] not in {"ordered", "shuffled"}:
        raise ValueError("probe.train_order must be ordered or shuffled")
    if not probe["test_orders"]:
        raise ValueError("probe.test_orders must not be empty")
    if any(order not in {"ordered", "shuffled"} for order in probe["test_orders"]):
        raise ValueError("probe.test_orders contains an unknown order")
    required_tasks = {("train", probe["train_order"])} | {
        ("test", order) for order in probe["test_orders"]
    }
    if not required_tasks.issubset(seen_tasks):
        raise ValueError(
            f"extraction.tasks is missing probe feature sets: "
            f"{sorted(required_tasks - seen_tasks)}"
        )
    if probe["selection_metric"] not in {
        "accuracy",
        "auroc",
        "scenario_macro_accuracy",
        "scenario_macro_auroc",
    }:
        raise ValueError(f"Unknown probe selection metric: {probe['selection_metric']}")
    if int(probe["cross_validation_folds"]) < 2:
        raise ValueError("probe.cross_validation_folds must be at least 2")
    if not probe["c_values"] or any(float(value) <= 0 for value in probe["c_values"]):
        raise ValueError("probe.c_values must contain positive values")
    if float(probe["tolerance_grad"]) <= 0:
        raise ValueError("probe.tolerance_grad must be positive")
    if probe["device"] not in {"cuda", "cpu", "auto"}:
        raise ValueError("probe.device must be cuda, cpu, or auto")


def protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(config["checkpoint"]).expanduser()
    checkpoint_info: dict[str, Any] = {"path": str(checkpoint.resolve())}
    if checkpoint.is_file():
        stat = checkpoint.stat()
        checkpoint_info.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "name": config["name"],
        "checkpoint": checkpoint_info,
        "dataset_root": str(Path(config["dataset_root"]).expanduser().resolve()),
        "model": config["model"],
        "extraction": config["extraction"],
    }


def protocol_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(protocol_payload(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
