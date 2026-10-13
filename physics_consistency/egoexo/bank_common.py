"""Frozen protocol contracts and small IO helpers for EgoExo consistency runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
LEGACY_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for _path in (REPO_ROOT, LEGACY_PROBE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from world_model_probe_rjob.physics_consistency_encoder import (
    WORLD_MODEL_BACKENDS,
    validate_world_model_encoder_config,
)


STANDARD_SPLIT_PREFIX = "egoexo_exo124_view3ood"
TRAIN_VIEWS = ("view01", "view02", "view04")
ID_VIEWS = TRAIN_VIEWS
OOD_VIEWS = ("view03",)
ALL_EVAL_VIEWS = ID_VIEWS + OOD_VIEWS

TARGET_SPECS = (
    {
        "name": "left_elbow_flexion",
        "field": "left_elbow_flexion",
        "side": "left",
        "family": "angle",
        "unit": "rad",
        "role": "cosine_law_input",
        "encoder_supervised": True,
    },
    {
        "name": "right_elbow_flexion",
        "field": "right_elbow_flexion",
        "side": "right",
        "family": "angle",
        "unit": "rad",
        "role": "cosine_law_input",
        "encoder_supervised": True,
    },
    {
        "name": "left_wrist_shoulder_dist",
        "field": "left_wrist_shoulder_dist",
        "side": "left",
        "family": "distance",
        "unit": "m",
        "role": "direct_endpoint",
        "encoder_supervised": False,
    },
    {
        "name": "right_wrist_shoulder_dist",
        "field": "right_wrist_shoulder_dist",
        "side": "right",
        "family": "distance",
        "unit": "m",
        "role": "direct_endpoint",
        "encoder_supervised": False,
    },
)
TARGET_NAMES = tuple(spec["name"] for spec in TARGET_SPECS)
TARGET_FIELDS = tuple(spec["field"] for spec in TARGET_SPECS)
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}
ANGLE_INDEX = {
    side: TARGET_INDEX[f"{side}_elbow_flexion"] for side in ("left", "right")
}
DISTANCE_INDEX = {
    side: TARGET_INDEX[f"{side}_wrist_shoulder_dist"] for side in ("left", "right")
}


def target_manifest() -> list[dict]:
    return [dict(spec) for spec in TARGET_SPECS]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def deep_update(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_with_base(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    if path in chain:
        raise ValueError(f"Cyclic YAML base chain: {[str(item) for item in (*chain, path)]}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    base = config.pop("base", None)
    if base is None:
        return config
    base_path = Path(base).expanduser()
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    return deep_update(_load_yaml_with_base(base_path, (*chain, path)), config)


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    config = deep_update(_load_yaml_with_base(path), overrides or {})
    config["config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    phase = config.get("experiment", {}).get("phase")
    if phase not in {"full", "smoke"}:
        raise ValueError("experiment.phase must be full or smoke")
    sampling = config.get("sampling", {})
    num_frames = int(sampling.get("num_frames", 0))
    tubelet = int(sampling.get("tubelet_size", 0))
    if num_frames <= 0 or tubelet <= 0 or num_frames % tubelet:
        raise ValueError("sampling.num_frames must be positive and divisible by tubelet_size")
    data = config.get("data", {})
    if str(data.get("train_split")) != "probe_train":
        raise ValueError("data.train_split is frozen to probe_train")
    if str(data.get("val_split")) != "probe_val":
        raise ValueError("data.val_split is frozen to probe_val")
    mode = config.get("evaluation", {}).get("mode")
    if phase == "full" and mode != "test":
        raise ValueError("Full protocol requires evaluation.mode=test")
    if phase == "smoke" and mode not in {"smoke", "val"}:
        raise ValueError("Smoke protocol must not read final test")
    if phase == "full":
        if num_frames != 64 or tubelet != 2 or float(sampling.get("fps", 0.0)) != 5.0:
            raise ValueError("Formal EgoExo protocol is frozen to 64 frames, 5 fps, tubelet 2")
        if data.get("split_prefix") != STANDARD_SPLIT_PREFIX:
            raise ValueError(f"Formal split_prefix must be {STANDARD_SPLIT_PREFIX}")
        if tuple(data.get("train_views", ())) != TRAIN_VIEWS:
            raise ValueError(f"Formal train views must be {TRAIN_VIEWS}")
        if tuple(data.get("id_views", ())) != ID_VIEWS:
            raise ValueError(f"Formal ID views must be {ID_VIEWS}")
        if tuple(data.get("ood_views", ())) != OOD_VIEWS:
            raise ValueError(f"Formal OOD views must be {OOD_VIEWS}")
        backend = config.get("encoder", {}).get("backend")
        if backend not in {"vjepa2_1", *WORLD_MODEL_BACKENDS}:
            raise ValueError(
                "Formal runs require encoder.backend to be vjepa2_1 or a "
                "registered world-model backend"
            )
        probe = config.get("probe", {})
        if probe.get("type", "independent_linear") != "independent_linear":
            raise ValueError("Formal primary protocol requires independent_linear heads")
    encoder = config.get("encoder", {})
    backend = encoder.get("backend")
    if backend == "vjepa2_1":
        if not encoder.get("checkpoint"):
            raise ValueError("encoder.checkpoint is required")
    elif backend in WORLD_MODEL_BACKENDS:
        validate_world_model_encoder_config(config)
    gate_path = config.get("kinematics", {}).get("gt_gate_metrics")
    if not gate_path:
        raise ValueError("kinematics.gt_gate_metrics is required")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
