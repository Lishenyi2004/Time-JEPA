"""Shared constants, configuration loading, and small IO helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
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


# Order is a public file-format contract.  The first nine are the main SE(2)
# variables; v_ego/a_ego are an explicitly auxiliary, seen-variable check.
TARGET_SPECS = (
    {"name": "vx_calib", "source": "state", "field": "velocities_calib[0]", "role": "main"},
    {"name": "vy_calib", "source": "state", "field": "velocities_calib[1]", "role": "main"},
    {"name": "yaw_rate", "source": "state", "field": "angular_velocities_calib[2]", "role": "main"},
    {"name": "trajectory_20_x", "source": "trajectory", "point": 20, "axis": 0, "horizon": 1.0, "role": "main"},
    {"name": "trajectory_20_y", "source": "trajectory", "point": 20, "axis": 1, "horizon": 1.0, "role": "main"},
    {"name": "trajectory_40_x", "source": "trajectory", "point": 40, "axis": 0, "horizon": 2.0, "role": "main"},
    {"name": "trajectory_40_y", "source": "trajectory", "point": 40, "axis": 1, "horizon": 2.0, "role": "main"},
    {"name": "trajectory_59_x", "source": "trajectory", "point": 59, "axis": 0, "horizon": 2.95, "role": "main"},
    {"name": "trajectory_59_y", "source": "trajectory", "point": 59, "axis": 1, "horizon": 2.95, "role": "main"},
    {"name": "v_ego", "source": "state", "field": "vEgo", "role": "auxiliary"},
    {"name": "a_ego", "source": "state", "field": "aEgo", "role": "auxiliary"},
)
TARGET_NAMES = tuple(item["name"] for item in TARGET_SPECS)
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}
STATE_FIELDS = (
    "velocities_calib[0]",
    "velocities_calib[1]",
    "angular_velocities_calib[2]",
    "vEgo",
    "aEgo",
)
TRAJECTORY_POINTS = (20, 40, 59)
PRIMARY_HORIZONS = ((1.0, 20), (2.0, 40), (2.95, 59))
VA_HORIZONS = (0.5, 1.0, 2.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
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
    config = _load_yaml_with_base(path)
    config = deep_update(config, overrides or {})
    config["config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    phase = config.get("experiment", {}).get("phase")
    if phase not in {"pilot", "full", "smoke"}:
        raise ValueError("experiment.phase must be pilot, full, or smoke")
    sampling = config.get("sampling", {})
    num_frames = int(sampling.get("num_frames", 0))
    tubelet = int(sampling.get("tubelet_size", 0))
    if num_frames <= 0 or tubelet <= 0 or num_frames % tubelet:
        raise ValueError("sampling.num_frames must be positive and divisible by tubelet_size")
    if phase in {"pilot", "full"}:
        if num_frames != 64 or int(sampling.get("frame_stride", 0)) != 5 or tubelet != 2:
            raise ValueError("Formal COVLA protocol is frozen to 64 frames, stride 5, tubelet 2")
        backend = config.get("encoder", {}).get("backend")
        if backend not in {"vjepa2_1", *WORLD_MODEL_BACKENDS}:
            raise ValueError(
                "pilot/full requires vjepa2_1 or a registered world-model backend"
            )
    eval_splits = list(config.get("data", {}).get("eval_splits", []))
    if phase == "pilot" and eval_splits != ["probe_val"]:
        raise ValueError("Pilot leakage guard: data.eval_splits must be exactly [probe_val]")
    if phase == "full" and eval_splits != ["test", "weather_ood_test"]:
        raise ValueError("Full protocol requires eval_splits: [test, weather_ood_test]")
    if phase == "smoke" and any(name in {"test", "weather_ood_test"} for name in eval_splits):
        raise ValueError("Smoke test must not read final test/OOD splits")
    if str(config.get("data", {}).get("train_split")) != "probe_train":
        raise ValueError("data.train_split is frozen to probe_train")
    if str(config.get("data", {}).get("val_split")) != "probe_val":
        raise ValueError("data.val_split is frozen to probe_val")
    checkpoint = config.get("encoder", {}).get("checkpoint")
    backend = config.get("encoder", {}).get("backend")
    if backend == "vjepa2_1" and not checkpoint:
        raise ValueError("encoder.checkpoint is required for vjepa2_1")
    if backend in WORLD_MODEL_BACKENDS:
        validate_world_model_encoder_config(config)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_manifest() -> list[dict]:
    return [dict(spec) for spec in TARGET_SPECS]
