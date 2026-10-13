"""Frozen contracts and configuration helpers for AgiBot arm reach rate."""

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
import yaml

from src.time_jepa.agibot_derived_targets import (
    END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS,
)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for _path in (REPO_ROOT, PROBE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from world_model_probe_rjob.physics_consistency_encoder import (
    WORLD_MODEL_BACKENDS,
    validate_world_model_encoder_config,
)

STANDARD_TASKS = (
    3400,
    3401,
    3402,
    3404,
    3405,
    3477,
    3641,
    3705,
    3777,
    4053,
)
TRAIN_VIEWS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
OOD_VIEWS = (
    "observation.images.head_left_fisheye",
    "observation.images.head_right_fisheye",
)
ALL_EVAL_VIEWS = TRAIN_VIEWS + OOD_VIEWS


PROTOCOL = "agibot_arm_reach_rate_current_v1"
TARGET_DEFINITION = "agibot_q14_endpoint_signed_velocity6_v1"
ARMS = ("left", "right")
AXES = ("x", "y", "z")

TARGET_SPECS = tuple(
    [
        {
            "name": f"q_left_{index}",
            "field": f"state/joint/position[{index}]",
            "family": "q",
            "arm": "left",
            "component": index,
            "encoder_supervised": False,
            "role": "physics_q_shot",
            "unit": "rad",
        }
        for index in range(7)
    ]
    + [
        {
            "name": f"q_right_{index}",
            "field": f"state/joint/position[{index + 7}]",
            "family": "q",
            "arm": "right",
            "component": index,
            "encoder_supervised": False,
            "role": "physics_q_shot",
            "unit": "rad",
        }
        for index in range(7)
    ]
    + [
        {
            "name": f"v_left_{axis}",
            "field": END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS[index],
            "family": "velocity",
            "arm": "left",
            "component": index,
            "axis": axis,
            "encoder_supervised": True,
            "role": "physics_readout",
            "unit": "m/s",
        }
        for index, axis in enumerate(AXES)
    ]
    + [
        {
            "name": f"v_right_{axis}",
            "field": END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS[index + 3],
            "family": "velocity",
            "arm": "right",
            "component": index,
            "axis": axis,
            "encoder_supervised": True,
            "role": "physics_readout",
            "unit": "m/s",
        }
        for index, axis in enumerate(AXES)
    ]
)
TARGET_NAMES = tuple(item["name"] for item in TARGET_SPECS)
TARGET_FIELDS = tuple(item["field"] for item in TARGET_SPECS)
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}
Q_INDICES = {
    arm: np.asarray(
        [TARGET_INDEX[f"q_{arm}_{index}"] for index in range(7)],
        dtype=np.int64,
    )
    for arm in ARMS
}
V_INDICES = {
    arm: np.asarray(
        [TARGET_INDEX[f"v_{arm}_{axis}"] for axis in AXES],
        dtype=np.int64,
    )
    for arm in ARMS
}
P_SLICES = {"left": slice(0, 3), "right": slice(3, 6)}


def target_manifest() -> list[dict]:
    return [dict(item) for item in TARGET_SPECS]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
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
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    # Avoid importing the multi-GB torch package in standalone evaluators.
    if (
        value.__class__.__module__.startswith("torch")
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
    ):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def deep_update(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path, chain: tuple[Path, ...] = ()) -> dict:
    path = Path(path).expanduser().resolve()
    if path in chain:
        raise ValueError(f"Cyclic YAML base chain: {[str(item) for item in (*chain, path)]}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    base = config.pop("base", None)
    if base is not None:
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        config = deep_update(load_yaml(base_path, (*chain, path)), config)
    config["config_path"] = str(path)
    return config


def validate_config(config: dict) -> None:
    experiment = config.get("experiment", {})
    phase = experiment.get("phase")
    if phase not in {"full", "smoke"}:
        raise ValueError("experiment.phase must be full or smoke")
    if experiment.get("protocol_name") != PROTOCOL:
        raise ValueError(f"experiment.protocol_name must be {PROTOCOL!r}")
    if experiment.get("reuse_heads_from"):
        raise ValueError("Head reuse is forbidden for reach-rate experiments")

    data = config.get("data", {})
    if str(data.get("train_split")) != "probe_train":
        raise ValueError("data.train_split is frozen to probe_train")
    if str(data.get("val_split")) != "probe_val":
        raise ValueError("data.val_split is frozen to probe_val")
    evaluation_mode = config.get("evaluation", {}).get("mode")
    if phase == "full" and evaluation_mode != "test":
        raise ValueError("Full protocol requires evaluation.mode=test")
    if phase == "smoke" and evaluation_mode not in {"smoke", "val"}:
        raise ValueError("Smoke must not read final test views")

    sampling = config.get("sampling", {})
    num_frames = int(sampling.get("num_frames", 0))
    tubelet = int(sampling.get("tubelet_size", 0))
    if num_frames <= 0 or tubelet <= 0 or num_frames % tubelet:
        raise ValueError("sampling.num_frames must be positive and divisible by tubelet_size")
    if phase == "full":
        if num_frames != 64 or tubelet != 2 or float(sampling.get("fps", 0.0)) != 5.0:
            raise ValueError("Formal protocol is frozen to 64 frames, 5 fps, tubelet 2")
        if tuple(int(item) for item in data.get("tasks", ())) != STANDARD_TASKS:
            raise ValueError(f"Formal AgiBot tasks must be {STANDARD_TASKS}")
        if tuple(data.get("train_views", ())) != TRAIN_VIEWS:
            raise ValueError(f"Formal train views must be {TRAIN_VIEWS}")
        if tuple(data.get("id_views", ())) != TRAIN_VIEWS:
            raise ValueError(f"Formal ID views must be {TRAIN_VIEWS}")
        if tuple(data.get("ood_views", ())) != OOD_VIEWS:
            raise ValueError(f"Formal OOD views must be {OOD_VIEWS}")
        backend = config.get("encoder", {}).get("backend")
        if backend not in {"vjepa2_1", *WORLD_MODEL_BACKENDS}:
            raise ValueError(
                "Formal runs require vjepa2_1 or a registered world-model backend"
            )
        if config.get("probe", {}).get("type") != "independent_linear":
            raise ValueError("Formal protocol requires independent_linear heads")

    encoder = config.get("encoder", {})
    if encoder.get("backend") == "vjepa2_1" and not encoder.get("checkpoint"):
        raise ValueError("encoder.checkpoint is required")
    if encoder.get("backend") in WORLD_MODEL_BACKENDS:
        validate_world_model_encoder_config(config)
    kinematics = config.get("kinematics", {})
    if not kinematics.get("urdf"):
        raise ValueError("kinematics.urdf is required")
    if not kinematics.get("gt_gate_metrics"):
        raise ValueError("kinematics.gt_gate_metrics is required")
    reach_rate = config.get("reach_rate", {})
    if reach_rate.get("primary_metric") != "active_formula_mae":
        raise ValueError("reach_rate.primary_metric must be active_formula_mae")
    if int(reach_rate.get("time_shift_tokens", 0)) <= 0:
        raise ValueError("reach_rate.time_shift_tokens must be positive")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    config = deep_update(load_yaml(path), overrides or {})
    validate_config(config)
    return config


def config_fingerprint(config: dict) -> str:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "target_definition": TARGET_DEFINITION,
        "targets": target_manifest(),
        "tasks": list(config["data"]["tasks"]),
        "split_prefix": config["data"]["split_prefix"],
        "sampling": config["sampling"],
        "primary_metric": config["reach_rate"]["primary_metric"],
        "time_shift_tokens": int(config["reach_rate"]["time_shift_tokens"]),
        "urdf": str(config["kinematics"]["urdf"]),
    }
    return canonical_hash(payload)


def shoulder_points_from_gate(gate: dict) -> dict[str, np.ndarray]:
    points = gate.get("shoulder_points_dataset")
    if not isinstance(points, dict):
        raise ValueError("GT Gate is missing shoulder_points_dataset")
    output = {}
    for arm in ARMS:
        value = np.asarray(points.get(arm), dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid {arm} shoulder point in GT Gate: {value}")
        output[arm] = value
    return output


__all__ = [
    "ALL_EVAL_VIEWS",
    "ARMS",
    "AXES",
    "OOD_VIEWS",
    "P_SLICES",
    "PROTOCOL",
    "Q_INDICES",
    "STANDARD_TASKS",
    "TARGET_DEFINITION",
    "TARGET_FIELDS",
    "TARGET_INDEX",
    "TARGET_NAMES",
    "TARGET_SPECS",
    "TRAIN_VIEWS",
    "V_INDICES",
    "atomic_json_dump",
    "canonical_hash",
    "config_fingerprint",
    "deep_update",
    "json_safe",
    "load_config",
    "load_yaml",
    "seed_everything",
    "sha256_file",
    "shoulder_points_from_gate",
    "target_manifest",
]
