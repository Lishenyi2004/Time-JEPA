"""Shared derived-target definitions for AgiBot encoder and probe datasets.

The versioned whole-body panels cover base motion, torso configuration, both
end effectors, and both grippers.  ``wholebody_cartesian_v2`` replaces the two
unsigned end-speed magnitudes in v1 with six signed Cartesian velocities
derived from the matching endpoint-position stream.  Series-derived fields
live here so encoder supervision and frozen-probe evaluation use byte-for-byte
identical target construction.
"""

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


AGIBOT_TARGET_DEFINITION_VERSION = "wholebody_cartesian_v1_2026-07-22"
AGIBOT_TARGET_DEFINITION_VERSION_V2 = "wholebody_cartesian_v2_2026-07-27"

BASE_TRANSLATION_SPEED = "state/robot/translation_speed"
BASE_ANGULAR_SPEED = "state/robot/angular_speed"
WAIST_PC1 = "state/waist/position_pc1"
WAIST_PC2 = "state/waist/position_pc2"
END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS = tuple(
    f"state/end/arm_position_velocity_world[{index}]" for index in range(6)
)

WHOLEBODY_CARTESIAN_V1_FIELDS = [
    BASE_TRANSLATION_SPEED,
    BASE_ANGULAR_SPEED,
    WAIST_PC1,
    WAIST_PC2,
    *[f"state/end/arm_position[{i}]" for i in range(6)],
    "state/end/velocity_linear_norm_left",
    "state/end/velocity_linear_norm_right",
    "state/left_effector/position",
    "state/right_effector/position",
]

WHOLEBODY_CARTESIAN_V2_FIELDS = [
    BASE_TRANSLATION_SPEED,
    BASE_ANGULAR_SPEED,
    WAIST_PC1,
    WAIST_PC2,
    *[f"state/end/arm_position[{i}]" for i in range(6)],
    *END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS,
    "state/left_effector/position",
    "state/right_effector/position",
]

JOINT_CONFIGURATION_FULL_V1_FIELDS = [
    f"state/joint/position[{i}]" for i in range(14)
]

SERIES_DERIVED_FIELDS = frozenset(
    (
        BASE_TRANSLATION_SPEED,
        BASE_ANGULAR_SPEED,
        WAIST_PC1,
        WAIST_PC2,
        *END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS,
    )
)

# Flat observation.state layout shared by the 162-D and 169-D shards.
END_ARM_POSITION_SLICE = slice(34, 40)
WAIST_SLICE = slice(85, 90)
ROBOT_POSITION_SLICE = slice(90, 93)
ROBOT_ORIENTATION_SLICE = slice(93, 97)
SERIES_DERIVED_MIN_STATE_LEN = 97

# Conservative gates for the G2 base pose stream.  Values outside these gates
# are treated as unavailable, not clipped into plausible-looking supervision.
BASE_MAX_TRANSLATION_SPEED_MPS = 2.0
BASE_MAX_ANGULAR_SPEED_RADPS = 4.0
END_EFFECTOR_MAX_LINEAR_SPEED_MPS = 2.0
QUATERNION_NORM_MIN = 0.9
QUATERNION_NORM_MAX = 1.1

DEFAULT_WAIST_PCA_PATH = (
    Path(__file__).resolve().parents[2]
    / "time-jepa"
    / "cache"
    / "agibot_wholebody_cartesian_v1_waist_pca.json"
)


def is_agibot_series_derived_field(field):
    return str(field) in SERIES_DERIVED_FIELDS


def agibot_target_definition_version(fields):
    """Return a cache version without invalidating historical v1 statistics."""

    if any(str(field) in END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS for field in fields):
        return AGIBOT_TARGET_DEFINITION_VERSION_V2
    return AGIBOT_TARGET_DEFINITION_VERSION


@lru_cache(maxsize=4)
def load_waist_pca_codebook(path=None):
    """Load the encoder-train-only, task-balanced waist PCA codebook."""

    configured_path = path or os.environ.get("AGIBOT_WAIST_PCA_PATH")
    codebook_path = Path(configured_path).expanduser() if configured_path else DEFAULT_WAIST_PCA_PATH
    if not codebook_path.is_file():
        raise FileNotFoundError(
            "AgiBot whole-body targets require the waist PCA codebook; "
            f"missing: {codebook_path}. Run time-jepa/fit_agibot_waist_pca.py."
        )
    with codebook_path.open("r") as handle:
        payload = json.load(handle)
    mean = np.asarray(payload.get("mean"), dtype=np.float64)
    components = np.asarray(payload.get("components"), dtype=np.float64)
    if mean.shape != (5,) or components.shape != (2, 5):
        raise RuntimeError(
            f"Invalid AgiBot waist PCA codebook shapes in {codebook_path}: "
            f"mean={mean.shape}, components={components.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise RuntimeError(f"Non-finite AgiBot waist PCA codebook: {codebook_path}")
    return mean, components, payload


def _central_neighbor_indices(num_rows):
    indices = np.arange(num_rows, dtype=np.int64)
    previous = np.maximum(indices - 1, 0)
    following = np.minimum(indices + 1, num_rows - 1)
    return previous, following


def _central_time_delta(timestamps):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    previous, following = _central_neighbor_indices(len(timestamps))
    delta_t = timestamps[following] - timestamps[previous]
    return previous, following, delta_t


def _base_translation_speed(timestamps, states):
    previous, following, delta_t = _central_time_delta(timestamps)
    position = np.asarray(states[:, ROBOT_POSITION_SLICE], dtype=np.float64)
    delta_xy = position[following, :2] - position[previous, :2]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.linalg.norm(delta_xy, axis=1) / delta_t
    valid = (
        np.isfinite(position[previous]).all(axis=1)
        & np.isfinite(position[following]).all(axis=1)
        & np.isfinite(values)
        & (delta_t > 0.0)
        & (values <= BASE_MAX_TRANSLATION_SPEED_MPS)
    )
    return np.where(valid, values, 0.0).astype(np.float32), valid.astype(np.float32)


def _base_angular_speed(timestamps, states):
    previous, following, delta_t = _central_time_delta(timestamps)
    quaternion = np.asarray(states[:, ROBOT_ORIENTATION_SLICE], dtype=np.float64)
    norms = np.linalg.norm(quaternion, axis=1)
    quaternion_valid = (
        np.isfinite(quaternion).all(axis=1)
        & (norms >= QUATERNION_NORM_MIN)
        & (norms <= QUATERNION_NORM_MAX)
    )
    safe_norms = np.where(quaternion_valid, norms, 1.0)
    unit_quaternion = quaternion / safe_norms[:, None]
    dots = np.sum(unit_quaternion[previous] * unit_quaternion[following], axis=1)
    # abs(dot) makes the geodesic rate invariant to the q <-> -q ambiguity.
    relative_angle = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = relative_angle / delta_t
    valid = (
        quaternion_valid[previous]
        & quaternion_valid[following]
        & np.isfinite(values)
        & (delta_t > 0.0)
        & (values <= BASE_MAX_ANGULAR_SPEED_RADPS)
    )
    return np.where(valid, values, 0.0).astype(np.float32), valid.astype(np.float32)


def _waist_principal_component(states, component_index):
    waist = np.asarray(states[:, WAIST_SLICE], dtype=np.float64)
    mean, components, _ = load_waist_pca_codebook()
    values = (waist - mean[None, :]) @ components[int(component_index)]
    valid = np.isfinite(waist).all(axis=1) & np.isfinite(values)
    return np.where(valid, values, 0.0).astype(np.float32), valid.astype(np.float32)


def _end_effector_world_linear_velocity(timestamps, states, component_index):
    """Differentiate the endpoint position in its recorded base/world frame."""

    component_index = int(component_index)
    if component_index < 0 or component_index >= 6:
        raise ValueError(f"Endpoint velocity component must be in [0, 6), got {component_index}")

    previous, following, delta_t = _central_time_delta(timestamps)
    position = np.asarray(states[:, END_ARM_POSITION_SLICE], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = (position[following] - position[previous]) / delta_t[:, None]

    arm_start = 0 if component_index < 3 else 3
    arm_end = arm_start + 3
    arm_speed = np.linalg.norm(velocity[:, arm_start:arm_end], axis=1)
    valid = (
        np.isfinite(position[previous, arm_start:arm_end]).all(axis=1)
        & np.isfinite(position[following, arm_start:arm_end]).all(axis=1)
        & np.isfinite(velocity[:, arm_start:arm_end]).all(axis=1)
        & (delta_t > 0.0)
        & (arm_speed <= END_EFFECTOR_MAX_LINEAR_SPEED_MPS)
    )
    values = velocity[:, component_index]
    return np.where(valid, values, 0.0).astype(np.float32), valid.astype(np.float32)


def derive_agibot_series_field(timestamps, states, field):
    """Return one derived per-frame field and its float validity mask."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"Expected aligned [N,D] AgiBot states/timestamps, got {states.shape} and {timestamps.shape}"
        )
    if states.shape[1] < SERIES_DERIVED_MIN_STATE_LEN:
        raise ValueError(
            f"AgiBot derived targets require state dim >= {SERIES_DERIVED_MIN_STATE_LEN}, "
            f"got {states.shape[1]}"
        )
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=np.float32), np.zeros(len(timestamps), dtype=np.float32)
    if field == BASE_TRANSLATION_SPEED:
        return _base_translation_speed(timestamps, states)
    if field == BASE_ANGULAR_SPEED:
        return _base_angular_speed(timestamps, states)
    if field == WAIST_PC1:
        return _waist_principal_component(states, 0)
    if field == WAIST_PC2:
        return _waist_principal_component(states, 1)
    if field in END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS:
        component_index = END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS.index(field)
        return _end_effector_world_linear_velocity(timestamps, states, component_index)
    raise ValueError(f"Unsupported AgiBot series-derived field: {field}")


def agibot_derived_display_name(field):
    scalar_names = {
        BASE_TRANSLATION_SPEED: "base_translation_speed",
        BASE_ANGULAR_SPEED: "base_angular_speed",
        WAIST_PC1: "waist_pc1",
        WAIST_PC2: "waist_pc2",
    }
    if field in scalar_names:
        return scalar_names[field]
    if field in END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS:
        index = END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS.index(field)
        arm = "left" if index < 3 else "right"
        axis = ("x", "y", "z")[index % 3]
        return f"end_velocity_world_{arm}_{axis}"
    return field
