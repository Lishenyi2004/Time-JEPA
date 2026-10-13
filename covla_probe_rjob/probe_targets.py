"""Target-field normalization and per-frame state/OXTS series loading."""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from probe_constants import DERIVED_COVLA_TARGETS, KITTI_OXTS_FIELDS, KITTI_TARGET_ALIASES, TARGET_ALIASES
from src.time_jepa.agibot_derived_targets import (
    JOINT_CONFIGURATION_FULL_V1_FIELDS,
    SERIES_DERIVED_MIN_STATE_LEN,
    WHOLEBODY_CARTESIAN_V1_FIELDS,
    WHOLEBODY_CARTESIAN_V2_FIELDS,
    agibot_derived_display_name,
    derive_agibot_series_field,
    is_agibot_series_derived_field,
)
from src.time_jepa.egoexo_pose_targets import EGOEXO_TARGET_GROUPS, normalize_egoexo_targets


def normalize_target_field(field):
    field = field.strip()
    return TARGET_ALIASES.get(field, field)


def normalize_kitti_target_field(field):
    field = field.strip()
    return KITTI_TARGET_ALIASES.get(field, field)


def target_display_name(field):
    if field in DERIVED_COVLA_TARGETS:
        return DERIVED_COVLA_TARGETS[field].get("display_name", field)
    for alias, target in TARGET_ALIASES.items():
        if field == target and alias == "yaw_rate":
            return alias
    return field


def kitti_target_display_name(field):
    for alias, target in KITTI_TARGET_ALIASES.items():
        if field == target and alias in ("vEgo", "aEgo", "yaw_rate"):
            return alias
    return field


AGIBOT_STATE_SLICES = {
    "state/left_effector/position": (0, 1),
    "state/right_effector/position": (1, 2),
    "state/end/wrench": (2, 14),
    "state/end/velocity": (14, 26),
    "state/end/arm_position": (34, 40),
    "state/joint/position": (40, 54),
    "state/joint/effort": (54, 68),
    "state/joint/velocity": (68, 82),
    "state/waist/position": (85, 90),
    "state/robot/position": (90, 93),
    "state/robot/orientation": (93, 97),
}

AGIBOT_STATE_MIN_LEN = max(end for _, end in AGIBOT_STATE_SLICES.values())


def normalize_egoexo_target_fields(fields):
    return normalize_egoexo_targets(fields)


def egoexo_target_display_name(field):
    return str(field)


AGIBOT_TARGET_GROUPS = {
    "joint_position_top4": [f"state/joint/position[{i}]" for i in range(4)],
    "joint_velocity_top4": [f"state/joint/velocity[{i}]" for i in range(4)],
    "joint_effector_v1": (
        [f"state/joint/position[{i}]" for i in range(4)]
        + [f"state/joint/velocity[{i}]" for i in range(4)]
        + [
            "state/end/velocity_norm",
            "state/left_effector/position",
            "state/right_effector/position",
        ]
    ),
    "bimanual_compact_v1": [
        "state/joint/velocity_norm_left",
        "state/joint/velocity_norm_right",
        "state/end/velocity_linear_norm_left",
        "state/end/velocity_linear_norm_right",
        "state/end/arm_position[1]",
        "state/end/arm_position[4]",
        "state/left_effector/position",
        "state/right_effector/position",
    ],
    "unseen_phys_v1": [
        "state/joint/effort_norm_left",
        "state/joint/effort_norm_right",
        "state/end/wrench_force_norm_left",
        "state/end/wrench_force_norm_right",
        "state/waist/position[4]",
        "state/waist/position[2]",
    ],
    # Left-arm supervision family: readout reads the 6 supervised LEFT-arm dims;
    # shot reads the dual 6 RIGHT-arm dims (held out from encoder training).
    # Keep the two groups dim-for-dim aligned so left/right transfer is a clean
    # mirror comparison.
    "left_arm_seen_v1": [
        "state/joint/velocity_norm_left",
        "state/end/velocity_linear_norm_left",
        "state/end/arm_position[1]",
        "state/left_effector/position",
        "state/joint/effort_norm_left",
        "state/end/wrench_force_norm_left",
    ],
    "left_arm_unseen_right_v1": [
        "state/joint/velocity_norm_right",
        "state/end/velocity_linear_norm_right",
        "state/end/arm_position[4]",
        "state/right_effector/position",
        "state/joint/effort_norm_right",
        "state/end/wrench_force_norm_right",
    ],
    # No-wrench v2 ablation. These groups remain dim-for-dim mirrored; v1 is
    # intentionally retained for reproducing the original 6-D experiment.
    "left_arm_seen_v2": [
        "state/joint/velocity_norm_left",
        "state/end/velocity_linear_norm_left",
        "state/end/arm_position[1]",
        "state/left_effector/position",
        "state/joint/effort_norm_left",
    ],
    "left_arm_unseen_right_v2": [
        "state/joint/velocity_norm_right",
        "state/end/velocity_linear_norm_right",
        "state/end/arm_position[4]",
        "state/right_effector/position",
        "state/joint/effort_norm_right",
    ],
    # Signed joint components for AgiBot physics-consistency probes.  These are
    # intentionally split by arm: each probe learns only one 7-DoF arm's q/qdot
    # pair, and the held-out physics eval uses the same arm only.
    "agibot_phys_signed_left_v1": (
        [f"state/joint/position[{i}]" for i in range(0, 7)]
        + [f"state/joint/velocity[{i}]" for i in range(0, 7)]
    ),
    "agibot_phys_signed_right_v1": (
        [f"state/joint/position[{i}]" for i in range(7, 14)]
        + [f"state/joint/velocity[{i}]" for i in range(7, 14)]
    ),
    # Readout is exactly the panel used to supervise the encoder.
    "wholebody_cartesian_v1": list(WHOLEBODY_CARTESIAN_V1_FIELDS),
    # New readout is exactly the 18-D v2 encoder-supervision panel.
    "wholebody_cartesian_v2": list(WHOLEBODY_CARTESIAN_V2_FIELDS),
    # Shot deliberately asks for signed internal joint configuration, which is
    # absent from the Cartesian encoder target but remains 14-D for parity.
    "joint_configuration_full_v1": list(JOINT_CONFIGURATION_FULL_V1_FIELDS),
    # Ordinary probe shot for the v2 encoder experiment. This is intentionally
    # distinct from the 14-D q-shot used by the physics formula: it stays at
    # eight broad, bilateral summaries related to the Cartesian readout while
    # remaining absent from encoder supervision.
    "wholebody_related_shot_v2": [
        "state/joint/position_norm_left",
        "state/joint/position_norm_right",
        "state/joint/velocity_norm_left",
        "state/joint/velocity_norm_right",
        "state/joint/effort_norm_left",
        "state/joint/effort_norm_right",
        "state/end/wrench_force_norm_left",
        "state/end/wrench_force_norm_right",
    ],
}

AGIBOT_TARGET_ALIASES = {
    "end_velocity_norm": "state/end/velocity_norm",
    "left_effector_scalar": "state/left_effector/position",
    "right_effector_scalar": "state/right_effector/position",
    "joint_position_norm_left": "state/joint/position_norm_left",
    "joint_position_norm_right": "state/joint/position_norm_right",
    "joint_velocity_norm_left": "state/joint/velocity_norm_left",
    "joint_velocity_norm_right": "state/joint/velocity_norm_right",
    "end_velocity_linear_norm_left": "state/end/velocity_linear_norm_left",
    "end_velocity_linear_norm_right": "state/end/velocity_linear_norm_right",
    "arm_position_left": "state/end/arm_position[1]",
    "arm_position_right": "state/end/arm_position[4]",
    "joint_effort_norm_left": "state/joint/effort_norm_left",
    "joint_effort_norm_right": "state/joint/effort_norm_right",
    "end_wrench_force_norm_left": "state/end/wrench_force_norm_left",
    "end_wrench_force_norm_right": "state/end/wrench_force_norm_right",
    "waist_position_4": "state/waist/position[4]",
    "waist_position_2": "state/waist/position[2]",
    "base_translation_speed": "state/robot/translation_speed",
    "base_angular_speed": "state/robot/angular_speed",
    "waist_pc1": "state/waist/position_pc1",
    "waist_pc2": "state/waist/position_pc2",
}


def normalize_agibot_target_field(field):
    field = field.strip()
    return AGIBOT_TARGET_ALIASES.get(field, field)


def normalize_agibot_target_fields(fields):
    normalized = []
    for field in fields:
        field = field.strip()
        if not field:
            continue
        if field in AGIBOT_TARGET_GROUPS:
            normalized.extend(AGIBOT_TARGET_GROUPS[field])
        else:
            normalized.append(normalize_agibot_target_field(field))
    return normalized


def agibot_target_display_name(field):
    if is_agibot_series_derived_field(field):
        return agibot_derived_display_name(field)

    scalar_names = {
        "state/end/velocity_norm": "end_velocity_norm",
        "state/left_effector/position": "left_effector_scalar",
        "state/right_effector/position": "right_effector_scalar",
        "state/joint/position_norm_left": "joint_position_norm_left",
        "state/joint/position_norm_right": "joint_position_norm_right",
        "state/joint/velocity_norm_left": "joint_velocity_norm_left",
        "state/joint/velocity_norm_right": "joint_velocity_norm_right",
        "state/end/velocity_linear_norm_left": "end_velocity_linear_norm_left",
        "state/end/velocity_linear_norm_right": "end_velocity_linear_norm_right",
        "state/end/arm_position[1]": "arm_position_left",
        "state/end/arm_position[4]": "arm_position_right",
        "state/joint/effort_norm_left": "joint_effort_norm_left",
        "state/joint/effort_norm_right": "joint_effort_norm_right",
        "state/end/wrench_force_norm_left": "end_wrench_force_norm_left",
        "state/end/wrench_force_norm_right": "end_wrench_force_norm_right",
        "state/waist/position[4]": "waist_position_4",
        "state/waist/position[2]": "waist_position_2",
    }
    if field in scalar_names:
        return scalar_names[field]

    indexed_names = {
        "state/joint/position": "joint_position",
        "state/joint/velocity": "joint_velocity",
        "state/joint/effort": "joint_effort",
        "state/end/arm_position": "arm_position",
        "state/end/wrench": "end_wrench",
        "state/waist/position": "waist_position",
    }
    for base, label in indexed_names.items():
        if field.startswith(f"{base}[") and field.endswith("]"):
            return f"{label}_{field[field.rfind('[') + 1:-1]}"
    return field


def read_payload_field(payload, field):
    field = normalize_target_field(field)
    if field in DERIVED_COVLA_TARGETS:
        return read_derived_payload_field(payload, field)
    if "[" in field and field.endswith("]"):
        base, index_text = field[:-1].split("[", 1)
        value = payload.get(base)
        try:
            index = int(index_text)
        except ValueError:
            return 0.0, 0.0
        is_valid = (
            isinstance(value, (list, tuple))
            and -len(value) <= index < len(value)
            and isinstance(value[index], (int, float))
            and math.isfinite(value[index])
        )
        return (float(value[index]) if is_valid else 0.0), (1.0 if is_valid else 0.0)

    value = payload.get(field)
    is_valid = value is not None and isinstance(value, (int, float)) and math.isfinite(value)
    return (float(value) if is_valid else 0.0), (1.0 if is_valid else 0.0)


def read_derived_payload_field(payload, field):
    spec = DERIVED_COVLA_TARGETS[field]
    if spec.get("source") != "trajectory":
        return 0.0, 0.0

    trajectory = payload.get("trajectory")
    point_index = int(spec["point_index"])
    axis_index = int(spec["axis_index"])
    try:
        point = trajectory[point_index]
        value = point[axis_index]
    except (IndexError, TypeError):
        return 0.0, 0.0

    is_valid = isinstance(value, (int, float)) and math.isfinite(value)
    return (float(value) if is_valid else 0.0), (1.0 if is_valid else 0.0)


def read_trajectory_xy(payload, point_indices=(-1,)):
    trajectory = payload.get("trajectory")
    num_outputs = 2 * len(tuple(point_indices))
    if not isinstance(trajectory, (list, tuple)) or not trajectory:
        return [0.0] * num_outputs, [0.0] * num_outputs

    values, valid = [], []
    for point_index in point_indices:
        try:
            point = trajectory[point_index]
        except (IndexError, TypeError):
            values.extend([0.0, 0.0])
            valid.extend([0.0, 0.0])
            continue
        is_valid = (
            isinstance(point, (list, tuple))
            and len(point) >= 2
            and isinstance(point[0], (int, float))
            and isinstance(point[1], (int, float))
            and math.isfinite(point[0])
            and math.isfinite(point[1])
        )
        if not is_valid:
            values.extend([0.0, 0.0])
            valid.extend([0.0, 0.0])
            continue
        values.extend([float(point[0]), float(point[1])])
        valid.extend([1.0, 1.0])
    return values, valid


def read_kitti_payload_field(payload, field):
    field = normalize_kitti_target_field(field)
    value = payload.get(field)
    is_valid = value is not None and isinstance(value, (int, float)) and math.isfinite(value)
    return (float(value) if is_valid else 0.0), (1.0 if is_valid else 0.0)


def _parse_indexed_field(field):
    if "[" not in field or not field.endswith("]"):
        return field, None
    base, index_text = field[:-1].split("[", 1)
    return base, int(index_text)


def read_agibot_state_field(state, field):
    field = normalize_agibot_target_field(field)
    if is_agibot_series_derived_field(field):
        raise ValueError(
            f"AgiBot field {field!r} is sequence-derived; use load_agibot_state_series"
        )
    if field == "state/end/velocity_norm":
        start, end = AGIBOT_STATE_SLICES["state/end/velocity"]
        value = float(np.linalg.norm(state[start:end]))
        return value, 1.0 if math.isfinite(value) else 0.0

    if field == "state/joint/position_norm_left":
        start, end = AGIBOT_STATE_SLICES["state/joint/position"]
        value = float(np.linalg.norm(state[start : start + 7]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/joint/position_norm_right":
        start, end = AGIBOT_STATE_SLICES["state/joint/position"]
        value = float(np.linalg.norm(state[start + 7 : end]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/joint/velocity_norm_left":
        start, end = AGIBOT_STATE_SLICES["state/joint/velocity"]
        value = float(np.linalg.norm(state[start : start + 7]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/joint/velocity_norm_right":
        start, end = AGIBOT_STATE_SLICES["state/joint/velocity"]
        value = float(np.linalg.norm(state[start + 7 : end]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/end/velocity_linear_norm_left":
        start, end = AGIBOT_STATE_SLICES["state/end/velocity"]
        value = float(np.linalg.norm(state[start : start + 3]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/end/velocity_linear_norm_right":
        start, end = AGIBOT_STATE_SLICES["state/end/velocity"]
        value = float(np.linalg.norm(state[start + 6 : start + 9]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/joint/effort_norm_left":
        start, end = AGIBOT_STATE_SLICES["state/joint/effort"]
        value = float(np.linalg.norm(state[start : start + 7]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/joint/effort_norm_right":
        start, end = AGIBOT_STATE_SLICES["state/joint/effort"]
        value = float(np.linalg.norm(state[start + 7 : end]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/end/wrench_force_norm_left":
        start, end = AGIBOT_STATE_SLICES["state/end/wrench"]
        value = float(np.linalg.norm(state[start : start + 3]))
        return value, 1.0 if math.isfinite(value) else 0.0
    if field == "state/end/wrench_force_norm_right":
        start, end = AGIBOT_STATE_SLICES["state/end/wrench"]
        value = float(np.linalg.norm(state[start + 6 : start + 9]))
        return value, 1.0 if math.isfinite(value) else 0.0

    base, index = _parse_indexed_field(field)
    if base not in AGIBOT_STATE_SLICES:
        raise ValueError(f"Unsupported AgiBot target field: {field}")
    start, end = AGIBOT_STATE_SLICES[base]
    values = state[start:end]
    if index is None:
        if values.size != 1:
            raise ValueError(f"AgiBot field {field!r} is vector-valued; use an indexed field")
        value = float(values[0])
    else:
        if index < 0 or index >= (end - start):
            raise ValueError(f"AgiBot field index out of range: {field}")
        value = float(state[start + index])
    return value, 1.0 if math.isfinite(value) else 0.0


@lru_cache(maxsize=512)
def load_state_series(states_path, target_fields_tuple):
    target_fields = [normalize_target_field(f) for f in target_fields_tuple]
    frame_ids, values, valid = [], [], []
    with open(states_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            frame_id_str, payload = next(iter(obj.items()))
            frame_ids.append(int(frame_id_str))
            row_values, row_valid = [], []
            for field in target_fields:
                value, is_valid = read_payload_field(payload, field)
                row_values.append(value)
                row_valid.append(is_valid)
            values.append(row_values)
            valid.append(row_valid)
    if not values:
        raise RuntimeError(f"Empty states file: {states_path}")
    frame_ids = np.array(frame_ids, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    valid = np.array(valid, dtype=np.float32)
    order = np.argsort(frame_ids)
    return frame_ids[order], values[order], valid[order]


@lru_cache(maxsize=512)
def load_trajectory_xy_series(states_path, point_indices_tuple=(-1,)):
    point_indices = tuple(int(index) for index in point_indices_tuple)
    frame_ids, values, valid = [], [], []
    with open(states_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            frame_id_str, payload = next(iter(obj.items()))
            frame_ids.append(int(frame_id_str))
            row_values, row_valid = read_trajectory_xy(payload, point_indices=point_indices)
            values.append(row_values)
            valid.append(row_valid)
    if not values:
        raise RuntimeError(f"Empty states file: {states_path}")
    frame_ids = np.array(frame_ids, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    valid = np.array(valid, dtype=np.float32)
    order = np.argsort(frame_ids)
    return frame_ids[order], values[order], valid[order]


@lru_cache(maxsize=256)
def load_kitti_oxts_series(oxts_dir, target_fields_tuple):
    oxts_dir = Path(oxts_dir)
    target_fields = [normalize_kitti_target_field(f) for f in target_fields_tuple]
    frame_ids, values, valid = [], [], []
    files = sorted(oxts_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"Empty KITTI OXTS directory: {oxts_dir}")
    for path in files:
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        parts = path.read_text().strip().split()
        payload = {}
        for name, text in zip(KITTI_OXTS_FIELDS, parts):
            try:
                payload[name] = float(text)
            except ValueError:
                payload[name] = None
        frame_ids.append(frame_id)
        row_values, row_valid = [], []
        for field in target_fields:
            value, is_valid = read_kitti_payload_field(payload, field)
            row_values.append(value)
            row_valid.append(is_valid)
        values.append(row_values)
        valid.append(row_valid)
    if not values:
        raise RuntimeError(f"No usable KITTI OXTS files under: {oxts_dir}")
    frame_ids = np.array(frame_ids, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    valid = np.array(valid, dtype=np.float32)
    order = np.argsort(frame_ids)
    return frame_ids[order], values[order], valid[order]


@lru_cache(maxsize=256)
def load_agibot_state_series(parquet_path, target_fields_tuple):
    import pandas as pd

    target_fields = tuple(normalize_agibot_target_fields(target_fields_tuple))
    df = pd.read_parquet(parquet_path, columns=["timestamp", "observation.state"])
    if len(df) == 0:
        raise RuntimeError(f"Empty AgiBot parquet: {parquet_path}")

    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    states = np.stack(
        [np.asarray(value, dtype=np.float32) for value in df["observation.state"]],
        axis=0,
    )
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    states = states[order]
    required_state_len = 0
    for field in target_fields:
        base, index = _parse_indexed_field(field)
        if is_agibot_series_derived_field(field):
            required_state_len = max(required_state_len, SERIES_DERIVED_MIN_STATE_LEN)
        elif field == "state/end/velocity_norm":
            required_state_len = max(required_state_len, AGIBOT_STATE_SLICES["state/end/velocity"][1])
        elif field in ("state/joint/position_norm_left", "state/joint/position_norm_right"):
            required_state_len = max(required_state_len, AGIBOT_STATE_SLICES["state/joint/position"][1])
        elif field in ("state/joint/velocity_norm_left", "state/joint/velocity_norm_right"):
            required_state_len = max(required_state_len, AGIBOT_STATE_SLICES["state/joint/velocity"][1])
        elif field in ("state/end/velocity_linear_norm_left", "state/end/velocity_linear_norm_right"):
            required_state_len = max(required_state_len, AGIBOT_STATE_SLICES["state/end/velocity"][1])
        elif field in ("state/joint/effort_norm_left", "state/joint/effort_norm_right"):
            required_state_len = max(required_state_len, AGIBOT_STATE_SLICES["state/joint/effort"][1])
        elif field in ("state/end/wrench_force_norm_left", "state/end/wrench_force_norm_right"):
            required_state_len = max(required_state_len, AGIBOT_STATE_SLICES["state/end/wrench"][1])
        elif base in AGIBOT_STATE_SLICES:
            start, end = AGIBOT_STATE_SLICES[base]
            required_state_len = max(required_state_len, end if index is None else start + index + 1)
    if states.shape[-1] < required_state_len:
        raise RuntimeError(
            f"Expected AgiBot observation.state dim at least {required_state_len}, "
            f"got {states.shape[-1]} in {parquet_path}"
        )

    values = np.zeros((states.shape[0], len(target_fields)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=np.float32)
    for col_idx, field in enumerate(target_fields):
        if is_agibot_series_derived_field(field):
            field_values, field_valid = derive_agibot_series_field(timestamps, states, field)
            values[:, col_idx] = field_values
            valid[:, col_idx] = field_valid
            continue
        for row_idx, state in enumerate(states):
            value, is_valid = read_agibot_state_field(state, field)
            if is_valid:
                values[row_idx, col_idx] = value
                valid[row_idx, col_idx] = 1.0

    return timestamps, values, valid
