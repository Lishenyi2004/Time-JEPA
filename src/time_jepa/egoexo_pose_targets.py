import json
import math
from functools import lru_cache

import numpy as np
import torch

from src.time_jepa.time_sampling import nearest_time_indices


# Increment whenever a same-named EgoExo target changes numerical semantics.
# Version 2 fixes z-up hand height and replaces the degenerate legacy torso
# lateral projection with the corrected torso inclination definition.
EGOEXO_TARGET_DEFINITION_VERSION = 2


EGOEXO_TARGET_GROUPS = {
    "whole_body_compact_v1": [
        "pelvis_speed",
        "torso_lateral_lean",
        "left_wrist_rel_speed",
        "right_wrist_rel_speed",
        "left_ankle_rel_speed",
        "right_ankle_rel_speed",
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_knee_flexion",
        "right_knee_flexion",
    ],
    "left_body_seen_v1": [
        "left_wrist_rel_speed",
        "left_elbow_flexion",
        "left_arm_extension",
        "left_ankle_rel_speed",
        "left_knee_flexion",
        "left_leg_extension",
    ],
    "right_body_shot_v1": [
        "right_wrist_rel_speed",
        "right_elbow_flexion",
        "right_arm_extension",
        "right_ankle_rel_speed",
        "right_knee_flexion",
        "right_leg_extension",
    ],
    "whole_body_compact_v2": [
        "pelvis_speed",
        "left_wrist_rel_speed",
        "right_wrist_rel_speed",
        "left_ankle_rel_speed",
        "right_ankle_rel_speed",
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_knee_flexion",
        "right_knee_flexion",
    ],
    # Exact-12D root/torso extension of whole_body_compact_v2. The original
    # 3D pelvis speed magnitude is split into ground-plane speed and signed
    # vertical velocity, then turning rate and torso inclination are added.
    "whole_body_kinematic_v3_12d": [
        "pelvis_planar_speed",
        "pelvis_vertical_velocity",
        "torso_yaw_rate",
        "torso_inclination",
        "left_wrist_rel_speed",
        "right_wrist_rel_speed",
        "left_ankle_rel_speed",
        "right_ankle_rel_speed",
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_knee_flexion",
        "right_knee_flexion",
    ],
    "left_body_seen_v2": [
        "left_wrist_rel_speed",
        "left_elbow_flexion",
        "left_ankle_rel_speed",
        "left_knee_flexion",
    ],
    "right_body_shot_v2": [
        "right_wrist_rel_speed",
        "right_elbow_flexion",
        "right_ankle_rel_speed",
        "right_knee_flexion",
    ],
    # Unseen-physics shot for the WHOLE encoder (whole_body_compact_v2).
    # None of these 6 are in whole_body_compact_v2's supervised set: the 4
    # angular velocities are first derivatives of the supervised elbow/knee
    # angles, inter_wrist_distance is a bilateral coordination quantity, and
    # pelvis_acceleration is the first derivative of the supervised pelvis_speed
    # (CoVLA velocity->acceleration analogue). GT gate: angular velocities
    # eps/step 0.15-0.19 (clean); pelvis_acc 0.38 (borderline, weakest dim).
    "whole_unseen_v1": [
        "left_elbow_angular_velocity",
        "right_elbow_angular_velocity",
        "left_knee_angular_velocity",
        "right_knee_angular_velocity",
        "inter_wrist_distance",
        "pelvis_acceleration",
    ],
    # Static-geometry unseen-physics shot for the WHOLE encoder. Unlike v1
    # (dynamic derivatives, which V-JEPA representations read out near-zero),
    # these are single-frame upper-body geometric quantities: static (no
    # differentiation) and ego-observable (upper body stays in frame), chosen
    # to be readable in ID and to survive the ego OOD view. None appear in
    # whole_body_compact_v2's supervised set. Prototype (8 takes): all CV>0.2,
    # valid 60-95%; shoulder_width was rejected (CV 0.12, near-constant).
    "whole_unseen_v2": [
        "inter_wrist_distance",
        "left_wrist_shoulder_dist",
        "right_wrist_shoulder_dist",
        "left_hand_height",
        "right_hand_height",
        "inter_elbow_distance",
    ],
    # Easy, low-order geometry shot associated with the original 9D encoder.
    # These fields share joints with the supervised quantities without
    # repeating the supervised speed/flexion values themselves.
    "whole9_shot_easy_v1": [
        "inter_wrist_distance",
        "left_wrist_shoulder_dist",
        "right_wrist_shoulder_dist",
        "inter_ankle_distance",
        "pelvis_support_height",
    ],
    # Nested extension for the 12D encoder: keep the same 5D test and add one
    # relative torso-rotation state associated with the new yaw/torso fields.
    "whole12_shot_easy_v1": [
        "inter_wrist_distance",
        "left_wrist_shoulder_dist",
        "right_wrist_shoulder_dist",
        "inter_ankle_distance",
        "pelvis_support_height",
        "shoulder_hip_twist",
    ],
}


def normalize_egoexo_targets(targets):
    if targets is None:
        return list(EGOEXO_TARGET_GROUPS["whole_body_compact_v2"])
    if isinstance(targets, dict):
        target_type = targets.get("type", "whole_body_compact_v2")
        if target_type not in EGOEXO_TARGET_GROUPS:
            raise ValueError(f"Unknown EgoExo target preset: {target_type}")
        return list(EGOEXO_TARGET_GROUPS[target_type])
    if isinstance(targets, str):
        if targets in EGOEXO_TARGET_GROUPS:
            return list(EGOEXO_TARGET_GROUPS[targets])
        return [part.strip() for part in targets.split(",") if part.strip()]
    normalized = []
    for target in targets:
        if target in EGOEXO_TARGET_GROUPS:
            normalized.extend(EGOEXO_TARGET_GROUPS[target])
        else:
            normalized.append(str(target))
    unknown = sorted(set(normalized) - set(sum(EGOEXO_TARGET_GROUPS.values(), [])))
    if unknown:
        raise ValueError(f"Unknown EgoExo target fields: {unknown}")
    return normalized


def _point(joints, name, min_views):
    value = joints.get(name)
    if not isinstance(value, dict):
        return None
    if float(value.get("num_views_for_3d", 0)) < float(min_views):
        return None
    point = np.asarray([value.get("x"), value.get("y"), value.get("z")], dtype=np.float64)
    return point if np.isfinite(point).all() else None


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return 0.5 * (a + b)


def _angle(a, b, c):
    if a is None or b is None or c is None:
        return np.nan
    u = a - b
    v = c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-12:
        return np.nan
    cosine = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return math.acos(cosine)


def _smooth_positions(values, valid, window):
    if window is None or int(window) <= 1:
        return values.copy(), valid.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    output = np.full_like(values, np.nan)
    output_valid = np.zeros_like(valid)
    for index in range(values.shape[0]):
        lo = max(0, index - radius)
        hi = min(values.shape[0], index + radius + 1)
        mask = valid[lo:hi]
        if mask.sum() < max(2, (hi - lo + 1) // 2):
            continue
        output[index] = np.median(values[lo:hi][mask], axis=0)
        output_valid[index] = True
    return output, output_valid


def _derivative(values, valid, times):
    output = np.full_like(values, np.nan)
    output_valid = np.zeros(valid.shape, dtype=bool)
    for index in range(values.shape[0]):
        left = index - 1
        right = index + 1
        if left >= 0 and right < values.shape[0] and valid[left] and valid[right]:
            dt = times[right] - times[left]
            if dt > 0:
                output[index] = (values[right] - values[left]) / dt
                output_valid[index] = True
        elif right < values.shape[0] and valid[index] and valid[right]:
            dt = times[right] - times[index]
            if dt > 0:
                output[index] = (values[right] - values[index]) / dt
                output_valid[index] = True
        elif left >= 0 and valid[left] and valid[index]:
            dt = times[index] - times[left]
            if dt > 0:
                output[index] = (values[index] - values[left]) / dt
                output_valid[index] = True
    return output, output_valid


def _signed_angle_derivative(unit_vectors, valid, times):
    """Differentiate 2D directions without constructing an absolute yaw."""
    output = np.full(unit_vectors.shape[0], np.nan, dtype=np.float64)
    output_valid = np.zeros(valid.shape, dtype=bool)
    for index in range(unit_vectors.shape[0]):
        left = index - 1
        right = index + 1
        if left >= 0 and right < unit_vectors.shape[0] and valid[left] and valid[right]:
            start, end = left, right
        elif right < unit_vectors.shape[0] and valid[index] and valid[right]:
            start, end = index, right
        elif left >= 0 and valid[left] and valid[index]:
            start, end = left, index
        else:
            continue
        dt = times[end] - times[start]
        if dt <= 0:
            continue
        a = unit_vectors[start]
        b = unit_vectors[end]
        cross = a[0] * b[1] - a[1] * b[0]
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        output[index] = np.arctan2(cross, dot) / dt
        output_valid[index] = True
    return output, output_valid


@lru_cache(maxsize=64)
def load_egoexo_pose_series(pose_json_path, pose_fps=30.0, min_views=2, smooth_window=5):
    with open(pose_json_path, "r") as handle:
        payload = json.load(handle)
    frame_ids = np.asarray(sorted(int(key) for key in payload), dtype=np.int64)
    times = frame_ids.astype(np.float64) / float(pose_fps)
    joint_names = [
        "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
        "left-wrist", "right-wrist", "left-hip", "right-hip",
        "left-knee", "right-knee", "left-ankle", "right-ankle",
    ]
    positions = {name: np.full((len(frame_ids), 3), np.nan, dtype=np.float64) for name in joint_names}
    position_valid = {name: np.zeros(len(frame_ids), dtype=bool) for name in joint_names}
    for row_index, frame_id in enumerate(frame_ids.tolist()):
        persons = payload.get(str(frame_id), [])
        if not persons:
            continue
        joints = persons[0].get("annotation3D", {})
        for name in joint_names:
            point = _point(joints, name, min_views=min_views)
            if point is not None:
                positions[name][row_index] = point
                position_valid[name][row_index] = True
    for name in joint_names:
        positions[name], position_valid[name] = _smooth_positions(
            positions[name], position_valid[name], smooth_window
        )

    pelvis = 0.5 * (positions["left-hip"] + positions["right-hip"])
    pelvis_valid = position_valid["left-hip"] & position_valid["right-hip"]
    shoulder = 0.5 * (positions["left-shoulder"] + positions["right-shoulder"])
    shoulder_valid = position_valid["left-shoulder"] & position_valid["right-shoulder"]

    pelvis_velocity, pelvis_velocity_valid = _derivative(pelvis, pelvis_valid, times)
    values = {
        "pelvis_speed": np.linalg.norm(pelvis_velocity, axis=1),
        "pelvis_planar_speed": np.linalg.norm(pelvis_velocity[:, :2], axis=1),
        "pelvis_vertical_velocity": pelvis_velocity[:, 2],
    }
    valid = {
        "pelvis_speed": pelvis_velocity_valid,
        "pelvis_planar_speed": pelvis_velocity_valid,
        "pelvis_vertical_velocity": pelvis_velocity_valid,
    }

    # Torso yaw uses the average shoulder/hip lateral axis projected onto the
    # z-up ground plane. Signed relative angles avoid absolute-yaw unwrapping
    # and remain invariant to a fixed horizontal rotation of the capture.
    shoulder_axis = positions["right-shoulder"] - positions["left-shoulder"]
    hip_axis = positions["right-hip"] - positions["left-hip"]
    body_lateral_xy = 0.5 * (shoulder_axis[:, :2] + hip_axis[:, :2])
    body_lateral_norm = np.linalg.norm(body_lateral_xy, axis=1)
    body_lateral_valid = (
        shoulder_valid & pelvis_valid & (body_lateral_norm > 1e-8)
    )
    body_lateral_unit = np.full_like(body_lateral_xy, np.nan)
    body_lateral_unit[body_lateral_valid] = (
        body_lateral_xy[body_lateral_valid]
        / body_lateral_norm[body_lateral_valid, None]
    )
    torso_yaw_rate, torso_yaw_rate_valid = _signed_angle_derivative(
        body_lateral_unit, body_lateral_valid, times
    )
    values["torso_yaw_rate"] = torso_yaw_rate
    valid["torso_yaw_rate"] = torso_yaw_rate_valid

    # Correct z-up torso inclination. The deprecated torso_lateral_lean name
    # is retained as a compatibility alias, but no longer uses the structurally
    # near-zero projection of the torso axis onto the hip lateral axis.
    torso_vector = shoulder - pelvis
    torso_norm = np.linalg.norm(torso_vector, axis=1)
    torso_horizontal_norm = np.linalg.norm(torso_vector[:, :2], axis=1)
    torso_inclination = np.arctan2(torso_horizontal_norm, torso_vector[:, 2])
    torso_inclination_valid = (
        shoulder_valid
        & pelvis_valid
        & (torso_norm > 1e-8)
        & (torso_vector[:, 2] > 0.0)
    )
    values["torso_inclination"] = torso_inclination
    valid["torso_inclination"] = torso_inclination_valid
    values["torso_lateral_lean"] = torso_inclination.copy()
    valid["torso_lateral_lean"] = torso_inclination_valid.copy()

    # pelvis_acceleration = d(pelvis_speed)/dt, signed scalar (whole_unseen_v1).
    # Second-order quantity from 2-view triangulated pose; smoothing already
    # applied to joint positions upstream. GT gate eps/step 0.38 (borderline).
    pelvis_acceleration, pelvis_acceleration_valid = _derivative(
        values["pelvis_speed"], valid["pelvis_speed"], times
    )
    values["pelvis_acceleration"] = pelvis_acceleration
    valid["pelvis_acceleration"] = pelvis_acceleration_valid

    # inter_wrist_distance = ||left_wrist - right_wrist|| (whole_unseen_v1).
    # Single-frame bilateral coordination quantity, no differentiation.
    inter_wrist_vector = positions["left-wrist"] - positions["right-wrist"]
    values["inter_wrist_distance"] = np.linalg.norm(inter_wrist_vector, axis=1)
    valid["inter_wrist_distance"] = position_valid["left-wrist"] & position_valid["right-wrist"]

    # Easy whole-body shot geometry: lower-body stance and pelvis height above
    # the bilateral ankle support level. Both definitions are single-frame and
    # use z as the verified vertical axis.
    inter_ankle_vector = positions["left-ankle"] - positions["right-ankle"]
    values["inter_ankle_distance"] = np.linalg.norm(inter_ankle_vector, axis=1)
    valid["inter_ankle_distance"] = (
        position_valid["left-ankle"] & position_valid["right-ankle"]
    )
    ankle_mid_z = 0.5 * (
        positions["left-ankle"][:, 2] + positions["right-ankle"][:, 2]
    )
    values["pelvis_support_height"] = pelvis[:, 2] - ankle_mid_z
    valid["pelvis_support_height"] = (
        pelvis_valid
        & position_valid["left-ankle"]
        & position_valid["right-ankle"]
    )

    # Static upper-body geometry (whole_unseen_v2): single-frame distances/heights,
    # no differentiation. Ego-observable; chosen for readable ID + robust OOD.
    inter_elbow_vector = positions["left-elbow"] - positions["right-elbow"]
    values["inter_elbow_distance"] = np.linalg.norm(inter_elbow_vector, axis=1)
    valid["inter_elbow_distance"] = position_valid["left-elbow"] & position_valid["right-elbow"]

    for side in ("left", "right"):
        wrist_shoulder = positions[f"{side}-wrist"] - positions[f"{side}-shoulder"]
        values[f"{side}_wrist_shoulder_dist"] = np.linalg.norm(wrist_shoulder, axis=1)
        valid[f"{side}_wrist_shoulder_dist"] = (
            position_valid[f"{side}-wrist"] & position_valid[f"{side}-shoulder"]
        )
        # Hand height = wrist vertical position relative to pelvis (z-up).
        values[f"{side}_hand_height"] = (positions[f"{side}-wrist"] - pelvis)[:, 2]
        valid[f"{side}_hand_height"] = position_valid[f"{side}-wrist"] & pelvis_valid

    # Signed shoulder-axis rotation relative to the hip axis on the xy ground
    # plane. This is an easy adjacent torso state for the 12D shot, not the
    # supervised torso_yaw_rate itself.
    shoulder_axis_xy = shoulder_axis[:, :2]
    hip_axis_xy = hip_axis[:, :2]
    shoulder_axis_xy_norm = np.linalg.norm(shoulder_axis_xy, axis=1)
    hip_axis_xy_norm = np.linalg.norm(hip_axis_xy, axis=1)
    shoulder_hip_cross = (
        hip_axis_xy[:, 0] * shoulder_axis_xy[:, 1]
        - hip_axis_xy[:, 1] * shoulder_axis_xy[:, 0]
    )
    shoulder_hip_dot = np.sum(hip_axis_xy * shoulder_axis_xy, axis=1)
    values["shoulder_hip_twist"] = np.arctan2(shoulder_hip_cross, shoulder_hip_dot)
    valid["shoulder_hip_twist"] = (
        shoulder_valid
        & pelvis_valid
        & (shoulder_axis_xy_norm > 1e-8)
        & (hip_axis_xy_norm > 1e-8)
    )

    for side in ("left", "right"):
        wrist_relative = positions[f"{side}-wrist"] - pelvis
        wrist_valid = position_valid[f"{side}-wrist"] & pelvis_valid
        wrist_velocity, wrist_velocity_valid = _derivative(wrist_relative, wrist_valid, times)
        values[f"{side}_wrist_rel_speed"] = np.linalg.norm(wrist_velocity, axis=1)
        valid[f"{side}_wrist_rel_speed"] = wrist_velocity_valid

        ankle_relative = positions[f"{side}-ankle"] - pelvis
        ankle_valid = position_valid[f"{side}-ankle"] & pelvis_valid
        ankle_velocity, ankle_velocity_valid = _derivative(ankle_relative, ankle_valid, times)
        values[f"{side}_ankle_rel_speed"] = np.linalg.norm(ankle_velocity, axis=1)
        valid[f"{side}_ankle_rel_speed"] = ankle_velocity_valid

        elbow = np.asarray([
            _angle(positions[f"{side}-shoulder"][i], positions[f"{side}-elbow"][i], positions[f"{side}-wrist"][i])
            for i in range(len(frame_ids))
        ])
        elbow_valid = (
            position_valid[f"{side}-shoulder"]
            & position_valid[f"{side}-elbow"]
            & position_valid[f"{side}-wrist"]
            & np.isfinite(elbow)
        )
        values[f"{side}_elbow_flexion"] = elbow
        valid[f"{side}_elbow_flexion"] = elbow_valid

        # d(elbow_flexion)/dt, signed rad/s (whole_unseen_v1). GT gate eps/step ~0.15-0.19.
        elbow_angvel, elbow_angvel_valid = _derivative(elbow, elbow_valid, times)
        values[f"{side}_elbow_angular_velocity"] = elbow_angvel
        valid[f"{side}_elbow_angular_velocity"] = elbow_angvel_valid

        knee = np.asarray([
            _angle(positions[f"{side}-hip"][i], positions[f"{side}-knee"][i], positions[f"{side}-ankle"][i])
            for i in range(len(frame_ids))
        ])
        knee_valid = (
            position_valid[f"{side}-hip"]
            & position_valid[f"{side}-knee"]
            & position_valid[f"{side}-ankle"]
            & np.isfinite(knee)
        )
        values[f"{side}_knee_flexion"] = knee
        valid[f"{side}_knee_flexion"] = knee_valid

        # d(knee_flexion)/dt, signed rad/s (whole_unseen_v1). GT gate eps/step ~0.17-0.19.
        knee_angvel, knee_angvel_valid = _derivative(knee, knee_valid, times)
        values[f"{side}_knee_angular_velocity"] = knee_angvel
        valid[f"{side}_knee_angular_velocity"] = knee_angvel_valid

        arm_extension = np.linalg.norm(positions[f"{side}-wrist"] - positions[f"{side}-shoulder"], axis=1)
        values[f"{side}_arm_extension"] = arm_extension
        valid[f"{side}_arm_extension"] = position_valid[f"{side}-wrist"] & position_valid[f"{side}-shoulder"]

        leg_extension = np.linalg.norm(positions[f"{side}-ankle"] - positions[f"{side}-hip"], axis=1)
        values[f"{side}_leg_extension"] = leg_extension
        valid[f"{side}_leg_extension"] = position_valid[f"{side}-ankle"] & position_valid[f"{side}-hip"]

    fields = sorted(values)
    return times, values, valid, fields


def build_egoexo_tubelet_targets(
    pose_json_path,
    sample_times,
    target_fields=None,
    tubelet_size=2,
    max_time_delta=0.05,
    pose_fps=30.0,
    min_views=2,
    smooth_window=5,
):
    target_fields = normalize_egoexo_targets(target_fields)
    source_times, values, valid, _ = load_egoexo_pose_series(
        str(pose_json_path), float(pose_fps), int(min_views), int(smooth_window)
    )
    indices, time_valid = nearest_time_indices(sample_times, source_times, max_delta=max_time_delta)
    frame_values = np.stack([values[field][indices] for field in target_fields], axis=-1)
    frame_valid = np.stack([valid[field][indices] for field in target_fields], axis=-1)
    frame_valid = frame_valid.astype(np.float32) * time_valid[:, None].astype(np.float32)
    frame_values = np.nan_to_num(frame_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tubelet_size = int(tubelet_size)
    usable = (len(sample_times) // tubelet_size) * tubelet_size
    frame_values = frame_values[:usable].reshape(-1, tubelet_size, len(target_fields))
    frame_valid = frame_valid[:usable].reshape(-1, tubelet_size, len(target_fields))
    counts = frame_valid.sum(axis=1)
    targets = (frame_values * frame_valid).sum(axis=1) / np.maximum(counts, 1.0)
    target_valid = (counts > 0).astype(np.float32)
    return torch.from_numpy(targets), torch.from_numpy(target_valid)
