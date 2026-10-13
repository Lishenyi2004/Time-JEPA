import json
import math

import numpy as np

from src.time_jepa.egoexo_pose_targets import (
    EGOEXO_TARGET_GROUPS,
    build_egoexo_tubelet_targets,
    load_egoexo_pose_series,
    normalize_egoexo_targets,
)


ORIGINAL_WHOLE9 = [
    "pelvis_speed",
    "left_wrist_rel_speed",
    "right_wrist_rel_speed",
    "left_ankle_rel_speed",
    "right_ankle_rel_speed",
    "left_elbow_flexion",
    "right_elbow_flexion",
    "left_knee_flexion",
    "right_knee_flexion",
]

WHOLE12 = [
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
]

SHOT9 = [
    "inter_wrist_distance",
    "left_wrist_shoulder_dist",
    "right_wrist_shoulder_dist",
    "inter_ankle_distance",
    "pelvis_support_height",
]

SHOT12 = SHOT9 + ["shoulder_hip_twist"]


def _rotate_xy(point, angle):
    point = np.asarray(point, dtype=np.float64)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [
            cosine * point[0] - sine * point[1],
            sine * point[0] + cosine * point[1],
            point[2],
        ],
        dtype=np.float64,
    )


def _joint(point):
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]),
        "num_views_for_3d": 2,
    }


def _make_pose_payload(num_frames=5, yaw_step=0.2, shoulder_hip_twist=0.2):
    payload = {}
    for frame_index in range(num_frames):
        yaw = yaw_step * frame_index
        translation = np.asarray([0.3 * frame_index, 0.0, 0.1 * frame_index])
        pelvis = translation + np.asarray([0.0, 0.0, 1.0])

        hip_left = pelvis + _rotate_xy([-0.2, 0.0, 0.0], yaw)
        hip_right = pelvis + _rotate_xy([0.2, 0.0, 0.0], yaw)

        shoulder_mid = pelvis + _rotate_xy([0.2, 0.0, 0.7], yaw)
        shoulder_left = shoulder_mid + _rotate_xy(
            [-0.3, 0.0, 0.0], yaw + shoulder_hip_twist
        )
        shoulder_right = shoulder_mid + _rotate_xy(
            [0.3, 0.0, 0.0], yaw + shoulder_hip_twist
        )

        local_points = {
            "left-elbow": [-0.45, 0.15, 0.55],
            "right-elbow": [0.45, -0.10, 0.55],
            "left-wrist": [-0.60, 0.25, 0.40],
            "right-wrist": [0.60, -0.10, 0.50],
            "left-knee": [-0.20, 0.08, -0.50],
            "right-knee": [0.20, -0.08, -0.50],
            "left-ankle": [-0.25, 0.0, -1.0],
            "right-ankle": [0.25, 0.0, -1.0],
        }
        world_points = {
            name: pelvis + _rotate_xy(point, yaw)
            for name, point in local_points.items()
        }
        world_points.update(
            {
                "left-shoulder": shoulder_left,
                "right-shoulder": shoulder_right,
                "left-hip": hip_left,
                "right-hip": hip_right,
            }
        )
        payload[str(frame_index)] = [
            {"annotation3D": {name: _joint(point) for name, point in world_points.items()}}
        ]
    return payload


def test_target_preset_dimensions_and_order():
    assert EGOEXO_TARGET_GROUPS["whole_body_compact_v2"] == ORIGINAL_WHOLE9
    assert EGOEXO_TARGET_GROUPS["whole_body_kinematic_v3_12d"] == WHOLE12
    assert EGOEXO_TARGET_GROUPS["whole9_shot_easy_v1"] == SHOT9
    assert EGOEXO_TARGET_GROUPS["whole12_shot_easy_v1"] == SHOT12
    assert normalize_egoexo_targets({"type": "whole_body_kinematic_v3_12d"}) == WHOLE12
    assert normalize_egoexo_targets("whole9_shot_easy_v1") == SHOT9


def test_corrected_zup_torso_and_new_kinematic_targets(tmp_path):
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(json.dumps(_make_pose_payload()))
    load_egoexo_pose_series.cache_clear()

    _, values, valid, _ = load_egoexo_pose_series(
        str(pose_path), pose_fps=1.0, min_views=2, smooth_window=1
    )

    np.testing.assert_allclose(values["pelvis_planar_speed"], 0.3, atol=1e-7)
    np.testing.assert_allclose(values["pelvis_vertical_velocity"], 0.1, atol=1e-7)
    np.testing.assert_allclose(values["torso_yaw_rate"], 0.2, atol=2e-3)

    expected_inclination = math.atan2(0.2, 0.7)
    np.testing.assert_allclose(values["torso_inclination"], expected_inclination, atol=1e-7)
    np.testing.assert_allclose(values["torso_lateral_lean"], expected_inclination, atol=1e-7)

    # The corrected definition uses relative z, not the rotating world-y coordinate.
    np.testing.assert_allclose(values["left_hand_height"], 0.4, atol=1e-7)
    np.testing.assert_allclose(values["right_hand_height"], 0.5, atol=1e-7)

    np.testing.assert_allclose(values["inter_ankle_distance"], 0.5, atol=1e-7)
    np.testing.assert_allclose(values["pelvis_support_height"], 1.0, atol=1e-7)
    np.testing.assert_allclose(values["shoulder_hip_twist"], 0.2, atol=1e-7)

    for field in (
        "pelvis_planar_speed",
        "pelvis_vertical_velocity",
        "torso_yaw_rate",
        "torso_inclination",
        "left_hand_height",
        "right_hand_height",
        "inter_ankle_distance",
        "pelvis_support_height",
        "shoulder_hip_twist",
    ):
        assert valid[field].all(), field


def test_new_presets_build_expected_tubelet_dimensions(tmp_path):
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(json.dumps(_make_pose_payload()))
    load_egoexo_pose_series.cache_clear()
    sample_times = np.arange(4, dtype=np.float64)

    for preset, expected_dim in (
        ("whole_body_compact_v2", 9),
        ("whole_body_kinematic_v3_12d", 12),
        ("whole9_shot_easy_v1", 5),
        ("whole12_shot_easy_v1", 6),
    ):
        target, target_valid = build_egoexo_tubelet_targets(
            str(pose_path),
            sample_times,
            target_fields=preset,
            tubelet_size=2,
            max_time_delta=1e-6,
            pose_fps=1.0,
            min_views=2,
            smooth_window=1,
        )
        assert target.shape == (2, expected_dim)
        assert target_valid.shape == (2, expected_dim)
        assert target_valid.bool().all()
