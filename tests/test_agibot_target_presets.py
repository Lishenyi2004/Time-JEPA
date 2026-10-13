import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
sys.path.insert(0, str(PROBE_DIR))

from probe_targets import (  # noqa: E402
    AGIBOT_TARGET_GROUPS,
    normalize_agibot_target_fields,
    read_agibot_state_field,
)
from src.time_jepa.agibot_derived_targets import (  # noqa: E402
    BASE_ANGULAR_SPEED,
    BASE_TRANSLATION_SPEED,
    END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS,
    JOINT_CONFIGURATION_FULL_V1_FIELDS,
    WHOLEBODY_CARTESIAN_V1_FIELDS,
    WHOLEBODY_CARTESIAN_V2_FIELDS,
    derive_agibot_series_field,
)
from src.time_jepa.agibot_time_dataset import TARGET_PRESETS, normalize_agibot_targets  # noqa: E402


LEFT_V2 = [
    "state/joint/velocity_norm_left",
    "state/end/velocity_linear_norm_left",
    "state/end/arm_position[1]",
    "state/left_effector/position",
    "state/joint/effort_norm_left",
]

RIGHT_V2 = [
    "state/joint/velocity_norm_right",
    "state/end/velocity_linear_norm_right",
    "state/end/arm_position[4]",
    "state/right_effector/position",
    "state/joint/effort_norm_right",
]

WHOLEBODY_RELATED_SHOT_V2 = [
    "state/joint/position_norm_left",
    "state/joint/position_norm_right",
    "state/joint/velocity_norm_left",
    "state/joint/velocity_norm_right",
    "state/joint/effort_norm_left",
    "state/joint/effort_norm_right",
    "state/end/wrench_force_norm_left",
    "state/end/wrench_force_norm_right",
]


def test_encoder_and_probe_use_the_same_left_arm_seen_v2_fields():
    assert TARGET_PRESETS["left_arm_seen_v2"] == LEFT_V2
    assert AGIBOT_TARGET_GROUPS["left_arm_seen_v2"] == LEFT_V2
    assert normalize_agibot_targets({"type": "left_arm_seen_v2"}) == LEFT_V2
    assert normalize_agibot_target_fields(["left_arm_seen_v2"]) == LEFT_V2


def test_v2_is_the_versioned_v1_no_wrench_ablation():
    assert TARGET_PRESETS["left_arm_seen_v1"][:-1] == LEFT_V2
    assert AGIBOT_TARGET_GROUPS["left_arm_seen_v1"][:-1] == LEFT_V2
    assert AGIBOT_TARGET_GROUPS["left_arm_unseen_right_v1"][:-1] == RIGHT_V2
    assert normalize_agibot_target_fields(["left_arm_unseen_right_v2"]) == RIGHT_V2
    assert all("wrench" not in field for field in LEFT_V2 + RIGHT_V2)


def test_wholebody_readout_exactly_matches_encoder_and_shot_is_14d():
    expected = list(WHOLEBODY_CARTESIAN_V1_FIELDS)
    assert len(expected) == 14
    assert len(set(expected)) == 14
    assert TARGET_PRESETS["wholebody_cartesian_v1"] == expected
    assert AGIBOT_TARGET_GROUPS["wholebody_cartesian_v1"] == expected
    assert normalize_agibot_targets({"type": "wholebody_cartesian_v1"}) == expected
    assert normalize_agibot_target_fields(["wholebody_cartesian_v1"]) == expected

    expected_shot = list(JOINT_CONFIGURATION_FULL_V1_FIELDS)
    assert len(expected_shot) == 14
    assert AGIBOT_TARGET_GROUPS["joint_configuration_full_v1"] == expected_shot
    assert normalize_agibot_target_fields(["joint_configuration_full_v1"]) == expected_shot
    assert set(expected).isdisjoint(expected_shot)


def test_wholebody_v2_readout_matches_encoder_and_ordinary_shot_is_8d():
    expected = list(WHOLEBODY_CARTESIAN_V2_FIELDS)
    assert len(expected) == 18
    assert len(set(expected)) == 18
    assert TARGET_PRESETS["wholebody_cartesian_v2"] == expected
    assert AGIBOT_TARGET_GROUPS["wholebody_cartesian_v2"] == expected
    assert normalize_agibot_targets({"type": "wholebody_cartesian_v2"}) == expected
    assert normalize_agibot_target_fields(["wholebody_cartesian_v2"]) == expected

    assert len(WHOLEBODY_RELATED_SHOT_V2) == 8
    assert len(set(WHOLEBODY_RELATED_SHOT_V2)) == 8
    assert (
        AGIBOT_TARGET_GROUPS["wholebody_related_shot_v2"]
        == WHOLEBODY_RELATED_SHOT_V2
    )
    assert (
        normalize_agibot_target_fields(["wholebody_related_shot_v2"])
        == WHOLEBODY_RELATED_SHOT_V2
    )
    assert set(expected).isdisjoint(WHOLEBODY_RELATED_SHOT_V2)


def test_endpoint_position_derivative_produces_signed_world_velocity():
    timestamps = np.array([0.0, 0.5, 1.5], dtype=np.float64)
    states = np.zeros((3, 97), dtype=np.float32)
    left_velocity = np.array([0.20, -0.10, 0.30], dtype=np.float32)
    right_velocity = np.array([-0.40, 0.50, -0.20], dtype=np.float32)
    velocity = np.concatenate([left_velocity, right_velocity])
    initial_position = np.array([0.3, 0.1, 0.5, 0.4, -0.1, 0.6], dtype=np.float32)
    states[:, 34:40] = initial_position[None, :] + timestamps[:, None] * velocity[None, :]

    for index, field in enumerate(END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS):
        values, valid = derive_agibot_series_field(timestamps, states, field)
        np.testing.assert_allclose(values, velocity[index], atol=1.0e-6)
        np.testing.assert_array_equal(valid, np.ones(3, dtype=np.float32))


def test_endpoint_velocity_outlier_is_masked_per_arm():
    timestamps = np.array([0.0, 1.0], dtype=np.float64)
    states = np.zeros((2, 97), dtype=np.float32)
    states[1, 34] = 3.0
    states[1, 37] = 0.5

    left_values, left_valid = derive_agibot_series_field(
        timestamps, states, END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS[0]
    )
    right_values, right_valid = derive_agibot_series_field(
        timestamps, states, END_EFFECTOR_WORLD_LINEAR_VELOCITY_FIELDS[3]
    )
    np.testing.assert_array_equal(left_values, np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(left_valid, np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(right_values, 0.5, atol=1.0e-6)
    np.testing.assert_array_equal(right_valid, np.ones(2, dtype=np.float32))


def test_wholebody_v2_shot_joint_position_norms_are_bilateral():
    state = np.zeros(97, dtype=np.float32)
    state[40:47] = 1.0
    state[47:54] = 2.0
    left, left_valid = read_agibot_state_field(
        state, "state/joint/position_norm_left"
    )
    right, right_valid = read_agibot_state_field(
        state, "state/joint/position_norm_right"
    )
    np.testing.assert_allclose(left, np.sqrt(7.0), atol=1.0e-6)
    np.testing.assert_allclose(right, 2.0 * np.sqrt(7.0), atol=1.0e-6)
    assert left_valid == 1.0
    assert right_valid == 1.0


def test_base_rates_are_physical_and_quaternion_sign_invariant():
    timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    states = np.zeros((3, 97), dtype=np.float32)
    states[:, 90] = [0.0, 1.0, 2.0]

    yaw = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    # Quaternion component order does not affect the dot-product geodesic.
    states[:, 95] = np.sin(yaw / 2.0)
    states[:, 96] = np.cos(yaw / 2.0)
    states[1, 93:97] *= -1.0

    translation, translation_valid = derive_agibot_series_field(
        timestamps, states, BASE_TRANSLATION_SPEED
    )
    angular, angular_valid = derive_agibot_series_field(
        timestamps, states, BASE_ANGULAR_SPEED
    )

    np.testing.assert_allclose(translation, 1.0, atol=1.0e-6)
    np.testing.assert_allclose(angular, 0.5, atol=1.0e-6)
    np.testing.assert_array_equal(translation_valid, np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(angular_valid, np.ones(3, dtype=np.float32))


def test_base_rate_outliers_are_masked_instead_of_clipped():
    timestamps = np.array([0.0, 1.0], dtype=np.float64)
    states = np.zeros((2, 97), dtype=np.float32)
    states[:, 90] = [0.0, 10.0]
    states[:, 96] = 1.0

    values, valid = derive_agibot_series_field(
        timestamps, states, BASE_TRANSLATION_SPEED
    )
    np.testing.assert_array_equal(values, np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(valid, np.zeros(2, dtype=np.float32))
