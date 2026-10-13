"""Frozen target schema for the EgoExo knee-chain experiment."""

from __future__ import annotations

from physics_consistency.egoexo.bank_common import (
    ALL_EVAL_VIEWS,
    ID_VIEWS,
    OOD_VIEWS,
    REPO_ROOT,
    STANDARD_SPLIT_PREFIX,
    TRAIN_VIEWS,
    atomic_json_dump,
    canonical_hash,
    deep_update,
    json_safe,
    load_config,
    safe_name,
    seed_everything,
    sha256_file,
)


TARGET_SPECS = (
    {
        "name": "left_knee_flexion",
        "field": "left_knee_flexion",
        "side": "left",
        "family": "angle",
        "unit": "rad",
        "role": "cosine_law_input",
        "encoder_supervised": True,
    },
    {
        "name": "right_knee_flexion",
        "field": "right_knee_flexion",
        "side": "right",
        "family": "angle",
        "unit": "rad",
        "role": "cosine_law_input",
        "encoder_supervised": True,
    },
    {
        "name": "left_leg_extension",
        "field": "left_leg_extension",
        "side": "left",
        "family": "distance",
        "unit": "m",
        "role": "direct_endpoint",
        "encoder_supervised": False,
    },
    {
        "name": "right_leg_extension",
        "field": "right_leg_extension",
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
    side: TARGET_INDEX[f"{side}_knee_flexion"] for side in ("left", "right")
}
DISTANCE_INDEX = {
    side: TARGET_INDEX[f"{side}_leg_extension"] for side in ("left", "right")
}


def target_manifest() -> list[dict]:
    return [dict(spec) for spec in TARGET_SPECS]

