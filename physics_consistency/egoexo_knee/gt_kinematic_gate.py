#!/usr/bin/env python3
"""Freeze and validate the EgoExo knee-chain cosine-law protocol."""

from __future__ import annotations

import argparse
import json

from physics_consistency.egoexo import gt_kinematic_gate as engine

from .bank_common import json_safe
from .pose_geometry import (
    SIDES,
    cosine_law_distance,
    load_take_leg_geometry,
    pool_window_leg_targets,
)


GT_GATE_PROTOCOL = "egoexo_take_calibrated_knee_cosine_law_v1"


def bind_protocol() -> None:
    """Bind knee geometry to the shared, tested limb-gate mechanics."""

    engine.SIDES = SIDES
    engine.cosine_law_distance = cosine_law_distance
    engine.load_take_arm_geometry = load_take_leg_geometry
    engine.pool_window_arm_targets = pool_window_leg_targets
    engine.GT_GATE_PROTOCOL = GT_GATE_PROTOCOL
    engine.FORMULA_NAME = "law_of_cosines_internal_knee_angle"
    engine.FORMULA_EXPRESSION = (
        "sqrt(L_thigh^2 + L_shin^2 - 2*L_thigh*L_shin*cos(theta))"
    )
    engine.MORPHOLOGY_DESCRIPTION = (
        "per-take temporal median thigh/shin lengths after 5-frame joint smoothing"
    )
    engine.INTERPRETATION_GUARD = (
        "This gate validates pose-derived hip-knee-ankle geometry, not independent "
        "sensors, forces, contacts, gait dynamics, or Newtonian human dynamics."
    )


def run_gate(config: dict, config_path=None) -> dict:
    bind_protocol()
    return engine.run_gate(config, config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, path = engine._load_yaml(args.config)
    result = run_gate(config, path)
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

