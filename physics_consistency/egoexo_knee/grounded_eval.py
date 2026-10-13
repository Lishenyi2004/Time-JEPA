#!/usr/bin/env python3
"""GT-grounded knee-chain evaluation using the shared limb evaluator."""

from __future__ import annotations

import argparse
import json

from physics_consistency.egoexo import grounded_eval as engine

from .bank_common import (
    ANGLE_INDEX,
    DISTANCE_INDEX,
    ID_VIEWS,
    OOD_VIEWS,
    TARGET_INDEX,
    TARGET_NAMES,
    TARGET_SPECS,
    json_safe,
)
from .pose_geometry import SIDES, cosine_law_distance, morphology_for_pose


GT_GATE_PROTOCOL = "egoexo_take_calibrated_knee_cosine_law_v1"
GROUNDED_PROTOCOL = "egoexo_gt_grounded_knee_cosine_law_v1"


def bind_protocol() -> None:
    """Bind the knee target schema to the shared grounded evaluator."""

    engine.ANGLE_INDEX = ANGLE_INDEX
    engine.DISTANCE_INDEX = DISTANCE_INDEX
    engine.ID_VIEWS = ID_VIEWS
    engine.OOD_VIEWS = OOD_VIEWS
    engine.TARGET_INDEX = TARGET_INDEX
    engine.TARGET_NAMES = TARGET_NAMES
    engine.TARGET_SPECS = TARGET_SPECS
    engine.SIDES = SIDES
    engine.cosine_law_distance = cosine_law_distance
    engine.morphology_for_pose = morphology_for_pose
    engine.GT_GATE_PROTOCOL = GT_GATE_PROTOCOL
    engine.GROUNDED_PROTOCOL = GROUNDED_PROTOCOL


def evaluate_prediction_file(*args, **kwargs):
    bind_protocol()
    return engine.evaluate_prediction_file(*args, **kwargs)


def evaluate_cross_view(*args, **kwargs):
    bind_protocol()
    return engine.evaluate_cross_view(*args, **kwargs)


def summarize_run(*args, **kwargs):
    bind_protocol()
    return engine.summarize_run(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pose-fps", type=float, default=30.0)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--split-name")
    args = parser.parse_args()
    result = evaluate_prediction_file(
        args.predictions,
        args.gt_gate,
        args.output_dir,
        {
            "fps": args.pose_fps,
            "min_views": args.min_views,
            "smooth_window": args.smooth_window,
        },
        split_name=args.split_name,
    )
    print(
        json.dumps(
            json_safe({"metrics": result["metrics_path"], "legs": result["arms"]}),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

