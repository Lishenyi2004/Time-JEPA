#!/usr/bin/env python3
"""Aggregate all formal EgoExo knee-chain runs."""

from __future__ import annotations

import argparse
import json

from physics_consistency.egoexo import aggregate_grounded as engine

from .bank_common import json_safe


AGGREGATE_PROTOCOL = "egoexo_knee_grounded_five_seed_aggregate_v1"


def aggregate(*args, **kwargs):
    engine.AGGREGATE_PROTOCOL = AGGREGATE_PROTOCOL
    return engine.aggregate(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="physics_consistency/egoexo_knee/results/model_runs",
    )
    parser.add_argument(
        "--output-dir",
        default="physics_consistency/egoexo_knee/results/aggregate",
    )
    parser.add_argument("--models", default="time,video_only,zero")
    parser.add_argument("--reference-model", default="time")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    args = parser.parse_args()
    result = aggregate(
        args.root,
        args.output_dir,
        models=engine._parse_csv_list(args.models),
        seeds=engine._parse_csv_list(args.seeds, int),
        reference_model=args.reference_model,
        bootstrap_samples=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
