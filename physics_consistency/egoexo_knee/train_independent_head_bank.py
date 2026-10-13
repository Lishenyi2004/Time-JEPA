#!/usr/bin/env python3
"""Train four knee-chain probes using the shared EgoExo training engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from physics_consistency.egoexo import bank_data as shared_data
from physics_consistency.egoexo import train_independent_head_bank as engine

from .bank_common import (
    TARGET_FIELDS,
    TARGET_NAMES,
    load_config,
    target_manifest,
)
from .grounded_eval import (
    evaluate_cross_view,
    evaluate_prediction_file,
    summarize_run,
)
from .gt_kinematic_gate import GT_GATE_PROTOCOL


def bind_protocol() -> None:
    """Bind knee targets while retaining the tested DDP/training mechanics."""

    shared_data.TARGET_FIELDS = TARGET_FIELDS
    shared_data.TARGET_NAMES = TARGET_NAMES
    engine.TARGET_NAMES = TARGET_NAMES
    engine.EgoExoPhysicsBankDataset = shared_data.EgoExoPhysicsBankDataset
    engine.get_or_compute_stats = shared_data.get_or_compute_stats
    engine.load_protocol_rows = shared_data.load_protocol_rows
    engine.evaluate_prediction_file = evaluate_prediction_file
    engine.evaluate_cross_view = evaluate_cross_view
    engine.summarize_run = summarize_run
    engine.target_manifest = target_manifest
    engine.GT_GATE_PROTOCOL = GT_GATE_PROTOCOL


def train(config: dict) -> Path:
    bind_protocol()
    return engine.train(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--reuse-heads-from")
    args = parser.parse_args()
    if args.reuse_heads_from:
        raise ValueError(
            "EgoExo knee runs train every seed independently; "
            "REUSE_HEADS_FROM must be empty."
        )
    overrides = {}
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = int(args.seed)
    if args.output_root:
        overrides.setdefault("output", {})["root"] = str(
            Path(args.output_root).expanduser()
        )
    train(load_config(args.config, overrides))


if __name__ == "__main__":
    main()

