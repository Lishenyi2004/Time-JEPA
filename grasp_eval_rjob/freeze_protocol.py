from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from grasp_eval_rjob.config import (
    evaluation_signature,
    load_config,
    protocol_digest,
    protocol_payload,
)


def freeze(config_paths: list[str | Path], output: str | Path) -> dict:
    configs = [load_config(path) for path in config_paths]
    if len(configs) != 3:
        raise ValueError(f"Expected exactly three model configs, got {len(configs)}")
    names = [str(config["name"]) for config in configs]
    if len(set(names)) != 3:
        raise ValueError(f"Model names are not unique: {names}")
    signatures = {evaluation_signature(config) for config in configs}
    if len(signatures) != 1:
        raise RuntimeError("The three configs do not share one evaluation protocol")

    payload = {
        "format_version": 1,
        "status": "fixed_before_grasp_model_results",
        "frozen_unix_time": time.time(),
        "evaluation_signature": signatures.pop(),
        "models": {
            config["name"]: {
                "protocol_digest": protocol_digest(config),
                "protocol": protocol_payload(config),
            }
            for config in configs
        },
        "decision_rule": {
            "primary": "8-property equal macro pair accuracy from AverageSurprise",
            "comparison": "Time-JEPA minus matched Video FT",
            "required_primary_delta": "> 0 percentage points",
            "required_nondegraded_properties": "at least 6/8 with delta >= 0",
            "single_property_guard": (
                "equal-property macro delta remains > 0 after excluding the "
                "largest-gain property"
            ),
            "bootstrap": (
                "10,000 paired pair-level resamples, stratified within the 16 "
                "families; same draw for Time-JEPA and Video FT; recompute all "
                "multi-label property metrics"
            ),
            "encoder_seed_uncertainty_included": False,
        },
        "chronology_note": (
            "The GRASP protocol and corrected eight-property aggregation were "
            "frozen after PEZ results but before any GRASP checkpoint evaluation."
        ),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_existing.pop("frozen_unix_time", None)
        comparable_new = dict(payload)
        comparable_new.pop("frozen_unix_time", None)
        if comparable_existing != comparable_new:
            raise RuntimeError(
                f"Refusing to overwrite a different frozen protocol: {destination}"
            )
        return existing
    temporary = destination.with_name(f".{destination.name}.tmp-p{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = freeze(args.config, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

