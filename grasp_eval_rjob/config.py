from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from grasp_eval_rjob.constants import FAMILY_TO_PROPERTIES, FAMILIES, PROPERTIES


DEFAULTS: dict[str, Any] = {
    "model": {
        "model_name": "vit_giant_xformers",
        "resolution": 384,
        "patch_size": 16,
        "tubelet_size": 2,
        "pred_depth": 24,
        "pred_embed_dim": 384,
        "pred_num_heads": 12,
        "num_mask_tokens": 2,
        "levels_predictor": 4,
        "img_temporal_dim_size": None,
        "uniform_power": True,
        "use_rope": True,
        "use_sdpa": True,
        "interpolate_rope": True,
        "modality_embedding": True,
        "wide_silu": True,
        "strict_checkpoint": True,
        "checkpoint_mmap": True,
        "weights_dtype": "float32",
    },
    "evaluation": {
        "frames_per_clip": 16,
        "frame_step": 10,
        "stride": 2,
        "context_frames": 2,
        "chunk_size": 1,
        "num_workers": 2,
        "decord_num_threads": 1,
        "use_bfloat16": True,
        "seed": 239,
        "max_pairs": None,
    },
    "scoring": {
        "bootstrap_samples": 10000,
        "bootstrap_seed": 239,
        "minimum_nondegraded_properties": 6,
        "require_leave_largest_gain_out_positive": True,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    config = _deep_update(copy.deepcopy(DEFAULTS), user_config)
    config["config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for required in (
        "name",
        "checkpoint",
        "checkpoint_sha256",
        "dataset_root",
        "data_audit",
        "output_dir",
    ):
        if not config.get(required):
            raise ValueError(f"Missing required config field: {required}")

    checksum = str(config["checkpoint_sha256"])
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")

    model = config["model"]
    evaluation = config["evaluation"]
    positive_ints = {
        "model.resolution": model["resolution"],
        "model.patch_size": model["patch_size"],
        "model.tubelet_size": model["tubelet_size"],
        "model.pred_depth": model["pred_depth"],
        "model.num_mask_tokens": model["num_mask_tokens"],
        "model.levels_predictor": model["levels_predictor"],
        "evaluation.frames_per_clip": evaluation["frames_per_clip"],
        "evaluation.frame_step": evaluation["frame_step"],
        "evaluation.stride": evaluation["stride"],
        "evaluation.context_frames": evaluation["context_frames"],
        "evaluation.chunk_size": evaluation["chunk_size"],
    }
    for field, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer; got {value!r}")

    # This evaluator intentionally has no context/model/property sweep.
    fixed = {
        "evaluation.frames_per_clip": (evaluation["frames_per_clip"], 16),
        "evaluation.frame_step": (evaluation["frame_step"], 10),
        "evaluation.stride": (evaluation["stride"], 2),
        "evaluation.context_frames": (evaluation["context_frames"], 2),
    }
    violations = [f"{key}={actual} (required {wanted})" for key, (actual, wanted) in fixed.items() if actual != wanted]
    if violations:
        raise ValueError("Fixed GRASP protocol violation: " + ", ".join(violations))
    if evaluation["context_frames"] % model["tubelet_size"]:
        raise ValueError("context_frames must be divisible by tubelet_size")
    if evaluation["frames_per_clip"] % model["tubelet_size"]:
        raise ValueError("frames_per_clip must be divisible by tubelet_size")
    if evaluation["context_frames"] >= evaluation["frames_per_clip"]:
        raise ValueError("context_frames must be shorter than frames_per_clip")
    if model["resolution"] % model["patch_size"]:
        raise ValueError("model.resolution must be divisible by model.patch_size")
    if model["weights_dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("model.weights_dtype must be float32 or bfloat16")
    if not isinstance(evaluation["num_workers"], int) or evaluation["num_workers"] < 0:
        raise ValueError("evaluation.num_workers must be a non-negative integer")
    max_pairs = evaluation.get("max_pairs")
    if max_pairs is not None and (not isinstance(max_pairs, int) or max_pairs <= 0):
        raise ValueError("evaluation.max_pairs must be null or a positive integer")

    scoring = config["scoring"]
    expected_nondegraded = 6  # ceil(0.70 * 8), translating the draft's 7/10 rule.
    if scoring["minimum_nondegraded_properties"] != expected_nondegraded:
        raise ValueError(
            "minimum_nondegraded_properties is frozen at 6/8 after correcting the "
            "upstream 10-property count"
        )
    if scoring["require_leave_largest_gain_out_positive"] is not True:
        raise ValueError(
            "require_leave_largest_gain_out_positive is frozen to enforce the "
            "single-property dominance guard"
        )


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_audit(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["data_audit"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen data audit {path}; run python -m grasp_eval_rjob.audit_data"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "dataset_root",
        "pair_count",
        "video_count",
        "manifest_sha256",
        "content_fingerprint",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Data audit {path} is missing fields: {sorted(missing)}")
    configured_root = str(Path(config["dataset_root"]).expanduser().resolve())
    if payload["dataset_root"] != configured_root:
        raise ValueError(
            f"Data audit root {payload['dataset_root']} != configured root {configured_root}"
        )
    return payload


def protocol_payload(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    checkpoint_info: dict[str, Any] = {
        "path": str(checkpoint),
        "sha256": str(config["checkpoint_sha256"]),
    }
    if checkpoint.is_file():
        checkpoint_info["bytes"] = checkpoint.stat().st_size
    audit_path = Path(config["data_audit"]).expanduser().resolve()
    audit = load_data_audit(config)
    return {
        "protocol_version": 1,
        "status": "fixed_before_grasp_model_results",
        "name": config["name"],
        "checkpoint": checkpoint_info,
        "dataset_root": str(Path(config["dataset_root"]).expanduser().resolve()),
        "data_audit": {
            "path": str(audit_path),
            "sha256": _sha256_file(audit_path),
            "manifest_sha256": audit["manifest_sha256"],
            "content_fingerprint": audit["content_fingerprint"],
            "pair_count": audit["pair_count"],
            "video_count": audit["video_count"],
        },
        "model": config["model"],
        "evaluation": config["evaluation"],
        "scoring": config["scoring"],
        "families": list(FAMILIES),
        "properties": list(PROPERTIES),
        "family_to_properties": {
            key: list(value) for key, value in FAMILY_TO_PROPERTIES.items()
        },
        "aggregation_note": (
            "Official JEPA multi-label family-to-property mapping; eight non-empty "
            "GRASP concepts. The JEPA paper's Table S2 count of ten is inconsistent "
            "with GRASP and the released aggregation code."
        ),
        "upstream_sources": {
            "grasp": {
                "repository": "https://github.com/i-machine-think/grasp",
                "commit": "676d13a1b5d4d86bc3d834529193bd7cce7d764f",
                "google_drive_file_id": "1_xPzN0MS3vVlci4yHL5FBH8N_ZpL6mt8",
                "archive_expected_bytes": 4980560348,
            },
            "jepa_intuitive_physics": {
                "repository": "https://github.com/facebookresearch/jepa-intuitive-physics",
                "commit": "302a2e821d908b4a8667e0c0e48fe733e43c9156",
                "mapping_source": "figures.ipynb",
                "mapping_source_sha256": "0d20bcd6b06cfdd063dc01733d3b1e2c208485f8c23cff757359f07aa8af576a",
                "loader_source_sha256": "dd07933a69214e51f9ecc0fe5f6613d6e078f787faf6fe92cf94e322cb79f28b",
                "config_source_sha256": "f5f7d65df2ecf42d7c3d6350d8c579cb20a993128f30eb5ab75f04d356d115a1",
            },
        },
    }


def protocol_digest(config: dict[str, Any]) -> str:
    serialized = json.dumps(
        protocol_payload(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def evaluation_signature(config: dict[str, Any]) -> str:
    payload = protocol_payload(config)
    shared = {
        key: payload[key]
        for key in (
            "dataset_root",
            "data_audit",
            "evaluation",
            "families",
            "properties",
            "family_to_properties",
        )
    }
    serialized = json.dumps(shared, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
