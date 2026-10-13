"""Strict configuration loader for isolated iVideoGPT domain adaptation.

The adaptation runner intentionally has its own schema.  It does not import or
mutate the V-JEPA training application, the shared probe parser, or the frozen
iVideoGPT probe entry point.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_MODES = ("video_only", "time")
VALID_DOMAINS = ("covla", "agibot", "egoexo")
REQUIRED_SNAPSHOT_FILES = (
    "tokenizer/config.json",
    "tokenizer/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/model.safetensors",
)


def _resolve_path(value: str, *, must_exist: bool) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def _require(mapping: Dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise RuntimeError(f"Missing required config key {section}.{key}")
    return mapping[key]


def _load_json_list(path: str) -> list:
    with Path(path).open("r") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Expected a non-empty JSON list: {path}")
    return values


def load_adapt_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"Config must be a YAML mapping: {config_path}")

    allowed_top = {"model", "data", "optimization", "physical", "runtime"}
    unknown_top = sorted(set(raw) - allowed_top)
    if unknown_top:
        raise RuntimeError(f"Unknown top-level config keys: {unknown_top}")

    cfg = copy.deepcopy(raw)
    model = cfg.setdefault("model", {})
    data = cfg.setdefault("data", {})
    optimization = cfg.setdefault("optimization", {})
    physical = cfg.setdefault("physical", {})
    runtime = cfg.setdefault("runtime", {})

    model["base_snapshot"] = _resolve_path(
        _require(model, "base_snapshot", "model"), must_exist=True
    )
    missing_snapshot = [
        name
        for name in REQUIRED_SNAPSHOT_FILES
        if not (Path(model["base_snapshot"]) / name).is_file()
    ]
    if missing_snapshot:
        raise RuntimeError(
            f"Incomplete base iVideoGPT snapshot: missing {missing_snapshot}"
        )

    fixed_model_values = {
        "input_frames": 64,
        "input_size": 64,
        "window_frames": 16,
        "context_frames": 2,
        "target_group_size": 2,
        "hidden_size": 1024,
    }
    for key, expected in fixed_model_values.items():
        value = int(model.setdefault(key, expected))
        if value != expected:
            raise ValueError(
                f"Registered iVideoGPT-OXE-64-M protocol requires model.{key}="
                f"{expected}, got {value}"
            )
        model[key] = value
    model.setdefault("attention_implementation", "sdpa")
    if model["attention_implementation"] not in ("sdpa", "eager"):
        raise ValueError("model.attention_implementation must be sdpa or eager")

    domain = str(_require(data, "domain", "data")).strip().lower()
    if domain not in VALID_DOMAINS:
        raise ValueError(f"data.domain must be one of {VALID_DOMAINS}, got {domain}")
    data["domain"] = domain
    data["root"] = _resolve_path(_require(data, "root", "data"), must_exist=True)
    data["encoder_train_split"] = _resolve_path(
        _require(data, "encoder_train_split", "data"), must_exist=True
    )
    split_values = _load_json_list(data["encoder_train_split"])
    data["encoder_train_split_size"] = len(split_values)
    data.setdefault("fps", 5.0)
    data.setdefault("frames", 64)
    data.setdefault("tubelet_size", 2)
    if float(data["fps"]) != 5.0 or int(data["frames"]) != 64:
        raise ValueError("The aligned protocol requires 64 frames sampled at 5 FPS")
    if int(data["tubelet_size"]) != 2:
        raise ValueError("The aligned physical protocol requires tubelet_size=2")
    if data.get("manifest_path"):
        data["manifest_path"] = _resolve_path(
            data["manifest_path"], must_exist=True
        )

    mode = str(_require(optimization, "mode", "optimization")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"optimization.mode must be one of {VALID_MODES}")
    optimization["mode"] = mode
    defaults = {
        "epochs": 20,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lr": 3.0e-5,
        "start_lr": 3.0e-6,
        "final_lr": 1.0e-6,
        "warmup_epochs": 1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed": 239,
    }
    for key, value in defaults.items():
        optimization.setdefault(key, value)
    for key in ("epochs", "per_device_batch_size", "gradient_accumulation_steps"):
        if int(optimization[key]) <= 0:
            raise ValueError(f"optimization.{key} must be positive")
        optimization[key] = int(optimization[key])

    physical_enabled = bool(physical.setdefault("enabled", mode == "time"))
    if physical_enabled != (mode == "time"):
        raise ValueError(
            "physical.enabled must be false for video_only and true for time"
        )
    if physical_enabled:
        physical["stats_cache"] = _resolve_path(
            _require(physical, "stats_cache", "physical"), must_exist=True
        )
        physical.setdefault("head_type", "conv1x1")
        physical.setdefault("lambda_time", 0.1)
        physical.setdefault("lambda_warmup_steps", 250)
        physical.setdefault("smooth_l1_beta", 0.5)
        _require(physical, "targets", "physical")

    runtime["output_dir"] = _resolve_path(
        _require(runtime, "output_dir", "runtime"), must_exist=False
    )
    runtime.setdefault("num_workers", 4)
    runtime.setdefault("persistent_workers", True)
    runtime.setdefault("pin_memory", True)
    runtime.setdefault("save_every_epochs", 1)
    runtime.setdefault("log_every_updates", 10)
    runtime.setdefault("mixed_precision", "bf16")
    runtime.setdefault("expected_world_size", 8)
    runtime.setdefault("expected_global_batch_size", 64)
    if runtime["mixed_precision"] != "bf16":
        raise ValueError("The registered adaptation protocol uses BF16 autocast")

    if domain == "covla":
        expected_split = int(data.setdefault("expected_encoder_train_videos", 6305))
        if len(split_values) != expected_split:
            raise RuntimeError(
                f"COVLA encoder split has {len(split_values)} IDs, expected {expected_split}"
            )
    elif domain == "egoexo":
        expected_split = int(data.setdefault("expected_encoder_train_windows", 7000))
        if len(split_values) != expected_split:
            raise RuntimeError(
                f"EgoExo encoder split has {len(split_values)} windows, expected {expected_split}"
            )
        views = list(data.setdefault("video_views", ["view01", "view02", "view04"]))
        if views != ["view01", "view02", "view04"]:
            raise ValueError(
                "Registered EgoExo protocol requires seen views view01/view02/view04"
            )
    else:
        video_keys = list(_require(data, "video_keys", "data"))
        expected_keys = [
            "observation.images.top_head",
            "observation.images.hand_left",
            "observation.images.hand_right",
        ]
        if video_keys != expected_keys:
            raise ValueError(
                "Registered AgiBot protocol requires top_head, hand_left, hand_right"
            )
        total_samples = int(data.setdefault("encoder_train_num_samples", 15000))
        if total_samples != 15000:
            raise ValueError(
                "AgiBot formal iVideoGPT adaptation is fixed to 15,000 samples"
            )
        if total_samples % len(video_keys):
            raise ValueError("AgiBot sample count must be divisible by its view count")
        required_windows = total_samples // len(video_keys)
        if required_windows > len(split_values):
            raise RuntimeError(
                f"AgiBot needs {required_windows} split windows, only {len(split_values)} exist"
            )
        data["selected_encoder_train_windows"] = required_windows

    cfg["config_path"] = str(config_path)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt iVideoGPT with video-only or video+time objectives"
    )
    parser.add_argument("--config", "--fname", dest="config", required=True)
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="Validate the formal protocol and paths without loading data/model",
    )
    parser.add_argument(
        "--data-check",
        action="store_true",
        help="Construct the aligned dataset, report its manifest, and exit",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional smoke-test cap on optimizer updates",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (intended for isolated smoke tests)",
    )
    return parser


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = load_adapt_config(args.config)
    if args.epochs is not None:
        cfg["optimization"]["epochs"] = int(args.epochs)
    if args.per_device_batch_size is not None:
        cfg["optimization"]["per_device_batch_size"] = int(
            args.per_device_batch_size
        )
    if args.gradient_accumulation_steps is not None:
        cfg["optimization"]["gradient_accumulation_steps"] = int(
            args.gradient_accumulation_steps
        )
    if args.num_workers is not None:
        cfg["runtime"]["num_workers"] = int(args.num_workers)
    if args.output_dir is not None:
        cfg["runtime"]["output_dir"] = _resolve_path(
            args.output_dir, must_exist=False
        )
    args.resolved_config = cfg
    return args
