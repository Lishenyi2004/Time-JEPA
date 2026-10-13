"""Configuration layer for PSG-JEPA probes built on the shared protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for path in (str(REPO_ROOT), str(SHARED_PROBE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_config as shared_config  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_UPSTREAM_ROOT = Path("third_party/PSG-JEPA")
DEFAULT_STATIC_CHECKPOINT = Path(
    ""
    "outputs_external/psg_jepa/agibot_static_seed3072/official_model_final_object.ckpt"
)


def _action(parser, dest):
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(f"Parser has no action {dest!r}")


def build_arg_parser():
    parser = shared_config.build_arg_parser()

    arch_action = _action(parser, "arch")
    arch_action.choices = ("psg_jepa_vit_tiny", "psg_jepa_vitg")
    arch_action.default = "psg_jepa_vit_tiny"

    family_action = _action(parser, "encoder_family")
    family_action.choices = ("psg_jepa",)
    family_action.default = "psg_jepa"

    checkpoint_action = _action(parser, "encoder_checkpoint")
    checkpoint_action.default = str(DEFAULT_STATIC_CHECKPOINT)
    checkpoint_action.help = "Complete official CrossModalJEPA_v2 model object produced by training"

    checkpoint_key_action = _action(parser, "checkpoint_key")
    checkpoint_key_action.choices = (
        "official_model_object",
        "exported_encoder_projector",
    )
    checkpoint_key_action.default = "official_model_object"

    parser.add_argument(
        "--base-config",
        default=None,
        help="Existing shared probe YAML inherited before PSG-specific overrides",
    )
    parser.add_argument(
        "--upstream-root",
        default=str(DEFAULT_UPSTREAM_ROOT),
        help="Pinned checkout of the authors' PSG-JEPA repository",
    )
    parser.add_argument(
        "--training-variant",
        choices=("psg_static", "psg_full", "lewm"),
        default="psg_static",
        help="Expected training objective, checked against checkpoint contents/metadata",
    )
    parser.add_argument(
        "--frame-microbatch-size",
        type=int,
        default=128,
        help="Number of independent RGB frames per frozen 2-D ViT forward",
    )
    parser.add_argument(
        "--state-stats-path",
        default=(
            "time-jepa/cache/"
            "agibot_3view_top_head_hand_left_hand_right_"
            "wholebody_cartesian_v2_64f5fps_stats.json"
        ),
        help="Encoder-train 18-D normalization used by the original PSG state head",
    )
    parser.add_argument(
        "--health-output-dir",
        default=None,
        help="Output directory for the original state-head diagnostic",
    )
    parser.add_argument(
        "--adapter-only-check",
        action="store_true",
        help="Load and audit the exact official model object, then exit",
    )
    return parser


def _read_yaml_mapping(config_path):
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config")
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"YAML config must be a mapping: {config_path}")
    return config


def _training_mapping(config, config_path):
    if "train" in config:
        values = config.get("train") or {}
    else:
        values = {
            key: value
            for key, value in config.items()
            if key not in ("rjob", "runtime", "base_config")
        }
    if not isinstance(values, dict):
        raise RuntimeError(f"YAML field 'train' must be a mapping: {config_path}")
    return {str(key).replace("-", "_"): value for key, value in values.items()}


def _resolve_base_config(base_value, child_path):
    candidate = Path(base_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    child_relative = (child_path.parent / candidate).resolve()
    if child_relative.is_file():
        return child_relative
    return (REPO_ROOT / candidate).resolve()


def _load_merged_yaml_defaults(config_path, stack=()):
    config_path = Path(config_path).expanduser().resolve()
    if config_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, config_path))
        raise RuntimeError(f"Cyclic base_config inheritance: {cycle}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    config = _read_yaml_mapping(config_path)
    defaults = {}
    resolved_base = None
    if config.get("base_config"):
        resolved_base = _resolve_base_config(config["base_config"], config_path)
        defaults.update(
            _load_merged_yaml_defaults(resolved_base, stack=(*stack, config_path))[0]
        )
    defaults.update(_training_mapping(config, config_path))
    if resolved_base is not None:
        defaults["base_config"] = str(resolved_base)
    return defaults, resolved_base


def parse_args(argv=None, *, require_checkpoint=True):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "--fname", dest="config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = build_arg_parser()
    if pre_args.config:
        yaml_defaults, _ = _load_merged_yaml_defaults(pre_args.config)
        yaml_defaults = shared_config._coerce_yaml_defaults(
            parser, yaml_defaults, pre_args.config
        )
        parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(argv)

    if args.config is not None:
        args.config = str(Path(args.config).expanduser().resolve())
    args.upstream_root = str(Path(args.upstream_root).expanduser().resolve())
    checkpoint = Path(args.encoder_checkpoint).expanduser().resolve()
    if require_checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(
            "PSG encoder_checkpoint must be the complete official model object; "
            f"not found: {checkpoint}"
        )
    args.encoder_checkpoint = str(checkpoint)
    args.state_stats_path = str(Path(args.state_stats_path).expanduser().resolve())

    required = {
        "encoder_family": "psg_jepa",
        "dataset": "agibot",
        "tubelet_size": 2,
    }
    for field, expected in required.items():
        actual = getattr(args, field)
        if actual != expected:
            raise ValueError(f"PSG probe requires {field}={expected!r}, got {actual!r}")
    if args.arch == "psg_jepa_vit_tiny":
        if args.checkpoint_key != "official_model_object":
            raise ValueError(
                "ViT-Tiny PSG probes require checkpoint_key='official_model_object'"
            )
        if args.training_variant not in ("psg_static", "lewm"):
            raise ValueError(
                "ViT-Tiny PSG probes only support training_variant psg_static/lewm"
            )
        if args.target_mode != "state":
            raise ValueError("Legacy ViT-Tiny PSG probes only support target_mode='state'")
    elif args.arch == "psg_jepa_vitg":
        if args.checkpoint_key != "exported_encoder_projector":
            raise ValueError(
                "ViT-g PSG probes require checkpoint_key='exported_encoder_projector'"
            )
        if args.training_variant not in ("psg_static", "psg_full"):
            raise ValueError(
                "ViT-g PSG probes only support training_variant psg_static/psg_full"
            )
        if args.target_mode not in ("state", "future_state"):
            raise ValueError(
                "ViT-g PSG probes only support target_mode state/future_state"
            )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported PSG architecture: {args.arch!r}")
    if args.frame_microbatch_size <= 0:
        raise ValueError("frame_microbatch_size must be positive")
    return args
