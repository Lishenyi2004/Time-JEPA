"""Configuration layer for the isolated iVideoGPT frozen-probe runner.

This is a separate parser, rather than an extension of ``world_model_config``,
so an iVideoGPT change cannot alter an already-running InternVideo experiment.
The YAML inheritance behavior intentionally matches the InternVideo runner and
lets each new config inherit an established zero-Time-JEPA probe protocol.
"""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for path in (str(REPO_ROOT), str(SHARED_PROBE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_config as shared_config  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover - converted to an actionable error below
    yaml = None


DEFAULT_SNAPSHOT = (
    REPO_ROOT / "iVideoGPT/checkpoints/ivideogpt-oxe-64-act-free-medium"
)
DEFAULT_CHECKPOINT = DEFAULT_SNAPSHOT / "transformer/model.safetensors"
REQUIRED_SNAPSHOT_FILES = (
    "tokenizer/config.json",
    "tokenizer/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/model.safetensors",
)


def _action(parser, dest):
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(f"Parser has no action {dest!r}")


def build_arg_parser():
    parser = shared_config.build_arg_parser()

    arch_action = _action(parser, "arch")
    arch_action.choices = ("ivideogpt_oxe_64_medium",)
    arch_action.default = "ivideogpt_oxe_64_medium"
    arch_action.help = "Pinned iVideoGPT OXE 64x64 action-free medium model"

    family_action = _action(parser, "encoder_family")
    family_action.choices = ("ivideogpt",)
    family_action.default = "ivideogpt"
    family_action.help = "iVideoGPT tokenizer plus causal transformer"

    checkpoint_action = _action(parser, "encoder_checkpoint")
    checkpoint_action.default = str(DEFAULT_CHECKPOINT)
    checkpoint_action.help = (
        "transformer/model.safetensors inside the complete local iVideoGPT snapshot"
    )

    checkpoint_key_action = _action(parser, "checkpoint_key")
    checkpoint_key_action.choices = ("huggingface",)
    checkpoint_key_action.default = "huggingface"
    checkpoint_key_action.help = "Checkpoint format marker; adapter loads the local snapshot"

    tubelet_action = _action(parser, "tubelet_size")
    tubelet_action.help = (
        "Physical target grouping only. Keep 2 for the existing 64-frame/32-step protocol."
    )

    parser.add_argument(
        "--base-config",
        default=None,
        help=(
            "Existing zero-probe YAML inherited before this config's overrides. "
            "Relative paths are resolved against the child config and then the repo root."
        ),
    )
    parser.add_argument(
        "--encoder-dtype",
        choices=("float32",),
        default="float32",
        help=(
            "The registered zero baseline uses the checkpoint-native FP32 tokenizer and "
            "transformer; lower-precision variants require a separate validation."
        ),
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
        help="Causal transformer attention implementation; SDPA is the formal default",
    )
    parser.add_argument(
        "--input-preprocess",
        choices=("direct", "center_crop", "letterbox"),
        default="direct",
        help=(
            "Spatial preprocessing before the checkpoint-native 64x64 tokenizer: "
            "direct square resize (registered baseline), aspect-preserving center "
            "crop, or aspect-preserving letterbox."
        ),
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=16,
        help="Native iVideoGPT segment length; the registered checkpoint requires 16",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=2,
        help="Native full-frame context length; the registered checkpoint requires 2",
    )
    parser.add_argument(
        "--adapter-only-check",
        action="store_true",
        help="Load tokenizer and transformer, print the frozen-state manifest, and exit",
    )
    return parser


def _read_yaml_mapping(config_path):
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config, but import yaml failed")
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"YAML config must be a mapping: {config_path}")
    return config


def _training_mapping(config, config_path):
    if "train" in config:
        train_config = config.get("train") or {}
    else:
        train_config = {
            key: value
            for key, value in config.items()
            if key not in ("rjob", "runtime", "base_config")
        }
    if not isinstance(train_config, dict):
        raise RuntimeError(f"YAML field 'train' must be a mapping: {config_path}")
    return {str(key).replace("-", "_"): value for key, value in train_config.items()}


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


def _validate_complete_snapshot(checkpoint):
    if checkpoint.name != "model.safetensors" or checkpoint.parent.name != "transformer":
        raise ValueError(
            "iVideoGPT encoder_checkpoint must end in transformer/model.safetensors; "
            f"got {checkpoint}"
        )
    snapshot_root = checkpoint.parent.parent
    missing = [
        relative_path
        for relative_path in REQUIRED_SNAPSHOT_FILES
        if not (snapshot_root / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete iVideoGPT snapshot at {snapshot_root}; missing files: {missing}"
        )
    return snapshot_root


def parse_args(argv=None):
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

    checkpoint = Path(args.encoder_checkpoint).expanduser().resolve()
    _validate_complete_snapshot(checkpoint)
    args.encoder_checkpoint = str(checkpoint)

    if args.tubelet_size != 2:
        raise ValueError("The aligned physical protocol requires tubelet_size=2")
    if args.window_frames != 16:
        raise ValueError("The pinned iVideoGPT model requires window_frames=16")
    if args.context_frames != 2:
        raise ValueError("The pinned iVideoGPT tokenizer requires context_frames=2")
    if args.encoder_dtype != "float32":
        raise ValueError("The registered iVideoGPT zero baseline requires encoder_dtype=float32")
    return args
