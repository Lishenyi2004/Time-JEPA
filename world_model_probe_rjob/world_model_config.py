"""Configuration layer for the isolated world-model probe runner.

The parser deliberately reuses the mature probe arguments, while adding a
small ``base_config`` inheritance mechanism.  World-model experiment files can
therefore inherit the exact current split/target/probe protocol and override
only encoder-specific settings.
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
except ImportError:  # pragma: no cover - caught with an actionable error below
    yaml = None


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "InternVideo/checkpoints/internvideo_next_large_p14_res224_f16/model.safetensors"
)
DEFAULT_VIDEOMAE_CHECKPOINT = (
    REPO_ROOT / "VideoMAEv2/checkpoints/mae-g/vit_g_hybrid_pt_1200e.pth"
)

ARCH_PROTOCOLS = {
    "internvideo_next_l": {
        "encoder_family": "internvideo_next",
        "checkpoint_key": "huggingface",
    },
    "videomaev2_g": {
        "encoder_family": "videomaev2",
        "checkpoint_key": "model",
    },
}


def _action(parser, dest):
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise KeyError(f"Parser has no action {dest!r}")


def build_arg_parser():
    parser = shared_config.build_arg_parser()

    arch_action = _action(parser, "arch")
    arch_action.choices = tuple(ARCH_PROTOCOLS)
    arch_action.default = "internvideo_next_l"
    arch_action.help = "Frozen world-model encoder exposed to the shared physical probe"

    family_action = _action(parser, "encoder_family")
    family_action.choices = ("internvideo_next", "videomaev2")
    family_action.default = "internvideo_next"
    family_action.help = "World-model encoder family used by this isolated runner"

    checkpoint_action = _action(parser, "encoder_checkpoint")
    checkpoint_action.default = None
    checkpoint_action.help = (
        "Complete local checkpoint file. If omitted, the selected family's pinned "
        "checkpoint path under this repository is used."
    )

    checkpoint_key_action = _action(parser, "checkpoint_key")
    checkpoint_key_action.choices = ("huggingface", "model")
    checkpoint_key_action.default = "huggingface"
    checkpoint_key_action.help = "Checkpoint format marker used for protocol validation"

    tubelet_action = _action(parser, "tubelet_size")
    tubelet_action.help = (
        "Physical-target/probe grouping size. Keep at 2 for 32 targets from 64 frames."
    )

    parser.add_argument(
        "--base-config",
        default=None,
        help=(
            "Optional existing probe YAML inherited before this config's overrides. "
            "Relative paths are resolved against the child config and then the repo root."
        ),
    )
    parser.add_argument(
        "--encoder-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Parameter dtype used to load the frozen encoder",
    )
    parser.add_argument(
        "--kernel-mode",
        choices=("auto", "native", "portable"),
        default="auto",
        help=(
            "auto uses all native fused kernels when installed, otherwise FlashAttention "
            "with equivalent PyTorch RMSNorm/MLP; native requires every optional fused extension"
        ),
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
        help=(
            "VideoMAE attention implementation. Full 64-frame extraction requires "
            "fused SDPA; eager is accepted only for the native-chunk diagnostic."
        ),
    )
    parser.add_argument(
        "--videomae-temporal-mode",
        choices=("full", "native_chunks"),
        default="full",
        help=(
            "full performs joint attention over all 64 frames; native_chunks runs four "
            "independent 16-frame clips and is a separately labelled fallback."
        ),
    )
    parser.add_argument(
        "--encoder-microbatch-size",
        type=int,
        default=1,
        help="Number of clips sent through the frozen VideoMAE encoder at once",
    )
    parser.add_argument(
        "--adapter-only-check",
        action="store_true",
        help="Load the local checkpoint, build/cache its position table, print metadata, and exit",
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

    default_checkpoints = {
        "internvideo_next": DEFAULT_CHECKPOINT,
        "videomaev2": DEFAULT_VIDEOMAE_CHECKPOINT,
    }
    checkpoint_value = args.encoder_checkpoint or default_checkpoints[args.encoder_family]
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint.is_file():
        checkpoint_description = (
            "the complete clean mae-g/vit_g_hybrid_pt_1200e.pth file"
            if args.encoder_family == "videomaev2"
            else "the local model.safetensors file"
        )
        raise FileNotFoundError(
            f"{args.encoder_family} encoder_checkpoint must be {checkpoint_description}; "
            f"not found: {checkpoint}"
        )
    args.encoder_checkpoint = str(checkpoint)
    expected_protocol = ARCH_PROTOCOLS[args.arch]
    for field in ("encoder_family", "checkpoint_key"):
        expected = expected_protocol[field]
        actual = getattr(args, field)
        if actual != expected:
            raise ValueError(
                f"arch={args.arch} requires {field}={expected}, got {actual}"
            )
    if args.tubelet_size != 2:
        raise ValueError(
            "The aligned physical protocol requires tubelet_size=2 as a target/probe "
            "grouping size."
        )
    if args.encoder_microbatch_size <= 0:
        raise ValueError("encoder_microbatch_size must be positive")
    if (
        args.encoder_family == "videomaev2"
        and args.videomae_temporal_mode == "full"
        and args.attention_implementation != "sdpa"
    ):
        raise ValueError(
            "VideoMAE full 64-frame mode requires attention_implementation=sdpa"
        )
    return args
