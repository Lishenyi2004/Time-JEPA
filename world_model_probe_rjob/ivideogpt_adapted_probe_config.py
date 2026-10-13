"""Config parser for frozen probes on adapted iVideoGPT checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_PROBE_DIR = REPO_ROOT / "covla_probe_rjob"
for path in (str(REPO_ROOT), str(SHARED_PROBE_DIR), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_config as shared_config  # noqa: E402

from ivideogpt_config import (  # noqa: E402
    _load_merged_yaml_defaults,
    _validate_complete_snapshot,
    build_arg_parser as build_zero_arg_parser,
)
from encoder_adapters.ivideogpt_adapted import resolve_adapted_transformer  # noqa: E402


def build_arg_parser():
    parser = build_zero_arg_parser()
    parser.add_argument(
        "--adapted-transformer-checkpoint",
        required=False,
        default=None,
        help="Formal epoch transformer/model.safetensors produced by adaptation",
    )
    parser.add_argument(
        "--adaptation-mode",
        choices=("video_only", "time"),
        default=None,
        help="Expected mode recorded in adaptation_manifest.json",
    )
    parser.add_argument("--adaptation-epoch", type=int, default=19)
    return parser


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
    if args.adapted_transformer_checkpoint is None:
        raise ValueError("adapted_transformer_checkpoint is required")
    if args.adaptation_mode is None:
        raise ValueError("adaptation_mode is required")

    base_checkpoint = Path(args.encoder_checkpoint).expanduser().resolve()
    _validate_complete_snapshot(base_checkpoint)
    args.encoder_checkpoint = str(base_checkpoint)

    _, weights_path, _, manifest = resolve_adapted_transformer(
        args.adapted_transformer_checkpoint
    )
    if manifest.get("adaptation_mode") != args.adaptation_mode:
        raise RuntimeError(
            f"Config mode {args.adaptation_mode} does not match checkpoint mode "
            f"{manifest.get('adaptation_mode')}"
        )
    args.adapted_transformer_checkpoint = str(weights_path)
    if args.tubelet_size != 2 or args.window_frames != 16 or args.context_frames != 2:
        raise ValueError("Adapted probes retain the registered 64f/16-window/2-context protocol")
    if args.encoder_dtype != "float32":
        raise ValueError("Formal adapted checkpoint extraction uses FP32")
    return args
