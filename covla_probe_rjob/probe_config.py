"""Argument parser and YAML config handling."""

import argparse
from pathlib import Path

import torch

try:
    import yaml
except ImportError:
    yaml = None

from probe_constants import ARCH_SPECS


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "--fname",
        dest="config",
        default=None,
        help="YAML config file. Values under top-level 'train:' become argparse defaults.",
    )
    parser.add_argument(
        "--covla-root",
        default="data/CoVLA-Dataset",
    )
    parser.add_argument(
        "--dataset",
        choices=("covla", "agibot", "egoexo"),
        default="covla",
        help="In-domain probe dataset. Use --ood kitti for KITTI; EgoExo uses the dedicated pose-target path.",
    )
    parser.add_argument(
        "--ood",
        choices=("none", "kitti"),
        default="none",
        help="Use an OOD probe dataset instead of CoVLA. 'kitti' reads kitti_<split>.json files.",
    )
    parser.add_argument(
        "--kitti-root",
        default="data/KITTI/raw",
        help="KITTI Raw root, used for logging and future relative split support.",
    )
    parser.add_argument(
        "--agibot-root",
        default="data/agibot",
    )
    parser.add_argument("--agibot-tasks", default="3641,3777")
    parser.add_argument("--agibot-video-key", default="observation.images.top_head")
    parser.add_argument(
        "--agibot-train-video-keys",
        default=None,
        help=(
            "Comma-separated AgiBot views used for probe training. Defaults to "
            "--agibot-video-key. Multiple views are expanded as additional rows "
            "on the same camera-agnostic split keys."
        ),
    )
    parser.add_argument(
        "--agibot-seen-video-keys",
        default="",
        help=(
            "Comma-separated AgiBot train-seen non-ID views evaluated on the test "
            "split and reported separately from held-out OOD views."
        ),
    )
    parser.add_argument(
        "--agibot-ood-video-keys",
        default="observation.images.hand_left,observation.images.hand_right",
        help="Comma-separated held-out AgiBot views evaluated after ID test; empty disables OOD view eval.",
    )
    parser.add_argument(
        "--agibot-shot-ood-protocol",
        choices=("full_8d", "same_side"),
        default="full_8d",
        help=(
            "Target aggregation for AgiBot current-state bilateral Shot OOD. "
            "same_side pairs head_left_fisheye with left-arm targets and "
            "head_right_fisheye with right-arm targets while retaining full/cross-side "
            "hard diagnostics. Readout and future-state configs must not enable "
            "same_side."
        ),
    )
    parser.add_argument(
        "--agibot-train-num-samples",
        type=int,
        default=0,
        help=(
            "Maximum expanded AgiBot probe train rows after multi-view split expansion. "
            "0 keeps all rows. Sampling is grouped by window/episode key to avoid "
            "partial-view leakage."
        ),
    )
    parser.add_argument(
        "--agibot-train-sample-seed",
        type=int,
        default=None,
        help="Seed for --agibot-train-num-samples; defaults to --seed.",
    )
    parser.add_argument(
        "--agibot-val-num-samples",
        type=int,
        default=0,
        help=(
            "Maximum expanded AgiBot probe validation rows after multi-view split "
            "expansion. 0 keeps all rows. Sampling is grouped by window/episode key."
        ),
    )
    parser.add_argument(
        "--agibot-val-sample-seed",
        type=int,
        default=None,
        help="Seed for --agibot-val-num-samples; defaults to --agibot-train-sample-seed + 1.",
    )
    parser.add_argument("--agibot-use-meta-episodes-only", action="store_true", default=True)
    parser.add_argument("--agibot-include-extra-episodes", dest="agibot_use_meta_episodes_only", action="store_false")
    parser.add_argument("--agibot-split-ratios", default="70,15,5,10")
    parser.add_argument(
        "--agibot-split-dir",
        default=None,
        help=(
            "Directory with agibot_<split>.json episode-key files (from "
            "make_agibot_splits.py). When set, the probe uses these persisted "
            "splits instead of on-the-fly shuffling, sharing the exact "
            "episode protocol with the encoder (no leakage, reproducible)."
        ),
    )
    parser.add_argument("--agibot-split-prefix", default="agibot_top_head")
    parser.add_argument("--agibot-max-time-delta", type=float, default=0.05)
    parser.add_argument("--agibot-video-backend", choices=("auto", "decord", "ffmpeg"), default="ffmpeg")
    parser.add_argument("--agibot-ffmpeg-timeout", type=float, default=120.0)
    parser.add_argument("--agibot-fps", type=float, default=5.0)
    parser.add_argument("--agibot-slice-stride-seconds", type=float, default=12.8)
    parser.add_argument("--egoexo-root", default="data/EgoExo4D_minimal")
    parser.add_argument(
        "--egoexo-video-view",
        default="best_exo",
        help="Single EgoExo ID view used for the main test metric.",
    )
    parser.add_argument(
        "--egoexo-train-video-keys",
        default=None,
        help=(
            "Comma-separated EgoExo views used during probe train/validation. "
            "One candidate is sampled per window, so multiple views do not multiply the split size. "
            "Defaults to --egoexo-video-view."
        ),
    )
    parser.add_argument(
        "--egoexo-seen-video-keys",
        default="",
        help="Comma-separated train-seen EgoExo views evaluated separately on the test split.",
    )
    parser.add_argument("--egoexo-ood-video-keys", default="aria01_214-1")
    parser.add_argument("--egoexo-split-dir", default="time-jepa/splits")
    parser.add_argument("--egoexo-split-prefix", default="egoexo_bestexo")
    parser.add_argument("--egoexo-manifest-path", default=None)
    parser.add_argument("--egoexo-fps", type=float, default=5.0)
    parser.add_argument("--egoexo-slice-stride-seconds", type=float, default=12.8)
    parser.add_argument("--egoexo-max-time-delta", type=float, default=0.05)
    parser.add_argument("--egoexo-pose-fps", type=float, default=30.0)
    parser.add_argument("--egoexo-min-views", type=int, default=2)
    parser.add_argument("--egoexo-smooth-window", type=int, default=5)
    parser.add_argument("--egoexo-video-backend", choices=("auto", "decord", "ffmpeg"), default="auto")
    parser.add_argument("--egoexo-ffmpeg-timeout", type=float, default=120.0)
    parser.add_argument(
        "--egoexo-stats-cache",
        default="time-jepa/cache/egoexo_bestexo_whole_body_compact_v2_64f5fps_stats.json",
        help=(
            "EgoExo target-normalization cache. The default follows the current 9D "
            "readout panel; shot configs must override it with their right-body cache."
        ),
    )
    parser.add_argument("--arch", choices=ARCH_SPECS.keys(), default="vitg_384")
    parser.add_argument("--encoder-checkpoint", default="vjepa2_1_vitg_384.pt")
    parser.add_argument(
        "--encoder-family",
        choices=("vjepa2_1", "legacy"),
        default="vjepa2_1",
        help="Use app.vjepa_2_1 encoder definitions for newly fine-tuned V-JEPA 2.1 checkpoints; legacy keeps the older src.models encoder.",
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("auto", "encoder", "target_encoder", "ema_encoder"),
        default="auto",
        help="Encoder state to load from checkpoint. auto prefers target_encoder, then encoder, then ema_encoder.",
    )
    parser.add_argument("--output-dir", default="time-jepa/outputs/covla_probe_vitg384")
    parser.add_argument(
        "--target-mode",
        choices=("state", "future_state", "trajectory_xy"),
        default="state",
        help=(
            "Target construction mode. 'state' reads scalar state fields from --targets; "
            "'future_state' predicts one or more states strictly after the input clip at "
            "--future-horizons-seconds (falling back to the legacy single "
            "--future-horizon-seconds); "
            "'trajectory_xy' reads CoVLA states/<video_id>.jsonl trajectory points and "
            "uses their x/y coordinates as a two-output readout target."
        ),
    )
    parser.add_argument(
        "--future-horizon-seconds",
        type=float,
        default=1.0,
        help=(
            "Legacy single forecast horizon measured from the final observed video "
            "frame. Used when --future-horizons-seconds is omitted; otherwise the "
            "largest configured horizon is retained here for compatibility."
        ),
    )
    parser.add_argument(
        "--future-horizons-seconds",
        default=None,
        help=(
            "Strictly increasing comma-separated forecast horizons measured from "
            "the final observed frame, for example 0.2,0.4,0.6,0.8,1.0. YAML "
            "configs may provide a list. When omitted, the legacy scalar "
            "--future-horizon-seconds is used."
        ),
    )
    parser.add_argument("--targets", default="vEgo,aEgo,yaw_rate")
    parser.add_argument(
        "--trajectory-point-index",
        type=int,
        default=-1,
        help=(
            "Single trajectory point used by --target-mode trajectory_xy when "
            "--trajectory-point-indices is not set. Negative values follow Python "
            "indexing; -1 is the final provided future trajectory point."
        ),
    )
    parser.add_argument(
        "--trajectory-point-indices",
        default=None,
        help=(
            "Comma-separated or YAML-list trajectory point indices used by --target-mode "
            "trajectory_xy. Example: 19,39,59 produces six outputs: x/y at each point."
        ),
    )
    parser.add_argument(
        "--probe-type",
        choices=(
            "official_ssv2",
            "attentive",
            "linear",
            "conv1x1",
            "future_conv1x1",
            "temporal_aligned",
            "temporal",
        ),
        default="official_ssv2",
        help=(
            "Probe for current physical readout. official_ssv2/attentive use the V-JEPA "
            "attentive pooling sequence head; temporal_aligned matches conv1x1 except that both "
            "Conv1d kernels are size 3; temporal keeps the legacy deeper temporal head. "
            "future_conv1x1 reads only the final contextualized temporal token and emits "
            "one state vector per configured post-clip horizon. "
            "Diving48-style multilevel and EK100-style anticipation "
            "require different encoder wrappers and are not folded into this current-readout script."
        ),
    )
    parser.add_argument("--probe-depth", type=int, default=4, help="Number of attentive probe blocks")
    parser.add_argument(
        "--probe-hidden-dim",
        type=int,
        default=0,
        help="Hidden dim for conv1x1/temporal_aligned/temporal probes; 0 uses encoder dim",
    )
    parser.add_argument(
        "--tubelet-size",
        type=int,
        default=2,
        help="Frozen encoder temporal tubelet size; current official V-JEPA checkpoint/model path keeps this at 2",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak probe LR")
    parser.add_argument("--start-lr", type=float, default=None, help="Warmup start LR; defaults to --lr")
    parser.add_argument("--final-lr", type=float, default=0.0, help="Final LR for cosine schedule")
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--final-weight-decay", type=float, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-dir", default="time-jepa/splits", help="Directory containing covla_<split>.json files")
    parser.add_argument("--train-split", default="train", help="Split name used to train the probe")
    parser.add_argument("--val-split", default="val", help="Split name used to select the best probe checkpoint")
    parser.add_argument("--test-split", default="test", help="Split name used for final metrics after training")
    parser.add_argument(
        "--covla-ood-test-split",
        default=None,
        help=(
            "Optional additional covla_<split>.json evaluated after the in-domain "
            "test split with the same frozen encoder, probe, and train-set target "
            "normalization. This is intended for within-CoVLA OOD protocols."
        ),
    )
    parser.add_argument("--max-train-videos", type=int, default=0)
    parser.add_argument("--max-val-videos", type=int, default=0)
    parser.add_argument("--max-test-videos", type=int, default=0, help="Maximum videos in each ID/CoVLA OOD test split; 0 keeps all videos")
    parser.add_argument("--frame-span", type=int, default=128, help="Approximate raw-frame span for one sampled clip when --frame-stride is 0")
    parser.add_argument("--frame-stride", type=int, default=0, help="Raw-frame interval between supervised input frames; 0 derives it from --frame-span")
    parser.add_argument("--video-num-threads", type=int, default=1, help="Number of decord threads per video reader")
    parser.add_argument("--keep-state-cache", action="store_true", help="Keep parsed state jsonl cache across epochs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-bfloat16", action="store_true")
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard log directory; defaults to <output-dir>/tensorboard")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging")
    parser.add_argument(
        "--estimated-persistence-checkpoint",
        default=None,
        help=(
            "Optional current-state probe checkpoint used to form a fair, video-only "
            "estimated persistence baseline for future_state evaluation. The probe "
            "must decode the same target dimensions as the future probe; its final "
            "temporal prediction is copied to every future horizon."
        ),
    )
    parser.add_argument(
        "--probe-checkpoint",
        default=None,
        help=(
            "Existing probe checkpoint to evaluate. With --eval-only this skips "
            "probe training and reuses the saved future-state probe."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate --probe-checkpoint without training a new probe.",
    )
    return parser


def _load_yaml_defaults(config_path):
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config, but import yaml failed")
    config_path = Path(config_path).expanduser()
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f"YAML config must be a mapping: {config_path}")

    if "train" in config:
        train_config = config.get("train") or {}
    else:
        train_config = {
            key: value
            for key, value in config.items()
            if key not in ("rjob", "runtime")
        }
    if not isinstance(train_config, dict):
        raise RuntimeError(f"YAML field 'train' must be a mapping: {config_path}")

    return {str(key).replace("-", "_"): value for key, value in train_config.items()}


def _parse_yaml_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _coerce_yaml_defaults(parser, defaults, config_path):
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in (argparse.SUPPRESS, "help")
    }
    unknown = sorted(key for key in defaults if key not in actions)
    if unknown:
        raise RuntimeError(
            f"Unknown train config key(s) in {config_path}: {unknown}. "
            "Use argparse destination names such as probe_type, batch_size, output_dir."
        )

    coerced = {}
    for key, value in defaults.items():
        action = actions[key]
        if key in (
            "targets",
            "trajectory_point_indices",
            "future_horizons_seconds",
            "agibot_tasks",
            "agibot_split_ratios",
            "agibot_train_video_keys",
            "agibot_seen_video_keys",
            "agibot_ood_video_keys",
            "egoexo_train_video_keys",
            "egoexo_seen_video_keys",
            "egoexo_ood_video_keys",
        ) and isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            value = _parse_yaml_bool(value)
        elif action.type is not None and value is not None:
            try:
                value = action.type(value)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid value for '{key}' in {config_path}: {value!r}"
                ) from exc

        if action.choices is not None and value is not None and value not in action.choices:
            raise RuntimeError(
                f"Invalid choice for '{key}' in {config_path}: {value!r}; "
                f"expected one of {list(action.choices)}"
            )
        coerced[key] = value
    return coerced


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "--fname", dest="config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = build_arg_parser()
    if pre_args.config:
        yaml_defaults = _load_yaml_defaults(pre_args.config)
        yaml_defaults = _coerce_yaml_defaults(parser, yaml_defaults, pre_args.config)
        parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(argv)
    if args.config is not None:
        args.config = str(Path(args.config).expanduser())
    return args
