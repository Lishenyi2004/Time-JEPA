"""Train a CoVLA probe on top of a frozen official V-JEPA encoder.

Default task: multi-output regression for vEgo, aEgo, yaw_rate.
This is the first formal CoVLA baseline after the official-probe sanity check:

    official V-JEPA encoder frozen + official attentive probe architecture
    randomly initialized + CoVLA labels.

The implementation is split across sibling ``probe_*`` modules:

    probe_paths      repo-root sys.path bootstrap
    probe_constants  arch specs, target aliases, ImageNet stats
    probe_utils      distributed / seeding / memory / IO helpers
    probe_targets    target-field normalization + state/OXTS series loading
    probe_sampling   frame sampling + video<->state alignment + tubelet targets
    probe_transforms clip preprocessing transform
    probe_datasets   metadata/splits + CoVLA/KITTI Dataset classes
    probe_stats      target normalization statistics
    probe_models     probe heads + build_probe factory
    probe_optim      optimizer + LR/WD schedules
    probe_engine     encoder loading, loss, evaluation, checkpointing
    probe_config     argument parser + YAML config
"""

import json
import random
from types import SimpleNamespace
from pathlib import Path

import probe_paths  # noqa: F401  (ensures REPO_ROOT is on sys.path before app/src imports)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from probe_config import parse_args
from probe_constants import ARCH_SPECS
from probe_datasets import (
    AgiBotProbeDataset,
    EgoExoProbeDataset,
    CoVLAProbeDataset,
    KITTIProbeDataset,
    load_kitti_split_rows,
    load_egoexo_split_rows,
    read_agibot_metadata,
    read_metadata,
    split_agibot_rows,
    split_agibot_rows_from_files,
    split_rows,
    split_rows_from_files,
)
from probe_engine import evaluate, load_encoder, masked_mse, save_checkpoint
from probe_models import build_probe
from probe_optim import build_optimizer_and_schedulers
from probe_sampling import (
    future_row_is_eligible,
    normalize_future_horizons_seconds,
)
from probe_stats import compute_or_broadcast_stats
from probe_targets import (
    agibot_target_display_name,
    egoexo_target_display_name,
    kitti_target_display_name,
    normalize_agibot_target_fields,
    normalize_egoexo_target_fields,
    normalize_kitti_target_field,
    normalize_target_field,
    target_display_name,
)
from probe_transforms import build_transform
from probe_utils import (
    build_summary_writer,
    cleanup_distributed,
    init_distributed,
    memory_report,
    print_main,
    reduce_sum,
    release_epoch_memory,
    seed_everything,
    seed_worker,
    torch_load,
    tqdm,
    unwrap_model,
)
from probe_view_metrics import (
    build_agibot_shot_ood_report,
    macro_average_view_metrics,
    validate_agibot_same_side_ood,
)


def parse_trajectory_point_indices(args):
    if args.trajectory_point_indices:
        return tuple(
            int(index_text.strip())
            for index_text in str(args.trajectory_point_indices).split(",")
            if index_text.strip()
        )
    return (int(args.trajectory_point_index),)


def trajectory_target_names(point_indices):
    return [
        f"trajectory_{point_index}_{axis}"
        for point_index in point_indices
        for axis in ("x", "y")
    ]


def dataset_kwargs(args, dataset_name):
    common = {
        "future_horizon_seconds": args.future_horizon_seconds,
        "future_horizons_seconds": args.future_horizons_seconds,
    }
    if dataset_name == "egoexo":
        return {
            **common,
            "egoexo_fps": args.egoexo_fps,
            "egoexo_max_time_delta": args.egoexo_max_time_delta,
            "egoexo_pose_fps": args.egoexo_pose_fps,
            "egoexo_min_views": args.egoexo_min_views,
            "egoexo_smooth_window": args.egoexo_smooth_window,
            "egoexo_video_backend": args.egoexo_video_backend,
            "egoexo_ffmpeg_timeout": args.egoexo_ffmpeg_timeout,
        }
    if dataset_name != "agibot":
        return common if dataset_name == "covla" else {}
    return {
        **common,
        "agibot_fps": args.agibot_fps,
        "agibot_max_time_delta": args.agibot_max_time_delta,
        "agibot_video_backend": args.agibot_video_backend,
        "agibot_ffmpeg_timeout": args.agibot_ffmpeg_timeout,
        "agibot_slice_stride_seconds": args.agibot_slice_stride_seconds,
    }


def parse_csv_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_checkpoint_targets(dataset_name, targets):
    requested = parse_csv_list(targets)
    if dataset_name == "agibot":
        return normalize_agibot_target_fields(requested)
    if dataset_name == "egoexo":
        return normalize_egoexo_target_fields(requested)
    return [normalize_target_field(target) for target in requested]


def _resolved_path(path):
    return str(Path(path).expanduser().resolve()) if path else None


def load_future_probe_checkpoint(
    checkpoint_path,
    args,
    dataset_name,
    target_fields,
    future_horizons_seconds,
):
    """Load and validate a trained future probe before evaluation-only setup."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"future probe checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    saved_args = checkpoint.get("args") or {}
    if not isinstance(saved_args, dict):
        raise RuntimeError(f"future probe checkpoint has no argparse metadata: {checkpoint_path}")

    expected = {
        "dataset": dataset_name,
        "target_mode": "future_state",
        "probe_type": "future_conv1x1",
        "arch": args.arch,
        "tubelet_size": args.tubelet_size,
    }
    for key, expected_value in expected.items():
        actual = saved_args.get(key)
        if actual != expected_value:
            raise ValueError(
                f"future probe checkpoint {key} mismatch: "
                f"checkpoint={actual!r}, evaluation={expected_value!r}"
            )
    for key in ("probe_depth", "probe_hidden_dim"):
        if saved_args.get(key) != getattr(args, key):
            raise ValueError(
                f"future probe checkpoint {key} mismatch: "
                f"checkpoint={saved_args.get(key)!r}, evaluation={getattr(args, key)!r}"
            )
    saved_targets = normalize_checkpoint_targets(dataset_name, saved_args.get("targets"))
    if saved_targets != list(target_fields):
        raise ValueError(
            "future probe checkpoint targets do not match the evaluation config: "
            f"checkpoint={saved_targets}, evaluation={list(target_fields)}"
        )
    saved_encoder = _resolved_path(saved_args.get("encoder_checkpoint"))
    eval_encoder = _resolved_path(args.encoder_checkpoint)
    if saved_encoder != eval_encoder:
        raise ValueError(
            "future probe checkpoint encoder does not match the evaluation config: "
            f"checkpoint={saved_encoder}, evaluation={eval_encoder}"
        )
    saved_horizons = normalize_future_horizons_seconds(
        saved_args.get("future_horizons_seconds"),
        saved_args.get("future_horizon_seconds", 1.0),
    )
    if saved_horizons != tuple(float(value) for value in future_horizons_seconds):
        raise ValueError(
            "future probe checkpoint horizons do not match the evaluation config: "
            f"checkpoint={saved_horizons}, evaluation={tuple(future_horizons_seconds)}"
        )
    if checkpoint.get("probe") is None:
        raise KeyError(f"future probe checkpoint lacks 'probe': {checkpoint_path}")
    if checkpoint.get("target_mean") is None or checkpoint.get("target_std") is None:
        raise KeyError(f"future probe checkpoint lacks target_mean/target_std: {checkpoint_path}")
    target_mean = torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32)
    target_std = torch.as_tensor(checkpoint["target_std"], dtype=torch.float32)
    expected_shape = (len(future_horizons_seconds), len(target_fields))
    if tuple(target_mean.shape) != expected_shape or tuple(target_std.shape) != expected_shape:
        raise ValueError(
            "future probe normalization shape does not match its configured horizons/targets: "
            f"mean={tuple(target_mean.shape)}, std={tuple(target_std.shape)}, "
            f"expected={expected_shape}"
        )
    if not torch.isfinite(target_mean).all() or not torch.isfinite(target_std).all():
        raise ValueError(f"future probe normalization contains non-finite values: {checkpoint_path}")
    if (target_std <= 0).any():
        raise ValueError(f"future probe normalization std must be positive: {checkpoint_path}")
    return checkpoint, target_mean, target_std


def load_current_probe_checkpoint(
    checkpoint_path,
    args,
    dataset_name,
    target_fields,
):
    """Load a current-state probe for eval-only runs.

    Current-state probes use one-dimensional target statistics, so validate
    their metadata explicitly and then reuse the ordinary evaluation path.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"current-state probe checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    saved_args = checkpoint.get("args") or {}
    if not isinstance(saved_args, dict):
        raise RuntimeError(
            f"current-state probe checkpoint has no argparse metadata: {checkpoint_path}"
        )
    expected = {
        "dataset": dataset_name,
        "target_mode": "state",
        "arch": args.arch,
        "tubelet_size": args.tubelet_size,
        "probe_type": args.probe_type,
    }
    for key, expected_value in expected.items():
        actual = saved_args.get(key)
        if actual != expected_value:
            raise ValueError(
                f"current-state probe checkpoint {key} mismatch: "
                f"checkpoint={actual!r}, evaluation={expected_value!r}"
            )
    for key in ("probe_depth", "probe_hidden_dim"):
        if saved_args.get(key) != getattr(args, key):
            raise ValueError(
                f"current-state probe checkpoint {key} mismatch: "
                f"checkpoint={saved_args.get(key)!r}, evaluation={getattr(args, key)!r}"
            )
    saved_targets = normalize_checkpoint_targets(dataset_name, saved_args.get("targets"))
    if saved_targets != list(target_fields):
        raise ValueError(
            "current-state probe checkpoint targets do not match the evaluation config: "
            f"checkpoint={saved_targets}, evaluation={list(target_fields)}"
        )
    saved_encoder = _resolved_path(saved_args.get("encoder_checkpoint"))
    eval_encoder = _resolved_path(args.encoder_checkpoint)
    if saved_encoder != eval_encoder:
        raise ValueError(
            "current-state probe checkpoint encoder does not match the evaluation config: "
            f"checkpoint={saved_encoder}, evaluation={eval_encoder}"
        )
    if checkpoint.get("probe") is None:
        raise KeyError(f"current-state probe checkpoint lacks 'probe': {checkpoint_path}")
    if checkpoint.get("target_mean") is None or checkpoint.get("target_std") is None:
        raise KeyError(
            f"current-state probe checkpoint lacks target_mean/target_std: {checkpoint_path}"
        )
    target_mean = torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32)
    target_std = torch.as_tensor(checkpoint["target_std"], dtype=torch.float32)
    expected_shape = (len(target_fields),)
    if tuple(target_mean.shape) != expected_shape or tuple(target_std.shape) != expected_shape:
        raise ValueError(
            "current-state probe normalization shape does not match targets: "
            f"mean={tuple(target_mean.shape)}, std={tuple(target_std.shape)}, "
            f"expected={expected_shape}"
        )
    if not torch.isfinite(target_mean).all() or not torch.isfinite(target_std).all():
        raise ValueError(
            f"current-state probe normalization contains non-finite values: {checkpoint_path}"
        )
    if (target_std <= 0).any():
        raise ValueError(
            f"current-state probe normalization std must be positive: {checkpoint_path}"
        )
    return checkpoint, target_mean, target_std


def validate_estimated_persistence_protocol(current_checkpoint, future_checkpoint, args):
    """Require both heads and the evaluation loader to use the same input clips."""
    current_args = current_checkpoint.get("args") or {}
    future_args = (
        future_checkpoint.get("args") or {}
        if future_checkpoint is not None
        else vars(args)
    )
    keys = ["frame_span", "frame_stride", "tubelet_size"]
    dataset_keys = {
        "covla": ("frame_span", "frame_stride"),
        "agibot": (
            "agibot_fps",
            "agibot_slice_stride_seconds",
            "agibot_max_time_delta",
            "agibot_video_backend",
        ),
        "egoexo": (
            "egoexo_fps",
            "egoexo_slice_stride_seconds",
            "egoexo_max_time_delta",
            "egoexo_pose_fps",
            "egoexo_min_views",
            "egoexo_smooth_window",
            "egoexo_video_backend",
        ),
    }
    keys.extend(dataset_keys.get(args.dataset, ()))
    mismatches = {}
    for key in dict.fromkeys(keys):
        eval_value = getattr(args, key, None)
        current_value = current_args.get(key)
        future_value = future_args.get(key)
        if current_value != future_value or future_value != eval_value:
            mismatches[key] = {
                "current_probe": current_value,
                "future_probe": future_value,
                "evaluation": eval_value,
            }
    if mismatches:
        raise ValueError(
            "estimated persistence and future probes must use the same input clip "
            f"sampling protocol; mismatches={mismatches}"
        )


def load_estimated_persistence_probe(
    checkpoint_path,
    args,
    arch,
    num_outputs,
    device,
    future_checkpoint,
):
    """Load a frozen current-state probe used for estimated persistence.

    A future-state probe receives a video clip and predicts states after the
    clip.  The fair persistence comparator first decodes the current state from
    that same clip with a separately trained current-state probe, then repeats
    that estimate at every future horizon.  The checkpoint stores the probe's
    own target normalization, which is deliberately kept separate from the
    future-label statistics.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "estimated persistence checkpoint does not exist: "
            f"{checkpoint_path}"
        )
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    saved_args = checkpoint.get("args") or {}
    if not isinstance(saved_args, dict):
        raise RuntimeError(
            "estimated persistence checkpoint has no serialized argparse dictionary: "
            f"{checkpoint_path}"
        )
    probe_args = SimpleNamespace(**vars(args))
    for name in ("probe_type", "probe_depth", "probe_hidden_dim", "tubelet_size"):
        if name in saved_args:
            setattr(probe_args, name, saved_args[name])
    if saved_args.get("dataset") != args.dataset:
        raise ValueError(
            "estimated persistence probe dataset does not match future probe dataset: "
            f"checkpoint={saved_args.get('dataset')!r}, evaluation={args.dataset!r}"
        )
    if saved_args.get("target_mode") != "state":
        raise ValueError(
            "estimated persistence checkpoint must be trained for current-state targets, "
            f"got target_mode={saved_args.get('target_mode')!r}"
        )
    current_targets = normalize_checkpoint_targets(args.dataset, saved_args.get("targets"))
    if current_targets != list(args.normalized_target_fields):
        raise ValueError(
            "estimated persistence probe targets do not match future targets: "
            f"checkpoint={current_targets}, evaluation={args.normalized_target_fields}"
        )
    if saved_args.get("arch") != args.arch:
        raise ValueError(
            "estimated persistence probe architecture does not match future probe: "
            f"checkpoint={saved_args.get('arch')!r}, evaluation={args.arch!r}"
        )
    if saved_args.get("tubelet_size") != args.tubelet_size:
        raise ValueError(
            "estimated persistence probe tubelet_size does not match future probe: "
            f"checkpoint={saved_args.get('tubelet_size')!r}, evaluation={args.tubelet_size!r}"
        )
    current_encoder = _resolved_path(saved_args.get("encoder_checkpoint"))
    future_encoder = _resolved_path(args.encoder_checkpoint)
    if current_encoder != future_encoder:
        raise ValueError(
            "estimated persistence probe encoder does not match future probe: "
            f"checkpoint={current_encoder}, evaluation={future_encoder}"
        )
    validate_estimated_persistence_protocol(checkpoint, future_checkpoint, args)
    if probe_args.probe_type == "future_conv1x1":
        raise ValueError(
            "estimated persistence checkpoint must be a current-state probe, "
            f"got probe_type={probe_args.probe_type!r}"
        )
    current_probe = build_probe(
        probe_args,
        arch,
        num_outputs=num_outputs,
        num_future_steps=1,
    )
    state = checkpoint.get("probe")
    if state is None:
        raise KeyError(f"estimated persistence checkpoint lacks 'probe': {checkpoint_path}")
    try:
        current_probe.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "estimated persistence probe is incompatible with the future target "
            f"dimension ({num_outputs}): {checkpoint_path}"
        ) from exc
    current_mean = checkpoint.get("target_mean")
    current_std = checkpoint.get("target_std")
    if current_mean is None or current_std is None:
        raise KeyError(
            "estimated persistence checkpoint must contain target_mean and target_std: "
            f"{checkpoint_path}"
        )
    current_mean = torch.as_tensor(current_mean, dtype=torch.float32)
    current_std = torch.as_tensor(current_std, dtype=torch.float32)
    if current_mean.ndim != 1 or current_std.shape != current_mean.shape:
        raise ValueError(
            "estimated persistence normalization stats must have shape [D], got "
            f"{tuple(current_mean.shape)} and {tuple(current_std.shape)}"
        )
    if current_mean.numel() != num_outputs:
        raise ValueError(
            "estimated persistence normalization dimension does not match future targets: "
            f"{current_mean.numel()} vs {num_outputs}"
        )
    if not torch.isfinite(current_mean).all() or not torch.isfinite(current_std).all():
        raise ValueError(f"estimated persistence normalization contains non-finite values: {checkpoint_path}")
    if (current_std <= 0).any():
        raise ValueError(f"estimated persistence normalization std must be positive: {checkpoint_path}")
    validate_estimated_persistence_protocol(checkpoint, future_checkpoint, args)
    current_probe.to(device).eval()
    for parameter in current_probe.parameters():
        parameter.requires_grad = False
    return current_probe, current_mean, current_std, {
        "checkpoint": str(checkpoint_path),
        "probe_type": probe_args.probe_type,
        "epoch": checkpoint.get("epoch"),
        "targets": current_targets,
    }


def dedupe_preserve_order(items):
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def agibot_row_group_key(row):
    return row.get("window_key") or row.get("episode_key") or row["video_id"]


def sample_agibot_train_rows(rows, max_samples, seed):
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    by_key = {}
    for row in rows:
        by_key.setdefault(agibot_row_group_key(row), []).append(row)
    group_keys = sorted(by_key)
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    selected_keys = []
    selected_count = 0
    for key in group_keys:
        group_size = len(by_key[key])
        if selected_keys and selected_count + group_size > max_samples:
            continue
        if not selected_keys and group_size > max_samples:
            selected_keys.append(key)
            selected_count += group_size
            break
        selected_keys.append(key)
        selected_count += group_size
        if selected_count >= max_samples:
            break
    selected_key_set = set(selected_keys)
    return [row for row in rows if agibot_row_group_key(row) in selected_key_set]


def read_agibot_metadata_for_views(args, video_keys):
    rows = []
    for video_key in video_keys:
        rows.extend(
            read_agibot_metadata(
                args.agibot_root,
                tasks=args.agibot_tasks,
                video_key=video_key,
                use_meta_episodes_only=args.agibot_use_meta_episodes_only,
            )
        )
    return rows


def main():
    args = parse_args()

    if args.eval_only and not args.probe_checkpoint:
        raise ValueError("--eval-only requires --probe-checkpoint")
    if args.probe_checkpoint and not args.eval_only:
        raise ValueError("--probe-checkpoint requires --eval-only")

    encoder_checkpoint = Path(args.encoder_checkpoint).expanduser()
    if not encoder_checkpoint.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint does not exist or is not a file: {encoder_checkpoint}"
        )
    args.encoder_checkpoint = str(encoder_checkpoint)

    if args.ood == "kitti":
        if args.train_split == "train":
            args.train_split = "probe_train"
        if args.val_split == "val":
            args.val_split = "probe_val"

    if args.tubelet_size != 2:
        raise ValueError(
            "This probe script keeps the frozen V-JEPA encoder architecture unchanged, "
            "so --tubelet-size must remain 2 for the current official checkpoint."
        )

    dist_info = init_distributed(args)
    distributed = dist_info["distributed"]
    rank = dist_info["rank"]
    is_main = dist_info["is_main"]
    device = dist_info["device"]

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    writer = build_summary_writer(args, output_dir, is_main)

    dataset_name = "kitti" if args.ood == "kitti" else args.dataset
    if args.target_mode == "future_state":
        if dataset_name == "kitti":
            raise ValueError("--target-mode future_state is not implemented for KITTI")
        future_horizons_seconds = normalize_future_horizons_seconds(
            args.future_horizons_seconds,
            args.future_horizon_seconds,
        )
        args.future_horizons_seconds = list(future_horizons_seconds)
        # Keep the scalar field as the maximum horizon for terminal-window
        # filtering and compatibility with existing launch tooling.
        args.future_horizon_seconds = max(future_horizons_seconds)
        if args.probe_type != "future_conv1x1":
            raise ValueError("--target-mode future_state requires --probe-type future_conv1x1")
    else:
        future_horizons_seconds = ()
        args.future_horizons_seconds = None
        if args.probe_type == "future_conv1x1":
            raise ValueError("--probe-type future_conv1x1 requires --target-mode future_state")
    if args.target_mode == "trajectory_xy" and dataset_name != "covla":
        raise ValueError("--target-mode trajectory_xy is only supported for CoVLA states/*.jsonl")
    trajectory_point_indices = parse_trajectory_point_indices(args)
    if args.target_mode == "trajectory_xy" and not trajectory_point_indices:
        raise ValueError("--target-mode trajectory_xy requires at least one trajectory point index")
    requested_target_fields = [t.strip() for t in args.targets.split(",") if t.strip()]
    if args.target_mode == "trajectory_xy":
        target_fields = ["trajectory_xy"]
        target_names = trajectory_target_names(trajectory_point_indices)
    elif dataset_name == "kitti":
        target_fields = [normalize_kitti_target_field(t) for t in requested_target_fields]
        target_names = [kitti_target_display_name(t) for t in target_fields]
    elif dataset_name == "agibot":
        target_fields = normalize_agibot_target_fields(requested_target_fields)
        target_names = [agibot_target_display_name(t) for t in target_fields]
    elif dataset_name == "egoexo":
        target_fields = normalize_egoexo_target_fields(requested_target_fields)
        target_names = [egoexo_target_display_name(t) for t in target_fields]
    else:
        target_fields = [normalize_target_field(t) for t in requested_target_fields]
        target_names = [target_display_name(t) for t in target_fields]
    args.normalized_target_fields = list(target_fields)
    if args.agibot_shot_ood_protocol == "same_side":
        if dataset_name != "agibot" or args.target_mode != "state":
            raise ValueError(
                "--agibot-shot-ood-protocol same_side is only valid for AgiBot "
                "current-state Shot probes"
            )
        validate_agibot_same_side_ood(
            target_fields,
            parse_csv_list(args.agibot_ood_video_keys),
        )
    arch = ARCH_SPECS[args.arch]
    covla_ood_rows = []
    future_checkpoint = None
    checkpoint_stats = None
    if args.eval_only:
        if args.target_mode == "future_state":
            future_checkpoint, checkpoint_mean, checkpoint_std = load_future_probe_checkpoint(
                args.probe_checkpoint,
                args,
                dataset_name,
                target_fields,
                future_horizons_seconds,
            )
        else:
            future_checkpoint, checkpoint_mean, checkpoint_std = load_current_probe_checkpoint(
                args.probe_checkpoint,
                args,
                dataset_name,
                target_fields,
            )
        checkpoint_stats = (checkpoint_mean, checkpoint_std)
    if args.covla_ood_test_split and dataset_name != "covla":
        raise ValueError("--covla-ood-test-split is only supported when --dataset covla")

    if dataset_name == "kitti":
        rows = []
        train_rows, val_rows, test_rows = load_kitti_split_rows(
            args.split_dir,
            split_names=(args.train_split, args.val_split, args.test_split),
        )
    elif dataset_name == "egoexo":
        agibot_id_video_key = args.egoexo_video_view
        agibot_train_video_keys = dedupe_preserve_order(
            parse_csv_list(args.egoexo_train_video_keys) or [agibot_id_video_key]
        )
        if agibot_id_video_key not in agibot_train_video_keys:
            agibot_train_video_keys.insert(0, agibot_id_video_key)
        agibot_seen_video_keys = [
            view
            for view in dedupe_preserve_order(parse_csv_list(args.egoexo_seen_video_keys))
            if view != agibot_id_video_key
        ]
        unseen_seen_views = sorted(set(agibot_seen_video_keys) - set(agibot_train_video_keys))
        if unseen_seen_views:
            raise ValueError(
                "EgoExo seen evaluation views must also be probe-training views; "
                f"missing from --egoexo-train-video-keys: {unseen_seen_views}"
            )
        ood_video_keys = dedupe_preserve_order(parse_csv_list(args.egoexo_ood_video_keys))
        ood_overlap = sorted(set(ood_video_keys) & set(agibot_train_video_keys))
        if ood_overlap:
            raise ValueError(
                "EgoExo OOD views must be held out from probe training; "
                f"overlap={ood_overlap}"
            )
        train_rows, val_rows = load_egoexo_split_rows(
            args.egoexo_root, args.egoexo_split_dir,
            split_names=(args.train_split, args.val_split),
            prefix=args.egoexo_split_prefix, video_view=agibot_train_video_keys,
            frames_per_clip=arch["num_frames"], fps=args.egoexo_fps,
            slice_stride_seconds=args.egoexo_slice_stride_seconds,
            manifest_path=args.egoexo_manifest_path,
        )
        (test_rows,) = load_egoexo_split_rows(
            args.egoexo_root, args.egoexo_split_dir,
            split_names=(args.test_split,),
            prefix=args.egoexo_split_prefix, video_view=agibot_id_video_key,
            frames_per_clip=arch["num_frames"], fps=args.egoexo_fps,
            slice_stride_seconds=args.egoexo_slice_stride_seconds,
            manifest_path=args.egoexo_manifest_path,
        )
        rows = train_rows + val_rows + test_rows
        agibot_seen_rows_by_view = {}
        for view in agibot_seen_video_keys:
            (view_rows,) = load_egoexo_split_rows(
                args.egoexo_root, args.egoexo_split_dir, split_names=(args.test_split,),
                prefix=args.egoexo_split_prefix, video_view=view,
                frames_per_clip=arch["num_frames"], fps=args.egoexo_fps,
                slice_stride_seconds=args.egoexo_slice_stride_seconds,
                manifest_path=args.egoexo_manifest_path,
            )
            agibot_seen_rows_by_view[view] = view_rows
        agibot_ood_rows_by_view = {}
        for view in ood_video_keys:
            (view_rows,) = load_egoexo_split_rows(
                args.egoexo_root, args.egoexo_split_dir, split_names=(args.test_split,),
                prefix=args.egoexo_split_prefix, video_view=view,
                frames_per_clip=arch["num_frames"], fps=args.egoexo_fps,
                slice_stride_seconds=args.egoexo_slice_stride_seconds,
                manifest_path=args.egoexo_manifest_path,
            )
            agibot_ood_rows_by_view[view] = view_rows
        train_rows_before_view_sample = len(train_rows)
        val_rows_before_view_sample = len(val_rows)
        train_sample_seed = args.seed
        val_sample_seed = args.seed + 1
    elif dataset_name == "agibot":
        agibot_id_video_key = args.agibot_video_key
        agibot_train_video_keys = dedupe_preserve_order(
            parse_csv_list(args.agibot_train_video_keys) or [agibot_id_video_key]
        )
        if agibot_id_video_key not in agibot_train_video_keys:
            agibot_train_video_keys.insert(0, agibot_id_video_key)
        agibot_seen_video_keys = [
            video_key
            for video_key in dedupe_preserve_order(parse_csv_list(args.agibot_seen_video_keys))
            if video_key != agibot_id_video_key
        ]

        rows = read_agibot_metadata_for_views(args, agibot_train_video_keys)
        if args.agibot_split_dir:
            train_rows, val_rows, test_rows = split_agibot_rows_from_files(
                rows,
                args.agibot_split_dir,
                split_names=(args.train_split, args.val_split, args.test_split),
                prefix=args.agibot_split_prefix,
            )
        else:
            print_main(
                is_main,
                "warning:",
                "agibot_split_dir not set; falling back to on-the-fly seeded split "
                "(NOT episode-disjoint from the encoder; results may be optimistic)",
            )
            _, train_rows, val_rows, test_rows = split_agibot_rows(
                rows,
                ratios=args.agibot_split_ratios,
                seed=args.seed,
                split_names=("encoder_train", args.train_split, args.val_split, args.test_split),
            )
        train_rows_before_view_sample = len(train_rows)
        train_sample_seed = args.seed if args.agibot_train_sample_seed is None else args.agibot_train_sample_seed
        val_rows_before_view_sample = len(val_rows)
        val_sample_seed = (
            train_sample_seed + 1
            if args.agibot_val_sample_seed is None
            else args.agibot_val_sample_seed
        )
        train_rows = sample_agibot_train_rows(
            train_rows,
            max_samples=args.agibot_train_num_samples,
            seed=train_sample_seed,
        )
        val_rows = sample_agibot_train_rows(
            val_rows,
            max_samples=args.agibot_val_num_samples,
            seed=val_sample_seed,
        )
        id_rows = read_agibot_metadata_for_views(args, [agibot_id_video_key])
        if args.agibot_split_dir:
            (test_rows,) = split_agibot_rows_from_files(
                id_rows,
                args.agibot_split_dir,
                split_names=(args.test_split,),
                prefix=args.agibot_split_prefix,
            )
        else:
            _, _, _, test_rows = split_agibot_rows(
                id_rows,
                ratios=args.agibot_split_ratios,
                seed=args.seed,
                split_names=("encoder_train", args.train_split, args.val_split, args.test_split),
            )
        agibot_seen_rows_by_view = {}
        for seen_video_key in agibot_seen_video_keys:
            seen_rows = read_agibot_metadata_for_views(args, [seen_video_key])
            if args.agibot_split_dir:
                (seen_test_rows,) = split_agibot_rows_from_files(
                    seen_rows,
                    args.agibot_split_dir,
                    split_names=(args.test_split,),
                    prefix=args.agibot_split_prefix,
                )
            else:
                _, _, _, seen_test_rows = split_agibot_rows(
                    seen_rows,
                    ratios=args.agibot_split_ratios,
                    seed=args.seed,
                    split_names=("encoder_train", args.train_split, args.val_split, args.test_split),
                )
            agibot_seen_rows_by_view[seen_video_key] = seen_test_rows
        agibot_ood_rows_by_view = {}
        for ood_video_key in parse_csv_list(args.agibot_ood_video_keys):
            if ood_video_key == agibot_id_video_key or ood_video_key in agibot_seen_video_keys:
                continue
            ood_rows = read_agibot_metadata_for_views(args, [ood_video_key])
            if args.agibot_split_dir:
                (ood_test_rows,) = split_agibot_rows_from_files(
                    ood_rows,
                    args.agibot_split_dir,
                    split_names=(args.test_split,),
                    prefix=args.agibot_split_prefix,
                )
            else:
                _, _, _, ood_test_rows = split_agibot_rows(
                    ood_rows,
                    ratios=args.agibot_split_ratios,
                    seed=args.seed,
                    split_names=("encoder_train", args.train_split, args.val_split, args.test_split),
                )
            agibot_ood_rows_by_view[ood_video_key] = ood_test_rows
    else:
        rows = read_metadata(args.covla_root)
    if dataset_name == "covla" and args.split_dir:
        train_rows, val_rows, test_rows = split_rows_from_files(
            rows,
            args.split_dir,
            split_names=(args.train_split, args.val_split, args.test_split),
        )
        if args.covla_ood_test_split:
            (covla_ood_rows,) = split_rows_from_files(
                rows,
                args.split_dir,
                split_names=(args.covla_ood_test_split,),
            )
    elif dataset_name == "covla":
        if args.covla_ood_test_split:
            raise ValueError("--covla-ood-test-split requires --split-dir")
        train_rows, val_rows, test_rows = split_rows(rows, args.train_ratio, args.val_ratio, args.seed)
    future_filter_counts = {}
    if args.target_mode == "future_state" and dataset_name in ("agibot", "egoexo"):
        future_fps = args.agibot_fps if dataset_name == "agibot" else args.egoexo_fps
        future_stride = (
            args.agibot_slice_stride_seconds
            if dataset_name == "agibot"
            else args.egoexo_slice_stride_seconds
        )

        def filter_future_rows(rows_for_role, role):
            before = len(rows_for_role)
            filtered = [
                row
                for row in rows_for_role
                if future_row_is_eligible(
                    row,
                    arch["num_frames"],
                    future_fps,
                    args.future_horizon_seconds,
                    slice_stride_seconds=future_stride,
                )
            ]
            future_filter_counts[role] = {"before": before, "after": len(filtered)}
            if not filtered:
                raise RuntimeError(f"No future-eligible rows remain for {role}")
            return filtered

        train_rows = filter_future_rows(train_rows, args.train_split)
        val_rows = filter_future_rows(val_rows, args.val_split)
        test_rows = filter_future_rows(test_rows, args.test_split)
        agibot_seen_rows_by_view = {
            view: filter_future_rows(rows_for_view, f"seen:{view}:{args.test_split}")
            for view, rows_for_view in agibot_seen_rows_by_view.items()
        }
        agibot_ood_rows_by_view = {
            view: filter_future_rows(rows_for_view, f"ood:{view}:{args.test_split}")
            for view, rows_for_view in agibot_ood_rows_by_view.items()
        }
    if args.max_train_videos > 0:
        train_rows = train_rows[: args.max_train_videos]
    if args.max_val_videos > 0:
        val_rows = val_rows[: args.max_val_videos]
    if args.max_test_videos > 0:
        test_rows = test_rows[: args.max_test_videos]
        covla_ood_rows = covla_ood_rows[: args.max_test_videos]
    if covla_ood_rows:
        id_ids = {
            row["video_id"]
            for rows_for_role in (train_rows, val_rows, test_rows)
            for row in rows_for_role
        }
        overlap = sorted(id_ids & {row["video_id"] for row in covla_ood_rows})
        if overlap:
            raise RuntimeError(
                "CoVLA OOD split overlaps probe-train/val/ID-test roles; "
                f"count={len(overlap)}, first={overlap[:5]}"
            )

    stats_rows = train_rows
    if dataset_name == "egoexo" and args.target_mode != "future_state":
        from src.time_jepa.multidomain import get_or_compute_domain_stats
        encoder_keys = json.loads(
            (Path(args.egoexo_split_dir) / f"{args.egoexo_split_prefix}_encoder_train.json").read_text()
        )
        stats = get_or_compute_domain_stats(
            domain_cfg={
                "name": "egoexo", "domain": "egoexo", "root": args.egoexo_root,
                "video_views": agibot_train_video_keys, "targets": list(target_fields),
                "episode_allowlist": encoder_keys, "manifest_path": args.egoexo_manifest_path,
                "split_prefix": args.egoexo_split_prefix,
                "slice_stride_seconds": args.egoexo_slice_stride_seconds,
                "max_time_delta": args.egoexo_max_time_delta, "pose_fps": args.egoexo_pose_fps,
                "min_views": args.egoexo_min_views, "smooth_window": args.egoexo_smooth_window,
                "stats_max_videos": 2000, "stats_seed": args.seed,
                "stats_cache": args.egoexo_stats_cache,
            },
            frames_per_clip=arch["num_frames"], tubelet_size=args.tubelet_size,
            fps=args.egoexo_fps, cache_path=args.egoexo_stats_cache,
            distributed=distributed, rank=rank, device=device,
        )
    elif checkpoint_stats is not None:
        stats = checkpoint_stats
    else:
        stats = compute_or_broadcast_stats(
            stats_rows,
            target_fields,
            num_frames=arch["num_frames"],
            dataset_name=dataset_name,
            frame_span=args.frame_span,
            frame_stride=args.frame_stride,
            tubelet_size=args.tubelet_size,
            target_mode=args.target_mode,
            trajectory_point_indices=trajectory_point_indices,
            agibot_fps=args.agibot_fps,
            agibot_max_time_delta=args.agibot_max_time_delta,
            egoexo_fps=args.egoexo_fps,
            egoexo_max_time_delta=args.egoexo_max_time_delta,
            egoexo_pose_fps=args.egoexo_pose_fps,
            egoexo_min_views=args.egoexo_min_views,
            egoexo_smooth_window=args.egoexo_smooth_window,
            future_horizon_seconds=args.future_horizon_seconds,
            future_horizons_seconds=args.future_horizons_seconds,
            seed=args.seed,
            distributed=distributed,
            rank=rank,
            device=device,
        )
    target_mean, target_std = stats
    print_main(is_main, "distributed:", {"enabled": distributed, "rank": rank, "world_size": dist_info["world_size"]})
    print_main(
        is_main,
        "dataset:",
        {
            "name": dataset_name,
            "ood": args.ood,
            "covla_root": str(Path(args.covla_root).resolve()) if dataset_name == "covla" else None,
            "kitti_root": str(Path(args.kitti_root).resolve()) if dataset_name == "kitti" else None,
            "agibot_root": str(Path(args.agibot_root).resolve()) if dataset_name == "agibot" else None,
            "agibot_tasks": args.agibot_tasks if dataset_name == "agibot" else None,
            "agibot_video_key": agibot_id_video_key if dataset_name == "agibot" else None,
            "agibot_train_video_keys": agibot_train_video_keys if dataset_name == "agibot" else None,
            "egoexo_root": str(Path(args.egoexo_root).resolve()) if dataset_name == "egoexo" else None,
            "egoexo_video_view": args.egoexo_video_view if dataset_name == "egoexo" else None,
            "egoexo_train_video_keys": agibot_train_video_keys if dataset_name == "egoexo" else None,
            "egoexo_seen_video_keys": agibot_seen_video_keys if dataset_name == "egoexo" else None,
            "egoexo_ood_video_keys": parse_csv_list(args.egoexo_ood_video_keys) if dataset_name == "egoexo" else None,
        },
    )
    print_main(is_main, "targets:", target_names)
    print_main(is_main, "target_fields:", target_fields)
    print_main(
        is_main,
        "target_mode:",
        {
            "mode": args.target_mode,
            "trajectory_point_indices": list(trajectory_point_indices) if args.target_mode == "trajectory_xy" else None,
            "agibot_fps": args.agibot_fps if dataset_name == "agibot" else None,
            "agibot_max_time_delta": args.agibot_max_time_delta if dataset_name == "agibot" else None,
            "future_horizon_seconds": (
                args.future_horizon_seconds if args.target_mode == "future_state" else None
            ),
            "future_horizons_seconds": (
                args.future_horizons_seconds if args.target_mode == "future_state" else None
            ),
            "future_reference": (
                "one state at each configured offset from the final observed frame"
                if args.target_mode == "future_state"
                else None
            ),
        },
    )
    print_main(is_main, "target_mean:", target_mean.tolist())
    print_main(is_main, "target_std:", target_std.tolist())
    print_main(
        is_main,
        "target_stats:",
        {
            "method": (
                "seeded_probe_train_per_horizon_future_targets"
                if args.target_mode == "future_state"
                else "seeded_random_train_videos_tubelet_targets"
            ),
            "dataset": dataset_name,
            "target_mode": args.target_mode,
            "trajectory_point_indices": list(trajectory_point_indices) if args.target_mode == "trajectory_xy" else None,
            "max_videos": 2000,
            "seed": args.seed,
            "computed_on_rank0": distributed,
            "future_horizon_seconds": (
                args.future_horizon_seconds if args.target_mode == "future_state" else None
            ),
            "future_horizons_seconds": (
                args.future_horizons_seconds if args.target_mode == "future_state" else None
            ),
        },
    )
    print_main(
        is_main,
        "split:",
        {
            args.train_split: len(train_rows),
            args.val_split: len(val_rows),
            args.test_split: len(test_rows),
            **(
                {args.covla_ood_test_split: len(covla_ood_rows)}
                if args.covla_ood_test_split
                else {}
            ),
            "train_role": args.train_split,
            "val_role": args.val_split,
            "test_role": args.test_split,
            "future_eligibility_filter": future_filter_counts,
        },
    )
    if dataset_name in ("agibot", "egoexo"):
        print_main(
            is_main,
            f"{dataset_name}_view_protocol:",
            {
                "id_view": agibot_id_video_key,
                "train_views": agibot_train_video_keys,
                "seen_views": sorted(agibot_seen_rows_by_view),
                "ood_views": sorted(agibot_ood_rows_by_view),
                "train_count_before_view_sample": train_rows_before_view_sample,
                "val_count_before_view_sample": val_rows_before_view_sample,
                "train_num_samples": args.agibot_train_num_samples if dataset_name == "agibot" else 0,
                "val_num_samples": args.agibot_val_num_samples if dataset_name == "agibot" else 0,
                "train_sample_seed": train_sample_seed,
                "val_sample_seed": val_sample_seed,
                "ood_test_counts": {
                    view: len(rows_for_view)
                    for view, rows_for_view in agibot_ood_rows_by_view.items()
                },
                "seen_test_counts": {
                    view: len(rows_for_view)
                    for view, rows_for_view in agibot_seen_rows_by_view.items()
                },
                "slice_stride_seconds": (
                    args.agibot_slice_stride_seconds
                    if dataset_name == "agibot"
                    else args.egoexo_slice_stride_seconds
                ),
                "split_prefix": (
                    args.agibot_split_prefix
                    if dataset_name == "agibot"
                    else args.egoexo_split_prefix
                ),
            },
        )
    print_main(is_main, "probe_type:", args.probe_type)
    dataset_cls = {
        "agibot": AgiBotProbeDataset,
        "egoexo": EgoExoProbeDataset,
        "covla": CoVLAProbeDataset,
        "kitti": KITTIProbeDataset,
    }[dataset_name]
    extra_dataset_kwargs = dataset_kwargs(args, dataset_name)
    print_main(
        is_main,
        "sequence_sampling:",
        {
            "num_frames": arch["num_frames"],
            "tubelet_size": args.tubelet_size,
            "target_time_steps": (
                len(future_horizons_seconds) if args.target_mode == "future_state"
                else max(1, arch["num_frames"] // args.tubelet_size)
            ),
            "frame_span": args.frame_span,
            "frame_stride": args.frame_stride,
            "fixed_split_frame_indices": dataset_name == "kitti",
            "effective_stride_600f": dataset_cls(
                [],
                target_fields=target_fields,
                num_frames=arch["num_frames"],
                transform=None,
                frame_span=args.frame_span,
                frame_stride=args.frame_stride,
                target_mode=args.target_mode if dataset_name != "kitti" else "state",
                trajectory_point_indices=trajectory_point_indices,
                **extra_dataset_kwargs,
            ).effective_frame_stride(600),
        },
    )

    transform = build_transform(arch["img_size"])
    train_dataset = dataset_cls(
        train_rows,
        target_fields=target_fields,
        num_frames=arch["num_frames"],
        transform=transform,
        random_clip=True,
        stats=stats,
        frame_span=args.frame_span,
        frame_stride=args.frame_stride,
        tubelet_size=args.tubelet_size,
        video_num_threads=args.video_num_threads,
        target_mode=args.target_mode if dataset_name != "kitti" else "state",
        trajectory_point_indices=trajectory_point_indices,
        **extra_dataset_kwargs,
    )
    val_dataset = dataset_cls(
        val_rows,
        target_fields=target_fields,
        num_frames=arch["num_frames"],
        transform=transform,
        random_clip=False,
        stats=stats,
        frame_span=args.frame_span,
        frame_stride=args.frame_stride,
        tubelet_size=args.tubelet_size,
        video_num_threads=args.video_num_threads,
        target_mode=args.target_mode if dataset_name != "kitti" else "state",
        trajectory_point_indices=trajectory_point_indices,
        **extra_dataset_kwargs,
    )
    test_dataset = dataset_cls(
        test_rows,
        target_fields=target_fields,
        num_frames=arch["num_frames"],
        transform=transform,
        random_clip=False,
        stats=stats,
        frame_span=args.frame_span,
        frame_stride=args.frame_stride,
        tubelet_size=args.tubelet_size,
        video_num_threads=args.video_num_threads,
        target_mode=args.target_mode if dataset_name != "kitti" else "state",
        trajectory_point_indices=trajectory_point_indices,
        **extra_dataset_kwargs,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist_info["world_size"],
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    ) if distributed else None
    val_eval_dataset = (
        Subset(val_dataset, list(range(rank, len(val_dataset), dist_info["world_size"])))
        if distributed
        else val_dataset
    )
    test_eval_dataset = (
        Subset(test_dataset, list(range(rank, len(test_dataset), dist_info["world_size"])))
        if distributed
        else test_dataset
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + rank)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 10000 + rank)
    test_generator = torch.Generator()
    test_generator.manual_seed(args.seed + 20000 + rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    test_loader = DataLoader(
        test_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )
    covla_ood_loader = None
    if covla_ood_rows:
        covla_ood_dataset = dataset_cls(
            covla_ood_rows,
            target_fields=target_fields,
            num_frames=arch["num_frames"],
            transform=transform,
            random_clip=False,
            stats=stats,
            frame_span=args.frame_span,
            frame_stride=args.frame_stride,
            tubelet_size=args.tubelet_size,
            video_num_threads=args.video_num_threads,
            target_mode=args.target_mode,
            trajectory_point_indices=trajectory_point_indices,
            **extra_dataset_kwargs,
        )
        covla_ood_eval_dataset = (
            Subset(
                covla_ood_dataset,
                list(range(rank, len(covla_ood_dataset), dist_info["world_size"])),
            )
            if distributed
            else covla_ood_dataset
        )
        covla_ood_generator = torch.Generator()
        covla_ood_generator.manual_seed(args.seed + 35000 + rank)
        covla_ood_loader = DataLoader(
            covla_ood_eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=covla_ood_generator,
        )
    seen_loaders = {}
    if dataset_name in ("agibot", "egoexo"):
        for seen_index, (seen_video_key, seen_rows) in enumerate(sorted(agibot_seen_rows_by_view.items())):
            seen_dataset = dataset_cls(
                seen_rows,
                target_fields=target_fields,
                num_frames=arch["num_frames"],
                transform=transform,
                random_clip=False,
                stats=stats,
                frame_span=args.frame_span,
                frame_stride=args.frame_stride,
                tubelet_size=args.tubelet_size,
                video_num_threads=args.video_num_threads,
                target_mode=args.target_mode,
                trajectory_point_indices=trajectory_point_indices,
                **extra_dataset_kwargs,
            )
            seen_eval_dataset = (
                Subset(seen_dataset, list(range(rank, len(seen_dataset), dist_info["world_size"])))
                if distributed
                else seen_dataset
            )
            seen_generator = torch.Generator()
            seen_generator.manual_seed(args.seed + 25000 + rank + seen_index)
            seen_loaders[seen_video_key] = DataLoader(
                seen_eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=seen_generator,
            )
    ood_loaders = {}
    if dataset_name in ("agibot", "egoexo"):
        for ood_index, (ood_video_key, ood_rows) in enumerate(sorted(agibot_ood_rows_by_view.items())):
            ood_dataset = dataset_cls(
                ood_rows,
                target_fields=target_fields,
                num_frames=arch["num_frames"],
                transform=transform,
                random_clip=False,
                stats=stats,
                frame_span=args.frame_span,
                frame_stride=args.frame_stride,
                tubelet_size=args.tubelet_size,
                video_num_threads=args.video_num_threads,
                target_mode=args.target_mode,
                trajectory_point_indices=trajectory_point_indices,
                **extra_dataset_kwargs,
            )
            ood_eval_dataset = (
                Subset(ood_dataset, list(range(rank, len(ood_dataset), dist_info["world_size"])))
                if distributed
                else ood_dataset
            )
            ood_generator = torch.Generator()
            ood_generator.manual_seed(args.seed + 30000 + rank + ood_index)
            ood_loaders[ood_video_key] = DataLoader(
                ood_eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=ood_generator,
            )

    builder = arch["builder"] if args.encoder_family == "vjepa2_1" else arch["legacy_builder"]
    encoder_kwargs = {
        "img_size": (arch["img_size"], arch["img_size"]),
        "num_frames": arch["num_frames"],
    }
    if args.encoder_family == "vjepa2_1":
        encoder_kwargs.update(
            {
                "modality_embedding": True,
                "interpolate_rope": True,
            }
        )
    encoder = builder(**encoder_kwargs)
    print_main(
        is_main,
        "encoder_build:",
        {
            "family": args.encoder_family,
            "arch": args.arch,
            "checkpoint_key": args.checkpoint_key,
            "kwargs": encoder_kwargs,
        },
    )
    print_main(
        is_main,
        "encoder_load:",
        load_encoder(encoder, args.encoder_checkpoint, checkpoint_key=args.checkpoint_key),
    )
    encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False

    probe = build_probe(
        args,
        arch,
        num_outputs=len(target_names),
        num_future_steps=(
            len(future_horizons_seconds)
            if args.target_mode == "future_state"
            else 1
        ),
    ).to(device)
    estimated_persistence_probe = None
    estimated_persistence_mean = None
    estimated_persistence_std = None
    estimated_persistence_info = None
    if args.estimated_persistence_checkpoint is not None:
        if args.target_mode != "future_state":
            raise ValueError(
                "--estimated-persistence-checkpoint is only valid with "
                "--target-mode future_state"
            )
        (
            estimated_persistence_probe,
            estimated_persistence_mean,
            estimated_persistence_std,
            estimated_persistence_info,
        ) = load_estimated_persistence_probe(
            args.estimated_persistence_checkpoint,
            args,
            arch,
            num_outputs=len(target_names),
            device=device,
            future_checkpoint=future_checkpoint,
        )
        estimated_persistence_info["checkpoint"] = str(
            Path(args.estimated_persistence_checkpoint).expanduser().resolve()
        )
    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [dist_info["local_rank"]], "output_device": dist_info["local_rank"]}
        probe = DistributedDataParallel(probe, **ddp_kwargs)
    seed_everything(args.seed + rank)
    optimizer, lr_scheduler, wd_scheduler = build_optimizer_and_schedulers(args, unwrap_model(probe), len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.use_bfloat16))
    amp_dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16

    best_val = float("inf")
    history = []
    for epoch in range(0 if args.eval_only else args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print_main(is_main, "memory_epoch_start:", {"epoch": epoch + 1, **memory_report(device)})
        probe.train()
        running_loss_sum = torch.zeros((), device=device)
        running_count = torch.zeros((), device=device)
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            leave=True,
            dynamic_ncols=True,
            disable=not is_main,
        )
        for step, batch in enumerate(pbar):
            videos = batch["video"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            # Use no_grad instead of inference_mode: the frozen encoder output
            # is still consumed by a trainable probe whose autograd graph needs
            # normal tensors for saving activations during backward.
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=device.type == "cuda",
                    dtype=amp_dtype,
                ):
                    features = encoder(videos)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                preds = probe(features.detach())
                loss = masked_mse(
                    preds,
                    targets,
                    valid,
                    balance_time_steps=(args.target_mode == "future_state"),
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if wd_scheduler is not None:
                wd_scheduler.step()

            running_loss_sum += loss.detach() * len(videos)
            running_count += len(videos)
            pbar.set_postfix(
                loss=f"{(running_loss_sum / running_count.clamp_min(1.0)).item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                wd=f"{optimizer.param_groups[0]['weight_decay']:.2e}",
            )

        train_loss_sum = reduce_sum(running_loss_sum, distributed)
        train_count = reduce_sum(running_count, distributed)
        train_loss = (train_loss_sum / train_count.clamp_min(1.0)).item()
        print_main(is_main, "memory_before_val:", {"epoch": epoch + 1, **memory_report(device)})
        val_metrics = evaluate(
            encoder,
            probe,
            val_loader,
            device,
            target_mean,
            target_std,
            distributed=distributed,
            is_main=is_main,
            desc=args.val_split,
            future_horizons_seconds=(
                future_horizons_seconds if args.target_mode == "future_state" else None
            ),
            estimated_persistence_probe=estimated_persistence_probe,
            estimated_persistence_target_mean=estimated_persistence_mean,
            estimated_persistence_target_std=estimated_persistence_std,
        )
        print_main(is_main, "memory_after_val:", {"epoch": epoch + 1, **memory_report(device)})
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val": val_metrics,
            "targets": target_names,
            "target_fields": target_fields,
            "probe_type": args.probe_type,
        }
        if is_main:
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            writer_step = epoch + 1
            if writer is not None:
                writer.add_scalar("loss/train", train_loss, writer_step)
                writer.add_scalar("loss/val", val_metrics["loss"], writer_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], writer_step)
                for target_name, mae, rmse, nrmse, pearson, ccc, r2, skill_train_mean in zip(
                    target_names,
                    val_metrics["mae"],
                    val_metrics["rmse"],
                    val_metrics["nrmse"],
                    val_metrics["pearson"],
                    val_metrics["ccc"],
                    val_metrics["r2"],
                    val_metrics["skill_train_mean"],
                ):
                    writer.add_scalar(f"mae/{target_name}", mae, writer_step)
                    writer.add_scalar(f"rmse/{target_name}", rmse, writer_step)
                    writer.add_scalar(f"nrmse/{target_name}", nrmse, writer_step)
                    writer.add_scalar(f"pearson/{target_name}", pearson, writer_step)
                    writer.add_scalar(f"ccc/{target_name}", ccc, writer_step)
                    writer.add_scalar(f"r2/{target_name}", r2, writer_step)
                    writer.add_scalar(f"skill_train_mean/{target_name}", skill_train_mean, writer_step)
                for horizon_key, horizon_metrics in val_metrics.get("per_horizon", {}).items():
                    persistence_metrics = horizon_metrics["baselines"].get("persistence")
                    estimated_persistence_metrics = horizon_metrics["baselines"].get(
                        "estimated_persistence"
                    )
                    for target_index, target_name in enumerate(target_names):
                        prefix = f"future_h{horizon_key}s/{target_name}"
                        writer.add_scalar(
                            f"{prefix}/nrmse",
                            horizon_metrics["nrmse"][target_index],
                            writer_step,
                        )
                        writer.add_scalar(
                            f"{prefix}/r2",
                            horizon_metrics["r2"][target_index],
                            writer_step,
                        )
                        if persistence_metrics is not None:
                            writer.add_scalar(
                                f"{prefix}/persistence_skill",
                                persistence_metrics["model_skill"][target_index],
                                writer_step,
                            )
                        if estimated_persistence_metrics is not None:
                            writer.add_scalar(
                                f"{prefix}/estimated_persistence_skill",
                                estimated_persistence_metrics["model_skill"][target_index],
                                writer_step,
                            )
                writer.flush()

            with (output_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2)

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    output_dir / "best_probe.pt",
                    args.encoder_checkpoint,
                    probe,
                    optimizer,
                    epoch + 1,
                    args,
                    stats,
                    val_metrics,
                )

        release_epoch_memory(device, clear_state_cache=not args.keep_state_cache)
        print_main(is_main, "memory_after_release:", {"epoch": epoch + 1, **memory_report(device)})

    if not args.eval_only:
        print_main(is_main, f"best_val_loss={best_val:.6f}")
        print_main(is_main, f"saved_best={output_dir / 'best_probe.pt'}")
    if distributed:
        dist.barrier()
    best_probe_path = Path(args.probe_checkpoint) if args.eval_only else output_dir / "best_probe.pt"
    if not best_probe_path.is_file() and args.eval_only:
        raise FileNotFoundError(f"Future probe checkpoint not found: {best_probe_path}")
    if best_probe_path.exists():
        checkpoint = torch_load(best_probe_path, map_location="cpu")
        unwrap_model(probe).load_state_dict(checkpoint["probe"])
        test_metrics = evaluate(
            encoder,
            probe,
            test_loader,
            device,
            target_mean,
            target_std,
            distributed=distributed,
            is_main=is_main,
            desc=args.test_split,
            future_horizons_seconds=(
                future_horizons_seconds if args.target_mode == "future_state" else None
            ),
            estimated_persistence_probe=estimated_persistence_probe,
            estimated_persistence_target_mean=estimated_persistence_mean,
            estimated_persistence_target_std=estimated_persistence_std,
        )
        covla_ood_metrics = None
        if covla_ood_loader is not None:
            covla_ood_metrics = evaluate(
                encoder,
                probe,
                covla_ood_loader,
                device,
                target_mean,
                target_std,
                distributed=distributed,
                is_main=is_main,
                desc=f"ood:{args.covla_ood_test_split}",
                future_horizons_seconds=(
                    future_horizons_seconds if args.target_mode == "future_state" else None
                ),
                estimated_persistence_probe=estimated_persistence_probe,
                estimated_persistence_target_mean=estimated_persistence_mean,
                estimated_persistence_target_std=estimated_persistence_std,
            )
        seen_metrics = {}
        for seen_video_key, seen_loader in sorted(seen_loaders.items()):
            seen_metrics[seen_video_key] = evaluate(
                encoder,
                probe,
                seen_loader,
                device,
                target_mean,
                target_std,
                distributed=distributed,
                is_main=is_main,
                desc=f"seen:{seen_video_key}",
                future_horizons_seconds=(
                    future_horizons_seconds if args.target_mode == "future_state" else None
                ),
                estimated_persistence_probe=estimated_persistence_probe,
                estimated_persistence_target_mean=estimated_persistence_mean,
                estimated_persistence_target_std=estimated_persistence_std,
            )
        ood_metrics = {}
        for ood_video_key, ood_loader in sorted(ood_loaders.items()):
            ood_metrics[ood_video_key] = evaluate(
                encoder,
                probe,
                ood_loader,
                device,
                target_mean,
                target_std,
                distributed=distributed,
                is_main=is_main,
                desc=f"ood:{ood_video_key}",
                future_horizons_seconds=(
                    future_horizons_seconds if args.target_mode == "future_state" else None
                ),
                estimated_persistence_probe=estimated_persistence_probe,
                estimated_persistence_target_mean=estimated_persistence_mean,
                estimated_persistence_target_std=estimated_persistence_std,
            )
        if is_main:
            agibot_shot_ood_report = (
                build_agibot_shot_ood_report(
                    ood_metrics,
                    target_fields,
                    target_names,
                )
                if dataset_name == "agibot"
                and args.agibot_shot_ood_protocol == "same_side"
                else None
            )
            id_view_metrics = (
                {agibot_id_video_key: test_metrics, **seen_metrics}
                if dataset_name in ("agibot", "egoexo")
                else {}
            )
            final_report = {
                "test": test_metrics,
                "ood_test": covla_ood_metrics,
                "seen_view_test": seen_metrics,
                "ood_view_test": ood_metrics,
                "id_view_test": id_view_metrics,
                "id_view_macro_test": macro_average_view_metrics(id_view_metrics),
                "ood_view_macro_test": macro_average_view_metrics(ood_metrics),
                "agibot_shot_ood_report": agibot_shot_ood_report,
                "best_val": checkpoint.get("metrics"),
                "best_epoch": checkpoint.get("epoch"),
                "targets": target_names,
                "target_fields": target_fields,
                "target_mode": args.target_mode,
                "future_horizon_seconds": (
                    args.future_horizon_seconds if args.target_mode == "future_state" else None
                ),
                "future_horizons_seconds": (
                    args.future_horizons_seconds if args.target_mode == "future_state" else None
                ),
                "trajectory_point_indices": list(trajectory_point_indices) if args.target_mode == "trajectory_xy" else None,
                "probe_type": args.probe_type,
                "probe_checkpoint": str(best_probe_path.resolve()),
                "estimated_persistence": estimated_persistence_info,
                "dataset": {
                    "name": dataset_name,
                    "ood": args.ood,
                    "covla_root": str(Path(args.covla_root).resolve()) if dataset_name == "covla" else None,
                    "covla_ood_test_split": args.covla_ood_test_split if dataset_name == "covla" else None,
                    "kitti_root": str(Path(args.kitti_root).resolve()) if dataset_name == "kitti" else None,
                    "agibot_root": str(Path(args.agibot_root).resolve()) if dataset_name == "agibot" else None,
                    "egoexo_root": str(Path(args.egoexo_root).resolve()) if dataset_name == "egoexo" else None,
                    "egoexo_video_view": args.egoexo_video_view if dataset_name == "egoexo" else None,
                    "egoexo_train_video_keys": agibot_train_video_keys if dataset_name == "egoexo" else None,
                    "egoexo_seen_video_keys": agibot_seen_video_keys if dataset_name == "egoexo" else None,
                    "egoexo_ood_video_keys": parse_csv_list(args.egoexo_ood_video_keys) if dataset_name == "egoexo" else None,
                    "egoexo_effective_ood_video_keys": sorted(agibot_ood_rows_by_view) if dataset_name == "egoexo" else None,
                    "agibot_tasks": args.agibot_tasks if dataset_name == "agibot" else None,
                    "agibot_video_key": agibot_id_video_key if dataset_name == "agibot" else None,
                    "agibot_train_video_keys": agibot_train_video_keys if dataset_name == "agibot" else None,
                    "agibot_seen_video_keys": agibot_seen_video_keys if dataset_name == "agibot" else None,
                    "agibot_ood_video_keys": parse_csv_list(args.agibot_ood_video_keys) if dataset_name == "agibot" else None,
                    "agibot_effective_ood_video_keys": sorted(agibot_ood_rows_by_view) if dataset_name == "agibot" else None,
                    "agibot_shot_ood_protocol": args.agibot_shot_ood_protocol if dataset_name == "agibot" else None,
                    "agibot_train_num_samples": args.agibot_train_num_samples if dataset_name == "agibot" else None,
                    "agibot_train_sample_seed": train_sample_seed if dataset_name == "agibot" else None,
                    "agibot_val_num_samples": args.agibot_val_num_samples if dataset_name == "agibot" else None,
                    "agibot_val_sample_seed": val_sample_seed if dataset_name == "agibot" else None,
                    "agibot_slice_stride_seconds": args.agibot_slice_stride_seconds if dataset_name == "agibot" else None,
                    "agibot_split_prefix": args.agibot_split_prefix if dataset_name == "agibot" else None,
                },
                "splits": {
                    "train": args.train_split,
                    "val": args.val_split,
                    "test": args.test_split,
                    "ood_test": args.covla_ood_test_split,
                    "counts": {
                        args.train_split: len(train_rows),
                        args.val_split: len(val_rows),
                        args.test_split: len(test_rows),
                        **(
                            {args.covla_ood_test_split: len(covla_ood_rows)}
                            if args.covla_ood_test_split
                            else {}
                        ),
                        **{
                            f"seen:{view}:{args.test_split}": len(rows_for_view)
                            for view, rows_for_view in (agibot_seen_rows_by_view.items() if dataset_name in ("agibot", "egoexo") else [])
                        },
                        **{
                            f"ood:{view}:{args.test_split}": len(rows_for_view)
                            for view, rows_for_view in (agibot_ood_rows_by_view.items() if dataset_name in ("agibot", "egoexo") else [])
                        },
                    },
                },
            }
            print(json.dumps(final_report, ensure_ascii=False))
            with (output_dir / "test_metrics.json").open("w") as f:
                json.dump(final_report, f, indent=2)
    else:
        print_main(is_main, f"warning: best checkpoint not found, skipped test eval: {best_probe_path}")
    if writer is not None:
        writer.close()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
