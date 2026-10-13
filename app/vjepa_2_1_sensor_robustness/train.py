"""Isolated V-JEPA 2.1 adapter for EgoExo sensor-target corruptions."""

import copy
import hashlib
import json
import logging
import os
from pathlib import Path

from app.vjepa_2_1 import train as base_train
from src.datasets.data_manager import init_data as standard_init_data
from src.time_jepa.egoexo_pose_targets import normalize_egoexo_targets
from src.time_jepa.egoexo_sensor_robustness_dataset import (
    make_egoexo_sensor_robustness_dataset,
    normalize_sensor_corruption_config,
)


_EGOEXO_DATASET_NAMES = {
    "egoexotimejepadataset",
    "egoexo_time_jepa",
    "egoexo_time",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_BASE_CODE_SHA256 = {
    "app/vjepa_2_1/train.py": (
        "64b56164721635d3c14de1630ec8e2e8fc2a3117f9161dcc3d13677ddbaa63a8"
    ),
    "src/datasets/data_manager.py": (
        "bb0d13a30edd104de18f2042167be253c6d76b87ef51b7b989c075c193237fc1"
    ),
    "src/time_jepa/egoexo_time_dataset.py": (
        "78a1b9f56fc9677153ac84f91140a94e23e21a249b500d469c6f1366444d40ba"
    ),
    "src/time_jepa/egoexo_pose_targets.py": (
        "cc4eaa4c29b9cd73a7f2d2621a18e34b63e7c417482abe07ebe3568a005625d4"
    ),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_base_code_snapshot(repo_root=_REPO_ROOT):
    observed = {}
    for relative_path, expected_sha256 in _EXPECTED_BASE_CODE_SHA256.items():
        path = Path(repo_root) / relative_path
        digest = _sha256(path)
        observed[relative_path] = digest
        if digest != expected_sha256:
            raise RuntimeError(
                "Sensor-robustness base-code snapshot mismatch for "
                f"{relative_path}: expected={expected_sha256}, observed={digest}. "
                "Refuse to run until the controlled diff is re-audited."
            )
    return observed


def _resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _load_clean_target_std(domain_cfg, target_fields):
    cache_path = domain_cfg.get("stats_cache")
    if not cache_path:
        raise ValueError("Gaussian target corruption requires the clean stats_cache")
    cache_path = _resolve_path(cache_path)
    with cache_path.open("r") as handle:
        payload = json.load(handle)
    cached_fields = payload.get("meta", {}).get("target_fields")
    if list(cached_fields or []) != list(target_fields):
        raise ValueError(
            "Clean stats target order does not match the configured EgoExo targets: "
            f"cache={cached_fields}, config={target_fields}"
        )
    std = payload.get("std")
    if not isinstance(std, list) or len(std) != len(target_fields):
        raise ValueError("Clean stats cache has an invalid std vector")
    return [float(value) for value in std], cache_path


def _validate_and_resolve_config(args):
    if args.get("app") != "vjepa_2_1_sensor_robustness":
        raise ValueError("Sensor robustness config must use its isolated app")
    time_jepa = args.get("time_jepa", {}) or {}
    if not bool(time_jepa.get("enabled", False)):
        raise ValueError("Sensor robustness requires time_jepa.enabled=true")
    if str(time_jepa.get("mode", "")).strip().lower() not in (
        "multi",
        "multidomain",
        "multi_domain",
    ):
        raise ValueError("Sensor robustness requires the formal multi_domain path")
    if str(time_jepa.get("head_type", "")).strip().lower() != "conv1x1":
        raise ValueError("Sensor robustness must keep the formal conv1x1 head")
    if str(time_jepa.get("feature_mode", "")).strip().lower() != "final":
        raise ValueError("Sensor robustness must keep final encoder features")

    domains = time_jepa.get("domains", {}) or {}
    if len(domains) != 1:
        raise ValueError("Sensor robustness requires exactly one EgoExo domain")
    domain_name, domain_cfg = next(iter(domains.items()))
    if str(domain_cfg.get("domain", domain_name)).strip().lower() != "egoexo":
        raise ValueError("Sensor robustness only supports the EgoExo domain")
    targets = domain_cfg.get("targets", {}) or {}
    if not isinstance(targets, dict) or targets.get("type") != "whole_body_compact_v2":
        raise ValueError("Sensor robustness must keep the formal full 9D target preset")
    temporal_mode = str(
        domain_cfg.get("target_temporal_mode", "tubelet_aligned")
    ).strip().lower().replace("-", "_")
    if temporal_mode not in ("aligned", "tubelet", "tubelet_aligned"):
        raise ValueError("Sensor robustness requires tubelet-aligned source targets")

    corruption = normalize_sensor_corruption_config(
        domain_cfg.get("target_corruption")
    )
    top_level_corruption = time_jepa.get("sensor_robustness")
    if top_level_corruption is None:
        raise ValueError(
            "time_jepa.sensor_robustness must duplicate the domain corruption config"
        )
    if normalize_sensor_corruption_config(top_level_corruption) != corruption:
        raise ValueError(
            "time_jepa.sensor_robustness and domains.egoexo.target_corruption differ"
        )

    debug_skip_save = bool(args.get("meta", {}).get("debug_skip_checkpoint_write", False))
    if debug_skip_save:
        if int(args.get("optimization", {}).get("epochs", 0)) != 1:
            raise ValueError("sensor-robustness debug runs must use exactly one epoch")
        if int(args.get("optimization", {}).get("ipe", 0)) > 2:
            raise ValueError("sensor-robustness debug runs must use at most two iterations")
    elif corruption["mode"] == "composite_debug":
        raise ValueError("composite_debug is allowed only for the no-checkpoint smoke run")

    target_fields = normalize_egoexo_targets(targets)
    target_std = None
    stats_path = None
    if corruption["noise_std"] > 0.0:
        target_std, stats_path = _load_clean_target_std(domain_cfg, target_fields)
    return domain_cfg, target_fields, corruption, target_std, stats_path, debug_skip_save


def _write_run_manifest(args, corruption, target_fields, stats_path, base_hashes):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output_dir = Path(args["folder"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "egoexo_sensor_robustness",
        "fine_tuning_targets_only": True,
        "probe_and_evaluation_labels": "clean",
        "target_fields": list(target_fields),
        "target_corruption": corruption,
        "clean_stats_cache": None if stats_path is None else str(stats_path),
        "clean_stats_sha256": None if stats_path is None else _sha256(stats_path),
        "base_code_sha256": base_hashes,
        "dataset_code_sha256": _sha256(
            _REPO_ROOT / "src/time_jepa/egoexo_sensor_robustness_dataset.py"
        ),
        "app_code_sha256": _sha256(Path(__file__)),
    }
    path = output_dir / "sensor_robustness_manifest.json"
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def _make_isolated_init_data(corruption, target_std):
    def init_data(*call_args, **kwargs):
        data = kwargs.get("data")
        if data is None and len(call_args) >= 3:
            data = call_args[2]
        if str(data).strip().lower() not in _EGOEXO_DATASET_NAMES:
            return standard_init_data(*call_args, **kwargs)
        if call_args:
            raise TypeError(
                "The isolated sensor-robustness adapter requires keyword arguments"
            )

        _, loader, sampler = make_egoexo_sensor_robustness_dataset(
            data_paths=kwargs.get("root_path"),
            batch_size=kwargs["batch_size"],
            frames_per_clip=kwargs.get("clip_len") or 64,
            dataset_fpcs=kwargs.get("dataset_fpcs"),
            fps=kwargs.get("fps", 5.0),
            num_clips=kwargs.get("num_clips", 1),
            transform=kwargs.get("transform"),
            shared_transform=kwargs.get("shared_transform"),
            rank=kwargs.get("rank", 0),
            world_size=kwargs.get("world_size", 1),
            datasets_weights=kwargs.get("datasets_weights"),
            collator=kwargs.get("collator"),
            drop_last=kwargs.get("drop_last", True),
            num_workers=kwargs.get("num_workers", 10),
            pin_mem=kwargs.get("pin_mem", True),
            persistent_workers=kwargs.get("persistent_workers", True),
            deterministic=kwargs.get("deterministic", True),
            log_dir=kwargs.get("log_dir"),
            egoexo_video_view=kwargs.get("egoexo_video_view", "best_exo"),
            egoexo_target_fields=kwargs.get("egoexo_target_fields"),
            physical_tubelet_size=kwargs.get("physical_tubelet_size", 2),
            egoexo_slice_stride_seconds=kwargs.get(
                "egoexo_slice_stride_seconds", 12.8
            ),
            egoexo_max_time_delta=kwargs.get("egoexo_max_time_delta", 0.05),
            egoexo_pose_fps=kwargs.get("egoexo_pose_fps", 30.0),
            egoexo_min_views=kwargs.get("egoexo_min_views", 2),
            egoexo_smooth_window=kwargs.get("egoexo_smooth_window", 5),
            egoexo_video_backend=kwargs.get("egoexo_video_backend", "auto"),
            egoexo_ffmpeg_timeout=kwargs.get("egoexo_ffmpeg_timeout", 120),
            return_physical_targets=kwargs.get("return_physical_targets", True),
            egoexo_episode_allowlist=kwargs.get("egoexo_episode_allowlist"),
            egoexo_manifest_path=kwargs.get("egoexo_manifest_path"),
            egoexo_split_prefix=kwargs.get(
                "egoexo_split_prefix", "egoexo_bestexo"
            ),
            egoexo_target_temporal_mode=kwargs.get(
                "egoexo_target_temporal_mode", "tubelet_aligned"
            ),
            egoexo_target_corruption=corruption,
            egoexo_target_std=target_std,
        )
        return loader, sampler

    return init_data


def _install_debug_checkpoint_stub(output_dir):
    original_save = base_train.torch.save
    output_dir = Path(output_dir).resolve()

    def debug_save(payload, path, *args, **kwargs):
        path = Path(path).resolve()
        if output_dir not in path.parents:
            raise RuntimeError(f"Refusing to suppress checkpoint outside {output_dir}: {path}")
        marker = Path(str(path) + ".debug-skipped.json")
        marker.write_text(
            json.dumps(
                {
                    "checkpoint_write_skipped": True,
                    "epoch": int(payload.get("epoch", -1)),
                    "loss": float(payload.get("loss", float("nan"))),
                    "keys": sorted(payload),
                },
                indent=2,
                sort_keys=True,
            )
        )

    base_train.torch.save = debug_save
    return original_save


def main(args, resume_preempt=False):
    base_hashes = _verify_base_code_snapshot()
    isolated_args = copy.deepcopy(args)
    (
        _,
        target_fields,
        corruption,
        target_std,
        stats_path,
        debug_skip_save,
    ) = _validate_and_resolve_config(isolated_args)
    _write_run_manifest(
        isolated_args,
        corruption=corruption,
        target_fields=target_fields,
        stats_path=stats_path,
        base_hashes=base_hashes,
    )
    logging.getLogger(__name__).info(
        "EgoExo sensor robustness corruption: %s", corruption
    )

    original_init_data = base_train.init_data
    original_torch_save = None
    base_train.init_data = _make_isolated_init_data(corruption, target_std)
    if debug_skip_save:
        original_torch_save = _install_debug_checkpoint_stub(isolated_args["folder"])
    try:
        return base_train.main(args=isolated_args, resume_preempt=resume_preempt)
    finally:
        base_train.init_data = original_init_data
        if original_torch_save is not None:
            base_train.torch.save = original_torch_save
