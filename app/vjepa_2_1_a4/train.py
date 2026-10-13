"""Isolated V-JEPA 2.1 adapter for EgoExo A4 misaligned supervision.

The production ``app.vjepa_2_1`` implementation is imported but never edited.
Only this process's module-level ``init_data`` reference is temporarily routed
to the dedicated A4 dataset factory.
"""

import copy
import hashlib
import logging
from pathlib import Path

from app.vjepa_2_1 import train as base_train
from src.datasets.data_manager import init_data as standard_init_data
from src.time_jepa.egoexo_a4_misaligned_dataset import (
    A4_EGOEXO_MISALIGNMENT_MODE,
    make_egoexo_a4_misaligned_dataset,
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
}


def _verify_base_code_snapshot(repo_root=_REPO_ROOT):
    observed = {}
    for relative_path, expected_sha256 in _EXPECTED_BASE_CODE_SHA256.items():
        path = Path(repo_root) / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[relative_path] = digest
        if digest != expected_sha256:
            raise RuntimeError(
                "A4 isolated app base-code snapshot mismatch for "
                f"{relative_path}: expected={expected_sha256}, observed={digest}. "
                "Refuse to run until the controlled diff is re-audited."
            )
    return observed


def _validate_and_resolve_a4_config(args):
    time_jepa = args.get("time_jepa", {}) or {}
    if not bool(time_jepa.get("enabled", False)):
        raise ValueError("A4 M2d requires time_jepa.enabled=true")
    if str(time_jepa.get("mode", "")).strip().lower() not in (
        "multi",
        "multidomain",
        "multi_domain",
    ):
        raise ValueError("A4 M2d requires the formal multi_domain data path")
    if str(time_jepa.get("head_type", "")).strip().lower() != "conv1x1":
        raise ValueError("A4 M2d must keep the formal conv1x1 training head")
    if str(time_jepa.get("feature_mode", "")).strip().lower() != "final":
        raise ValueError("A4 M2d must keep final encoder features")

    domains = time_jepa.get("domains", {}) or {}
    if len(domains) != 1:
        raise ValueError("A4 M2d requires exactly one EgoExo domain")
    domain_name, domain = next(iter(domains.items()))
    if str(domain.get("domain", domain_name)).strip().lower() != "egoexo":
        raise ValueError("A4 M2d only supports the EgoExo domain")
    targets = domain.get("targets", {}) or {}
    if not isinstance(targets, dict) or targets.get("type") != "whole_body_compact_v2":
        raise ValueError("A4 M2d must keep the formal full 9D target preset")
    if domain.get("target_temporal_mode", "tubelet_aligned") not in (
        "aligned",
        "tubelet",
        "tubelet_aligned",
    ):
        raise ValueError("A4 M2d must keep tubelet-aligned source targets")

    misalignment = time_jepa.get("physical_misalignment", {}) or {}
    if not isinstance(misalignment, dict) or not misalignment.get("enabled", False):
        raise ValueError("A4 M2d requires physical_misalignment.enabled=true")
    mode = str(misalignment.get("mode", "")).strip().lower().replace("-", "_")
    if mode != A4_EGOEXO_MISALIGNMENT_MODE:
        raise ValueError(
            f"A4 M2d requires mode={A4_EGOEXO_MISALIGNMENT_MODE!r}"
        )
    seed = int(misalignment.get("seed", args.get("meta", {}).get("seed", 239)))
    if seed != int(args.get("meta", {}).get("seed", seed)):
        raise ValueError("A4 pairing seed must equal the formal encoder seed")

    pairing = domain.get("target_pairing", {}) or {}
    if pairing.get("mode") != mode or int(pairing.get("seed", seed)) != seed:
        raise ValueError(
            "time_jepa.domains.egoexo.target_pairing must duplicate the A4 "
            "mode/seed so the checkpoint domain manifest is self-describing"
        )
    return mode, seed


def _make_isolated_init_data(args, mode, seed):
    output_folder = Path(args["folder"])
    manifest_path = output_folder / "a4_target_pairing_manifest.json"

    def init_data(*call_args, **kwargs):
        data = kwargs.get("data")
        if data is None and len(call_args) >= 3:
            data = call_args[2]
        if str(data).strip().lower() not in _EGOEXO_DATASET_NAMES:
            return standard_init_data(*call_args, **kwargs)
        if call_args:
            raise TypeError("The isolated A4 data adapter requires keyword arguments")

        dataset, loader, sampler = make_egoexo_a4_misaligned_dataset(
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
            egoexo_misalignment_mode=mode,
            egoexo_misalignment_seed=seed,
            pairing_manifest_path=(
                manifest_path if int(kwargs.get("rank", 0)) == 0 else None
            ),
        )
        return loader, sampler

    return init_data


def main(args, resume_preempt=False):
    observed_hashes = _verify_base_code_snapshot()
    logging.getLogger(__name__).info(
        "A4 isolated base-code snapshot verified: %s",
        observed_hashes,
    )
    isolated_args = copy.deepcopy(args)
    mode, seed = _validate_and_resolve_a4_config(isolated_args)
    original_init_data = base_train.init_data
    base_train.init_data = _make_isolated_init_data(isolated_args, mode, seed)
    try:
        return base_train.main(args=isolated_args, resume_preempt=resume_preempt)
    finally:
        base_train.init_data = original_init_data
