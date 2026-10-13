#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pprint
import random
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from covla_jepa_rjob.split_config import (
    configured_encoder_train_split as resolve_configured_encoder_train_split,
    encoder_train_split_required,
    expected_encoder_train_videos,
    resolve_encoder_split_enabled,
)

try:
    import torch

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
except Exception:
    pass

from torch.distributed.elastic.multiprocessing.errors import record

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, help="YAML config file for V-JEPA training")
    parser.add_argument(
        "--covla-encoder-train-split",
        default=None,
        help="Optional covla_<split>.json used to restrict CoVLAVideoDataset encoder/JEPA training videos",
    )
    parser.add_argument(
        "--covla-train-split",
        default=None,
        help="Backward-compatible alias for --covla-encoder-train-split",
    )
    parser.add_argument(
        "--agibot-encoder-train-split",
        default=None,
        help=(
            "Optional AgiBot encoder split JSON. Supports episode keys "
            "'task_<id>/<session>/episode_<idx>' and window keys "
            "'task_<id>/<session>/episode_<idx>/window_<idx>'; prevents "
            "probe/test leakage."
        ),
    )
    parser.add_argument(
        "--egoexo-encoder-train-split",
        default=None,
        help="Optional EgoExo camera-agnostic encoder split JSON.",
    )
    parser.add_argument(
        "--filtered-metadata-dir",
        default=None,
        help="Directory for generated filtered CoVLA metadata; defaults to <folder>/filtered_metadata",
    )
    return parser.parse_args()


def _resolve_covla_metadata(data_path):
    data_path = Path(data_path).expanduser()
    if data_path.is_dir():
        dataset_root = data_path
        metadata_path = dataset_root / "metadata.jsonl"
    else:
        metadata_path = data_path
        dataset_root = metadata_path.parent
    if not metadata_path.exists():
        raise FileNotFoundError(f"CoVLA metadata file not found: {metadata_path}")
    return dataset_root.resolve(), metadata_path.resolve()


def _write_filtered_metadata(data_path, train_ids, output_path):
    dataset_root, metadata_path = _resolve_covla_metadata(data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    found_ids = set()
    with metadata_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            video_id = record.get("video_id")
            if video_id not in train_ids:
                continue
            file_name = record.get("file_name")
            if not file_name:
                raise RuntimeError(f"Missing file_name for CoVLA video_id={video_id}")
            video_path = Path(file_name)
            if not video_path.is_absolute():
                video_path = dataset_root / video_path
            if not video_path.exists():
                raise FileNotFoundError(f"Missing CoVLA video for train split: {video_path}")
            filtered_record = dict(record)
            filtered_record["file_name"] = str(video_path.resolve())
            kept.append(filtered_record)
            found_ids.add(video_id)

    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)
    return str(output_path.resolve()), found_ids, len(kept), str(metadata_path)


def apply_covla_train_split(params, split_path, filtered_metadata_dir=None):
    data_cfg = params.get("data", {})
    dataset_type = str(data_cfg.get("dataset_type", "")).lower()
    time_jepa_cfg = params.get("time_jepa", {}) or {}
    time_jepa_mode = str(time_jepa_cfg.get("mode", "")).lower()
    time_jepa_domains = time_jepa_cfg.get("domains", {}) or {}
    covla_domain_cfg = next(
        (
            domain_cfg
            for config_name, domain_cfg in time_jepa_domains.items()
            if isinstance(domain_cfg, dict)
            and str(domain_cfg.get("domain", config_name)).strip().lower()
            == "covla"
        ),
        None,
    )
    has_covla_domain = (
        time_jepa_mode in ("multi", "multidomain", "multi_domain")
        and isinstance(covla_domain_cfg, dict)
    )
    if dataset_type not in (
        "covlavideodataset",
        "covla_video",
        "covla",
        "covlatimejepadataset",
        "covla_time_jepa",
        "covla_time",
    ) and not has_covla_domain:
        return params

    split_path = Path(split_path).expanduser().resolve()
    split_stem = split_path.stem
    with split_path.open("r") as f:
        train_ids = set(json.load(f))
    if not train_ids:
        raise RuntimeError(f"Empty CoVLA train split: {split_path}")
    expected_train_videos = expected_encoder_train_videos(params, "covla")
    if (
        expected_train_videos is not None
        and len(train_ids) != expected_train_videos
    ):
        raise RuntimeError(
            "CoVLA encoder split size does not match the configured expectation: "
            f"split={split_path} actual={len(train_ids)} "
            f"expected={expected_train_videos}"
        )

    dataset_paths = data_cfg.get("datasets", [])
    if has_covla_domain:
        domain_paths = covla_domain_cfg.get(
            "root",
            covla_domain_cfg.get("datasets", dataset_paths),
        )
        dataset_paths = domain_paths
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    if not dataset_paths:
        raise RuntimeError("CoVLAVideoDataset requires data.datasets before applying train split")

    if filtered_metadata_dir is None:
        filtered_metadata_dir = Path(params["folder"]) / "filtered_metadata"
    else:
        filtered_metadata_dir = Path(filtered_metadata_dir)
    filtered_metadata_dir.mkdir(parents=True, exist_ok=True)

    new_paths = []
    new_weights = []
    old_weights = data_cfg.get("datasets_weights")
    found_all = set()
    reports = []
    for idx, data_path in enumerate(dataset_paths):
        output_path = filtered_metadata_dir / f"{split_stem}_{idx}.jsonl"
        filtered_path, found_ids, kept_count, source_metadata = _write_filtered_metadata(
            data_path, train_ids, output_path
        )
        found_all.update(found_ids)
        if kept_count > 0:
            new_paths.append(filtered_path)
            if old_weights is not None:
                new_weights.append(old_weights[idx])
        reports.append(
            {
                "source_metadata": source_metadata,
                "filtered_metadata": filtered_path,
                "kept_videos": kept_count,
            }
        )

    missing_ids = sorted(train_ids - found_all)
    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} ids from CoVLA train split were not found in metadata; "
            f"first={missing_ids[:5]}"
        )
    if not new_paths:
        raise RuntimeError(f"No CoVLA train videos kept from split: {split_path}")
    filtered_video_count = sum(report["kept_videos"] for report in reports)
    if filtered_video_count != len(train_ids):
        raise RuntimeError(
            "Filtered CoVLA metadata count does not match the unique encoder "
            f"split IDs: filtered={filtered_video_count} split_ids={len(train_ids)}"
        )

    data_cfg["datasets"] = new_paths
    if old_weights is not None:
        data_cfg["datasets_weights"] = new_weights
    if has_covla_domain:
        covla_domain_cfg["root"] = new_paths
        if old_weights is not None and covla_domain_cfg.get("datasets_weights") is not None:
            covla_domain_cfg["datasets_weights"] = new_weights
    params["covla_train_split"] = {
        "split_path": str(split_path),
        "num_train_ids": len(train_ids),
        "verified_num_train_videos": filtered_video_count,
        "expected_num_train_videos": expected_train_videos,
        "reports": reports,
    }
    return params


def _unique_preserve_order(values):
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _agibot_video_keys(domain_cfg):
    video_keys = domain_cfg.get(
        "video_keys",
        domain_cfg.get("video_key", "observation.images.top_head"),
    )
    if isinstance(video_keys, (list, tuple)):
        normalized = [str(video_key) for video_key in video_keys]
    else:
        normalized = [str(video_keys)]
    normalized = _unique_preserve_order(normalized)
    if not normalized:
        raise RuntimeError("AgiBot requires at least one training video key")
    return normalized


def _agibot_video_key_count(domain_cfg):
    return len(_agibot_video_keys(domain_cfg))


def _apply_agibot_sample_limit(domain_cfg, split_keys):
    view_sample_counts = domain_cfg.get("encoder_train_view_num_samples")
    if view_sample_counts is not None:
        if not isinstance(view_sample_counts, dict):
            raise RuntimeError(
                "AgiBot encoder_train_view_num_samples must be a mapping from "
                "video key to sample count"
            )
        video_keys = _agibot_video_keys(domain_cfg)
        configured_views = {str(key) for key in view_sample_counts}
        expected_views = set(video_keys)
        if configured_views != expected_views:
            raise RuntimeError(
                "AgiBot encoder_train_view_num_samples keys must exactly match "
                f"the configured training views; missing={sorted(expected_views - configured_views)} "
                f"extra={sorted(configured_views - expected_views)}"
            )

        normalized_counts = {
            video_key: int(view_sample_counts[video_key])
            for video_key in video_keys
        }
        nonpositive = {
            video_key: count
            for video_key, count in normalized_counts.items()
            if count <= 0
        }
        if nonpositive:
            raise RuntimeError(
                "AgiBot encoder_train_view_num_samples values must be positive; "
                f"got {nonpositive}"
            )

        requested_num_samples = sum(normalized_counts.values())
        configured_total = domain_cfg.get("encoder_train_num_samples")
        if (
            configured_total is not None
            and int(configured_total) != requested_num_samples
        ):
            raise RuntimeError(
                "AgiBot encoder_train_num_samples must equal the sum of "
                "encoder_train_view_num_samples; "
                f"got total={configured_total}, view_sum={requested_num_samples}"
            )
        if requested_num_samples > len(split_keys):
            raise RuntimeError(
                "AgiBot disjoint per-view quotas exceed the encoder split size; "
                f"requested={requested_num_samples}, available={len(split_keys)}"
            )

        sample_seed = int(
            domain_cfg.get(
                "encoder_train_sample_seed",
                domain_cfg.get("stats_seed", 0),
            )
        )
        shuffled_keys = list(split_keys)
        random.Random(sample_seed).shuffle(shuffled_keys)
        view_episode_allowlists = {}
        offset = 0
        for video_key in video_keys:
            count = normalized_counts[video_key]
            view_episode_allowlists[video_key] = sorted(
                shuffled_keys[offset : offset + count]
            )
            offset += count
        selected_keys = shuffled_keys[:offset]
        return selected_keys, view_episode_allowlists, {
            "requested_num_samples": requested_num_samples,
            "num_views": len(video_keys),
            "view_num_samples": normalized_counts,
            "num_allowlist_keys": len(selected_keys),
            "estimated_num_samples": len(selected_keys),
            "sample_seed": sample_seed,
            "view_allowlists_disjoint": True,
        }

    target_samples = domain_cfg.get("encoder_train_num_samples")
    if target_samples is None:
        return list(split_keys), None, None

    target_samples = int(target_samples)
    if target_samples <= 0:
        raise RuntimeError("AgiBot encoder_train_num_samples must be positive or null")

    num_views = _agibot_video_key_count(domain_cfg)
    if target_samples % num_views != 0:
        raise RuntimeError(
            "AgiBot encoder_train_num_samples must be divisible by the number of training views; "
            f"got encoder_train_num_samples={target_samples}, num_views={num_views}"
        )
    max_split_keys = target_samples // num_views
    if max_split_keys <= 0:
        raise RuntimeError(
            "AgiBot encoder_train_num_samples is too small for the configured training views"
        )
    if max_split_keys >= len(split_keys):
        limited_keys = list(split_keys)
    else:
        sample_seed = int(domain_cfg.get("encoder_train_sample_seed", domain_cfg.get("stats_seed", 0)))
        rng = random.Random(sample_seed)
        limited_keys = rng.sample(list(split_keys), max_split_keys)
    return limited_keys, None, {
        "requested_num_samples": target_samples,
        "num_views": num_views,
        "num_allowlist_keys": len(limited_keys),
        "estimated_num_samples": len(limited_keys) * num_views,
        "sample_seed": int(domain_cfg.get("encoder_train_sample_seed", domain_cfg.get("stats_seed", 0))),
    }


def apply_agibot_encoder_train_split(params, split_path):
    """Restrict AgiBot encoder training to the persisted episode/window keys."""
    time_jepa_cfg = params.get("time_jepa", {}) or {}
    time_jepa_mode = str(time_jepa_cfg.get("mode", "")).lower()
    if time_jepa_mode not in ("multi", "multidomain", "multi_domain"):
        raise RuntimeError(
            "--agibot-encoder-train-split currently requires time_jepa.mode=multi_domain "
            "(AgiBot-only training can use multi_domain with a single agibot domain)"
        )
    domains = time_jepa_cfg.get("domains", {}) or {}
    agibot_domains = [
        cfg
        for cfg in domains.values()
        if isinstance(cfg, dict) and str(cfg.get("domain", "")).lower() == "agibot"
    ]
    if not agibot_domains:
        raise RuntimeError("No agibot domain found under time_jepa.domains")

    split_path = Path(split_path).expanduser().resolve()
    with split_path.open("r") as f:
        episode_keys = json.load(f)
    if not episode_keys:
        raise RuntimeError(f"Empty AgiBot encoder split: {split_path}")
    episode_keys = _unique_preserve_order(episode_keys)
    reports = []
    for cfg in agibot_domains:
        (
            domain_episode_keys,
            view_episode_allowlists,
            limit_report,
        ) = _apply_agibot_sample_limit(cfg, episode_keys)
        cfg["episode_allowlist"] = sorted(domain_episode_keys)
        if view_episode_allowlists is not None:
            cfg["view_episode_allowlists"] = view_episode_allowlists
        else:
            cfg.pop("view_episode_allowlists", None)
        report = {
            "name": cfg.get("name"),
            "video_key": cfg.get("video_key", "observation.images.top_head"),
            "num_allowlist_keys": len(domain_episode_keys),
            "estimated_num_samples": (
                len(domain_episode_keys)
                if view_episode_allowlists is not None
                else len(domain_episode_keys) * _agibot_video_key_count(cfg)
            ),
        }
        if view_episode_allowlists is not None:
            report["view_num_samples"] = {
                video_key: len(keys)
                for video_key, keys in view_episode_allowlists.items()
            }
            report["view_allowlists_disjoint"] = True
        if limit_report is not None:
            report.update(limit_report)
        reports.append(report)
    params["agibot_encoder_train_split"] = {
        "split_path": str(split_path),
        "num_episodes": len(episode_keys),
        "num_split_keys": len(episode_keys),
        "reports": reports,
    }
    return params



def apply_egoexo_encoder_train_split(params, split_path):
    time_jepa_cfg = params.get("time_jepa", {}) or {}
    mode = str(time_jepa_cfg.get("mode", "")).lower()
    if mode not in ("multi", "multidomain", "multi_domain"):
        raise RuntimeError("--egoexo-encoder-train-split requires time_jepa.mode=multi_domain")
    domains = time_jepa_cfg.get("domains", {}) or {}
    egoexo_domains = [cfg for cfg in domains.values() if isinstance(cfg, dict) and str(cfg.get("domain", "")).lower() == "egoexo"]
    if not egoexo_domains:
        raise RuntimeError("No egoexo domain found under time_jepa.domains")
    split_path = Path(split_path).expanduser().resolve()
    with split_path.open("r") as handle:
        keys = _unique_preserve_order(json.load(handle))
    if not keys:
        raise RuntimeError(f"Empty EgoExo encoder split: {split_path}")
    for cfg in egoexo_domains:
        cfg["episode_allowlist"] = sorted(keys)
    params["egoexo_encoder_train_split"] = {"split_path": str(split_path), "num_split_keys": len(keys)}
    return params


def configured_encoder_train_split(params, domain_name):
    """Return one domain's YAML-declared encoder split, if configured."""
    return resolve_configured_encoder_train_split(
        params,
        domain_name,
        repo_root=REPO_ROOT,
    )


def configured_encoder_split_enabled(params, domain_name):
    flag_name = f"ENABLE_{str(domain_name).upper()}_SPLIT"
    flag = str(os.environ.get(flag_name, "auto")).strip().lower()
    return resolve_encoder_split_enabled(flag, params, domain_name)


@record
def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    logger = get_logger(force=True)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    logger.info(f"called-params {args.fname}")

    with open(args.fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    covla_train_split = (
        args.covla_encoder_train_split
        or args.covla_train_split
        or (
            configured_encoder_train_split(params, "covla")
            if configured_encoder_split_enabled(params, "covla")
            else None
        )
    )
    if encoder_train_split_required(params, "covla") and not covla_train_split:
        raise RuntimeError(
            "CoVLA config requires data.encoder_train_split, but no split path "
            "was resolved"
        )
    if covla_train_split:
        params = apply_covla_train_split(
            params,
            split_path=covla_train_split,
            filtered_metadata_dir=args.filtered_metadata_dir,
        )
        if rank == 0:
            covla_split_info = params.get("covla_train_split")
            if covla_split_info is not None:
                logger.info("restricted CoVLAVideoDataset to train split: %s", covla_split_info)

    agibot_train_split = (
        args.agibot_encoder_train_split
        or (
            configured_encoder_train_split(params, "agibot")
            if configured_encoder_split_enabled(params, "agibot")
            else None
        )
    )
    if encoder_train_split_required(params, "agibot") and not agibot_train_split:
        raise RuntimeError(
            "AgiBot config requires an encoder_train_split, but no split path "
            "was resolved"
        )
    if agibot_train_split:
        params = apply_agibot_encoder_train_split(params, agibot_train_split)
        if rank == 0:
            logger.info(
                "restricted AgiBot encoder domain to train split: %s",
                params["agibot_encoder_train_split"],
            )

    egoexo_train_split = (
        args.egoexo_encoder_train_split
        or (
            configured_encoder_train_split(params, "egoexo")
            if configured_encoder_split_enabled(params, "egoexo")
            else None
        )
    )
    if encoder_train_split_required(params, "egoexo") and not egoexo_train_split:
        raise RuntimeError(
            "EgoExo config requires an encoder_train_split, but no split path "
            "was resolved"
        )
    if egoexo_train_split:
        params = apply_egoexo_encoder_train_split(params, egoexo_train_split)
        if rank == 0:
            logger.info("restricted EgoExo encoder domain to train split: %s", params["egoexo_encoder_train_split"])

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = Path(params["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        params_path = folder / "params-pretrain.yaml"
        with params_path.open("w") as f:
            yaml.dump(params, f)

    world_size, rank = init_distributed()
    logger.info(f"Running... (rank: {rank}/{world_size})")

    app_main(params["app"], args=params)


if __name__ == "__main__":
    main()
