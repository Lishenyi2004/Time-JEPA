from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from grasp_eval_rjob.config import load_config, protocol_digest
from grasp_eval_rjob.constants import FAMILY_TO_PROPERTIES, FAMILIES, PROPERTIES


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-p{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-p{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw_dir = Path(config["output_dir"]).expanduser().resolve() / "raw"
    paths = sorted(raw_dir.glob("rank_*_of_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No evaluation shards found in {raw_dir}")
    records: list[dict[str, Any]] = []
    by_world: dict[int, list[tuple[int, Path]]] = {}
    for path in paths:
        parts = path.stem.split("_")
        if len(parts) != 4 or parts[0] != "rank" or parts[2] != "of":
            continue
        rank, world_size = int(parts[1]), int(parts[3])
        by_world.setdefault(world_size, []).append((rank, path))
    if len(by_world) != 1:
        raise RuntimeError(f"Found shard sets from multiple world sizes: {sorted(by_world)}")
    world_size, ranked = next(iter(by_world.items()))
    ranks = sorted(rank for rank, _ in ranked)
    if ranks != list(range(world_size)):
        raise RuntimeError(f"Incomplete shard set: {ranks}; expected 0..{world_size - 1}")
    expected_digest = protocol_digest(config)
    dataset_sizes: set[int] = set()
    code_digests: set[str] = set()
    seen: set[str] = set()
    for rank, path in sorted(ranked):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("protocol_digest") != expected_digest:
            raise RuntimeError(
                f"Protocol mismatch in {path}: {shard.get('protocol_digest')} != {expected_digest}"
            )
        if int(shard.get("rank", -1)) != rank or int(shard.get("world_size", -1)) != world_size:
            raise RuntimeError(f"Rank metadata mismatch in {path}")
        dataset_sizes.add(int(shard["dataset_size"]))
        code_sha256 = shard.get("code_sha256")
        if not isinstance(code_sha256, dict) or not code_sha256:
            raise RuntimeError(f"Shard {path} does not record evaluator code hashes")
        code_digests.add(json.dumps(code_sha256, sort_keys=True))
        for record in shard["records"]:
            pair_id = str(record["pair_id"])
            if pair_id in seen:
                raise RuntimeError(f"Duplicate pair record across shards: {pair_id}")
            seen.add(pair_id)
            records.append(record)
    if len(dataset_sizes) != 1:
        raise RuntimeError(f"Shards disagree on dataset size: {sorted(dataset_sizes)}")
    if len(code_digests) != 1:
        raise RuntimeError("Shards were produced by different evaluator source revisions")
    dataset_size = dataset_sizes.pop()
    if len(records) != dataset_size:
        raise RuntimeError(f"Expected {dataset_size} records, found {len(records)}")
    records.sort(key=lambda row: int(row["dataset_index"]))
    expected_indices = list(range(dataset_size))
    actual_indices = [int(row["dataset_index"]) for row in records]
    if actual_indices != expected_indices:
        raise RuntimeError("Pair records do not cover a contiguous deterministic index set")
    return records, dataset_size


def video_score_table(
    records: list[dict[str, Any]],
    *,
    context_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for record in records:
        possible = torch.as_tensor(record["possible_losses"], dtype=torch.float32).numpy()
        impossible = torch.as_tensor(record["impossible_losses"], dtype=torch.float32).numpy()
        starts = torch.as_tensor(record["window_start"], dtype=torch.int64).numpy()
        prediction_starts = torch.as_tensor(record["prediction_start"], dtype=torch.int64).numpy()
        if (
            len(possible) != len(impossible)
            or len(possible) != len(starts)
            or len(starts) != len(prediction_starts)
        ):
            raise RuntimeError(f"Window length mismatch for {record['pair_id']}")
        if not np.array_equal(prediction_starts, starts + int(context_frames)):
            raise RuntimeError(f"Prediction starts are inconsistent for {record['pair_id']}")
        if not np.isfinite(possible).all() or not np.isfinite(impossible).all():
            raise RuntimeError(f"Non-finite surprise for {record['pair_id']}")
        base = {
            "dataset_index": int(record["dataset_index"]),
            "pair_id": str(record["pair_id"]),
            "family": str(record["family"]),
            "file_name": str(record["file_name"]),
            "properties": "|".join(record["properties"]),
            "num_windows": int(len(starts)),
            "context_frames": int(context_frames),
            "frame_step": 10,
            "window_stride": 2,
            "frames_per_clip": 16,
        }
        for plausibility, values, relative_path in (
            ("possible", possible, record["possible_relative_path"]),
            ("impossible", impossible, record["impossible_relative_path"]),
        ):
            for window_index, (start, prediction_start, surprise) in enumerate(
                zip(starts, prediction_starts, values)
            ):
                window_rows.append(
                    {
                        **base,
                        "plausibility": plausibility,
                        "relative_path": str(relative_path),
                        "window_index": int(window_index),
                        "window_start": int(start),
                        "prediction_start": int(prediction_start),
                        "surprise": float(surprise),
                    }
                )
            video_rows.append(
                {
                    **base,
                    "plausibility": plausibility,
                    "relative_path": str(relative_path),
                    "average_surprise": float(values.mean()),
                    "max_surprise": float(values.max()),
                }
            )
        for aggregation, possible_score, impossible_score in (
            ("average", float(possible.mean()), float(impossible.mean())),
            ("maximum", float(possible.max()), float(impossible.max())),
        ):
            pair_rows.append(
                {
                    **base,
                    "aggregation": aggregation,
                    "possible_surprise": possible_score,
                    "impossible_surprise": impossible_score,
                    "margin": impossible_score - possible_score,
                    # Strict inequality agrees with the published evaluator.
                    "correct": int(impossible_score > possible_score),
                }
            )
    return (
        pd.DataFrame(window_rows),
        pd.DataFrame(video_rows),
        pd.DataFrame(pair_rows),
    )


def _bootstrap_macro(
    pair_table: pd.DataFrame,
    *,
    groups: Iterable[str],
    group_membership: dict[str, set[str]],
    aggregation: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    subset = pair_table[pair_table["aggregation"] == aggregation].copy()
    if subset.empty or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_family_means: dict[str, np.ndarray] = {}
    family_sizes: dict[str, int] = {}
    for family, frame in subset.groupby("family", sort=False):
        correct = frame["correct"].to_numpy(dtype=np.float64)
        draws = rng.integers(0, len(correct), size=(samples, len(correct)))
        boot_family_means[str(family)] = correct[draws].mean(axis=1)
        family_sizes[str(family)] = len(correct)

    group_distributions = []
    group_list = list(groups)
    for group in group_list:
        families = sorted(group_membership[group] & set(boot_family_means))
        if not families:
            continue
        denominator = sum(family_sizes[family] for family in families)
        distribution = sum(
            boot_family_means[family] * family_sizes[family] for family in families
        ) / denominator
        group_distributions.append(distribution)
    if not group_distributions:
        return float("nan"), float("nan")
    values = np.stack(group_distributions, axis=0).mean(axis=0) * 100.0
    low, high = np.nanquantile(values, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    pair_table: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    groups_by_type = [("family", FAMILIES), ("property", PROPERTIES)]
    for aggregation in ("average", "maximum"):
        subset = pair_table[pair_table["aggregation"] == aggregation]
        rows.append(
            {
                "aggregation": aggregation,
                "group_type": "overall_micro",
                "group": "All",
                "n_pairs": int(len(subset)),
                "accuracy": float(subset["correct"].mean() * 100.0),
                "margin_mean": float(subset["margin"].mean()),
            }
        )
        for group_type, group_list in groups_by_type:
            if group_type == "family":
                column = "family"
                group_frames = {group: subset[subset[column] == group] for group in group_list}
            else:
                # A family pair is intentionally included in each mapped
                # property; this reproduces the released multi-label code.
                group_frames = {
                    group: subset[
                        subset["family"].map(
                            lambda family: group in FAMILY_TO_PROPERTIES[str(family)]
                        )
                    ]
                    for group in group_list
                }
            accuracies = []
            for group, frame in group_frames.items():
                if frame.empty:
                    continue
                accuracy = float(frame["correct"].mean() * 100.0)
                accuracies.append(accuracy)
                rows.append(
                    {
                        "aggregation": aggregation,
                        "group_type": group_type,
                        "group": group,
                        "n_pairs": int(len(frame)),
                        "accuracy": accuracy,
                        "margin_mean": float(frame["margin"].mean()),
                    }
                )
            macro = float(np.mean(accuracies))
            if group_type == "family":
                membership = {group: {group} for group in group_list}
            else:
                membership = {
                    group: {
                        family
                        for family, mapped in FAMILY_TO_PROPERTIES.items()
                        if group in mapped
                    }
                    for group in group_list
                }
            low, high = _bootstrap_macro(
                subset,
                groups=group_list,
                group_membership=membership,
                aggregation=aggregation,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "aggregation": aggregation,
                    "group_type": f"{group_type}_macro",
                    "group": "All",
                    "n_pairs": int(len(subset)),
                    "accuracy": macro,
                    "margin_mean": float(subset["margin"].mean()),
                    "bootstrap_low": low,
                    "bootstrap_high": high,
                }
            )

    summary = pd.DataFrame(rows)
    property_summary = summary[
        (summary["group_type"] == "property") & (summary["aggregation"] == "average")
    ].copy()
    return summary, property_summary


def score(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, dataset_size = load_records(config)
    evaluation = config["evaluation"]
    windows, videos, pairs = video_score_table(
        records, context_frames=int(evaluation["context_frames"])
    )
    summary, property_summary = summarize(
        pairs,
        bootstrap_samples=int(config["scoring"]["bootstrap_samples"]),
        bootstrap_seed=int(config["scoring"]["bootstrap_seed"]),
    )
    output_dir = Path(config["output_dir"]).expanduser().resolve() / "scores"
    _atomic_csv(windows, output_dir / "per_window.csv")
    _atomic_csv(videos, output_dir / "per_video.csv")
    _atomic_csv(pairs, output_dir / "per_pair.csv")
    _atomic_csv(summary, output_dir / "summary.csv")
    _atomic_csv(
        summary[summary["group_type"] == "family"],
        output_dir / "family_summary.csv",
    )
    _atomic_csv(property_summary, output_dir / "property_summary.csv")
    _atomic_json(
        {
            "protocol_digest": protocol_digest(config),
            "dataset_size": dataset_size,
            "video_count": int(len(videos)),
            "pair_count": int(len(pairs) // 2),
            "window_count": int(len(windows)),
            "context_frames": int(evaluation["context_frames"]),
            "aggregation_primary": "average",
        },
        output_dir / "score_manifest.json",
    )
    return windows, videos, pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    _, _, _, summary = score(config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
