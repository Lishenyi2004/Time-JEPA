#!/usr/bin/env python3
"""Aggregate five-seed GT-grounded COVLA metrics and paired video bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .bank_common import PRIMARY_HORIZONS, atomic_json_dump, json_safe
    from .grounded_eval import SCOPES
except ImportError:
    from bank_common import PRIMARY_HORIZONS, atomic_json_dump, json_safe
    from grounded_eval import SCOPES


MODELS = ("time", "video_only", "zero")
SPLITS = ("test", "weather_ood_test")
LOWER_IS_BETTER_METRICS = (
    "direct_trajectory_to_gt_mae_2d_m",
    "rollout_to_gt_trajectory_mae_2d_m",
    "rollout_to_gt_rollout_mae_2d_m",
)
SEED_METRICS = LOWER_IS_BETTER_METRICS + (
    "gt_oracle_floor_mae_2d_m",
    "gt_anchored_conditioned_macro_auroc",
)


def _mean_std(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std_across_probe_seeds": float(np.std(values, ddof=1)),
        "values": values.tolist(),
    }


def _bootstrap_ci(estimate: float, values: np.ndarray) -> dict:
    return {
        "estimate": float(estimate),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _metric_from_json(payload: dict, scope: str, horizon: str) -> dict:
    overall = payload["grounded_metrics"][scope][horizon]["conditions"]["overall"]
    hard = payload["gt_anchored_hard_negative"][scope][horizon]
    return {
        "direct_trajectory_to_gt_mae_2d_m": overall["direct_trajectory_to_gt"][
            "mae_2d_m"
        ],
        "rollout_to_gt_trajectory_mae_2d_m": overall[
            "rollout_to_gt_trajectory"
        ]["mae_2d_m"],
        "rollout_to_gt_rollout_mae_2d_m": overall["rollout_to_gt_rollout"][
            "mae_2d_m"
        ],
        "gt_oracle_floor_mae_2d_m": overall["gt_oracle_floor"]["mae_2d_m"],
        "gt_anchored_conditioned_macro_auroc": hard["conditioned_macro_auroc"],
        "sample_count": overall["count"],
        "gross_excluded_count": overall["gross_excluded_count"],
    }


def _load_per_video(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            count = int(row["count"])
            if count <= 0:
                continue
            parsed = {
                "video_id": row["video_id"],
                "capture_date": row["capture_date"],
                "horizon": str(float(row["horizon_seconds"])),
                "scope": row["scope"],
                "count": count,
            }
            for metric in LOWER_IS_BETTER_METRICS:
                parsed[metric] = float(row[metric])
            rows.append(parsed)
    return rows


def aggregate_grounded(
    root: str | Path,
    seeds: list[int],
    bootstrap: int,
    random_seed: int,
    models=MODELS,
    reference_model: str = "time",
) -> dict:
    root = Path(root).expanduser().resolve()
    models = tuple(str(model) for model in models)
    if not models or len(set(models)) != len(models):
        raise ValueError(f"models must be non-empty and unique: {models}")
    if reference_model not in models:
        raise ValueError(
            f"reference_model={reference_model!r} is not present in models={models}"
        )
    seed_rows = []
    per_video = defaultdict(lambda: defaultdict(list))
    capture_dates = {}
    source_hashes = {}
    quality_audit = {}
    for model in models:
        for seed in seeds:
            run = root / model / f"seed_{seed}"
            for split in SPLITS:
                json_path = run / f"{split}_grounded.json"
                csv_path = run / f"{split}_grounded_per_video.csv"
                if not json_path.is_file() or not csv_path.is_file():
                    raise FileNotFoundError(
                        f"Missing grounded output for {model}/seed_{seed}/{split}"
                    )
                payload = json.loads(json_path.read_text())
                if (
                    payload.get("model_name") != model
                    or int(payload.get("probe_seed")) != seed
                    or payload.get("split") != split
                ):
                    raise RuntimeError(f"Grounded metadata mismatch: {json_path}")
                source_hashes[f"{model}/seed_{seed}/{split}"] = payload[
                    "prediction_sha256"
                ]
                quality_audit.setdefault(split, payload["gross_quality_audit"])
                if quality_audit[split] != payload["gross_quality_audit"]:
                    raise RuntimeError(
                        f"GT quality audit differs across model/seed for {split}"
                    )
                for scope in SCOPES:
                    for horizon, _ in PRIMARY_HORIZONS:
                        key = str(horizon)
                        seed_rows.append(
                            {
                                "model": model,
                                "seed": int(seed),
                                "split": split,
                                "scope": scope,
                                "horizon": key,
                                **_metric_from_json(payload, scope, key),
                            }
                        )
                for row in _load_per_video(csv_path):
                    key = (
                        model,
                        split,
                        row["scope"],
                        row["horizon"],
                        row["video_id"],
                    )
                    for metric in LOWER_IS_BETTER_METRICS:
                        per_video[key][metric].append(row[metric])
                    date_key = (split, row["video_id"])
                    previous = capture_dates.setdefault(date_key, row["capture_date"])
                    if previous != row["capture_date"]:
                        raise RuntimeError(f"Capture-date mismatch for {date_key}")

    seed_summary = {}
    for model in models:
        seed_summary[model] = {}
        for split in SPLITS:
            seed_summary[model][split] = {}
            for scope in SCOPES:
                seed_summary[model][split][scope] = {}
                for horizon, _ in PRIMARY_HORIZONS:
                    key = str(horizon)
                    selected = [
                        row
                        for row in seed_rows
                        if row["model"] == model
                        and row["split"] == split
                        and row["scope"] == scope
                        and row["horizon"] == key
                    ]
                    if len(selected) != len(seeds):
                        raise RuntimeError(
                            f"Expected {len(seeds)} rows for {model}/{split}/{scope}/{key}"
                        )
                    seed_summary[model][split][scope][key] = {
                        metric: _mean_std([row[metric] for row in selected])
                        for metric in SEED_METRICS
                    }

    rng = np.random.default_rng(int(random_seed))
    comparisons = {}
    for comparator in models:
        if comparator == reference_model:
            continue
        comparison_name = f"{reference_model}_vs_{comparator}"
        comparisons[comparison_name] = {}
        for split in SPLITS:
            comparisons[comparison_name][split] = {}
            for scope in SCOPES:
                comparisons[comparison_name][split][scope] = {}
                for horizon, _ in PRIMARY_HORIZONS:
                    key = str(horizon)
                    reference_ids = {
                        item[4]
                        for item in per_video
                        if item[:4] == (reference_model, split, scope, key)
                    }
                    comparator_ids = {
                        item[4]
                        for item in per_video
                        if item[:4] == (comparator, split, scope, key)
                    }
                    if reference_ids != comparator_ids:
                        raise RuntimeError(
                            f"Paired IDs differ for {comparison_name}/{split}/{scope}/{key}"
                        )
                    ids = sorted(reference_ids)
                    date_groups = defaultdict(list)
                    for index, video_id in enumerate(ids):
                        date_groups[capture_dates[(split, video_id)]].append(index)
                    metric_output = {}
                    for metric in LOWER_IS_BETTER_METRICS:
                        reference_values = np.asarray(
                            [
                                np.mean(
                                    per_video[(reference_model, split, scope, key, video_id)][
                                        metric
                                    ]
                                )
                                for video_id in ids
                            ]
                        )
                        comparator_values = np.asarray(
                            [
                                np.mean(
                                    per_video[(comparator, split, scope, key, video_id)][
                                        metric
                                    ]
                                )
                                for video_id in ids
                            ]
                        )
                        delta = comparator_values - reference_values
                        boot = np.empty(bootstrap, dtype=np.float64)
                        for replicate in range(bootstrap):
                            if split == "weather_ood_test":
                                sampled = np.concatenate(
                                    [
                                        rng.choice(indices, size=len(indices), replace=True)
                                        for indices in date_groups.values()
                                    ]
                                )
                            else:
                                sampled = rng.integers(0, len(ids), size=len(ids))
                            boot[replicate] = float(np.mean(delta[sampled]))
                        estimate = float(np.mean(delta))
                        comparator_mean = float(np.mean(comparator_values))
                        metric_output[
                            f"{metric}_delta_comparator_minus_reference_positive_favors_reference"
                        ] = {
                            **_bootstrap_ci(estimate, boot),
                            "relative_improvement_percent": float(
                                100.0 * estimate / max(comparator_mean, 1e-12)
                            ),
                            "p_reference_not_better": float(
                                (1 + np.sum(boot <= 0)) / (bootstrap + 1)
                            ),
                        }
                        if reference_model == "time":
                            metric_output[
                                f"{metric}_delta_baseline_minus_time_positive_favors_time"
                            ] = {
                                **metric_output[
                                    f"{metric}_delta_comparator_minus_reference_positive_favors_reference"
                                ],
                                "p_time_not_better": metric_output[
                                    f"{metric}_delta_comparator_minus_reference_positive_favors_reference"
                                ]["p_reference_not_better"],
                            }
                    comparisons[comparison_name][split][scope][key] = {
                        "video_count": int(len(ids)),
                        "capture_date_counts": {
                            date: len(indices)
                            for date, indices in sorted(date_groups.items())
                        },
                        **metric_output,
                    }

    paired_probe_seed = {}
    lookup = {
        (row["model"], row["seed"], row["split"], row["scope"], row["horizon"]): row
        for row in seed_rows
    }
    for comparator in models:
        if comparator == reference_model:
            continue
        comparison_name = f"{reference_model}_vs_{comparator}"
        paired_probe_seed[comparison_name] = {}
        for split in SPLITS:
            paired_probe_seed[comparison_name][split] = {}
            for scope in SCOPES:
                paired_probe_seed[comparison_name][split][scope] = {}
                for horizon, _ in PRIMARY_HORIZONS:
                    key = str(horizon)
                    metric_deltas = defaultdict(list)
                    for seed in seeds:
                        reference_row = lookup[
                            (reference_model, seed, split, scope, key)
                        ]
                        comparator_row = lookup[
                            (comparator, seed, split, scope, key)
                        ]
                        for metric in LOWER_IS_BETTER_METRICS:
                            metric_deltas[
                                f"{metric}_comparator_minus_reference"
                            ].append(
                                comparator_row[metric] - reference_row[metric]
                            )
                        metric_deltas[
                            "gt_anchored_auroc_reference_minus_comparator"
                        ].append(
                            reference_row["gt_anchored_conditioned_macro_auroc"]
                            - comparator_row["gt_anchored_conditioned_macro_auroc"]
                        )
                    if reference_model == "time":
                        for metric in LOWER_IS_BETTER_METRICS:
                            metric_deltas[f"{metric}_baseline_minus_time"] = list(
                                metric_deltas[
                                    f"{metric}_comparator_minus_reference"
                                ]
                            )
                        metric_deltas[
                            "gt_anchored_auroc_time_minus_baseline"
                        ] = list(
                            metric_deltas[
                                "gt_anchored_auroc_reference_minus_comparator"
                            ]
                        )
                    paired_probe_seed[comparison_name][split][scope][key] = {
                        name: _mean_std(values)
                        for name, values in metric_deltas.items()
                    }

    per_date = {}
    for model in models:
        per_date[model] = {}
        for split in SPLITS:
            per_date[model][split] = {}
            for scope in SCOPES:
                per_date[model][split][scope] = {}
                for horizon, _ in PRIMARY_HORIZONS:
                    key = str(horizon)
                    date_values = defaultdict(lambda: defaultdict(list))
                    for item_key, metrics in per_video.items():
                        item_model, item_split, item_scope, item_horizon, video_id = item_key
                        if (item_model, item_split, item_scope, item_horizon) != (
                            model,
                            split,
                            scope,
                            key,
                        ):
                            continue
                        date = capture_dates[(split, video_id)]
                        for metric in LOWER_IS_BETTER_METRICS:
                            date_values[date][metric].append(float(np.mean(metrics[metric])))
                    per_date[model][split][scope][key] = {
                        date: {
                            "video_count": len(values[LOWER_IS_BETTER_METRICS[0]]),
                            **{
                                metric: float(np.mean(values[metric]))
                                for metric in LOWER_IS_BETTER_METRICS
                            },
                        }
                        for date, values in sorted(date_values.items())
                    }

    controls = {}
    for split in SPLITS:
        control_path = root / "controls" / f"{split}_controls.json"
        if not control_path.is_file():
            raise FileNotFoundError(f"Missing grounded controls: {control_path}")
        controls[split] = json.loads(control_path.read_text())

    return {
        "schema": 1,
        "experiment": "covla_gt_grounded_kinematic_evaluation_v1",
        "root": str(root),
        "models": list(models),
        "reference_model": reference_model,
        "probe_seeds": [int(seed) for seed in seeds],
        "primary_scope": "all",
        "sensitivity_scope": "gross_screened",
        "quality_audit": quality_audit,
        "seed_summary": seed_summary,
        "paired_video_bootstrap": {
            "replicates": int(bootstrap),
            "seed": int(random_seed),
            "method": "average five probe seeds within video, then paired resample videos; weather OOD preserves capture-date counts",
            "comparisons": comparisons,
        },
        "paired_probe_seed_diagnostics": paired_probe_seed,
        "capture_date_summary_after_averaging_probe_seeds": per_date,
        "degenerate_controls": controls,
        "source_prediction_hashes": source_hashes,
        "seed_rows": seed_rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    default_root = Path(__file__).resolve().parent / "results" / "grounded_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=default_root)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--reference-model", default="time")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=73100)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError("Formal aggregate is frozen to probe seeds 0,1,2,3,4")
    models = tuple(
        item.strip() for item in args.models.split(",") if item.strip()
    )
    result = aggregate_grounded(
        args.root,
        seeds,
        args.bootstrap,
        args.random_seed,
        models=models,
        reference_model=args.reference_model,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.root).expanduser().resolve() / "aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(json_safe(result), output_dir / "aggregate_summary.json")
    _write_csv(output_dir / "seed_metrics.csv", result["seed_rows"])
    print(
        json.dumps(
            {"status": "COMPLETE", "output_dir": str(output_dir)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
