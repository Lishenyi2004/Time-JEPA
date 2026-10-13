#!/usr/bin/env python3
"""Aggregate the 3-model x 5-seed AgiBot reach-rate experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from physics_consistency.agibot_reach_rate.common import (
    ARMS,
    PROTOCOL,
    TARGET_DEFINITION,
    atomic_json_dump,
    json_safe,
)


MODELS = ("time", "video_only", "zero")
FULL_SPLITS = (
    "id__top_head",
    "id__hand_left",
    "id__hand_right",
    "ood__head_left_fisheye",
    "ood__head_right_fisheye",
)
METRICS = (
    "q_macro_nrmse",
    "direction_mean_deg",
    "velocity_macro_nrmse",
    "formula_mae",
    "formula_nmae",
    "formula_rmse",
    "formula_pearson",
    "active_formula_mae",
    "active_formula_nmae",
    "active_formula_rmse",
    "active_formula_pearson",
    "gt_q_pred_v_mae",
    "pred_q_gt_v_mae",
    "gt_q_gt_v_mae",
    "raw_geometry_oracle_mae",
    "zero_mae",
    "train_mean_mae",
    "active_zero_mae",
    "active_train_mean_mae",
    "active_error_reduction_vs_zero",
    "active_error_reduction_vs_train_mean",
    "wrong_arm_preference",
    "time_shift_preference",
    "active_wrong_arm_preference",
    "active_time_shift_preference",
)
LOWER_IS_BETTER = {
    "q_macro_nrmse",
    "direction_mean_deg",
    "velocity_macro_nrmse",
    "formula_mae",
    "formula_nmae",
    "formula_rmse",
    "active_formula_mae",
    "active_formula_nmae",
    "active_formula_rmse",
    "gt_q_pred_v_mae",
    "pred_q_gt_v_mae",
    "active_zero_mae",
    "active_train_mean_mae",
}
COMPARISON_METRICS = (
    "active_formula_mae",
    "formula_mae",
    "velocity_macro_nrmse",
    "q_macro_nrmse",
    "direction_mean_deg",
    "active_wrong_arm_preference",
    "active_time_shift_preference",
)


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else np.nan


def _mean_std(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "mean": float(np.mean(array)) if len(array) else np.nan,
        "std_across_probe_seeds": (
            float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        ),
        "seed_count": int(len(array)),
        "values": array.tolist(),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _read_episode_csv(
    path: Path,
    model: str,
    seed: int,
    split: str,
) -> list[dict]:
    output = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "split": split,
                    "episode_key": row["episode_key"],
                    "task_id": int(row["task_id"]),
                    "arm": row["arm"],
                    **{
                        metric: _float(row.get(metric))
                        for metric in METRICS
                    },
                }
            )
    return output


def _read_pooled_json(
    path: Path,
    model: str,
    seed: int,
    split: str,
) -> list[dict]:
    payload = json.loads(path.read_text())
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("target_definition") != TARGET_DEFINITION
        or payload.get("model") != model
        or int(payload.get("seed", -1)) != int(seed)
        or payload.get("split") != split
    ):
        raise RuntimeError(f"Metric identity mismatch: {path}")
    rows = []
    for arm in ARMS:
        values = payload["arm_flattened_metrics"][arm]
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "split": split,
                "arm": arm,
                **{metric: _float(values.get(metric)) for metric in METRICS},
            }
        )
    rows.append(
        {
            "model": model,
            "seed": int(seed),
            "split": split,
            "arm": "macro",
            **{
                metric: _float(
                    payload["flattened_metrics"].get(metric)
                )
                for metric in METRICS
            },
        }
    )
    return rows


def _scope_definitions(
    splits: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    scopes = {split: (split,) for split in splits}
    id_splits = tuple(split for split in splits if split.startswith("id__"))
    ood_splits = tuple(
        split for split in splits if split.startswith("ood__")
    )
    if id_splits:
        scopes["id_all"] = id_splits
    if ood_splits:
        scopes["ood_all"] = ood_splits
    scopes["all_views"] = splits
    return scopes


def _collapse_scope(
    rows: list[dict], members: tuple[str, ...]
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] in members:
            key = (
                row["model"],
                row["seed"],
                row["episode_key"],
                row["task_id"],
                row["arm"],
            )
            grouped[key].append(row)
    output = []
    for (model, seed, episode, task, arm), values in grouped.items():
        output.append(
            {
                "model": model,
                "seed": seed,
                "episode_key": episode,
                "task_id": task,
                "arm": arm,
                "view_count": len(values),
                **{
                    metric: _mean(row[metric] for row in values)
                    for metric in METRICS
                },
            }
        )
    return output


def _task_macro(rows: list[dict], metric: str) -> float:
    by_task = defaultdict(list)
    for row in rows:
        if np.isfinite(row[metric]):
            by_task[row["task_id"]].append(row[metric])
    return _mean(_mean(values) for values in by_task.values())


def _paired_values(
    rows: list[dict],
    reference_model: str,
    comparator: str,
    arm: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = defaultdict(list)
    tasks = {}
    for row in rows:
        if row["model"] not in {reference_model, comparator} or row["arm"] != arm:
            continue
        key = (row["model"], row["episode_key"])
        grouped[key].append(row[metric])
        old = tasks.setdefault(row["episode_key"], row["task_id"])
        if old != row["task_id"]:
            raise RuntimeError(
                f"Task mismatch for episode {row['episode_key']}"
            )
    reference_episodes = {
        episode for model, episode in grouped if model == reference_model
    }
    comparator_episodes = {
        episode for model, episode in grouped if model == comparator
    }
    if reference_episodes != comparator_episodes:
        raise RuntimeError(
            f"Unpaired episodes for {reference_model} vs {comparator}/{arm}/{metric}"
        )
    deltas, task_ids = [], []
    for episode in sorted(reference_episodes):
        reference_value = _mean(grouped[(reference_model, episode)])
        comparator_value = _mean(grouped[(comparator, episode)])
        if not np.isfinite(reference_value) or not np.isfinite(comparator_value):
            continue
        delta = (
            comparator_value - reference_value
            if metric in LOWER_IS_BETTER
            else reference_value - comparator_value
        )
        deltas.append(delta)
        task_ids.append(tasks[episode])
    return (
        np.asarray(deltas, dtype=np.float64),
        np.asarray(task_ids, dtype=np.int64),
    )


def _stratified_bootstrap(
    deltas: np.ndarray,
    tasks: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
    reference_model: str = "time",
) -> dict:
    task_ids = sorted(set(tasks.tolist()))
    if not task_ids:
        return {"count": 0}
    observed = _mean(
        np.mean(deltas[tasks == task]) for task in task_ids
    )
    samples = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        task_means = []
        for task in task_ids:
            values = deltas[tasks == task]
            indices = rng.integers(0, len(values), size=len(values))
            task_means.append(float(np.mean(values[indices])))
        samples[replicate] = np.mean(task_means)
    result = {
        "count": int(len(deltas)),
        "task_count": int(len(task_ids)),
        "improvement_reference_vs_comparator_or_error_reduction": observed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "reference_model": reference_model,
        "probability_reference_better": float(np.mean(samples > 0.0)),
    }
    if reference_model == "time":
        result["improvement_time_minus_baseline_or_error_reduction"] = observed
        result["probability_time_better"] = result[
            "probability_reference_better"
        ]
    return result


def aggregate(
    root: str | Path,
    output_dir: str | Path,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    splits: tuple[str, ...] = FULL_SPLITS,
    reference_model: str = "time",
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20260729,
) -> dict:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    models = tuple(str(model) for model in models)
    seeds = tuple(int(seed) for seed in seeds)
    if not models or len(set(models)) != len(models):
        raise ValueError(f"models must be non-empty and unique: {models}")
    if reference_model not in models:
        raise ValueError(
            f"reference_model={reference_model!r} is not present in models={models}"
        )
    all_episode_rows, pooled_rows = [], []
    completion_paths = []
    for model in models:
        for seed in seeds:
            run_dir = root / model / f"seed_{seed}"
            completion_path = run_dir / "COMPLETED.json"
            if not completion_path.is_file():
                raise FileNotFoundError(
                    f"Missing completed run: {completion_path}"
                )
            completion = json.loads(completion_path.read_text())
            if (
                completion.get("status") != "COMPLETE"
                or completion.get("protocol") != PROTOCOL
                or completion.get("target_definition")
                != TARGET_DEFINITION
                or completion.get("model_name") != model
                or int(completion.get("probe_seed", -1)) != seed
            ):
                raise RuntimeError(
                    f"Completion identity mismatch: {completion_path}"
                )
            completion_paths.append(str(completion_path))
            for split in splits:
                metrics_path = run_dir / "metrics" / f"{split}.json"
                episode_path = (
                    run_dir / "metrics" / f"{split}_per_episode.csv"
                )
                if not metrics_path.is_file() or not episode_path.is_file():
                    raise FileNotFoundError(
                        f"Missing metrics for {model}/seed_{seed}/{split}"
                    )
                pooled_rows.extend(
                    _read_pooled_json(
                        metrics_path, model, seed, split
                    )
                )
                all_episode_rows.extend(
                    _read_episode_csv(
                        episode_path, model, seed, split
                    )
                )

    scopes = _scope_definitions(splits)
    collapsed = {
        scope: _collapse_scope(all_episode_rows, members)
        for scope, members in scopes.items()
    }
    per_seed_rows = []
    per_task_seed_rows = []
    for scope, rows in collapsed.items():
        for model in models:
            for seed in seeds:
                for arm in ARMS:
                    selected = [
                        row
                        for row in rows
                        if row["model"] == model
                        and row["seed"] == seed
                        and row["arm"] == arm
                    ]
                    if not selected:
                        raise RuntimeError(
                            f"No rows for {scope}/{model}/seed_{seed}/{arm}"
                        )
                    per_seed_rows.append(
                        {
                            "scope": scope,
                            "model": model,
                            "seed": seed,
                            "arm": arm,
                            "episode_count": len(selected),
                            "task_count": len(
                                {row["task_id"] for row in selected}
                            ),
                            **{
                                metric: _task_macro(selected, metric)
                                for metric in METRICS
                            },
                        }
                    )
                    for task_id in sorted(
                        {row["task_id"] for row in selected}
                    ):
                        task_rows = [
                            row
                            for row in selected
                            if row["task_id"] == task_id
                        ]
                        per_task_seed_rows.append(
                            {
                                "scope": scope,
                                "model": model,
                                "seed": seed,
                                "arm": arm,
                                "task_id": int(task_id),
                                "episode_count": len(task_rows),
                                **{
                                    metric: _mean(
                                        row[metric] for row in task_rows
                                    )
                                    for metric in METRICS
                                },
                            }
                        )

    main_rows = []
    summary = {}
    for scope in scopes:
        summary[scope] = {}
        for model in models:
            summary[scope][model] = {}
            for arm in ARMS:
                selected = [
                    row
                    for row in per_seed_rows
                    if row["scope"] == scope
                    and row["model"] == model
                    and row["arm"] == arm
                ]
                metric_summary = {
                    metric: _mean_std(
                        row[metric] for row in selected
                    )
                    for metric in METRICS
                }
                summary[scope][model][arm] = metric_summary
                main_rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "arm": arm,
                        "seed_count": len(selected),
                        "active_formula_mae_mps_mean": metric_summary[
                            "active_formula_mae"
                        ]["mean"],
                        "active_formula_mae_mps_std": metric_summary[
                            "active_formula_mae"
                        ]["std_across_probe_seeds"],
                        "active_formula_mae_cmps_mean": 100.0
                        * metric_summary["active_formula_mae"]["mean"],
                        "active_formula_mae_cmps_std": 100.0
                        * metric_summary["active_formula_mae"][
                            "std_across_probe_seeds"
                        ],
                        "formula_mae_cmps_mean": 100.0
                        * metric_summary["formula_mae"]["mean"],
                        "formula_mae_cmps_std": 100.0
                        * metric_summary["formula_mae"][
                            "std_across_probe_seeds"
                        ],
                        "formula_pearson_mean": metric_summary[
                            "formula_pearson"
                        ]["mean"],
                        "active_zero_mae_cmps_mean": 100.0
                        * metric_summary["active_zero_mae"]["mean"],
                        "active_train_mean_mae_cmps_mean": 100.0
                        * metric_summary["active_train_mean_mae"][
                            "mean"
                        ],
                        "active_error_reduction_vs_zero_cmps_mean": 100.0
                        * metric_summary[
                            "active_error_reduction_vs_zero"
                        ]["mean"],
                        "active_error_reduction_vs_train_mean_cmps_mean": 100.0
                        * metric_summary[
                            "active_error_reduction_vs_train_mean"
                        ]["mean"],
                        "velocity_macro_nrmse_mean": metric_summary[
                            "velocity_macro_nrmse"
                        ]["mean"],
                        "q_macro_nrmse_mean": metric_summary[
                            "q_macro_nrmse"
                        ]["mean"],
                        "direction_mean_deg": metric_summary[
                            "direction_mean_deg"
                        ]["mean"],
                        "active_wrong_arm_preference": metric_summary[
                            "active_wrong_arm_preference"
                        ]["mean"],
                        "active_time_shift_preference": metric_summary[
                            "active_time_shift_preference"
                        ]["mean"],
                    }
                )

    rng = np.random.default_rng(int(bootstrap_seed))
    comparison_rows = []
    for scope, rows in collapsed.items():
        for comparator in models:
            if comparator == reference_model:
                continue
            for arm in ARMS:
                for metric in COMPARISON_METRICS:
                    deltas, tasks = _paired_values(
                        rows, reference_model, comparator, arm, metric
                    )
                    comparison_rows.append(
                        {
                            "scope": scope,
                            "comparison": f"{reference_model}_vs_{comparator}",
                            "arm": arm,
                            "metric": metric,
                            **_stratified_bootstrap(
                                deltas,
                                tasks,
                                bootstrap_replicates,
                                rng,
                                reference_model,
                            ),
                        }
                    )

    per_task_rows = []
    task_groups = defaultdict(list)
    for row in per_task_seed_rows:
        task_groups[
            (
                row["scope"],
                row["model"],
                row["arm"],
                row["task_id"],
            )
        ].append(row)
    for (scope, model, arm, task_id), rows in sorted(
        task_groups.items()
    ):
        result_row = {
            "scope": scope,
            "model": model,
            "arm": arm,
            "task_id": task_id,
            "seed_count": len(rows),
        }
        for metric in METRICS:
            values = _mean_std(row[metric] for row in rows)
            result_row[f"{metric}_mean"] = values["mean"]
            result_row[f"{metric}_std"] = values[
                "std_across_probe_seeds"
            ]
        per_task_rows.append(result_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main_table.csv"
    seed_path = output_dir / "per_seed_task_macro.csv"
    pooled_path = output_dir / "per_seed_pooled_split_metrics.csv"
    task_path = output_dir / "per_task.csv"
    paired_path = output_dir / "paired_comparisons.csv"
    _write_csv(main_path, main_rows)
    _write_csv(seed_path, per_seed_rows)
    _write_csv(pooled_path, pooled_rows)
    _write_csv(task_path, per_task_rows)
    _write_csv(paired_path, comparison_rows)
    result = {
        "schema": 1,
        "status": "COMPLETE",
        "protocol": PROTOCOL,
        "target_definition": TARGET_DEFINITION,
        "models": list(models),
        "reference_model": reference_model,
        "seeds": list(seeds),
        "splits": list(splits),
        "primary_scope": "id_all",
        "primary_metric": "active_formula_mae",
        "aggregation": (
            "episode mean within view scope, task macro within seed, "
            "mean/std across five probe seeds"
        ),
        "bootstrap": {
            "method": "paired episode, stratified task-macro",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
        },
        "summary": summary,
        "files": {
            "main_table": str(main_path),
            "per_seed_task_macro": str(seed_path),
            "per_seed_pooled_split_metrics": str(pooled_path),
            "per_task": str(task_path),
            "paired_comparisons": str(paired_path),
        },
        "completion_files": completion_paths,
    }
    aggregate_path = output_dir / "aggregate.json"
    atomic_json_dump(json_safe(result), aggregate_path)
    result["aggregate_path"] = str(aggregate_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parent / "results" / "model_runs"
    default_output = Path(__file__).resolve().parent / "results" / "aggregate"
    parser.add_argument("--root", default=default_root)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--reference-model", default="time")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    args = parser.parse_args()
    result = aggregate(
        args.root,
        args.output_dir,
        models=tuple(item.strip() for item in args.models.split(",") if item.strip()),
        seeds=tuple(int(item) for item in args.seeds.split(",") if item.strip()),
        reference_model=args.reference_model,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate": result["aggregate_path"],
                "main_table": result["files"]["main_table"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
