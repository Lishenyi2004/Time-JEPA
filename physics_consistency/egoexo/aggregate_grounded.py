#!/usr/bin/env python3
"""Aggregate EgoExo grounded runs across probe seeds and paired takes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .bank_common import atomic_json_dump, json_safe, sha256_file
except ImportError:
    from bank_common import atomic_json_dump, json_safe, sha256_file


DEFAULT_MODELS = ("time", "video_only", "zero")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
METRICS = (
    "physics_mae_m",
    "physics_active_mae_m",
    "direct_distance_mae_m",
    "oracle_mae_m",
    "law_overall_macro_auroc",
    "law_conditioned_macro_auroc",
    "law_strict_recall_at_1",
)
TAKE_METRICS = (
    "physics_mae_m",
    "physics_active_mae_m",
    "direct_distance_mae_m",
    "law_overall_macro_auroc",
    "law_conditioned_macro_auroc",
)
HIGHER_IS_BETTER = {
    "law_overall_macro_auroc",
    "law_conditioned_macro_auroc",
    "law_strict_recall_at_1",
}
AGGREGATE_PROTOCOL = "egoexo_grounded_five_seed_aggregate_v1"


def _parse_csv_list(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def _as_float(value) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _finite_mean(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else np.nan


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_run(root: Path, model: str, seed: int) -> dict:
    run_dir = root / model / f"seed_{seed}"
    completion_path = run_dir / "COMPLETED.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"Incomplete run: {completion_path}")
    completion = json.loads(completion_path.read_text())
    if completion.get("status") != "COMPLETE":
        raise RuntimeError(f"Run status is not COMPLETE: {completion_path}")
    if completion.get("model_name") != model or int(completion.get("probe_seed", -1)) != seed:
        raise RuntimeError(f"Run identity mismatch: {completion_path}")
    summary_path = Path(completion["run_summary"])
    summary = json.loads(summary_path.read_text())
    return {
        "model": model,
        "seed": seed,
        "run_dir": str(run_dir),
        "completion_path": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "completion": completion,
        "summary": summary,
    }


def _seed_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        for role, arms in run["summary"]["roles"].items():
            for side, metrics in arms.items():
                row = {
                    "model": run["model"],
                    "seed": run["seed"],
                    "role": role,
                    "side": side,
                    "view_count": metrics["view_count"],
                }
                row.update(
                    {metric: _as_float(metrics.get(metric, np.nan)) for metric in METRICS}
                )
                rows.append(row)
    return rows


def _aggregate_seed_metrics(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["role"], row["side"])].append(row)
    output = {}
    for (model, role, side), items in sorted(grouped.items()):
        destination = output.setdefault(model, {}).setdefault(role, {}).setdefault(side, {})
        destination["seed_count"] = len(items)
        for metric in METRICS:
            values = np.asarray([_as_float(item[metric]) for item in items], dtype=np.float64)
            finite = values[np.isfinite(values)]
            destination[metric] = {
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "sample_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "values": values.tolist(),
            }
    return output


def _take_rows(run: dict) -> list[dict]:
    rows = []
    for split_name, output in run["completion"]["outputs"].items():
        role = split_name.split("__", 1)[0]
        metrics_payload = json.loads(Path(output["metrics"]).read_text())
        for row in _read_csv(metrics_payload["per_take_csv"]):
            converted = {
                "model": run["model"],
                "seed": int(run["seed"]),
                "role": role,
                "view": row["view"],
                "take_uid": row["take_uid"],
                "participant_uid": row["participant_uid"],
                "capture_uid": row["capture_uid"],
                "task_id": row["task_id"],
                "side": row["side"],
            }
            for metric in TAKE_METRICS:
                converted[metric] = _as_float(row.get(metric))
            rows.append(converted)
    return rows


def _collapse_take_rows(rows: list[dict]) -> dict[tuple, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["role"], row["side"], row["take_uid"])].append(row)
    output = {}
    for key, items in grouped.items():
        output[key] = {
            "participant_uid": items[0]["participant_uid"],
            "capture_uid": items[0]["capture_uid"],
            "task_id": items[0]["task_id"],
            **{
                metric: _finite_mean(item[metric] for item in items)
                for metric in TAKE_METRICS
            },
        }
    return output


def _paired_bootstrap(
    collapsed: dict,
    reference_model: str,
    comparator: str,
    role: str,
    side: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict:
    reference_keys = {
        key[3]
        for key in collapsed
        if key[0] == reference_model and key[1] == role and key[2] == side
    }
    comparator_keys = {
        key[3]
        for key in collapsed
        if key[0] == comparator and key[1] == role and key[2] == side
    }
    takes = sorted(reference_keys & comparator_keys)
    if not takes:
        return {"take_count": 0}
    reference = np.asarray(
        [collapsed[(reference_model, role, side, take)][metric] for take in takes],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [collapsed[(comparator, role, side, take)][metric] for take in takes],
        dtype=np.float64,
    )
    valid = np.isfinite(reference) & np.isfinite(baseline)
    reference, baseline = reference[valid], baseline[valid]
    if metric in HIGHER_IS_BETTER:
        differences = reference - baseline
        definition = f"{reference_model}_minus_{comparator}"
    else:
        differences = baseline - reference
        definition = f"{comparator}_minus_{reference_model}"
    if not len(differences):
        return {"take_count": 0}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(differences), size=(int(samples), len(differences)))
    distribution = np.mean(differences[draws], axis=1)
    estimate = float(np.mean(differences))
    result = {
        "take_count": int(len(differences)),
        "difference_definition": definition,
        "estimate": estimate,
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "reference_model": reference_model,
        "p_reference_not_better": float(
            (np.sum(distribution <= 0.0) + 1) / (len(distribution) + 1)
        ),
        "reference_mean": float(np.mean(reference)),
        "comparator_mean": float(np.mean(baseline)),
        "relative_improvement": (
            float(estimate / abs(np.mean(baseline)))
            if abs(np.mean(baseline)) > 1e-12
            else np.nan
        ),
    }
    # Keep the original three-model output fields stable for existing readers.
    if reference_model == "time":
        result["p_time_not_better"] = result["p_reference_not_better"]
        result["time_mean"] = result["reference_mean"]
    return result


def aggregate(
    root: str | Path,
    output_dir: str | Path,
    models=DEFAULT_MODELS,
    seeds=DEFAULT_SEEDS,
    reference_model: str = "time",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260723,
) -> dict:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    models = tuple(str(model) for model in models)
    seeds = tuple(int(seed) for seed in seeds)
    if not models:
        raise ValueError("models must contain at least one model")
    if len(set(models)) != len(models):
        raise ValueError(f"models contains duplicates: {models}")
    if reference_model not in models:
        raise ValueError(
            f"reference_model={reference_model!r} is not present in models={models}"
        )
    if not seeds:
        raise ValueError("seeds must contain at least one probe seed")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    runs = [_load_run(root, model, int(seed)) for model in models for seed in seeds]
    seed_rows = _seed_rows(runs)
    _write_csv(output_dir / "seed_metrics.csv", seed_rows)
    take_rows = [row for run in runs for row in _take_rows(run)]
    collapsed = _collapse_take_rows(take_rows)
    paired = {}
    for comparator in models:
        if comparator == reference_model:
            continue
        for role in ("id", "ood"):
            for side in ("left", "right"):
                for metric in TAKE_METRICS:
                    key = f"{comparator}/{role}/{side}/{metric}"
                    paired[key] = _paired_bootstrap(
                        collapsed,
                        reference_model,
                        comparator,
                        role,
                        side,
                        metric,
                        bootstrap_samples,
                        bootstrap_seed,
                    )
    result = {
        "schema": 1,
        "protocol": AGGREGATE_PROTOCOL,
        "root": str(root),
        "models": list(models),
        "reference_model": reference_model,
        "seeds": [int(seed) for seed in seeds],
        "run_count": len(runs),
        "seed_aggregate": _aggregate_seed_metrics(seed_rows),
        "paired_take_bootstrap": paired,
        "bootstrap": {"samples": int(bootstrap_samples), "seed": int(bootstrap_seed)},
        "inputs": [
            {
                key: run[key]
                for key in (
                    "model",
                    "seed",
                    "run_dir",
                    "completion_path",
                    "completion_sha256",
                    "summary_path",
                    "summary_sha256",
                )
            }
            for run in runs
        ],
        "seed_metrics_csv": str(output_dir / "seed_metrics.csv"),
        "interpretation_guard": (
            "Bootstrap clusters are takes after averaging probe seeds and repeated views; "
            "confidence intervals do not include encoder-training seed variance."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "aggregate_summary.json"
    atomic_json_dump(json_safe(result), path)
    result["path"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="physics_consistency/egoexo/results/model_runs",
    )
    parser.add_argument(
        "--output-dir",
        default="physics_consistency/egoexo/results/aggregate",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--reference-model", default="time")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()
    result = aggregate(
        args.root,
        args.output_dir,
        models=_parse_csv_list(args.models),
        seeds=_parse_csv_list(args.seeds, int),
        reference_model=args.reference_model,
        bootstrap_samples=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
